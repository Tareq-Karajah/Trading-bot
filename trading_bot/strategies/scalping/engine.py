from datetime import datetime, timezone, time
from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.core.models import OHLCV
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

# --- SignalIntent Contract (Strictly from Spec) ---

class TPPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    tp1_distance: Optional[float]
    tp1_size_pct: float
    tp2_distance: Optional[float]
    trail_atr_mult: float

class TimeoutPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_bars: Optional[int]
    mandatory_exit_utc: Optional[str]

class ExecutionConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_spread_bp: float
    max_slippage_bp: float
    min_quote_fresh_ms: int

class SignalIntent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    strategy_name: str
    direction: int # +1, -1
    score: int # 0-5
    entry_type: str # "MARKET", "LIMIT", "STOP_LIMIT"
    entry_trigger: float
    sl_distance: float
    tp_plan: TPPlan
    timeout_plan: TimeoutPlan
    regime_context: RegimeState
    macro_context: MacroState
    execution_constraints: ExecutionConstraints
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Scalping Decision Model ---

class ScalpingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    signal_intent: Optional[SignalIntent]
    bb_signal: bool
    rsi_value: float
    ema_200_value: float
    trend_day_filter: bool
    score: int
    consecutive_losses: int
    halted_today: bool
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Scalping Engine ---

class ScalpingEngine:
    """
    Implements Scalping Strategy.
    Spec Section 10.
    """
    
    # Constants
    MIN_CANDLES = 200
    CONSECUTIVE_LOSS_LIMIT = 5
    
    # Gates
    GATE_MAX_SPREAD_BP = 2.0
    GATE_SESSION_START = 8 # 08:00 UTC
    GATE_SESSION_END = 17 # 17:00 UTC (Inclusive? "in [08:00, 17:00]" usually inclusive)
    
    # Signal
    BB_STD_DEV = 2.0
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    EMA_200_PERIOD = 200
    VMA_PERIOD = 20
    VOL_MULT_ENTRY = 1.10
    VOL_MULT_BONUS = 1.50
    TREND_DAY_ATR_MULT = 2.50
    
    # Management
    SL_ATR_MULT = 1.0
    TP1_ATR_MULT = 0.5
    TP1_SIZE_PCT = 0.50
    TRAIL_ATR_MULT = 1.00
    TIMEOUT_BARS = 6
    
    # Constraints
    EXEC_MAX_SPREAD_BP = 2.0
    EXEC_MAX_SLIPPAGE_BP = 3.0
    EXEC_MIN_FRESHNESS_MS = 300

    def __init__(self) -> None:
        self._consecutive_losses: int = 0
        self._halted_today: bool = False
        self._last_reset_date: Optional[datetime] = None

    def _check_daily_reset(self, now_utc: datetime) -> None:
        """
        Reset halt and counters at 00:00 UTC.
        """
        if self._last_reset_date is None or now_utc.date() > self._last_reset_date.date():
            self._consecutive_losses = 0
            self._halted_today = False
            self._last_reset_date = now_utc

    def record_trade_result(self, was_win: bool) -> None:
        """
        Update consecutive losses based on trade outcome.
        """
        if was_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.CONSECUTIVE_LOSS_LIMIT:
                self._halted_today = True

    def _calculate_indicators(self, closes: np.ndarray[np.float64, np.dtype[np.float64]], volumes: np.ndarray[np.float64, np.dtype[np.float64]]) -> Tuple[float, float, float, float, float, float]:
        """
        Compute indicators for the LAST candle.
        Returns: bb_upper, bb_lower, rsi, ema_200, vma_20, sma_20
        """
        # BB (SMA 20)
        sma_20 = float(np.mean(closes[-20:]))
        std_20 = float(np.std(closes[-20:]))
        bb_upper = sma_20 + self.BB_STD_DEV * std_20
        bb_lower = sma_20 - self.BB_STD_DEV * std_20
        
        # RSI 14
        deltas = np.diff(closes)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        # Calculate full RSI series to get the correct current value using history
        if len(gains) < 14:
             return 0.0, 0.0, 50.0, 0.0, 0.0, 0.0
             
        avg_gain = float(np.mean(gains[:14]))
        avg_loss = float(np.mean(losses[:14]))
        
        for i in range(14, len(gains)):
            avg_gain = (avg_gain * 13 + gains[i]) / 14.0
            avg_loss = (avg_loss * 13 + losses[i]) / 14.0
            
        rs = avg_gain / avg_loss if avg_loss > 0 else 0.0
        if avg_loss == 0:
            rsi = 100.0
        else:
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
        # EMA 200 - Not used in evaluate (passed in), but computed here for completeness if needed
        ema_200 = float(np.mean(closes[:200])) if len(closes) >= 200 else 0.0
        
        # VMA 20
        vma_20 = float(np.mean(volumes[-20:]))
        
        return float(bb_upper), float(bb_lower), float(rsi), float(ema_200), float(vma_20), float(sma_20)

    def evaluate(
        self,
        candles_m5: List[OHLCV],      # must have >= 200 candles
        ema_200_m5: float,            # Use passed EMA 200
        ema_20_4h: float,
        ema_50_4h: float,
        atr_14_m5: float,
        spread_live_bp: float,
        permissions: DispatcherPermissions,
        now_utc: datetime,
    ) -> ScalpingDecision:
        
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        self._check_daily_reset(now_utc)
        
        # Min Data Guard
        if len(candles_m5) < self.MIN_CANDLES:
            raise ValueError(f"Insufficient candles: {len(candles_m5)} < {self.MIN_CANDLES}")
            
        # Data Extraction
        closes = np.array([c.close for c in candles_m5], dtype=np.float64)
        volumes = np.array([c.volume for c in candles_m5], dtype=np.float64)
        current_close = float(closes[-1])
        current_volume = float(volumes[-1])
        
        # Indicator Computation
        bb_upper, bb_lower, rsi, _, vma_20, _ = self._calculate_indicators(closes, volumes)
        
        # 1. Gates
        # Gate 1: Regime == LOW_VOL strictly
        gate_regime = (permissions.regime == RegimeState.LOW_VOL)
        
        # Gate 2: Macro != EVENT_RISK
        gate_macro = (permissions.macro_bias != MacroState.MACRO_EVENT_RISK)
        
        # Gate 3: Blackout
        gate_blackout = (not permissions.blackout_active)
        
        # Gate 4: Spread <= 2.0
        gate_spread = (spread_live_bp <= self.GATE_MAX_SPREAD_BP)
        
        # Gate 5: Session [08:00, 17:00]
        # "in [08:00, 17:00]" -> 08:00:00 to 17:00:00 inclusive.
        # Hour check: 8 <= h <= 17.
        # If 17, min must be 0 and sec must be 0 (implied, usually M5 close time).
        hour = now_utc.hour
        gate_session = False
        if 8 <= hour < 17:
            gate_session = True
        elif hour == 17:
            if now_utc.minute == 0:
                gate_session = True
                
        # Also check Halt
        # Halt resets at 00:00. This is handled by _check_daily_reset called above.
        # So we just check current state.
        gate_halt = (not self._halted_today)
        
        gates_passed = (gate_regime and gate_macro and gate_blackout and gate_spread and gate_session and gate_halt)
        
        signal_intent = None
        bb_signal = False
        trend_day_filter = False
        score = 0
        
        if gates_passed:
            direction = 0
            
            # Trend Day Filter Check (Common to both)
            # |Close - EMA_200| < 2.5 * ATR
            dist_ema = abs(current_close - ema_200_m5)
            trend_limit = self.TREND_DAY_ATR_MULT * atr_14_m5
            if dist_ema >= trend_limit:
                trend_day_filter = True # Filter ACTIVE (Suppress signal)
            
            if not trend_day_filter:
                # Long Logic
                # Close < Lower BB
                # RSI < 30
                # Close > EMA 200
                # Volume > 1.1 * VMA
                if (current_close < bb_lower and 
                    rsi < self.RSI_OVERSOLD and 
                    current_close > ema_200_m5 and 
                    current_volume > self.VOL_MULT_ENTRY * vma_20):
                    
                    if permissions.direction_constraint != -1:
                        direction = 1
                        bb_signal = True
                        
                # Short Logic
                # Close > Upper BB
                # RSI > 70
                # Close < EMA 200
                # Volume > 1.1 * VMA
                if direction == 0:
                    if (current_close > bb_upper and 
                        rsi > self.RSI_OVERBOUGHT and 
                        current_close < ema_200_m5 and 
                        current_volume > self.VOL_MULT_ENTRY * vma_20):
                        
                        if permissions.direction_constraint != 1:
                            direction = -1
                            bb_signal = True
                            
                if direction != 0:
                    # Scoring
                    score = 0
                    score += 2 # BB Breakout (Mandatory)
                    
                    # HTF Trend Aligned
                    # "EMA_20_4H direction matches direction"
                    # We have ema_20_4h and ema_50_4h?
                    # Spec says "HTF trend aligned (EMA_20_4H direction matches direction)".
                    # How do we determine "EMA_20_4H direction"? Slope? Relation to price? Relation to 50 EMA?
                    # Usually "Trend Aligned" means "Price relative to EMA" or "EMA Slope".
                    # Prompt input has `ema_20_4h`.
                    # Without history of EMA_20_4H, we can't determine slope.
                    # Maybe it means "EMA_20_4H relative to EMA_50_4H"?
                    # Or "Current Price relative to EMA_20_4H"?
                    # Let's check `docs/specification.md` Section 10? (I can't read it now, locked).
                    # I must rely on prompt.
                    # Prompt: "+1 if HTF trend aligned (EMA_20_4H direction matches direction)"
                    # Ambiguous. "EMA_20_4H direction".
                    # Options:
                    # 1. Slope of EMA 20 4H (Need prev value).
                    # 2. Price > EMA 20 4H (Long), Price < EMA (Short).
                    # 3. EMA 20 4H > EMA 50 4H (Long).
                    # Given "ema_50_4h" is passed in but not mentioned in signal conditions, 
                    # it is likely used for THIS alignment check.
                    # So: Long if EMA_20_4H > EMA_50_4H. Short if EMA_20_4H < EMA_50_4H.
                    aligned = False
                    if direction == 1:
                        if ema_20_4h > ema_50_4h:
                            aligned = True
                    else:
                        if ema_20_4h < ema_50_4h:
                            aligned = True
                            
                    if aligned:
                        score += 1
                        
                    # Volume Bonus
                    if current_volume > self.VOL_MULT_BONUS * vma_20:
                        score += 1
                        
                    # RSI Zone
                    # Long: < 30. Short: > 70.
                    # This is already a mandatory condition for signal?
                    # "Signal Conditions: RSI(14) < 30".
                    # "Scoring: +1 if RSI in zone (< 30 long; > 70 short)".
                    # So if signal exists, this point is guaranteed?
                    # Yes. "mandatory — if this fails, score cannot reach 4"?
                    # No, BB is mandatory +2.
                    # RSI is mandatory for signal generation.
                    # So score always starts at 2 (BB) + 1 (RSI) = 3.
                    # Need 1 more (HTF or Vol Bonus).
                    score += 1
                    
                    if score >= 4:
                        # Construct Intent
                        sl_dist = self.SL_ATR_MULT * atr_14_m5
                        
                        signal_intent = SignalIntent(
                            strategy_name="SCALP",
                            direction=direction,
                            score=score,
                            entry_type="MARKET",
                            entry_trigger=float(current_close),
                            sl_distance=float(sl_dist),
                            tp_plan=TPPlan(
                                tp1_distance=float(self.TP1_ATR_MULT * atr_14_m5),
                                tp1_size_pct=self.TP1_SIZE_PCT,
                                tp2_distance=None,
                                trail_atr_mult=self.TRAIL_ATR_MULT
                            ),
                            timeout_plan=TimeoutPlan(
                                max_bars=self.TIMEOUT_BARS,
                                mandatory_exit_utc=None
                            ),
                            regime_context=permissions.regime,
                            macro_context=permissions.macro_bias,
                            execution_constraints=ExecutionConstraints(
                                max_spread_bp=self.EXEC_MAX_SPREAD_BP,
                                max_slippage_bp=self.EXEC_MAX_SLIPPAGE_BP,
                                min_quote_fresh_ms=self.EXEC_MIN_FRESHNESS_MS
                            )
                        )
                        
                        self._publish_signal(signal_intent)

        return ScalpingDecision(
            signal_intent=signal_intent,
            bb_signal=bb_signal,
            rsi_value=float(rsi),
            ema_200_value=float(ema_200_m5),
            trend_day_filter=trend_day_filter,
            score=score,
            consecutive_losses=self._consecutive_losses,
            halted_today=self._halted_today,
            timestamp_utc=now_utc
        )

    def _publish_signal(self, intent: SignalIntent) -> None:
        """
        Publish SIGNAL_INTENT to Redis. Stub.
        """
        pass
