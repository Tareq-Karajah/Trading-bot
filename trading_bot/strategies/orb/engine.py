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

# --- ORB Decision Model ---

class ORBDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    signal_intent: Optional[SignalIntent]
    or_high: Optional[float]
    or_low: Optional[float]
    or_width: Optional[float]
    orb_quality_ok: bool
    buf_final: float
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- ORB Engine ---

class ORBEngine:
    """
    Implements Opening Range Breakout (ORB) Strategy.
    Spec Section 8.
    """
    
    # Frozen Constants (Spec 17.4)
    OR_START_TIME = time(7, 0)
    OR_END_TIME = time(8, 0)
    MIN_CANDLES = 4
    
    # Quality Band
    QUALITY_LOWER_ATR_MULT = 0.25
    QUALITY_LOWER_MEDIAN_MULT = 0.70
    QUALITY_UPPER_ATR_MULT = 1.80
    QUALITY_UPPER_MEDIAN_MULT = 2.00
    
    # Adaptive Buffer
    BUF_BASE_ATR_MULT = 0.15
    BUF_CLAMP_MIN_ATR_MULT = 0.10
    BUF_CLAMP_MAX_ATR_MULT = 0.25
    
    # Entry
    VOL_CONFIRMATION_MULT = 1.20
    
    # Management
    SL_CAP_ATR_MULT = 1.50
    TP1_RR = 1.0
    TP1_SIZE_PCT = 0.50
    TRAIL_ATR_MULT = 2.00
    MANDATORY_EXIT_DEFAULT = "20:45"
    MANDATORY_EXIT_FRIDAY = "19:30"
    
    # Constraints
    EXEC_MAX_SPREAD_BP = 3.0
    EXEC_MAX_SLIPPAGE_BP = 5.0
    EXEC_MIN_QUOTE_FRESH_MS = 500

    def __init__(self) -> None:
        self.new_session()

    def new_session(self) -> None:
        """
        Reset session state at 07:00 UTC.
        """
        self._or_high: float = -float('inf')
        self._or_low: float = float('inf')
        self._range_candles: int = 0
        self._session_trade_taken: bool = False
        self._session_valid: bool = False

    def on_range_candle(self, candle: OHLCV) -> None:
        """
        Update Opening Range with M15 candles between 07:00-08:00 UTC.
        """
        if candle.high > self._or_high:
            self._or_high = candle.high
        if candle.low < self._or_low:
            self._or_low = candle.low
        self._range_candles += 1
        
        # Validate range at end of window (implied by usage)
        if self._range_candles >= self.MIN_CANDLES:
            self._session_valid = True

    def evaluate(
        self,
        candle: OHLCV,
        atr_14_m15: float,
        atr_daily_14: float,
        or_median_20: float,
        vma_20: float,
        permissions: DispatcherPermissions,
    ) -> ORBDecision:
        
        timestamp = candle.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # 1. Validate Session & Quality
        if not self._session_valid or self._or_high == -float('inf') or self._or_low == float('inf'):
            return ORBDecision(
                signal_intent=None, or_high=None, or_low=None, or_width=None,
                orb_quality_ok=False, buf_final=0.0, timestamp_utc=timestamp
            )
            
        or_width = self._or_high - self._or_low
        
        # Quality Band Filter
        lower_bound = max(self.QUALITY_LOWER_ATR_MULT * atr_daily_14, self.QUALITY_LOWER_MEDIAN_MULT * or_median_20)
        upper_bound = min(self.QUALITY_UPPER_ATR_MULT * atr_daily_14, self.QUALITY_UPPER_MEDIAN_MULT * or_median_20)
        
        orb_quality_ok = (lower_bound <= or_width <= upper_bound)
        
        if not orb_quality_ok:
            return ORBDecision(
                signal_intent=None, or_high=self._or_high, or_low=self._or_low, or_width=or_width,
                orb_quality_ok=False, buf_final=0.0, timestamp_utc=timestamp
            )

        # 2. Adaptive Buffer Calculation
        # Avoid div by zero
        or_quality_ratio = or_width / or_median_20 if or_median_20 > 0 else 1.0
        buf_base = self.BUF_BASE_ATR_MULT * atr_14_m15
        buf_final_raw = buf_base * (1.0 + 0.30 * (1.0 - or_quality_ratio))
        
        buf_min = self.BUF_CLAMP_MIN_ATR_MULT * atr_14_m15
        buf_max = self.BUF_CLAMP_MAX_ATR_MULT * atr_14_m15
        
        buf_final = max(buf_min, min(buf_final_raw, buf_max))

        # 3. Entry Logic
        signal_intent = None
        
        # Check permissions and session limit
        if (permissions.allow_orb and 
            not self._session_trade_taken and 
            not permissions.blackout_active):
            
            # Shock Abort Check: TR > 2.0 * ATR on last bar
            # candle is the last bar (M15)
            # We don't have PrevClose, so we use High - Low as TR approximation or best effort.
            # Assuming 'candle' is the M15 bar that just closed (or is forming?).
            # "Call for each M15 candle after 08:00 UTC". Usually implies closed candle.
            # We'll use High-Low.
            tr_current = candle.high - candle.low
            shock_abort = (tr_current > 2.0 * atr_14_m15)
            
            if not shock_abort:
                direction = 0
                
                # Long Condition
                long_trigger = self._or_high + buf_final
                if (candle.close > long_trigger and 
                    candle.volume > self.VOL_CONFIRMATION_MULT * vma_20 and 
                    permissions.direction_constraint >= 0):
                    direction = 1
                    
                # Short Condition
                short_trigger = self._or_low - buf_final
                if (candle.close < short_trigger and 
                    candle.volume > self.VOL_CONFIRMATION_MULT * vma_20 and 
                    permissions.direction_constraint <= 0):
                    direction = -1
                    
                if direction != 0:
                    # 4. Score Computation
                    score = 0
                    score += 2 # Quality band passed (mandatory here)
                    
                    if permissions.direction_constraint == direction:
                        score += 1
                        
                    # Volume > 1.20 * VMA (Already checked for entry, so +1)
                    score += 1
                    
                    # NOT shock_abort (Already checked, so +1)
                    score += 1
                    
                    # Total max possible: 2 + 1 + 1 + 1 = 5.
                    
                    if score >= 4:
                        # 5. Construct SignalIntent
                        
                        if direction == 1:
                            # Long: Entry = High + buf
                            # SL Level = Low - buf
                            entry_price = float(self._or_high + buf_final)
                            stop_level_raw = float(self._or_low - buf_final)
                            dist = entry_price - stop_level_raw
                        else:
                            # Short: Entry = Low - buf
                            # SL Level = High + buf
                            entry_price = float(self._or_low - buf_final)
                            stop_level_raw = float(self._or_high + buf_final)
                            dist = stop_level_raw - entry_price
                            
                        # Cap Distance
                        cap = self.SL_CAP_ATR_MULT * atr_14_m15
                        sl_distance = min(dist, cap)
                        
                        # Mandatory Exit
                        is_friday = (timestamp.weekday() == 4)
                        exit_time = self.MANDATORY_EXIT_FRIDAY if is_friday else self.MANDATORY_EXIT_DEFAULT
                        
                        signal_intent = SignalIntent(
                            strategy_name="ORB",
                            direction=direction,
                            score=score,
                            entry_type="STOP_LIMIT",
                            entry_trigger=entry_price,
                            sl_distance=float(sl_distance),
                            tp_plan=TPPlan(
                                tp1_distance=float(sl_distance * self.TP1_RR),
                                tp1_size_pct=self.TP1_SIZE_PCT,
                                tp2_distance=None,
                                trail_atr_mult=self.TRAIL_ATR_MULT
                            ),
                            timeout_plan=TimeoutPlan(
                                max_bars=None,
                                mandatory_exit_utc=exit_time
                            ),
                            regime_context=permissions.regime,
                            macro_context=permissions.macro_bias,
                            execution_constraints=ExecutionConstraints(
                                max_spread_bp=self.EXEC_MAX_SPREAD_BP,
                                max_slippage_bp=self.EXEC_MAX_SLIPPAGE_BP,
                                min_quote_fresh_ms=self.EXEC_MIN_QUOTE_FRESH_MS
                            )
                        )
                        
                        self._session_trade_taken = True
                        self._publish_signal(signal_intent)

        return ORBDecision(
            signal_intent=signal_intent,
            or_high=self._or_high,
            or_low=self._or_low,
            or_width=or_width,
            orb_quality_ok=orb_quality_ok,
            buf_final=buf_final,
            timestamp_utc=timestamp
        )

    def _publish_signal(self, intent: SignalIntent) -> None:
        """
        Publish SIGNAL_INTENT to Redis. Stub.
        """
        pass
