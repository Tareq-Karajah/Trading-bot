import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.core.models import OHLCV
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

# --- SignalIntent Contract (as per docs/signal_contracts.md) ---

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

# --- Swing Decision Model ---

class SwingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    swing_dir: int        # +1, -1, or 0
    confidence: float
    brake_active: bool
    brake_modifier: float
    score: int
    signal_intent: Optional[SignalIntent]
    mom_1m: float
    mom_3m: float
    mom_6m: float
    mom_12m: float
    atr_daily_20: float
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Swing Engine ---

class SwingEngine:
    """
    Implements TSMOM Multi-Horizon Swing Strategy.
    """
    
    # Frozen weights (Spec 17.3)
    WEIGHT_1M = 0.10
    WEIGHT_3M = 0.20
    WEIGHT_6M = 0.30
    WEIGHT_12M = 0.40
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.30
    BRAKE_MOM_1M_Z = -0.50  # -0.50 sigma
    BRAKE_ATR_RATIO = 1.15
    BRAKE_MODIFIER_ACTIVE = 0.40
    BRAKE_MODIFIER_INACTIVE = 1.00
    
    # Constraints
    EXEC_MAX_SPREAD_BP = 10.0
    EXEC_MAX_SLIPPAGE_BP = 15.0
    EXEC_MIN_QUOTE_FRESH_MS = 2000
    
    # Data Requirements
    MIN_BARS = 253 # 252 for 12m + 1 current

    def evaluate(
        self,
        daily_closes: List[float],
        atr_fast: float,
        atr_slow: float,
        permissions: DispatcherPermissions,
        now_utc: datetime,
    ) -> SwingDecision:
        
        if len(daily_closes) < self.MIN_BARS:
            raise ValueError(f"Insufficient daily closes: {len(daily_closes)} < {self.MIN_BARS}")
            
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # 1. Compute Volatility-Adjusted Momentum
        closes = np.array(daily_closes, dtype=np.float64)
        # Handle zero or negative prices if any (though generator should fix this)
        # Returns: (C[t] - C[t-1]) / C[t-1]
        # Guard against division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(closes) / closes[:-1]
        
        # Replace NaN/Inf with 0.0 to prevent crash, though inputs should be clean
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        def calc_realized_vol(window: int) -> float:
            # StdDev(daily_returns, N) * sqrt(252)
            # Use last N returns
            if len(returns) < window: return 1.0 # Should be guarded by MIN_BARS
            window_returns = returns[-window:]
            return float(np.std(window_returns, ddof=1) * np.sqrt(252))

        def calc_mom(window: int, vol: float) -> float:
            # (Close[t] - Close[t-window]) / vol
            if vol == 0: return 0.0
            price_t = closes[-1]
            price_t_minus_n = closes[-1 - window]
            return float((price_t - price_t_minus_n) / vol)

        vol_21d = calc_realized_vol(21)
        vol_63d = calc_realized_vol(63)
        vol_126d = calc_realized_vol(126)
        vol_252d = calc_realized_vol(252)
        
        mom_1m = calc_mom(21, vol_21d)
        mom_3m = calc_mom(63, vol_63d)
        mom_6m = calc_mom(126, vol_126d)
        mom_12m = calc_mom(252, vol_252d)
        
        # 2. Confidence & Direction
        # Explicit casting to float for mypy and precision safety
        sign_1m = float(np.sign(mom_1m))
        sign_3m = float(np.sign(mom_3m))
        sign_6m = float(np.sign(mom_6m))
        sign_12m = float(np.sign(mom_12m))

        confidence_raw = (
            self.WEIGHT_1M * sign_1m +
            self.WEIGHT_3M * sign_3m +
            self.WEIGHT_6M * sign_6m +
            self.WEIGHT_12M * sign_12m
        )
        
        # Round to 10 decimal places per instructions
        confidence = round(float(confidence_raw), 10)
        
        swing_dir = 0
        if confidence > self.CONFIDENCE_THRESHOLD:
            swing_dir = 1
        elif confidence < -self.CONFIDENCE_THRESHOLD:
            swing_dir = -1
            
        # 3. Turning-Point Brake
        # brake_cond_3: mom_1m < -0.50 * StdDev(mom_1m_series, 52w)
        
        brake_cond_3 = False
        if len(closes) >= 274: # 273 previous + current
            # Optimization: We only need StdDev of mom_1m.
            pass # Logic implemented below
            
        # Implementation of Brake Conditions
        # 1. swing_dir != 0
        c1 = (swing_dir != 0)
        
        # 2. sign(mom_1m) != swing_dir
        c2 = (np.sign(mom_1m) != swing_dir)
        
        # 3. mom_1m < -0.50 * StdDev(mom_1m, 52w)
        c3 = False
        if c1 and c2: # Optimization: only check c3 if c1/c2 pass
            if len(daily_closes) >= 274:
                mom_series = []
                start_idx = len(closes) - 252
                if start_idx >= 22:
                    # To optimize, we can extract slices once
                    # We need history of 252 points.
                    # Each point i (from 0 to 251) corresponds to time T-252+i
                    # We need closes and vol at each point.
                    # Recomputing vol 252 times is slow (252 * 21 days).
                    # But required for correctness if "realized_vol_21d" changes over time.
                    # We accept the cost for now as "evaluate" is daily/low freq.
                    
                    for i in range(252):
                        idx = start_idx + i
                        p_now = closes[idx]
                        p_prev = closes[idx-21]
                        
                        # Vol calculation window: [idx-21 : idx]
                        # returns array is aligned such that returns[k] = C[k+1]/C[k]-1?
                        # No, returns = diff(closes)/closes[:-1].
                        # returns[k] corresponds to change from closes[k] to closes[k+1].
                        # So returns ending at idx (exclusive of future) means returns[idx-21 : idx].
                        # closes[idx] is the price at idx.
                        # returns[idx-1] is (closes[idx]-closes[idx-1])/closes[idx-1].
                        
                        subset_returns = returns[idx-21 : idx]
                        if len(subset_returns) == 21:
                            vol = np.std(subset_returns, ddof=1) * np.sqrt(252)
                            if vol > 0:
                                m = (p_now - p_prev) / vol
                                mom_series.append(m)
                    
                    if len(mom_series) > 0:
                        std_mom = np.std(mom_series, ddof=1)
                        if mom_1m < (-0.50 * std_mom):
                            c3 = True
        
        # 4. ATR_fast / ATR_slow > 1.15
        # Guard div by zero
        c4 = False
        if atr_slow > 0:
            c4 = (atr_fast / atr_slow) > 1.15
            
        brake_active = c1 and c2 and c3 and c4
        brake_modifier = self.BRAKE_MODIFIER_ACTIVE if brake_active else self.BRAKE_MODIFIER_INACTIVE
        
        # 4. Score Computation
        score = 0
        if swing_dir != 0:
            # +1 if sign matches swing_dir
            if np.sign(mom_12m) == swing_dir: score += 1
            if np.sign(mom_6m) == swing_dir: score += 1
            if np.sign(mom_3m) == swing_dir: score += 1
            if np.sign(mom_1m) == swing_dir: score += 1
            if not brake_active: score += 1
            
        # 5. Signal Intent Emission
        signal_intent = None
        
        # Conditions to emit:
        # score >= 4
        # allow_swing_rebalance == True
        # direction_constraint allows direction (0 or match)
        
        dir_allowed = False
        if permissions.direction_constraint == 0:
            dir_allowed = True
        elif permissions.direction_constraint == swing_dir:
            dir_allowed = True
            
        if (score >= 4) and (permissions.allow_swing_rebalance) and dir_allowed and (swing_dir != 0):
            atr_daily_20 = self._calculate_pseudo_atr(closes, 20)
            
            signal_intent = SignalIntent(
                strategy_name="SWING",
                direction=swing_dir,
                score=score,
                entry_type="MARKET",
                entry_trigger=float(closes[-1]),
                sl_distance=float(3.0 * atr_daily_20),
                tp_plan=TPPlan(
                    tp1_distance=None,
                    tp1_size_pct=0.50,
                    tp2_distance=None,
                    trail_atr_mult=3.00
                ),
                timeout_plan=TimeoutPlan(
                    max_bars=None,
                    mandatory_exit_utc="00:05"
                ),
                regime_context=permissions.regime,
                macro_context=permissions.macro_bias,
                execution_constraints=ExecutionConstraints(
                    max_spread_bp=self.EXEC_MAX_SPREAD_BP,
                    max_slippage_bp=self.EXEC_MAX_SLIPPAGE_BP,
                    min_quote_fresh_ms=self.EXEC_MIN_QUOTE_FRESH_MS
                )
            )
            
            # Publish event stub
            self._publish_signal(signal_intent)
        else:
            # Even if no signal, we need to populate atr_daily_20 for Decision object
            atr_daily_20 = self._calculate_pseudo_atr(closes, 20)

        return SwingDecision(
            swing_dir=swing_dir,
            confidence=confidence,
            brake_active=brake_active,
            brake_modifier=brake_modifier,
            score=score,
            signal_intent=signal_intent,
            mom_1m=float(mom_1m),
            mom_3m=float(mom_3m),
            mom_6m=float(mom_6m),
            mom_12m=float(mom_12m),
            atr_daily_20=atr_daily_20,
            timestamp_utc=now_utc
        )

    def _calculate_pseudo_atr(self, closes: np.ndarray[Any, np.dtype[np.float64]], period: int) -> float:
        # Fallback ATR using only closes: Avg(Abs(Close - PrevClose))
        if len(closes) < period + 1:
            return 0.0
        
        tr_series = np.abs(np.diff(closes))
        return float(np.mean(tr_series[-period:]))

    def _publish_signal(self, intent: SignalIntent) -> None:
        """
        Publish SIGNAL_INTENT to Redis. Stub.
        """
        pass
