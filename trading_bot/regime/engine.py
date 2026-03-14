from typing import List
from trading_bot.core.models import OHLCV

def calculate_true_range(current: OHLCV, previous_close: float) -> float:
    """
    Calculate True Range (TR) for a single candle.
    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    """
    hl = current.high - current.low
    h_pc = abs(current.high - previous_close)
    l_pc = abs(current.low - previous_close)
    return max(hl, h_pc, l_pc)

def calculate_atr(candles: List[OHLCV], period: int) -> float:
    """
    Calculate Average True Range (ATR) using Wilder's Smoothing.
    Returns 0.0 if insufficient data (len(candles) < period + 1).
    """
    if len(candles) < period + 1:
        return 0.0

    # Initial TRs
    trs = []
    for i in range(1, len(candles)):
        tr = calculate_true_range(candles[i], candles[i-1].close)
        trs.append(tr)
    
    if len(trs) < period:
        return 0.0

    # First ATR is simple average of first 'period' TRs
    # Wilder's smoothing: ATR[t] = (ATR[t-1] * (n-1) + TR[t]) / n
    
    current_atr = sum(trs[:period]) / period
    
    for i in range(period, len(trs)):
        current_atr = (current_atr * (period - 1) + trs[i]) / period
        
    return current_atr

# Re-exporting regime classes for cleaner imports if needed, 
# but they are defined in the same file below in the original code.
# I will combine the indicator logic and the regime logic into this file as requested.
# "Move ATR/TR calculations from trading_bot/signals/indicators.py to trading_bot/regime/engine.py or a private helper inside the regime module"

from enum import Enum
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

class RegimeState(str, Enum):
    LOW_VOL = "LOW_VOL"
    MID_VOL = "MID_VOL"
    HIGH_VOL = "HIGH_VOL"
    SHOCK_EVENT = "SHOCK_EVENT"


class RegimeDecision(BaseModel):
    """
    Output model for the Regime Engine.
    """
    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str
    current_regime: RegimeState
    atr_fast: float
    atr_slow: float
    ratio: float
    risk_scalar: float
    shock_detected: bool
    timestamp: datetime
    
    # Hysteresis info (optional/extra, good for debugging)
    pending_regime: Optional[RegimeState] = None
    confirmation_count: int = 0

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


@dataclass
class RegimeContext:
    current_regime: RegimeState = RegimeState.MID_VOL
    pending_regime: Optional[RegimeState] = None
    confirmation_count: int = 0
    bars_since_shock: int = 999  # Large number = no recent shock
    last_processed_time: Optional[datetime] = None


class RegimeEngine:
    """
    Volatility Regime Engine.
    Classifies market state into LOW_VOL, MID_VOL, HIGH_VOL, or SHOCK_EVENT.
    """
    
    # Frozen defaults from Spec 17.1
    ATR_FAST_PERIOD = 14
    ATR_SLOW_PERIOD = 200
    
    THRESHOLD_LOW = 0.85
    THRESHOLD_HIGH = 1.25
    
    HYSTERESIS_BARS = 3  # Spec Section 5.2 and 17.1
    SHOCK_THRESHOLD_MULT = 2.0
    SHOCK_COOLDOWN_BARS = 12
    
    RISK_SCALARS = {
        RegimeState.LOW_VOL: 1.0,
        RegimeState.MID_VOL: 0.8,
        RegimeState.HIGH_VOL: 0.5,
        RegimeState.SHOCK_EVENT: 0.0,
    }

    def __init__(self) -> None:
        # State tracking per (symbol, timeframe)
        self._contexts: Dict[str, RegimeContext] = {}

    def _get_context(self, key: str) -> RegimeContext:
        if key not in self._contexts:
            self._contexts[key] = RegimeContext()
        return self._contexts[key]

    def evaluate(self, symbol: str, timeframe: str, candles: List[OHLCV]) -> RegimeDecision:
        """
        Evaluate the current regime for a given symbol and timeframe.
        """
        if not candles:
            raise ValueError("Candles list cannot be empty")

        key = f"{symbol}:{timeframe}"
        ctx = self._get_context(key)
        last_candle = candles[-1]
        
        # 1. Calculate Indicators
        atr_fast = calculate_atr(candles, self.ATR_FAST_PERIOD)
        atr_slow = calculate_atr(candles, self.ATR_SLOW_PERIOD)
        
        # Safety: avoid division by zero
        if atr_slow == 0:
            ratio = 1.0 # Default to neutral/mid
        else:
            ratio = atr_fast / atr_slow

        # 2. Detect Shock
        # TR > 2.0 * ATR(14)
        prev_atr_fast = calculate_atr(candles[:-1], self.ATR_FAST_PERIOD) if len(candles) > 1 else atr_fast
        
        if len(candles) > 1:
            current_tr = calculate_true_range(last_candle, candles[-2].close)
        else:
            current_tr = last_candle.high - last_candle.low

        shock_threshold = self.SHOCK_THRESHOLD_MULT * prev_atr_fast
        is_shock = (current_tr > shock_threshold) and (prev_atr_fast > 0)

        # 3. Determine Raw Regime (ignoring shock/hysteresis for a moment)
        if ratio < self.THRESHOLD_LOW:
            raw_regime = RegimeState.LOW_VOL
        elif ratio > self.THRESHOLD_HIGH:
            raw_regime = RegimeState.HIGH_VOL
        else:
            raw_regime = RegimeState.MID_VOL

        # 4. State Machine Update
        is_new_bar = (ctx.last_processed_time != last_candle.timestamp)
        
        if is_new_bar:
            ctx.last_processed_time = last_candle.timestamp
            
            # Shock Logic
            if is_shock:
                ctx.current_regime = RegimeState.SHOCK_EVENT
                ctx.bars_since_shock = 0
                ctx.pending_regime = None
                ctx.confirmation_count = 0
                self._publish_event("SHOCK_EVENT", symbol, timeframe, last_candle.timestamp, {})
            
            elif ctx.current_regime == RegimeState.SHOCK_EVENT:
                # In Shock Cooldown
                ctx.bars_since_shock += 1
                
                if ctx.bars_since_shock >= self.SHOCK_COOLDOWN_BARS:
                    # Recovery Protocol (Spec 5.3)
                    # if ATR_fast <= 1.25 * ATR_slow -> MID_VOL
                    # else -> HIGH_VOL
                    # "never jump directly to LOW_VOL"
                    if ratio <= self.THRESHOLD_HIGH:
                         new_regime = RegimeState.MID_VOL
                    else:
                         new_regime = RegimeState.HIGH_VOL
                    
                    ctx.current_regime = new_regime
                    # Reset hysteresis
                    ctx.pending_regime = None
                    ctx.confirmation_count = 0
                    
                    self._publish_event("REGIME_UPDATE", symbol, timeframe, last_candle.timestamp, {"regime": ctx.current_regime})
            
            else:
                # Normal Operation with Hysteresis
                ctx.bars_since_shock += 1 # Keep counting even if large
                
                if raw_regime != ctx.current_regime:
                    # Hysteresis Reset Logic (Issue 3 fix)
                    # If pending regime exists but new raw_regime is different from pending, reset.
                    if ctx.pending_regime is not None and raw_regime != ctx.pending_regime:
                        ctx.pending_regime = raw_regime
                        ctx.confirmation_count = 1
                    elif raw_regime == ctx.pending_regime:
                        ctx.confirmation_count += 1
                    else:
                        ctx.pending_regime = raw_regime
                        ctx.confirmation_count = 1
                    
                    if ctx.confirmation_count >= self.HYSTERESIS_BARS:
                        ctx.current_regime = raw_regime
                        ctx.pending_regime = None
                        ctx.confirmation_count = 0
                        self._publish_event("REGIME_UPDATE", symbol, timeframe, last_candle.timestamp, {"regime": ctx.current_regime})
                else:
                    # Raw matches current, reset pending
                    ctx.pending_regime = None
                    ctx.confirmation_count = 0

        # 5. Construct Decision
        decision = RegimeDecision(
            symbol=symbol,
            timeframe=timeframe,
            current_regime=ctx.current_regime,
            atr_fast=atr_fast,
            atr_slow=atr_slow,
            ratio=ratio,
            risk_scalar=self.RISK_SCALARS[ctx.current_regime],
            shock_detected=is_shock,
            timestamp=last_candle.timestamp,
            pending_regime=ctx.pending_regime,
            confirmation_count=ctx.confirmation_count
        )
        
        return decision

    def _publish_event(self, event_type: str, symbol: str, timeframe: str, timestamp: datetime, payload: Dict[str, Any]) -> None:
        """
        Stub for Redis event publishing.
        Allowed as internal infrastructure.
        """
        # In a real impl, this would push to Redis.
        # For now, we can log or pass.
        pass
