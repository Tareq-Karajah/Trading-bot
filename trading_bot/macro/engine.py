from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.macro.calendar import CalendarSeverity

# --- Enums from Spec ---

class MacroState(str, Enum):
    MACRO_BULL_GOLD = "MACRO_BULL_GOLD"
    MACRO_BEAR_GOLD = "MACRO_BEAR_GOLD"
    MACRO_NEUTRAL = "MACRO_NEUTRAL"
    MACRO_EVENT_RISK = "MACRO_EVENT_RISK"

# --- Models ---

class MacroBiasDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    macro_state: MacroState
    confidence: float
    macro_risk_modifier: float
    usd_impulse: float
    yield_impulse: float
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

class MacroBiasEngine:
    """
    Classifies the macro environment for Gold (XAU/USD).
    """
    
    # Frozen Thresholds (Spec 17.2)
    BULL_GOLD_USD_THRESH = -0.005
    BULL_GOLD_YIELD_THRESH = -0.003
    
    BEAR_GOLD_USD_THRESH = 0.005
    BEAR_GOLD_YIELD_THRESH = 0.003
    
    IMPULSE_WINDOW = 5  # 5-day ROC
    MIN_DATA_POINTS = 6 # Need current + 5 previous
    
    RISK_OFF_ATR_MULT = 2.0
    
    def __init__(self) -> None:
        self._current_state: MacroState = MacroState.MACRO_NEUTRAL
        self._current_risk_modifier: float = 1.0

    async def evaluate(
        self,
        usdx_closes: List[float],
        tlt_closes: List[float],
        atr_daily: float,
        atr_baseline_252d: float,
        cal_severity: CalendarSeverity = CalendarSeverity.NONE,
    ) -> MacroBiasDecision:
        """
        Evaluate macro state based on USD, Yields, and Event Risk.
        """
        
        # 1. Calculate Impulses
        usd_impulse = 0.0
        yield_impulse = 0.0
        
        # Fallback: fewer than 6 values
        if len(usdx_closes) >= self.MIN_DATA_POINTS:
            current = usdx_closes[-1]
            prev_5 = usdx_closes[-1 - self.IMPULSE_WINDOW]
            # Avoid division by zero
            if prev_5 != 0:
                usd_impulse = (current - prev_5) / prev_5

        if len(tlt_closes) >= self.MIN_DATA_POINTS:
            current = tlt_closes[-1]
            prev_5 = tlt_closes[-1 - self.IMPULSE_WINDOW]
            # Yield impulse = -(TLT_impulse) because TLT price inverse to yield
            # Spec 6: yield_impulse = -(TLT_close - TLT_close[5]) / TLT_close[5]
            if prev_5 != 0:
                tlt_impulse = (current - prev_5) / prev_5
                yield_impulse = -tlt_impulse

        # 2. Check Risk-Off / Event Risk
        # Spec 6: risk_off = ATR_daily > 2.0 x ATR_baseline_252d
        risk_off = atr_daily > (self.RISK_OFF_ATR_MULT * atr_baseline_252d)
        
        # Spec 6: if cal_severity in [HIGH, CRITICAL] OR risk_off
        is_event_risk = (cal_severity in [CalendarSeverity.HIGH, CalendarSeverity.CRITICAL]) or risk_off
        
        # 3. Classify State
        if is_event_risk:
            macro_state = MacroState.MACRO_EVENT_RISK
        elif usd_impulse < self.BULL_GOLD_USD_THRESH and yield_impulse < self.BULL_GOLD_YIELD_THRESH:
            macro_state = MacroState.MACRO_BULL_GOLD
        elif usd_impulse > self.BEAR_GOLD_USD_THRESH and yield_impulse > self.BEAR_GOLD_YIELD_THRESH:
            macro_state = MacroState.MACRO_BEAR_GOLD
        else:
            macro_state = MacroState.MACRO_NEUTRAL
            
        # 4. Calculate Outputs
        # confidence = min(1.0, (abs(usd_impulse)/0.005 + abs(yield_impulse)/0.003) / 2)
        # Set confidence = 1.0 when macro_state = MACRO_EVENT_RISK
        if macro_state == MacroState.MACRO_EVENT_RISK:
            confidence = 1.0
        else:
            raw_conf = (abs(usd_impulse) / 0.005 + abs(yield_impulse) / 0.003) / 2.0
            confidence = min(1.0, raw_conf)
            
        # Risk Modifier
        risk_map = {
            MacroState.MACRO_BULL_GOLD: 1.0,
            MacroState.MACRO_BEAR_GOLD: 1.0,
            MacroState.MACRO_NEUTRAL: 1.0,
            MacroState.MACRO_EVENT_RISK: 0.5
        }
        macro_risk_modifier = risk_map[macro_state]
        
        # 5. Update Internal State & Publish if Changed
        prev_state = self._current_state
        self._current_state = macro_state
        self._current_risk_modifier = macro_risk_modifier
        
        decision = MacroBiasDecision(
            macro_state=macro_state,
            confidence=confidence,
            macro_risk_modifier=macro_risk_modifier,
            usd_impulse=usd_impulse,
            yield_impulse=yield_impulse,
            timestamp_utc=datetime.now(timezone.utc)
        )
        
        if macro_state != prev_state:
            await self._publish_event(decision)
            
        return decision

    def get_current_state(self) -> MacroState:
        return self._current_state

    def get_risk_modifier(self) -> float:
        return self._current_risk_modifier

    async def _publish_event(self, decision: MacroBiasDecision) -> None:
        """
        Publish MACRO_BIAS_UPDATE to Redis.
        Stub for now.
        """
        pass
