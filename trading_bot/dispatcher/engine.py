from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState
from trading_bot.macro.calendar import MarketCalendarService, EventRiskState, CalendarSeverity

class DispatcherPermissions(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    allow_swing_rebalance: bool
    allow_orb: bool
    allow_news: bool
    allow_scalp: bool
    direction_constraint: int  # +1 | -1 | 0
    macro_bias: MacroState
    regime: RegimeState
    risk_scalar: float
    blackout_active: bool
    blackout_reason: Optional[str]

class StrategyDispatcher:
    """
    Permissions Engine that determines which strategies are allowed to run.
    Pure logic, no side effects.
    """
    
    def __init__(self) -> None:
        self._last_permissions: Optional[DispatcherPermissions] = None

    def evaluate(
        self,
        regime: RegimeState,
        macro_state: MacroState,
        risk_scalar: float,
        swing_dir: int,
        now_utc: datetime,
        calendar_service: MarketCalendarService,
        event_risk_state: EventRiskState,
    ) -> DispatcherPermissions:
        
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # 1. Determine Blackout
        blackout_active = False
        blackout_reason = None
        
        # Spec: HIGH or CRITICAL calendar event scheduled within ±600 seconds of now_utc
        # We need to iterate events from calendar service.
        # But calendar_service interface doesn't expose raw events list directly in a "get all" way 
        # that fits this query easily without modifying it.
        # However, we are allowed to use the service.
        # Let's inspect `calendar_service._events` (it's internal but accessible in Python).
        # Or better, we can assume the service has them.
        # Strict adherence: "You may modify only... trading_bot/dispatcher/..."
        # I cannot add methods to CalendarService.
        # I must access `_events` or rely on `get_current_severity` if it supported window check.
        # But `get_current_severity` uses Pre/Live/Post windows which vary per event.
        # Blackout logic is fixed ±600s.
        # I will access `_events` directly as it's the only way without modifying CalendarService.
        
        for event in calendar_service._events:
            if event.severity in [CalendarSeverity.HIGH, CalendarSeverity.CRITICAL]:
                # Check ±600s
                diff = (event.scheduled_time_utc - now_utc).total_seconds()
                if abs(diff) <= 600:
                    blackout_active = True
                    blackout_reason = f"{event.event_name} @ {event.scheduled_time_utc}"
                    break # Stop on first blackout match

        # 2. Determine Direction Constraint (Spec 6.3)
        if macro_state == MacroState.MACRO_BULL_GOLD:
            # max(swing_dir, 0) or +1? 
            # Spec text: "max(swing_dir, 0) or +1"
            # This is ambiguous. "OR".
            # Let's interpret typical "Bull" logic: bias is Long (+1).
            # If swing_dir is Short (-1), result is 0 (Block).
            # If swing_dir is Neutral (0) or Long (+1), result is +1? Or matching swing?
            # "max(swing_dir, 0)" -> if -1 -> 0. if 0 -> 0. if 1 -> 1.
            # So it effectively blocks shorts.
            # But the "or +1" part suggests forcing +1?
            # Let's check Test requirements:
            # "MACRO_BULL_GOLD with swing_dir = -1 → direction_constraint = +1"
            # Wait, test requirement says: "with swing_dir = -1 -> direction_constraint = +1"
            # This means we FORCE Long even if swing is Short? That seems like "Counter-Trend".
            # Or does it mean "Bias is +1, so we only allow +1"?
            # If constraint is +1, and strategy wants -1, strategy is blocked.
            # The dispatcher output is `direction_constraint`.
            # If output is +1, then strategies must trade Long.
            # So if swing_dir is -1, and we output +1, we are saying "Only Longs allowed".
            # This matches "MACRO_BULL_GOLD with swing_dir = -1 → direction_constraint = +1".
            direction_constraint = 1
            
        elif macro_state == MacroState.MACRO_BEAR_GOLD:
            direction_constraint = -1
            
        elif macro_state == MacroState.MACRO_NEUTRAL:
            direction_constraint = swing_dir
            
        else: # MACRO_EVENT_RISK
            direction_constraint = 0

        # 3. Determine Permissions via Priority Table (Spec 6.2)
        
        allow_scalp = False
        allow_orb = False
        allow_news = False
        allow_swing_rebalance = True # Default True in all priorities, monitor only
        
        # Priority 1: SHOCK_EVENT or approved news active
        is_shock = (regime == RegimeState.SHOCK_EVENT)
        is_whitelisted_news = (event_risk_state == EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW)
        
        if is_shock or is_whitelisted_news:
            # P1
            allow_scalp = False
            allow_orb = False
            allow_news = True
            # direction_constraint already set from macro_state above (but wait)
            # Spec 6.2 Table says: "direction_constraint = from macro_state" for P1/P2.
            # My logic in step 2 handles this generally.
            
        elif blackout_active:
            # P2
            allow_scalp = False
            allow_orb = False
            allow_news = False
            # allow_swing_rebalance = True (monitor)
            
        else:
            # P3, P4, P5 checks
            hour = now_utc.hour
            is_london_open = (hour == 7) # 07:00-07:59
            # Liquid session: 07:00 to 17:00 (end of 16:59?) or 17:00 included?
            # "07:00-17:00 UTC". Usually implies [07, 17).
            is_liquid = (7 <= hour < 17)
            
            if is_london_open:
                # Priority 3
                allow_scalp = False
                allow_orb = True
                
                # "allow_news = standby (True only if whitelisted event active)"
                # But we handled "whitelisted active" in P1.
                # If we are here, is_whitelisted_news is False.
                # So allow_news is False.
                allow_news = False
                
                # "direction_constraint = from swing_dir"
                # My Step 2 logic used macro_state.
                # Spec 6.2 Table overrides Step 2 for P3/P4/P5?
                # P3 Table: "direction_constraint = from swing_dir"
                # P1/P2 Table: "from macro_state"
                # So I need to re-apply constraint logic based on Priority level.
                direction_constraint = swing_dir

            elif (regime == RegimeState.LOW_VOL) and is_liquid:
                # Priority 4
                allow_scalp = True
                allow_orb = False
                allow_news = False
                direction_constraint = swing_dir
                
            else:
                # Priority 5 (Default)
                allow_scalp = False
                allow_orb = False
                allow_news = False
                direction_constraint = swing_dir

        # Re-apply P1/P2 override for direction constraint?
        # The table says P1/P2 use "from macro_state".
        # P3/P4/P5 use "from swing_dir".
        # So if we are in P1 or P2, we should revert to the logic in Step 2.
        # If we are in P3/P4/P5, we use `swing_dir` directly.
        
        if (is_shock or is_whitelisted_news) or blackout_active:
            # Re-calculate using Step 2 logic (Macro based)
            # Code duplication or reuse?
            # Let's reuse the Step 2 result.
            # But I overwrote it in P3/P4/P5 blocks.
            # Let's fix structure.
            pass 
            # It was calculated in Step 2.
            # If P3/P4/P5, I overwrote it with `swing_dir`.
            # But wait, if I entered the `if is_london_open` block, I am NOT in P1/P2.
            # So the overwrite is correct.
            # However, I need to ensure `direction_constraint` variable holds the MACRO-derived value initially.
            # Yes, Step 2 did that.
            # So:
            # 1. Calc Macro-based constraint (val_macro).
            # 2. P1/P2: use val_macro.
            # 3. P3/P4/P5: use swing_dir.
            pass

        # Refined Logic Flow:
        # 1. Calc Macro Constraint
        if macro_state == MacroState.MACRO_BULL_GOLD:
            macro_constraint = 1
        elif macro_state == MacroState.MACRO_BEAR_GOLD:
            macro_constraint = -1
        elif macro_state == MacroState.MACRO_NEUTRAL:
            macro_constraint = swing_dir
        else: # EVENT_RISK
            macro_constraint = 0
        if is_shock or is_whitelisted_news:
            # P1
            allow_scalp = False
            allow_orb = False
            allow_news = True
            final_constraint = macro_constraint
            
        elif blackout_active:
            # P2
            allow_scalp = False
            allow_orb = False
            allow_news = False
            final_constraint = macro_constraint
            
        elif is_london_open:
            # P3
            allow_scalp = False
            allow_orb = True
            allow_news = False
            final_constraint = swing_dir
            
        elif (regime == RegimeState.LOW_VOL) and is_liquid:
            # P4
            allow_scalp = True
            allow_orb = False
            allow_news = False
            final_constraint = swing_dir
            
        else:
            # P5
            allow_scalp = False
            allow_orb = False
            allow_news = False
            final_constraint = swing_dir

        permissions = DispatcherPermissions(
            allow_swing_rebalance=allow_swing_rebalance,
            allow_orb=allow_orb,
            allow_news=allow_news,
            allow_scalp=allow_scalp,
            direction_constraint=final_constraint,
            macro_bias=macro_state,
            regime=regime,
            risk_scalar=risk_scalar,
            blackout_active=blackout_active,
            blackout_reason=blackout_reason
        )
        
        # Publish Update if Changed
        if self._last_permissions != permissions:
            self._publish_update(permissions, now_utc)
            self._last_permissions = permissions
            
        return permissions

    def _publish_update(self, permissions: DispatcherPermissions, timestamp: datetime) -> None:
        """
        Publish DISPATCHER_UPDATE to Redis. Stub.
        """
        pass
