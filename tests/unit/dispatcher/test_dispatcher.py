import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from trading_bot.dispatcher.engine import StrategyDispatcher, DispatcherPermissions
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState
from trading_bot.macro.calendar import MarketCalendarService, EventRiskState, CalendarEvent, Tier, CalendarSeverity

@pytest.fixture
def dispatcher():
    return StrategyDispatcher()

@pytest.fixture
def mock_calendar():
    service = MagicMock(spec=MarketCalendarService)
    service._events = [] # Allow direct access for blackout check
    return service

def create_event(name, time, severity=CalendarSeverity.HIGH):
    return CalendarEvent(
        event_id="e1", event_name=name, tier=Tier.T1, severity=severity,
        scheduled_time_utc=time, pre_window_seconds=60, post_window_seconds=60,
        is_whitelisted_for_news_breakout=True
    )

# --- Priority 1 Tests ---

def test_p1_shock_event(dispatcher, mock_calendar):
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.SHOCK_EVENT,
        macro_state=MacroState.MACRO_NEUTRAL,
        risk_scalar=0.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    
    assert perms.allow_news is True
    assert perms.allow_scalp is False
    assert perms.allow_orb is False
    assert perms.direction_constraint == 1 # Neutral -> swing_dir

def test_p1_whitelisted_news(dispatcher, mock_calendar):
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.MID_VOL,
        macro_state=MacroState.MACRO_NEUTRAL,
        risk_scalar=1.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW
    )
    
    assert perms.allow_news is True
    assert perms.allow_scalp is False
    assert perms.allow_orb is False

# --- Priority 2 Tests ---

def test_p2_blackout_window(dispatcher, mock_calendar):
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    # Event 5 mins from now (within 600s)
    evt = create_event("Blackout Event", now + timedelta(minutes=5), CalendarSeverity.CRITICAL)
    mock_calendar._events = [evt]
    
    perms = dispatcher.evaluate(
        regime=RegimeState.MID_VOL,
        macro_state=MacroState.MACRO_NEUTRAL,
        risk_scalar=1.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.PRE_EVENT_WINDOW
    )
    
    assert perms.blackout_active is True
    assert "Blackout Event" in perms.blackout_reason
    assert perms.allow_news is False # Blackout overrides News (unless P1 whitelisted active, but here we passed PRE)
    assert perms.allow_scalp is False
    assert perms.allow_orb is False

def test_blackout_boundary(dispatcher, mock_calendar):
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    # Exactly 600s -> Blackout
    evt_in = create_event("In", now + timedelta(seconds=600))
    mock_calendar._events = [evt_in]
    perms = dispatcher.evaluate(RegimeState.MID_VOL, MacroState.MACRO_NEUTRAL, 1.0, 1, now, mock_calendar, EventRiskState.NO_EVENT_RISK)
    assert perms.blackout_active is True
    
    # 601s -> No Blackout
    evt_out = create_event("Out", now + timedelta(seconds=601))
    mock_calendar._events = [evt_out]
    perms = dispatcher.evaluate(RegimeState.MID_VOL, MacroState.MACRO_NEUTRAL, 1.0, 1, now, mock_calendar, EventRiskState.NO_EVENT_RISK)
    assert perms.blackout_active is False

# --- Priority 3 Tests ---

def test_p3_london_open(dispatcher, mock_calendar):
    # 07:30 UTC
    now = datetime(2023, 1, 1, 7, 30, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.MID_VOL,
        macro_state=MacroState.MACRO_BULL_GOLD, # Macro says +1
        risk_scalar=1.0,
        swing_dir=-1, # Swing says -1
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    
    assert perms.allow_orb is True
    assert perms.allow_scalp is False
    # P3 uses swing_dir for constraint, ignoring Macro
    assert perms.direction_constraint == -1 

# --- Priority 4 Tests ---

def test_p4_low_vol_liquid(dispatcher, mock_calendar):
    # 10:00 UTC (Liquid) + LOW_VOL
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.LOW_VOL,
        macro_state=MacroState.MACRO_BULL_GOLD,
        risk_scalar=1.0,
        swing_dir=-1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    
    assert perms.allow_scalp is True
    assert perms.allow_orb is False
    # P4 uses swing_dir
    assert perms.direction_constraint == -1

# --- Priority 5 Tests ---

def test_p5_default(dispatcher, mock_calendar):
    # 20:00 UTC (Illiquid)
    now = datetime(2023, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.LOW_VOL,
        macro_state=MacroState.MACRO_NEUTRAL,
        risk_scalar=1.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    
    assert perms.allow_scalp is False
    assert perms.allow_orb is False
    assert perms.allow_news is False
    assert perms.direction_constraint == 1

# --- Direction Constraint Tests (Macro Logic) ---

def test_direction_constraint_bull(dispatcher, mock_calendar):
    # Trigger P1 (Shock) to use Macro logic
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    # Bull + Swing(-1) -> +1
    perms = dispatcher.evaluate(
        regime=RegimeState.SHOCK_EVENT,
        macro_state=MacroState.MACRO_BULL_GOLD,
        risk_scalar=0.0,
        swing_dir=-1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    assert perms.direction_constraint == 1

def test_direction_constraint_bear(dispatcher, mock_calendar):
    # Trigger P1
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    # Bear + Swing(+1) -> -1
    perms = dispatcher.evaluate(
        regime=RegimeState.SHOCK_EVENT,
        macro_state=MacroState.MACRO_BEAR_GOLD,
        risk_scalar=0.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    assert perms.direction_constraint == -1

def test_direction_constraint_event_risk(dispatcher, mock_calendar):
    # Trigger P1
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    perms = dispatcher.evaluate(
        regime=RegimeState.SHOCK_EVENT,
        macro_state=MacroState.MACRO_EVENT_RISK,
        risk_scalar=0.0,
        swing_dir=1,
        now_utc=now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    assert perms.direction_constraint == 0

# --- Redis Event Tests ---

def test_redis_publish_deduplication(dispatcher, mock_calendar):
    now = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    with patch.object(dispatcher, '_publish_update') as mock_pub:
        # Call 1
        dispatcher.evaluate(RegimeState.MID_VOL, MacroState.MACRO_NEUTRAL, 1.0, 1, now, mock_calendar, EventRiskState.NO_EVENT_RISK)
        mock_pub.assert_called_once()
        
        # Call 2 (Identical)
        mock_pub.reset_mock()
        dispatcher.evaluate(RegimeState.MID_VOL, MacroState.MACRO_NEUTRAL, 1.0, 1, now, mock_calendar, EventRiskState.NO_EVENT_RISK)
        mock_pub.assert_not_called()
        
        # Call 3 (Change)
        mock_pub.reset_mock()
        dispatcher.evaluate(RegimeState.SHOCK_EVENT, MacroState.MACRO_NEUTRAL, 1.0, 1, now, mock_calendar, EventRiskState.NO_EVENT_RISK)
        mock_pub.assert_called_once()

# --- Coverage Gap Tests ---

def test_evaluate_naive_datetime(dispatcher, mock_calendar):
    # Covers Line 45: naive datetime handling
    naive_now = datetime(2023, 1, 1, 10, 0, 0) # No tzinfo
    perms = dispatcher.evaluate(
        regime=RegimeState.MID_VOL,
        macro_state=MacroState.MACRO_NEUTRAL,
        risk_scalar=1.0,
        swing_dir=1,
        now_utc=naive_now,
        calendar_service=mock_calendar,
        event_risk_state=EventRiskState.NO_EVENT_RISK
    )
    # Should not crash, result logic irrelevant but must succeed
    assert perms is not None
