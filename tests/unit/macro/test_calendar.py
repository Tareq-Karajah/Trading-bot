import pytest
from datetime import datetime, timedelta, timezone
from trading_bot.macro.calendar import MarketCalendarService, CalendarEvent, Tier, CalendarSeverity, EventRiskState

@pytest.fixture
def calendar_service():
    return MarketCalendarService()

@pytest.fixture
def sample_event():
    return CalendarEvent(
        event_id="evt_1",
        event_name="Test Event",
        tier=Tier.T1,
        severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        pre_window_seconds=300,  # 5 min
        post_window_seconds=180, # 3 min
        is_whitelisted_for_news_breakout=True
    )

def test_load_events(calendar_service, sample_event):
    calendar_service.load_events([sample_event])
    assert len(calendar_service._events) == 1
    assert calendar_service._events[0].event_id == "evt_1"

def test_window_classification(calendar_service, sample_event):
    calendar_service.load_events([sample_event])
    sched = sample_event.scheduled_time_utc
    
    # 1. Pre-Event: 5 min before
    # Window: [11:55:00, 12:00:00)
    dt_pre = sched - timedelta(seconds=299)
    assert calendar_service.get_active_event_risk(dt_pre) == EventRiskState.PRE_EVENT_WINDOW
    
    # Boundary check: Exactly start of pre-window
    dt_pre_start = sched - timedelta(seconds=300)
    assert calendar_service.get_active_event_risk(dt_pre_start) == EventRiskState.PRE_EVENT_WINDOW
    
    # Outside Pre-window (before)
    dt_early = sched - timedelta(seconds=301)
    assert calendar_service.get_active_event_risk(dt_early) == EventRiskState.NO_EVENT_RISK

    # 2. Live Event: [12:00:00, 12:03:00]
    # Whitelisted -> WHITELISTED_NEWS_BREAKOUT_WINDOW
    assert calendar_service.get_active_event_risk(sched) == EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW
    assert calendar_service.get_active_event_risk(sched + timedelta(seconds=180)) == EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW
    
    # 3. Post Event: (12:03:00, 12:08:00] (300s cooldown)
    dt_post = sched + timedelta(seconds=181)
    assert calendar_service.get_active_event_risk(dt_post) == EventRiskState.POST_EVENT_WINDOW
    
    dt_post_end = sched + timedelta(seconds=180 + 300)
    assert calendar_service.get_active_event_risk(dt_post_end) == EventRiskState.POST_EVENT_WINDOW
    
    # Outside Post-window (after)
    dt_late = sched + timedelta(seconds=180 + 301)
    assert calendar_service.get_active_event_risk(dt_late) == EventRiskState.NO_EVENT_RISK

def test_non_whitelisted_event(calendar_service):
    event = CalendarEvent(
        event_id="evt_2",
        event_name="Minor Event",
        tier=Tier.T3,
        severity=CalendarSeverity.LOW,
        scheduled_time_utc=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        pre_window_seconds=60,
        post_window_seconds=60,
        is_whitelisted_for_news_breakout=False
    )
    calendar_service.load_events([event])
    sched = event.scheduled_time_utc
    
    # Should be LIVE_EVENT_WINDOW, not WHITELISTED
    assert calendar_service.get_active_event_risk(sched) == EventRiskState.LIVE_EVENT_WINDOW

def test_overlapping_events(calendar_service):
    # Event A: Low severity, active now
    evt_a = CalendarEvent(
        event_id="a", event_name="A", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    # Event B: Critical severity, active now
    evt_b = CalendarEvent(
        event_id="b", event_name="B", tier=Tier.T1, severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=True
    )
    
    calendar_service.load_events([evt_a, evt_b])
    sched = evt_a.scheduled_time_utc
    
    # Severity should be CRITICAL (highest)
    assert calendar_service.get_current_severity(sched) == CalendarSeverity.CRITICAL
    
    # Risk State should be WHITELISTED (highest priority)
    assert calendar_service.get_active_event_risk(sched) == EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW

def test_get_current_severity_tiers(calendar_service):
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Test NONE
    calendar_service.load_events([])
    assert calendar_service.get_current_severity(now) == CalendarSeverity.NONE
    
    # Test LOW
    evt_low = CalendarEvent(
        event_id="l", event_name="L", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    calendar_service.load_events([evt_low])
    assert calendar_service.get_current_severity(now) == CalendarSeverity.LOW
    
    # Test HIGH
    evt_high = CalendarEvent(
        event_id="h", event_name="H", tier=Tier.T2, severity=CalendarSeverity.HIGH,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    calendar_service.load_events([evt_high])
    assert calendar_service.get_current_severity(now) == CalendarSeverity.HIGH
    
    # Test CRITICAL (override)
    evt_crit = CalendarEvent(
        event_id="c", event_name="C", tier=Tier.T1, severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    calendar_service.load_events([evt_high, evt_crit])
    assert calendar_service.get_current_severity(now) == CalendarSeverity.CRITICAL

def test_get_whitelisted_event(calendar_service):
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Non-whitelisted active
    evt_nw = CalendarEvent(
        event_id="nw", event_name="NW", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    calendar_service.load_events([evt_nw])
    assert calendar_service.get_whitelisted_event(now) is None
    
    # Whitelisted active
    evt_w = CalendarEvent(
        event_id="w", event_name="W", tier=Tier.T1, severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=True
    )
    calendar_service.load_events([evt_nw, evt_w])
    result = calendar_service.get_whitelisted_event(now)
    assert result is not None
    assert result.event_id == "w"

def test_naive_datetime_handling(calendar_service, sample_event):
    calendar_service.load_events([sample_event])
    naive_now = datetime(2023, 1, 1, 12, 0, 0) # Naive
    
    # Should not raise error and treat as UTC
    assert calendar_service.get_current_severity(naive_now) == CalendarSeverity.CRITICAL

# --- Coverage Gap Tests ---

def test_calendar_event_naive_datetime_validator():
    # Covers Line 45: ensure_utc validator branch for naive inputs
    naive_dt = datetime(2023, 1, 1, 12, 0, 0)
    event = CalendarEvent(
        event_id="naive", event_name="Naive", tier=Tier.T1, severity=CalendarSeverity.LOW,
        scheduled_time_utc=naive_dt,
        pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=False
    )
    assert event.scheduled_time_utc.tzinfo == timezone.utc

def test_active_risk_naive_datetime_guard(calendar_service):
    # Covers Line 71: get_active_event_risk guard
    naive_now = datetime(2023, 1, 1, 12, 0, 0)
    # Should not raise
    state = calendar_service.get_active_event_risk(naive_now)
    assert state == EventRiskState.NO_EVENT_RISK

def test_whitelisted_event_naive_datetime_guard(calendar_service):
    # Covers Line 144: get_whitelisted_event guard
    naive_now = datetime(2023, 1, 1, 12, 0, 0)
    result = calendar_service.get_whitelisted_event(naive_now)
    assert result is None

def test_whitelisted_event_logic_branch(calendar_service):
    # Covers Lines 159-160: candidate logic when event exists but is not whitelisted/active
    # Case 1: Whitelisted event exists but we are NOT in the window (e.g. POST window)
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Event ended 5 mins ago (POST window)
    evt = CalendarEvent(
        event_id="past", event_name="Past", tier=Tier.T1, severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=now - timedelta(minutes=10),
        pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=True
    )
    calendar_service.load_events([evt])
    
    # Logic: _classify_event_state returns POST_EVENT_WINDOW
    # line 156 check `if state == WHITELISTED` fails
    result = calendar_service.get_whitelisted_event(now)
    assert result is None

def test_whitelisted_event_candidate_replacement(calendar_service):
    # Covers Lines 159-160: elif severity_rank[event.severity] > severity_rank[candidate.severity]
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Event 1: Whitelisted, LOW severity
    evt1 = CalendarEvent(
        event_id="w1", event_name="W1", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=True
    )
    
    # Event 2: Whitelisted, HIGH severity
    evt2 = CalendarEvent(
        event_id="w2", event_name="W2", tier=Tier.T2, severity=CalendarSeverity.HIGH,
        scheduled_time_utc=now, pre_window_seconds=60, post_window_seconds=60, is_whitelisted_for_news_breakout=True
    )
    
    # Load both
    calendar_service.load_events([evt1, evt2])
    
    result = calendar_service.get_whitelisted_event(now)
    assert result is not None
    assert result.event_id == "w2"
    assert result.severity == CalendarSeverity.HIGH
