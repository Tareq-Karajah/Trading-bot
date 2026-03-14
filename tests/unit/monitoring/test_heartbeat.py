import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from trading_bot.monitoring.heartbeat import DeadManSwitch

@pytest.fixture
def dms():
    return DeadManSwitch()

def test_beat_updates_timestamp(dms):
    original_beat = dms._last_beat
    # Mock datetime.now to ensure it moves forward
    with patch("trading_bot.monitoring.heartbeat.datetime") as mock_dt:
        mock_dt.now.return_value = original_beat + timedelta(seconds=10)
        # We need to ensure ensure_utc works if called inside check/seconds_since...
        # But beat() just calls now().
        # Mocking datetime.now requires careful handling of side effects if used multiple times.
        # Simplest is to just sleep or rely on mock return value change.
        
        # Let's just create a new DMS and set _last_beat manually for better control
        pass

    # Better approach:
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    with patch("trading_bot.monitoring.heartbeat.datetime") as mock_dt:
        new_time = datetime(2023, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        mock_dt.now.return_value = new_time
        dms.beat()
        
    assert dms._last_beat == new_time

def test_check_returns_true_within_60s(dms):
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # 59 seconds later
    now = datetime(2023, 1, 1, 12, 0, 59, tzinfo=timezone.utc)
    assert dms.check(now) is True

def test_check_returns_true_at_60s(dms):
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # 60 seconds later (exact boundary)
    # Usually check returns True if elapsed <= timeout.
    # 60 <= 60 is True.
    now = datetime(2023, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
    assert dms.check(now) is True

def test_check_returns_false_after_60s(dms):
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # 61 seconds later
    now = datetime(2023, 1, 1, 12, 1, 1, tzinfo=timezone.utc)
    assert dms.check(now) is False

def test_seconds_since_last_beat(dms):
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    now = datetime(2023, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    
    assert dms.seconds_since_last_beat(now) == 30.0

def test_naive_datetime_input(dms):
    # DMS should handle naive datetime by assuming/converting to UTC or erroring?
    # Spec usually implies UTC everywhere.
    # Implementation should handle it safely.
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Naive input
    now_naive = datetime(2023, 1, 1, 12, 0, 30) 
    # Python 3.12+ might warn, but let's see implementation.
    # If implementation forces UTC, it should work if values are comparable or converted.
    # Comparison between offset-naive and offset-aware raises TypeError.
    # So implementation MUST convert.
    
    # Test that it DOES NOT raise
    try:
        assert dms.check(now_naive) is True
    except TypeError:
        pytest.fail("DeadManSwitch.check failed with naive datetime")

def test_seconds_since_last_beat_naive(dms):
    dms._last_beat = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    now_naive = datetime(2023, 1, 1, 12, 0, 30)
    
    # Should convert and work
    assert dms.seconds_since_last_beat(now_naive) == 30.0
