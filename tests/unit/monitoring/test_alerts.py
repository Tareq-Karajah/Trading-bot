import pytest
from datetime import datetime, timezone
from trading_bot.monitoring.alerts import AlertManager, AlertSeverity

@pytest.fixture
def alert_manager():
    return AlertManager()

def test_daily_loss_critical(alert_manager):
    # Threshold: < -3.0
    alerts = alert_manager.check_and_fire("daily_pnl_pct", -3.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "daily_loss_critical"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].threshold == -3.0
    assert "PagerDuty" in alerts[0].channel
    assert "Telegram" in alerts[0].channel

def test_daily_loss_warning(alert_manager):
    # Threshold: < -2.0
    alerts = alert_manager.check_and_fire("daily_pnl_pct", -2.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "daily_loss_warning"
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].threshold == -2.0
    assert "Telegram" in alerts[0].channel
    
    # Boundary check: -2.0 (should not fire)
    alerts = alert_manager.check_and_fire("daily_pnl_pct", -2.0, {})
    assert len(alerts) == 0
    
    # Boundary check: -1.9 (should not fire)
    alerts = alert_manager.check_and_fire("daily_pnl_pct", -1.9, {})
    assert len(alerts) == 0

def test_weekly_drawdown(alert_manager):
    # Threshold: < -5.0
    alerts = alert_manager.check_and_fire("weekly_pnl_pct", -5.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "weekly_drawdown"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].threshold == -5.0
    assert "PagerDuty" in alerts[0].channel
    assert "Email" in alerts[0].channel

def test_peak_drawdown(alert_manager):
    # Threshold: > 15.0
    alerts = alert_manager.check_and_fire("active_drawdown_pct", 15.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "peak_drawdown"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].threshold == 15.0

def test_api_error_rate(alert_manager):
    # Threshold: > 5.0
    alerts = alert_manager.check_and_fire("api_error_rate", 5.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "api_error_rate"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].threshold == 5.0

def test_strategy_viability(alert_manager):
    # Threshold: < 0.0 (negative rolling ER)
    alerts = alert_manager.check_and_fire("rolling_expected_return", -0.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "strategy_viability"
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].threshold == 0.0

def test_strategy_suspended(alert_manager):
    # Threshold: > 0.0 (True)
    alerts = alert_manager.check_and_fire("strategy_suspended", 1.0, {"strategy": "TEST"})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "strategy_suspended"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert "TEST" in alerts[0].message

def test_dead_man_switch_alert(alert_manager):
    # Threshold: > 60.0
    alerts = alert_manager.check_and_fire("seconds_since_heartbeat", 61.0, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "dead_man_switch"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].threshold == 60.0

def test_slippage_alert(alert_manager):
    # Threshold: > 3.0
    alerts = alert_manager.check_and_fire("slippage_ratio", 3.1, {})
    assert len(alerts) == 1
    assert alerts[0].alert_name == "slippage_alert"
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[0].threshold == 3.0

def test_alert_naive_datetime():
    from trading_bot.monitoring.alerts import Alert
    
    # Create Alert with naive datetime
    naive_dt = datetime(2023, 1, 1, 12, 0, 0)
    alert = Alert(
        alert_name="test",
        severity=AlertSeverity.WARNING,
        channel=["test"],
        message="test",
        value=1.0,
        threshold=1.0,
        timestamp_utc=naive_dt
    )
    
    assert alert.timestamp_utc.tzinfo == timezone.utc
