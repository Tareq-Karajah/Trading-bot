import pytest
from datetime import datetime, timezone, time
from unittest.mock import MagicMock

from trading_bot.execution.quality_gate import ExecutionQualityGate, ExecutionDecision
from trading_bot.risk.engine import SignalIntent, ExecutionConstraints, TPPlan, TimeoutPlan
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def gate():
    return ExecutionQualityGate()

@pytest.fixture
def base_signal():
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    
    return SignalIntent(
        strategy_name="SWING", direction=1, score=5, entry_type="MARKET", entry_trigger=2000.0,
        sl_distance=20.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime.now(timezone.utc)
    )

def test_blocked_spread_ratio(gate, base_signal):
    # Ratio > 3.0 -> Blocked
    # Spread 4.0, Median 1.0 => Ratio 4.0
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=4.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "BLOCKED"
    assert "SPREAD_RATIO_HIGH" in decision.blocked_reason
    assert decision.routing_mode is None

def test_blocked_spread_ratio_boundary(gate, base_signal):
    # Ratio = 3.0 exactly -> Not blocked by this (Assuming Allowed/Degraded)
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=3.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    # 3.0 is Degradation threshold (1.8 < 3.0). So DEGRADED.
    assert decision.allowed_state == "DEGRADED" 
    assert decision.blocked_reason is None

def test_blocked_quote_age_swing(gate, base_signal):
    # SWING allows 2000ms.
    # Age 1500ms. Should pass (Allowed).
    # Even though Global Limit is 1000, Strategy overrides.
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=1500,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "ALLOWED"

    # Age 2001ms. Blocked.
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=2001,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    assert decision.allowed_state == "BLOCKED"
    assert "STRATEGY_QUOTE_STALE" in decision.blocked_reason

def test_blocked_api_error(gate, base_signal):
    # Error >= 0.05 -> Blocked
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.05,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "BLOCKED"
    assert "API_ERROR_RATE" in decision.blocked_reason

def test_blocked_api_error_boundary(gate, base_signal):
    # Error 0.049 -> Allowed
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.049,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "ALLOWED"

def test_session_swing_allowed(gate, base_signal):
    # SWING has None session. Always allowed.
    # Test at 00:00 (Midnight)
    
    now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    
    assert decision.allowed_state == "ALLOWED"

def test_session_orb_blocked(gate, base_signal):
    # ORB 07:00-20:00.
    # Test 06:59 -> Blocked.
    
    orb_signal = base_signal.model_copy(update={"strategy_name": "ORB"})
    # Update constraints manually as SignalIntent usually carries them
    # But `gate` looks up session from its own dict.
    # It uses `signal.execution_constraints` for numeric limits.
    # So we don't need to update signal constraints for session test.
    
    now = datetime(2026, 1, 1, 6, 59, tzinfo=timezone.utc)
    decision = gate.evaluate(
        orb_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    
    assert decision.allowed_state == "BLOCKED"
    assert "OUTSIDE_SESSION" in decision.blocked_reason

def test_session_scalp_narrow(gate, base_signal):
    # SCALP 08:00-17:00.
    scalp_signal = base_signal.model_copy(update={"strategy_name": "SCALP"})
    
    # 08:00 -> Allowed
    now = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    decision = gate.evaluate(
        scalp_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "ALLOWED"
    
    # 17:00 -> Allowed
    now = datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)
    decision = gate.evaluate(
        scalp_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "ALLOWED"
    
    # 17:01 -> Blocked
    now = datetime(2026, 1, 1, 17, 1, tzinfo=timezone.utc)
    decision = gate.evaluate(
        scalp_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "BLOCKED"

def test_degraded_spread_ratio(gate, base_signal):
    # Ratio > 1.80 -> Degraded
    # Ratio 2.0.
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=2.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "DEGRADED"
    assert decision.degraded_action == "REDUCE_SIZE_50PCT"
    assert decision.routing_mode == "LIMIT"

def test_degraded_slippage(gate, base_signal):
    # Expected slippage > max_slippage -> Degraded
    # Max slippage 15.0 (Swing).
    # Spread 20.0 triggers MAX_SPREAD_EXCEEDED (Swing limit is 10.0)!
    # We must use spread < 10.0.
    # Use Spread 8.0.
    # Expected = 4.0 + impact.
    # Need impact > 11.0.
    # 11 = (lots / 10000) * 100 -> lots = 1100.
    
    decision = gate.evaluate(
        base_signal, spread_live_bp=8.0, spread_median_100=8.0, quote_age_ms=100,
        order_lots=1200.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.allowed_state == "DEGRADED"
    assert decision.expected_slippage_bp > 15.0

def test_per_strategy_spread_constraint(gate, base_signal):
    # ORB max spread 3.0.
    # Spread 3.1 -> Blocked.
    
    orb_ec = ExecutionConstraints(max_spread_bp=3.0, max_slippage_bp=5.0, min_quote_fresh_ms=500)
    orb_signal = base_signal.model_copy(update={"strategy_name": "ORB", "execution_constraints": orb_ec})
    
    # Ensure Median is high enough so Ratio < 3.0 (Global Block)
    # Median 2.0 -> Ratio 1.55 (OK).
    
    # Also ensure time is within session
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    decision = gate.evaluate(
        orb_signal, spread_live_bp=3.1, spread_median_100=2.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    
    assert decision.allowed_state == "BLOCKED"
    assert "MAX_SPREAD_EXCEEDED" in decision.blocked_reason

def test_scalp_quote_freshness(gate, base_signal):
    # SCALP limit 300ms.
    # 299ms -> Allowed.
    
    scalp_ec = ExecutionConstraints(max_spread_bp=2.0, max_slippage_bp=3.0, min_quote_fresh_ms=300)
    scalp_signal = base_signal.model_copy(update={"strategy_name": "SCALP", "execution_constraints": scalp_ec})
    
    # Mock time to be within session (08:00-17:00)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    decision = gate.evaluate(
        scalp_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=299,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "ALLOWED"
    
    # 301ms -> Blocked
    decision = gate.evaluate(
        scalp_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=301,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "BLOCKED"

def test_news_session_extended(gate, base_signal):
    # NEWS allowed until 21:00.
    # 20:30 -> Allowed.
    
    news_ec = ExecutionConstraints(max_spread_bp=3.0, max_slippage_bp=5.0, min_quote_fresh_ms=500)
    news_signal = base_signal.model_copy(update={"strategy_name": "NEWS", "execution_constraints": news_ec})
    
    now = datetime(2026, 1, 1, 20, 30, tzinfo=timezone.utc)
    decision = gate.evaluate(
        news_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "ALLOWED"
    
    # 21:01 -> Blocked
    now = datetime(2026, 1, 1, 21, 1, tzinfo=timezone.utc)
    decision = gate.evaluate(
        news_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=now
    )
    assert decision.allowed_state == "BLOCKED"

def test_execution_decision_schema(gate, base_signal):
    # Verify fields
    decision = gate.evaluate(
        base_signal, spread_live_bp=1.0, spread_median_100=1.0, quote_age_ms=100,
        order_lots=1.0, adv_daily_lots=10000.0, api_error_rate_5min=0.0,
        now_utc=datetime.now(timezone.utc)
    )
    
    assert decision.order_intent_id.startswith("sig_")
    assert decision.routing_mode == "LIMIT"
    assert decision.timestamp_utc.tzinfo == timezone.utc
    assert decision.actual_slippage_bp is None

def test_ensure_utc_validator(gate):
    # Test line 29: ensure_utc validator
    # Pass naive datetime, should become UTC
    naive = datetime(2026, 1, 1, 12, 0)
    decision = ExecutionDecision(
        allowed_state="ALLOWED", blocked_reason=None, spread_snapshot=1.0, spread_vs_median=1.0,
        quote_freshness_ms=100, expected_slippage_bp=1.0, routing_mode="LIMIT",
        order_intent_id="test", timestamp_utc=naive
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc
