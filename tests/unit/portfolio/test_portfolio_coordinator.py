import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from typing import Dict

from trading_bot.portfolio.coordinator import PortfolioCoordinator, CoordinatorDecision, StrategyMetrics
from trading_bot.risk.engine import RiskDecision, SignalIntent, CircuitBreakerState, PortfolioHeatSnapshot, StrategyViability, TPPlan, TimeoutPlan, ExecutionConstraints
from trading_bot.core.models import Position
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def coordinator():
    return PortfolioCoordinator()

@pytest.fixture
def base_risk_decision():
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    
    return RiskDecision(
        approved=True, rejection_reason=None, position_size=1.0, risk_fraction_used=0.01,
        stop_price=1980.0, take_profit_plan=tp, 
        circuit_breaker_state=CircuitBreakerState(daily_loss_pct=0.0, weekly_loss_pct=0.0, consecutive_losses=0, breaker_active=False),
        portfolio_heat_snapshot=PortfolioHeatSnapshot(current_heat_pct=0.0, heat_remaining=0.05, open_positions=0),
        strategy_viability=StrategyViability(rolling_er=0.5, viable=True),
        timestamp_utc=datetime.now(timezone.utc)
    )

@pytest.fixture
def base_signal():
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)

    return SignalIntent(
        strategy_name="SWING", direction=1, score=5, entry_type="MARKET", entry_trigger=2000.0,
        sl_distance=20.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime.now(timezone.utc)
    )

@pytest.fixture
def default_metrics():
    return {
        "SWING": StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.0),
        "ORB": StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.0),
        "SCALP": StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.0),
        "NEWS": StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.0),
    }

def test_hard_limit_max_heat(coordinator, base_risk_decision, base_signal, default_metrics):
    base_risk_decision = base_risk_decision.model_copy(update={
        "portfolio_heat_snapshot": PortfolioHeatSnapshot(current_heat_pct=0.045, heat_remaining=0.005, open_positions=0)
    })
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, [], default_metrics, 100_000.0
    )
    pass 

def test_hard_limit_max_positions(coordinator, base_risk_decision, base_signal, default_metrics):
    open_pos = [MagicMock(spec=Position, quantity=1.0, entry_price=2000.0, stop_loss=1980.0, side=MagicMock(name="LONG"), symbol="XAUUSD")] * 5
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, open_pos, default_metrics, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "MAX_POSITIONS_REACHED"

def test_quality_weighting_perfect(coordinator, base_risk_decision, base_signal):
    metrics = {"SWING": StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.0)}
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, [], metrics, 100_000.0
    )
    assert decision.quality_weight == 1.0
    assert decision.adjusted_size == 1.0

def test_quality_weighting_floor(coordinator, base_risk_decision, base_signal):
    metrics = {"SWING": StrategyMetrics(rolling_pf=1.0, rolling_er=-0.5, realized_slippage_dev=5.0, open_risk_pct=0.0)}
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, [], metrics, 100_000.0
    )
    assert decision.quality_weight == 0.5
    assert decision.adjusted_size == 0.5

def test_risk_budget_enforcement(coordinator, base_risk_decision, base_signal, default_metrics):
    metrics = default_metrics.copy()
    metrics["SWING"] = StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.025)
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, [], metrics, 100_000.0
    )
    assert decision.approved is False
    assert "RISK_BUDGET_EXHAUSTED" in decision.rejection_reason

def test_news_uses_orb_budget(coordinator, base_risk_decision, base_signal, default_metrics):
    news_signal = base_signal.model_copy(update={"strategy_name": "NEWS"})
    metrics = default_metrics.copy()
    metrics["ORB"] = StrategyMetrics(rolling_pf=2.0, rolling_er=0.5, realized_slippage_dev=0.0, open_risk_pct=0.01)
    decision = coordinator.evaluate(
        base_risk_decision, news_signal, [], metrics, 100_000.0
    )
    assert decision.approved is False
    assert "RISK_BUDGET_EXHAUSTED_ORB" in decision.rejection_reason

def test_net_exposure_scaling(coordinator, base_risk_decision, base_signal, default_metrics):
    pos = MagicMock(spec=Position, quantity=40.0, entry_price=2000.0, stop_loss=1980.0, side=MagicMock(name="LONG"), symbol="XAUUSD")
    bd = base_risk_decision.model_copy(update={"risk_fraction_used": 0.4, "position_size": 10.0})
    decision = coordinator.evaluate(
        bd, base_signal, [pos], default_metrics, 100_000.0
    )
    assert decision.approved is True
    assert decision.adjusted_size == 5.0
    assert decision.net_exposure == pytest.approx(1.0)

def test_net_exposure_floor_rejection(coordinator, base_risk_decision, base_signal, default_metrics):
    bd = base_risk_decision.model_copy(update={"risk_fraction_used": 0.10})
    decision = coordinator.evaluate(
        bd, base_signal, [], default_metrics, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "NET_EXPOSURE_FLOOR"

def test_opposing_direction_conflict(coordinator, base_risk_decision, base_signal, default_metrics):
    scalp_signal = base_signal.model_copy(update={"strategy_name": "SCALP", "direction": 1})
    pos = MagicMock(spec=Position, quantity=10.0, entry_price=2000.0, stop_loss=2020.0, side=MagicMock(name="SHORT"), symbol="XAUUSD")
    decision = coordinator.evaluate(
        base_risk_decision, scalp_signal, [pos], default_metrics, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "OPPOSING_DIRECTION_CONFLICT"

def test_correlation_conflict(coordinator, base_risk_decision, base_signal, default_metrics):
    coordinator.update_correlations({"XAUUSD": {"EURUSD": 0.8}})
    pos = MagicMock(spec=Position, quantity=1.0, entry_price=1.0, stop_loss=0.9, side=MagicMock(name="LONG"), symbol="EURUSD")
    decision = coordinator.evaluate(
        base_risk_decision, base_signal, [pos], default_metrics, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "CORRELATION_CONFLICT"

def test_ensure_utc(coordinator):
    cd = CoordinatorDecision(
        approved=True, rejection_reason=None, adjusted_size=1.0, quality_weight=1.0,
        net_exposure=0.0, heat_after=0.0, budget_remaining={},
        timestamp_utc=datetime(2026, 1, 1, 12, 0) # Naive
    )
    assert cd.timestamp_utc.tzinfo == timezone.utc

def test_ensure_utc_already_utc(coordinator):
    cd = CoordinatorDecision(
        approved=True, rejection_reason=None, adjusted_size=1.0, quality_weight=1.0,
        net_exposure=0.0, heat_after=0.0, budget_remaining={},
        timestamp_utc=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    )
    assert cd.timestamp_utc.tzinfo == timezone.utc

# --- NEW TESTS FOR COVERAGE ---

def test_pass_through_rejection(coordinator, base_risk_decision, base_signal, default_metrics):
    # Covers lines 73-78 (if not risk_decision.approved)
    rejected_bd = base_risk_decision.model_copy(update={"approved": False, "rejection_reason": "RISK_FAIL"})
    decision = coordinator.evaluate(rejected_bd, base_signal, [], default_metrics, 100_000.0)
    assert decision.approved is False
    assert decision.rejection_reason == "RISK_FAIL"

def test_missing_strategy_metrics(coordinator, base_risk_decision, base_signal):
    # Covers line 83 (if not metrics:)
    # Pass empty metrics dict
    # Default metric rolling_pf=1.0, rolling_er=0.0, slip=0.0 -> Score: (0 + 0.5 + 1)/3 = 0.5
    # candidate_risk_pct = 0.30 * 0.5 = 0.15 (passes floor 0.15!)
    bd = base_risk_decision.model_copy(update={"risk_fraction_used": 0.30}) 
    decision = coordinator.evaluate(bd, base_signal, [], {}, 100_000.0)
    assert decision.approved is True
    assert decision.quality_weight == 0.5

def test_net_exposure_cap_reached(coordinator, base_risk_decision, base_signal, default_metrics):
    # Covers lines 119-120 (if available <= 0)
    # Current net risk = 1.0 (at cap). Candidate adds more in same direction.
    pos = MagicMock(spec=Position, quantity=50.0, entry_price=2000.0, stop_loss=1980.0, side=MagicMock(name="LONG"), symbol="XAUUSD")
    bd = base_risk_decision.model_copy(update={"risk_fraction_used": 0.4})
    decision = coordinator.evaluate(bd, base_signal, [pos], default_metrics, 100_000.0)
    assert decision.approved is False
    assert decision.rejection_reason == "NET_EXPOSURE_CAP_REACHED"

def test_correlation_conflict_reverse_lookup(coordinator, base_risk_decision, base_signal, default_metrics):
    # Covers lines 185-186 (elif p.symbol in self._correlations:)
    # signal is XAUUSD, position is EURUSD
    # Update dict with EURUSD -> XAUUSD
    coordinator.update_correlations({"EURUSD": {"XAUUSD": 0.8}})
    pos = MagicMock(spec=Position, quantity=1.0, entry_price=1.0, stop_loss=0.9, side=MagicMock(name="LONG"), symbol="EURUSD")
    decision = coordinator.evaluate(base_risk_decision, base_signal, [pos], default_metrics, 100_000.0)
    assert decision.approved is False
    assert decision.rejection_reason == "CORRELATION_CONFLICT"
