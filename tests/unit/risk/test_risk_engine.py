import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from typing import List

from trading_bot.risk.engine import (
    RiskEngine, RiskDecision, SignalIntent, 
    TPPlan, TimeoutPlan, ExecutionConstraints,
    CircuitBreakerState, PortfolioHeatSnapshot, StrategyViability
)
from trading_bot.core.models import Position
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def risk_engine():
    return RiskEngine()

@pytest.fixture
def base_signal():
    return SignalIntent(
        strategy_name="SWING",
        direction=1,
        score=5,
        entry_type="MARKET",
        entry_trigger=2000.0,
        sl_distance=20.0,
        tp_plan=TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0),
        timeout_plan=TimeoutPlan(max_bars=10, mandatory_exit_utc=None),
        regime_context=RegimeState.LOW_VOL,
        macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    )

def test_position_sizing_base(risk_engine, base_signal):
    # SWING r = 0.01
    equity = 100_000.0
    contract_oz = 100.0 # Standard lot
    sl_dist = 20.0
    
    # Base lots = (100k * 0.01) / (20 * 100) = 1000 / 2000 = 0.5 lots
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [], 0.0, 0.0, equity
    )
    
    assert decision.approved is True
    assert decision.position_size == 0.5
    assert decision.risk_fraction_used == 0.01

def test_position_sizing_modifiers(risk_engine, base_signal):
    equity = 100_000.0
    contract_oz = 100.0
    
    # Base 0.5 lots
    # Modifiers: Regime 0.5 (HIGH_VOL), Macro 0.5 (EVENT), Quality 0.8
    # 0.5 * 0.5 * 0.5 * 0.8 = 0.5 * 0.2 = 0.1 lots
    
    decision = risk_engine.evaluate(
        base_signal, equity, 0.5, 0.5, 0.8, contract_oz, 0.01, [], 0.0, 0.0, equity
    )
    
    assert decision.position_size == 0.1
    assert decision.risk_fraction_used == pytest.approx(0.01 * 0.5 * 0.5 * 0.8)

def test_broker_min_lot(risk_engine, base_signal):
    equity = 2000.0 # Increased equity so 0.01 lot is < 1.5% risk (Risk=20/2000=1%)
    contract_oz = 100.0
    sl_dist = 20.0
    # Base = (2000 * 0.01) / 2000 = 0.01 lots
    # If we lower equity slightly to make base < 0.01
    equity = 1500.0 
    # Base = 15 / 2000 = 0.0075
    # Clamped to 0.01.
    # Risk = 0.01 * 100 * 20 = 20.
    # 20 / 1500 = 1.33% < 1.5%. OK.
    
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [], 0.0, 0.0, equity
    )
    
    assert decision.position_size == 0.01 # Clamped to min

def test_circuit_breaker_shock(risk_engine, base_signal):
    # Shock -> regime_risk_scalar = 0.0
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 0.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "SHOCK_EVENT_ACTIVE"

def test_circuit_breaker_daily_hard(risk_engine, base_signal):
    # Daily PnL -3.1%
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], -0.031, 0.0, 100_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "DAILY_LOSS_HARD_LIMIT"
    assert decision.circuit_breaker_state.breaker_active is True

def test_circuit_breaker_daily_soft_scalp(risk_engine, base_signal):
    scalp_signal = base_signal.model_copy(update={"strategy_name": "SCALP"})
    
    # PnL -1.6% (SCALP limit -1.5%)
    decision = risk_engine.evaluate(
        scalp_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], -0.016, 0.0, 100_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "DAILY_LOSS_SOFT_LIMIT_SCALP"
    
    # Should persist
    decision2 = risk_engine.evaluate(
        scalp_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], -0.010, 0.0, 100_000.0
    )
    assert decision2.approved is False
    assert decision2.rejection_reason == "DAILY_LOSS_SOFT_LIMIT_SCALP"

def test_circuit_breaker_weekly_hard(risk_engine, base_signal):
    # Weekly -8.1%
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, -0.081, 100_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "WEEKLY_LOSS_LIMIT"

def test_circuit_breaker_peak_drawdown(risk_engine, base_signal):
    # Peak equity 200k. Current 160k. DD = 40k/200k = 20% > 15%
    decision = risk_engine.evaluate(
        base_signal, 160_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 200_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "PEAK_DRAWDOWN_SHUTDOWN"

def test_consecutive_loss_reduction(risk_engine, base_signal):
    # Need ER > 0 to avoid suspension.
    # Record 10 wins first.
    for _ in range(10):
        risk_engine.record_trade_result("SWING", 1.0, True)
        
    # 5 losses
    for _ in range(5):
        risk_engine.record_trade_result("SWING", -1.0, False)
        
    # Should trigger reduction
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    
    # Base 0.5. Reduced 50% -> 0.25
    assert decision.position_size == 0.25
    
    # 10 trades later -> back to normal?
    # Actually logic says "for next 10 trades".
    # Evaluating DOES NOT decrement counter. record_trade_result does.
    # So we need to simulate 10 trades.
    
    # Let's verify counter logic.
    # We are in reduction mode.
    for _ in range(10):
        # We record trades (win or loss doesn't matter for counter decrement)
        risk_engine.record_trade_result("SWING", 1.0, True)
        
    # Now counter should be 0.
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.position_size == 0.5

def test_viability_monitor_suspend(risk_engine, base_signal):
    # ER < -0.10 -> Suspend
    # Need to populate history.
    # p*b - (1-p) < -0.10
    # If all losses, p=0. ER = -1.
    
    risk_engine.record_trade_result("SWING", -1.0, False)
    # Only 1 trade? VIABILITY_WINDOW is 30.
    # Logic: "Fewer than 30 trades → use available trades"
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "STRATEGY_SUSPENDED_LOW_ER"
    assert decision.strategy_viability.viable is False

def test_viability_monitor_degrade(risk_engine, base_signal):
    # ER > -0.10 but <= 0 OR PF <= 1.10
    # ER=0 -> Mod 0.5
    # p=0.5. b=1.0. ER = 0.5*1 - 0.5 = 0.
    
    risk_engine.record_trade_result("SWING", 1.0, True)
    risk_engine.record_trade_result("SWING", -1.0, False)
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    
    assert decision.approved is True
    # Base 0.5 * 0.5 (Viability) = 0.25
    assert decision.position_size == 0.25
    assert decision.strategy_viability.viable is False # mod 0.5 implies not viable

def test_portfolio_heat_cap(risk_engine, base_signal):
    equity = 100_000.0
    contract_oz = 100.0
    # 1 lot risk = 1 * 100 * 20 = $2000 = 2% heat.
    
    # Create existing position with 4% heat (2 lots)
    pos = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=2.0,
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    
    # New trade: 0.5 lots = 1% heat. Total 5%. Should PASS.
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [pos], 0.0, 0.0, equity
    )
    assert decision.approved is True
    assert decision.portfolio_heat_snapshot.current_heat_pct == pytest.approx(0.04)
    assert decision.portfolio_heat_snapshot.heat_remaining == pytest.approx(0.01)
    
    # New trade: 1.0 lots = 2% heat. Total 6%. Should FAIL.
    pos2 = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=2.1, # 4.2% heat
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    # 4.2% + 1.0% (0.5 lots) = 5.2% > 5%.
    
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [pos2], 0.0, 0.0, equity
    )
    assert decision.approved is False
    assert decision.rejection_reason == "PORTFOLIO_HEAT_EXCEEDED"

def test_max_single_position_risk(risk_engine, base_signal):
    equity = 10_000.0 # Adjusted equity so min lot risk is manageable
    # Min lot 0.1. Risk = 0.1 * 100 * 20 = 200.
    # 200 / 10000 = 2%.
    # 2% < 5% (Total Heat) BUT > 1.5% (Single Limit).
    
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, 100.0, 0.1, [], 0.0, 0.0, equity
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "MAX_SINGLE_POSITION_RISK"

def test_max_positions_limit(risk_engine, base_signal):
    equity = 100_000.0
    pos = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=0.1,
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    open_positions = [pos] * 5 # 5 positions
    
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, 100.0, 0.01, open_positions, 0.0, 0.0, equity
    )
    
    assert decision.approved is False
    assert decision.rejection_reason == "MAX_POSITIONS_REACHED"

def test_resets(risk_engine):
    risk_engine._daily_halt_strategies.add("SCALP")
    risk_engine._hard_daily_halt_active = True
    risk_engine.reset_daily_state()
    assert len(risk_engine._daily_halt_strategies) == 0
    assert risk_engine._hard_daily_halt_active is False
    
    risk_engine._weekly_halt_active = True
    risk_engine.reset_weekly_state()
    assert risk_engine._weekly_halt_active is False

def test_history_limit(risk_engine):
    for i in range(35):
        risk_engine.record_trade_result("SWING", 1.0, True)
    
    assert len(risk_engine._trade_history["SWING"]) == 30

def test_new_strategy_history(risk_engine):
    # Record trade for unknown strategy
    risk_engine.record_trade_result("UNKNOWN_STRAT", 1.0, True)
    assert len(risk_engine._trade_history["UNKNOWN_STRAT"]) == 1

def test_ensure_utc_naive_conversion():
    # SignalIntent with naive datetime
    tp = TPPlan(tp1_distance=1.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    
    naive_dt = datetime(2026, 1, 1, 12, 0)
    si = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=naive_dt
    )
    assert si.timestamp.tzinfo == timezone.utc
    
    # RiskDecision with naive datetime
    rd = RiskDecision(
        approved=True, rejection_reason=None, position_size=1.0, risk_fraction_used=0.01,
        stop_price=100.0, take_profit_plan=tp, 
        circuit_breaker_state=CircuitBreakerState(daily_loss_pct=0.0, weekly_loss_pct=0.0, consecutive_losses=0, breaker_active=False),
        portfolio_heat_snapshot=PortfolioHeatSnapshot(current_heat_pct=0.0, heat_remaining=0.05, open_positions=0),
        strategy_viability=StrategyViability(rolling_er=1.0, viable=True),
        timestamp_utc=naive_dt
    )
    assert rd.timestamp_utc.tzinfo == timezone.utc

def test_ensure_utc_aware_conversion():
    # Test lines 51 and 92: Input is already aware
    aware_dt = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    # SignalIntent
    tp = TPPlan(tp1_distance=1.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    
    si = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=aware_dt
    )
    assert si.timestamp == aware_dt
    
    # RiskDecision
    rd = RiskDecision(
        approved=True, rejection_reason=None, position_size=1.0, risk_fraction_used=0.01,
        stop_price=100.0, take_profit_plan=tp, 
        circuit_breaker_state=CircuitBreakerState(daily_loss_pct=0.0, weekly_loss_pct=0.0, consecutive_losses=0, breaker_active=False),
        portfolio_heat_snapshot=PortfolioHeatSnapshot(current_heat_pct=0.0, heat_remaining=0.05, open_positions=0),
        strategy_viability=StrategyViability(rolling_er=1.0, viable=True),
        timestamp_utc=aware_dt
    )
    assert rd.timestamp_utc == aware_dt

def test_is_circuit_breaker_active_direct(risk_engine):
    # Test line 237
    assert risk_engine.is_circuit_breaker_active() is False
    risk_engine._hard_daily_halt_active = True
    assert risk_engine.is_circuit_breaker_active() is True

def test_circuit_breaker_daily_soft_orb(risk_engine, base_signal):
    # Test line 281: ORB limit -2.0%
    orb_signal = base_signal.model_copy(update={"strategy_name": "ORB"})
    
    # -1.9% -> Pass
    decision = risk_engine.evaluate(
        orb_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], -0.019, 0.0, 100_000.0
    )
    assert decision.approved is True
    
    # -2.1% -> Fail
    decision = risk_engine.evaluate(
        orb_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], -0.021, 0.0, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "DAILY_LOSS_SOFT_LIMIT_ORB"
    assert "ORB" in risk_engine._daily_halt_strategies

def test_short_signal_sizing(risk_engine, base_signal):
    # Test line 399: Short signal stop price
    short_signal = base_signal.model_copy(update={"direction": -1, "entry_trigger": 2000.0, "sl_distance": 20.0})
    
    decision = risk_engine.evaluate(
        short_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    
    # Stop price = 2000 + 20 = 2020
    assert decision.stop_price == 2020.0

def test_zero_equity_risk_calc(risk_engine):
    # Test line 456
    pos = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=1.0,
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    risk = risk_engine._calculate_position_risk(pos, 0.0, 100.0)
    assert risk == 0.0

def test_publish_decision_stub(risk_engine):
    # Test line 462
    risk_engine._publish_decision(MagicMock(), "TEST")

def test_consecutive_loss_reset_after_10(risk_engine, base_signal):
    # Prime with wins to keep ER healthy
    # Need ER > 0.
    # 5 losses coming up.
    # To keep ER > 0 after 5 losses, we need enough wins.
    # P*B - (1-P) > 0.
    # Let's add 20 wins.
    # Total 25 trades. 20 wins, 5 losses.
    # P = 20/25 = 0.8.
    # B = 1.0.
    # ER = 0.8*1 - 0.2 = 0.6 > 0. Safe.
    
    for _ in range(20):
        risk_engine.record_trade_result("SWING", 1.0, True)

    # Setup: 5 losses to trigger reduction
    for _ in range(5):
        risk_engine.record_trade_result("SWING", -1.0, False)
        
    # Check reduction active
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.position_size == 0.25 # 50% of 0.5
    
    # Process 9 trades (wins to keep ER high)
    for _ in range(9):
        risk_engine.record_trade_result("SWING", 1.0, True)
        
    # Still reduced? (Counter was 10. Decremented 9 times. Value is 1. > 0 implies active)
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.position_size == 0.25
    
    # Process 10th trade
    risk_engine.record_trade_result("SWING", 1.0, True)
    # Counter 1 -> 0.
    
    # Now evaluate. Counter 0. Normal size.
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.position_size == 0.5

def test_viability_boundary_zero(risk_engine, base_signal):
    # ER = 0.0 exactly. Viable = False. Modifier = 0.50.
    # P * B - (1 - P) = 0
    # Let P = 0.5. 0.5*B - 0.5 = 0 -> B = 1.0.
    # 1 Win (1R), 1 Loss (1R).
    risk_engine.record_trade_result("SWING", 1.0, True)
    risk_engine.record_trade_result("SWING", -1.0, False)
    
    er = risk_engine._calculate_rolling_er("SWING")
    assert er == 0.0
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.strategy_viability.viable is False
    assert decision.position_size == 0.25 # 0.5 * 0.5

def test_viability_boundary_negative_point_one(risk_engine, base_signal):
    # ER = -0.10 exactly.
    # Spec: ER < -0.10 -> Suspend.
    # So ER = -0.10 -> Modifier 0.50 (Not suspended, just degraded).
    # Need to construct trades for ER = -0.10.
    # P*B - (1-P) = -0.1
    # Let P = 0.5. 0.5*B - 0.5 = -0.1 -> 0.5*B = 0.4 -> B = 0.8.
    # Win 0.8R. Loss 1.0R.
    
    risk_engine.record_trade_result("SWING", 0.8, True)
    risk_engine.record_trade_result("SWING", -1.0, False)
    
    er = risk_engine._calculate_rolling_er("SWING")
    assert er == pytest.approx(-0.10)
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.approved is True # Not suspended
    assert decision.position_size == 0.25 # Degraded

    # Now make it slightly worse: -0.11
    # P=0.5. 0.5*B - 0.5 = -0.11 -> 0.5*B = 0.39 -> B = 0.78.
    risk_engine.reset_daily_state() # Just to be clean, though irrelevant
    # Clear history
    risk_engine._trade_history["SWING"] = []
    risk_engine.record_trade_result("SWING", 0.78, True)
    risk_engine.record_trade_result("SWING", -1.0, False)
    
    decision = risk_engine.evaluate(
        base_signal, 100_000.0, 1.0, 1.0, 1.0, 100.0, 0.01, [], 0.0, 0.0, 100_000.0
    )
    assert decision.approved is False
    assert decision.rejection_reason == "STRATEGY_SUSPENDED_LOW_ER"

def test_portfolio_heat_exact_boundary(risk_engine, base_signal):
    equity = 100_000.0
    contract_oz = 100.0
    
    # Target: 5.0% heat.
    # Create position with 4.0% heat.
    # New trade adds 1.0% heat.
    
    # 4% of 100k = 4000 risk.
    # Qty 2.0. SL 20. 2 * 100 * 20 = 4000.
    pos = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=2.0,
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    
    # New trade: Base signal 0.5 lots.
    # Risk: 0.5 * 100 * 20 = 1000 = 1%.
    # Total = 5%.
    
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [pos], 0.0, 0.0, equity
    )
    assert decision.approved is True
    
    # Slightly more heat: 4.001% + 1% > 5%.
    pos2 = Position(
        symbol="XAUUSD", side="LONG", entry_price=2000.0, quantity=2.001,
        stop_loss=1980.0, tp1=2020.0, tp2=2040.0, atr=10.0,
        opened_at=datetime.now(timezone.utc)
    )
    decision = risk_engine.evaluate(
        base_signal, equity, 1.0, 1.0, 1.0, contract_oz, 0.01, [pos2], 0.0, 0.0, equity
    )
    assert decision.approved is False
    assert decision.rejection_reason == "PORTFOLIO_HEAT_EXCEEDED"
