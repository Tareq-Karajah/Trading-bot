import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import MagicMock

from trading_bot.execution.paper_engine import PaperExecutionEngine, PaperOrderState, PaperPosition, TradeResult, AuditEntry
from trading_bot.execution.quality_gate import ExecutionDecision
from trading_bot.risk.engine import SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

# --- Fixtures ---

@pytest.fixture
def engine():
    return PaperExecutionEngine()

@pytest.fixture
def base_signal():
    return SignalIntent(
        strategy_name="SWING",
        direction=1, # Long
        score=5,
        entry_type="BREAKOUT",
        entry_trigger=100.0,
        sl_distance=2.0, # SL at 98.0
        tp_plan=TPPlan(
            tp1_distance=4.0, # TP1 at 104.0
            tp1_size_pct=0.5,
            tp2_distance=None,
            trail_atr_mult=3.0
        ),
        timeout_plan=TimeoutPlan(
            max_bars=0, # No limit for Swing
            mandatory_exit_utc=None
        ),
        regime_context=RegimeState.MID_VOL,
        macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ExecutionConstraints(
            max_spread_bp=10.0,
            max_slippage_bp=15.0,
            min_quote_fresh_ms=2000
        ),
        timestamp=datetime.now(timezone.utc)
        # Removed symbol="TEST_SYM" as it is not in SignalIntent definition
    )

@pytest.fixture
def execution_decision():
    return ExecutionDecision(
        allowed_state="ALLOWED",
        blocked_reason=None,
        spread_snapshot=1.0,
        spread_vs_median=1.0,
        quote_freshness_ms=100,
        expected_slippage_bp=1.0,
        routing_mode="LIMIT",
        order_intent_id="test_id",
        degraded_action=None,
        timestamp_utc=datetime.now(timezone.utc)
    )

# --- Tests ---

def test_constants(engine):
    assert engine.PAPER_ONLY is True
    assert engine.DRY_RUN is True

def test_submit_order_creates_pending_position(engine, base_signal, execution_decision):
    # Pass symbol explicitly
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    
    assert pos.order_id is not None
    assert pos.state == PaperOrderState.PENDING
    assert pos.symbol == "TEST_SYM"
    assert pos.current_size == 10.0
    assert pos.stop_price == 98.0
    assert pos.tp1_price == 104.0
    assert pos.initial_stop_price == 98.0
    
    # Audit Log
    logs = engine.get_audit_log()
    assert len(logs) == 1
    assert logs[0].event == "ORDER_SUBMITTED"

def test_submit_order_blocked_raises(engine, base_signal, execution_decision):
    blocked_decision = execution_decision.model_copy(update={
        "allowed_state": "BLOCKED",
        "blocked_reason": "TEST_BLOCK"
    })
    
    with pytest.raises(ValueError, match="EXECUTION_BLOCKED: TEST_BLOCK"):
        engine.submit_order(blocked_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")

def test_submit_order_degraded_halves_size(engine, base_signal, execution_decision):
    degraded_decision = execution_decision.model_copy(update={
        "allowed_state": "DEGRADED",
        "degraded_action": "REDUCE_SIZE_50PCT"
    })
    
    pos = engine.submit_order(degraded_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    assert pos.current_size == 5.0 # Halved

def test_on_bar_close_fills_entry(engine, base_signal, execution_decision):
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    
    now = datetime.now(timezone.utc)
    results = engine.on_bar_close("TEST_SYM", close=100.0, high=101.0, low=99.0, now_utc=now)
    
    assert len(results) == 0 # Just filled, no trade closed
    assert engine.get_open_positions()[0].state == PaperOrderState.FILLED
    
    logs = engine.get_audit_log()
    assert logs[-1].event == "ORDER_FILLED"

def test_sl_hit(engine, base_signal, execution_decision):
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Move price to SL (98.0)
    results = engine.on_bar_close("TEST_SYM", close=97.0, high=99.0, low=97.0, now_utc=datetime.now(timezone.utc))
    
    assert len(results) == 1
    res = results[0]
    assert res.exit_reason == "SL_HIT"
    assert res.exit_price == 98.0 # Filled at SL price exactly
    assert res.pnl_r == -1.0 # Lost 1R (Entry 100, SL 98, Exit 98)
    
    assert len(engine.get_open_positions()) == 0

def test_tp1_hit_partial_and_trail_activation(engine, base_signal, execution_decision):
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Move price to TP1 (104.0)
    # High hits 105.0
    results = engine.on_bar_close("TEST_SYM", close=103.0, high=105.0, low=102.0, now_utc=datetime.now(timezone.utc))
    
    assert len(results) == 1 # Partial result
    res = results[0]
    assert res.exit_reason == "TP1_HIT"
    assert res.exit_price == 104.0
    assert res.pnl_r == 2.0 # (104-100)/(100-98) = 4/2 = 2R
    
    # Check remaining position
    assert len(engine.get_open_positions()) == 1
    rem_pos = engine.get_open_positions()[0]
    assert rem_pos.current_size == 5.0 # Halved
    assert rem_pos.state == PaperOrderState.PARTIALLY_FILLED
    assert rem_pos.trail_active is True
    # SL moved to BE + buffer (100 + 0.05% = 100.05)
    assert rem_pos.stop_price == 100.05

def test_trailing_stop_hit(engine, base_signal, execution_decision):
    # Setup: Fill and Hit TP1 to activate trail
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    engine.on_bar_close("TEST_SYM", 103.0, 105.0, 102.0, datetime.now(timezone.utc)) # TP1 Hit
    
    # Current State: Trail Active. 
    # Highest Close so far?
    # Fill bar close: 100.0. TP1 bar close: 103.0.
    # Highest Close = 103.0.
    # Trail Mult (Swing) = 3.0. ATR = 1.0.
    # Trail Price = 103.0 - 3*1 = 100.0.
    # Current Stop (BE) = 100.05.
    
    # Next bar: Price goes up to 110.0.
    engine.on_bar_close("TEST_SYM", 110.0, 111.0, 109.0, datetime.now(timezone.utc))
    
    rem_pos = engine.get_open_positions()[0]
    # Highest Close = 110.0.
    # Trail Price = 110.0 - 3.0 = 107.0.
    assert rem_pos.trail_price == 107.0
    
    # Next bar: Price drops to 106.0 (crosses 107.0)
    results = engine.on_bar_close("TEST_SYM", 106.0, 108.0, 105.0, datetime.now(timezone.utc))
    
    assert len(results) == 1
    res = results[0]
    assert res.exit_reason == "TRAILING_STOPPED"
    assert res.exit_price == 107.0
    assert len(engine.get_open_positions()) == 0

def test_scalp_timeout(engine, base_signal, execution_decision):
    # Modify signal for SCALP
    scalp_signal = base_signal.model_copy(update={
        "strategy_name": "SCALP",
        "timeout_plan": TimeoutPlan(max_bars=2, mandatory_exit_utc=None)
    })
    
    pos = engine.submit_order(execution_decision, scalp_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Bar 1 (bars_held becomes 1)
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    # Bar 2 (bars_held becomes 2 -> Expiry)
    results = engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    assert len(results) == 1
    assert results[0].exit_reason == "EXPIRED"
    assert len(engine.get_open_positions()) == 0

def test_orb_friday_exit(engine, base_signal, execution_decision):
    # ORB Signal
    orb_signal = base_signal.model_copy(update={
        "strategy_name": "ORB",
        "timeout_plan": TimeoutPlan(max_bars=0, mandatory_exit_utc="20:45")
    })
    
    pos = engine.submit_order(execution_decision, orb_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Friday 19:31
    friday = datetime(2023, 10, 27, 19, 31, tzinfo=timezone.utc) # Oct 27 2023 is Friday
    assert friday.weekday() == 4
    
    results = engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, now_utc=friday)
    
    assert len(results) == 1
    assert results[0].exit_reason == "EXPIRED"

def test_force_close(engine, base_signal, execution_decision):
    pos = engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Force close
    res = engine.force_close(pos.order_id, reason="TEST_FORCE")
    
    assert res.exit_reason == "MANUALLY_CLOSED"
    assert res.exit_price == 100.0 # Entry price was highest close so far
    assert len(engine.get_open_positions()) == 0

def test_short_position_lifecycle(engine, base_signal, execution_decision):
    # 1. Submit Short
    short_signal = base_signal.model_copy(update={
        "direction": -1, # Short
        "entry_trigger": 100.0,
        "sl_distance": 2.0, # SL 102.0
        "tp_plan": TPPlan(
            tp1_distance=4.0, # TP1 96.0
            tp1_size_pct=0.5,
            tp2_distance=None,
            trail_atr_mult=3.0
        )
    })
    
    pos = engine.submit_order(execution_decision, short_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    assert pos.direction == -1
    assert pos.stop_price == 102.0
    assert pos.tp1_price == 96.0
    
    # 2. Fill
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    assert engine.get_open_positions()[0].state == PaperOrderState.FILLED

    # 3. TP1 Hit (Low touches 96.0)
    results = engine.on_bar_close("TEST_SYM", 97.0, 98.0, 95.0, datetime.now(timezone.utc))
    assert len(results) == 1
    res = results[0]
    assert res.exit_reason == "TP1_HIT"
    assert res.exit_price == 96.0
    assert res.pnl_r == 2.0 # (100-96)/(102-100) = 4/2 = 2R
    
    pos = engine.get_open_positions()[0]
    assert pos.current_size == 5.0
    # SL moved to BE - buffer (100 - 0.05% = 99.95)
    assert pos.stop_price == 99.95
    assert pos.trail_active is True
    
    # 4. Trail Update (Lowest Close logic)
    # Price drops to 90.0
    engine.on_bar_close("TEST_SYM", 90.0, 92.0, 89.0, datetime.now(timezone.utc))
    pos = engine.get_open_positions()[0]
    # Highest Close logic for Short is actually "Lowest Close"?
    # Spec: "SWING: Highest_Close - 3.0 x ATR"
    # Wait. For Short, trailing stop should be ABOVE price.
    # Usually "Lowest Close + 3.0 x ATR".
    # Let's check implementation.
    # "else: # Short ... if close < pos.highest_close: pos.highest_close = close"
    # "new_trail = pos.highest_close + (pos.trail_multiplier * pos.atr_at_entry)"
    # So `highest_close` variable is used to store the "best price" (lowest for short).
    assert pos.highest_close == 90.0
    assert pos.trail_price == 93.0 # 90 + 3*1
    
    # 5. Trail Hit
    # Price rises to 94.0 (crosses 93.0)
    results = engine.on_bar_close("TEST_SYM", 94.0, 95.0, 93.5, datetime.now(timezone.utc))
    assert len(results) == 1
    assert results[0].exit_reason == "TRAILING_STOPPED"
    assert results[0].exit_price == 93.0

def test_short_sl_hit(engine, base_signal, execution_decision):
    short_signal = base_signal.model_copy(update={
        "direction": -1,
        "entry_trigger": 100.0,
        "sl_distance": 2.0 # SL 102.0
    })
    engine.submit_order(execution_decision, short_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc)) # Fill
    
    # Hit SL (High >= 102.0)
    results = engine.on_bar_close("TEST_SYM", 101.0, 103.0, 100.0, datetime.now(timezone.utc))
    assert len(results) == 1
    assert results[0].exit_reason == "SL_HIT"
    assert results[0].exit_price == 102.0

def test_force_close_short(engine, base_signal, execution_decision):
    short_signal = base_signal.model_copy(update={"direction": -1, "entry_trigger": 100.0})
    pos = engine.submit_order(execution_decision, short_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    # Price moves in favor (90.0)
    engine.on_bar_close("TEST_SYM", 90.0, 92.0, 89.0, datetime.now(timezone.utc))
    
    res = engine.force_close(pos.order_id, "TEST")
    # Exit at "highest_close" (best price) which is 90.0
    assert res.exit_price == 90.0
    # PnL: Entry 100, Exit 90. Profit 10. Risk 2 (default). 5R.
    assert res.pnl_r > 0

def test_safety_rails(engine):
    # Patch constants
    # Since they are class attributes, we can patch them on the instance or class
    # But type hint says they are final-ish.
    # Let's try patching the class
    original_paper = PaperExecutionEngine.PAPER_ONLY
    
    try:
        PaperExecutionEngine.PAPER_ONLY = False
        with pytest.raises(RuntimeError, match="SAFETY_VIOLATION"):
            # Need a fresh engine or just call submit
            # The check is in submit_order
            eng = PaperExecutionEngine()
            # Mock args
            eng.submit_order(MagicMock(), MagicMock(), 1.0, 1.0, "S")
    finally:
        PaperExecutionEngine.PAPER_ONLY = original_paper

def test_multiple_symbols(engine, base_signal, execution_decision):
    engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="SYM1")
    engine.submit_order(execution_decision, base_signal, atr=1.0, quantity=10.0, symbol="SYM2")
    
    # Process SYM1
    engine.on_bar_close("SYM1", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    positions = engine.get_open_positions()
    sym1_pos = next(p for p in positions if p.symbol == "SYM1")
    sym2_pos = next(p for p in positions if p.symbol == "SYM2")
    
    assert sym1_pos.state == PaperOrderState.FILLED
    assert sym2_pos.state == PaperOrderState.PENDING

def test_orb_normal_day_exit(engine, base_signal, execution_decision):
    # ORB on a Tuesday
    orb_signal = base_signal.model_copy(update={
        "strategy_name": "ORB",
        "timeout_plan": TimeoutPlan(max_bars=0, mandatory_exit_utc="20:45")
    })
    engine.submit_order(execution_decision, orb_signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    # Tuesday 20:46
    tuesday = datetime(2023, 10, 24, 20, 46, tzinfo=timezone.utc)
    assert tuesday.weekday() == 1
    
    results = engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, now_utc=tuesday)
    assert len(results) == 1
    assert results[0].exit_reason == "EXPIRED"

def test_invalid_exit_time_format(engine, base_signal, execution_decision):
    signal = base_signal.model_copy(update={
        "timeout_plan": TimeoutPlan(max_bars=0, mandatory_exit_utc="INVALID")
    })
    engine.submit_order(execution_decision, signal, atr=1.0, quantity=10.0, symbol="TEST_SYM")
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    
    # Should not crash, just ignore
    engine.on_bar_close("TEST_SYM", 100.0, 101.0, 99.0, datetime.now(timezone.utc))
    assert engine.get_open_positions()[0].state == PaperOrderState.FILLED

def test_force_close_invalid_id(engine):
    with pytest.raises(ValueError, match="Order ID .* not found"):
        engine.force_close("INVALID", "TEST")

def test_naive_datetime_handling(engine, base_signal, execution_decision):
    # Test ensure_utc validators
    
    # Submit with naive timestamp in decision (if possible)
    # ExecutionDecision validator might catch it first, but let's try
    # We can test the Validator directly on AuditEntry or TradeResult
    
    entry = AuditEntry(
        order_id="1", event="E", price=1.0, 
        timestamp_utc=datetime(2023, 1, 1), # Naive
        details={}
    )
    assert entry.timestamp_utc.tzinfo == timezone.utc
    
    res = TradeResult(
        order_id="1", strategy_name="S", direction=1, entry_price=1.0, exit_price=1.0,
        pnl_r=0.0, exit_reason="R", bars_held=1,
        timestamp_utc=datetime(2023, 1, 1) # Naive
    )
    assert res.timestamp_utc.tzinfo == timezone.utc
    
    pos = PaperPosition(
        order_id="1", strategy_name="S", symbol="S", direction=1, entry_price=1.0,
        current_size=1.0, initial_size=1.0, stop_price=1.0, initial_stop_price=1.0,
        tp1_price=None, trail_active=False, trail_price=None, highest_close=1.0,
        bars_held=0, state=PaperOrderState.PENDING, atr_at_entry=1.0, pnl_r=0.0,
        entry_time_utc=datetime(2023, 1, 1) # Naive
    )
    assert pos.entry_time_utc.tzinfo == timezone.utc

