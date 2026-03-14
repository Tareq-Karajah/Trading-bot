import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from trading_bot.strategies.swing.engine import SwingEngine, SwingDecision, SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def swing_engine():
    return SwingEngine()

@pytest.fixture
def permissions():
    return DispatcherPermissions(
        allow_swing_rebalance=True,
        allow_orb=False,
        allow_news=False,
        allow_scalp=False,
        direction_constraint=0,
        macro_bias=MacroState.MACRO_NEUTRAL,
        regime=RegimeState.LOW_VOL,
        risk_scalar=1.0,
        blackout_active=False,
        blackout_reason=None
    )

def generate_closes(length, trend=0.0):
    """
    Generate synthetic daily closes.
    trend: daily return drift (e.g. 0.001 for bull, -0.001 for bear)
    """
    closes = [100.0]
    for i in range(1, length):
        change = trend + np.random.normal(0, 0.01) # Random walk with drift
        closes.append(closes[-1] * (1 + change))
    return closes

def generate_perfect_trend(length, direction=1):
    """
    Generate a perfect trend where price strictly increases or decreases
    to ensure all momentum horizons align.
    Uses multiplicative factor to ensure prices remain positive.
    """
    closes = []
    current_price = 100.0
    
    # 0.5% daily change
    factor = 1.005 if direction == 1 else 0.995
    
    for _ in range(length):
        closes.append(current_price)
        current_price *= factor
        
    return closes

# --- Momentum Computation Tests ---

def test_momentum_all_positive(swing_engine, permissions):
    # Strictly increasing prices -> all horizons positive
    closes = generate_perfect_trend(300, direction=1)
    
    decision = swing_engine.evaluate(
        closes, 1.0, 1.0, permissions, datetime.now(timezone.utc)
    )
    
    assert decision.mom_1m > 0
    assert decision.mom_3m > 0
    assert decision.mom_6m > 0
    assert decision.mom_12m > 0
    assert decision.confidence == pytest.approx(1.0) # 0.1+0.2+0.3+0.4
    assert decision.swing_dir == 1

def test_momentum_all_negative(swing_engine, permissions):
    # Strictly decreasing prices -> all horizons negative
    closes = generate_perfect_trend(300, direction=-1)
    
    decision = swing_engine.evaluate(
        closes, 1.0, 1.0, permissions, datetime.now(timezone.utc)
    )
    
    assert decision.confidence == pytest.approx(-1.0)
    assert decision.swing_dir == -1

def test_mixed_momentum_neutral(swing_engine, permissions):
    # Construct scenario: 
    # Long term (12m, 6m) positive (0.4 + 0.3 = 0.7)
    # Short term (3m, 1m) negative (-0.2 - 0.1 = -0.3)
    # Net = 0.4 -> Bullish? 
    # Wait, 0.7 - 0.3 = 0.4. That's > 0.30 -> Bull.
    # Need to tune to get < 0.30.
    # 12m(+), 6m(-), 3m(-), 1m(-) -> 0.4 - 0.3 - 0.2 - 0.1 = -0.2.
    # -0.2 is within [-0.3, 0.3] -> Neutral.
    
    # 12m positive: Price[t] > Price[t-252]
    # Others negative: Price[t] < Price[t-126], Price[t] < Price[t-63], etc.
    
    # Simple synthetic construction:
    closes = [100.0] * 300
    # Set key pivot points
    # T = 299 (last index)
    # T-252 = 47. 
    # T-126 = 173.
    # T-63 = 236.
    # T-21 = 278.
    
    closes[299] = 100.0
    closes[47] = 90.0  # 12m positive (100 > 90)
    closes[173] = 110.0 # 6m negative (100 < 110)
    closes[236] = 110.0 # 3m negative
    closes[278] = 110.0 # 1m negative
    
    # Ensure volatility is non-zero so div doesn't fail
    # Add noise to others
    for i in range(300):
        if i not in [299, 47, 173, 236, 278]:
            closes[i] = 100.0 + np.random.normal(0, 0.1)
            
    decision = swing_engine.evaluate(
        closes, 1.0, 1.0, permissions, datetime.now(timezone.utc)
    )
    
    # Check signs
    assert decision.mom_12m > 0
    assert decision.mom_6m < 0
    assert decision.mom_3m < 0
    assert decision.mom_1m < 0
    
    # Score: 0.4*1 + 0.3*(-1) + 0.2*(-1) + 0.1*(-1) = 0.4 - 0.6 = -0.2
    assert decision.confidence == pytest.approx(-0.2)
    assert decision.swing_dir == 0

def test_confidence_boundary_exact(swing_engine, permissions):
    # Construct: 12m +, 6m 0, 3m -, 1m +
    # Weights: 0.4, 0.3, 0.2, 0.1
    # 0.4(+) + 0.3(0) + 0.2(-) + 0.1(+) = 0.4 + 0 - 0.2 + 0.1 = 0.3
    
    closes = [100.0] * 300
    closes[299] = 100.0
    closes[47] = 90.0   # 12m + (100 > 90)
    closes[173] = 100.0 # 6m 0 (100 == 100)
    closes[236] = 110.0 # 3m - (100 < 110)
    closes[278] = 90.0  # 1m + (100 > 90)
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    
    # With rounding to 10 decimal places, 0.3000000000...4 is rounded to 0.3
    # Wait, float(0.30000000000000004) rounded to 10 decimals is 0.3
    # So confidence should be exactly 0.3
    # Logic: if confidence > 0.30 -> +1.
    # 0.3 is NOT > 0.30. So swing_dir should be 0.
    # Let's verify assertion:
    assert decision.confidence == pytest.approx(0.30, abs=1e-9)
    assert decision.swing_dir == 0 # 0.3 is not > 0.3

    # Boundary -0.30 -> swing_dir = 0
    # 0.4(-) + 0.3(0) + 0.2(+) + 0.1(-) = -0.4 + 0 + 0.2 - 0.1 = -0.3
    closes[47] = 110.0  # 12m -
    closes[173] = 100.0 # 6m 0
    closes[236] = 90.0  # 3m +
    closes[278] = 110.0 # 1m -
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    assert decision.confidence == pytest.approx(-0.30, abs=1e-9)
    assert decision.swing_dir == 0 # < -0.30 required for -1

# --- Brake Tests ---

def test_brake_activation(swing_engine, permissions):
    # Need all 4 conditions:
    # 1. swing_dir != 0 (Let's go Bull, +1)
    # 2. sign(mom_1m) != swing_dir (So mom_1m must be negative)
    # 3. mom_1m < -0.5 * StdDev (Strong negative pull)
    # 4. ATR_fast / ATR_slow > 1.15
    
    # Construct closes:
    # Bullish long term (12m, 6m, 3m positive) -> Confidence > 0.3
    # Sharp drop recently (1m negative strong)
    
    closes = [100.0] * 300
    # Long term bull
    closes[47] = 80.0
    closes[173] = 85.0
    closes[236] = 90.0
    
    # 1m negative strong. Current = 100. Prev_21 = 115.
    closes[299] = 100.0
    closes[278] = 115.0 
    
    # Ensure Cond 3 holds: mom_1m < -0.5 * StdDev
    # With limited data (300), code might skip rolling calc if < 274?
    # No, 300 > 274. So it will compute.
    # We need mom_1m to be an outlier.
    # To make StdDev small, history of mom_1m should be flat (near 0).
    # Then current mom_1m is large negative.
    # We set history to constant price, so mom ~ 0.
    # Then the dip at the end creates outlier.
    
    # ATR Ratio > 1.15
    atr_fast = 1.2
    atr_slow = 1.0
    
    decision = swing_engine.evaluate(closes, atr_fast, atr_slow, permissions, datetime.now(timezone.utc))
    
    assert decision.swing_dir == 1 # 0.4+0.3+0.2-0.1 = 0.8
    assert decision.mom_1m < 0
    assert decision.brake_active is True
    assert decision.brake_modifier == 0.40

def test_brake_inactive_single_fail(swing_engine, permissions):
    # Fail Cond 4 (ATR Ratio)
    closes = generate_perfect_trend(300, 1) # Bull
    # Force 1m negative to trigger potential brake logic
    # Make sure we don't go negative on price
    closes[-1] = closes[-22] * 0.9 # Sharp drop
    
    atr_fast = 1.0
    atr_slow = 1.0 # Ratio 1.0 < 1.15
    
    decision = swing_engine.evaluate(closes, atr_fast, atr_slow, permissions, datetime.now(timezone.utc))
    assert decision.brake_active is False
    assert decision.brake_modifier == 1.0

# --- Scoring & Intent Tests ---

def test_score_perfect_bull(swing_engine, permissions):
    # All aligned + No Brake
    closes = generate_perfect_trend(300, 1)
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    
    assert decision.score == 5 # 4 horizons + 1 brake
    assert decision.signal_intent is not None
    assert decision.signal_intent.direction == 1
    assert decision.signal_intent.score == 5

def test_score_brake_active(swing_engine, permissions):
    # Aligned but Brake Active?
    # Brake requires sign(mom_1m) != swing_dir.
    # If all horizons aligned, mom_1m matches swing_dir.
    # So Brake Cond 2 fails.
    # Thus, if Score = 5, Brake CANNOT be active.
    # If Brake is active, mom_1m mismatch -> Score loses 1 point for mom_1m mismatch
    # AND Score loses 1 point for Brake active?
    # Scoring logic:
    # +1 if mom_12m == swing
    # +1 if mom_6m == swing
    # +1 if mom_3m == swing
    # +1 if mom_1m == swing
    # +1 if NOT brake_active
    
    # If Brake Active:
    # mom_1m != swing (Cond 2). So mom_1m score = 0.
    # Brake active = True. So brake score = 0.
    # Max score = 3 (12m, 6m, 3m aligned).
    # So if Brake is active, score is at most 3.
    # SignalIntent emitted only if score >= 4.
    # Therefore, Brake Active -> No Signal Intent?
    # Let's verify Spec.
    # "Only emit SignalIntent if score >= 4".
    # This implies Brake effectively kills new entries?
    # Yes, makes sense. Brake = "Turning Point" -> Danger -> Don't enter.
    
    # Let's verify this logic with test.
    closes = [100.0] * 300
    # Bull setup 12, 6, 3
    closes[47] = 80.0
    closes[173] = 85.0
    closes[236] = 90.0
    # 1m crash
    closes[299] = 100.0
    closes[278] = 115.0
    
    decision = swing_engine.evaluate(closes, 1.2, 1.0, permissions, datetime.now(timezone.utc))
    
    assert decision.brake_active is True
    assert decision.score <= 3 # 12m, 6m, 3m match (+3). 1m fail. Brake fail.
    assert decision.signal_intent is None

def test_signal_suppression_score(swing_engine, permissions):
    # Create mixed signal score 3
    # 12m +, 6m -, 3m +, 1m +
    # Conf = 0.4 - 0.3 + 0.2 + 0.1 = 0.4 -> Bull (+1)
    # Score:
    # 12m (+) == +1 -> Yes
    # 6m (-) == +1 -> No
    # 3m (+) == +1 -> Yes
    # 1m (+) == +1 -> Yes
    # Brake inactive -> Yes
    # Total = 4. Signal should emit.
    
    # Wait, test requirement says "Score < 4 suppresses". 4 emits.
    # Let's try Score 3.
    # 12m +, 6m +, 3m -, 1m -
    # Conf = 0.4 + 0.3 - 0.2 - 0.1 = 0.4 -> Bull (+1)
    # Score:
    # 12m match
    # 6m match
    # 3m fail
    # 1m fail
    # Brake inactive (likely, 1m is - vs swing +, but need ATR ratio etc. Assume inactive)
    # Score = 3 (12m, 6m, Brake).
    
    closes = [100.0] * 300
    closes[299] = 100.0
    closes[47] = 90.0   # 12m +
    closes[173] = 90.0  # 6m +
    closes[236] = 110.0 # 3m -
    closes[278] = 110.0 # 1m -
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    assert decision.swing_dir == 1
    assert decision.score == 3
    assert decision.signal_intent is None

def test_direction_constraint_suppression(swing_engine, permissions):
    # Perfect Bull signal
    closes = generate_perfect_trend(300, 1)
    
    # Constraint -1 (Bear only)
    # Use object replacement since it's a Pydantic model (frozen)
    # But permissions fixture is function scoped, we can just create new one
    perms_bear = DispatcherPermissions(
        allow_swing_rebalance=True, allow_orb=False, allow_news=False, allow_scalp=False,
        direction_constraint=-1, # Block Longs
        macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL, risk_scalar=1.0,
        blackout_active=False, blackout_reason=None
    )
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, perms_bear, datetime.now(timezone.utc))
    assert decision.swing_dir == 1
    assert decision.signal_intent is None # Blocked by constraint

def test_allow_rebalance_suppression(swing_engine, permissions):
    closes = generate_perfect_trend(300, 1)
    perms_off = DispatcherPermissions(
        allow_swing_rebalance=False, # Block
        allow_orb=False, allow_news=False, allow_scalp=False, direction_constraint=0,
        macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL, risk_scalar=1.0,
        blackout_active=False, blackout_reason=None
    )
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, perms_off, datetime.now(timezone.utc))
    assert decision.signal_intent is None

# --- Schema & Guards ---

def test_min_data_guard(swing_engine, permissions):
    closes = [100.0] * 252 # One short
    with pytest.raises(ValueError, match="Insufficient daily closes"):
        swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))

def test_signal_intent_schema(swing_engine, permissions):
    closes = generate_perfect_trend(300, 1)
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    
    intent = decision.signal_intent
    assert intent.strategy_name == "SWING"
    assert intent.direction == 1
    assert intent.tp_plan.tp1_size_pct == 0.50
    assert intent.execution_constraints.max_spread_bp == 10.0
    assert intent.timestamp.tzinfo == timezone.utc

def test_redis_publish_stub(swing_engine, permissions):
    closes = generate_perfect_trend(300, 1)
    
    with patch.object(swing_engine, '_publish_signal') as mock_pub:
        swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
        mock_pub.assert_called_once()

def test_naive_datetime_handling(swing_engine, permissions):
    closes = generate_perfect_trend(300, 1)
    naive_now = datetime(2023, 1, 1, 12, 0, 0)
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, naive_now)
    assert decision.timestamp_utc.tzinfo == timezone.utc

# --- Coverage Fixes ---

def test_signal_intent_naive_datetime_validator():
    # Covers Lines 50-52: SignalIntent validator branch
    intent = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=TPPlan(tp1_distance=None, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=3.0),
        timeout_plan=TimeoutPlan(max_bars=None, mandatory_exit_utc="00:05"),
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ExecutionConstraints(max_spread_bp=10, max_slippage_bp=15, min_quote_fresh_ms=2000),
        timestamp=datetime(2023, 1, 1, 12, 0, 0) # Naive
    )
    assert intent.timestamp.tzinfo == timezone.utc

def test_swing_decision_naive_datetime_validator():
    # Covers Line 76: SwingDecision validator branch
    decision = SwingDecision(
        swing_dir=1, confidence=1.0, brake_active=False, brake_modifier=1.0, score=5, signal_intent=None,
        mom_1m=1.0, mom_3m=1.0, mom_6m=1.0, mom_12m=1.0, atr_daily_20=1.0,
        timestamp_utc=datetime(2023, 1, 1, 12, 0, 0) # Naive
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc

def test_calculate_pseudo_atr_guard(swing_engine):
    # Covers Line 321: Guard for insufficient closes in private method
    closes = np.array([100.0] * 5, dtype=np.float64) # Period is 20, so 5 < 21
    atr = swing_engine._calculate_pseudo_atr(closes, 20)
    assert atr == 0.0

def test_brake_logic_loop_branch(swing_engine, permissions):
    # Covers Line 266: if len(subset_returns) == 21
    # We need to trigger the loop in brake logic (len > 274)
    # And ensure the slice logic works.
    closes = generate_perfect_trend(300, 1)
    
    # We need to ensure brake conditions 1 & 2 are met to enter the loop block
    # C1: swing_dir != 0 (True for perfect trend)
    # C2: sign(mom_1m) != swing_dir (False for perfect trend)
    # So we need to fake C2.
    # mom_1m needs to be negative while swing is positive.
    # Swing uses all 4 horizons. 
    # If 1m is negative but 12m, 6m, 3m are positive -> Swing likely positive.
    # 0.4+0.3+0.2-0.1 = 0.8 -> Swing +1.
    # 1m is -1. So C2 passed.
    
    # Modify perfect trend: make recent prices drop to flip 1m mom
    # mom_1m uses T and T-21.
    closes[-1] = closes[-22] * 0.9 # Sharp drop
    
    # Now evaluate. This should enter the loop.
    # The coverage gap at line 266 check `if len(subset_returns) == 21`.
    # Since we provide 300 points, and the loop iterates 252 times,
    # it extracts slices of size 21. It should pass.
    # The test confirms no crash and logic runs.
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, permissions, datetime.now(timezone.utc))
    # We don't check brake active/inactive here, just that we exercised the code path.
    assert decision.mom_1m < 0
    assert decision.swing_dir == 1

def test_signal_intent_explicit_utc_validator():
    # Covers Line 52: SignalIntent validator return v (already UTC)
    now_utc = datetime.now(timezone.utc)
    intent = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=TPPlan(tp1_distance=None, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=3.0),
        timeout_plan=TimeoutPlan(max_bars=None, mandatory_exit_utc="00:05"),
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ExecutionConstraints(max_spread_bp=10, max_slippage_bp=15, min_quote_fresh_ms=2000),
        timestamp=now_utc # Explicit UTC
    )
    assert intent.timestamp == now_utc
    assert intent.timestamp.tzinfo == timezone.utc

def test_direction_constraint_explicit_match(swing_engine):
    # Covers Line 266: dir_allowed = True when constraint == swing_dir
    closes = generate_perfect_trend(300, 1) # swing_dir = 1
    
    # Permissions with constraint = 1 (Long only)
    perms_long = DispatcherPermissions(
        allow_swing_rebalance=True, allow_orb=False, allow_news=False, allow_scalp=False,
        direction_constraint=1, # Match swing_dir
        macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL, risk_scalar=1.0,
        blackout_active=False, blackout_reason=None
    )
    
    decision = swing_engine.evaluate(closes, 1.0, 1.0, perms_long, datetime.now(timezone.utc))
    
    # Should have signal
    assert decision.swing_dir == 1
    assert decision.signal_intent is not None
    assert decision.signal_intent.direction == 1
