import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from trading_bot.strategies.scalping.engine import (
    ScalpingEngine, ScalpingDecision, SignalIntent,
    TPPlan, TimeoutPlan, ExecutionConstraints
)
from trading_bot.core.models import OHLCV
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def scalping_engine():
    return ScalpingEngine()

@pytest.fixture
def permissions():
    return DispatcherPermissions(
        allow_swing_rebalance=True, allow_orb=False, allow_news=False, allow_scalp=True,
        direction_constraint=0, macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL,
        risk_scalar=1.0, blackout_active=False, blackout_reason=None
    )

def create_candle(price: float, volume: float = 1000.0, timestamp: datetime = None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    return OHLCV(
        symbol="XAUUSD", timeframe="5m", open=price, high=price+1, low=price-1, close=price,
        volume=volume, timestamp=timestamp
    )

def generate_candles(count: int, price: float = 100.0, volume: float = 1000.0) -> list[OHLCV]:
    now = datetime.now(timezone.utc)
    candles = []
    for i in range(count):
        t = now - timedelta(minutes=(count - i) * 5)
        candles.append(create_candle(price, volume, t))
    return candles

# --- Activation Gates Tests ---

def test_gate_regime_fail(scalping_engine, permissions):
    perms = permissions.model_copy(update={"regime": RegimeState.HIGH_VOL})
    candles = generate_candles(200)
    decision = scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 1.0, perms, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None

def test_gate_macro_fail(scalping_engine, permissions):
    perms = permissions.model_copy(update={"macro_bias": MacroState.MACRO_EVENT_RISK})
    candles = generate_candles(200)
    decision = scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 1.0, perms, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None

def test_gate_blackout_fail(scalping_engine, permissions):
    perms = permissions.model_copy(update={"blackout_active": True})
    candles = generate_candles(200)
    decision = scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 1.0, perms, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None

def test_gate_spread_fail(scalping_engine, permissions):
    candles = generate_candles(200)
    # Spread 2.1 > 2.0
    decision = scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 2.1, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None

def test_gate_spread_boundary(scalping_engine, permissions):
    # Construct valid LONG signal:
    # BB Lower ~ 90. Close 80.
    # RSI < 30 (Dumping).
    # Close 80 > EMA 200 (70).
    # Vol > 1.1 * VMA.
    # Trend day filter OK.

    # Setup candles
    c_base = generate_candles(200, 100.0, 1000.0)
    
    # Drop to 80 over 15 bars.
    # Keep volume at 1000 for history to keep VMA low.
    for i in range(15):
        idx = -(15 - i)
        c_base[idx] = create_candle(100.0 - i, 1000.0, c_base[idx].timestamp)
    
    # Last candle: Close 80. Volume 1200.
    # VMA (all 1000) = 1000.
    # 1.1 * 1000 = 1100.
    # 1200 > 1100. Entry Vol OK.
    c_base[-1] = create_candle(80.0, 1200.0, c_base[-1].timestamp)

    # Spread 2.0
    decision = scalping_engine.evaluate(c_base, 70.0, 100.0, 90.0, 10.0, 2.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is not None

def test_gate_session_fail_early(scalping_engine, permissions):
    candles = generate_candles(200)
    # 07:59
    decision = scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 1.0, permissions, datetime(2026, 3, 20, 7, 59, tzinfo=timezone.utc))
    assert decision.signal_intent is None

def test_gate_session_pass_start(scalping_engine, permissions):
    # 08:00
    c_base = generate_candles(200, 100.0, 1000.0)
    # Force signal
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1000.0) # Low history vol
    # Last candle 80.
    c_base[-1] = create_candle(80.0, 1200.0)
    
    decision = scalping_engine.evaluate(c_base, 70.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is not None

def test_gate_session_pass_end(scalping_engine, permissions):
    # 17:00
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1000.0)
    c_base[-1] = create_candle(80.0, 1200.0)
    
    decision = scalping_engine.evaluate(c_base, 70.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 17, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is not None

def test_gate_session_fail_late(scalping_engine, permissions):
    # 17:01
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1200.0)
    c_base[-1] = create_candle(85.0, 1200.0)
    
    decision = scalping_engine.evaluate(c_base, 80.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 17, 1, tzinfo=timezone.utc))
    assert decision.signal_intent is None

# --- Signal Logic Tests ---

def test_long_signal_conditions(scalping_engine, permissions):
    # Prepare data for Long
    c_base = generate_candles(200, 100.0, 1000.0)
    # Drop prices to trigger RSI < 30 and Close < BB Lower
    for i in range(20):
        c_base[-(20-i)] = create_candle(100.0 - i, 1200.0) # 100..81
    
    # Close 81. SMA20 ~ 90. BB Lower ~ 80?
    # Need Close < BB Lower.
    # Increase Volatility?
    # Let's make prior candles stable 100, then drop SHARPLY.
    # Last 20: 19 at 100, 1 at 85.
    # SMA = (1900 + 85)/20 = 99.25.
    # Std: ~ sqrt(mean((x-99.25)^2)). (19*0.75^2 + 14.25^2)/20 ~ (10 + 200)/20 ~ 10. Sqrt ~ 3.
    # BB Lower = 99.25 - 2*3 = 93.
    # Close 85 < 93. OK.
    
    c_base = generate_candles(200, 100.0, 1000.0)
    c_base[-1] = create_candle(85.0, 1200.0) # Vol > 1100
    
    # Need RSI < 30.
    # Single drop 100->85 might not drive RSI < 30 if prev 14 were flat?
    # Flat -> RSI 50 or 100? If no loss/gain, RSI undefined or 50.
    # We need sustained drop.
    # Let's drop 100, 95, 90, 85 over last 4 bars.
    for i in range(4):
        c_base[-(4-i)] = create_candle(100.0 - (i+1)*5, 1200.0) # 95, 90, 85, 80.
    # Close 80.
    
    # EMA 200 needs to be < Close for Long.
    # Close 80. So EMA 200 must be < 80. Say 75.
    
    # Trend Day Filter: |80 - 75| = 5. ATR = 10. 5 < 25. OK.
    
    decision = scalping_engine.evaluate(c_base, 75.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is not None
    assert decision.signal_intent.direction == 1
    assert decision.bb_signal is True

def test_short_signal_conditions(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    # Rise 100, 105, 110, 115, 120.
    for i in range(4):
        c_base[-(4-i)] = create_candle(100.0 + (i+1)*5, 1200.0) # 105, 110, 115, 120.
    
    # Close 120.
    # EMA 200 must be > 120. Say 125.
    # Trend |120-125| = 5 < 25. OK.
    
    decision = scalping_engine.evaluate(c_base, 125.0, 100.0, 110.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is not None
    assert decision.signal_intent.direction == -1

def test_trend_day_filter_suppression(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    # Drop to 80.
    c_base[-1] = create_candle(80.0, 1200.0)
    
    # EMA 200 = 40. Dist = 40. ATR = 10.
    # 40 > 2.5*10=25. Filter Active.
    
    decision = scalping_engine.evaluate(c_base, 40.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None
    assert decision.trend_day_filter is True

def test_bb_fail_suppression(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    # Flat price 100. Close 100. BB Lower ~ 90.
    # Close 100 > 90. No breakout.
    
    decision = scalping_engine.evaluate(c_base, 90.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None
    assert decision.bb_signal is False

# --- Scoring Tests ---

def test_score_max_5(scalping_engine, permissions):
    # Long Setup
    c_base = generate_candles(200, 100.0, 1000.0)
    # Drop to make Close < Lower BB and RSI < 30
    # Drop to 80.
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1000.0) # Vol 1000 keeps VMA low
    c_base[-1] = create_candle(80.0, 1600.0) # Spike > 1.5 * 1000
    
    # Close 80. EMA 200 = 75.
    # Aligned: EMA_20_4H > EMA_50_4H (Long)
    
    decision = scalping_engine.evaluate(c_base, 75.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    
    # Score:
    # +2 BB
    # +1 Aligned (100 > 90)
    # +1 Vol Bonus (1600 > 1500)
    # +1 RSI Zone (Assuming drop triggered < 30)
    assert decision.signal_intent.score == 5

def test_score_4_no_vol_bonus(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1000.0) # Vol 1000 keeps VMA low
    c_base[-1] = create_candle(80.0, 1200.0) # 1200 > 1.1*1000 (Entry OK) but < 1.5*1000 (Bonus Fail)
    
    decision = scalping_engine.evaluate(c_base, 75.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    
    # Score: 2(BB) + 1(Align) + 1(RSI) + 0(Vol) = 4
    assert decision.signal_intent.score == 4

def test_score_suppression_low(scalping_engine, permissions):
    # Unaligned Trend
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1000.0)
    c_base[-1] = create_candle(80.0, 1200.0)
    
    # EMA 20 4H (80) < EMA 50 4H (90) -> Short Trend.
    # Signal is Long. Unaligned.
    
    decision = scalping_engine.evaluate(c_base, 75.0, 80.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    
    # Score: 2(BB) + 0(Align) + 1(RSI) + 0(Vol) = 3.
    # Suppressed < 4.
    assert decision.signal_intent is None
    assert decision.score == 3

# --- Daily Halt Tests ---

def test_daily_halt_trigger(scalping_engine, permissions):
    # Initialize reset date to avoid auto-reset during evaluate
    scalping_engine._last_reset_date = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
    
    # 4 losses
    for _ in range(4):
        scalping_engine.record_trade_result(False)
    assert scalping_engine._halted_today is False
    
    # 5th loss
    scalping_engine.record_trade_result(False)
    assert scalping_engine._halted_today is True
    
    # Next eval should be blocked
    c_base = generate_candles(200, 100.0, 1000.0) # Should pass gates if not halted
    decision = scalping_engine.evaluate(c_base, 100.0, 100.0, 100.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    assert decision.signal_intent is None
    assert decision.halted_today is True

def test_daily_halt_reset_on_win(scalping_engine):
    for _ in range(4):
        scalping_engine.record_trade_result(False)
    assert scalping_engine._consecutive_losses == 4
    
    scalping_engine.record_trade_result(True)
    assert scalping_engine._consecutive_losses == 0

def test_daily_halt_reset_new_day(scalping_engine, permissions):
    scalping_engine._halted_today = True
    scalping_engine._consecutive_losses = 5
    scalping_engine._last_reset_date = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
    
    c_base = generate_candles(200)
    # Next day
    decision = scalping_engine.evaluate(c_base, 100.0, 100.0, 100.0, 10.0, 1.0, permissions, datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc))
    
    assert decision.halted_today is False
    assert scalping_engine._consecutive_losses == 0

# --- Min Data Guard ---

def test_min_data_guard(scalping_engine, permissions):
    candles = generate_candles(199)
    with pytest.raises(ValueError):
        scalping_engine.evaluate(candles, 100.0, 100.0, 100.0, 1.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))

# --- Signal Intent Schema ---

def test_signal_intent_schema(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1600.0)
    c_base[-1] = create_candle(80.0, 1600.0)
    
    # Close 80.
    
    decision = scalping_engine.evaluate(c_base, 75.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
    
    intent = decision.signal_intent
    assert intent.strategy_name == "SCALP"
    assert intent.execution_constraints.max_spread_bp == 2.0
    assert intent.tp_plan.tp1_distance == 5.0 # 0.5 * ATR(10)
    assert intent.sl_distance == 10.0 # 1.0 * ATR(10)
    assert intent.timeout_plan.max_bars == 6

# --- Redis Publish ---

def test_redis_publish_called(scalping_engine, permissions):
    c_base = generate_candles(200, 100.0, 1000.0)
    for i in range(15):
        c_base[-(15-i)] = create_candle(100.0 - i, 1600.0)
    c_base[-1] = create_candle(80.0, 1600.0)
        
    with patch.object(scalping_engine, '_publish_signal') as mock_pub:
        scalping_engine.evaluate(c_base, 75.0, 100.0, 90.0, 10.0, 1.0, permissions, datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc))
        mock_pub.assert_called_once()

def test_naive_datetime_handling(scalping_engine, permissions):
    c_base = generate_candles(200)
    naive = datetime(2026, 3, 20, 10, 0)
    # Should not raise error and convert
    decision = scalping_engine.evaluate(c_base, 100.0, 100.0, 100.0, 1.0, 3.0, permissions, naive) # Spread 3.0 fails gate
    assert decision.timestamp_utc.tzinfo == timezone.utc

def test_ensure_utc_validators():
    # Helper models
    tp = TPPlan(tp1_distance=1.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=6, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=2.0, max_slippage_bp=3.0, min_quote_fresh_ms=300)
    
    # SignalIntent - Naive
    si_naive = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime(2026, 1, 1, 12, 0)
    )
    assert si_naive.timestamp.tzinfo == timezone.utc
    
    # SignalIntent - Aware (Already UTC)
    si_aware = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=1.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    )
    assert si_aware.timestamp.tzinfo == timezone.utc

    # ScalpingDecision - Naive
    sd_naive = ScalpingDecision(
        signal_intent=None, bb_signal=False, rsi_value=50.0, ema_200_value=100.0,
        trend_day_filter=False, score=0, consecutive_losses=0, halted_today=False,
        timestamp_utc=datetime(2026, 1, 1, 12, 0)
    )
    assert sd_naive.timestamp_utc.tzinfo == timezone.utc

    # ScalpingDecision - Aware
    sd_aware = ScalpingDecision(
        signal_intent=None, bb_signal=False, rsi_value=50.0, ema_200_value=100.0,
        trend_day_filter=False, score=0, consecutive_losses=0, halted_today=False,
        timestamp_utc=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    )
    assert sd_aware.timestamp_utc.tzinfo == timezone.utc

def test_calculate_indicators_short_data(scalping_engine):
    # Test internal method with insufficient data
    closes = np.array([100.0] * 10, dtype=np.float64)
    volumes = np.array([1000.0] * 10, dtype=np.float64)
    
    # Should return defaults
    bb_u, bb_l, rsi, ema, vma, sma = scalping_engine._calculate_indicators(closes, volumes)
    assert rsi == 50.0
    assert ema == 0.0
