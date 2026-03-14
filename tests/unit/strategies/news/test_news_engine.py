import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from trading_bot.strategies.news.engine import NewsBreakoutEngine, NewsDecision, SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
from trading_bot.core.models import OHLCV
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.macro.calendar import CalendarEvent, Tier, CalendarSeverity
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

@pytest.fixture
def news_engine():
    return NewsBreakoutEngine()

@pytest.fixture
def permissions():
    return DispatcherPermissions(
        allow_swing_rebalance=True, allow_orb=False, allow_news=True, allow_scalp=False,
        direction_constraint=0, macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL,
        risk_scalar=1.0, blackout_active=False, blackout_reason=None
    )

@pytest.fixture
def event_t1():
    return CalendarEvent(
        event_id="1", event_name="NFP", tier=Tier.T1, severity=CalendarSeverity.CRITICAL,
        scheduled_time_utc=datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc),
        pre_window_seconds=300, post_window_seconds=300, is_whitelisted_for_news_breakout=True
    )

def create_candle(price: float, timestamp: datetime):
    return OHLCV(
        symbol="XAUUSD", timeframe="5m", open=price, high=price+1, low=price-1, close=price,
        volume=1000, timestamp=timestamp
    )

# --- Pre-News Range Tests ---

def test_pre_news_range_calc(news_engine, permissions, event_t1):
    # 5 candles: Highs 101..105, Lows 99..95
    candles = []
    base_time = event_t1.scheduled_time_utc - timedelta(minutes=25)
    for i in range(5):
        c = OHLCV(
            symbol="XAU", timeframe="5m", open=100, high=100 + (i+1), low=100 - (i+1), close=100,
            volume=1000, timestamp=base_time + timedelta(minutes=i*5)
        )
        candles.append(c)
        
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=10))
    
    # High = 105. Low = 95.
    # ATR = 10. Buf = 0.2 * 10 = 2.0.
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    
    assert decision.pre_news_high == 105.0
    assert decision.pre_news_low == 95.0
    assert decision.buf_news == 2.0

# --- Gate Tests ---

def test_gate_spread_fail(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=10))
    
    # Spread 4.0 > 3.0 Max
    decision = news_engine.evaluate(
        candles, current, 10.0, 4.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.gates_passed is False

def test_gate_spread_dynamic_fail(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=10))
    
    # Median 1.0. Limit = min(3.0, 2.0*1.0) = 2.0.
    # Live 2.5 > 2.0. Fail.
    decision = news_engine.evaluate(
        candles, current, 10.0, 2.5, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.gates_passed is False

def test_gate_freshness_fail(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=10))
    
    # Age 600 > 500
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 600, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.gates_passed is False

def test_gate_slippage_fail(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=10))
    
    # Slip = Spread/2 + (Order/ADV)*100
    # Spread 2.0 -> 1.0.
    # Order 1000, ADV 10000 -> 0.1 * 100 = 10.0.
    # Total 11.0 > 5.0.
    decision = news_engine.evaluate(
        candles, current, 10.0, 2.0, 1.0, 100, 1000.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.gates_passed is False

def test_gate_session_fail(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    
    # 06:00 UTC (Outside 07-21)
    early = datetime(2026, 3, 20, 6, 0, tzinfo=timezone.utc)
    current = create_candle(100, early)
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, early, permissions
    )
    assert decision.gates_passed is False

# --- Entry & Expiry Tests ---

def test_entry_long_t1_pass(news_engine, permissions, event_t1):
    # High 101 (from create_candle(100) -> High 101). Buf 2. Trigger 103.
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    # Close 104 > 103. Time + 60s (within 3m).
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=60))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    
    assert decision.signal_intent is not None
    assert decision.signal_intent.direction == 1
    assert decision.signal_intent.entry_trigger == 103.0 # Trigger, not close

def test_entry_t1_expired(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    # + 181s (3m 1s) -> Expired
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=181))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.expiry_ok is False
    assert decision.signal_intent is None

def test_entry_t2_expiry(news_engine, permissions):
    event_t2 = CalendarEvent(
        event_id="2", event_name="CPI", tier=Tier.T2, severity=CalendarSeverity.HIGH,
        scheduled_time_utc=datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc),
        pre_window_seconds=300, post_window_seconds=300, is_whitelisted_for_news_breakout=True
    )
    candles = [create_candle(100, event_t2.scheduled_time_utc)] * 5
    
    # + 121s (2m 1s) -> Expired for T2
    current = create_candle(103, event_t2.scheduled_time_utc + timedelta(seconds=121))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t2, event_t2.scheduled_time_utc, permissions
    )
    assert decision.expiry_ok is False

def test_entry_t3_expiry(news_engine, permissions):
    event_t3 = CalendarEvent(
        event_id="3", event_name="Goods", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc),
        pre_window_seconds=300, post_window_seconds=300, is_whitelisted_for_news_breakout=True
    )
    candles = [create_candle(100, event_t3.scheduled_time_utc)] * 5
    
    # + 61s (1m 1s) -> Expired for T3
    current = create_candle(103, event_t3.scheduled_time_utc + timedelta(seconds=61))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t3, event_t3.scheduled_time_utc, permissions
    )
    assert decision.expiry_ok is False

# --- Chase Limit Tests ---

def test_chase_limit_fail(news_engine, permissions, event_t1):
    # Range 99-101 (Width 2). Buf 2. Trigger Long 103.
    # SL Dist = 103 - (99 - 2) = 103 - 97 = 6.
    # Limit = 1.5 * 6 = 9.
    # Max Price = 103 + 9 = 112.
    
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    # Close 113. Deviation 10 > 9.
    current = create_candle(113, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.signal_intent is None # Suppressed

# --- Daily Limit & Reset ---

def test_daily_limit_enforcement(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    # Trade 1
    d1 = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert d1.daily_trade_count == 1
    assert d1.signal_intent is not None
    
    # Trade 2
    d2 = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert d2.daily_trade_count == 2
    assert d2.signal_intent is not None
    
    # Trade 3 (Fail)
    d3 = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert d3.daily_trade_count == 2
    assert d3.signal_intent is None

def test_daily_reset(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    
    # Day 1
    t1 = datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc)
    current1 = create_candle(104, t1 + timedelta(seconds=30))
    news_engine.evaluate(candles, current1, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, t1, permissions)
    assert news_engine._daily_trade_count == 1
    
    # Day 2
    t2 = datetime(2026, 3, 21, 13, 30, tzinfo=timezone.utc)
    current2 = create_candle(104, t2 + timedelta(seconds=30))
    d2 = news_engine.evaluate(candles, current2, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, t2, permissions)
    assert d2.daily_trade_count == 1 # Reset to 0 then +1

# --- Signal Intent Schema ---

def test_signal_intent_fields(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    
    intent = decision.signal_intent
    assert intent.strategy_name == "NEWS"
    assert intent.tp_plan.tp1_size_pct == 1.00
    assert intent.tp_plan.trail_atr_mult == 0.0
    assert intent.execution_constraints.max_spread_bp == 3.0 # T1
    
    # TP = 2 * SL
    # Trigger 103. SL Level 97. Dist 6.
    # TP Dist = 12.
    assert intent.sl_distance == 6.0
    assert intent.tp_plan.tp1_distance == 12.0

# --- Coverage Fixes ---

def test_short_entry(news_engine, permissions, event_t1):
    # Low 99. Buf 2. Trigger 97.
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(96, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    
    assert decision.signal_intent.direction == -1
    assert decision.signal_intent.entry_trigger == 97.0

def test_insufficient_data(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 4 # < 5
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=30))
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.signal_intent is None

def test_redis_publish_stub(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    current = create_candle(104, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    with patch.object(news_engine, '_publish_signal') as mock_pub:
        news_engine.evaluate(candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions)
        mock_pub.assert_called_once()

def test_no_signal_triggered(news_engine, permissions, event_t1):
    # Gates pass, Expiry OK, but Price inside range
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    # Range 99-101. Buf 2. Trigger L 103, S 97.
    # Current 100 (Inside).
    current = create_candle(100, event_t1.scheduled_time_utc + timedelta(seconds=30))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.signal_intent is None
    assert decision.gates_passed is True

def test_validators_naive_coverage():
    # SignalIntent
    naive = datetime(2026, 3, 20, 12, 0)
    intent = SignalIntent(
        strategy_name="NEWS", direction=1, score=5, entry_type="MARKET", entry_trigger=100.0,
        sl_distance=10.0, tp_plan=TPPlan(tp1_distance=20.0, tp1_size_pct=1.0, tp2_distance=None, trail_atr_mult=0.0),
        timeout_plan=TimeoutPlan(max_bars=1, mandatory_exit_utc=None),
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ExecutionConstraints(max_spread_bp=3.0, max_slippage_bp=5.0, min_quote_fresh_ms=500),
        timestamp=naive
    )
    assert intent.timestamp.tzinfo == timezone.utc
    
    # NewsDecision
    decision = NewsDecision(
        signal_intent=None, pre_news_high=100.0, pre_news_low=90.0, buf_news=1.0,
        gates_passed=True, expiry_ok=True, daily_trade_count=0, timestamp_utc=naive
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc

def test_entry_t2_pass(news_engine, permissions):
    event_t2 = CalendarEvent(
        event_id="2", event_name="CPI", tier=Tier.T2, severity=CalendarSeverity.HIGH,
        scheduled_time_utc=datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc),
        pre_window_seconds=300, post_window_seconds=300, is_whitelisted_for_news_breakout=True
    )
    candles = [create_candle(100, event_t2.scheduled_time_utc)] * 5
    current = create_candle(104, event_t2.scheduled_time_utc + timedelta(seconds=60))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t2, event_t2.scheduled_time_utc, permissions
    )
    assert decision.signal_intent is not None
    assert decision.signal_intent.execution_constraints.max_spread_bp == 2.5 # T2

def test_entry_t3_pass(news_engine, permissions):
    event_t3 = CalendarEvent(
        event_id="3", event_name="Goods", tier=Tier.T3, severity=CalendarSeverity.LOW,
        scheduled_time_utc=datetime(2026, 3, 20, 13, 30, tzinfo=timezone.utc),
        pre_window_seconds=300, post_window_seconds=300, is_whitelisted_for_news_breakout=True
    )
    candles = [create_candle(100, event_t3.scheduled_time_utc)] * 5
    current = create_candle(104, event_t3.scheduled_time_utc + timedelta(seconds=30))
    
    decision = news_engine.evaluate(
        candles, current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t3, event_t3.scheduled_time_utc, permissions
    )
    assert decision.signal_intent is not None
    assert decision.signal_intent.execution_constraints.max_spread_bp == 3.0 # T3

def test_manual_validator_call():
    # Force coverage of ensure_utc
    naive = datetime(2026, 3, 20, 12, 0)
    res = SignalIntent.ensure_utc(naive)
    assert res.tzinfo == timezone.utc
    
    res2 = NewsDecision.ensure_utc(naive)
    assert res2.tzinfo == timezone.utc
    
    # Also test already aware path (lines 53, 74)
    aware = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    res3 = SignalIntent.ensure_utc(aware)
    assert res3 == aware

def test_evaluate_naive_datetime_defensive(news_engine, permissions, event_t1):
    # Mock OHLCV to have naive timestamp to trigger defensive check in evaluate
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    naive_ts = datetime(2026, 3, 20, 13, 30, 30) # Naive
    
    mock_candle = MagicMock(spec=OHLCV)
    mock_candle.high = 104
    mock_candle.low = 102
    mock_candle.close = 104
    mock_candle.timestamp = naive_ts
    
    # We need to ensure attribute access works for evaluate logic
    # evaluate uses: current_candle.timestamp, current_candle.close
    
    decision = news_engine.evaluate(
        candles, mock_candle, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, event_t1.scheduled_time_utc, permissions
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc

def test_naive_datetime_handling(news_engine, permissions, event_t1):
    candles = [create_candle(100, event_t1.scheduled_time_utc)] * 5
    # Naive times
    naive_event = event_t1.scheduled_time_utc.replace(tzinfo=None)
    naive_current = create_candle(103, naive_event + timedelta(seconds=30))
    # Note: create_candle doesn't enforce naive, but we pass it.
    # OHLCV validator enforces UTC? No, create_candle makes it.
    # If OHLCV validator enforces UTC, then naive_current.timestamp will be UTC.
    # Let's bypass create_candle helper if needed, or ensure OHLCV validator works.
    # OHLCV model has ensure_utc. So it will convert.
    # But `event_time_utc` passed to evaluate can be naive.
    
    decision = news_engine.evaluate(
        candles, naive_current, 10.0, 1.0, 1.0, 100, 1.0, 10000.0, event_t1, naive_event, permissions
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc
