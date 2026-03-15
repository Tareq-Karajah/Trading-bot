import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from trading_bot.pipeline import TradingPipeline, PipelineConfig, PipelineResult
from trading_bot.core.models import OHLCV

@pytest.fixture
def valid_config():
    return PipelineConfig(
        symbols=["XAUUSD"],
        timeframes=["M5", "M15", "H4", "D1"],
        equity=100_000.0,
        contract_oz=100.0,
        broker_min_lot=0.01,
        paper_only=True,
        dry_run=True
    )

@pytest.fixture
def sample_candle():
    return OHLCV(
        symbol="XAUUSD",
        timeframe="M5",
        timestamp=datetime.now(timezone.utc),
        open=2000.0,
        high=2010.0,
        low=1990.0,
        close=2005.0,
        volume=1000.0
    )

def test_pipeline_instantiation(valid_config):
    pipeline = TradingPipeline(valid_config)
    assert pipeline.config.equity == 100_000.0
    assert pipeline.PAPER_ONLY is True
    assert pipeline.DRY_RUN is True

def test_startup_guard_paper_only_false():
    original_val = TradingPipeline.PAPER_ONLY
    try:
        TradingPipeline.PAPER_ONLY = False
        with pytest.raises(RuntimeError, match="LIVE TRADING NOT PERMITTED"):
            TradingPipeline(PipelineConfig(symbols=["XAUUSD"], timeframes=["M5"], equity=100000.0, contract_oz=100.0, broker_min_lot=0.01))
    finally:
        TradingPipeline.PAPER_ONLY = original_val

def test_startup_guard_dry_run_false():
    original_val = TradingPipeline.DRY_RUN
    try:
        TradingPipeline.DRY_RUN = False
        with pytest.raises(RuntimeError, match="LIVE TRADING NOT PERMITTED"):
            TradingPipeline(PipelineConfig(symbols=["XAUUSD"], timeframes=["M5"], equity=100000.0, contract_oz=100.0, broker_min_lot=0.01))
    finally:
        TradingPipeline.DRY_RUN = original_val

def test_config_guard_paper_only_false():
    with pytest.raises(RuntimeError, match="Config LIVE TRADING NOT PERMITTED"):
        TradingPipeline(PipelineConfig(symbols=["XAUUSD"], timeframes=["M5"], equity=100000.0, contract_oz=100.0, broker_min_lot=0.01, paper_only=False))

def test_config_guard_dry_run_false():
    with pytest.raises(RuntimeError, match="Config LIVE TRADING NOT PERMITTED"):
        TradingPipeline(PipelineConfig(symbols=["XAUUSD"], timeframes=["M5"], equity=100000.0, contract_oz=100.0, broker_min_lot=0.01, dry_run=False))

@pytest.mark.asyncio
async def test_process_candle_other_timeframes(valid_config, sample_candle):
    pipeline = TradingPipeline(valid_config)
    
    await pipeline.process_candle("XAUUSD", "M15", sample_candle)
    await pipeline.process_candle("XAUUSD", "H4", sample_candle)
    await pipeline.process_candle("XAUUSD", "D1", sample_candle)
    
    assert len(pipeline.m15_buffer) == 1
    assert len(pipeline.h4_buffer) == 1
    assert len(pipeline.d1_buffer) == 1

@pytest.mark.asyncio
async def test_process_candle(valid_config, sample_candle):
    pipeline = TradingPipeline(valid_config)
    
    result = await pipeline.process_candle("XAUUSD", "M5", sample_candle)
    
    assert isinstance(result, PipelineResult)
    assert result.candle == sample_candle
    assert result.timestamp_utc.tzinfo == timezone.utc
    assert len(pipeline.m5_buffer) == 1

@pytest.mark.asyncio
async def test_process_candle_all_strategies(valid_config, sample_candle):
    pipeline = TradingPipeline(valid_config)
    
    pipeline.dispatcher.evaluate = MagicMock()
    from trading_bot.dispatcher.engine import DispatcherPermissions
    from trading_bot.macro.engine import MacroState
    from trading_bot.regime.engine import RegimeState
    mock_permissions = DispatcherPermissions(
        allow_swing_rebalance=True,
        allow_orb=True,
        allow_news=True,
        allow_scalp=True,
        risk_scalar=1.0,
        direction_constraint=0,
        macro_bias=MacroState.MACRO_NEUTRAL,
        regime=RegimeState.LOW_VOL,
        blackout_active=False,
        blackout_reason=None,
        reason="TEST"
    )
    pipeline.dispatcher.evaluate.return_value = mock_permissions

    from trading_bot.risk.engine import SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    
    sig = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=2000.0,
        sl_distance=20.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime.now(timezone.utc)
    )

    class MockDecision:
        def __init__(self, sig):
            self.signal_intent = sig
            
    mock_dec = MockDecision(sig)

    pipeline.swing_engine.evaluate = MagicMock(return_value=mock_dec)
    pipeline.orb_engine.evaluate = MagicMock(return_value=mock_dec)
    pipeline.news_engine.evaluate = MagicMock(return_value=mock_dec)
    pipeline.scalp_engine.evaluate = MagicMock(return_value=mock_dec)

    # RiskEngine
    from trading_bot.risk.engine import RiskDecision, CircuitBreakerState, PortfolioHeatSnapshot, StrategyViability
    rd = RiskDecision(
        approved=True, rejection_reason=None, position_size=1.0, risk_fraction_used=0.01,
        stop_price=1980.0, take_profit_plan=tp, 
        circuit_breaker_state=CircuitBreakerState(daily_loss_pct=0.0, weekly_loss_pct=0.0, consecutive_losses=0, breaker_active=False),
        portfolio_heat_snapshot=PortfolioHeatSnapshot(current_heat_pct=0.0, heat_remaining=0.05, open_positions=0),
        strategy_viability=StrategyViability(rolling_er=0.5, viable=True),
        timestamp_utc=datetime.now(timezone.utc)
    )
    pipeline.risk_engine.evaluate = MagicMock(return_value=rd)

    # Coordinator
    from trading_bot.portfolio.coordinator import CoordinatorDecision
    cd = CoordinatorDecision(
        approved=True, rejection_reason=None, adjusted_size=1.0, quality_weight=1.0,
        net_exposure=0.0, heat_after=0.0, budget_remaining={},
        timestamp_utc=datetime.now(timezone.utc)
    )
    pipeline.portfolio_coordinator.evaluate = MagicMock(return_value=cd)

    # Quality Gate
    from trading_bot.execution.quality_gate import ExecutionDecision
    ed = ExecutionDecision(
        allowed_state="ALLOWED", rejection_reason=None,
        blocked_reason=None,
        adjusted_quantity=1.0, current_spread_bp=1.0, current_slippage_bp=1.0,
        spread_snapshot=1.0, spread_vs_median=1.0, quote_freshness_ms=1000,
        expected_slippage_bp=1.0, routing_mode="TEST", order_intent_id="TEST_123",
        timestamp_utc=datetime.now(timezone.utc)
    )
    pipeline.quality_gate.evaluate = MagicMock(return_value=ed)
    
    # Paper Engine
    pipeline.paper_engine.submit_order = MagicMock()
    pipeline.paper_engine.on_bar_close = MagicMock(return_value=[])
    
    res = await pipeline.process_candle("XAUUSD", "M5", sample_candle)
    
    # Should have 4 signals (swing, orb, news, scalp)
    assert len(res.signals) == 4

@pytest.mark.asyncio
async def test_process_candle_engine_exceptions(valid_config, sample_candle):
    # Covers line 133-134 (macro_engine exception) and others
    pipeline = TradingPipeline(valid_config)
    
    # Mock to throw exception
    class BadMacroEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("Macro fail")
            
    class BadRegimeEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("Regime fail")
            
    class BadSwingEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("Swing fail")
            
    class BadOrbEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("Orb fail")
            
    class BadNewsEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("News fail")

    class BadScalpEngine:
        def evaluate(self, *args, **kwargs):
            raise Exception("Scalp fail")
            
    pipeline.macro_engine = BadMacroEngine()
    pipeline.regime_engine = BadRegimeEngine()
    pipeline.swing_engine = BadSwingEngine()
    pipeline.orb_engine = BadOrbEngine()
    pipeline.news_engine = BadNewsEngine()
    pipeline.scalp_engine = BadScalpEngine()
    
    from trading_bot.dispatcher.engine import DispatcherPermissions
    from trading_bot.macro.engine import MacroState
    from trading_bot.regime.engine import RegimeState
    mock_permissions = DispatcherPermissions(
        allow_swing_rebalance=True,
        allow_orb=True,
        allow_news=True,
        allow_scalp=True,
        risk_scalar=1.0,
        direction_constraint=0,
        macro_bias=MacroState.MACRO_NEUTRAL,
        regime=RegimeState.LOW_VOL,
        blackout_active=False,
        blackout_reason=None,
        reason="TEST"
    )
        
    pipeline.dispatcher.evaluate = MagicMock(return_value=mock_permissions)
    
    res = await pipeline.process_candle("XAUUSD", "M5", sample_candle)
    
    # Should complete without throwing
    from trading_bot.regime.engine import RegimeState
    assert res.regime == RegimeState.LOW_VOL
    assert len(res.signals) == 0

@pytest.mark.asyncio
async def test_process_candle_with_signals_and_execution(valid_config, sample_candle):
    pipeline = TradingPipeline(valid_config)
    
    # We want to cover the inner loop for signals -> risk -> coordinator -> quality -> execution
    # Let's mock the dispatcher to allow swing, and mock swing engine to return a signal.
    
    pipeline.dispatcher.evaluate = MagicMock()
    # Need to return actual DispatcherPermissions to pass Pydantic validation
    from trading_bot.dispatcher.engine import DispatcherPermissions
    from trading_bot.regime.engine import RegimeState
    from trading_bot.macro.engine import MacroState
    mock_permissions = DispatcherPermissions(
        allow_swing_rebalance=True,
        allow_orb=False,
        allow_news=False,
        allow_scalp=False,
        risk_scalar=1.0,
        direction_constraint=0,
        macro_bias=MacroState.MACRO_NEUTRAL,
        regime=RegimeState.LOW_VOL,
        blackout_active=False,
        blackout_reason=None,
        reason="TEST"
    )
    pipeline.dispatcher.evaluate.return_value = mock_permissions
    
    from trading_bot.risk.engine import SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
    from trading_bot.regime.engine import RegimeState
    from trading_bot.macro.engine import MacroState
    
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)

    sig = SignalIntent(
        strategy_name="SWING", direction=1, score=5, entry_type="MARKET", entry_trigger=2000.0,
        sl_distance=20.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime.now(timezone.utc)
    )
    
    class MockDecision:
        def __init__(self, sig):
            self.signal_intent = sig
            
    mock_dec = MockDecision(sig)
    pipeline.swing_engine.evaluate = MagicMock(return_value=mock_dec)
    
    # We also need to mock RiskEngine to return approved RiskDecision
    from trading_bot.risk.engine import RiskDecision, CircuitBreakerState, PortfolioHeatSnapshot, StrategyViability
    rd = RiskDecision(
        approved=True, rejection_reason=None, position_size=1.0, risk_fraction_used=0.01,
        stop_price=1980.0, take_profit_plan=tp, 
        circuit_breaker_state=CircuitBreakerState(daily_loss_pct=0.0, weekly_loss_pct=0.0, consecutive_losses=0, breaker_active=False),
        portfolio_heat_snapshot=PortfolioHeatSnapshot(current_heat_pct=0.0, heat_remaining=0.05, open_positions=0),
        strategy_viability=StrategyViability(rolling_er=0.5, viable=True),
        timestamp_utc=datetime.now(timezone.utc)
    )
    pipeline.risk_engine.evaluate = MagicMock(return_value=rd)
    
    # Coordinator
    from trading_bot.portfolio.coordinator import CoordinatorDecision
    cd = CoordinatorDecision(
        approved=True, rejection_reason=None, adjusted_size=1.0, quality_weight=1.0,
        net_exposure=0.0, heat_after=0.0, budget_remaining={},
        timestamp_utc=datetime.now(timezone.utc)
    )
    pipeline.portfolio_coordinator.evaluate = MagicMock(return_value=cd)
    
    # Quality Gate
    from trading_bot.execution.quality_gate import ExecutionDecision
    ed = ExecutionDecision(
        allowed_state="ALLOWED", rejection_reason=None,
        blocked_reason=None,
        adjusted_quantity=1.0, current_spread_bp=1.0, current_slippage_bp=1.0,
        spread_snapshot=1.0, spread_vs_median=1.0, quote_freshness_ms=1000,
        expected_slippage_bp=1.0, routing_mode="TEST", order_intent_id="TEST_123",
        timestamp_utc=datetime.now(timezone.utc)
    )
    # The pipeline accesses quality_gate.evaluate
    # We must ensure allowed_state is correctly matched. The code does `if str(exec_decision.allowed_state) != "BLOCKED":`
    pipeline.quality_gate.evaluate = MagicMock(return_value=ed)
    
    # We also need to mock the paper engine's submit_order correctly.
    # In Pipeline:
    # if exec_decision.approved and state_str != "BLOCKED":
    # It might be failing because state_str == "ExecutionState.ALLOWED"?
    # "ExecutionState.ALLOWED" != "BLOCKED". So it should pass.
    # Why is submit_order not called?
    # Because of an exception! 
    # "except Exception: pass" masks errors!
    # Let's remove the try/except block in the test, or verify the arguments match.
    # Wait, the pipeline has `except Exception:` block inside the loop.
    # Let's check `self.paper_engine.submit_order` signature:
    # `def submit_order(self, execution_decision: ExecutionDecision, signal: SignalIntent, atr: float, quantity: float, symbol: str) -> PaperPosition`
    # Mocking `submit_order` shouldn't raise exception unless `return_value` doesn't match? No, it's a mock.
    # Maybe `self.risk_engine.evaluate` threw exception?
    # Signature: `def evaluate(self, signal: SignalIntent, open_positions: list[Position], equity: float, win_rate: float, avg_win_loss_ratio: float, consecutive_losses: int) -> RiskDecision`
    # Mock args match.
    # Let's see what is really happening.
    
    # Paper Engine
    pipeline.paper_engine.submit_order = MagicMock()
    pipeline.paper_engine.on_bar_close = MagicMock(return_value=[])
    
    res = await pipeline.process_candle("XAUUSD", "M5", sample_candle)
    
    assert len(res.signals) == 1
    assert len(res.risk_decisions) == 1
    assert len(res.coordinator_decisions) == 1
    assert len(res.execution_decisions) == 1
    
    pipeline.paper_engine.submit_order.assert_called_once()
    pipeline.paper_engine.on_bar_close.assert_called_once()

@pytest.mark.asyncio
async def test_stop_calls_graceful_shutdown(valid_config):
    pipeline = TradingPipeline(valid_config)
    
    # Mock get_open_positions to return a dummy
    mock_pos = MagicMock()
    mock_pos.order_id = "test_123"
    pipeline.paper_engine.get_open_positions = MagicMock(return_value=[mock_pos])
    pipeline.paper_engine.force_close = MagicMock()
    
    await pipeline.stop()
    
    pipeline.paper_engine.force_close.assert_called_once_with("test_123", reason="SHUTDOWN")

@pytest.mark.asyncio
async def test_start_method(valid_config):
    pipeline = TradingPipeline(valid_config)
    # Start doesn't do much yet, just ensure it runs without error
    await pipeline.start()

@pytest.mark.asyncio
async def test_process_candle_execution_exceptions(valid_config, sample_candle):
    pipeline = TradingPipeline(valid_config)
    
    pipeline.dispatcher.evaluate = MagicMock()
    from trading_bot.dispatcher.engine import DispatcherPermissions
    from trading_bot.regime.engine import RegimeState
    from trading_bot.macro.engine import MacroState
    pipeline.dispatcher.evaluate.return_value = DispatcherPermissions(
        allow_swing_rebalance=True, allow_orb=False, allow_news=False, allow_scalp=False,
        risk_scalar=1.0, direction_constraint=0, macro_bias=MacroState.MACRO_NEUTRAL, regime=RegimeState.LOW_VOL,
        blackout_active=False, blackout_reason=None, reason="TEST"
    )

    from trading_bot.risk.engine import SignalIntent, TPPlan, TimeoutPlan, ExecutionConstraints
    from trading_bot.regime.engine import RegimeState
    from trading_bot.macro.engine import MacroState
    tp = TPPlan(tp1_distance=20.0, tp1_size_pct=0.5, tp2_distance=None, trail_atr_mult=1.0)
    to = TimeoutPlan(max_bars=10, mandatory_exit_utc=None)
    ec = ExecutionConstraints(max_spread_bp=10.0, max_slippage_bp=15.0, min_quote_fresh_ms=2000)
    sig = SignalIntent(
        strategy_name="TEST", direction=1, score=5, entry_type="MARKET", entry_trigger=2000.0,
        sl_distance=20.0, tp_plan=tp, timeout_plan=to,
        regime_context=RegimeState.LOW_VOL, macro_context=MacroState.MACRO_NEUTRAL,
        execution_constraints=ec,
        timestamp=datetime.now(timezone.utc)
    )

    class MockDecision:
        def __init__(self, sig):
            self.signal_intent = sig
            
    mock_dec = MockDecision(sig)
    pipeline.swing_engine.evaluate = MagicMock(return_value=mock_dec)
    
    # Exception in RiskEngine
    pipeline.risk_engine.evaluate = MagicMock(side_effect=Exception("Risk fail"))
    
    res = await pipeline.process_candle("XAUUSD", "M5", sample_candle)
    assert len(res.risk_decisions) == 0

def test_startup_guard_main():
    pass

def test_pipeline_result_naive_datetime(sample_candle):
    # Test that a naive datetime is converted to UTC
    naive_dt = datetime.now()  # no tzinfo
    result = PipelineResult(
        candle=sample_candle,
        regime=None,
        macro_state=None,
        permissions=None,
        signals=[],
        risk_decisions=[],
        coordinator_decisions=[],
        execution_decisions=[],
        paper_fills=[],
        timestamp_utc=naive_dt
    )
    assert result.timestamp_utc.tzinfo == timezone.utc
