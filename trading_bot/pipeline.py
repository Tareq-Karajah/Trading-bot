from datetime import datetime, timezone
from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Imports from internal modules ---
from trading_bot.core.models import OHLCV
from trading_bot.data.ingestion import DataIngestion
from trading_bot.macro.engine import MacroBiasEngine, MacroState
from trading_bot.regime.engine import RegimeEngine, RegimeState
from trading_bot.dispatcher.engine import StrategyDispatcher, DispatcherPermissions
from trading_bot.strategies.swing.engine import SwingEngine
from trading_bot.strategies.orb.engine import ORBEngine
from trading_bot.strategies.news.engine import NewsBreakoutEngine
from trading_bot.strategies.scalping.engine import ScalpingEngine
from trading_bot.risk.engine import RiskEngine, RiskDecision, SignalIntent
from trading_bot.portfolio.coordinator import PortfolioCoordinator, CoordinatorDecision, StrategyMetrics
from trading_bot.execution.quality_gate import ExecutionQualityGate, ExecutionDecision
from trading_bot.execution.paper_engine import PaperExecutionEngine, TradeResult
from trading_bot.monitoring.monitor import SystemHealthMonitor

class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    symbols: List[str]
    timeframes: List[str]
    equity: float
    contract_oz: float
    broker_min_lot: float
    paper_only: bool = True
    dry_run: bool = True

class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    candle: OHLCV
    regime: Optional[RegimeState]
    macro_state: Optional[MacroState]
    permissions: Optional[DispatcherPermissions]
    signals: List[SignalIntent]
    risk_decisions: List[RiskDecision]
    coordinator_decisions: List[CoordinatorDecision]
    execution_decisions: List[ExecutionDecision]
    paper_fills: List[TradeResult]
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

class TradingPipeline:
    PAPER_ONLY: bool = True
    DRY_RUN: bool = True

    def __init__(self, config: PipelineConfig) -> None:
        if not self.PAPER_ONLY or not self.DRY_RUN:
            raise RuntimeError("LIVE TRADING NOT PERMITTED — PAPER_ONLY and DRY_RUN must both be True")
        if not config.paper_only or not config.dry_run:
            raise RuntimeError("Config LIVE TRADING NOT PERMITTED — paper_only and dry_run must both be True")
            
        self.config = config
        
        # Instantiate modules
        self.data_ingestion = DataIngestion()
        self.macro_engine = MacroBiasEngine()
        self.regime_engine = RegimeEngine()
        self.dispatcher = StrategyDispatcher()
        self.swing_engine = SwingEngine()
        self.orb_engine = ORBEngine()
        self.news_engine = NewsBreakoutEngine()
        self.scalp_engine = ScalpingEngine()
        self.risk_engine = RiskEngine()
        self.portfolio_coordinator = PortfolioCoordinator()
        self.quality_gate = ExecutionQualityGate()
        self.paper_engine = PaperExecutionEngine()
        self.monitor = SystemHealthMonitor()
        
        # Buffers (simplified for pipeline integration)
        # In a real app, these would be managed by DataIngestion or a shared BufferManager.
        # We simulate passing them as lists of OHLCV
        self.m5_buffer: List[OHLCV] = []
        self.m15_buffer: List[OHLCV] = []
        self.h4_buffer: List[OHLCV] = []
        self.d1_buffer: List[OHLCV] = []
        
        self.current_equity = config.equity

    async def start(self) -> None:
        """Start data ingestion and begin processing."""
        # Typically this would start background tasks.
        # For this stage, we just provide the hook.
        pass

    async def stop(self) -> None:
        """Graceful shutdown — close all positions via paper engine."""
        open_positions = self.paper_engine.get_open_positions()
        for pos in open_positions:
            self.paper_engine.force_close(pos.order_id, reason="SHUTDOWN")
            
        # Log shutdown
        self.monitor.logger.info("PIPELINE_SHUTDOWN", extra={"details": {"closed_positions": len(open_positions)}})

    async def process_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: OHLCV,
    ) -> PipelineResult:
        """Process one candle through the full pipeline."""
        
        now_utc = datetime.now(timezone.utc)
        
        # 1. Update Buffers
        if timeframe == "M5":
            self.m5_buffer.append(candle)
        elif timeframe == "M15":
            self.m15_buffer.append(candle)
        elif timeframe == "H4":
            self.h4_buffer.append(candle)
        elif timeframe == "D1":
            self.d1_buffer.append(candle)
            
        # 2. MacroBiasEngine
        from trading_bot.macro.engine import MacroState, MacroBiasDecision
        macro_state = MacroState.MACRO_NEUTRAL # Placeholder
        try:
            # Extract closes
            d1_closes = [c.close for c in self.d1_buffer]
            # Provide dummy tlt_closes and atr values
            macro_decision = await self.macro_engine.evaluate(
                usdx_closes=d1_closes,
                tlt_closes=d1_closes, # Mock
                atr_daily=1.0, # Mock
                atr_baseline_252d=1.0 # Mock
            )
            macro_state = macro_decision.macro_state
        except Exception:
            pass

        # 3. RegimeEngine
        regime_state = RegimeState.LOW_VOL # Placeholder fallback
        try:
            regime_decision = self.regime_engine.evaluate(
                symbol=symbol,
                timeframe=timeframe,
                candles=self.m5_buffer
            )
            regime_state = regime_decision.current_regime
        except Exception:
            pass
            
        # 4. StrategyDispatcher
        from trading_bot.macro.engine import MacroState
        # Need to handle missing modules gracefully or use mock objects if allowed
        class MockCalendar:
            _events: list[str] = []
        
        from trading_bot.macro.calendar import EventRiskState
        event_risk = EventRiskState.NO_EVENT_RISK

        permissions = self.dispatcher.evaluate(
            regime=regime_state,
            macro_state=macro_state,
            risk_scalar=1.0,
            swing_dir=0,
            now_utc=now_utc,
            calendar_service=MockCalendar(), # type: ignore
            event_risk_state=event_risk
        )
        
        # 5-8. Strategies
        signals: List[Any] = []
        
        # Swing
        if getattr(permissions, "allow_swing_rebalance", False):
            try:
                d1_closes = [c.close for c in self.d1_buffer]
                swing_dec = self.swing_engine.evaluate(
                    daily_closes=d1_closes,
                    atr_fast=1.0,
                    atr_slow=1.0,
                    permissions=permissions,
                    now_utc=now_utc
                )
                if swing_dec and swing_dec.signal_intent:
                    signals.append(swing_dec.signal_intent)
            except Exception:
                pass
                
        # ORB
        if getattr(permissions, "allow_orb", False):
            try:
                orb_dec = self.orb_engine.evaluate(
                    candle=candle,
                    atr_14_m15=1.0,
                    atr_daily_14=1.0,
                    or_median_20=1.0,
                    vma_20=1.0,
                    permissions=permissions
                )
                if orb_dec and getattr(orb_dec, "signal_intent", None):
                    signals.append(orb_dec.signal_intent)
            except Exception:
                pass
                
        # News
        if getattr(permissions, "allow_news", False):
            try:
                m5_closes = [c.close for c in self.m5_buffer]
                class DummyEvent:
                    pass
                news_dec = self.news_engine.evaluate(
                    m5_closes=m5_closes,
                    current_candle=candle,
                    atr_14_m5=1.0,
                    spread_live_bp=1.0,
                    spread_median_100=1.0,
                    quote_age_ms=100,
                    order_lots=1.0,
                    adv_daily_lots=100.0,
                    event=DummyEvent(), # type: ignore
                    event_time_utc=now_utc,
                    permissions=permissions
                )
                if news_dec and getattr(news_dec, "signal_intent", None):
                    signals.append(news_dec.signal_intent)
            except Exception:
                pass
                
        # Scalping
        if getattr(permissions, "allow_scalp", False):
            try:
                scalp_dec = self.scalp_engine.evaluate(
                    candles_m5=self.m5_buffer,
                    ema_200_m5=1.0,
                    ema_20_4h=1.0,
                    ema_50_4h=1.0,
                    atr_14_m5=1.0,
                    spread_live_bp=1.0,
                    permissions=permissions,
                    now_utc=now_utc
                )
                if scalp_dec and getattr(scalp_dec, "signal_intent", None):
                    signals.append(scalp_dec.signal_intent)
            except Exception:
                pass

        # 9-12. Execution Flow
        risk_decisions: List[RiskDecision] = []
        coordinator_decisions: List[CoordinatorDecision] = []
        execution_decisions: List[ExecutionDecision] = []
        
        # Fetch open positions from PaperEngine (it uses PaperPosition, but PortfolioCoordinator expects Position)
        # We need to map them if needed, or assume PortfolioCoordinator handles duck-typing.
        # Let's assume an empty list for now if mapping is complex, or map them.
        open_paper_positions = self.paper_engine.get_open_positions()
        # Mock Position list for Coordinator
        open_positions: list[Any] = []
        
        strategy_metrics: dict[str, StrategyMetrics] = {} # Mock metrics

        for signal in signals:
            # 9. RiskEngine
            try:
                risk_decision = self.risk_engine.evaluate(
                    signal=signal,
                    equity=self.current_equity,
                    regime_risk_scalar=1.0,
                    macro_risk_modifier=1.0,
                    quality_weight=1.0,
                    contract_oz_per_lot=self.config.contract_oz,
                    broker_min_lot_size=self.config.broker_min_lot,
                    open_positions=open_positions,
                    daily_pnl_pct=0.0,
                    weekly_pnl_pct=0.0,
                    peak_equity=self.current_equity
                )
                risk_decisions.append(risk_decision)
                
                # 10. PortfolioCoordinator
                coord_decision = self.portfolio_coordinator.evaluate(
                    risk_decision=risk_decision,
                    signal=signal,
                    open_positions=open_positions,
                    strategy_metrics=strategy_metrics,
                    equity=self.current_equity,
                    contract_oz=self.config.contract_oz
                )
                coordinator_decisions.append(coord_decision)
                
                # 11. ExecutionQualityGate
                exec_decision = self.quality_gate.evaluate(
                    signal=signal,
                    spread_live_bp=1.0, # Mock
                    spread_median_100=1.0, # Mock
                    quote_age_ms=100, # Mock
                    order_lots=1.0, # Mock
                    adv_daily_lots=100.0, # Mock
                    api_error_rate_5min=0.0, # Mock
                    now_utc=now_utc
                )
                execution_decisions.append(exec_decision)
                
                # 12. PaperExecutionEngine
                state_str = str(getattr(exec_decision.allowed_state, "value", exec_decision.allowed_state))
                # ExecutionDecision doesn't have 'approved' in Stage 11 spec, it uses allowed_state (ALLOWED, DEGRADED, BLOCKED)
                if state_str != "BLOCKED":
                    # Paper engine needs atr, quantity, symbol
                    # We will use dummy atr=1.0, quantity=adjusted_size
                    self.paper_engine.submit_order(
                        execution_decision=exec_decision,
                        signal=signal,
                        atr=1.0,
                        quantity=coord_decision.adjusted_size,
                        symbol=symbol
                    )
            except Exception:
                pass
                
        # Process existing open orders in PaperEngine on bar close
        paper_fills = self.paper_engine.on_bar_close(
            symbol=symbol,
            close=candle.close,
            high=candle.high,
            low=candle.low,
            now_utc=now_utc
        )
        
        # 13. SystemHealthMonitor
        self.monitor.record_pnl(0.0, 0.0) # Dummy update
        
        return PipelineResult(
            candle=candle,
            regime=regime_state,
            macro_state=macro_state,
            permissions=permissions,
            signals=signals,
            risk_decisions=risk_decisions,
            coordinator_decisions=coordinator_decisions,
            execution_decisions=execution_decisions,
            paper_fills=paper_fills,
            timestamp_utc=now_utc
        )
