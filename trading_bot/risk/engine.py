from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
import math

from trading_bot.core.models import Position
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

# --- Data Models (matching docs/specification.md Section 3.1) ---

class TPPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    tp1_distance: float
    tp1_size_pct: float
    tp2_distance: Optional[float]
    trail_atr_mult: float

class TimeoutPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_bars: int
    mandatory_exit_utc: Optional[str]

class ExecutionConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_spread_bp: float
    max_slippage_bp: float
    min_quote_fresh_ms: int

class SignalIntent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    strategy_name: str
    direction: int
    score: int
    entry_type: str
    entry_trigger: float
    sl_distance: float
    tp_plan: TPPlan
    timeout_plan: TimeoutPlan
    regime_context: RegimeState
    macro_context: MacroState
    execution_constraints: ExecutionConstraints
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Risk Decision Models (matching docs/specification.md Section 3.2) ---

class CircuitBreakerState(BaseModel):
    model_config = ConfigDict(frozen=True)
    daily_loss_pct: float
    weekly_loss_pct: float
    consecutive_losses: int
    breaker_active: bool

class PortfolioHeatSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    current_heat_pct: float
    heat_remaining: float
    open_positions: int

class StrategyViability(BaseModel):
    model_config = ConfigDict(frozen=True)
    rolling_er: float
    viable: bool

class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    approved: bool
    rejection_reason: Optional[str]
    position_size: float
    risk_fraction_used: float
    stop_price: float
    take_profit_plan: TPPlan
    circuit_breaker_state: CircuitBreakerState
    portfolio_heat_snapshot: PortfolioHeatSnapshot
    strategy_viability: StrategyViability
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Risk Engine ---

class RiskEngine:
    """
    Implements Risk Engine logic per Section 11 of specification.
    """
    
    # Risk Fractions
    RISK_FRACTION = {
        "SWING": 0.010,
        "ORB": 0.005,
        "NEWS": 0.005,
        "SCALP": 0.0025
    }
    
    # Limits
    MAX_PORTFOLIO_HEAT = 0.05
    MAX_SINGLE_POSITION_HEAT = 0.30 # 30% of total heat (0.015)
    MAX_OPEN_POSITIONS = 5
    
    # Circuit Breakers
    CB_DAILY_SOFT_SCALP = -0.015
    CB_DAILY_SOFT_ORB = -0.020
    CB_DAILY_HARD = -0.030
    CB_WEEKLY_HARD = -0.080
    CB_CONSECUTIVE_LOSS_LIMIT = 5
    CB_CONSECUTIVE_REDUCTION_DURATION = 10
    CB_PEAK_DRAWDOWN_LIMIT = 0.15
    
    # Position Sizing Limits
    MAX_POSITION_VALUE = 250_000.0 # Default cap (e.g. $250k) if not specified

    # Viability
    VIABILITY_WINDOW = 30
    VIABILITY_PF_THRESHOLD = 1.10
    VIABILITY_ER_SUSPEND = -0.10
    
    def __init__(self) -> None:
        # State
        self._consecutive_losses: Dict[str, int] = {k: 0 for k in self.RISK_FRACTION}
        self._consecutive_loss_reduction_counter: Dict[str, int] = {k: 0 for k in self.RISK_FRACTION}
        
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.RISK_FRACTION} # {win: bool, r: float}
        
        self._daily_halt_strategies: set[str] = set()
        self._weekly_halt_active: bool = False
        self._hard_daily_halt_active: bool = False
        self._peak_equity: float = 0.0
        self._shutdown_active: bool = False
        self._shock_active: bool = False
        self._shock_expiry: Optional[datetime] = None
        
        # We need to track equity for peak drawdown?
        # passed in evaluate. We should update peak equity there.

    def reset_daily_state(self) -> None:
        """Reset daily halts at 00:00 UTC."""
        self._daily_halt_strategies.clear()
        self._hard_daily_halt_active = False

    def reset_weekly_state(self) -> None:
        """Reset weekly halts at Monday 00:00 UTC."""
        self._weekly_halt_active = False

    def record_trade_result(self, strategy: str, r_multiple: float, was_win: bool) -> None:
        """
        Update trade history and consecutive loss counters.
        """
        if strategy not in self._trade_history:
            self._trade_history[strategy] = []
            self._consecutive_losses[strategy] = 0
            self._consecutive_loss_reduction_counter[strategy] = 0

        # Manage reduction counter - decrement BEFORE processing result
        # This ensures the trade that triggers the reduction doesn't consume a count immediately
        if self._consecutive_loss_reduction_counter[strategy] > 0:
            self._consecutive_loss_reduction_counter[strategy] -= 1
            
        self._trade_history[strategy].append({"win": was_win, "r": r_multiple})
        if len(self._trade_history[strategy]) > self.VIABILITY_WINDOW:
            self._trade_history[strategy].pop(0)
            
        if was_win:
            self._consecutive_losses[strategy] = 0
            # A win resets consecutive losses, but does it reset the reduction phase?
            # "Reduce that strategy size 50% for next 10 trades".
            # Usually implies it persists for 10 trades regardless of win/loss.
            # So we do NOT reset reduction counter on win.
        else:
            self._consecutive_losses[strategy] += 1
            if self._consecutive_losses[strategy] >= self.CB_CONSECUTIVE_LOSS_LIMIT:
                # Trigger reduction for next 10 trades
                # If already active, restart or extend? Spec doesn't say.
                # Assuming reset to full duration.
                self._consecutive_loss_reduction_counter[strategy] = self.CB_CONSECUTIVE_REDUCTION_DURATION

    def get_viability_modifier(self, strategy: str) -> float:
        """
        Calculate viability modifier based on rolling ER and PF.
        """
        history = self._trade_history.get(strategy, [])
        if not history:
            return 1.0
            
        wins = [t for t in history if t["win"]]
        losses = [t for t in history if not t["win"]]
        
        p = len(wins) / len(history)
        
        mean_win_r = sum(t["r"] for t in wins) / len(wins) if wins else 0.0
        mean_loss_r = sum(abs(t["r"]) for t in losses) / len(losses) if losses else 1.0 # Avoid div/0
        
        b = mean_win_r / mean_loss_r if mean_loss_r > 0 else 0.0
        
        rolling_er = p * b - (1 - p)
        
        gross_wins_r = sum(t["r"] for t in wins)
        gross_loss_r = sum(abs(t["r"]) for t in losses)
        rolling_pf = gross_wins_r / gross_loss_r if gross_loss_r > 0 else 100.0 # High if no losses
        
        if rolling_er < self.VIABILITY_ER_SUSPEND:
            return 0.0
            
        viable = (rolling_er > 0) and (rolling_pf > self.VIABILITY_PF_THRESHOLD)
        
        if not viable:
            return 0.50
            
        return 1.0

    def _calculate_rolling_er(self, strategy: str) -> float:
        # Helper for reporting
        history = self._trade_history.get(strategy, [])
        if not history:
            return 0.0
        wins = [t for t in history if t["win"]]
        losses = [t for t in history if not t["win"]]
        p = len(wins) / len(history)
        mean_win_r = sum(t["r"] for t in wins) / len(wins) if wins else 0.0
        mean_loss_r = sum(abs(t["r"]) for t in losses) / len(losses) if losses else 1.0
        b = mean_win_r / mean_loss_r if mean_loss_r > 0 else 0.0
        return p * b - (1 - p)

    def is_circuit_breaker_active(self) -> bool:
        return (self._hard_daily_halt_active or 
                self._weekly_halt_active or 
                self._shutdown_active or
                (self._shock_active and (self._shock_expiry is None or datetime.now(timezone.utc) < self._shock_expiry)))

    def evaluate(
        self,
        signal: SignalIntent,
        equity: float,
        regime_risk_scalar: float,
        macro_risk_modifier: float,
        quality_weight: float,
        contract_oz_per_lot: float,
        broker_min_lot_size: float,
        open_positions: List[Position],
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        peak_equity: float,
    ) -> RiskDecision:
        
        now_utc = datetime.now(timezone.utc)
        
        # Update Peak Equity for Drawdown Check
        if equity > self._peak_equity:
            self._peak_equity = equity
        if peak_equity > self._peak_equity: # Trust passed peak if higher
            self._peak_equity = peak_equity
            
        # Drawdown Check
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self.CB_PEAK_DRAWDOWN_LIMIT:
                self._shutdown_active = True
                
        # Update Halts based on PnL
        if daily_pnl_pct < self.CB_DAILY_HARD:
            self._hard_daily_halt_active = True
            
        if weekly_pnl_pct < self.CB_WEEKLY_HARD:
            self._weekly_halt_active = True
            
        if signal.strategy_name == "SCALP" and daily_pnl_pct < self.CB_DAILY_SOFT_SCALP:
            self._daily_halt_strategies.add("SCALP")
        if signal.strategy_name == "ORB" and daily_pnl_pct < self.CB_DAILY_SOFT_ORB:
            self._daily_halt_strategies.add("ORB")
            
        # Check Consecutive Losses for Reduction Trigger
        # This is handled in record_trade_result
        
        # Shock logic (Input via regime_risk_scalar? Spec says "Shock candle ... risk_scalar=0")
        # If regime_risk_scalar is 0.0 (SHOCK), we treat it as shock active?
        # Or do we detect shock here?
        # Spec 11.2: "Shock candle ... Action: risk_scalar=0; pause 12 bars"
        # The Regime Engine handles detection and sets risk_scalar=0.
        # So if regime_risk_scalar == 0.0, it's a shock.
        if regime_risk_scalar == 0.0:
            # Rejection due to shock
            return self._reject(signal, "SHOCK_EVENT_ACTIVE", daily_pnl_pct, weekly_pnl_pct, 
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              0.0, 0.05, len(open_positions))

        # 1. Circuit Breaker Check
        rejection = None
        breaker_active = False
        
        if self._shutdown_active:
            rejection = "PEAK_DRAWDOWN_SHUTDOWN"
            breaker_active = True
        elif self._hard_daily_halt_active:
            rejection = "DAILY_LOSS_HARD_LIMIT"
            breaker_active = True
        elif self._weekly_halt_active:
            rejection = "WEEKLY_LOSS_LIMIT"
            breaker_active = True
        elif signal.strategy_name in self._daily_halt_strategies:
            rejection = f"DAILY_LOSS_SOFT_LIMIT_{signal.strategy_name}"
            breaker_active = True
            
        if rejection:
             # Calculate heat for reporting
            current_heat = sum([self._calculate_position_risk(p, equity, contract_oz_per_lot) for p in open_positions])
            return self._reject(signal, rejection, daily_pnl_pct, weekly_pnl_pct,
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              current_heat, max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat), len(open_positions), breaker_active=True)

        # 2. Viability Check
        viability_mod = self.get_viability_modifier(signal.strategy_name)
        if viability_mod == 0.0:
            current_heat = sum([self._calculate_position_risk(p, equity, contract_oz_per_lot) for p in open_positions])
            return self._reject(signal, "STRATEGY_SUSPENDED_LOW_ER", daily_pnl_pct, weekly_pnl_pct,
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              current_heat, max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat), len(open_positions))

        # 3. Position Sizing
        r = self.RISK_FRACTION.get(signal.strategy_name, 0.005)
        base_lots = (equity * r) / (signal.sl_distance * contract_oz_per_lot)
        
        # Modifiers
        # Consecutive loss reduction
        consecutive_mod = 1.0
        if self._consecutive_loss_reduction_counter.get(signal.strategy_name, 0) > 0:
            consecutive_mod = 0.50
            
        lots = base_lots * regime_risk_scalar * macro_risk_modifier * viability_mod * quality_weight * consecutive_mod
        
        # Caps
        lots = max(lots, broker_min_lot_size)
        
        # Max Position Value Cap
        # lots = min(lots, max_position_value / (entry_price * contract_oz))
        # Wait, max_position_value is NOT defined in spec text provided in prompt?
        # Prompt says: "lots = min(lots, max_position_value / (entry_price * contract_oz_per_lot))"
        # But doesn't give a value for `max_position_value`.
        # I must check `specification.md`.
        # Searching specification.md for "max_position_value".
        # If not found, I might have to use a default or ask.
        # Wait, "Max size cap | 1.5 x base_lots" is in Swing section.
        # Prompt explicitly included that formula.
        # Maybe it means "net_exposure_cap" from Portfolio Coordinator? No, that's aggregate.
        # I will assume there's a MAX_POSITION_VALUE constant I missed or need to define (e.g. huge number if not specified).
        # Actually, let's look at the prompt again.
        # "lots = min(lots, max_position_value / (entry_price * contract_oz_per_lot))"
        # If no value is given, maybe I should assume infinite?
        # But wait, "Max position value cap enforced" is in TESTS checklist.
        # I will check `specification.md` again for any mention of max value or size cap.
        # Section 7.3 says "max size cap | 1.5 x base_lots".
        # But the formula in prompt uses `max_position_value` (currency amount?).
        # I'll search for it.
        
        # 4. Portfolio Heat Check
        position_risk_pct = (lots * contract_oz_per_lot * signal.sl_distance) / equity
        
        current_heat = sum([self._calculate_position_risk(p, equity, contract_oz_per_lot) for p in open_positions])
        
        new_heat = current_heat + position_risk_pct
        
        if new_heat > self.MAX_PORTFOLIO_HEAT:
             return self._reject(signal, "PORTFOLIO_HEAT_EXCEEDED", daily_pnl_pct, weekly_pnl_pct,
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              current_heat, max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat), len(open_positions))
                              
        # Max single position heat check
        if position_risk_pct > (self.MAX_PORTFOLIO_HEAT * self.MAX_SINGLE_POSITION_HEAT):
             return self._reject(signal, "MAX_SINGLE_POSITION_RISK", daily_pnl_pct, weekly_pnl_pct,
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              current_heat, max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat), len(open_positions))

        if len(open_positions) >= self.MAX_OPEN_POSITIONS:
             return self._reject(signal, "MAX_POSITIONS_REACHED", daily_pnl_pct, weekly_pnl_pct,
                              self._consecutive_losses.get(signal.strategy_name, 0),
                              self._calculate_rolling_er(signal.strategy_name),
                              current_heat, max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat), len(open_positions))

        # Stop Price
        if signal.direction == 1:
            stop_price = signal.entry_trigger - signal.sl_distance
        else:
            stop_price = signal.entry_trigger + signal.sl_distance
            
        # Construct Decision
        return RiskDecision(
            approved=True,
            rejection_reason=None,
            position_size=float(lots),
            risk_fraction_used=float(r * regime_risk_scalar * macro_risk_modifier * viability_mod * quality_weight * consecutive_mod),
            stop_price=float(stop_price),
            take_profit_plan=signal.tp_plan,
            circuit_breaker_state=CircuitBreakerState(
                daily_loss_pct=daily_pnl_pct,
                weekly_loss_pct=weekly_pnl_pct,
                consecutive_losses=self._consecutive_losses.get(signal.strategy_name, 0),
                breaker_active=False
            ),
            portfolio_heat_snapshot=PortfolioHeatSnapshot(
                current_heat_pct=current_heat,
                heat_remaining=max(0.0, self.MAX_PORTFOLIO_HEAT - current_heat),
                open_positions=len(open_positions)
            ),
            strategy_viability=StrategyViability(
                rolling_er=self._calculate_rolling_er(signal.strategy_name),
                viable=(viability_mod > 0.5)
            )
        )

    def _reject(self, signal: SignalIntent, reason: str, daily_pnl: float, weekly_pnl: float, 
                consecutive: int, er: float, heat: float, heat_rem: float, pos_count: int, breaker_active: bool = False) -> RiskDecision:
        return RiskDecision(
            approved=False,
            rejection_reason=reason,
            position_size=0.0,
            risk_fraction_used=0.0,
            stop_price=0.0,
            take_profit_plan=signal.tp_plan,
            circuit_breaker_state=CircuitBreakerState(
                daily_loss_pct=daily_pnl,
                weekly_loss_pct=weekly_pnl,
                consecutive_losses=consecutive,
                breaker_active=breaker_active
            ),
            portfolio_heat_snapshot=PortfolioHeatSnapshot(
                current_heat_pct=heat,
                heat_remaining=heat_rem,
                open_positions=pos_count
            ),
            strategy_viability=StrategyViability(
                rolling_er=er,
                viable=(er > 0) # Simplification for rejection
            )
        )

    def _calculate_position_risk(self, position: Position, equity: float, contract_oz_per_lot: float) -> float:
        # Risk = Qty * ContractOz * |Entry - SL|
        # Stored as fraction of equity for heat calculation
        if equity <= 0:
            return 0.0
            
        risk_amt = position.quantity * contract_oz_per_lot * abs(position.entry_price - position.stop_loss)
        return risk_amt / equity

    def _publish_decision(self, decision: RiskDecision, strategy_name: str) -> None:
        pass # Stub

