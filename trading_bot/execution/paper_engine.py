from datetime import datetime, timezone, time
from typing import List, Optional, Dict, Any
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.execution.quality_gate import ExecutionDecision
from trading_bot.risk.engine import SignalIntent

# --- Enums & Data Models ---

class PaperOrderState(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    SL_HIT = "SL_HIT"
    TP1_HIT = "TP1_HIT"
    TRAILING_STOPPED = "TRAILING_STOPPED"
    EXPIRED = "EXPIRED"
    MANUALLY_CLOSED = "MANUALLY_CLOSED"

class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    order_id: str
    event: str
    price: float
    timestamp_utc: datetime
    details: Dict[str, Any]

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

class TradeResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    order_id: str
    strategy_name: str
    direction: int
    entry_price: float
    exit_price: float
    pnl_r: float
    exit_reason: str
    bars_held: int
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

class PaperPosition(BaseModel):
    """
    Mutable position tracking for Paper Execution Engine.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    order_id: str
    strategy_name: str
    symbol: str
    direction: int    # +1 | -1
    entry_price: float
    current_size: float  # lots remaining
    initial_size: float
    stop_price: float
    initial_stop_price: float # Added for R-calc
    tp1_price: Optional[float]
    trail_active: bool
    trail_price: Optional[float]
    highest_close: float  # for trail computation
    bars_held: int
    state: PaperOrderState
    entry_time_utc: datetime
    atr_at_entry: float
    pnl_r: float   # running R-multiple
    
    # Configuration for specific strategy rules
    mandatory_exit_utc: Optional[str] = None # For ORB
    max_bars: Optional[int] = None           # For SCALP
    tp1_distance: float = 0.0                # For calculating TP1 fill
    trail_multiplier: Optional[float] = None # For calculating trail

    @field_validator("entry_time_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Engine ---

class PaperExecutionEngine:
    """
    Paper Execution Engine (Stage 12).
    Simulates order execution without exchange connection.
    Strictly enforces PAPER_ONLY and DRY_RUN.
    """
    
    PAPER_ONLY: bool = True
    DRY_RUN: bool = True
    
    def __init__(self) -> None:
        self._positions: Dict[str, PaperPosition] = {}
        self._audit_log: List[AuditEntry] = []
        
    def submit_order(
        self,
        execution_decision: ExecutionDecision,
        signal: SignalIntent,
        atr: float,
        quantity: float,
        symbol: str,
    ) -> PaperPosition:
        """
        Submit an order for paper execution.
        """
        # 1. Safety Checks
        if not self.PAPER_ONLY or not self.DRY_RUN:
             raise RuntimeError("SAFETY_VIOLATION: PaperExecutionEngine must have PAPER_ONLY=True and DRY_RUN=True")
             
        # 2. Handle Blocked
        if execution_decision.allowed_state == "BLOCKED":
            raise ValueError(f"EXECUTION_BLOCKED: {execution_decision.blocked_reason}")
            
        # 3. Determine Size (Degraded logic)
        initial_size = quantity
        if execution_decision.allowed_state == "DEGRADED":
             initial_size *= 0.5
             
        # 4. Determine Stop and TP
        if signal.direction == 1: # Long
            stop_price = signal.entry_trigger - signal.sl_distance
            tp1_price = signal.entry_trigger + signal.tp_plan.tp1_distance
        else: # Short
            stop_price = signal.entry_trigger + signal.sl_distance
            tp1_price = signal.entry_trigger - signal.tp_plan.tp1_distance
            
        # 5. Determine Trail Multiplier
        trail_mult = None
        if signal.strategy_name == "SWING":
            trail_mult = 3.0
        elif signal.strategy_name == "ORB":
            trail_mult = 2.0
        elif signal.strategy_name == "SCALP":
            trail_mult = 1.0
        # NEWS has no trail
        
        # 6. Create Position
        order_id = str(uuid4())
        position = PaperPosition(
            order_id=order_id,
            strategy_name=signal.strategy_name,
            symbol=symbol,
            direction=signal.direction,
            entry_price=signal.entry_trigger,
            current_size=initial_size,
            initial_size=initial_size,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            tp1_price=tp1_price,
            trail_active=False,
            trail_price=None,
            highest_close=signal.entry_trigger,
            bars_held=0,
            state=PaperOrderState.PENDING,
            entry_time_utc=execution_decision.timestamp_utc,
            atr_at_entry=atr,
            pnl_r=0.0,
            mandatory_exit_utc=signal.timeout_plan.mandatory_exit_utc,
            max_bars=signal.timeout_plan.max_bars,
            tp1_distance=signal.tp_plan.tp1_distance,
            trail_multiplier=trail_mult
        )
        
        self._positions[order_id] = position
        
        # 7. Audit Log
        self._log_audit(
            order_id=order_id,
            event="ORDER_SUBMITTED",
            price=signal.entry_trigger,
            timestamp=execution_decision.timestamp_utc,
            details={
                "strategy": signal.strategy_name,
                "direction": signal.direction,
                "initial_size": initial_size,
                "stop_price": stop_price,
                "tp1_price": tp1_price,
                "degraded": (execution_decision.allowed_state == "DEGRADED")
            }
        )
        
        return position

    def on_bar_close(
        self,
        symbol: str,
        close: float,
        high: float,
        low: float,
        now_utc: datetime,
    ) -> List[TradeResult]:
        """
        Process bar close for all open positions.
        """
        completed_trades: List[TradeResult] = []
        
        for order_id, pos in list(self._positions.items()):
            if pos.symbol != symbol:
                continue
                
            # 1. Handle PENDING -> FILLED
            if pos.state == PaperOrderState.PENDING:
                pos.state = PaperOrderState.FILLED
                # Fill at entry_trigger (ideal fill)
                fill_price = pos.entry_price
                
                # If we are filling NOW, we might want to check if the current bar would have stopped us out immediately?
                # "Entry: filled at entry_trigger price on next available bar close".
                # Usually implies we entered at the START of this bar (or end of previous), and now we check results?
                # Or we enter AT the close of this bar?
                # "filled ... on next available bar close".
                # If we fill AT close, we can't stop out on THIS bar.
                # So we just log fill and continue.
                self._log_audit(
                    order_id=order_id,
                    event="ORDER_FILLED",
                    price=fill_price,
                    timestamp=now_utc,
                    details={"fill_type": "PAPER_IDEAL"}
                )
                continue

            # 2. Update State for Active Positions
            pos.bars_held += 1
            
            # Update Highest/Lowest Close for Trail
            if pos.direction == 1: # Long
                if close > pos.highest_close:
                    pos.highest_close = close
            else: # Short
                if close < pos.highest_close:
                    pos.highest_close = close
            
            # 3. Check Exits
            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None
            fully_closed = False
            
            # A. Check SL (Hard Stop)
            sl_hit = False
            if pos.direction == 1:
                if low <= pos.stop_price:
                    sl_hit = True
            else:
                if high >= pos.stop_price:
                    sl_hit = True
            
            if sl_hit:
                exit_reason = "SL_HIT"
                exit_price = pos.stop_price
                fully_closed = True
            
            # B. Check TP1 (if not hit)
            if not fully_closed and not pos.trail_active and pos.tp1_price:
                tp_hit = False
                if pos.direction == 1:
                    if high >= pos.tp1_price:
                        tp_hit = True
                else:
                    if low <= pos.tp1_price:
                        tp_hit = True
                
                if tp_hit:
                    # Partial Close
                    self._log_audit(
                        order_id=order_id,
                        event="TP1_HIT",
                        price=pos.tp1_price,
                        timestamp=now_utc,
                        details={"original_size": pos.current_size}
                    )
                    
                    # Reduce Size (50% of INITIAL size usually, or remaining?)
                    # Prompt: "TP1_HIT: 50% of position closed".
                    # Usually 50% of initial.
                    # "Partial fill handling ... After TP1: current_size = 0.5 * initial_size" (from TESTS section).
                    trade_size = pos.initial_size * 0.5
                    pos.current_size -= trade_size
                    
                    # Create TradeResult for the partial
                    risk_dist = abs(pos.entry_price - pos.initial_stop_price)
                    
                    r_mult = abs(pos.tp1_price - pos.entry_price) / risk_dist if risk_dist > 0 else 0.0
                    
                    completed_trades.append(TradeResult(
                        order_id=order_id,
                        strategy_name=pos.strategy_name,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=pos.tp1_price,
                        pnl_r=r_mult,
                        exit_reason="TP1_HIT",
                        bars_held=pos.bars_held,
                        timestamp_utc=now_utc
                    ))
                    
                    # Move SL to Breakeven
                    # "entry ± 0.05% fee buffer"
                    buffer = pos.entry_price * 0.0005
                    if pos.direction == 1:
                        pos.stop_price = pos.entry_price + buffer
                    else:
                        pos.stop_price = pos.entry_price - buffer
                        
                    # Activate Trail
                    pos.trail_active = True
                    pos.state = PaperOrderState.PARTIALLY_FILLED
                    
            # C. Check Trail (if active)
            if not fully_closed and pos.trail_active and pos.trail_multiplier:
                # Update Trail Price
                # "SWING trail = Highest_Close - 3.0 × ATR"
                # "filled at trailing_stop_price on bar where Close crosses it"
                new_trail = 0.0
                if pos.direction == 1:
                    new_trail = pos.highest_close - (pos.trail_multiplier * pos.atr_at_entry)
                    # Ratchet up only
                    if pos.trail_price is None or new_trail > pos.trail_price:
                        pos.trail_price = new_trail
                else:
                    new_trail = pos.highest_close + (pos.trail_multiplier * pos.atr_at_entry)
                    # Ratchet down only
                    if pos.trail_price is None or new_trail < pos.trail_price:
                        pos.trail_price = new_trail
                
                # Check Hit (Close price crosses trail)
                trail_hit = False
                if pos.direction == 1:
                    if close <= pos.trail_price: # Mypy safe: trail_price is set if logic reaches here? 
                        # Wait, pos.trail_price might be None initially?
                        # Yes, but we just set it above (if it was None, we set it).
                        # Logic: if pos.trail_price is None or new_trail > pos.trail_price.
                        # So pos.trail_price IS set.
                        # But Mypy doesn't know that easily.
                        # I'll assert pos.trail_price is not None if I rely on it.
                        pass
                    if pos.trail_price is not None and close <= pos.trail_price:
                        trail_hit = True
                else:
                    if pos.trail_price is not None and close >= pos.trail_price:
                        trail_hit = True
                        
                if trail_hit and pos.trail_price is not None:
                    exit_reason = "TRAILING_STOPPED"
                    exit_price = pos.trail_price
                    fully_closed = True

            # D. Check Timeouts
            if not fully_closed:
                # Scalp Max Bars
                if pos.max_bars and pos.bars_held >= pos.max_bars:
                    exit_reason = "EXPIRED"
                    exit_price = close
                    fully_closed = True
                
                # ORB Mandatory Exit
                elif pos.mandatory_exit_utc:
                    # Parse exit time
                    # Format "HH:MM". Assume UTC.
                    try:
                        exit_dt = datetime.strptime(pos.mandatory_exit_utc, "%H:%M").replace(
                            year=now_utc.year, month=now_utc.month, day=now_utc.day, tzinfo=timezone.utc
                        )
                        # Handle ORB Friday 19:30 special case?
                        # Prompt: "ORB: force_close at 20:45 UTC (19:30 UTC Fridays)"
                        # The `signal.timeout_plan` should already have the correct time?
                        # Or do we enforce Friday logic here?
                        # "ORB Friday mandatory exit at 19:30 UTC confirmed" in Tests.
                        # Signal generation logic usually sets the time.
                        # BUT, if `mandatory_exit_utc` is just a string "20:45", we need to adjust for Friday here?
                        # "SignalIntent ... timeout_plan ... mandatory_exit_utc".
                        # If the Strategy Engine sets it, it should be correct.
                        # But if it's static in config, we might need to adjust.
                        # Prompt: "MANDATORY EXIT RULES Per strategy: ORB: force_close at 20:45 UTC (19:30 UTC Fridays)".
                        # If `mandatory_exit_utc` passed in is "20:45", and it's Friday, we should override?
                        # I'll check if `now_utc` is Friday (weekday 4).
                        if pos.strategy_name == "ORB" and now_utc.weekday() == 4:
                            # Friday
                            # Force 19:30 check
                            friday_exit = now_utc.replace(hour=19, minute=30, second=0, microsecond=0)
                            if now_utc >= friday_exit:
                                exit_reason = "EXPIRED"
                                exit_price = close
                                fully_closed = True
                        else:
                            # Normal check
                            if now_utc >= exit_dt:
                                exit_reason = "EXPIRED"
                                exit_price = close
                                fully_closed = True
                    except ValueError:
                        pass # Invalid time format
            
            # Execute Full Close if triggered
            if fully_closed and exit_reason is not None and exit_price is not None:
                # Log
                self._log_audit(
                    order_id=order_id,
                    event=exit_reason,
                    price=exit_price,
                    timestamp=now_utc,
                    details={"bars_held": pos.bars_held}
                )
                
                # Calculate PnL R
                risk_dist = abs(pos.entry_price - pos.initial_stop_price)
                r_mult = abs(exit_price - pos.entry_price) / risk_dist if risk_dist > 0 else 0.0
                if pos.direction == 1:
                    if exit_price < pos.entry_price: r_mult = -r_mult
                else:
                    if exit_price > pos.entry_price: r_mult = -r_mult

                completed_trades.append(TradeResult(
                    order_id=order_id,
                    strategy_name=pos.strategy_name,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    pnl_r=r_mult,
                    exit_reason=exit_reason,
                    bars_held=pos.bars_held,
                    timestamp_utc=now_utc
                ))
                
                # Remove position
                del self._positions[order_id]

        return completed_trades

    def force_close(self, order_id: str, reason: str) -> TradeResult:
        if order_id not in self._positions:
            raise ValueError(f"Order ID {order_id} not found")
            
        pos = self._positions[order_id]
        now_utc = datetime.now(timezone.utc)
        
        # Use highest_close as best proxy for current price (since we don't have live feed here)
        exit_price = pos.highest_close
        
        self._log_audit(
            order_id=order_id,
            event="MANUALLY_CLOSED",
            price=exit_price,
            timestamp=now_utc,
            details={"reason": reason}
        )
        
        risk_dist = abs(pos.entry_price - pos.initial_stop_price)
        r_mult = abs(exit_price - pos.entry_price) / risk_dist if risk_dist > 0 else 0.0
        if pos.direction == 1:
            if exit_price < pos.entry_price: r_mult = -r_mult
        else:
             if exit_price > pos.entry_price: r_mult = -r_mult
             
        result = TradeResult(
            order_id=order_id,
            strategy_name=pos.strategy_name,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl_r=r_mult,
            exit_reason="MANUALLY_CLOSED",
            bars_held=pos.bars_held,
            timestamp_utc=now_utc
        )
        
        del self._positions[order_id]
        return result

    def _log_audit(self, order_id: str, event: str, price: float, timestamp: datetime, details: Dict[str, Any]) -> None:
        entry = AuditEntry(
            order_id=order_id,
            event=event,
            price=price,
            timestamp_utc=timestamp,
            details=details
        )
        self._audit_log.append(entry)

    def get_audit_log(self) -> List[AuditEntry]:
        return self._audit_log
    
    def get_open_positions(self) -> List[PaperPosition]:
        return list(self._positions.values())
