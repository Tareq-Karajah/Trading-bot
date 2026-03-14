from datetime import datetime, timezone, time
from typing import Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator
import math

from trading_bot.risk.engine import SignalIntent

# --- Data Models (matching docs/signal_contracts.md Section 3.3) ---

class ExecutionDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    allowed_state: str # "ALLOWED", "DEGRADED", "BLOCKED"
    blocked_reason: Optional[str]
    spread_snapshot: float
    spread_vs_median: float
    quote_freshness_ms: int
    expected_slippage_bp: float
    actual_slippage_bp: Optional[float] = None
    routing_mode: Optional[str] # "LIMIT", "MARKET_FALLBACK", or None if blocked
    order_intent_id: str
    degraded_action: Optional[str] = None # "REDUCE_SIZE_50PCT"
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Execution Quality Gate ---

class ExecutionQualityGate:
    """
    Implements Execution Quality Gate logic per Section 13 of specification.
    """
    
    # State Constants
    STATE_ALLOWED = "ALLOWED"
    STATE_DEGRADED = "DEGRADED"
    STATE_BLOCKED = "BLOCKED"
    
    # Routing Constants
    ROUTING_LIMIT = "LIMIT"
    ROUTING_MARKET_FALLBACK = "MARKET_FALLBACK" # Note: Default is LIMIT, Fallback handled in execution engine
    
    # Degraded Actions
    ACTION_REDUCE = "REDUCE_SIZE_50PCT"
    
    # Thresholds (Frozen per Spec 13.1)
    LIMIT_SPREAD_RATIO_BLOCK = 3.0
    LIMIT_SPREAD_RATIO_DEGRADE = 1.80
    LIMIT_QUOTE_AGE_BLOCK = 1000 # ms
    LIMIT_API_ERROR_BLOCK = 0.05 # 5%
    
    # Liquid Hours (Default)
    LIQUID_START = time(7, 0)
    LIQUID_END = time(20, 0)
    
    # Per-Strategy Constraints (Spec 13.2)
    STRATEGY_CONSTRAINTS: Dict[str, Dict[str, object]] = {
        "SWING": {
            "max_spread_bp": 10.0,
            "max_slippage_bp": 15.0,
            "min_quote_fresh_ms": 2000,
            "session": None # No restriction
        },
        "ORB": {
            "max_spread_bp": 3.0,
            "max_slippage_bp": 5.0,
            "min_quote_fresh_ms": 500,
            "session": (time(7, 0), time(20, 0))
        },
        "NEWS": {
            # T1/T2 split handled in News Engine? Or here?
            # Spec 13.2 says "NEWS T1/T2: 3.0/2.5".
            # But SignalIntent doesn't specify Tier explicitly, just "NEWS".
            # We'll assume strict (T2) or use Signal constraints?
            # SignalIntent has `execution_constraints` field passed from Strategy!
            # So we should respect Signal's constraints if present.
            # But Spec 13.2 defines "Frozen" constraints.
            # Strategy Engine sets them in Signal.
            # We should validate against Spec or trust Signal?
            # Spec 13.1 says "expected_slippage_bp > signal.max_slippage_bp".
            # So we use Signal's values.
            # BUT 13.2 lists them.
            # I will use the Session constraints from here, and numerical limits from Signal (which should match).
            # But to be safe/compliant with "Per-Strategy Constraints" requirement, I will enforce Session here.
            "session": (time(7, 0), time(21, 0)) # Extended to 21:00
        },
        "SCALP": {
            "max_spread_bp": 2.0,
            "max_slippage_bp": 3.0,
            "min_quote_fresh_ms": 300,
            "session": (time(8, 0), time(17, 0))
        }
    }

    def evaluate(
        self,
        signal: SignalIntent,
        spread_live_bp: float,
        spread_median_100: float,
        quote_age_ms: int,
        order_lots: float,
        adv_daily_lots: float,
        api_error_rate_5min: float,
        now_utc: datetime,
    ) -> ExecutionDecision:
        
        # 1. Derived Metrics
        # spread_ratio = spread_live_bp / rolling_median_spread_100
        spread_ratio = spread_live_bp / spread_median_100 if spread_median_100 > 0 else 999.0
        
        # expected_slippage_bp = spread_live_bp / 2 + (order_lots / ADV_daily_lots) * 100
        # (Assuming order_lots is small relative to ADV, avoiding div/0)
        market_impact = (order_lots / adv_daily_lots * 100) if adv_daily_lots > 0 else 0.0
        expected_slippage_bp = (spread_live_bp / 2.0) + market_impact
        
        # 2. Strategy Constraints Lookup
        constraints = self.STRATEGY_CONSTRAINTS.get(signal.strategy_name, {})
        # Mypy sees `constraints` as Dict[str, object], so .get returns object.
        # We need to cast session to Optional[Tuple[time, time]]
        session_obj = constraints.get("session")
        session_window: Optional[tuple[time, time]] = None
        if isinstance(session_obj, tuple):
            session_window = session_obj
        
        # Signal constraints (from Strategy Engine) override defaults for numeric limits?
        # Spec 13.1 uses `signal.max_slippage_bp`.
        # So we trust signal for numeric limits.
        # But Session is enforced here.
        
        max_spread = signal.execution_constraints.max_spread_bp
        max_slippage = signal.execution_constraints.max_slippage_bp
        min_freshness = signal.execution_constraints.min_quote_fresh_ms
        
        # 3. BLOCKED Logic
        blocked_reason = None
        
        # Global Blockers
        if spread_ratio > self.LIMIT_SPREAD_RATIO_BLOCK:
            blocked_reason = f"SPREAD_RATIO_HIGH_{spread_ratio:.2f}"
        # NOTE: Removed LIMIT_QUOTE_AGE_BLOCK (1000) check here because Strategy Constraints (Spec 13.2) override it.
        # Swing allows 2000. Scalp allows 300.
        # We enforce `quote_age_ms > min_freshness` later.
        elif api_error_rate_5min >= self.LIMIT_API_ERROR_BLOCK:
            blocked_reason = f"API_ERROR_RATE_{api_error_rate_5min:.2f}"
        
        # Session Blockers (Per Strategy)
        if not blocked_reason and session_window:
            start, end = session_window
            current_time = now_utc.time()
            # Handle crossing midnight? Spec times are simple day ranges (07-20).
            # Simple comparison.
            if not (start <= current_time <= end):
                blocked_reason = f"OUTSIDE_SESSION_{current_time}"
        
        # Global Liquid Hours for others?
        # Spec 13.1: "session outside liquid hours (07:00-20:00 UTC)"
        # This applies if NO specific strategy session?
        # SWING has "None". So it's EXEMPT.
        # ORB, NEWS, SCALP have specific sessions.
        # So "Global Liquid Hours" effectively applies to generic strategies if added.
        # But SWING is explicitly exempt.
        # I'll rely on `session_window` being None for SWING.
        # If strategy is UNKNOWN, default to Liquid Hours?
        # I'll stick to defined strategies.
        
        # Strategy Specific Constraints (that cause BLOCK)
        # Spec 13.2: "max_spread_bp" -> Is this a BLOCK or DEGRADE?
        # Spec 13.1 doesn't list max_spread in BLOCKED conditions.
        # But Spec 13.2 says "Per-Strategy Constraints".
        # Usually max_spread is a hard gate.
        # Spec 10.1 (Scalp) "spread_live_bp <= 2.0". Gate.
        # News 9.2 "spread_live_bp <= 3.0". Gate.
        # So exceeding max_spread_bp IS A BLOCK.
        if not blocked_reason and spread_live_bp > max_spread:
            blocked_reason = f"MAX_SPREAD_EXCEEDED_{spread_live_bp}"
            
        # Quote Freshness per strategy
        # Spec 13.2 lists min_quote_fresh_ms.
        # Spec 13.1 lists "quote_age_ms > 1000" as Global Block.
        # But SWING allows 2000.
        # SCALP allows 300.
        # So we must check Strategy Limit.
        # If Strategy Limit < Global Limit (e.g. Scalp 300 < 1000), Strategy limit triggers first.
        # If Strategy Limit > Global Limit (e.g. Swing 2000 > 1000), Global limit triggers first?
        # Conflict: "EXECUTION_BLOCKED if ... quote_age_ms > 1000".
        # But Swing allows 2000.
        # Spec 13.2 is "Per-Strategy Constraints".
        # Spec 13.1 is "State Machine Logic".
        # Usually Strategy Specific overrides Global.
        # So for SWING, we ignore global 1000 and use 2000.
        # For SCALP, we use 300.
        # So I should use `min_freshness` from Signal (which reflects strategy) instead of hardcoded 1000?
        # Or `max(1000, min_freshness)`?
        # Spec 13.1 is explicit: "quote_age_ms > 1000".
        # But Spec 13.2 Table says Swing 2000.
        # I will implement: Block if quote_age > min_freshness.
        # And I will assume Signal's min_freshness is correct.
        # AND I will apply the 1000 limit ONLY if strategy doesn't override it?
        # Or does 1000 apply to everyone except Swing?
        # Actually, ORB(500), News(500), Scalp(300) are all < 1000.
        # Only Swing(2000) is > 1000.
        # I will respect the Strategy Constraint primarily.
        
        # Note: We rely on Strategy Constraints (min_freshness) which are correctly set per strategy.
        # We ignore global 1000ms check here because strategies define their own limits (some >1000, some <1000).
        if not blocked_reason and quote_age_ms > min_freshness:
            blocked_reason = f"STRATEGY_QUOTE_STALE_{quote_age_ms}ms"

        if blocked_reason:
            return ExecutionDecision(
                allowed_state=self.STATE_BLOCKED,
                blocked_reason=blocked_reason,
                spread_snapshot=spread_live_bp,
                spread_vs_median=spread_ratio,
                quote_freshness_ms=quote_age_ms,
                expected_slippage_bp=expected_slippage_bp,
                routing_mode=None,
                order_intent_id=f"sig_{hash(signal.timestamp)}",
                degraded_action=None,
                timestamp_utc=now_utc
            )

        # 4. DEGRADED Logic
        # EXECUTION_DEGRADED if NOT BLOCKED AND ANY of:
        #   spread_ratio > 1.80
        #   expected_slippage_bp > signal.max_slippage_bp
        #   degraded_action = "REDUCE_SIZE_50PCT" (Outcome)
        
        is_degraded = False
        degraded_reason = None
        
        if spread_ratio > self.LIMIT_SPREAD_RATIO_DEGRADE:
            is_degraded = True
            degraded_reason = "HIGH_SPREAD_RATIO"
        elif expected_slippage_bp > max_slippage:
            is_degraded = True
            degraded_reason = "HIGH_EXPECTED_SLIPPAGE"
            
        if is_degraded:
            return ExecutionDecision(
                allowed_state=self.STATE_DEGRADED,
                blocked_reason=None,
                spread_snapshot=spread_live_bp,
                spread_vs_median=spread_ratio,
                quote_freshness_ms=quote_age_ms,
                expected_slippage_bp=expected_slippage_bp,
                routing_mode=self.ROUTING_LIMIT,
                order_intent_id=f"sig_{hash(signal.timestamp)}",
                degraded_action=self.ACTION_REDUCE,
                timestamp_utc=now_utc
            )

        # 5. ALLOWED Logic
        return ExecutionDecision(
            allowed_state=self.STATE_ALLOWED,
            blocked_reason=None,
            spread_snapshot=spread_live_bp,
            spread_vs_median=spread_ratio,
            quote_freshness_ms=quote_age_ms,
            expected_slippage_bp=expected_slippage_bp,
            routing_mode=self.ROUTING_LIMIT,
            order_intent_id=f"sig_{hash(signal.timestamp)}",
            degraded_action=None,
            timestamp_utc=now_utc
        )
