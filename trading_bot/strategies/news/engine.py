from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading_bot.core.models import OHLCV
from trading_bot.dispatcher.engine import DispatcherPermissions
from trading_bot.macro.calendar import CalendarEvent, Tier
from trading_bot.regime.engine import RegimeState
from trading_bot.macro.engine import MacroState

# --- SignalIntent Contract (Strictly from Spec) ---

class TPPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    tp1_distance: Optional[float]
    tp1_size_pct: float
    tp2_distance: Optional[float]
    trail_atr_mult: float

class TimeoutPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_bars: Optional[int]
    mandatory_exit_utc: Optional[str]

class ExecutionConstraints(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_spread_bp: float
    max_slippage_bp: float
    min_quote_fresh_ms: int

class SignalIntent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    strategy_name: str
    direction: int # +1, -1
    score: int # 0-5
    entry_type: str # "MARKET", "LIMIT", "STOP_LIMIT"
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

# --- News Decision Model ---

class NewsDecision(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    signal_intent: Optional[SignalIntent]
    pre_news_high: float
    pre_news_low: float
    buf_news: float
    gates_passed: bool
    expiry_ok: bool
    daily_trade_count: int
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- News Breakout Engine ---

class NewsBreakoutEngine:
    """
    Implements News Breakout Strategy.
    Spec Section 9.
    """
    
    # Constants
    MAX_DAILY_TRADES = 2
    BUF_ATR_MULT = 0.20
    MIN_RR = 2.0
    CHASE_LIMIT_MULT = 1.5
    
    # Gate Thresholds
    GATE_MAX_SPREAD_BP = 3.0
    GATE_MAX_FRESHNESS_MS = 500
    GATE_MAX_SLIPPAGE_BP = 5.0
    GATE_SESSION_START = 7 # 07:00 UTC
    GATE_SESSION_END = 21 # 21:00 UTC (inclusive/exclusive? "in [07:00, 21:00]" usually inclusive)
    
    # Execution Constraints
    EXEC_MAX_SPREAD_T1 = 3.0
    EXEC_MAX_SPREAD_T2 = 2.5
    EXEC_MAX_SPREAD_T3 = 3.0 # Fallback
    EXEC_MAX_SLIPPAGE = 5.0
    EXEC_MIN_FRESHNESS = 500

    def __init__(self) -> None:
        self._daily_trade_count: int = 0
        self._last_reset_date: Optional[datetime] = None

    def _check_daily_reset(self, now_utc: datetime) -> None:
        """
        Reset daily trade count at 00:00 UTC.
        """
        if self._last_reset_date is None or now_utc.date() > self._last_reset_date.date():
            self._daily_trade_count = 0
            self._last_reset_date = now_utc

    def evaluate(
        self,
        candles_m5: List[OHLCV],      # last 5 pre-news M5 candles
        current_candle: OHLCV,         # candle after event time
        atr_14_m5: float,
        spread_live_bp: float,
        spread_median_100: float,
        quote_age_ms: int,
        order_lots: float,
        adv_daily_lots: float,
        event: CalendarEvent,
        event_time_utc: datetime,
        permissions: DispatcherPermissions,
    ) -> NewsDecision:
        
        now_utc = current_candle.timestamp
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        if event_time_utc.tzinfo is None:
            event_time_utc = event_time_utc.replace(tzinfo=timezone.utc)

        self._check_daily_reset(now_utc)
        
        # 1. Calculate Pre-News Range
        # Requires 5 candles
        if not candles_m5 or len(candles_m5) < 5:
             # Fallback or error? Return no signal.
             return NewsDecision(
                 signal_intent=None, pre_news_high=0.0, pre_news_low=0.0, buf_news=0.0,
                 gates_passed=False, expiry_ok=False, daily_trade_count=self._daily_trade_count,
                 timestamp_utc=now_utc
             )
             
        # Use last 5 candles passed in
        range_candles = candles_m5[-5:]
        pre_news_high = max(c.high for c in range_candles)
        pre_news_low = min(c.low for c in range_candles)
        buf_news = self.BUF_ATR_MULT * atr_14_m5
        
        # 2. Check Gates
        # Gate 1: Spread
        spread_limit = min(self.GATE_MAX_SPREAD_BP, 2.0 * spread_median_100)
        spread_ok = (spread_live_bp <= spread_limit)
        
        # Gate 2: Freshness
        freshness_ok = (quote_age_ms < self.GATE_MAX_FRESHNESS_MS)
        
        # Gate 3: Slippage
        # expected_slip = spread_live_bp / 2 + (order_lots / ADV_daily_lots) * 100
        # Guard adv_daily_lots zero
        adv_lots = adv_daily_lots if adv_daily_lots > 0 else 1.0 # Prevent div/0
        expected_slip = (spread_live_bp / 2.0) + (order_lots / adv_lots) * 100.0
        slippage_ok = (expected_slip < self.GATE_MAX_SLIPPAGE_BP)
        
        # Gate 4: Session
        # [07:00, 21:00] UTC
        hour = now_utc.hour
        session_ok = (7 <= hour <= 21)
        
        gates_passed = spread_ok and freshness_ok and slippage_ok and session_ok
        
        # 3. Expiry Check
        expiry_ok = False
        time_diff_sec = (now_utc - event_time_utc).total_seconds()
        
        # T1: 3 min (180s), T2: 2 min (120s), T3: 1 min (60s)
        # "cancel if no trigger within X minutes" -> means if current time > limit, expired.
        # So we must be <= limit.
        limit_sec = 0
        if event.tier == Tier.T1:
            limit_sec = 180
        elif event.tier == Tier.T2:
            limit_sec = 120
        else: # T3
            limit_sec = 60
            
        # Assuming we are strictly AFTER event time (post-release)
        # If time_diff_sec < 0 (pre-event), we shouldn't be here or it's not expired but maybe too early?
        # Prompt says "Entry Window ... X min post-release".
        # If we are evaluating `current_candle` which is "candle after event time", then time_diff >= 0.
        if 0 <= time_diff_sec <= limit_sec:
            expiry_ok = True
            
        # 4. Entry Logic
        signal_intent = None
        
        if gates_passed and expiry_ok and self._daily_trade_count < self.MAX_DAILY_TRADES:
            direction = 0
            entry_trigger = 0.0
            
            # Long
            long_trigger = pre_news_high + buf_news
            if current_candle.close > long_trigger:
                if permissions.direction_constraint != -1: # -1 suppresses Long
                    direction = 1
                    entry_trigger = long_trigger
                    
            # Short
            if direction == 0: # Only check if not already Long
                short_trigger = pre_news_low - buf_news
                if current_candle.close < short_trigger:
                    if permissions.direction_constraint != 1: # 1 suppresses Short
                        direction = -1
                        entry_trigger = short_trigger
                        
            if direction != 0:
                # 5. Score & Intent
                score = 0
                score += 2 # Whitelisted (Mandatory, implicit since we are running News engine with event)
                score += 1 # Gates passed
                score += 1 # Expiry ok
                
                # Calculate SL/TP
                sl_distance = 0.0
                if direction == 1:
                    # SL = Opposite side (Low) - buf_news?
                    # Spec: "sl_distance: distance from entry to opposite side of pre-news range + buf_news"
                    # Entry = High + buf
                    # Opposite = Low
                    # Wait, "opposite side of pre-news range + buf_news"
                    # Does it mean (Low + buf) or (Low - buf)?
                    # Usually SL for Long is below Low. So (Low - buf).
                    # "distance from entry to opposite side... + buf_news" could mean:
                    # Dist = Entry - (Opposite_Level)
                    # Opposite_Level for Long = Low - buf_news.
                    # Entry = High + buf_news.
                    # Dist = (High + buf) - (Low - buf) = High - Low + 2*buf.
                    # Let's verify text: "opposite side of pre-news range + buf_news" might refer to the *calculation* of the level?
                    # Or "distance ... is (Range + 2*Buf)"?
                    # "opposite side ... + buf_news" is ambiguous syntax.
                    # Let's look at Short: Entry = Low - buf. Opposite = High. SL Level = High + buf.
                    # So SL is "High + buf_news".
                    # For Long, SL is "Low - buf_news".
                    # So "opposite side ... +/- buf_news".
                    # Let's compute SL Level.
                    stop_level = pre_news_low - buf_news if direction == 1 else pre_news_high + buf_news
                    sl_distance = abs(entry_trigger - stop_level)
                else:
                    stop_level = pre_news_high + buf_news
                    sl_distance = abs(entry_trigger - stop_level)
                    
                # R:R Check
                # tp_target = 2.0 * sl_distance
                # Check R:R >= 2.0.
                # Since we set TP = 2.0 * SL, R:R is exactly 2.0.
                # So "if R:R >= 2.0 confirmed" is always true if we define it so.
                # BUT we need to check if `tp_target` is viable?
                # No, just "Signal suppressed if tp_target < 2.0 * sl_distance".
                # Since we construct TP as 2.0 * SL, this check passes by definition unless SL=0.
                if sl_distance > 0:
                    score += 1
                
                # Chase Limit
                # "never enter if entry price > 1.5 * SL_distance from trigger level"
                # current_candle.close is the "entry price" (Market order executed now).
                # Wait, entry_type is "MARKET".
                # But we trigger if Close > Trigger.
                # So actual fill might be at Close (or worse).
                # Deviation = abs(Close - Trigger).
                # If Deviation > 1.5 * SL_Distance -> Suppress.
                deviation = abs(current_candle.close - entry_trigger)
                chase_ok = (deviation <= self.CHASE_LIMIT_MULT * sl_distance)
                
                if score >= 4 and chase_ok and sl_distance > 0:
                    # Construct Intent
                    # Max spread per Tier
                    max_spread = self.EXEC_MAX_SPREAD_T3
                    if event.tier == Tier.T1:
                        max_spread = self.EXEC_MAX_SPREAD_T1
                    elif event.tier == Tier.T2:
                        max_spread = self.EXEC_MAX_SPREAD_T2
                        
                    # TP Plan
                    # tp1_distance = 2.0 * sl_distance (Target)
                    # tp1_size_pct = 1.00
                    
                    signal_intent = SignalIntent(
                        strategy_name="NEWS",
                        direction=direction,
                        score=score,
                        entry_type="MARKET",
                        entry_trigger=entry_trigger, # Trigger level, not current price
                        sl_distance=float(sl_distance),
                        tp_plan=TPPlan(
                            tp1_distance=float(2.0 * sl_distance),
                            tp1_size_pct=1.00,
                            tp2_distance=None,
                            trail_atr_mult=0.0
                        ),
                        timeout_plan=TimeoutPlan(
                            # "max_bars: per tier expiry (T1=3min, T2=2min, T3=1min in M5 bars)"
                            # 3 min is < 1 bar (5 min). So we set 1 bar as minimum.
                            max_bars=1, 
                            mandatory_exit_utc=None
                        ),
                        regime_context=permissions.regime,
                        macro_context=permissions.macro_bias,
                        execution_constraints=ExecutionConstraints(
                            max_spread_bp=max_spread,
                            max_slippage_bp=self.EXEC_MAX_SLIPPAGE,
                            min_quote_fresh_ms=self.EXEC_MIN_FRESHNESS
                        )
                    )
                    
                    self._daily_trade_count += 1
                    self._publish_signal(signal_intent)

        return NewsDecision(
            signal_intent=signal_intent,
            pre_news_high=float(pre_news_high),
            pre_news_low=float(pre_news_low),
            buf_news=float(buf_news),
            gates_passed=gates_passed,
            expiry_ok=expiry_ok,
            daily_trade_count=self._daily_trade_count,
            timestamp_utc=now_utc
        )

    def _publish_signal(self, intent: SignalIntent) -> None:
        """
        Publish SIGNAL_INTENT to Redis. Stub.
        """
        pass
