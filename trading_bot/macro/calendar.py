from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Enums from Spec ---

class Tier(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"

class CalendarSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EventRiskState(str, Enum):
    NO_EVENT_RISK = "NO_EVENT_RISK"
    PRE_EVENT_WINDOW = "PRE_EVENT_WINDOW"
    LIVE_EVENT_WINDOW = "LIVE_EVENT_WINDOW"
    POST_EVENT_WINDOW = "POST_EVENT_WINDOW"
    WHITELISTED_NEWS_BREAKOUT_WINDOW = "WHITELISTED_NEWS_BREAKOUT_WINDOW"

# --- Models ---

class CalendarEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    event_id: str
    event_name: str
    tier: Tier
    severity: CalendarSeverity
    scheduled_time_utc: datetime
    pre_window_seconds: int
    post_window_seconds: int
    is_whitelisted_for_news_breakout: bool

    @field_validator("scheduled_time_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- Service ---

class MarketCalendarService:
    """
    Manages calendar events and determines event risk state.
    """
    
    POST_EVENT_COOLDOWN_SECONDS = 300
    
    def __init__(self) -> None:
        self._events: List[CalendarEvent] = []

    def load_events(self, events: List[CalendarEvent]) -> None:
        """
        Load a list of calendar events. Overwrites existing events.
        """
        self._events = events

    def get_active_event_risk(self, now_utc: datetime) -> EventRiskState:
        """
        Determine the highest priority event risk state for the given time.
        """
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        highest_risk_state = EventRiskState.NO_EVENT_RISK
        
        # Priority order for states (higher index = higher priority)
        # NO_EVENT_RISK < POST < PRE < LIVE < WHITELISTED
        # However, Spec says: "Overlapping events: use highest severity".
        # But here we are returning State, not Severity.
        # Let's infer priority: 
        # WHITELISTED_NEWS_BREAKOUT_WINDOW (Trading Allowed) vs LIVE_EVENT_WINDOW (Blocked)?
        # Spec 9.1: T1/T2/T3 are whitelisted.
        # Spec 6.2 Priority Table: 
        # 1. SHOCK or approved news active (Whitelisted)
        # 2. Blackout (HIGH/CRITICAL)
        # So Whitelisted > Live/Blackout.
        
        # Let's determine state for each event and pick the "most significant".
        # We need a hierarchy.
        # 1. WHITELISTED_NEWS_BREAKOUT_WINDOW (Specific active trade window)
        # 2. LIVE_EVENT_WINDOW (General danger zone)
        # 3. PRE_EVENT_WINDOW (Preparation)
        # 4. POST_EVENT_WINDOW (Cooldown)
        # 5. NO_EVENT_RISK
        
        state_priority = {
            EventRiskState.NO_EVENT_RISK: 0,
            EventRiskState.POST_EVENT_WINDOW: 1,
            EventRiskState.PRE_EVENT_WINDOW: 2,
            EventRiskState.LIVE_EVENT_WINDOW: 3,
            EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW: 4
        }
        
        current_best_priority = -1
        
        for event in self._events:
            state = self._classify_event_state(event, now_utc)
            priority = state_priority[state]
            
            if priority > current_best_priority:
                current_best_priority = priority
                highest_risk_state = state
        
        return highest_risk_state

    def get_current_severity(self, now_utc: datetime) -> CalendarSeverity:
        """
        Get the highest severity among all active events (Pre/Live/Post).
        """
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        max_severity = CalendarSeverity.NONE
        
        severity_rank = {
            CalendarSeverity.NONE: 0,
            CalendarSeverity.LOW: 1,
            CalendarSeverity.HIGH: 2,
            CalendarSeverity.CRITICAL: 3
        }
        
        for event in self._events:
            if self._is_event_active(event, now_utc):
                if severity_rank[event.severity] > severity_rank[max_severity]:
                    max_severity = event.severity
                    
        return max_severity

    def get_whitelisted_event(self, now_utc: datetime) -> Optional[CalendarEvent]:
        """
        Return the event if we are currently in a WHITELISTED_NEWS_BREAKOUT_WINDOW.
        If multiple, return the one with highest severity.
        """
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
            
        candidate: Optional[CalendarEvent] = None
        severity_rank = {
            CalendarSeverity.NONE: 0,
            CalendarSeverity.LOW: 1,
            CalendarSeverity.HIGH: 2,
            CalendarSeverity.CRITICAL: 3
        }

        for event in self._events:
            state = self._classify_event_state(event, now_utc)
            if state == EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW:
                if candidate is None:
                    candidate = event
                elif severity_rank[event.severity] > severity_rank[candidate.severity]:
                    candidate = event
        
        return candidate

    def _is_event_active(self, event: CalendarEvent, now_utc: datetime) -> bool:
        """
        Check if time is within Pre, Live, or Post window of the event.
        """
        start = event.scheduled_time_utc - timedelta(seconds=event.pre_window_seconds)
        end = event.scheduled_time_utc + timedelta(seconds=event.post_window_seconds) + timedelta(seconds=self.POST_EVENT_COOLDOWN_SECONDS)
        return start <= now_utc <= end

    def _classify_event_state(self, event: CalendarEvent, now_utc: datetime) -> EventRiskState:
        # Window logic:
        # - PRE_EVENT_WINDOW: now is within pre_window_seconds before scheduled_time_utc
        # - LIVE_EVENT_WINDOW: now is within post_window_seconds after scheduled_time_utc
        # - WHITELISTED_NEWS_BREAKOUT_WINDOW: event is whitelisted AND within LIVE_EVENT_WINDOW
        # - POST_EVENT_WINDOW: now is 0 to 300 seconds after LIVE_EVENT_WINDOW ends
        
        sched = event.scheduled_time_utc
        pre_start = sched - timedelta(seconds=event.pre_window_seconds)
        live_end = sched + timedelta(seconds=event.post_window_seconds)
        post_end = live_end + timedelta(seconds=self.POST_EVENT_COOLDOWN_SECONDS)
        
        # Strict inequalities or inclusive?
        # Spec says "within". Typically inclusive boundaries or half-open.
        # "now is within pre_window_seconds before" -> [sched - pre, sched)
        # "now is within post_window_seconds after" -> [sched, sched + post]
        
        if pre_start <= now_utc < sched:
            return EventRiskState.PRE_EVENT_WINDOW
        
        if sched <= now_utc <= live_end:
            if event.is_whitelisted_for_news_breakout:
                return EventRiskState.WHITELISTED_NEWS_BREAKOUT_WINDOW
            else:
                return EventRiskState.LIVE_EVENT_WINDOW
                
        if live_end < now_utc <= post_end:
            return EventRiskState.POST_EVENT_WINDOW
            
        return EventRiskState.NO_EVENT_RISK
