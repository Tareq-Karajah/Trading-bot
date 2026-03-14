from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class Alert(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    alert_name: str
    severity: AlertSeverity
    channel: List[str]
    message: str
    value: float
    threshold: float
    timestamp_utc: datetime

    @field_validator("timestamp_utc")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

class AlertManager:
    """
    Manages alert thresholds and generation based on metric values.
    Thresholds are frozen per spec Section 15.
    """
    
    def check_and_fire(
        self,
        metric_name: str,
        value: float,
        context: Dict[str, Any],
    ) -> List[Alert]:
        alerts: List[Alert] = []
        now_utc = datetime.now(timezone.utc)
        
        # 1. Daily PnL Checks
        if metric_name == "daily_pnl_pct":
            # daily_loss_critical: daily_pnl < -3%
            if value < -3.0:
                alerts.append(Alert(
                    alert_name="daily_loss_critical",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty", "Telegram"],
                    message=f"CRITICAL: Daily PnL {value:.2f}% below -3.0% limit",
                    value=value,
                    threshold=-3.0,
                    timestamp_utc=now_utc
                ))
            # daily_loss_warning: daily_pnl < -2%
            elif value < -2.0:
                alerts.append(Alert(
                    alert_name="daily_loss_warning",
                    severity=AlertSeverity.WARNING,
                    channel=["Telegram"],
                    message=f"WARNING: Daily PnL {value:.2f}% below -2.0% warning",
                    value=value,
                    threshold=-2.0,
                    timestamp_utc=now_utc
                ))

        # 2. Weekly PnL Checks
        elif metric_name == "weekly_pnl_pct":
            # weekly_drawdown: weekly_pnl < -5%
            if value < -5.0:
                alerts.append(Alert(
                    alert_name="weekly_drawdown",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty", "Email"],
                    message=f"CRITICAL: Weekly PnL {value:.2f}% below -5.0% limit",
                    value=value,
                    threshold=-5.0,
                    timestamp_utc=now_utc
                ))

        # 3. Drawdown Checks
        elif metric_name == "active_drawdown_pct":
            # peak_drawdown: drawdown > 15%
            if value > 15.0:
                alerts.append(Alert(
                    alert_name="peak_drawdown",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty"],
                    message=f"CRITICAL: Drawdown {value:.2f}% exceeds 15.0% limit",
                    value=value,
                    threshold=15.0,
                    timestamp_utc=now_utc
                ))

        # 4. Error Rate Checks
        elif metric_name == "api_error_rate":
            # api_error_rate: error_rate > 5%
            if value > 5.0:
                alerts.append(Alert(
                    alert_name="api_error_rate",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty"],
                    message=f"CRITICAL: API Error Rate {value:.2f}% exceeds 5.0% limit",
                    value=value,
                    threshold=5.0,
                    timestamp_utc=now_utc
                ))

        # 5. Strategy Viability (Rolling ER)
        elif metric_name == "rolling_expected_return":
            # strategy_viability: rolling_ER negative
            if value < 0.0:
                alerts.append(Alert(
                    alert_name="strategy_viability",
                    severity=AlertSeverity.WARNING,
                    channel=["Telegram"],
                    message=f"WARNING: Strategy Rolling ER {value:.4f} is negative",
                    value=value,
                    threshold=0.0,
                    timestamp_utc=now_utc
                ))

        # 6. Strategy Suspended
        elif metric_name == "strategy_suspended":
            # strategy_suspended: value > 0 (True)
            if value > 0.0:
                alerts.append(Alert(
                    alert_name="strategy_suspended",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty"],
                    message=f"CRITICAL: Strategy {context.get('strategy', 'UNKNOWN')} suspended",
                    value=value,
                    threshold=0.0,
                    timestamp_utc=now_utc
                ))

        # 7. Dead Man Switch
        elif metric_name == "seconds_since_heartbeat":
            # dead_man_switch: no heartbeat > 60s
            if value > 60.0:
                alerts.append(Alert(
                    alert_name="dead_man_switch",
                    severity=AlertSeverity.CRITICAL,
                    channel=["PagerDuty"],
                    message=f"CRITICAL: No heartbeat for {value:.1f}s (>60s)",
                    value=value,
                    threshold=60.0,
                    timestamp_utc=now_utc
                ))

        # 8. Slippage
        elif metric_name == "slippage_ratio":
            # slippage_alert: slippage > 3x expected
            # Value here represents ratio or actual? Spec says "slippage > 3x expected"
            # If metric_name passed is "slippage_ratio" (Actual / Expected), then > 3.0
            if value > 3.0:
                alerts.append(Alert(
                    alert_name="slippage_alert",
                    severity=AlertSeverity.WARNING,
                    channel=["Telegram"],
                    message=f"WARNING: Slippage {value:.2f}x expected (>3.0x)",
                    value=value,
                    threshold=3.0,
                    timestamp_utc=now_utc
                ))

        return alerts
