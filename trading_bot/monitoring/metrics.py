from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Dict, Optional

class MetricsCollector:
    """
    Centralized metrics collection using Prometheus.
    Exposes application health and performance indicators.
    """
    
    # Business Metrics
    active_positions = Gauge('trading_bot_active_positions', 'Number of currently open positions')
    daily_pnl = Gauge('trading_bot_daily_pnl', 'Realized daily P&L')
    signal_count = Counter('trading_bot_signal_count', 'Total trading signals generated', ['symbol', 'side'])
    
    # System Metrics
    order_latency_ms = Histogram('trading_bot_order_latency_ms', 'Order execution latency in milliseconds')
    api_errors = Counter('trading_bot_api_errors', 'Exchange API error count', ['exchange', 'error_type'])
    
    def __init__(self, registry=None):
        self.registry = registry

    @staticmethod
    def start_http_server(port: int = 8000) -> None:
        """
        Start the Prometheus metrics server.
        """
        from prometheus_client import start_http_server
        start_http_server(port)
