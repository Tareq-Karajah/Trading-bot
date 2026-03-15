from datetime import datetime, date
from typing import List, Optional, Dict
from pydantic import BaseModel, ConfigDict
import numpy as np

class TradeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    strategy: str
    r_multiple: float
    was_win: bool
    spread_cost_bp: float
    commission_pct: float
    slippage_pct: float
    holding_hours: float

class WindowResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    start_date: datetime
    end_date: datetime
    is_sharpe: float
    oos_sharpe: float
    is_maxdd: float
    oos_maxdd: float
    is_trades: int
    oos_trades: int

class WalkForwardResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    windows: List[WindowResult]
    aggregate_sharpe: float
    aggregate_maxdd: float
    oos_is_ratio: float   # mean OOS Sharpe / mean IS Sharpe
    passes_gate: bool     # oos_is_ratio >= 0.70

class WalkForwardBacktest:
    """
    Implements Walk-Forward Analysis protocol.
    Section 14.1: IS 36m, OOS 6m, Step 3m.
    """
    
    def _calculate_metrics(self, trades: List[TradeRecord]) -> dict[str, float]:
        if not trades:
            return {"sharpe": 0.0, "maxdd": 0.0, "trades": 0.0}
            
        returns: List[float] = []
        for t in trades:
            # Net Return Calculation
            cost_bp = t.spread_cost_bp + (t.commission_pct * 10000) + (t.slippage_pct * 10000)
            
            # Convert cost_bp to R. Assuming 1R = 100bp (1%).
            cost_r = cost_bp / 100.0
            
            net_r = t.r_multiple - cost_r
            returns.append(net_r)
            
        # Create daily series
        daily_pnl: Dict[date, float] = {}
        for t, r in zip(trades, returns):
            # Aggregation logic: Sum returns for the same day
            date_key = t.timestamp.date()
            daily_pnl[date_key] = daily_pnl.get(date_key, 0.0) + r
            
        sorted_dates = sorted(daily_pnl.keys())
        
        day_returns = list(daily_pnl.values())
        
        # If only 1 day of returns, std is undefined (or 0).
        # Sharpe is 0.
        if len(day_returns) < 2:
            mean_ret = np.mean(day_returns)
            std_ret = 0.0
        else:
            mean_ret = np.mean(day_returns)
            std_ret = float(np.std(day_returns, ddof=1))
        
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        
        # MaxDD on Equity Curve (cumulative sum of R)
        cum_r = np.cumsum(day_returns)
        peak = -999999.0
        max_dd = 0.0
        for x in cum_r:
            if x > peak:
                peak = float(x)
            dd = peak - x
            if dd > max_dd:
                max_dd = float(dd)
                
        return {"sharpe": float(sharpe), "maxdd": float(max_dd), "trades": float(len(trades))}

    def run(
        self,
        trade_returns: List[TradeRecord],
        is_months: int = 36,
        oos_months: int = 6,
        step_months: int = 3,
    ) -> WalkForwardResult:
        if not trade_returns:
             return WalkForwardResult(
                 windows=[], aggregate_sharpe=0.0, aggregate_maxdd=0.0, oos_is_ratio=0.0, passes_gate=False
             )

        # Sort trades
        sorted_trades = sorted(trade_returns, key=lambda x: x.timestamp)
        start_date = sorted_trades[0].timestamp
        end_date = sorted_trades[-1].timestamp
        
        # Generate Windows
        windows: List[WindowResult] = []
        
        # We need to iterate by months.
        # Simple month addition helper
        def add_months(dt: datetime, months: int) -> datetime:
            new_month = dt.month + months
            year_add = (new_month - 1) // 12
            new_month = (new_month - 1) % 12 + 1
            # Handle day overflow (e.g. Jan 31 + 1 month -> Feb 28/29)
            # Simplification: set to day 1? No, we need precise windows.
            # But trades are sparse.
            # Let's keep day, but clamp to max day of month.
            import calendar
            max_day = calendar.monthrange(dt.year + year_add, new_month)[1]
            return dt.replace(year=dt.year + year_add, month=new_month, day=min(dt.day, max_day))

        current_start = start_date
        
        while True:
            is_end = add_months(current_start, is_months)
            oos_end = add_months(is_end, oos_months)
            
            # Allow partial OOS window if data ends early
            if is_end > end_date:
                break
            
            # Filter Trades
            is_trades = [t for t in sorted_trades if current_start <= t.timestamp < is_end]
            oos_trades = [t for t in sorted_trades if is_end <= t.timestamp < oos_end]
            
            if not is_trades: # Skip empty windows? Or record 0?
                 # If no IS trades, we can't optimize/validate. Skip.
                 current_start = add_months(current_start, step_months)
                 continue

            is_metrics = self._calculate_metrics(is_trades)
            oos_metrics = self._calculate_metrics(oos_trades)
            
            # Skip windows where IS sharpe cannot be calculated or IS trades are empty
            # If is_metrics has no trades, _calculate_metrics returns {'sharpe': 0, 'maxdd': 0, 'trades': 0}
            # The previous implementation assumed 'trades' key exists, but _calculate_metrics helper didn't add it in the "if not trades" case.
            
            windows.append(WindowResult(
                start_date=current_start,
                end_date=oos_end,
                is_sharpe=is_metrics["sharpe"],
                oos_sharpe=oos_metrics["sharpe"],
                is_maxdd=is_metrics["maxdd"],
                oos_maxdd=oos_metrics["maxdd"],
                is_trades=int(is_metrics["trades"]),
                oos_trades=int(oos_metrics["trades"])
            ))
            
            current_start = add_months(current_start, step_months)
            
        # Aggregate Metrics
        if not windows:
             return WalkForwardResult(
                 windows=[], aggregate_sharpe=0.0, aggregate_maxdd=0.0, oos_is_ratio=0.0, passes_gate=False
             )

        mean_is_sharpe = float(np.mean([w.is_sharpe for w in windows]))
        mean_oos_sharpe = float(np.mean([w.oos_sharpe for w in windows]))
        
        # Ratio
        # If mean_is_sharpe is 0 or negative, ratio is tricky.
        # If negative IS sharpe, optimization failed.
        # If mean_is_sharpe <= 0, we can't really trust the ratio.
        # Let's assume ratio is 0 if IS <= 0, unless OOS is also negative?
        # Standard WFA: if IS <= 0, strategy failed IS.
        
        if mean_is_sharpe <= 0:
            ratio = 0.0
        else:
            ratio = mean_oos_sharpe / mean_is_sharpe
            
        # Passes Gate: ratio >= 0.70
        passes = ratio >= 0.70
        
        return WalkForwardResult(
            windows=windows,
            aggregate_sharpe=float(mean_oos_sharpe),
            aggregate_maxdd=float(np.mean([w.oos_maxdd for w in windows])),
            oos_is_ratio=float(ratio),
            passes_gate=passes
        )
