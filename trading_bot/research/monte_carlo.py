from typing import List, Optional, Any, cast
from pydantic import BaseModel, ConfigDict
import numpy as np
import random

class MonteCarloResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    ruin_probability: float
    median_return: float
    percentile_5: float
    percentile_95: float
    max_drawdown_median: float
    passes_gate: bool

class MonteCarloSimulator:
    """
    Monte Carlo Simulator for risk of ruin and equity curve analysis.
    Section 14.2.
    """
    
    def run(
        self,
        trade_r_multiples: List[float],
        risk_fraction: float,
        n_simulations: int = 10_000,
        n_trades: int = 252,
        ruin_threshold: float = 0.50,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        if not trade_r_multiples:
             return MonteCarloResult(
                 ruin_probability=0.0, median_return=0.0, percentile_5=0.0, percentile_95=0.0, max_drawdown_median=0.0, passes_gate=False
             )

        rng = np.random.default_rng(seed)
        
        ruin_count = 0
        final_returns = []
        max_drawdowns: List[float] = []
        
        # We need to simulate equity curves.
        # Equity starts at 1.0.
        # Ruin if equity < 0.5 (if threshold 0.5).
        
        trades_arr = np.array(trade_r_multiples)
        
        for _ in range(n_simulations):
            # Resample
            sim_trades = rng.choice(trades_arr, size=n_trades, replace=True)
            
            # Compute Equity Curve
            # Assuming Fixed Fractional: Equity_t = Equity_{t-1} * (1 + R * risk)
            # Or additive: Equity_t = Equity_{t-1} + (R * risk)
            # "compute equity curve ... if equity < 50% of start ... count as ruin"
            # Ruin is typically associated with geometric compounding (multiplicative).
            # "risk_fraction" implies fractional sizing.
            # We will use multiplicative.
            
            returns = sim_trades * risk_fraction
            equity_curve = np.cumprod(1 + returns)
            
            # Check Ruin (Start is 1.0. Prepend 1.0 to check from start?)
            # Usually ruin can happen on first trade.
            # equity_curve contains values after trade 1, 2, ...
            # Check min.
            min_equity: float = float(np.min(equity_curve))
            if min_equity < (1.0 - ruin_threshold): # ruin_threshold 0.5 means < 0.5
                ruin_count += 1
                
            final_returns.append(equity_curve[-1] - 1.0) # Total return
            
            # Max Drawdown for this sim
            # DD from peak
            peak = 1.0
            max_dd = 0.0
            # Need to iterate or vectorized accumulation
            # Vectorized DD:
            # peaks = np.maximum.accumulate(np.insert(equity_curve, 0, 1.0))
            # dds = (peaks - np.insert(equity_curve, 0, 1.0)) / peaks
            # max_dd = np.max(dds)
            # Slightly faster:
            running_max = np.maximum.accumulate(equity_curve)
            # Need to handle if running_max starts > 1.0? 
            # If first trade is loss, peak is 1.0.
            # We should prepend 1.0
            curve_with_start = np.insert(equity_curve, 0, 1.0)
            running_max = np.maximum.accumulate(curve_with_start)
            drawdowns = (running_max - curve_with_start) / running_max
            max_drawdowns.append(float(np.max(drawdowns)))
            
        ruin_prob = ruin_count / n_simulations
        
        final_returns_arr = np.array(final_returns, dtype=float)
        max_drawdowns_arr = np.array(max_drawdowns, dtype=float)
        
        # Numpy bug workaround: Use list for q to avoid scalar _NoValueType error in _amin
        # Mypy issue: np.percentile returns Any or floating, we need to cast to array to index
        p5 = cast(np.ndarray[Any, Any], np.percentile(final_returns_arr, [5]))
        p95 = cast(np.ndarray[Any, Any], np.percentile(final_returns_arr, [95]))
        
        return MonteCarloResult(
            ruin_probability=float(ruin_prob),
            median_return=float(np.median(final_returns_arr)),
            percentile_5=float(p5[0]),
            percentile_95=float(p95[0]),
            max_drawdown_median=float(np.median(max_drawdowns_arr)),
            passes_gate=(ruin_prob < 0.02)
        )
