from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict

from trading_bot.research.backtest import WalkForwardResult
from trading_bot.research.monte_carlo import MonteCarloResult

class DeploymentGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    approved: bool
    failed_gates: List[str]
    gate_details: Dict[str, Any]

class DeploymentGate:
    """
    Final decision gate for strategy deployment.
    Section 14.3.
    """
    
    def evaluate(
        self,
        wf_result: WalkForwardResult,
        mc_result: MonteCarloResult,
        max_drawdown_wf: float,
        profit_factor_wf: float,
        paper_trading_days: int,
        paper_sharpe: float,
    ) -> DeploymentGateResult:
        
        failed_gates: List[str] = []
        details: Dict[str, Any] = {
            "G1_oos_is_ratio": wf_result.oos_is_ratio,
            "G2_ruin_prob": mc_result.ruin_probability,
            "G3_max_dd": max_drawdown_wf,
            "G4_profit_factor": profit_factor_wf,
            "G5_paper_days": paper_trading_days,
            "G6_paper_sharpe": paper_sharpe
        }
        
        # G1: wf_result.oos_is_ratio >= 0.70
        if wf_result.oos_is_ratio < 0.70:
            failed_gates.append("G1_OOS_IS_RATIO")
            
        # G2: mc_result.ruin_probability < 0.02
        # Boundary: 0.02 fails (strictly less than)
        if mc_result.ruin_probability >= 0.02:
            failed_gates.append("G2_RUIN_PROBABILITY")
            
        # G3: max_drawdown_wf < 0.15
        # Boundary: 0.15 fails (strictly less than)
        if max_drawdown_wf >= 0.15:
            failed_gates.append("G3_MAX_DRAWDOWN")
            
        # G4: profit_factor_wf > 1.50
        # Boundary: 1.50 fails (strictly greater than)
        if profit_factor_wf <= 1.50:
            failed_gates.append("G4_PROFIT_FACTOR")
            
        # G5: paper_trading_days >= 30
        if paper_trading_days < 30:
            failed_gates.append("G5_PAPER_TRADING_DAYS")
            
        # G6: paper_sharpe > 1.0
        if paper_sharpe <= 1.0:
            failed_gates.append("G6_PAPER_SHARPE")
            
        approved = len(failed_gates) == 0
        
        return DeploymentGateResult(
            approved=approved,
            failed_gates=failed_gates,
            gate_details=details
        )
