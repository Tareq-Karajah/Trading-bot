import pytest
from unittest.mock import MagicMock
from trading_bot.research.deployment_gate import DeploymentGate, DeploymentGateResult
from trading_bot.research.backtest import WalkForwardResult
from trading_bot.research.monte_carlo import MonteCarloResult

@pytest.fixture
def gate():
    return DeploymentGate()

@pytest.fixture
def passing_wf():
    return WalkForwardResult(
        windows=[], aggregate_sharpe=1.5, aggregate_maxdd=0.10, oos_is_ratio=0.8, passes_gate=True
    )

@pytest.fixture
def passing_mc():
    return MonteCarloResult(
        ruin_probability=0.01, median_return=0.1, percentile_5=0.0, percentile_95=0.2, max_drawdown_median=0.1, passes_gate=True
    )

def test_all_pass(gate, passing_wf, passing_mc):
    res = gate.evaluate(
        wf_result=passing_wf,
        mc_result=passing_mc,
        max_drawdown_wf=0.14,
        profit_factor_wf=1.51,
        paper_trading_days=30,
        paper_sharpe=1.1
    )
    assert res.approved is True
    assert len(res.failed_gates) == 0

def test_g1_fail(gate, passing_wf, passing_mc):
    # oos_is_ratio < 0.70
    wf = passing_wf.model_copy(update={"oos_is_ratio": 0.69})
    res = gate.evaluate(wf, passing_mc, 0.14, 1.51, 30, 1.1)
    assert res.approved is False
    assert "G1_OOS_IS_RATIO" in res.failed_gates

def test_g1_boundary(gate, passing_wf, passing_mc):
    # oos_is_ratio = 0.70 (Pass)
    wf = passing_wf.model_copy(update={"oos_is_ratio": 0.70})
    res = gate.evaluate(wf, passing_mc, 0.14, 1.51, 30, 1.1)
    assert res.approved is True

def test_g2_fail(gate, passing_wf, passing_mc):
    # ruin >= 0.02
    mc = passing_mc.model_copy(update={"ruin_probability": 0.025})
    res = gate.evaluate(passing_wf, mc, 0.14, 1.51, 30, 1.1)
    assert res.approved is False
    assert "G2_RUIN_PROBABILITY" in res.failed_gates

def test_g2_boundary(gate, passing_wf, passing_mc):
    # ruin = 0.02 (Fail, strictly less than)
    mc = passing_mc.model_copy(update={"ruin_probability": 0.02})
    res = gate.evaluate(passing_wf, mc, 0.14, 1.51, 30, 1.1)
    assert res.approved is False
    assert "G2_RUIN_PROBABILITY" in res.failed_gates

def test_g3_fail(gate, passing_wf, passing_mc):
    # maxdd >= 0.15
    res = gate.evaluate(passing_wf, passing_mc, max_drawdown_wf=0.16, profit_factor_wf=1.51, paper_trading_days=30, paper_sharpe=1.1)
    assert res.approved is False
    assert "G3_MAX_DRAWDOWN" in res.failed_gates

def test_g3_boundary(gate, passing_wf, passing_mc):
    # maxdd = 0.15 (Fail, strictly less than)
    res = gate.evaluate(passing_wf, passing_mc, max_drawdown_wf=0.15, profit_factor_wf=1.51, paper_trading_days=30, paper_sharpe=1.1)
    assert res.approved is False
    assert "G3_MAX_DRAWDOWN" in res.failed_gates

def test_g4_fail(gate, passing_wf, passing_mc):
    # pf <= 1.50
    res = gate.evaluate(passing_wf, passing_mc, 0.14, profit_factor_wf=1.49, paper_trading_days=30, paper_sharpe=1.1)
    assert res.approved is False
    assert "G4_PROFIT_FACTOR" in res.failed_gates

def test_g4_boundary(gate, passing_wf, passing_mc):
    # pf = 1.50 (Fail, strictly greater than)
    res = gate.evaluate(passing_wf, passing_mc, 0.14, profit_factor_wf=1.50, paper_trading_days=30, paper_sharpe=1.1)
    assert res.approved is False
    assert "G4_PROFIT_FACTOR" in res.failed_gates

def test_g5_fail(gate, passing_wf, passing_mc):
    # days < 30
    res = gate.evaluate(passing_wf, passing_mc, 0.14, 1.51, paper_trading_days=29, paper_sharpe=1.1)
    assert res.approved is False
    assert "G5_PAPER_TRADING_DAYS" in res.failed_gates

def test_g6_fail(gate, passing_wf, passing_mc):
    # sharpe <= 1.0
    res = gate.evaluate(passing_wf, passing_mc, 0.14, 1.51, 30, paper_sharpe=0.9)
    assert res.approved is False
    assert "G6_PAPER_SHARPE" in res.failed_gates

def test_multiple_failures(gate, passing_wf, passing_mc):
    res = gate.evaluate(passing_wf, passing_mc, 0.20, 1.0, 10, 0.5)
    assert len(res.failed_gates) >= 3
    assert "G3_MAX_DRAWDOWN" in res.failed_gates
    assert "G4_PROFIT_FACTOR" in res.failed_gates
    assert "G5_PAPER_TRADING_DAYS" in res.failed_gates
