import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from trading_bot.macro.engine import MacroBiasEngine, MacroState, MacroBiasDecision
from trading_bot.macro.calendar import CalendarSeverity

@pytest.fixture
def engine():
    return MacroBiasEngine()

@pytest.mark.asyncio
async def test_macro_neutral_fallback(engine):
    # Insufficient data (< 6 points)
    closes = [100.0] * 5
    
    decision = await engine.evaluate(
        usdx_closes=closes,
        tlt_closes=closes,
        atr_daily=1.0,
        atr_baseline_252d=1.0,
        cal_severity=CalendarSeverity.NONE
    )
    
    assert decision.macro_state == MacroState.MACRO_NEUTRAL
    assert decision.usd_impulse == 0.0
    assert decision.yield_impulse == 0.0
    assert decision.macro_risk_modifier == 1.0

@pytest.mark.asyncio
async def test_macro_bull_gold(engine):
    # USD impulse < -0.005, Yield impulse < -0.003
    # USDX: 100 -> 99.4 (-0.6%) -> -0.006
    # TLT: 100 -> 100.4 (+0.4%) -> TLT impulse +0.004 -> Yield impulse -0.004
    
    usdx = [100.0, 100.0, 100.0, 100.0, 100.0, 99.4]
    tlt =  [100.0, 100.0, 100.0, 100.0, 100.0, 100.4]
    
    decision = await engine.evaluate(
        usdx_closes=usdx,
        tlt_closes=tlt,
        atr_daily=1.0,
        atr_baseline_252d=1.0
    )
    
    assert decision.usd_impulse == pytest.approx(-0.006)
    assert decision.yield_impulse == pytest.approx(-0.004)
    assert decision.macro_state == MacroState.MACRO_BULL_GOLD
    assert decision.macro_risk_modifier == 1.0

@pytest.mark.asyncio
async def test_macro_bear_gold(engine):
    # USD impulse > 0.005, Yield impulse > 0.003
    # USDX: 100 -> 100.6 (+0.6%) -> +0.006
    # TLT: 100 -> 99.6 (-0.4%) -> TLT impulse -0.004 -> Yield impulse +0.004
    
    usdx = [100.0, 100.0, 100.0, 100.0, 100.0, 100.6]
    tlt =  [100.0, 100.0, 100.0, 100.0, 100.0, 99.6]
    
    decision = await engine.evaluate(
        usdx_closes=usdx,
        tlt_closes=tlt,
        atr_daily=1.0,
        atr_baseline_252d=1.0
    )
    
    assert decision.macro_state == MacroState.MACRO_BEAR_GOLD
    assert decision.macro_risk_modifier == 1.0

@pytest.mark.asyncio
async def test_macro_event_risk_via_calendar(engine):
    # Strong Bull signals, but High Severity Calendar
    usdx = [100.0, 100.0, 100.0, 100.0, 100.0, 99.0] # -1%
    tlt =  [100.0, 100.0, 100.0, 100.0, 100.0, 101.0] # Yield -1%
    
    decision = await engine.evaluate(
        usdx_closes=usdx,
        tlt_closes=tlt,
        atr_daily=1.0,
        atr_baseline_252d=1.0,
        cal_severity=CalendarSeverity.HIGH
    )
    
    assert decision.macro_state == MacroState.MACRO_EVENT_RISK
    assert decision.macro_risk_modifier == 0.5
    assert decision.confidence == 1.0

@pytest.mark.asyncio
async def test_macro_event_risk_via_risk_off(engine):
    # Neutral signals, but Risk Off (ATR > 2x Baseline)
    usdx = [100.0] * 6
    tlt = [100.0] * 6
    
    decision = await engine.evaluate(
        usdx_closes=usdx,
        tlt_closes=tlt,
        atr_daily=2.1,
        atr_baseline_252d=1.0
    )
    
    assert decision.macro_state == MacroState.MACRO_EVENT_RISK
    assert decision.macro_risk_modifier == 0.5

@pytest.mark.asyncio
async def test_exact_boundary_values(engine):
    # USD = -0.005 (Boundary, strictly < required for Bull) -> Neutral
    usdx = [100.0, 100.0, 100.0, 100.0, 100.0, 99.5]
    tlt =  [100.0, 100.0, 100.0, 100.0, 100.0, 100.3] # Yield -0.003
    
    decision = await engine.evaluate(usdx_closes=usdx, tlt_closes=tlt, atr_daily=1.0, atr_baseline_252d=1.0)
    assert decision.macro_state == MacroState.MACRO_NEUTRAL
    
    # USD = +0.005 (Boundary) -> Neutral
    usdx = [100.0, 100.0, 100.0, 100.0, 100.0, 100.5]
    decision = await engine.evaluate(usdx_closes=usdx, tlt_closes=tlt, atr_daily=1.0, atr_baseline_252d=1.0)
    assert decision.macro_state == MacroState.MACRO_NEUTRAL

@pytest.mark.asyncio
async def test_publish_event_on_change(engine):
    usdx = [100.0] * 6
    tlt = [100.0] * 6
    
    with patch.object(engine, '_publish_event', new_callable=AsyncMock) as mock_pub:
        # 1. Neutral -> Neutral (No change)
        await engine.evaluate(usdx, tlt, 1.0, 1.0)
        mock_pub.assert_not_called()
        
        # 2. Neutral -> Event Risk (Change)
        await engine.evaluate(usdx, tlt, 1.0, 1.0, CalendarSeverity.HIGH)
        mock_pub.assert_called_once()
        
        # 3. Event Risk -> Event Risk (No change)
        mock_pub.reset_mock()
        await engine.evaluate(usdx, tlt, 1.0, 1.0, CalendarSeverity.CRITICAL) # Still Event Risk
        mock_pub.assert_not_called()

# --- Coverage Gap Tests ---

def test_macro_decision_naive_datetime_validator():
    # Covers Line 32: validator branch for naive inputs
    naive_dt = datetime(2023, 1, 1, 12, 0, 0)
    decision = MacroBiasDecision(
        macro_state=MacroState.MACRO_NEUTRAL,
        confidence=1.0,
        macro_risk_modifier=1.0,
        usd_impulse=0.0,
        yield_impulse=0.0,
        timestamp_utc=naive_dt
    )
    assert decision.timestamp_utc.tzinfo == timezone.utc

@pytest.mark.asyncio
async def test_getters_and_publish_stub(engine):
    # Covers Lines 144, 147 (getters) and 154 (publish stub)
    usdx = [100.0] * 6
    tlt = [100.0] * 6
    
    # Trigger a state change to ensure publish is called (even if stub)
    # Neutral -> Event Risk
    await engine.evaluate(usdx, tlt, 1.0, 1.0, CalendarSeverity.HIGH)
    
    # Check getters return correct updated state
    assert engine.get_current_state() == MacroState.MACRO_EVENT_RISK
    assert engine.get_risk_modifier() == 0.5
    
    # Call _publish_event directly to ensure the line is covered even if mocked elsewhere
    # (Though logic flow hits it, direct call guarantees coverage count)
    decision = MacroBiasDecision(
        macro_state=MacroState.MACRO_NEUTRAL,
        confidence=1.0,
        macro_risk_modifier=1.0,
        usd_impulse=0.0,
        yield_impulse=0.0,
        timestamp_utc=datetime.now(timezone.utc)
    )
    await engine._publish_event(decision)
