# XAU/USD Gold Bot — System Specification
**Version:** v3.0  
**Status:** LOCKED — do not modify  
**Date:** March 2026  
**Authority:** This file supersedes all previous versions (v2.1, v2.3)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Pipeline & Responsibilities](#2-module-pipeline--responsibilities)
3. [Contract Schemas](#3-contract-schemas)
4. [Macro Bias Engine](#4-macro-bias-engine)
5. [Volatility Regime Engine](#5-volatility-regime-engine)
6. [Strategy Dispatcher](#6-strategy-dispatcher)
7. [Swing Strategy — TSMOM Multi-Horizon](#7-swing-strategy--tsmom-multi-horizon)
8. [ORB Strategy — Opening Range Breakout](#8-orb-strategy--opening-range-breakout)
9. [News Breakout Strategy](#9-news-breakout-strategy)
10. [Scalping Strategy — Optional](#10-scalping-strategy--optional)
11. [Risk Engine](#11-risk-engine)
12. [Portfolio Coordinator](#12-portfolio-coordinator)
13. [Execution Quality Gate](#13-execution-quality-gate)
14. [Backtesting & Validation](#14-backtesting--validation)
15. [Monitoring & Alerting](#15-monitoring--alerting)
16. [Implementation Roadmap — Trae Stages](#16-implementation-roadmap--trae-stages)
17. [Numeric Threshold Appendix](#17-numeric-threshold-appendix)

---

## 1. System Overview

### 1.1 Problem Statement

Public gold bots fail via five recurring patterns:

| Failure | Evidence | Root Cause |
|---|---|---|
| Tail Risk | golduxe: 81.2% win rate, 69.3% DD | RR < 0.5; no loss cap |
| Collapsed RR | Gold Bot XAU-USD: RR = 0.19 | Small fixed TP, runaway SL |
| Overfitting | CITY GOLD HUNTER: PF 6.02 on 3-day sample | No walk-forward validation |
| News Blindness | Multiple blowups at 13:30 UTC Fridays | No event detection |
| Fixed Sizing | 2× loss on high-vol days | No volatility adjustment |

### 1.2 Design Objectives

| Metric | Target | Hard Reject Below |
|---|---|---|
| Sharpe Ratio | > 1.5 | < 0.8 |
| Max Drawdown | < 15% | > 25% |
| Profit Factor | > 1.5 | < 1.2 |
| Expectancy | > 0.5R | ≤ 0R |
| Monte Carlo ROR | < 2% | > 5% |
| OOS / IS Sharpe | ≥ 70% | < 50% |

---

## 2. Module Pipeline & Responsibilities

### 2.1 Pipeline Order

```
Data Ingestion
  → Macro Bias Engine
    → Volatility Regime Engine
      → Strategy Dispatcher
        → Strategy Engines (Swing, ORB, News, Scalp)
          → Risk Engine
            → Portfolio Coordinator
              → Execution Quality Gate
                → Execution Engine
                  → Monitoring & Audit
```

### 2.2 Module Responsibility Charter

| Module | Single Responsibility | Output | Must NOT Do |
|---|---|---|---|
| Data Ingestion | Normalise OHLCV; detect gaps; maintain buffers | OHLCV events per timeframe | Strategy logic, risk logic |
| Macro Bias Engine | Classify gold macro environment | `MACRO_STATE` + confidence [0–1] | Trade directly; be a signal source |
| Volatility Regime Engine | Classify volatility state with hysteresis | `REGIME_STATE` + `risk_scalar` | Dispatch strategy permissions |
| Strategy Dispatcher | Map (regime, macro, time, session) → permissions | `DispatcherPermissions` object | Emit orders or signals |
| Strategy Engines | Detect setup; score it; package intent | `SignalIntent` object | Size position; check risk; execute |
| Risk Engine | Transform intent into sized, stopped decision | `RiskDecision` object | Execute orders; manage portfolio |
| Portfolio Coordinator | Resolve conflicts; cap heat; weight quality | Updated `RiskDecision` or rejection | Generate signals; execute |
| Execution Quality Gate | Check real-world execution feasibility | `ExecutionDecision` object | Trade; size; generate signals |
| Execution Engine | Submit and manage orders | Order state transitions + audit log | Generate signals; size; risk-check |
| Monitoring & Audit | Observe system health; emit alerts | Metrics + alert events | Modify system behaviour |

---

## 3. Contract Schemas

All inter-module communication uses exactly three normalised contracts.  
No module may invent its own data format for downstream consumption.

### 3.1 SignalIntent  *(Strategy Engine → Risk Engine)*

| Field | Type | Allowed Values | Validation Rule |
|---|---|---|---|
| `strategy_name` | str | `"SWING"`, `"ORB"`, `"NEWS"`, `"SCALP"` | Must be one of four allowed values |
| `direction` | int | `+1` (long), `-1` (short) | Must be exactly +1 or -1 |
| `score` | int | 0 – 5 | Must be ≥ 4 to pass to Risk Engine |
| `entry_type` | str | `"MARKET"`, `"LIMIT"`, `"STOP_LIMIT"` | Must be one of three allowed values |
| `entry_trigger` | float | Any positive float | Must be > 0 |
| `sl_distance` | float | Any positive float | Must be > 0; in price units |
| `tp_plan.tp1_distance` | float | Any positive float | Must be > 0 |
| `tp_plan.tp1_size_pct` | float | 0.0 – 1.0 | Fraction closed at TP1 |
| `tp_plan.tp2_distance` | float \| null | Positive float or null | null = trail only after TP1 |
| `tp_plan.trail_atr_mult` | float | > 0 | Trailing stop = Highest_Close − N × ATR |
| `timeout_plan.max_bars` | int | > 0 | Hard exit after N bars |
| `timeout_plan.mandatory_exit_utc` | str \| null | `"HH:MM"` or null | Time-based mandatory exit |
| `regime_context` | REGIME_STATE | See Section 5 | Must match current regime |
| `macro_context` | MACRO_STATE | See Section 4 | Must match current macro state |
| `execution_constraints.max_spread_bp` | float | > 0 | Per-strategy values in Section 17.5/17.6 |
| `execution_constraints.max_slippage_bp` | float | > 0 | Per-strategy values |
| `execution_constraints.min_quote_fresh_ms` | int | > 0 | Per-strategy values |

### 3.2 RiskDecision  *(Risk Engine → Portfolio Coordinator → Execution Quality Gate)*

| Field | Type | Description |
|---|---|---|
| `approved` | bool | Whether the trade may proceed |
| `rejection_reason` | str \| null | Populated if `approved = false` |
| `position_size` | float | Lots, after all modifiers applied |
| `risk_fraction_used` | float | Actual `r` applied after vol scalar |
| `stop_price` | float | Absolute stop-loss price |
| `take_profit_plan` | tp_plan | Forwarded from SignalIntent |
| `circuit_breaker_state.daily_loss_pct` | float | Current day realised + unrealised P&L % |
| `circuit_breaker_state.weekly_loss_pct` | float | Current week P&L % |
| `circuit_breaker_state.consecutive_losses` | int | Count of sequential losing trades |
| `circuit_breaker_state.breaker_active` | bool | True if any CB is currently triggered |
| `portfolio_heat_snapshot.current_heat_pct` | float | Sum of all open position risks |
| `portfolio_heat_snapshot.heat_remaining` | float | `5.0% − current_heat_pct` |
| `portfolio_heat_snapshot.open_positions` | int | Count of currently open positions |
| `strategy_viability.rolling_er` | float | Expectancy R on last 30 trades |
| `strategy_viability.viable` | bool | False if rolling ER ≤ 0 |

### 3.3 ExecutionDecision  *(Execution Quality Gate → Execution Engine)*

| Field | Type | Description |
|---|---|---|
| `allowed_state` | str | `"ALLOWED"`, `"DEGRADED"`, `"BLOCKED"` |
| `blocked_reason` | str \| null | Populated if state is BLOCKED |
| `spread_snapshot` | float | Live spread in basis points |
| `spread_vs_median` | float | Ratio: live spread / rolling 100-tick median |
| `quote_freshness_ms` | int | Age of last price quote in milliseconds |
| `expected_slippage_bp` | float | Pre-fill slippage estimate |
| `actual_slippage_bp` | float \| null | Populated post-execution |
| `routing_mode` | str | `"LIMIT"` or `"MARKET_FALLBACK"` |
| `order_intent_id` | str | UUID linking back to originating SignalIntent |
| `degraded_action` | str \| null | `"REDUCE_SIZE_50PCT"` when DEGRADED |

---

## 4. Macro Bias Engine

### 4.1 State Enum

| State | Meaning |
|---|---|
| `MACRO_BULL_GOLD` | USD falling + real yields falling → gold bullish |
| `MACRO_BEAR_GOLD` | USD rising + real yields rising → gold bearish |
| `MACRO_NEUTRAL` | Neither threshold crossed |
| `MACRO_EVENT_RISK` | High-impact event imminent or active; or risk-off spike |

### 4.2 Input Proxies

| Input | Proxy | Fallback |
|---|---|---|
| USD impulse | 5-day ROC of USDX/DXY daily close | Set `usd_impulse = 0` (neutral) |
| Real-yield impulse | 5-day ROC of US10Y or TLT (inverted) | Set `yield_impulse = 0` |
| Calendar severity | Calendar service: NONE / LOW / HIGH / CRITICAL | Derive from shock candle |
| Risk-off override | VIX proxy > 25; fallback: `ATR_daily > 2.0 × ATR_baseline_252d` | ATR-based always available |

### 4.3 Classification Logic

```
usd_impulse   = (USDX_close - USDX_close[5]) / USDX_close[5]
yield_impulse = (TIPS_close  - TIPS_close[5]) / TIPS_close[5]

if cal_severity in [HIGH, CRITICAL] OR risk_off:
    macro_state = MACRO_EVENT_RISK
elif usd_impulse < -0.005 AND yield_impulse < -0.003:
    macro_state = MACRO_BULL_GOLD
elif usd_impulse > +0.005 AND yield_impulse > +0.003:
    macro_state = MACRO_BEAR_GOLD
else:
    macro_state = MACRO_NEUTRAL
```

### 4.4 Effect on Strategy Permissions

| Macro State | Swing | ORB | News | Scalping | Sizing Modifier |
|---|---|---|---|---|---|
| MACRO_BULL_GOLD | Long bias | Long only | Long preferred | Long only | × 1.0 |
| MACRO_BEAR_GOLD | Short bias | Short only | Short preferred | Short only | × 1.0 |
| MACRO_NEUTRAL | Both | Both | Both | Both | × 1.0 |
| MACRO_EVENT_RISK | Reduce 50%; no new entries | BLOCKED | Whitelist only | BLOCKED | × 0.5 |

---

## 5. Volatility Regime Engine

### 5.1 State Enum

| State | ATR Ratio Condition | risk_scalar |
|---|---|---|
| `LOW_VOL` | ratio < 0.85 | 1.00 |
| `MID_VOL` | 0.85 ≤ ratio ≤ 1.25 | 0.80 |
| `HIGH_VOL` | ratio > 1.25 | 0.50 |
| `SHOCK_EVENT` | TR_current > 2.0 × ATR_fast | 0.00 |

Where: `ratio = ATR_fast(14, M5) / ATR_slow(200, M5)`

### 5.2 Hysteresis Rule

Regime state changes only after **3 consecutive confirming bars**.  
`SHOCK_EVENT` overrides immediately — no hysteresis required for shocks.

### 5.3 Shock Recovery Protocol

```
Cooldown: 12 bars = 60 minutes after shock

if bars_since_shock >= 12 AND ATR_fast <= 1.25 × ATR_slow:
    regime = MID_VOL   # never jump directly to LOW_VOL
    risk_scalar = 0.80
elif bars_since_shock >= 12:
    regime = HIGH_VOL
    risk_scalar = 0.50
```

---

## 6. Strategy Dispatcher

### 6.1 Output Schema

See `DispatcherPermissions` in Section 3 / `docs/event_contracts.md`.

### 6.2 Priority Table

| Priority | Condition | Scalp | ORB | News | Swing | Direction |
|---|---|---|---|---|---|---|
| 1 | SHOCK_EVENT or approved news active | ✗ | ✗ | ✓ | monitor | From macro |
| 2 | Blackout (±10 min around HIGH/CRITICAL news) | ✗ | ✗ | ✗ | monitor | From macro |
| 3 | London open 07:00–08:00 UTC; no shock; no blackout | ✗ | ✓ | standby | ✓ | From swing_dir |
| 4 | LOW_VOL + liquid session (07:00–17:00); no event | ✓ | ✗ | ✗ | ✓ | From swing_dir |
| 5 | Any other time | ✗ | ✗ | ✗ | ✓ | From swing_dir |

### 6.3 Direction Constraint

```
swing_dir = sign(confidence_score)  # from multi-horizon TSMOM

if macro_state == MACRO_BULL_GOLD:  direction_constraint = max(swing_dir, 0) or +1
if macro_state == MACRO_BEAR_GOLD:  direction_constraint = min(swing_dir, 0) or -1
if macro_state == MACRO_NEUTRAL:    direction_constraint = swing_dir
if macro_state == MACRO_EVENT_RISK: direction_constraint = 0  # news only, any dir
```

---

## 7. Swing Strategy — TSMOM Multi-Horizon

### 7.1 Signal Computation  *(Monday 00:05 UTC)*

```
mom_1m  = (Close[t] - Close[t-21])  / realized_vol_21d
mom_3m  = (Close[t] - Close[t-63])  / realized_vol_63d
mom_6m  = (Close[t] - Close[t-126]) / realized_vol_126d
mom_12m = (Close[t] - Close[t-252]) / realized_vol_252d

# Inverse-volatility weighted score
confidence = 0.10×sign(mom_1m) + 0.20×sign(mom_3m) + 0.30×sign(mom_6m) + 0.40×sign(mom_12m)

if   confidence >  +0.30:  swing_dir = +1
elif confidence <  -0.30:  swing_dir = -1
else:                      swing_dir =  0
```

### 7.2 Turning-Point Brake

Activates when **all four** conditions hold:

```
brake_cond_1 = swing_dir != 0
brake_cond_2 = sign(mom_1m) != swing_dir
brake_cond_3 = mom_1m < -0.50 × StdDev(mom_1m_series, 52w)
brake_cond_4 = ATR_fast / ATR_slow > 1.15

if ALL four: risk_scalar_override = 0.40
```

### 7.3 Position Sizing

```
base_lots = (Equity × 0.010) / (ATR(20,Daily) × contract_oz)
lots = base_lots × regime_risk_scalar × macro_risk_modifier × brake_modifier
lots = min(lots, 1.5 × base_lots)
```

### 7.4 Exits

- **Rebalance:** every Monday 00:05 UTC; close and reverse if `sign(confidence)` flips
- **Chandelier stop:** `Highest_Close(since_entry) − 3.0 × ATR(20, Daily)`
- **Neutral hesitation:** if confidence crosses 0 but `|confidence| < 0.30`, reduce 50% and wait one week

---

## 8. ORB Strategy — Opening Range Breakout

### 8.1 Opening Range

- **Window:** 07:00–08:00 UTC (London open)
- `OR_High = MAX(High)` over window
- `OR_Low  = MIN(Low)`  over window

### 8.2 Quality Band

```
lower_bound = max(0.25 × ATR_daily_14, 0.70 × OR_median_20)
upper_bound = min(1.80 × ATR_daily_14, 2.00 × OR_median_20)
orb_quality_ok = lower_bound ≤ OR_width ≤ upper_bound
```

If `orb_quality_ok = false` → `NO_TRADE`

### 8.3 Adaptive Buffer

```
or_quality_ratio = OR_width / OR_median_20
buf_base  = 0.15 × ATR(14, M15)
buf_final = buf_base × (1.0 + 0.30 × (1.0 - or_quality_ratio))
buf_final = clamp(buf_final, 0.10 × ATR_14_M15, 0.25 × ATR_14_M15)
```

### 8.4 Entry Conditions

```
Long:  Close(M15) > OR_High + buf_final
       AND Volume > 1.20 × VMA(20, M15)
       AND direction_constraint >= 0
       AND NOT blackout_active

Short: Close(M15) < OR_Low - buf_final
       AND Volume > 1.20 × VMA(20, M15)
       AND direction_constraint <= 0

HARD RULE: one trade per session; no re-entry after stop-out
```

### 8.5 Trade Management

| Parameter | Value |
|---|---|
| Stop-Loss | `OR_Low − buf_final`; cap at `1.50 × ATR(14, M15)` |
| TP1 (50%) | `Entry + 1.0 × SL_distance`; on hit → SL to breakeven |
| Trailing (50%) | `Highest_Close − 2.0 × ATR(14, M15)` per M15 bar |
| Mandatory exit | 20:45 UTC (19:30 UTC Fridays) |
| Shock abort | `TR(M15) > 2.0 × ATR(14)` in last bar before entry → skip |

---

## 9. News Breakout Strategy

### 9.1 Event Whitelist

| Tier | Events | Entry Window |
|---|---|---|
| T1 — Always trade | NFP (Fri 13:30), FOMC Rate Decision, FOMC Press Conf | 3 min post-release |
| T2 — Trade if spread OK | US CPI (Wed 13:30), US GDP (Advance) | 2 min post-release |
| T3 — Trade if HIGH_VOL | Durable Goods Orders (Thu 13:30) | 1 min post-release |
| BLOCKED | All non-whitelisted events | Never |

### 9.2 Pre-Trade Gates  *(all must pass)*

```
spread_gate:   spread_live_bp ≤ min(3.0, 2.0 × spread_median_100)
freshness:     quote_age_ms < 500
slippage:      expected_slippage_bp < 5.0
session:       time_utc in [07:00, 21:00]
```

### 9.3 Entry Logic

```
Pre-news range: 5 M5 bars before event
buf_news = 0.20 × ATR(14, M5)

Long:  Close(M5) > pre_news_high + buf_news  AND all_gates_pass
Short: Close(M5) < pre_news_low  - buf_news  AND all_gates_pass

Expiry: T1 = 3 min, T2 = 2 min, T3 = 1 min after event time
Never enter if entry > 1.5 × SL_distance from trigger level
```

### 9.4 Trade Management

| Parameter | Value |
|---|---|
| Stop-Loss | Opposite side of pre-news range minus/plus buf_news |
| Take-Profit | `Entry ± 2.0 × SL_distance` (R:R = 2.0 minimum) |
| Max per day | 2 News Breakout trades |

---

## 10. Scalping Strategy — Optional

**Priority:** 4th — launch only after Stages 01–11 fully validated.

### 10.1 Activation Gates  *(all must pass)*

```
regime == LOW_VOL (strictly)
macro != MACRO_EVENT_RISK
NOT blackout_active
spread_live_bp ≤ 2.0
time_utc in [08:00, 17:00]
```

### 10.2 Signal Conditions (Long)

```
Close(M5) < BB_Lower(SMA_20, 2.0σ)
RSI(14) < 30
Close(M5) > EMA(200, M5)
Volume > 1.10 × VMA(20, M5)
|Close - EMA(200,M5)| < 2.50 × ATR(14,M5)

Signal Score must be ≥ 4/5:
  +2: BB breakout (mandatory)
  +1: HTF trend aligned (EMA_20_4H direction matches)
  +1: Volume > 1.50 × VMA(20)
  +1: RSI in zone (< 30 long; > 70 short)
```

### 10.3 Trade Management

| Parameter | Value |
|---|---|
| Stop-Loss | `Entry ± 1.0 × ATR(14, M5)` |
| TP1 (50%) | `Entry ∓ 0.5 × ATR`; then SL to breakeven |
| Trailing (50%) | `Highest_Close − 1.0 × ATR` per bar |
| Time stop | 6 bars = 30 minutes |
| Daily halt | 5 consecutive scalping losses → halt for remainder of day |

---

## 11. Risk Engine

### 11.1 Universal Position Sizing

```
base_lots = (Equity × r) / (SL_distance × contract_oz_per_lot)
lots = base_lots
     × regime_risk_scalar          # Regime Engine: 0.0–1.0
     × macro_risk_modifier          # 0.5 if MACRO_EVENT_RISK, else 1.0
     × viability_modifier           # 0.5 if rolling_ER near 0; 0.0 if suspended
     × quality_weight               # Portfolio Coordinator: 0.5–1.0
lots = max(lots, broker_min_lot_size)
```

### 11.2 Circuit Breaker Table

| Trigger | Condition | Action | Reset |
|---|---|---|---|
| Shock candle | `TR(M5) > 2.0 × ATR_fast` | SHOCK_EVENT; `risk_scalar = 0`; pause 12 bars | Auto per Section 5.3 |
| Daily loss — soft | Scalp: `< −1.5%`; ORB: `< −2.0%` | Halt that strategy for rest of day | Auto at 00:00 UTC |
| Daily loss — hard | Portfolio `< −3.0%` | Close ALL positions; halt all new entries | Auto at 00:00 UTC |
| Weekly loss | Portfolio `< −8%` week-open | Full halt 7 days | Manual review |
| 5 consecutive losses | Any strategy | Reduce that strategy size 50% for next 10 trades | Auto after 10 trades |
| ER turns negative | `rolling_ER(30) ≤ 0` | `viability_modifier = 0.50`; alert ops | Manual + 30-trade window |
| ER strongly negative | `rolling_ER(30) < −0.10` | `viability_modifier = 0.00`; suspend strategy | Manual review |
| API error rate | > 10% in 5-minute window | Halt new entries; alert ops | Manual |
| Peak drawdown | > 15% from equity peak | Full shutdown; close all positions | Manual + capital review |

### 11.3 Strategy Viability Monitor

```
p          = count(wins) / 30      # last 30 closed trades
b          = mean(win_R) / mean(|loss_R|)
rolling_ER = p × b - (1 - p)
rolling_PF = sum(gross_wins_R) / sum(|gross_loss_R|)

viable = (rolling_ER > 0) AND (rolling_PF > 1.10)
```

---

## 12. Portfolio Coordinator

### 12.1 Hard Limits

```
max_portfolio_heat    = 5.0%    # sum of all open position risks
max_single_position   = 30%     # no strategy > 30% of total heat
max_open_positions    = 5
net_exposure_cap      = 1.0     # |pos_swing + pos_orb + pos_scalp| in risk units
net_exposure_floor    = 0.15    # if below, go flat (avoid churn)
```

### 12.2 Quality-Weighted Allocation

```
quality_weight = f(rolling_PF, rolling_ER, realized_slippage_deviation)
quality_weight = clamp(quality_weight, 0.50, 1.00)
```

### 12.3 Risk Budget

| Strategy | Budget | Risk/Trade |
|---|---|---|
| Swing TSMOM | 60% | ~1.0% vol-target |
| ORB | 25% | 0.50% |
| Scalping | 15% | 0.25% |
| News Breakout | From ORB budget | 0.50% |

---

## 13. Execution Quality Gate

### 13.1 State Machine

```
EXECUTION_BLOCKED   if: spread_ratio > 3.0
                     OR  quote_age_ms > 1000
                     OR  api_error_rate_5min >= 0.05
                     OR  session outside liquid hours (07:00–20:00 UTC)

EXECUTION_DEGRADED  if: NOT BLOCKED
                     AND (spread_ratio > 1.80
                          OR expected_slippage_bp > signal.max_slippage_bp)
                     Action: REDUCE_SIZE_50PCT

EXECUTION_ALLOWED   otherwise
```

### 13.2 Per-Strategy Constraints

| Strategy | max_spread_bp | max_slippage_bp | min_quote_fresh_ms | Session |
|---|---|---|---|---|
| Swing | 10.0 | 15.0 | 2000 | None |
| ORB | 3.0 | 5.0 | 500 | 07:00–20:00 UTC |
| News T1/T2 | 3.0 / 2.5 | 5.0 | 500 | 07:00–21:00 UTC |
| Scalping | 2.0 | 3.0 | 300 | 08:00–17:00 UTC |

---

## 14. Backtesting & Validation

### 14.1 Walk-Forward Protocol

| Phase | Window | Gate |
|---|---|---|
| In-Sample | 36 months rolling | Document chosen parameters |
| Out-of-Sample | 6 months | OOS Sharpe ≥ 70% of IS Sharpe |
| Step | 3-month roll | All windows must pass OOS gate |

### 14.2 Transaction Cost Model

```
Total = Spread + Commission + Slippage + FundingCost
Spread:     dynamic; 2× wider at opens and within 5 min of news
Commission: 0.10% per side
Slippage:   0.05% baseline + (order_lots / ADV_lots) × 0.10
Funding:    Swing positions held > 8 hours → broker swap rate
```

### 14.3 Deployment Gate

All of the following must pass before live capital is allocated:

- [ ] Walk-forward OOS Sharpe ≥ 70% of IS Sharpe across all windows
- [ ] Monte Carlo ruin probability < 2%
- [ ] Max Drawdown < 15% in walk-forward
- [ ] Profit Factor > 1.5 in walk-forward
- [ ] 30+ consecutive paper trading days with Sharpe > 1.0
- [ ] All circuit breakers tested against historical NFP and FOMC bars
- [ ] All API keys: withdrawal permissions DISABLED; IP whitelist ACTIVE
- [ ] Dead-man switch tested: closes all positions if heartbeat silent > 60s
- [ ] All monitoring alerts tested end-to-end
- [ ] Human risk review sign-off obtained

---

## 15. Monitoring & Alerting

| Alert | Severity | Channel | SLA |
|---|---|---|---|
| Daily loss > 2% | WARNING | Telegram | 1 hour |
| Daily loss > 3% (hard CB) | CRITICAL | PagerDuty + Telegram | Immediate |
| Weekly drawdown > 5% | CRITICAL | PagerDuty + Email | Same day |
| Peak drawdown > 15% | CRITICAL | PagerDuty | Immediate shutdown |
| API error rate > 5% | CRITICAL | PagerDuty | Immediate |
| rolling_ER turns negative | WARNING | Telegram | Review within 1 hour |
| Strategy suspended | CRITICAL | PagerDuty | Immediate manual review |
| Dead-man switch (no heartbeat > 60s) | CRITICAL | PagerDuty | Watchdog closes all positions |
| Slippage > 3× expected | WARNING | Telegram | Review execution path |

---

## 16. Implementation Roadmap — Trae Stages

| Stage | Goal | Gate |
|---|---|---|
| 00 | Contract freeze | All schemas explicit; zero code; no file outside allowed list |
| 01 | Data ingestion | Clean data on 5+ years verified; mocked IO tests pass |
| 02 | Regime + shock | All LOW/MID/HIGH/SHOCK transitions tested including hysteresis |
| 03 | Macro + calendar | Explicit state output; no trading logic; fallback paths tested |
| 04 | Dispatcher | Deterministic permissions; all edge cases covered |
| 05 | Swing engine | Paper rebalance intent only; turning-point brake tested |
| 06 | ORB engine | One trade/day; quality bands tested; no re-entry after stop |
| 07 | News engine | Strict expiry; no-chase enforced; all tiers tested |
| 08 | Scalping engine | Strictest gates validated; paper only; LOW_VOL gate enforced |
| 09 | Risk engine | RiskDecision emitted only; all CBs fire on test inputs |
| 10 | Portfolio coordinator | Reproducible state transitions; heat cap enforced |
| 11 | Execution quality gate | BLOCKED/DEGRADED/ALLOWED state machine tested |
| 12 | Paper execution | Paper fills only; 30-day paper run; full audit trail |
| 13 | Monitoring & ops | All alerts fire in test; no secret leakage; dead-man switch tested |
| 14 | Research harness | ROR < 2%; OOS Sharpe ≥ 70% IS; cost model applied |

---

## 17. Numeric Threshold Appendix

All values below are **frozen defaults**. Any change requires a backtest comparison and human approval.

### 17.1 Regime Engine

| Parameter | Value |
|---|---|
| ATR_fast period | 14 bars M5 |
| ATR_slow period | 200 bars M5 |
| LOW_VOL upper bound | ratio < 0.85 |
| HIGH_VOL lower bound | ratio > 1.25 |
| Hysteresis confirmation | 3 consecutive bars |
| Shock candle threshold | TR > 2.0 × ATR_fast |
| Shock cooldown | 12 bars (60 min) |
| LOW_VOL risk_scalar | 1.00 |
| MID_VOL risk_scalar | 0.80 |
| HIGH_VOL risk_scalar | 0.50 |
| SHOCK risk_scalar | 0.00 |

### 17.2 Macro Bias Engine

| Parameter | Value |
|---|---|
| USD impulse window | 5-day ROC |
| MACRO_BULL_GOLD threshold | USD ROC < −0.005 AND yield ROC < −0.003 |
| MACRO_BEAR_GOLD threshold | USD ROC > +0.005 AND yield ROC > +0.003 |
| Risk-off VIX threshold | > 25 |
| Risk-off ATR fallback | ATR_daily > 2.0 × ATR_baseline_252d |
| MACRO_EVENT_RISK sizing modifier | 0.50 |

### 17.3 Swing TSMOM

| Parameter | Value |
|---|---|
| Horizon weights | 1M=0.10, 3M=0.20, 6M=0.30, 12M=0.40 |
| Bullish threshold | confidence > +0.30 |
| Bearish threshold | confidence < −0.30 |
| Neutral zone | −0.30 to +0.30 |
| Turning-point brake: 1M threshold | `mom_1m < −0.50 × StdDev(mom_1m, 52w)` |
| Turning-point brake: ATR threshold | ATR_fast/ATR_slow > 1.15 |
| Turning-point brake size reduction | 0.40 (40% of normal) |
| Rebalance | Every Monday 00:05 UTC |
| Chandelier stop multiplier | 3.0 × ATR(20, Daily) |
| Vol target sigma | 1.0% daily equity risk |
| Max size cap | 1.5 × base_lots |

### 17.4 ORB

| Parameter | Value |
|---|---|
| Opening range window | 07:00–08:00 UTC |
| Quality band lower | `max(0.25 × ATR_daily, 0.70 × OR_median_20)` |
| Quality band upper | `min(1.80 × ATR_daily, 2.00 × OR_median_20)` |
| Adaptive buf base | 0.15 × ATR(14, M15) |
| Adaptive buf clamp | [0.10 × ATR, 0.25 × ATR] |
| Volume confirmation | > 1.20 × VMA(20, M15) |
| SL cap | 1.50 × ATR(14, M15) |
| Trailing stop | 2.0 × ATR(14, M15) |
| Mandatory exit | 20:45 UTC (19:30 UTC Fridays) |
| Retest tolerance | 0.5 × buf_final (breakout-retest mode) |

### 17.5 News Breakout

| Parameter | Value |
|---|---|
| Pre-news range window | 5 M5 bars before event |
| buf_news | 0.20 × ATR(14, M5) |
| Max spread T1 | 3.0 bp |
| Max spread T2 | 2.5 bp |
| Max expected slippage | 5.0 bp |
| Quote freshness limit | 500 ms |
| T1 entry window | 3 min post-release |
| T2 entry window | 2 min post-release |
| T3 entry window | 1 min post-release |
| TP target | 2.0 × SL_distance (R:R = 2.0) |
| Max chase limit | 1.5 × SL_distance from trigger |
| Max per day | 2 trades |

### 17.6 Scalping

| Parameter | Value |
|---|---|
| Regime gate | LOW_VOL strictly |
| BB parameters | SMA(20), 2.0σ |
| RSI period / levels | 14 / < 30 long; > 70 short |
| EMA trend filter | EMA(200, M5) |
| Volume confirmation | > 1.10 × VMA(20, M5) |
| Trend-day filter | `|Close − EMA(200)| < 2.50 × ATR(14)` |
| Max spread ceiling | 2.0 bp |
| SL distance | 1.0 × ATR(14, M5) |
| TP1 distance | 0.5 × ATR(14, M5) |
| Trailing multiplier | 1.0 × ATR (post-TP1) |
| Time stop | 6 bars = 30 min |
| Daily consecutive loss halt | 5 losses → halt for rest of day |
| Min viable win rate (ER > 0) | 63% at 0.65R avg win |
