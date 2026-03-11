# AI Guardrails — Automated Trading Bot
**Status:** LOCKED  
**Version:** v3.0  
**Applies to:** All AI agents (Trae, Copilot, Cursor, Claude, etc.)

This document defines **non-negotiable constraints** for any AI agent working on
this repository. The goal is to protect capital, correctness, and architectural
integrity at every stage of development.

---

## 1. Source of Truth

- **`docs/specification.md`** is the **single authoritative specification**
- It is **LOCKED** — no AI agent may modify it
- **`docs/ai-guardrails.md`** (this file) is **LOCKED** — no AI agent may modify it
- Any deviation, conflict, or ambiguity between a prompt and the spec must be
  **escalated to a human immediately** — never resolved silently
- If instructions conflict, **the specification always wins**

**Contract hierarchy (highest to lowest authority):**
1. `docs/specification.md`
2. `docs/ai-guardrails.md`
3. Stage-specific prompt
4. Inline code comments

---

## 2. Scope Control

AI agents MUST:
- Work only within the currently active Git branch
- Modify only files **explicitly listed** in the current stage prompt under "Allowed Files"
- Read any file freely, but treat unlisted files as read-only
- Stop and ask if task scope is unclear or a required file is not in the allowed list

AI agents MUST NOT:
- Create, delete, rename, or merge branches
- Push directly to `main`
- Rewrite or squash git history
- Modify files outside the declared allowed list
- Modify `docs/specification.md` or `docs/ai-guardrails.md` under any circumstances

---

## 3. Module Boundaries

The system is divided into **10 strict modules** as defined in `docs/specification.md` Section 2:

```
Data Ingestion → Macro Bias Engine → Volatility Regime Engine →
Strategy Dispatcher → Strategy Engines → Risk Engine →
Portfolio Coordinator → Execution Quality Gate → Execution Engine →
Monitoring & Audit
```

AI agents MUST:
- Respect each module's single-responsibility charter
- Ensure modules communicate only via the normalised contracts:
  `SignalIntent → RiskDecision → ExecutionDecision`
- Never allow a module to perform the responsibility of another module

AI agents MUST NOT:
- Allow a Strategy Engine to size positions, check risk, or submit orders
- Allow the Risk Engine to generate signals or execute trades
- Allow the Dispatcher to emit orders or signals (permissions only)
- Allow the Execution Engine to generate signals, size positions, or manage risk
- Create cross-module dependencies not defined in the specification

---

## 4. Strategy & Risk Integrity

AI agents MUST NOT:
- Change trading strategy logic
- Modify risk parameters or their defaults (see Threshold Appendix, spec Sections 17.1–17.6)
- Adjust stop-loss, take-profit, position sizing, or circuit breaker rules
- Add heuristics, optimisations, or statistical inference not in the spec
- Introduce machine learning, neural networks, or adaptive parameter tuning
- Add new indicators, signals, or entry conditions not in the spec
- Modify the multi-horizon TSMOM weights, turning-point brake thresholds,
  ORB quality band bounds, or News Breakout whitelist

All strategy and risk behaviour must remain:
- **Deterministic** — same inputs always produce same outputs
- **Rule-based** — no probabilistic decisions
- **Config-driven** — all parameters sourced from config, not hardcoded

**The following numeric values are frozen and must not be altered by any agent:**

| Parameter | Frozen Value |
|---|---|
| ATR_fast period | 14 bars M5 |
| ATR_slow period | 200 bars M5 |
| Hysteresis confirmation | 3 consecutive bars |
| Shock candle threshold | TR > 2.0 × ATR_fast |
| Shock cooldown | 12 bars (60 min) |
| TSMOM horizon weights | 1M=0.10, 3M=0.20, 6M=0.30, 12M=0.40 |
| Turning-point brake ATR threshold | ATR_fast/ATR_slow > 1.15 |
| Turning-point brake size reduction | 40% of normal |
| ORB quality band lower | max(0.25 × ATR_d, 0.70 × OR_median) |
| ORB quality band upper | min(1.80 × ATR_d, 2.00 × OR_median) |
| News buf | 0.20 × ATR(14, M5) |
| Scalping spread ceiling | 2.0 bp |
| Daily hard loss halt | −3.0% portfolio |
| Peak drawdown shutdown | −15% from peak |
| Monte Carlo ruin gate | ROR < 2% |

---

## 5. Contract Enforcement

All inter-module communication must use the three normalised contracts.  
No module may invent its own data format for downstream consumption.

**`SignalIntent`** — written by: Strategy Engines. Read by: Risk Engine.  
**`RiskDecision`** — written by: Risk Engine. Read by: Portfolio Coordinator → Execution Quality Gate.  
**`ExecutionDecision`** — written by: Execution Quality Gate. Read by: Execution Engine.

AI agents MUST NOT:
- Add fields to these schemas without human approval and spec update
- Remove or rename fields
- Change field types
- Allow any module to read a contract it is not designated to read

---

## 6. Execution Safety

AI agents MUST NOT:
- Connect to live exchanges
- Place real or simulated orders outside the designated Paper Execution stage (Stage 12+)
- Enable live trading code paths
- Disable or bypass safety flags (`DRY_RUN`, `PAPER_ONLY`, circuit breakers)
- Remove or weaken the Execution Quality Gate

The system must default to:
- `DRY_RUN = True`
- `PAPER_ONLY = True`
- No external side effects
- No capital exposure

**Stage-specific execution permissions:**

| Stage | Execution Permitted |
|---|---|
| 00–11 | No execution of any kind |
| 12 | Paper fills only; no live orders |
| 13 | Monitoring only; no orders |
| 14 | Backtest / research harness; no live orders |
| Live | Only after human sign-off checklist complete (see spec Section 16) |

---

## 7. Async & Runtime Rules

- Async execution is **not allowed** unless the stage prompt explicitly requires it
- WebSocket and event-loop code is only permitted in Stage 01 (Data Ingestion) and Stage 12+ (Execution)
- Async stubs (function signatures or placeholders) are permitted when preparing
  for future WebSocket-based modules, subject to these constraints:
  - No implementation logic
  - No event loops
  - No runtime side effects
  - Must be clearly marked with `# STUB — implementation deferred to Stage N`

---

## 8. Code Quality Requirements

All AI-generated code MUST:
- Be production-grade — no shortcuts justified by "it's just a prototype"
- Include full Python type hints on all functions and class attributes
- Be deterministic and fully testable with mocked external dependencies
- Fail safely — raise typed exceptions; no silent failures; no bare `except`
- Avoid side effects on import
- Include validation for all critical data (especially contract fields)
- Use structured logging (`structlog` or `logging` with JSON formatter) — no `print()`
- Pass `mypy --strict` with zero errors
- Achieve 100% branch coverage on all non-stub modules before stage gate

Stubs, TODOs, or placeholder logic are **not allowed** unless explicitly
permitted by the stage prompt and marked `# PERMITTED STUB — Stage N`.

---

## 9. Logging & Secrets

- API keys, tokens, credentials, and account identifiers must **never** appear in:
  - Log output
  - Exception messages
  - Test fixtures
  - Comments
- All secrets must be sourced from AWS Secrets Manager or HashiCorp Vault
- Logging must be explicit, structured, and level-appropriate:
  - `DEBUG` — detailed diagnostic (disabled in production)
  - `INFO` — normal state transitions and events
  - `WARNING` — degraded execution quality, viability alerts
  - `ERROR` — circuit breaker triggers, API failures
  - `CRITICAL` — capital-at-risk events, system shutdown
- No debug prints in production code paths

---

## 10. When to Stop

The AI agent MUST **STOP immediately** and request human clarification if:

- A requirement in the stage prompt is ambiguous
- The stage prompt conflicts with `docs/specification.md`
- A requested change touches strategy logic, risk parameters, or execution behaviour
- A task could expose capital or enable live trading
- A contract schema field cannot be implemented without adding an assumption
- Any gate criterion for the current stage cannot be met within the allowed file scope
- The specification contains a threshold described qualitatively without a numeric value
  (check the Threshold Appendix, spec Sections 17.1–17.6 first)

**Do not interpret. Do not resolve ambiguity silently. Stop and escalate.**

---

## 11. Stage Gate Enforcement

Each stage has binary pass/fail gate criteria defined in its prompt.  
No stage begins until the preceding stage's gates all pass.

AI agents MUST NOT:
- Begin Stage N+1 work within a Stage N prompt
- Partially implement a future stage "to save time"
- Skip gate criteria on the grounds that a change is "obviously safe"

Human approval is required at every stage gate before proceeding.

---

## 12. Enforcement

Violation of any rule in this document is grounds for:
- **Immediate rejection** of the change
- **Manual rollback** of all files in the affected stage
- **Task restart** under stricter supervision
- Escalation to human review of all preceding stage outputs

---

**This document is mandatory.  
Compliance is required before any code is written, reviewed, or merged.  
When in doubt, stop and ask.**
