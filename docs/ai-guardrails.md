# AI Guardrails — Automated Trading Bot

Status: LOCKED  
Applies to: All AI agents (Trae, Copilot, Cursor, etc.)

This document defines **non-negotiable constraints** for any AI agent
working on this repository. The goal is to protect capital, correctness,
and architectural integrity.

---

## 1. Source of Truth

- The file **docs/specification.md** is the **single authoritative specification**
- It is **LOCKED** and must not be modified by any AI agent
- Any deviation, conflict, or ambiguity must be escalated to a human
- If instructions conflict, the specification always wins

---

## 2. Scope Control

AI agents MUST:
- Work only within the currently active Git branch
- Modify only files explicitly allowed by the current task/prompt
- Stop and ask if task scope is unclear

AI agents MUST NOT:
- Create, delete, rename, or merge branches
- Push directly to `main`
- Rewrite or squash git history
- Modify files outside the declared scope

---

## 3. Strategy & Risk Integrity

AI agents MUST NOT:
- Change trading strategy logic
- Modify risk parameters or defaults
- Adjust stop-loss, take-profit, or sizing rules
- Add assumptions, heuristics, or optimizations
- Introduce machine learning or statistical inference

All strategy and risk behavior must remain:
- Deterministic
- Rule-based
- Config-driven

---

## 4. Execution Safety

AI agents MUST NOT:
- Connect to live exchanges
- Place real or simulated orders
- Enable live trading paths
- Disable safety flags such as `DRY_RUN`

The system must default to:
- `DRY_RUN = True`
- No external side effects
- No capital exposure

---

## 5. Async & Runtime Rules

- Async execution is **not allowed** unless explicitly required by the task
- Async stubs (function signatures or placeholders) are permitted when preparing
  for future WebSocket-based modules
- Async stubs MUST contain:
  - No implementation logic
  - No event loops
  - No runtime side effects

---

## 6. Code Quality Requirements

All AI-generated code MUST:
- Be production-grade
- Include full Python type hints
- Be deterministic and testable
- Fail safely (no silent failures)
- Avoid side effects on import
- Include validation for all critical data

Stubs, TODOs, or placeholder logic are **not allowed**
unless explicitly permitted by the task.

---

## 7. Logging & Secrets

- Secrets (API keys, tokens, credentials) must never be logged or printed
- Logging must be explicit and structured
- No debug prints in production code

---

## 8. When to Stop

The AI agent MUST STOP and ask for clarification if:
- Requirements are ambiguous
- Specification conflicts with the prompt
- A requested change affects strategy, risk, or execution behavior
- A task could expose capital or enable live trading

---

## 9. Enforcement

Violation of any rule in this document is grounds for:
- Immediate rejection of the change
- Manual rollback
- Task restart under stricter supervision

---

**This document is mandatory.  
Compliance is required before any code is merged.**
