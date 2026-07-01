# Roadmap Specs

Specifications for the four improvement phases identified in the post-rebuild critique
(2026-07-01). Each spec is self-contained: motivation, current state, design, acceptance
gates, and test plan. Implementation order and dependencies below.

| Spec | Title | Effort | Status |
|---|---|---|---|
| [R1](R1-trustworthy-numbers.md) | Trustworthy numbers: conformal intervals, calibrated novelty, robust optimization | ~1–2 d | Draft |
| [R2](R2-physical-validity.md) | Physical validity: fixed-age targets, volume balance, workability guard | ~1–2 d | Draft |
| [R3](R3-dormant-value.md) | Dormant value: active learning, tensile, mix-ticket export, session-export fix | ~1–2 d | Draft |
| [R4](R4-engineering.md) | Engineering: slider rebinding, app modularization, CI hardening, benchmarks | ~1 d | Draft |

## Ordering & dependencies

```
R1.1 (conformal intervals) ──► R1.3 (robust optimization uses the calibrated bound)
R1.2 (novelty threshold)   ──► R1.3 (in-support constraint uses the threshold)
R1.3 ──► R2.NSGA constraint plumbing (shared pymoo constraint mechanism)
R2.1 (fixed age) — independent, can start immediately
R3, R4 — independent of each other; R3.1 (active learning) benefits from R1 intervals
```

**Recommended sequence: R1 → R2 → (R3 ∥ R4).** R1+R2 are the difference between a
demo and a tool an engineer can lean on; the single most urgent item is R1.1's
coverage test, because the UI currently displays intervals whose central claim
("90%") is unverified.

## Delegation readiness (sonnet-class implementer)

Each spec ends with an **Implementation notes (sonnet-ready)** section covering the traps.
Summary of what to hand over cold vs. what needs care:

- **Ready as-is:** R1.2, R2.2, R2.3, R3.1, **R3.2 (easiest — start here)**, R3.3, R4.3, R4.4.
- **Do carefully, guidance in-spec:** R1.1 (CQR indexing — the coverage test is the net),
  R1.3 (keep function defaults `robust=False`), R4.2 (mechanical moves only).
- **Streamlit keyed-widget trap — read R4's note first:** R4.1 and R3.4. This is the most
  common failure (`StreamlitAPIException` on setting a widget key after instantiation); the
  working callback pattern is written out in `R4-engineering.md`.
- **Escalate, do not attempt:** the R2.1 *flow retrain* (2-D age conditioning). Ship the
  interim (route to GA when age is fixed); leave the retrain as a senior task.

The acceptance gates are the safety net throughout: implement, run the gate, iterate — do
not weaken a gate to make it pass.

## Conventions

- Every item lands with regression tests; a spec is "done" when its acceptance gates
  pass in CI (heavy TF tests may skip; everything else must run).
- Behavior-changing items update `docs/TECHNICAL_REPORT.md` and, where user-visible,
  `docs/WORKFLOW.md` in the same commit.
- Retrained artifacts (`models/*.json`, `models/support.npz`, `models/amortizer/`)
  are regenerated together and committed together, so they stay mutually consistent
  (see the staleness mechanism in `src/amortized.py`).
