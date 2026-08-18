# Parallel agent delegation — workflow and retrospective

**Origin:** the R7.5 remediation fan-out (2026-08-16): four sonnet-class agents,
isolated git worktrees, disjoint file ownership, one orchestrator. All four packages
landed — final suite 161 passed / 1 skipped, zero regressions, zero merge conflicts —
but the run surfaced six operational failure modes that cost several hours of wall
clock and ~2× the token budget. This document is the critique of that run and the
workflow that falls out of it. `docs/WORKFLOW.md` is the *app's* workflow; this is
the *engineering process* for delegating a partitioned spec to concurrent agents.

The one-line summary: **the design-level machinery worked; the operational machinery
failed.** Ownership partitioning, frozen surfaces, and gate discipline produced four
clean packages with no cross-package breakage. Everything that went wrong was
logistics: workspace provisioning, CPU budgeting, test scheduling, and the
orchestrator's own reporting discipline.

---

## 1. Retrospective — what failed, with evidence

Each failure mode below is stated with its measured evidence and root cause, because
a retrospective that says "communication could be better" teaches nothing.

### F1. Every agent's worktree was missing the spec

**Evidence:** all four worktrees were based on `origin/main` (`784a50a`) because the
prior PR merged mid-session; the R7.5 spec existed only on the unmerged feature
branch. `test -f docs/specs/R7.5-*.md` failed in every worktree. The agents worked
from the condensed summaries in their prompts — and one prompt (WP-2) referred to
"the table of twelve density values" *without inlining it*, leaving that agent to
guess the exact values the spec existed to pin.

**Root cause:** the orchestrator assumed worktrees inherit the feature branch and
never verified. A referenced-but-unreadable document is worse than no document: the
agent believes it is spec-compliant while working from a paraphrase.

**Countermeasure:** push every artifact agents must read *before* dispatch, then
verify presence in each worktree immediately after spawn (a one-line `test -f`
loop). Never reference a document in a prompt without confirming the agent can open
it — or inline the load-bearing content (tables, pinned values) directly.

### F2. The thundering herd: 19 pytest processes on 4 cores

**Evidence:** load average peaked at 27 on a 4-core box. The full suite runs in
134 s uncontended; under the herd it took 988 s or never returned. Several runs sat
20+ minutes at 60–70 %CPU each, all thrashing.

**Root cause:** every agent was told to "establish a baseline first" — a correct
instruction per-agent that becomes 4 simultaneous full-suite runs plus the
orchestrator's own. The instruction failed to compose.

**Countermeasure:** the orchestrator runs the baseline **once**, before dispatch,
and hands the counts to each agent in its prompt. Agents run only their owned test
files while iterating. The single integration-gate full run belongs to the
orchestrator, at harvest time. Budget: at most ⌈cores/2⌉ suite-heavy processes
concurrently; on a 4-core box that means the fan-out width can be 4 but the *test*
concurrency must be ~2.

### F3. Agent stall loops: background-and-wait

**Evidence:** agents backgrounded a test run, ended their turn "waiting for the
notification," woke, found no result (the run had been starved or killed), and
re-armed — repeatedly. WP-1 spent 213k tokens and 161 tool calls over 84 minutes on
~0.5 d of work; WP-4, same scope class, spent 105k/61. Three of four agents ended at
least one turn in this state.

**Root cause:** a subagent that ends its turn waiting has no user to interject and
no guarantee its background child survives; under contention this becomes an
indefinite wake/stall loop.

**Countermeasure:** prompts must state the test policy explicitly: run tests in the
**foreground** with a timeout; never background-and-wait; if a targeted run exceeds
~5 minutes, kill it and *report contention* rather than waiting. Waiting is an
orchestrator behavior, not an agent behavior.

### F4. A confident, false incident report

**Evidence:** WP-3 reported its worktree "destroyed mid-session by an out-of-band
process," attributed the destruction to a specific commit on the main checkout, and
warned the sibling worktrees were at live risk. Direct inspection showed all sibling
worktrees intact with their changes present; nothing had been deleted by that
commit. (Its own worktree had been deregistered by the harness after it made zero
edits — a cleanup, not an attack.)

**Root cause:** the agent assembled a plausible causal narrative from correlated
timestamps and reported it as diagnosis, with evidence-shaped formatting ("evidence,
not speculation") that made it *more* convincing, not more true.

**Countermeasure:** treat every agent incident report as a hypothesis. Verify
against the filesystem and git before acting on it or relaying it upward. The
inverse also happened and deserves the same rule: the orchestrator briefly recorded
"WP-1 never touched the core file" from a stale mid-flight check — state must be
read at harvest time, not remembered from an earlier observation.

### F5. The orchestrator reported "verified" before the evidence existed

**Evidence:** the three-package integration was reported as verified while the full
suite cross-check had actually been killed — by the orchestrator's own cleanup
`pkill -f pytest`, which was aimed at the herd and took out the verification run
with it. The per-package targeted suites (26/28/37 passing) were real; the
cross-package claim was not yet.

**Root cause:** two errors compounding — an indiscriminate kill-sweep, and equating
"a run was started" with "the run completed."

**Countermeasure:** "verified" means the evidence artifact exists (a log tail with
counts). Kill-sweeps must select by working directory (`readlink /proc/PID/cwd`),
never by process name, so your own verification run survives. The integration
arithmetic check — baseline + Σ(new tests) = final count — is cheap and catches
silent test loss; run it every time (here: 141 + 20 = 161 ✓).

### F6. Scope creep, and a wrong correction of it

**Evidence:** WP-4 delivered its package plus drive-by modernization (import
resorting, `timezone.utc`→`UTC`, annotation style). The orchestrator reverted the
churn — and one revert (`str | None` → `Optional[str]`) was itself wrong, flagged
by ruff as the only new lint error in the tree. The agent had been right.

**Root cause:** agents default to leaving code "better than found"; orchestrators
default to reverting anything unfamiliar. Both are taste. Neither is evidence.

**Countermeasure:** prompts: "no drive-by modernization — your diff is your scope."
Integration: judge de-creep by measurable delta (lint rule codes vs the base,
behavior, test counts), not style preference. The lint-delta gate used here —
*no new rule codes vs main* rather than absolute clean — is the right shape whenever
the linter itself has drifted (this repo: unpinned ruff, 338 pre-existing errors on
main that fail CI before tests run).

---

## 2. What to keep — the machinery that worked

- **Partition by exclusive file ownership, not by finding.** Six of nine defects
  lived in one module; partitioning by finding would have serialised everything or
  guaranteed conflicts. Four packages, zero merge conflicts.
- **Frozen surfaces, named explicitly, with test files as tripwires.** WP-1 changed
  312 lines in the file *containing* the frozen carbon functions and left them
  bit-identical — because the prompt named the hazard ("you own the file they live
  in") and the gate ("these two test files must pass unmodified"). Verified
  numerically at integration (0.53 / 292.6 / 0.93), not just by trust.
- **Worktree isolation.** Concurrent commits cannot race; harvest is a clean
  `cherry-pick` (WP-1) or reviewed `diff | apply` (WP-2/4).
- **A live correction channel.** The missing density table (F1) was fixed by
  message before WP-2 committed guessed values — all twelve landed exactly per spec.
  Fan-out without a steering channel turns every dispatch error into a redo.
- **Gate discipline inherited from the spec:** unit, conservation, and
  frozen-surface checks only; never "tune until it matches the literature." No
  agent attempted coefficient-tuning. The spec saying *why* ("converts an
  honestly-wrong model into a dishonestly-right-looking one") appears to matter.
- **Serial integration.** One package at a time, targeted tests after each, full
  suite once at the end — any breakage attributes to exactly one package.
- **Agents reporting what the spec got wrong.** Explicitly requesting deviations
  ("do not paper over a problem to report success") produced the ruff-drift finding
  and the defensive-floor test correction instead of silent accommodation.

---

## 3. The workflow (v2)

### Phase 0 — Preflight (orchestrator, before any spawn)

1. **Publish the inputs.** Commit and push the spec and anything agents must read.
   Record the intended base SHA.
2. **Check the environment.** Venv exists, core imports pass, expected skips known
   (e.g. no-TF ⇒ amortized tests skip). A container reclaim mid-session wiped the
   venv once already; assume it can happen again.
3. **Run the baseline once.** Record pass/skip counts; put them in every prompt.
   Agents do not run baselines.
4. **Budget concurrency.** Fan-out width ≤ number of truly disjoint packages;
   test-run concurrency ≤ ⌈cores/2⌉. If those conflict, stagger dispatch.

### Phase 1 — Dispatch

5. **One agent per package, worktree isolation.** Each prompt carries: the exclusive
   file list; the frozen surfaces *with the ownership-hazard warning*; the test
   policy (targeted files only, foreground, timeout, report contention instead of
   waiting); the baseline counts; the no-push rule; and the report format — commit
   SHA, worktree path, before/after counts, gates met, measured key numbers,
   deviations, and anything the spec got wrong.
6. **Verify the workspaces immediately after spawn:** each worktree has the spec
   (`test -f`) and the right base (`git log -1`). Fix by copy + message *now* —
   minutes matter before agents commit to a wrong path.

### Phase 2 — Monitor

7. **Act on notifications; never poll.** On any "waiting"/stall report, read the
   worktree directly — `git status`/`diff` is ground truth; the agent's narrative
   is a hypothesis (F4).
8. **On any slowness signal, check load first** and kill orphaned runs *by cwd*
   (F5), leaving your own verification runs alive.
9. **Steer with messages, not respawns**, while the agent's context is still the
   cheapest asset you have. Stop an agent that has produced its diff and stalled;
   harvest its work rather than waiting for its self-report.

### Phase 3 — Integration (serial, orchestrator-owned)

10. **Harvest one package:** cherry-pick its commit if it made one; otherwise
    review-and-apply its diff.
11. **De-creep with evidence:** anything outside the owned-file scope or the
    package's purpose is reverted only if the lint/behavior delta supports the
    revert (F6).
12. **Gate the package:** its targeted tests plus the frozen-surface test files.
    Commit with the measured numbers in the message (the repo's established style).
13. **After the last package:** one full suite; the arithmetic check
    (baseline + Σ new = final); lint delta vs the base; push.
14. **PR:** if the branch's previous PR merged mid-session, rebase onto main and
    open a *new* PR — never stack on merged history.

### Phase 4 — Retrospective

15. If a new failure mode appeared, add it to §1 with its evidence. Add the
    effort-log row (`docs/specs/README.md`) with estimate vs actual, separating
    implementation time from failure-recovery time.

---

## 4. Calibration numbers from the R7.5 run

Recorded so the next estimate is a measurement, not a guess.

| Quantity | Value |
|---|---|
| Packages / agents | 4, fully concurrent (spec predicted ~0.7 d wall vs ~1.25 d serial) |
| Actual wall clock | ~5 h end-to-end; implementation itself roughly on-estimate, the rest was failure recovery (F1–F5) |
| Full suite, uncontended | 134 s (161 passed, 1 skipped) |
| Full suite, under load 27 | 988 s, or never completed |
| Peak contention | 19 pytest processes, 4 cores, load 27.5 |
| Token cost, stalled agent (WP-1) | 213k tokens, 161 tool calls, 84 min |
| Token cost, comparable non-stalled scope (WP-4) | 105k tokens, 61 tool calls |
| Cross-package breakage at integration | zero; frozen surfaces bit-identical |
| Dispatch errors caught by the correction channel | 2 (missing spec, missing density table) |
| False incident reports requiring verification | 1 (F4), plus 1 stale orchestrator observation |

The headline calibration: **the fan-out's parallelism saved roughly the time the
fan-out's logistics failures spent.** With Phase 0/1 done properly — baseline once,
workspaces verified, test policy explicit — the same four packages plausibly land in
~2 h wall. The design machinery needs no changes; the checklist above is where the
next multiple comes from.

**First validation of v2 (WP-5, 2026-08-18).** The remaining Wave B package ran
solo under this workflow: baseline handed in the prompt, the F1 workspace fix as an
explicit step 0 (the worktree base was verified good within seconds of spawn),
foreground-only targeted tests, no full-suite runs by the agent. Result: 96k tokens,
43 tool calls, ~6 minutes wall, zero stalls, zero corrections needed — versus 213k
tokens / 161 calls / 84 minutes for the comparable-scope stalled agent under v1. The
agent also correctly flagged a scope-boundary judgment call (the mix ticket's
pre-existing exotic reconciliation gap) instead of silently expanding scope to fix
it — the report-deviations clause doing its job.
