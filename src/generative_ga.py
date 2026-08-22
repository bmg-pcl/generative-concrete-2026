"""
generative_ga.py - A simple, transparent generative model for inverse mix design.

This is the honest first-step generator described in docs/FIX_PLAN.md (Phase 3).
It replaces the old noise sampler in bayesian.py, which returned the *same* cloud
regardless of the requested strength.

The idea is deliberately easy to follow:

    1. We already have a trained forward model:  strength = predict(mix).
    2. To design a mix for a TARGET strength, we search for mixes whose predicted
       strength is close to the target -- i.e. we minimise |predict(mix) - target|.
    3. We do that search with the existing GeneticOptimizer (a small GA), over
       bounds CLAMPED TO THE TRAINING DATA ENVELOPE so candidates stay realistic
       (in-distribution) rather than extrapolating.
    4. The "generative" output is not a single answer but the top-K spread of the
       final population -- a set of distinct mixes that all hit the target. That
       spread is our transparent stand-in for a posterior.

No TensorFlow, no black boxes: everything here is a GA over a known objective.

Optional compliance constraint (spec docs/specs/R8.2-exposure-compliance.md, WP-3
item 5): a caller may pass `compliance=(pack_id, class_id)` to `design()` /
`sample()` / `best_mix()` to bias the search toward an exposure class. This is
OFF BY DEFAULT (`compliance=None`) and, when off, executes none of the new code
paths below -- the default objective is untouched line-for-line, so default runs
stay bit-identical to before this feature existed (see tests/test_generative.py's
`test_default_path_is_bit_identical_to_no_compliance`).

UNKNOWN handling decision (must be justified per WP-3b instructions): for the
*optimizer's* penalty/constraint, an UNKNOWN rule counts as a VIOLATION, not a
pass. This mirrors compliance.py's own discipline ("a rule that cannot be
evaluated is not a pass") and, critically, closes an exploit an optimizer is
uniquely positioned to find: a GA/NSGA search that treated UNKNOWN as "free"
would actively steer toward whatever a pack happens to leave unsourced (e.g. an
SCM fraction no jurisdiction pack currently states) to relax the search, and
would do so *silently* -- the population would drift toward exactly the gap in
the reference data with nothing in the objective signalling it. Treating UNKNOWN
as violated is the conservative choice: it means a class most of whose rules are
absent from a pack (e.g. every en206.json class, which omits max_scm_fraction
everywhere) may never reach a true PASS verdict via this engine, only FAIL or
UNKNOWN -- but "we can't certify this against an incomplete pack" is the honest
answer, not a bug to route around. `design_compliant()` reports this honestly:
it returns `found=False` rather than presenting the closest candidate as if it
had passed.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ga import GeneticOptimizer
from .models import StrengthPredictor
from .chemistry_simple import calculate_embodied_carbon
from .data_fetcher import load_data
from .physical import volume_error, enforce_volume, VOLUME_TOLERANCE
from .compliance import check_compliance, load_packs

# Canonical parameter order (matches the UCI columns and StrengthPredictor inputs).
PARAM_NAMES: List[str] = [
    "cement", "slag", "ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age",
]

# Robust-mode out-of-support penalty weight (MPa-scale). See docs/specs/R1.
OOS_PENALTY_WEIGHT = 10.0
# Volume-balance penalty weight (MPa-scale per m³ of imbalance). See docs/specs/R2.
VOLUME_PENALTY_WEIGHT = 200.0
# Compliance-constraint penalty weight (MPa-scale). See docs/specs/R8.2, WP-3 item 5.
#
# Unlike OOS/VOLUME above, the quantity this multiplies (`compliance_violation()`)
# is already a *normalised* magnitude -- roughly the fractional amount each failing
# or unknown rule misses its limit by, summed across rules -- not a raw physical
# unit, so its weight is chosen relative to the other two weights' *roles* rather
# than to shared units. VOLUME_PENALTY_WEIGHT (200) exists to make a small, always-
# on physical infeasibility (a mix that doesn't fill ~1 m³) dominate the search
# regardless of how well it hits the strength target. A requested exposure class is
# the same kind of hard, non-negotiable requirement -- a design that fails it is not
# deliverable no matter how close its strength match is -- so COMPLIANCE_PENALTY_WEIGHT
# is set to sit in that same dominant regime: large enough that even one violated
# rule outweighs the ~0-70 MPa range of plausible target-strength errors, but far
# above OOS_PENALTY_WEIGHT (10), which only has to *nudge* the search away from
# extrapolated territory, not exclude it outright.
COMPLIANCE_PENALTY_WEIGHT = 150.0


def data_envelope(param_names: List[str] = PARAM_NAMES) -> np.ndarray:
    """
    Per-parameter (min, max) bounds taken from the training data.

    Searching inside this envelope keeps generated mixes in-distribution, which is
    exactly what the old inverse planner failed to do (it emitted water below the
    dataset minimum, forcing the forward model to extrapolate).
    """
    df = load_data()
    return np.array([(df[name].min(), df[name].max()) for name in param_names], dtype=float)


def resolve_compliance_target(compliance: Optional[Tuple[str, str]]) -> Tuple[Optional[dict], Optional[str]]:
    """Resolve an optional `(pack_id, class_id)` pair to `(pack_dict, class_id)`.

    Returns `(None, None)` when `compliance` is `None` -- the off-by-default case
    every default-path call goes through. Uses `load_packs(include_hidden=True)`:
    the optimizer is an engine, not the UI's jurisdiction picker, so the hidden
    `_fixture` pack (used by this module's and nsga's own tests) must resolve
    exactly like a real jurisdiction pack -- `load_packs()`'s default hidden-pack
    exclusion is a presentation concern for the UI dropdown, not an engine one.
    """
    if compliance is None:
        return None, None
    pack_id, cls = compliance
    packs = load_packs(include_hidden=True)
    if pack_id not in packs:
        raise KeyError(f"unknown compliance pack '{pack_id}' (available: {sorted(packs)})")
    pack = packs[pack_id]
    if cls not in pack.get("classes", {}):
        raise KeyError(f"pack '{pack_id}' has no class '{cls}' "
                        f"(available: {sorted(pack.get('classes', {}))})")
    return pack, cls


def compliance_violation(mix: Dict[str, float], pack: dict, cls: str, strength_lo: float,
                          *, unknown_counts_as_violation: bool = True) -> float:
    """Scalar violation magnitude (>= 0; 0 means fully compliant) for one mix
    against one exposure class, for use as a penalty term (GA) or a pymoo
    inequality constraint value (NSGA -- see nsga.py's MixDesignProblem, which
    imports this function so both optimizers score compliance identically).

    Always checks strength against `strength_lo` (the conformal lower bound),
    never a mean -- this is R8.2's core design decision, applied identically
    whether or not the caller's optimizer run is in "robust" mode (see the
    module docstring's "Optional compliance constraint" note).

    Each FAILing rule contributes the fractional amount by which it misses its
    limit (e.g. ~0.10 for a value 10% over a max, or 10% under a min), so a
    near-miss is penalised less than a gross violation. Each UNKNOWN rule
    contributes a fixed unit penalty when `unknown_counts_as_violation` is True
    (the default -- see the module docstring's "UNKNOWN handling decision").
    """
    result = check_compliance(mix, pack, cls, strength_lo=strength_lo)
    if result["verdict"] == "PASS":
        return 0.0
    total = 0.0
    for rule in result["rules"]:
        if rule["result"] == "FAIL":
            required, actual = rule["required"], rule["actual"]
            if required:
                total += abs(actual - required) / abs(required)
            else:
                total += 1.0
        elif rule["result"] == "UNKNOWN" and unknown_counts_as_violation:
            total += 1.0
    return total


class _InverseDesignerBase:
    """
    Shared plumbing for metaheuristic inverse designers.

    A subclass only has to implement `design()` -- run some optimizer and return the
    final population ranked best-first. Everything else (the objective, turning the
    ranked population into a sample cloud, the single-best-mix convenience) lives
    here, so the GA and ACO variants stay tiny and identical apart from the search.
    """

    def __init__(
        self,
        predictor: Optional[StrengthPredictor] = None,
        param_names: List[str] = PARAM_NAMES,
        bounds: Optional[np.ndarray] = None,
        jitter_frac: float = 0.02,
    ):
        self.predictor = predictor or StrengthPredictor()
        self.param_names = param_names
        self.bounds = data_envelope(param_names) if bounds is None else np.asarray(bounds, dtype=float)
        self.n_dims = len(self.bounds)
        self._age_idx = param_names.index("age")
        # Jitter used when expanding the elite set into a sample cloud, as a
        # fraction of each parameter's range.
        self._jitter = jitter_frac * (self.bounds[:, 1] - self.bounds[:, 0])

    # -- objective -----------------------------------------------------------
    def _make_objective(self, target_strength: float, carbon_target: Optional[float],
                        robust: bool = False, compliance_pack: Optional[dict] = None,
                        compliance_cls: Optional[str] = None):
        """
        Returns a scalar error to MINIMISE:

            error = |strength(mix) - target_strength|                    (always)
                  + max(0, carbon(mix) - carbon_target)                  (only if carbon_target given)
                  + OOS_PENALTY_WEIGHT·max(0, novelty - thresh)          (robust only)
                  + COMPLIANCE_PENALTY_WEIGHT·compliance_violation(...)  (only if a
                                                                           compliance target given)

        With `robust=True`, `strength` is the conformal lower bound (the guaranteed
        strength) rather than the mean, and an out-of-support penalty pulls the search
        away from extrapolated regions the prediction can't be trusted in.

        When `compliance_pack`/`compliance_cls` are given (both None by default --
        this whole branch is skipped in that case, which is how the default path
        stays bit-identical), the compliance term always checks strength against the
        conformal lower bound, independent of `robust`: R8.2's core design decision
        is to never certify a spec-relevant pass on a mean prediction, so even a
        non-robust run (whose primary strength term above uses the mean) recomputes
        the interval for the compliance check specifically.
        """
        threshold = self.predictor.support_threshold() if robust else None

        def objective(theta: np.ndarray) -> float:
            if robust:
                lo, _, _ = self.predictor.predict_interval(theta)
                strength = float(lo[0])
            else:
                strength = self.predictor.predict(theta)
            error = abs(strength - target_strength)
            mix = dict(zip(self.param_names, theta))
            if carbon_target is not None:
                error += max(0.0, calculate_embodied_carbon(mix) - carbon_target)
            # Physical validity: penalise mixes that don't fill ~1 m³ (always on).
            error += VOLUME_PENALTY_WEIGHT * max(0.0, volume_error(mix) - VOLUME_TOLERANCE)
            if robust:
                nov = float(self.predictor.novelty(theta)[0])
                error += OOS_PENALTY_WEIGHT * max(0.0, nov - threshold)
            if compliance_pack is not None:
                if robust:
                    strength_lo_c = strength  # already the conformal lower bound above
                else:
                    lo_c, _, _ = self.predictor.predict_interval(theta)
                    strength_lo_c = float(lo_c[0])
                viol = compliance_violation(mix, compliance_pack, compliance_cls, strength_lo_c)
                error += COMPLIANCE_PENALTY_WEIGHT * viol
            return float(error)

        return objective

    def _effective_bounds(self, age: Optional[float]) -> np.ndarray:
        """Search bounds with the age dimension pinned to `age` (degenerate bound) when
        a fixed design age is requested, so the optimizer treats age as a condition, not
        a free variable it can exploit (e.g. prescribing a 365-day cure)."""
        if age is None:
            return self.bounds
        b = self.bounds.copy()
        b[self._age_idx] = (float(age), float(age))
        return b

    def _rank(self, optimizer, objective) -> Tuple[np.ndarray, np.ndarray]:
        """Collect an optimizer's final population + global best, ranked best-first."""
        best, _ = optimizer.get_best()
        population = np.vstack([best, optimizer.population])
        errors = np.array([objective(ind) for ind in population])
        order = np.argsort(errors)
        return population[order], errors[order]

    # -- to be provided by subclasses ---------------------------------------
    def design(self, target_strength: float, carbon_target: Optional[float] = None,
               compliance: Optional[Tuple[str, str]] = None,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- generative interface (shared) --------------------------------------
    def sample(
        self,
        target_strength: float,
        n_samples: int = 2000,
        carbon_target: Optional[float] = None,
        robust: bool = False,
        age: Optional[float] = None,
        compliance: Optional[Tuple[str, str]] = None,
    ) -> np.ndarray:
        """
        Produce an (n_samples, n_dims) cloud of target-conditioned mixes.

        We take the elite quarter of the final population and resample it with a
        small Gaussian jitter (clipped to the data envelope). This turns a handful
        of good solutions into a smooth spread suitable for the dashboard surface,
        while keeping every sample tied to the requested target.

        `compliance`, as everywhere in this module, is an optional `(pack_id,
        class_id)` pair (default `None` -- off) biasing the underlying search
        toward the requested exposure class; see `design_compliant()` if you need
        an honest yes/no on whether a compliant design was actually found rather
        than a biased-but-unverified cloud.
        """
        ranked, _ = self.design(target_strength, carbon_target, robust=robust, age=age,
                                 compliance=compliance)
        n_elite = max(10, len(ranked) // 4)
        elite = ranked[:n_elite]

        eb = self._effective_bounds(age)
        idx = np.random.randint(0, len(elite), n_samples)
        jitter = np.random.normal(0.0, 1.0, (n_samples, self.n_dims)) * self._jitter
        samples = elite[idx] + jitter
        samples = np.clip(samples, eb[:, 0], eb[:, 1])
        # Physical validity: repair any sample that violates the volume balance.
        samples = np.array([[enforce_volume(dict(zip(self.param_names, s)))[p]
                             for p in self.param_names] for s in samples])
        samples = np.clip(samples, eb[:, 0], eb[:, 1])
        if age is not None:
            samples[:, self._age_idx] = float(age)  # jitter/repair must not un-pin age
        return samples

    def best_mix(self, target_strength: float, carbon_target: Optional[float] = None,
                 robust: bool = False, age: Optional[float] = None,
                 compliance: Optional[Tuple[str, str]] = None) -> Dict[str, float]:
        """Single best mix as a dict -- for one-shot callers like inverse_plan_mix.
        Repaired to satisfy the volume balance if the search left it slightly off.
        See `sample()`'s docstring on `compliance` -- this is the search bias, not
        a verified pass; use `design_compliant()` for a checked answer."""
        ranked, _ = self.design(target_strength, carbon_target, robust=robust, age=age,
                                 compliance=compliance)
        return enforce_volume(dict(zip(self.param_names, ranked[0])))

    def design_compliant(
        self,
        target_strength: float,
        compliance: Tuple[str, str],
        carbon_target: Optional[float] = None,
        robust: bool = False,
        age: Optional[float] = None,
        **design_kwargs,
    ) -> Dict:
        """Run a compliance-constrained design and report HONESTLY whether a
        verified-compliant mix was actually found.

        `design()` minimises a *penalised* objective -- the compliance penalty
        pulls the population toward the requested class but is a soft bias, not a
        certificate. This wrapper re-checks every candidate in the final ranked
        population, best-first, with the real `check_compliance()` engine (always
        against the conformal lower bound, per R8.2), and returns the first one
        that is a true PASS. If none of the final population verifies as PASS --
        which, per this module's UNKNOWN-handling decision, includes any mix left
        with an UNKNOWN rule -- it returns `found=False` rather than presenting
        the closest (but unverified) candidate as compliant.

        Returns:
            {"found": bool, "mix": dict | None, "result": check_compliance dict | None,
             "strength_basis": "conformal_lower_bound", "pack_id": str, "class": str,
             "checked": [(mix, check_compliance result), ...]}  # every candidate examined
        """
        pack, cls = resolve_compliance_target(compliance)
        ranked, _ = self.design(target_strength, carbon_target, robust=robust, age=age,
                                 compliance=compliance, **design_kwargs)
        checked = []
        for theta in ranked:
            mix = enforce_volume(dict(zip(self.param_names, theta)))
            theta_repaired = np.array([mix[p] for p in self.param_names])
            lo, _, _ = self.predictor.predict_interval(theta_repaired)
            result = check_compliance(mix, pack, cls, strength_lo=float(lo[0]))
            checked.append((mix, result))
            if result["verdict"] == "PASS":
                return {"found": True, "mix": mix, "result": result,
                        "strength_basis": "conformal_lower_bound",
                        "pack_id": pack["pack_id"], "class": cls, "checked": checked}
        return {"found": False, "mix": None, "result": None,
                "strength_basis": "conformal_lower_bound",
                "pack_id": pack["pack_id"], "class": cls, "checked": checked}


class PopulationInverseDesigner(_InverseDesignerBase):
    """
    GA-based inverse designer: given a target strength (and optional carbon target),
    generate a spread of realistic mixes that achieve it.
    """

    def design(
        self,
        target_strength: float,
        carbon_target: Optional[float] = None,
        pop_size: int = 80,
        generations: int = 40,
        robust: bool = False,
        age: Optional[float] = None,
        compliance: Optional[Tuple[str, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the GA and return the final population sorted best-first.

        `compliance=(pack_id, class_id)` (default None -- off) adds a penalty
        term biasing the search toward that exposure class; see the module
        docstring and `design_compliant()`."""
        pack, cls = resolve_compliance_target(compliance)
        objective = self._make_objective(target_strength, carbon_target, robust=robust,
                                          compliance_pack=pack, compliance_cls=cls)
        optimizer = GeneticOptimizer(
            objective_fn=objective,
            bounds=self._effective_bounds(age).tolist(),
            pop_size=pop_size,
            maximize=False,  # we are minimising the target-match error
        )
        optimizer.run(generations)
        return self._rank(optimizer, objective)


class AntColonyInverseDesigner(_InverseDesignerBase):
    """
    ACO-based inverse designer: identical interface to the GA variant, but uses
    Ant Colony Optimization for continuous domains (ACO_R, see src/aco.py) as the
    search engine. Useful as an independent metaheuristic to compare against the GA.
    """

    def design(
        self,
        target_strength: float,
        carbon_target: Optional[float] = None,
        n_ants: int = 40,
        archive_size: int = 20,
        generations: int = 40,
        robust: bool = False,
        age: Optional[float] = None,
        compliance: Optional[Tuple[str, str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ACO_R and return the final solution archive sorted best-first.

        `compliance=(pack_id, class_id)` (default None -- off) adds a penalty
        term biasing the search toward that exposure class; see the module
        docstring and `design_compliant()`."""
        # Imported lazily so importing this module doesn't require the ACO engine.
        from .aco import AntColonyOptimizer

        pack, cls = resolve_compliance_target(compliance)
        objective = self._make_objective(target_strength, carbon_target, robust=robust,
                                          compliance_pack=pack, compliance_cls=cls)
        optimizer = AntColonyOptimizer(
            objective_fn=objective,
            bounds=self._effective_bounds(age).tolist(),
            n_ants=n_ants,
            archive_size=archive_size,
            maximize=False,
        )
        optimizer.run(generations)
        return self._rank(optimizer, objective)


if __name__ == "__main__":
    for name, cls in (("GA", PopulationInverseDesigner), ("ACO", AntColonyInverseDesigner)):
        designer = cls()
        print(f"=== {name} inverse designer ===")
        for target in (25.0, 45.0, 65.0):
            mixes, errors = designer.design(target, generations=40)
            best = mixes[0]
            achieved = designer.predictor.predict(best)
            print(
                f"  target={target:>4.0f} MPa -> achieved={achieved:5.1f} MPa "
                f"(err={errors[0]:.2f}) | cement={best[0]:.0f} slag={best[1]:.0f} "
                f"ash={best[2]:.0f} water={best[3]:.0f} age={best[7]:.0f}"
            )
