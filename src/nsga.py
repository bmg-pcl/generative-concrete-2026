"""
nsga.py - True multi-objective optimization of mix design with NSGA-II / NSGA-III.

Where the amortized flow (src/amortized.py) and the GA/ACO designers answer a
*conditional* question -- "give me mixes that hit strength X" -- NSGA answers the
*multi-objective* question: "what is the whole trade-off surface between strength,
carbon, and cost?" It returns a Pareto front, not a single-target cloud.

The two approaches compose (see docs/WORKFLOW.md):
  * The flow / GA is fast and target-conditioned; NSGA is slower but maps the whole
    front with no target and no scalarization weights.
  * The flow can WARM-START NSGA: seed its initial population with realistic,
    in-envelope mixes near a target so it converges faster and stays in-distribution.

Built on pymoo (https://pymoo.org). Imported lazily-guarded so the rest of the app
works without it.

Optional compliance constraint (spec docs/specs/R8.2-exposure-compliance.md, WP-3
item 5): `run_nsga(..., compliance=(pack_id, class_id))` (default `None` -- off)
adds ONE pymoo inequality constraint -- not a fourth objective, which would
change the front's dimensionality and invalidate every existing hypervolume
number. The constraint reuses `generative_ga.compliance_violation()` verbatim
(the same scoring `generative_ga`'s GA penalty uses -- one mechanism, not two),
and the front is always checked against the conformal LOWER bound of strength
(R8.2's core design decision), independent of `robust`. See
`generative_ga`'s module docstring for the UNKNOWN-handling decision this
constraint inherits (UNKNOWN counts as a violation).

R8.5 P1 (coherence contract, kept forever): `MixDesignProblem._evaluate`'s carbon
objective column is `carbon_for_mode(mix_dict(x), advanced, **carbon_kwargs)` for
every candidate -- the SAME call `ui_logic.compute_metrics`/`scalarized_fitness`
make -- so the front's carbon column is always the number the ticket would show
for that mix under the identical config, `transport_detail` included whenever
`carbon_kwargs` carries it (`self.carbon_kwargs` is forwarded unmodified via
`**`, never filtered to a subset of keys). See tests/test_nsga.py's parametrized
`test_p1_coherence_*` tests, this spec's durable artifact.

Optional slump-support constraint (spec docs/specs/R8.5-optimizer-capability-
integration.md, P3): `run_nsga(..., slump_target=<float>)` (default `None` --
off) adds ONE pymoo inequality constraint requiring every front member to lie
within the SLUMP model's OWN support envelope (R8.1 -- distinct from, and
narrower than, the strength envelope every other constraint here uses) --
mirroring the shape of the existing robust in-support constraint exactly
(novelty minus threshold, feasible <= 0). Unlike the GA/ACO objective in
`generative_ga._make_objective`, this constraint does NOT also add a soft
"match this cm value" term: NSGA is a targetless, whole-front search by
design (see the module docstring above -- "no target and no scalarization
weights"), so `slump_target`'s role here is only to SWITCH the constraint on;
the numeric value is not otherwise used inside the search. What the front
actually looks like at the end is verified honestly with `properties.
slump_estimate` (never inferred from the soft constraint alone), same
"never present the constraint's own bookkeeping as a verified answer"
discipline as the compliance block above -- see the `"slump"` key in
`run_nsga`'s return dict.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .generative_ga import (
    PARAM_NAMES, data_envelope, resolve_compliance_target, compliance_violation,
    SLUMP_SP_DOSING_NOTE,
)
from .ui_logic import mix_dict, carbon_term
from .chemistry_simple import calculate_mix_cost
from .physical import volume_error, VOLUME_TOLERANCE
from .compliance import check_compliance
from .properties import get_slump_model, slump_estimate, SLUMP_FEATURES

try:
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    _PYMOO_AVAILABLE = True
except Exception:  # pragma: no cover - only when pymoo is absent
    _PYMOO_AVAILABLE = False


def pymoo_available() -> bool:
    return _PYMOO_AVAILABLE


if _PYMOO_AVAILABLE:

    class MixDesignProblem(Problem):
        """3-objective mix design: MAXIMISE strength, MINIMISE carbon and cost.

        pymoo minimises, so strength enters as -strength. Decision variables are the
        8 mix parameters, box-bounded to the training-data envelope.
        """

        def __init__(self, predictor, bounds, advanced, costs, carbon_kwargs=None, robust=False,
                     compliance_pack=None, compliance_cls=None, robust_carbon=False,
                     slump_target=None):
            # Constraints: volume balance (always) + in-support (robust only)
            # + compliance (only when a compliance target is given) + slump
            # support (only when a slump target is given). This is a
            # CONSTRAINT, not a 4th objective -- deliberately, so front
            # dimensionality (and every existing hypervolume number) is unchanged.
            n_constr = (1 + (1 if robust else 0) + (1 if compliance_pack is not None else 0)
                       + (1 if slump_target is not None else 0))
            super().__init__(n_var=len(bounds), n_obj=3, n_ieq_constr=n_constr,
                             xl=bounds[:, 0], xu=bounds[:, 1])
            self.predictor = predictor
            self.advanced = advanced
            self.costs = costs
            self.carbon_kwargs = carbon_kwargs or {}
            self.robust = robust
            self.threshold = predictor.support_threshold() if robust else None
            self.compliance_pack = compliance_pack
            self.compliance_cls = compliance_cls
            # R8.5 P2: swaps the carbon OBJECTIVE COLUMN for its +1.96*sigma upper
            # bound (ui_logic.carbon_term) -- dimensionality unchanged, still 3
            # objectives (see the module docstring). Default False -- bit-identical.
            self.robust_carbon = robust_carbon
            # R8.5 P3: slump-support CONSTRAINT switch -- see the module docstring's
            # "Optional slump-support constraint" note. `None` (default) adds no
            # constraint and evaluates none of the code below -- bit-identical.
            self.slump_target = slump_target
            self.slump_model = get_slump_model() if slump_target is not None else None

        def _evaluate(self, X, out, *args, **kwargs):
            if self.robust:
                # Optimize the conformal lower bound (guaranteed strength).
                strength, _, _ = self.predictor.predict_interval(X)
            else:
                strength = self.predictor.predict_batch(X)
            carbon = np.array([carbon_term(mix_dict(x), self.advanced, self.carbon_kwargs,
                                           robust_carbon=self.robust_carbon) for x in X])
            cost = np.array([calculate_mix_cost(mix_dict(x), self.costs) for x in X])
            out["F"] = np.column_stack([-strength, carbon, cost])
            # Physical-validity constraint (<=0 feasible): the front is batchable by
            # construction. In robust mode, also require in-support.
            g_vol = np.array([volume_error(mix_dict(x)) for x in X]) - VOLUME_TOLERANCE
            constraints = [g_vol]
            if self.robust:
                constraints.append(self.predictor.novelty(X) - self.threshold)
            if self.compliance_pack is not None:
                # Compliance is always checked against the conformal LOWER bound
                # (R8.2's core design decision), independent of `robust` -- so
                # recompute the interval here even when the objective above used
                # the mean (non-robust mode).
                if self.robust:
                    strength_lo = strength  # already the lower bound above
                else:
                    strength_lo, _, _ = self.predictor.predict_interval(X)
                g_compliance = np.array([
                    compliance_violation(mix_dict(x), self.compliance_pack, self.compliance_cls,
                                          float(slo))
                    for x, slo in zip(X, strength_lo)
                ])
                constraints.append(g_compliance)
            if self.slump_model is not None:
                # Slump-support-ONLY constraint (see the module docstring's
                # "Optional slump-support constraint" note): novelty minus
                # threshold, feasible <= 0 -- same shape as the robust
                # in-support constraint above, against the SLUMP model's own
                # envelope (R8.1), not the strength model's.
                x_slump = X[:, [PARAM_NAMES.index(f) for f in SLUMP_FEATURES]]
                g_slump = self.slump_model.novelty(x_slump) - self.slump_model.support_threshold()
                constraints.append(g_slump)
            out["G"] = np.column_stack(constraints) if len(constraints) > 1 else constraints[0]

    class _FrontHistory(Callback):
        """Record the best value of each objective per generation for a convergence view."""

        def __init__(self):
            super().__init__()
            self.data["best_strength"] = []
            self.data["min_carbon"] = []
            self.data["min_cost"] = []

        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            self.data["best_strength"].append(float(-F[:, 0].min()))
            self.data["min_carbon"].append(float(F[:, 1].min()))
            self.data["min_cost"].append(float(F[:, 2].min()))


def _seed_sampling(seed_population: Optional[np.ndarray], pop_size: int, bounds: np.ndarray):
    """Build an initial (pop_size, n_var) population from an optional warm-start seed."""
    n_var = len(bounds)
    if seed_population is None or len(seed_population) == 0:
        return None  # let pymoo use its default random sampling
    seed = np.atleast_2d(np.asarray(seed_population, dtype=float))
    if len(seed) >= pop_size:
        return seed[:pop_size]
    # Pad with random in-envelope mixes so the initial population is full.
    n_pad = pop_size - len(seed)
    pad = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pad, n_var))
    return np.vstack([seed, pad])


def run_nsga(
    predictor,
    advanced: bool = False,
    costs: Optional[Dict[str, float]] = None,
    algorithm: str = "nsga2",
    pop_size: int = 60,
    n_gen: int = 40,
    seed_population: Optional[np.ndarray] = None,
    n_partitions: int = 12,
    random_seed: int = 1,
    param_names: List[str] = PARAM_NAMES,
    bounds: Optional[np.ndarray] = None,
    carbon_kwargs: Optional[dict] = None,
    robust: bool = False,
    age: Optional[float] = None,
    compliance: Optional[Tuple[str, str]] = None,
    robust_carbon: bool = False,
    slump_target: Optional[float] = None,
) -> Dict:
    """
    Run NSGA-II or NSGA-III and return the Pareto front.

    Args:
        algorithm: "nsga2" or "nsga3".
        seed_population: optional (m, 8) array of warm-start mixes (e.g. from the
            amortized flow at a target strength).
        n_partitions: NSGA-III reference-direction resolution (more = denser front).
        compliance: optional `(pack_id, class_id)` pair (default `None` -- off).
            When given, front members are constrained (not scored as a 4th
            objective -- the front stays 3-objective strength/carbon/cost) to
            satisfy that exposure class, checked against the conformal LOWER
            bound of strength. See the module docstring.
        robust_carbon: R8.5 P2 (default False, independent of `robust`). Swaps
            the carbon OBJECTIVE COLUMN for its +1.96*sigma upper bound
            (`ui_logic.carbon_term`) -- front dimensionality stays 3-objective
            (see the module docstring's coherence-contract note). Default False
            is bit-identical to before this flag existed.
        slump_target: R8.5 P3 (default `None` -- off). Adds ONE inequality
            constraint requiring every front member to lie within the SLUMP
            model's own support envelope (R8.1) -- see the module docstring's
            "Optional slump-support constraint" note. `None` skips the whole
            branch (bit-identical).

    Returns dict with the front mixes and their objective values (natural units),
    plus a per-generation convergence history, and `"carbon_basis"` ("point" or
    "upper_95", per `robust_carbon`) disclosing what the "carbon" column is.
    When `compliance` is given, also includes a `"compliance"` block reporting,
    per front member, the real `check_compliance()` verdict -- never inferred
    from the optimizer's soft constraint alone -- and an honest `"all_pass"` flag.
    When `slump_target` is given, also includes a `"slump"` block (same honesty
    discipline, verified with `properties.slump_estimate` on every front member,
    never inferred from the constraint's own bookkeeping) -- see its assembly
    below for the exact shape.
    """
    if not _PYMOO_AVAILABLE:
        raise ImportError("pymoo is not installed. `pip install pymoo` to use NSGA-II/III.")

    bounds = data_envelope(param_names) if bounds is None else np.asarray(bounds, dtype=float)
    if age is not None:  # pin age as a fixed design condition (not a free variable)
        bounds = bounds.copy()
        bounds[param_names.index("age")] = (float(age), float(age))
    compliance_pack, compliance_cls = resolve_compliance_target(compliance)
    problem = MixDesignProblem(predictor, bounds, advanced, costs,
                               carbon_kwargs=carbon_kwargs, robust=robust,
                               compliance_pack=compliance_pack, compliance_cls=compliance_cls,
                               robust_carbon=robust_carbon, slump_target=slump_target)
    sampling = _seed_sampling(seed_population, pop_size, bounds)

    if algorithm.lower() == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
        kwargs = {"ref_dirs": ref_dirs, "pop_size": max(pop_size, len(ref_dirs))}
        if sampling is not None:
            sampling = _seed_sampling(seed_population, kwargs["pop_size"], bounds)
            kwargs["sampling"] = sampling
        algo = NSGA3(**kwargs)
        algo_name = "NSGA-III"
    else:
        kwargs = {"pop_size": pop_size}
        if sampling is not None:
            kwargs["sampling"] = sampling
        algo = NSGA2(**kwargs)
        algo_name = "NSGA-II"

    callback = _FrontHistory()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=random_seed,
                   callback=callback, verbose=False)

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    # Report MEAN strength for display so the numbers match the Compare tab, even when
    # robust mode optimized the lower bound (-F[:,0] is the lower bound in that case).
    strength = predictor.predict_batch(X)
    order = np.argsort(strength)  # sort the front by strength for display
    hist = callback.data
    out = {
        "algorithm": algo_name,
        "mixes": X[order],
        "strength": strength[order],
        "carbon": F[order, 1],
        "cost": F[order, 2],
        "history": {
            "best_strength": hist["best_strength"],
            "min_carbon": hist["min_carbon"],
            "min_cost": hist["min_cost"],
        },
        "front_size": len(X),
        "compliance": None,
        # R8.5 P2: discloses what the "carbon" column above IS -- the point
        # total, or (robust_carbon=True) its +1.96*sigma upper bound.
        "carbon_basis": "upper_95" if robust_carbon else "point",
        "slump": None,
    }
    if compliance_pack is not None:
        # Verify HONESTLY with the real engine -- the constraint above pulled the
        # search toward compliance but pymoo can still return an infeasible front
        # (e.g. no feasible region was found within n_gen), so this must not be
        # inferred from the constraint alone. Always checked against the
        # conformal LOWER bound, per R8.2.
        X_ordered = X[order]
        strength_lo, _, _ = predictor.predict_interval(X_ordered)
        results = []
        all_pass = True
        for x, slo in zip(X_ordered, strength_lo):
            r = check_compliance(mix_dict(x), compliance_pack, compliance_cls,
                                 strength_lo=float(slo))
            results.append(r)
            if r["verdict"] != "PASS":
                all_pass = False
        out["compliance"] = {
            "pack_id": compliance_pack["pack_id"],
            "class": compliance_cls,
            "strength_basis": "conformal_lower_bound",
            "all_pass": all_pass,
            "results": results,
        }
    if slump_target is not None:
        # R8.5 P3: verify HONESTLY with properties.slump_estimate (the real,
        # per-mix support gate) -- the constraint above pulled the search
        # toward slump support but pymoo can still return a front with
        # infeasible members (e.g. no feasible region was found within
        # n_gen), same "never infer from the constraint's own bookkeeping"
        # discipline as the compliance block above. NSGA is targetless (see
        # the module docstring), so this discloses the REACHED slump values
        # and whether each is honestly in-support -- it does not itself
        # decide a single "found" the way recommend_recipe's single design
        # does; a caller wanting that can read `all_in_support` directly.
        X_ordered = X[order]
        slump_results = [slump_estimate(mix_dict(x)) for x in X_ordered]
        all_in_support = all(r["in_support"] for r in slump_results)
        out["slump"] = {
            "target": float(slump_target),
            "basis": "model",
            "all_in_support": all_in_support,
            "in_support_count": sum(1 for r in slump_results if r["in_support"]),
            "front_size": len(X_ordered),
            "results": slump_results,
            "note": SLUMP_SP_DOSING_NOTE if all_in_support else None,
        }
    return out


if __name__ == "__main__":
    from .models import StrengthPredictor

    for algo in ("nsga2", "nsga3"):
        out = run_nsga(StrengthPredictor(), algorithm=algo, pop_size=60, n_gen=25)
        print(f"=== {out['algorithm']} ===  front size {out['front_size']}")
        print(f"  strength {out['strength'].min():.1f}-{out['strength'].max():.1f} MPa | "
              f"carbon {out['carbon'].min():.0f}-{out['carbon'].max():.0f} | "
              f"cost ${out['cost'].min():.0f}-${out['cost'].max():.0f}")
