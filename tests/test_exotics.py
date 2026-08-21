"""
Tests for exotic admixtures (src/exotics.py).

The key contract: exotics must NOT move predicted strength unless the switch is
explicitly enabled (the strength model knows nothing about them), while cost and
carbon always apply.

R7.5 WP-2: every exotic also carries a density_kg_m3 in the registry now (see
docs/specs/R7.5-chemistry-remediation.md WP-2), surfaced via
materials.exotic_densities_view() and consumed by physical.mix_volume's optional
`exotic` argument -- exercised in tests/test_physical.py and tests/test_materials.py.
"""
from src.exotics import (
    EXOTIC_ADMIXTURES,
    compliance_warnings,
    exotic_carbon,
    exotic_cost,
    exotic_strength_delta,
)
from src.materials import exotic_densities_view


def _mix(**kw):
    """A full exotic dict with the given overrides, rest zero."""
    d = {k: 0 for k in EXOTIC_ADMIXTURES}
    d.update(kw)
    return d


def test_strength_switch_off_by_default():
    exotic = _mix(silica_fume=50, nano_silica=5, steel_fiber=80)
    # Default and explicit-off must both be zero.
    assert exotic_strength_delta(exotic) == 0.0
    assert exotic_strength_delta(exotic, enabled=False) == 0.0


def test_strength_switch_on_applies_linear_estimate():
    exotic = _mix(silica_fume=50)  # factor 0.08 -> 4.0 MPa
    assert exotic_strength_delta(exotic, enabled=True) == 50 * EXOTIC_ADMIXTURES["silica_fume"]["strength_factor"]

    # Multiple exotics sum.
    exotic2 = _mix(silica_fume=50, nano_silica=5)
    expected = (50 * EXOTIC_ADMIXTURES["silica_fume"]["strength_factor"]
                + 5 * EXOTIC_ADMIXTURES["nano_silica"]["strength_factor"])
    assert exotic_strength_delta(exotic2, enabled=True) == expected


def test_empty_mix_is_zero_everywhere():
    empty = _mix()
    assert exotic_strength_delta(empty, enabled=True) == 0.0
    assert exotic_carbon(empty) == 0.0
    assert exotic_cost(empty) == 0.0


def test_carbon_and_cost_always_apply():
    exotic = _mix(graphene_oxide=1)  # carbon 50, cost 500
    assert exotic_carbon(exotic) == EXOTIC_ADMIXTURES["graphene_oxide"]["carbon_factor"]
    assert exotic_cost(exotic) == EXOTIC_ADMIXTURES["graphene_oxide"]["cost"]


def test_every_admixture_has_a_strength_factor():
    for name, props in EXOTIC_ADMIXTURES.items():
        assert "strength_factor" in props, f"{name} missing strength_factor"


def test_every_admixture_has_a_positive_density():
    """R7.5 WP-2: every exotic in the registry must declare a physically sensible
    density so mix_volume's `exotic` argument (src/physical.py) can account for its
    displaced volume -- without this, dosing an exotic silently breaks the 'fills
    1 m3' batchability guarantee (see docs/specs/R7.5-chemistry-remediation.md)."""
    densities = exotic_densities_view()
    for name in EXOTIC_ADMIXTURES:
        assert name in densities, f"{name} missing density_kg_m3"
        assert densities[name] > 0, f"{name} has a non-positive density"


# --- R8.0 WP-D2: compliance warnings ---------------------------------------------
def test_compliance_warnings_flags_dosed_calcium_chloride():
    exotic = _mix(calcium_chloride=5)
    warnings = compliance_warnings(exotic)
    assert len(warnings) == 1
    assert "ACI 318 ch. 19 chloride limits / EN 206" in warnings[0]
    assert "calcium_chloride" in warnings[0]


def test_compliance_warnings_empty_when_undosed_or_empty():
    assert compliance_warnings(_mix(calcium_chloride=0)) == []
    assert compliance_warnings({}) == []


def test_compliance_warnings_ignores_dosed_materials_without_restrictions():
    exotic = _mix(silica_fume=50, nano_silica=5)
    assert compliance_warnings(exotic) == []
