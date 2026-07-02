# R6 — Materials Platform: Pluggable Admixtures + Carbon Provenance

**Status:** Draft · **Effort:** ~2–3 d · **Depends on:** none hard; R5.2 (CLI) benefits;
motivated by `docs/BUSINESS_REPORT.md` §3/§5/§7 (provenance-grade carbon is the
procurement currency; hardcoded materials block regional/novel evaluation)

This spec is both the technical report answering two design questions and the
implementation roadmap:

- **Q1: Should each element (material) carry its own embodied-carbon value?**
- **Q2: How do we differentiate clinker from a carbon-captured plant vs a standard one,
  or a hydro-powered kiln vs natural gas?**

---

## 1. Current state (why adding a material is a code change)

Material knowledge is scattered across four hardcoded registries plus one JSON:

| Where | What it holds | Used by |
|---|---|---|
| `src/exotics.py :: EXOTIC_ADMIXTURES` | dosage bounds, carbon factor, cost, category, placeholder strength factor | Compare tab, metrics |
| `src/chemistry_simple.py :: CARBON_FACTORS` | A1–A3 carbon factor per core material (single scalar) | every carbon path |
| `src/chemistry_simple.py :: UNIT_COSTS` | $/kg per core material | every cost path |
| `src/physical.py :: DENSITIES` | particle densities for volume balance | volume constraint/repair |
| `data/oxide_compositions.json` | oxide compositions + clinker_factor per cement type | advanced tier (Bogue, clinker carbon) |

Adding one new SCM today means editing 3–5 files, and the carbon "value" is a bare
scalar: no system boundary, no region, no vintage, no source, no way to carry a
supplier EPD. `carbon_from_clinker` already decomposes clinker into calcination
(0.53 kg/kg) + kiln fuel (a single default parameter, 0.35) — the right skeleton,
not yet steerable by fuel type, electricity source, or capture.

## 2. Design: one material registry

### 2.1 `data/materials.json` + `src/materials.py`

One declarative record per material; everything above becomes a *view* of it.

```jsonc
{
  "schema_version": 1,
  "materials": {
    "ggbs_slag": {
      "name": "GGBS (blast-furnace slag)",
      "category": "scm",                 // binder | scm | aggregate | water | admixture | fiber | nano
      "density_kg_m3": 2900,
      "dosage": {"default": 0, "max": 360, "unit": "kg/m3"},
      "unit_cost": {"value": 0.08, "currency": "USD/kg", "source": "regional default"},
      "carbon": [                         // ordered candidates — see §3 resolution
        {"value": 0.052, "unit": "kgCO2e/kg", "boundary": "A1-A3", "region": "EU",
         "vintage": 2023, "source": "database", "reference": "ICE v3", "uncertainty": 0.3}
      ],
      "chemistry": {"oxides_key": "SLAG", "reactivity_class": "latent-hydraulic"},
      "water_demand_delta": 0.0,          // kg water per kg material (heuristic hook)
      "strength_treatment": "trained_feature"   // see §5
    }
  }
}
```

`src/materials.py` exposes `get_material(key)`, `all_materials(category=...)`, and
**back-compat views** so nothing downstream changes in R6.1:
`carbon_factors_view() -> {key: float}`, `unit_costs_view()`, `densities_view()`,
`exotics_view()`. The four legacy dicts become deprecated re-exports of these views —
the single-source-of-truth moves without a big-bang refactor.

### 2.2 How you add a new admixture (the user-facing recipe)

1. Append one record to `data/materials.json` (name, category, density, dosage bounds,
   cost, at least one carbon record, `strength_treatment`).
2. If it participates in advanced chemistry, add its oxide row to
   `data/oxide_compositions.json` and reference it via `chemistry.oxides_key`.
3. Run `pytest tests/test_materials.py` — schema validation tells you what is missing.
4. It appears in the UI (Compare dosing expander, Config factors editor) and in every
   carbon/cost path with **no code change**. Strength participation follows §5.

## 3. Q1 — should every element have an embodied-carbon value?

**Yes — every material record must carry at least one carbon record, but never a bare
scalar.** A single number cannot answer "compared to what, measured how, valid where?",
and (per `BUSINESS_REPORT.md` §3) an unprovenanced number actively *mis-ranks* the
cleanest suppliers.

Each carbon entry is a record: `value, unit, boundary (A1–A3 default), region, vintage,
source ∈ {epd | database | parametric | placeholder}, reference, uncertainty`.

**Resolution order** (first match wins), applied by one function
`resolve_carbon(material, project_ctx)`:

1. **Project override** — the user typed a factor in Config (today's behavior, kept).
2. **Supplier EPD** attached to the project (source = `epd`). Measured beats modeled.
3. **Regional database** record matching the project region (source = `database`).
4. **Parametric model** — computed from a process descriptor (clinker: §4).
5. **Placeholder default** — allowed, but the record says so.

The **mix ticket prints the resolved source per material** (`carbon_kgCO2,cement,...,
source=epd:EPD-XYZ-2025`). A ticket that discloses "placeholder" is honest; a ticket
that hides it is a liability. This single line is what makes the tool's output
EPD-conversation-ready rather than indicative-only.

Total mix carbon remains `Σ mass_i × resolve_carbon(i) + transport(A4)` — unchanged
math, provenance-aware inputs.

## 4. Q2 — differentiating clinker sources (capture, kiln fuel, electricity)

One EPD, when it exists, settles the question — rule 2 above. The parametric model
(rule 4) is for the majority case where it does not exist yet, and for *what-if*
analysis ("what does my mix look like if the cement comes from Brevik-style capture?").

### 4.1 Clinker source descriptor

Generalize `carbon_from_clinker` from `(clinker_factor, kiln_fuel_carbon=0.35)` to a
descriptor stored per cement material (and editable in Config):

```jsonc
"clinker_source": {
  "clinker_factor": 0.95,          // OPC 0.95, PLC ~0.85, LC3 ~0.50 (existing JSON)
  "kiln_fuel": "natural_gas",      // coal | petcoke | natural_gas | alt_waste | biomass | electric
  "electricity": "grid_EU",        // grid_<region> | hydro | nuclear | ppa_renewable
  "capture": {"rate": 0.0,          // fraction of stack CO2 captured (process + combustion)
               "energy_kwh_per_tCO2": 0}   // capture energy demand, charged to `electricity`
}
```

Per-kg-clinker carbon becomes:

```
process      = 0.525                                   # calcination — chemistry, fuel-independent
combustion   = FUEL_EF[kiln_fuel]                      # coal ~0.40, petcoke ~0.42, natgas ~0.28,
                                                       # alt_waste ~0.15*, biomass ~0.05*, electric 0
electric_kwh = KILN_KWH[kiln_fuel] + capture.rate>0 ? capture_energy : 0
scope2       = electric_kwh × GRID_EF[electricity]     # hydro ≈ 0.004, EU grid ~0.25 kg/kWh, ...
clinker_EC   = (process + combustion) × (1 − capture.rate) + scope2
cement_EC    = clinker_EC × clinker_factor + grinding_electricity + minor_constituents
```

(*accounting conventions for waste/biomass fuels differ by jurisdiction — the record's
`reference` field says which convention the factor uses.)

### 4.2 What this expresses that today's model cannot

| Scenario | kiln_fuel | electricity | capture | Approx. clinker EC (kg/kg) |
|---|---|---|---|---|
| Standard OPC, coal kiln | coal | grid | 0 | ~0.93 |
| Standard OPC, natural gas | natural_gas | grid | 0 | ~0.81 |
| Same plant + 90% CCUS, grid-powered capture | natural_gas | grid_EU | 0.90 | ~0.14 |
| Same + hydro-powered capture | natural_gas | hydro | 0.90 | **~0.08** |
| Electrified kiln on hydro | electric | hydro | 0 | **~0.53 → mostly irreducible calcination** |
| Electrified kiln on hydro + capture | electric | hydro | 0.90 | ~0.06 |

The table *is* the Scope discussion made mechanical: fuel switching moves emissions
within Scope 1; electrification moves them from the producer's Scope 1 to Scope 2 —
where the electricity source decides whether they vanish (hydro) or merely relocate
(fossil grid); capture cuts gross stack emissions but charges its energy penalty to
Scope 2. The descriptor keeps those effects distinguishable instead of collapsing them
into one editable scalar; the resolved record carries the assumptions onto the ticket.

The existing "OPC vs LC3" selector becomes a preset over this descriptor, not a
separate mechanism.

## 5. Strength-model treatment (how a new material affects predictions)

The XGBoost predictor has a **fixed 8-feature input** (`PARAM_NAMES`) trained on UCI
data. A registry entry cannot honestly inject a new feature. Three declared tiers:

- `trained_feature` — the material *is* one of the 8 model inputs (cement, slag, ash,
  water, SP, aggregates, age). Full prediction support.
- `delta_estimate` — opt-in linear placeholder (today's exotics switch, unchanged
  semantics and disclaimer). The registry field replaces the hardcoded
  `strength_factor`.
- `inert` — cost/carbon/volume only; never moves strength.

Promotion path from `delta_estimate` to real support is the **Calibration tab**: upload
lab results including the new material, which (future work, noted not promised) extends
the feature set and retrains — the registry field then flips per material. Until then
the tiering keeps the honesty guarantee: unvalidated effects are opt-in and labelled.

## 6. Roadmap slices

| Slice | Scope | Acceptance gates |
|---|---|---|
| **R6.1 Registry + views** | `data/materials.json`, `src/materials.py`, schema validation; legacy dicts become views; migrate existing 7 core + 12 exotic materials verbatim | All existing tests green with dicts replaced by views; `test_materials.py` schema test; grep gate: no literal factor tables outside `materials.py`/JSON |
| **R6.2 Carbon provenance** | carbon records + `resolve_carbon` + resolution order; ticket prints per-material source; Config override wires to rule 1 | Resolution-order unit tests (EPD beats database beats parametric beats placeholder); ticket shows `source=`; totals unchanged for the default library (regression) |
| **R6.3 Clinker source descriptor** | descriptor schema; generalized `carbon_from_clinker`; Config UI (fuel/electricity/capture selectors); OPC/LC3 become presets | Worked-example table in §4.2 reproduced by tests (±10%); capture energy charged to the selected electricity EF; Scope1/Scope2 split reported in the advanced breakdown |
| **R6.4 Pluggable UI + docs** | Compare dosing expanders and Config factor editors generated from the registry; "add a material" walkthrough in docs; new-material end-to-end test (add via JSON in a tmp dir → appears in metrics) | AppTest: registry-added material shows in UI and moves carbon/cost; docs recipe verified by the end-to-end test |

Order: R6.1 → R6.2 → (R6.3 ∥ R6.4).

## 7. Out of scope

- A2/A4/A5 lifecycle modules beyond the existing transport heuristic (A4) — boundary
  extensions ride on the record schema later.
- Automatic EPD ingestion (PDF/ILCD parsing) — records are entered manually first.
- Retraining the strength model with new features (promotion path noted in §5;
  separate senior task).

## Implementation notes (sonnet-ready)

- **R6.1 is a mechanical migration** — copy values exactly; the regression gate is that
  every existing carbon/cost/density number is bit-identical through the views. Do not
  "improve" factors while moving them.
- **Keep `resolve_carbon` pure** (material record + project ctx in, record out) so it is
  trivially testable and the ticket can print the winning record's fields.
- **R6.3 trap:** capture applies to `(process + combustion)`, *not* to Scope 2 capture
  energy — the penalty term must survive capture. The §4.2 table tests exist to catch
  exactly this inversion.
- The Config carbon-factor editor currently writes scalars into
  `st.session_state.carbon_factors`; in R6.2 that becomes rule-1 override records —
  keep the session-export schema versioned (bump to v3) and accept v2 imports.
