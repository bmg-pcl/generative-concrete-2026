# Low-Emission Cement & Concrete: Economics, Market Fragmentation, and Why Proponent Goals Stall

**Audience:** business/strategy readers deciding where a mix-design intelligence tool fits.
**Companion:** the technical response to this analysis is `docs/specs/R6-materials-platform.md`
(pluggable materials, carbon provenance, clinker-source differentiation).

Figures below are indicative, order-of-magnitude values from the public literature (IEA,
GCCA, WBCSD/CSI protocol, ICE database, RMI/ETC analyses). They are for framing, not for
reporting; anything used in an actual disclosure must come from supplier-specific EPDs —
which is itself one of this report's conclusions.

---

## 1. The stakes, in one paragraph

Concrete is the most-used man-made material on earth (~30 Gt/yr; ~14 bn m³), and cement —
its active ingredient — contributes roughly **7–8% of global CO₂ emissions**. About 60% of
a clinker kiln's emissions are *process* emissions from calcination (CaCO₃ → CaO + CO₂,
~0.52–0.53 t CO₂/t clinker) that no fuel switch can remove; the remaining ~40% is kiln
fuel. Unlike power or road transport, there is no single drop-in substitute: decarbonizing
concrete is a portfolio problem — less clinker per tonne of cement, less cement per m³ of
concrete, less concrete per structure, cleaner kilns, and capture on what remains.

## 2. The economics of low-emission concrete

### 2.1 The abatement ladder (cheapest first)

| Lever | Typical CO₂ reduction | Indicative cost today | Constraint |
|---|---|---|---|
| Mix optimization (this tool's territory) | 5–20% per m³ | **Negative to zero** — often saves money | Specification rules; risk aversion; information |
| Clinker substitution: SCMs (GGBS, fly ash) | 20–50% of binder EC | Low; SCMs often cheaper than clinker | **Supply-constrained** — fly ash falling with coal retirement; slag capped by (declining) blast-furnace iron |
| Portland-limestone & LC3-type cements | ~10% (PLC) to ~30–40% (LC3) | Near-parity | Standards acceptance; calcined-clay capacity build-out |
| Alternative kiln fuels (waste, biomass) | Up to ~40% of fuel term | Site-dependent, often modest | Permitting, fuel logistics |
| Kiln electrification / hydrogen | Fuel term → near zero (if clean power) | High, immature | Grid capacity & price; kiln redesign |
| CCUS on the kiln | 90%+ of gross stack | **$60–120/t CO₂**; roughly doubles cement production cost | Capex, transport & storage, policy support |
| Novel chemistries (alkali-activated, etc.) | Deep, case-specific | Variable | Standards, durability track record, precursor supply |

Two structural facts shape the market:

1. **The cheapest abatement is informational, not industrial.** A large share of concrete
   is over-specified and over-cemented relative to its performance requirement. Getting
   the *right* mix accepted — more SCM, less binder, right strength class, right age for
   the actual loading date — costs nothing to negative money. It is blocked by process,
   not by physics or price.
2. **The expensive abatement is invisible at project level.** Cement is ~10–15% of the
   cost of ready-mix concrete; concrete is a single-digit percentage of construction
   cost. Even the worst case — CCUS cement at roughly 2× cement price — lands around
   **≤1% on total project cost** (the well-known RMI/ETC result). A premium that is
   crushing at the cement plant's P&L is rounding error in the developer's pro forma.
   The problem is that *no contract in the chain passes that willingness-to-pay down to
   the kiln*.

### 2.2 The green premium paradox

The proponent (owner/developer/infrastructure agency) increasingly *wants* low-carbon
concrete — for SBTi commitments, PAS 2080 obligations, green financing covenants, Buy
Clean eligibility, or planning conditions. The premium they would rationally absorb
(≤1% of capex) vastly exceeds what suppliers need to fund the transition. Yet low-carbon
cement routinely fails to find buyers at any premium, because between the proponent's
balance sheet and the kiln sit three procurement events, each awarded on lowest
conforming price with carbon absent from the award criteria. This is not a technology
gap. It is a **market-design and information gap**.

## 3. Scope 1/2/3: who owns the emission vs who owns the target

| Actor | Their Scope 1 | Their Scope 2 | The concrete emission sits in their… |
|---|---|---|---|
| Cement producer | Calcination + kiln fuel (**the** emission) | Grinding/plant electricity | Scope 1 — they own the lever |
| Ready-mix producer | Truck/plant fuel (minor) | Plant electricity (minor) | **Scope 3 cat. 1** (purchased cement) |
| Contractor | Site plant/fuel (minor) | Site power | **Scope 3 cat. 1** (purchased concrete) |
| Proponent / owner | Negligible for materials | — | **Scope 3** (capital goods / purchased goods & services) |

The structure of the problem in one sentence: **the actor with the Scope 3 *target* is
three to four contracts away from the actor with the Scope 1 *lever*, and every contract
in between is a commodity transaction that strips the carbon signal out of the price.**

Three consequences:

- **Double-invisibility.** For the ready-mix producer and contractor, embodied cement
  carbon is Scope 3 — historically unreported or estimated from industry averages, so
  a genuinely cleaner clinker (captured, or hydro-electrified) earns no ledger credit
  anywhere downstream unless supplier-specific data (an EPD) travels with the material.
- **Averages destroy the incentive.** If a proponent's Scope 3 inventory uses regional
  average emission factors, buying the best cement in the country changes their reported
  number by exactly zero. Supplier-specific factors are what make procurement matter —
  which is why EPD infrastructure (EN 15804 / ISO 14025) is not paperwork; it is the
  price signal.
- **Boundary shifts change the story.** An electrified kiln moves the fuel term from
  the producer's Scope 1 to Scope 2 — near-zero on hydro or a clean PPA, not-zero on a
  fossil grid. A capture plant cuts gross Scope 1 but adds Scope 2 capture energy. Any
  tool that carries one number per material without provenance (boundary, energy source,
  capture assumptions) will mis-rank exactly the suppliers doing the most.

## 4. The fragmented value chain: design → procure → supply/R&D → utilise

### 4.1 The seams and their failure modes

```
DESIGN                PROCURE                 SUPPLY / R&D              UTILISE
structural engineer → contractor/QS        → ready-mix + cement      → owner/proponent
specifies the recipe   buys lowest            producer holds the        holds the carbon
or the performance     conforming bid         innovation                target & the value
```

**Design → Procure.** The engineer's incentive is liability minimization, not carbon.
Prescriptive ("deemed-to-satisfy") specification — minimum cement content, maximum w/b,
capped SCM fractions, 28-day strength regardless of actual loading age — is the safe
harbor. Every prescriptive clause removes a degree of freedom the supplier could have
used to cut carbon at no performance cost. The 28-day convention alone forces early
strength that demands clinker, when much of real construction loads members far later.

**Procure → Supply.** Concrete is tendered as a commodity line item. Carbon rarely
appears in award criteria; when the low-carbon mix costs 3% more per m³, it loses,
even though that premium is noise at project level (see 2.2). Short bid windows also
preclude the mix trials a supplier needs to derisk a novel design.

**Supply ↔ R&D.** Producers *have* low-carbon products and pipelines (blended cements,
calcined clay, capture projects). But R&D investment is throttled by demand uncertainty:
why scale calcined clay if prescriptive specs in your market cap SCM substitution at
levels the 1990s set? The demand signal the proponent would happily send never survives
the two seams above.

**Utilise (the proponent).** Holds the Scope 3 target, the green financing terms, and
the willingness to pay — and typically enters the process after the specification is
written and exits before the material is batched. The party with the goal has the least
contact with the decision.

### 4.2 Result

Each actor optimizes locally and rationally; the chain outcome is high-carbon concrete
that nobody individually chose. Fragmentation is not a metaphor here — it is the specific
mechanism by which a ≤1% project-level premium fails to purchase a 30–50% emission cut
that is technically available today.

## 5. Specification regimes: performance vs prescriptive, by jurisdiction

The single largest structural blocker is the specification regime. **Prescriptive
(design/recipe) specs** dictate ingredients (min cement content, max w/b, SCM caps).
**Performance specs** state required outcomes (strength class, durability/exposure
resistance, sometimes a carbon ceiling) and free the supplier to meet them however
their materials allow.

| Jurisdiction | Framework | De facto regime | Notes for low-carbon mixes |
|---|---|---|---|
| EU (general) | EN 206 + **national annexes** | Prescriptive-leaning | Deemed-to-satisfy tables (min binder, max w/b per exposure class) vary **by country**; the same low-carbon mix can be compliant in one member state and non-conforming next door |
| UK | BS 8500 | Mixed, comparatively progressive | Broad SCM allowances (high GGBS replacement long normalized); active work on removing minimum-cement-content logic; PAS 2080 pulls carbon into infrastructure procurement |
| US | ACI 318/301, ASTM C150/C595/**C1157**, state DOTs | Mostly prescriptive in practice | C1157 is a fully performance-based cement spec with historically weak adoption; PLC (Type IL) swept the market in the 2020s showing standards *can* move; DOTs remain a patchwork of recipe rules |
| Canada | CSA A23.1 | **Dual-track** | Offers an explicit performance alternative in which the supplier owns the mix — often cited as the model clause |
| Australia/NZ | AS 1379 / AS 3600 | Prescriptive-leaning | Special-class concrete allows performance negotiation; geopolymer/AAM pilots proceed outside the main standard |
| France | NF EN 206/CN + low-carbon labels | Prescriptive base + labels | National low-carbon concrete labels create a demand pull the base standard lacks |
| Nordics | EN 206 national annexes | Progressive on substitution | Higher SCM acceptance; public procurement carbon ceilings appearing |
| India / emerging | IS 456 etc. | Prescriptive | Where most future concrete will be poured; LC3 origin research but standards lag |

Policy overlays are converging on the same instrument — **a carbon number attached to
the material at procurement time**: EU CBAM (definitive phase from 2026) prices imported
clinker/cement carbon; ETS free-allocation phase-out prices domestic; US Buy Clean
(federal + state) sets EPD-based ceilings for public purchases; green building schemes
and lender covenants do the same privately. All of them presuppose *supplier-specific,
provenance-carrying carbon data* — reinforcing §3.

**National variation is itself a fragmentation cost.** A multinational supplier cannot
amortize one low-carbon product across markets; a global proponent cannot write one
concrete carbon standard into its projects. Each border resets the deemed-to-satisfy
table.

## 6. What closes the gap

1. **Performance specification with a carbon axis.** Specify strength-at-actual-loading-
   age, durability by exposure class, and an embodied-carbon ceiling per m³ — not a
   recipe. (CSA's performance alternative is the template.)
2. **EPD-grade, supplier-specific carbon accounting as the procurement currency** — with
   provenance: system boundary, kiln fuel, electricity source, capture. Averages
   neutralize procurement; provenance activates it.
3. **Early supplier engagement** — mix trials before tender, so the low-carbon option is
   a derisked bid, not a variation claim.
4. **Proponent-side mandates that survive the seams** — carbon ceilings written into the
   employer's requirements and passed down contractually (PAS 2080 pattern), so the
   engineer and the QS are executing the proponent's goal rather than filtering it out.
5. **Quantified trade-off tooling shared across the seams** — a common model in which the
   engineer, the buyer, and the supplier see the same strength/carbon/cost surface,
   the same uncertainty, and the same provenance. Fragmentation persists in contracts;
   it does not have to persist in information.

## 7. Implications for this product

This tool already occupies seam 5: one forward model, one carbon path, one Pareto surface
shared by design (Compare/Inverse Design), procurement framing (cost + carbon +
guaranteed-strength bound), supply (Calibration: retrain on the producer's own materials;
active learning tells the lab what to test next), and the proponent (mix tickets that
carry predictions, intervals, carbon breakdown, and the config that produced them).

The gaps this analysis exposes map directly onto the roadmap:

- **Carbon provenance is currently one scalar per material.** §3 and §5 show that is not
  procurement-grade: it cannot distinguish a captured clinker from a standard one, or a
  hydro-electrified kiln from natural gas, and it cannot carry an EPD.
  → `R6-materials-platform.md` (provenance-tagged factors; clinker source descriptor;
  EPD override).
- **Materials are hardcoded**, so regional SCMs and novel admixtures — exactly where the
  supply-side innovation is happening — require code changes to evaluate.
  → R6 pluggable material registry.
- **No headless interface** for procurement/batch workflows (evaluate N bids' mixes under
  one project config). → `R5-operability.md` CLI mode.

The strategic claim, stated plainly: the binding constraint on low-emission concrete is
neither chemistry nor cost but **the loss of information and incentive across four
contractual seams**. A tool that makes performance, carbon (with provenance), and cost
commensurable at every seam is attacking the constraint itself.
