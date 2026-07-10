# Concrete thinking

**The cheapest gigatonne of carbon abatement on earth is hiding in the small print of
a concrete specification. Nobody's contract tells them to go and get it.**

*A briefing in the style of a certain Anglophone weekly. Figures are indicative,
order-of-magnitude values from the public literature (IEA, GCCA, WBCSD/CSI, the ICE
database, RMI/ETC analyses) — for framing, not for reporting. Anything destined for a
disclosure must come from supplier-specific EPDs, which, as it happens, is one of this
briefing's conclusions. The technical response is `docs/specs/R6-materials-platform.md`.*

---

If cement were a country, only China and America would out-pollute it. Mankind pours
some 30 billion tonnes of concrete a year — the most-used material on the planet after
water — and the grey powder that glues it together accounts for 7–8% of global CO₂.
Worse, most of that carbon is not a fuel choice but a chemical fact: three-fifths of a
kiln's emissions come from calcination, the reaction that turns limestone into clinker
and releases about half a kilogramme of CO₂ for every kilogramme made. No windmill
fixes stoichiometry.

Yet the industry's dirty secret is not that decarbonising concrete is hard. It is that
a large slice of it is nearly free, and the market has organised itself, with some
care, not to buy it.

## Pour decisions

The abatement ladder starts below zero. Much of the world's concrete is over-engineered:
more cement than the job requires, strength specified for 28 days when the structure
will not bear load for six months, supplementary materials (slag, fly ash, calcined
clay) capped at fractions set decades ago. Simply designing the right mix — the
territory of the tool this repository contains — cuts 5–20% of a pour's emissions and
usually saves money. The next rungs cost little: Portland-limestone cements (which
quietly conquered America in the 2020s) trim ~10%; LC3-style blends offer 30–40% at
near price-parity, if the standards allow them in. Only at the top of the ladder does
it get expensive: carbon capture on the kiln runs $60–120 per tonne of CO₂ and roughly
doubles the cost of making cement.

"Doubles" sounds ruinous. It is not, and here is the arithmetic that makes the whole
subject interesting. Cement is a tenth or so of the cost of ready-mixed concrete;
concrete is a single-digit share of a construction budget. Even gold-plated,
fully-captured cement adds less than 1% to the cost of a building — the well-known
Rocky Mountain Institute result. A premium that terrifies a cement producer's P&L is a
rounding error in a developer's pro forma. Call it the green-premium paradox: the
willingness to pay exists at the top of the chain, the ability to abate exists at the
bottom, and every contract in between is written to make sure they never meet.

## Scope creep, or the lack of it

Carbon accounting explains why. Under the GHG Protocol's three scopes, the kiln's
emissions are the cement-maker's Scope 1 — the only party with its hand on the lever.
To the ready-mix producer buying that cement, the same tonnes are Scope 3, category 1.
To the contractor buying the concrete, Scope 3 again. To the developer or
infrastructure agency — the "proponent" whose net-zero pledge, green bond covenant or
planning condition set the whole thing in motion — they are Scope 3, three or four
contractual arm's-lengths away. The actor with the target is separated from the actor
with the lever by a chain of commodity transactions, each awarded on lowest conforming
price, each stripping the carbon signal out like a customs officer confiscating
contraband.

Averages finish the job. Where a proponent's Scope 3 inventory is built on
industry-average emission factors — as most still are — buying the cleanest cement in
the country changes the reported number by precisely nothing. Supplier-specific
declarations (EPDs, under EN 15804) are routinely dismissed as paperwork; they are in
fact the price signal, the only mechanism by which a captured clinker or a
hydro-powered kiln is worth more than its filthy twin. And the ledger is fussy:
electrify a kiln and the fuel emissions do not vanish but migrate to Scope 2, where
their fate depends on whether the electrons come from a Norwegian fjord or a lignite
grid. A tool that carries one unprovenanced number per material will mis-rank
precisely the suppliers trying hardest.

## Set in their ways

Follow the value chain and the seams appear on cue. The structural engineer, paid to
minimise liability rather than carbon, reaches for the prescriptive specification —
minimum cement content, maximum water-binder ratio, SCM caps, the 28-day strength
ritual — because "deemed to satisfy" is a safe harbour and innovation is a deposition
waiting to happen. The quantity surveyor then tenders concrete as a commodity line
item; when the low-carbon mix bids 3% more per cubic metre, it loses, the project's
≤1% tolerance for green premium notwithstanding, because nobody put carbon in the
award criteria. The supplier, sitting on blended cements and a calcined-clay pipeline,
sees the demand signal die two seams upstream and throttles its R&D accordingly. And
the proponent — owner of the target, the financing terms and the willingness to pay —
enters after the specification is frozen and exits before the first truck is batched.
Four rational actors, one irrational outcome. Nobody chose the high-carbon pour;
everybody's contract did.

The specifications deserve their own indictment, because they are national, various
and proudly incompatible. Europe's EN 206 delegates the crucial tables to national
annexes, so an identical low-carbon mix can be legal in one member state and
non-conforming across the border. Britain's BS 8500 is comparatively permissive on
slag and is inching towards abolishing minimum cement content — the single most
carbon-hostile clause in world construction. America's ASTM C1157, a fully
performance-based cement standard, has existed since the 1990s and is admired roughly
as much as it is used; the state DOTs remain a patchwork of recipes. Canada's CSA
A23.1 offers the model everyone cites: a clean dual track in which the buyer may
specify outcomes and leave the mix to the party who actually knows the materials.
Australia leans prescriptive; France bolts low-carbon labels onto a prescriptive base;
India, where most of the future's concrete will be poured, follows recipes. For a
multinational supplier this Babel means no low-carbon product can be amortised across
markets; for a global proponent, no single concrete standard can be written into its
projects. Fragmentation is not a metaphor. It is a tariff.

Policy, at least, is converging on the right instrument from both sides of the
Atlantic: attach a carbon number to the material at the moment of purchase. The EU's
CBAM prices imported clinker from 2026 as free ETS allowances wind down; America's Buy
Clean programmes set EPD-based ceilings for public procurement. Both presuppose the
same thing the private fixes do — provenance-grade, supplier-specific carbon data
travelling with the invoice.

## Breaking the mould

What would actually move the market is unglamorous. Specify performance — strength at
the age the structure will really be loaded, durability by exposure class, an embodied
carbon ceiling per cubic metre — and let suppliers meet it however their materials
allow. Make the EPD, not the industry average, the number that procurement scores.
Engage the supplier before the tender, so the low-carbon mix arrives as a de-risked
bid rather than a variation claim. Write the proponent's carbon ceiling into the
employer's requirements so it survives the seams contractually (the PAS 2080 pattern),
instead of evaporating at the first change of hands. None of this requires a
breakthrough; all of it requires the four parties to look at the same numbers.

Which is the narrow, self-interested point of the software in this repository: one
forward model, one carbon path with provenance, one strength-carbon-cost surface that
an engineer, a buyer, a supplier's lab and a proponent's carbon accountant can stare
at together — with uncertainty stated, extrapolation flagged, and a mix ticket that
discloses whether each carbon figure came from a supplier's EPD or a database default
(`R6-materials-platform.md`); plus a headless interface for scoring bids in batches
(`R5-operability.md`). The chain's contracts will stay fragmented for years. Its
information no longer has to be.

The kicker is the arithmetic this briefing started with. A gigatonne-scale emission,
abatable at a project-level premium of less than 1%, remains unbought — not for want
of chemistry, capital or even goodwill, but because four contracts in a row round it
to zero. Concrete, it turns out, is the easy part. The hard part is the paperwork.
