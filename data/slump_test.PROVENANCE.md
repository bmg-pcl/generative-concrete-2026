# `slump_test.data` — provenance

**Dataset:** UCI Concrete Slump Test (I-Cheng Yeh), 103 instances, 11 columns.
Canonical home: <https://archive.ics.uci.edu/dataset/182/concrete+slump+test>

**Why this file is committed rather than downloaded.** `src/data_fetcher.py` registers this
corpus as `kaggle_slump` with the canonical UCI URL, but the development environment's network
policy denies `archive.ics.uci.edu` at the proxy gateway (403 on CONNECT). Committing the file
keeps a fresh clone and CI self-sufficient — the same reasoning as `Concrete_Data.xls`.

**Retrieved from a third-party mirror**, since the canonical host was unreachable:
`https://raw.githubusercontent.com/natalipivnitskaya/Strength-of-Concrete-Analysis/master/data/slump_test.data`

**Integrity checks performed before committing** (a mirror is not an authority, so the file was
validated against the dataset's documented shape rather than trusted):

- SHA-256: `6745397c9c9d9aa37c59c554d6d866f7975decddeb3a140a485b0862fca11d70`
- 103 data rows + 1 header — matches UCI's documented instance count exactly.
- Columns exactly: `No, Cement, Slag, Fly ash, Water, SP, Coarse Aggr., Fine Aggr., SLUMP(cm),
  FLOW(cm), Compressive Strength (28-day)(Mpa)` — matches the documented schema.
- 0 nulls, 0 duplicate rows.
- Every column physically plausible: cement 137–374, water 160–240, SP 4.4–19,
  coarse 708–1050, fine 641–902 kg/m³; slump 0–29 cm; flow 20–78 cm; strength 17.2–58.5 MPa.

**If you can reach UCI**, re-download from the canonical URL and diff against this file. Any
difference should be treated as this mirror being wrong, not UCI.
