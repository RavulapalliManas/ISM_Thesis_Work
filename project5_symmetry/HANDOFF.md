# HANDOFF — project5_symmetry

Written 2026-07-15. State at the end of a long session on the biology and a numerical audit.
Companion doc: **`BIOLOGY_LEDGER.md`** (the full record of what was verified, refuted, and fixed).
The standing rules that came out of this session live in `~/.claude/CLAUDE.md` §5 (numbers) and §6 (memory).

**Source of truth:** `Report/elife/main_best.tex` (eLife class; edit here). `Report/elife/main.tex` is
the frozen pre-reframe baseline — ignore. The biorxiv version must be re-ported after edits settle.
As of the previous handoff both versions compiled cleanly; this session's ~35 edits have NOT been
recompiled (build integrity checked programmatically — all \cite/\ref resolve, no orphan labels).

---

## 0. THE ONE THING TO DO FIRST

**Decide what to do about two numbers the paper states that we cannot reproduce.** They are the only
edits left that change what the paper *claims* rather than how it reads, so they are yours, not mine.
See §3. Everything else this session is either done or is mechanical cleanup.

---

## 1. WHERE THE PROJECT IS

Strong specialist paper (eLife / PLoS Comp Biol tier), technically near-submittable, **not**
submittable yet. Science core solid and untouched:

- `axis` vs `parity` — matched information, opposite equivariance, opposite fold (0.552 vs 0.955).
  No rival model can state this prediction; no animal experiment can run it.
- A theorem with a 1/|G| bound the data HIT (C4: 0.275 against a predicted 0.25).
- **Harland et al. 2017 is a published causal test of the law with zero free parameters, read that
  way by nobody.** The single best asset in the paper.

The limitation no editing fixes: **it is a simulation plus a reinterpretation of the existing
literature — no new animal data.** The experiment that would break it open is unrun.

---

## 2. WHAT LANDED THIS SESSION (done, in the paper, verified)

1. **Per-unit heterogeneity kills the BVC objection.** New pipeline `unit_heterogeneity.py` — 56,000
   units, FIRST per-unit data in the project (everything before was per-network means). The fold is
   NOT carried by a subpopulation: 93% of units fold in s2/axis (median 0.94); most vs least
   boundary-tuned quartile fold at 0.947 vs 0.923 — 0.024 on an effect of 0.9; r(border,fold)=0.16.
   A BVC account needs that difference to be the whole story; it is a rounding error. Written into the
   BVC rebuttal section.

2. **Information/equivariance double dissociation.** `mixed = ev_conj - ev_add` tracks INFORMATION
   (ordered full>parity>axis>const, ~flat across arenas); phase decoding tracks EQUIVARIANCE (axis vs
   parity, same one bit, differ by 0.40). Same networks, two orthogonal causes. New Results subsection.
   NOTE: mixed selectivity is NOT a fold readout — 88% of its rise is in the C1 arena; reported as an
   information readout with that null stated.

3. **~30 audit defects fixed** across three passes. Load-bearing: "n=6 per arena" was n=4 (C4);
   "110 of 112 networks" was 103 with a SYSTEMATIC failure (all 8 s4/axis prefer X/C4, contradicting
   our own decoder — now stated); two "exact floor" p-values were normal approximations; CSCG table
   row inverted vs our own quoted text; abstract "folds completely" -> "almost completely" (residual is
   real, p=0.008); G reverted to its falsified definition in 3 places (fixed to "subgroup OF the
   environment's symmetry group that the compass can't break").

4. **A mechanism I invented, retracted.** Claimed s1's bidirectional tuning reproduces Alexander et
   al.'s egocentric-boundary mechanism. It does not — `const` (no compass) has FLAT tuning, and our
   agent samples headings uniformly so the occupancy confound is absent. Alexander now cited as an
   alternative we CANNOT rule out; answered with Harland's lesion (a fact about the map).

5. **New verified bib entries:** shibata1994, sugar2011, vangroen1992, li2025, naber2001, alexander2020.
   Anatomy claim narrowed to "no substantial direct EXCITATORY route" (li2025 found a GABAergic
   RSPd->CA1 projection — cited AGAINST ourselves).

---

## 3. THE OPEN DECISION — orphan numbers (DO THIS FIRST)

Three numbers were in the paper with NO code producing them (confirmed: the words appear nowhere in
the repo outside the manuscript). I wrote `orphan_metrics.py`, validated gridness on a synthetic
hexagonal grid (+0.994) and a place field (-0.091), recomputed. Results in `Report/data/orphan_metrics.csv`
(112 networks):

| claim | paper | recomputed | what to do |
|---|---|---|---|
| gridness max | 0.08 | **0.028** | SURVIVES, stronger. "No grid cells" is safe. Update the number. |
| frac units gridness>0.08 | 0 | 0.002 | matches. fine. |
| **Fano factor of field counts** | **0.52** | **1.157** | **CONCLUSION MAY FLIP.** Paper says "dispersed, Poisson decisively rejected." Fano~1.16 is Poisson-LIKE. Re-examine. |
| **omnidirectionality median** | **0.64** | **0.345** | **WRONG DIRECTION.** Our place cells are LESS omnidirectional than claimed — a real limitation. State it honestly. |
| omnidirectionality mean | 0.56 | 0.379 | same. |
| frac unit-pairs >0.8 | 0.35 | 0.168 | same. |

CAVEAT: a mismatch could mean the paper's number was wrong OR that my definition differs from whatever
(uncaptured) code produced the original. No way to tell — there was no script. Either way the paper
now cites numbers we cannot reproduce, so they must change. I did NOT edit these in; the Fano and
omnidirectionality changes alter what the paper claims — your call.
Locations: `main_best.tex` ~L970 (Fano), ~L1482 & ~L1784 (gridness), ~L1803-1805 (omni).

---

## 4. STILL OPEN (mechanical, lower priority)

- **More untraceable numbers** the audit flagged, not yet chased: field-size-by-horizon
  (30.6/38.1/38.5/37.4), corridor-conditioning series (0.84...1.00 / 0.63->0.52), Isomap orbit
  distances (0.17-0.18 vs 0.85-0.89). Same rule: find the script or cut.
- **`field_stats.csv` vs `cell_properties.csv` disagree** on field counts (1.781 vs 1.746); only one
  cited. Pick one.
- **Validation-suite holes** (`test_compass_symmetry.py`, 20/20 but): every synthetic has ONE unit
  (aggregation over 500 untested); `rot_gain` has no calibrated null; `cx` uses the same formula as
  the analysis so a transpose would be invisible.
- **Figures 11 -> ~6.** Abstract leads with repetition; repetition is currently figure 10. Cut hard.
- **Full LaTeX compile**, then re-port to biorxiv.

---

## 5. THE BIOLOGY / HASSELMO PLAN (full record in BIOLOGY_LEDGER.md)

- **Zhang 2022 demoted** antecedent -> consequent (Harland's intact radial condition + anatomy show
  the hippocampus does NOT read those dysgranular cells).
- **The experiment the account stands on:** {one cue, two cues} x {sham, HD lesion}. Repetition in
  exactly ONE of four cells, nothing fitted. Two cells already observed. Hasselmo has the apparatus;
  Dudchenko/Wood/Grieves have the lesion AND the place cells.
- **Send Hasselmo an INTRODUCTION, not an analysis** (he has no place cells). Do NOT propose the AB
  rate-map symmetry analysis (arithmetic corollary of his own Fig 4 for the 60% conjunctive cells).
  DO offer: his MEC HD null is already our "MEC doesn't fold" arm; the severance experiment his
  Reviewer 1 demanded and he declined (only simulation can run it).
- **Say the hard thing first:** Alexander et al. 2020 (his own lab) is the strongest deflationary
  account and we cannot rule it out. Answer = it explains tuning, not Harland's lesion.
- Traps (ledger §5): use the MEC **HD** null not the grid null (hexagonal lattice is
  inversion-symmetric); "symmetry tracks environment" is Zhang's result not LaChance's; the POR paper
  is LaChance/Graham/Shapiro/Morris/Taube 2022, and its "cannot differentiate N from S" quote is a
  setup the authors immediately REBUT — do not quote it alone.

---

## 6. GIT / LOGISTICS

- **15 commits unpushed.** User must run `git push origin main` (classifier blocks me on this remote).
  Never put AI/Claude attribution in commits.
- Uncommitted: `main_best.tex`, `references.bib` (modified); new: `BIOLOGY_LEDGER.md`, `HANDOFF.md`,
  four scripts (`compass_symmetry.py`, `test_compass_symmetry.py`, `unit_heterogeneity.py`,
  `orphan_metrics.py`), three CSVs. `dst.html`/`sa.html` are stray, not mine — ignore/clean.
- **Checkpoints + data on the external drive**, not in the repo:
  `/Volumes/Crucial X6/prnn_backup/checkpoints` and `.../data`. 112 checkpoints, all verified loadable.
  All 54 CSVs parse cleanly — **NO corruption** (user worried about the cloud transfer; it is fine).
- All new analyses CPU-only, memory-bounded (CLAUDE.md §6), run in minutes.

---

## 7. HONEST STANDING NOTE

Three audits, three rounds of real errors — several mine, some invented mechanisms. **Do not report
this paper as clean.** Assume the next pass finds more; keep the numbers-are-guilty discipline
(CLAUDE.md §5). The habit that worked all session: the user refused to trust a result, and every time
they pushed, the check found something real.
