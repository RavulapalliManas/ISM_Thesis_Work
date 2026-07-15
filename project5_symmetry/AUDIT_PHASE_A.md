# AUDIT — Phase A (CERTIFY/AUDIT), 2026-07-15

Read-only SMVF audit of all three documents (eLife `main_best.tex`, `supplementary.tex`,
`biorxiv/main.tex`), plus adversarial primary-source citation checks on the load-bearing biology.
This is AUDIT output, **not** a PASS certification — open MAJOR items remain (see below).
Provenance companion to `HANDOFF.md` §3 and `BIOLOGY_LEDGER.md`.

## Edits applied this pass (eLife `main_best.tex` only; bioRxiv re-port pending)

| # | location | old | new | basis |
|---|---|---|---|---|
| 1 | Fano (Results) | Fano 0.52 | **0.65** | `orphan_metrics.csv`, s1/full, mean of 10 seeds = 0.649±0.066 |
| 2 | Table 1 gridness | gridness ≤0.08 | **(full compass, gridness ≤0.08)** | s1/full max = 0.083; folded encodings reach 0.50 |
| 3 | Limitations gridness | mean −0.35, max +0.08, none >0.3 | **full-compass: mean −0.29, max +0.08; invariant encodings reach 0.50 via field repetition** | `orphan_metrics.csv` all-full = −0.289; global max 0.5031 (s1/axis) |
| 4 | Zhang onefold (Disc.) | only 3/8 bidirectional | **1/8** | Zhang 2022 PMC9761682: "only 1/8 had a BD score higher than shuffle" |
| 5 | LaChance severance ×2 | "requested in review and declined" | **"the authors name as necessary future work…did not run"** | provenance of reviewer request unverifiable; authors' own future-work framing is sourced |

Omnidirectionality (mean 0.56 / med 0.64 / 35%) verified correct against s1/full
(0.586/0.632/36.3%) — **left unchanged**.

## The §3 orphan-number alarm was a mis-aggregation (RESOLVED)

HANDOFF §3 "recomputed" column = mean/median over **all 112 networks**; the paper's claims are the
**s1/full** baseline. Reproduced exactly: all-112 Fano mean 1.157, omni median 0.345, gridness_max
mean 0.028. On the correct condition, Fano (0.65<1, under-dispersed) and omnidirectionality
(0.59/0.63/36%) both **hold**. Not a conclusion flip. Numbers guilty until proven innocent applied to
the HANDOFF itself (CLAUDE.md §5.1).

## Open findings (not yet edited — need author decision)

**MAJOR**
- **A1 Anatomy citation.** "tracing places the subiculum in the target list of the granular area and
  not the dysgranular [shibata1994, vangroen1992]" is not cleanly supported: shibata1994's own title
  targets the *retrohippocampal region* (post/pre/parasubiculum + entorhinal), not subiculum proper;
  vangroen1992 is the *dysgranular* paper (supports only the negative half). Re-attribute the
  granular→subiculum positive to sugar2011 (A29c→Sub) and mind the subiculum-vs-subicular-complex
  distinction. Narrow "no excitatory route" claim survives via li2025 + sugar2011.
- ~~**A2 Untraceable numbers.**~~ **RESOLVED — regenerated with new traceable scripts.**
  `field_area_horizon.py` → `field_area_horizon.csv` (33.8/40.7/42.1/39.7; step-then-plateau holds);
  `corridor_dwell.py` → `corridor_dwell.csv` (translation 0.56→0.51 decays, rotation 0.71→0.99 rises;
  opposite-signature confound control holds). Proposed text updates in `NUMBER_MAP.md`, **pending
  review** (fresh trajectories → values differ slightly from the untraceable originals).
- ~~**A3 lachance2024 authorship.**~~ **RESOLVED — bib is correct.** Crossref + PMC11399390 both
  confirm **LaChance, Patrick A. & Hasselmo, Michael E.** (two authors). The earlier agent's "Taube"
  was wrong; no change needed. (Flagged rather than edited — correct call.)

**MINOR**
- **SI cross-refs off by one:** main text "Fig. S9" (CA1) → actually S10; "Fig. S10" (coset/phase) →
  actually S11. The "Initialization" figure is S9. (S12 isomap, S13 population correct.)
- **field_stats.csv (1.781) vs cell_properties.csv (1.746)** disagree on s1/full field count; paper
  uses 1.78 = field_stats. Pick/annotate the authoritative file.
- **li2025 "recovers the same split"** overstates (dorsal/ventral axis, not granular/dysgranular).
- **Harland nuances:** "abolish the HD signal" is Harland's cited *background* property, not measured
  in that study; "intact rats did not repeat" is stronger than their "less prominent" (d=0.07 / 82%
  low justify it); confirm sparsity convention (0.32→0.37 = "less sparse").
- **LaChance postrhinal p** 3.53×10⁻⁵ rounded to 4×10⁻⁵ (defensible).
- ~~**bioRxiv materially stale**~~ **RESOLVED — full re-port done.** bioRxiv was missing 9 whole
  subsections (~700 lines, the entire biology Discussion) plus all number corrections. Regenerated
  from the corrected eLife body with float conversion (figure->figure*, table->table*), 4 missing
  figure PDFs copied in, abstract synced. Now 2142 lines / 25 subsections (= eLife), builds clean
  with pdflatex (34 pp, 0 undefined refs, 0 overfull hboxes). SI cross-refs ported; supplementary.tex
  is shared/single. Both eLife and bioRxiv verified to compile.

## Citation verdicts (adversarial, primary-source)

- **Harland 2017** (PMC5607353): 9/9 numbers CONFIRMED to the digit (verified twice).
- **Zhang 2022** (PMC9761682): CONFIRMED except 3/8→1/8 (fixed); A10 quote verbatim.
- **LaChance 2024** (PMC11399390): B1–B6 CONFIRMED; B7 provenance unverifiable (softened).
- **Alexander 2020** (PMC7035004): CONFIRMED, fairly and hedgedly cited.
- **Anatomy:** li2025 (b) GABAergic→CA1 CONFIRMED (PMC11736107); sugar2011 CONFIRMED (caution quote);
  naber2001 CONFIRMED; shibata1994 UNVERIFIABLE (paywalled) + secondary sources contradict cited use.

## Number audit — COMPLETE (full map in `NUMBER_MAP.md`)

Every reported number recomputed from its CSV. Result: the numerical **spine reproduces to the
digit** — Tables 2/3/5, geometry (incl. 103/112 assignment and the 8/8 axis-in-C4 systematic
exception), fold ratios, the full lesion dissociation, remapping, both compartment mazes, mixed
selectivity, unit heterogeneity (all correlations + quartiles), BVC, replay coverage, weak-cue
sweep, learned/noisy compass, Hockeimer, Isomap. The paper's numbers are traceable and correct.

Residual issues (all in `NUMBER_MAP.md`):
- **3 MINOR discrepancies:** sequenceness under axis 0.28 → recomputed **0.205** (s2; "approaches
  floor" still holds); shuffled-rollout coverage 1.50× → **1.57×**; position-only variance endpoint
  0.360 is the *axis* value (const is 0.312).
- **2 UNTRACEABLE (A2 confirmed):** field-size-by-horizon (30.6/38.1/38.5/37.4) — no code;
  corridor-conditioning rotation series (0.84→1.00) — no code; translation series (0.63→0.52)
  doesn't match the only script that computes the method.

## Not exhaustively run this pass
Full math-auditor (Theorem/Eq.1 read, internally sound; not formally re-derived) and Marr-level
passes were spot-checked, no blockers surfaced. Table 4 (bound) is internally consistent
(predicted = ½ + ½·distinguishable) and its measured column = Table 2 (verified).
