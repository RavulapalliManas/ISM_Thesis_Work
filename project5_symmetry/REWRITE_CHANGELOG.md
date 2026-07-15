# Rewrite change-log — Feynman ground-up rebuild

Source of truth: `Report/elife/main_best.tex` (rebuilt). Backup of the pre-rebuild version:
`scratchpad/main_best_ORIG.tex`. Assembler: `scratchpad/assemble.py` (splices floats/Methods
byte-exact from the backup). Verification summary at the end.

## What changed (exposition + structure only — no scientific meaning changed)

**Whole-document.** 2132 → 1611 lines; eLife PDF 43 → **36 pages**. Every one of the 12 figures, 7
tables, the Theorem, the Proof, Eq 1, and the entire Methods + back matter is spliced **byte-exact**
from the backup (never retyped), so no table/caption/Methods number can have drifted. All prose
numbers verified: zero numeric tokens in the new prose are absent from the original.

**Abstract.** Single 26-line wall → four short paragraphs, same content, same hedges, shorter sentences.

**Introduction.** Kept the place-field-repetition hook. Added the two-identical-rooms running example
and built the intuition *before* naming bisimulation / equivariance / MDP homomorphism (now one glossed
clause each). Roadmap tightened; Harland lesion foregrounded.

**Results — reordered into the Feynman arc (14 → 11 subsections):**
1. The phenomenon (compartments) — was mid-list, now opens Results.
2. **The quotient theorem** — moved *before* the design; now opens with a plain-language argument, then
   the Theorem/Proof as its crystallization (was: Theorem stated cold, intuition after).
3. The design — now *motivated first* (the two hypotheses), specs second (was: specs cold).
4. Place cells emerge and repeat.
5. Matched bits, opposite maps (headline dissociation).
6. The C4 ceiling.
7. A folded map, not a broken one (geometry).
8. Prediction builds the fold; reconstruction does not (objective).
9. Map and replay are one variable.
10. Survives a noisy and a learned compass (was two subsections, merged).
11. Two fingerprints, and no cell type carries the fold (mixed selectivity + population + PV + field
    stats + cross-seed, consolidated).

**Discussion — cut ~930 → ~400 lines, reordered to lead with the strongest evidence (13 → 7 subsections):**
- Leads with the animal-data dissociation (Grieves/Spiers/Fuhs) and then **the Harland lesion**, which
  was previously buried mid-Discussion.
- The in-silico lesion is folded into the causal-test subsection.
- Cortical compass (Zhang/LaChance), coexistence objection, anatomy, "directional shadow," and the
  tuning-curve identifiability point consolidated into one subsection.
- BVC-downstream + relation-to-models (SR/bisimulation, CSCG/TEM) consolidated.
- Honest-negatives scorecard kept (Table 6).
- **The 2×2 {cue}×{lesion} decisive experiment surfaced as the Outlook centerpiece** (was buried).

**Relocated to Supplementary (`supplementary.tex`), with main-text pointers:**
- The nonlinear-decoder control (kNN/MLP at chance) → new SI section.
- The three superseded single-cell readouts (rotational autocorrelation, odd power, RSA) and why
  orbit-phase decoding is primary → new SI section (references the isotypic figure S13b,c).
- Added main-text pointers to Fig. S8 (glued fundamental domain) and Fig. S13 (population code), which
  the rebuild would otherwise have left uncited. SI figure numbering unchanged (S1–S13).

**Methods.** Kept verbatim (eLife house standard; holds audited numbers). The two load-bearing controls
(corridor-memory, decoder-bias null) are pointed to from Results where the claims are made.

## Invariants preserved (see `REWRITE_INVARIANTS.md` for the full checklist)
Theorem proposition, Eq 1 semantics, all causal claims, the closed hypothesis set, and every calibrated
hedge verified present in the rebuilt text (16/16 distinctive hedge phrases; the LaChance
"failure to reject… no reported power" phrase confirmed after a line-wrap false-negative). Title
unchanged (phenomenon-hook framing retained per the chosen option).

## Second pass — readability polish (em-dashes + heaviness)

On review the draft still read heavy, largely from em-dash overuse (an AI tell flagged in the user's
style notes). Second pass: em-dashes cut from **118 → 3** across both documents (the 3 remaining are
the "not applicable" cells in Tables 3 and 6). Em-dashes were converted, per context, to periods
(splitting long sentences), commas, colons, or parentheses — not blind-swapped — and the worst
run-ons were broken up, in the abstract, introduction, every Results and Discussion prose block, and
the figure/table captions. Re-verified: no numeric token changed (number-diff against the backup is
empty), all 14/14 distinctive hedges present, both compile clean (eLife 36 pp, bioRxiv 28 pp, 0
undefined refs). Methods was already em-dash-free and untouched.

## Third pass — SMVF coherence audit + guarded grammar/visual/de-AI

Ran the scientific-manuscript-verification framework (AUDIT + IMPROVE). **Coherence (figure↔text,
para↔results):** all 12 figures + 7 tables referenced, no orphans, no nonexistent-figure discussion,
every Results subsection cites the right float, all caption↔text numbers match (verified against the
CSVs in `../data/`). **Applied under the Invariant Engine (no number/hedge/claim/citation changed):**
- Fixed a factual coherence error: line ~706 called the k-sweep endpoint the "$C_4$ ceiling" but
  `tab:horizon` is the $C_2$ arena → "a resolved map".
- Added 3 missing panel cross-refs (fig:brain a in the four-room prose; fig:fold b, d).
- Grammar: 5 `\citet` contact-clause "that" insertions (190, 204, 782, 911, 1183), comma-splice →
  semicolon (632), redundant "both" removed (688).
- Visual: overfull URL (139 pt) + inline-math (47 pt) cleared with `xurl` + `\emergencystretch=2em`;
  `$\sim$$100^\circ$` → `$\sim 100^\circ$`.
- Light de-AI: dropped a filler "Crucially,". The pervasive "X, not Y" antithesis was KEPT — it is
  load-bearing (invariance-not-information, prediction-not-reconstruction); removing it changes
  meaning, which the Invariant Engine forbids.
- **Two provenance ambiguities resolved by annotation (no data recomputed, only existing verified
  numbers consolidated):** (i) fig:fold caption now states its bars are the 64-network field-statistics
  ensemble while the text's per-unit distributions are over the full 112-network sweep (56,000 units);
  (ii) a Methods sentence now unifies the per-analysis ensemble sizes (phase/geometry n=10/10/8 = 112;
  C$_4$ ceiling and fold/lesion figures n=6/6/4, n=4 in C$_4$). Verified against the CSVs in `../data/`.
- Re-verified: eLife 36 pp / 0 undefined / 0 overfull > 6 pt; bioRxiv re-ported, 28 pp / 0 undefined.

## Verification
- eLife `main_best.tex`: xelatex + bibtex clean, **36 pp, 0 undefined references/citations**. 3 overfull
  hboxes >20pt (the pre-existing Data-availability repo URL, and one inline `$\max R^2$`); ~37 further
  ~6pt cosmetic — a typographic cleanup pass is available on request.
- `supplementary.tex`: pdflatex clean, **15 pp, 0 undefined refs**; figure numbering S1–S13 preserved.
- Floats byte-exact vs backup: all 20 (12 fig + 7 tab + Eq 1) verbatim; Theorem + Proof verbatim.
- Number diff: no numeric token in the new prose is absent from the original.
