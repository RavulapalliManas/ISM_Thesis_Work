# Handoff — project5_symmetry paper (as of tonight)

The paper is a predictive-RNN / cognitive-map study. **Central claim (phenomenon-first, quotient law):**
a predictive cognitive map represents space only *up to the symmetries the agent's self-motion
cannot break*; place-field repetition is that fold, not a hippocampal failure.

Both paper versions **compile right now.** Everything below is CPU-only from here — no GPU needed.

---

## 1. Where things stand

**Two paper versions (keep separate):**
- `Report/elife/main_best.tex` — eLife class, 22 pp. **This is the source of truth; edit here.**
  (`Report/elife/main.tex` is the frozen pre-reframe baseline; ignore.)
- `Report/biorxiv/main.tex` — HenriquesLab bioRxiv template, 16 pp. **Auto-generated** from the
  eLife body. After editing the eLife version, regenerate with:
  `python3 Report/biorxiv/port_from_elife.py` then `pdflatex; bibtex; pdflatex; pdflatex` in
  `Report/biorxiv/`. (The template's class was patched: cleveref must load after hyperref.)

**Build commands** (from the respective dir): eLife uses `xelatex; bibtex; xelatex; xelatex`;
bioRxiv uses `pdflatex; bibtex; pdflatex; pdflatex`. To render a page for eyeballing without
poppler: `gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r95 -dFirstPage=N -dLastPage=N -o out.png main_best.pdf`.

**Compute is gone** (RunPod pod died). Nothing critical was lost: all 36 result CSVs are in
`Report/data/`, and trained checkpoints + a full backup are on the SSD (see file map). Any new
analysis runs on the laptop CPU.

---

## 2. What was done this session

- **Four adversarial referee reviews** (neuroscience, ML theory, statistics, editorial) were run
  and synthesized. All rate the core science sound; all said "major revision" on framing/rigor.
- **Phenomenon-first reframe** applied: new title, abstract, intro all lead with the repetition
  puzzle; formal **Theorem** (hypotheses + trajectory proof) added; two redundant horizon sections
  merged; offline-replay compressed.
- **CA1 demoted** (unanimous referee call): out of the abstract and the main figure; the real-data
  scatter is now the supplementary `figS_ca1`; Fig 6 is two model panels (four-room + city-block).
  Framed as *consistent-not-confirmatory* (controls show it is not separable from a conjunctive
  place-by-direction code).
- **Correctness fixes:** "identical information"→"identical entropy"; Muller/Knierim citation error
  →Derdikman; SR "completes"→"instantiates" + MDP-homomorphism citations; learned-compass wording;
  learned-compass sample sizes corrected (C1 n=10, C4 n=8, from the item-7 rerun that finished
  before compute died).
- **`avoid-ai-writing` skill** applied: zero flagged AI-isms, em dashes down to 7 (all table
  "n/a" cells).
- **Item 8 (weak symmetry-breaking → rate remapping)** was launched but the pod died mid-compile,
  so it is written up as a **pre-registered prediction**, not a result. Code is on the (dead) pod
  only; the arena change (`environments/arena.py` tint, `environments/compartment_arenas.py`,
  `experiments/run_multi.py` `weakbreak` group) is in the local repo and smoke-tested.

---

## 3. RESUME HERE — in-progress analyses (CPU, from checkpoints)

Two new analyses were being added to convert the last referee gap into a measured result:

### #1 Permutation null (IMPORTANT — it changed a claim)
`analysis/perm_null_geometry.py`. A validation run (1 seed, 40 perms) gave the key finding
(preserved in `Report/data/perm_null_validation.csv`):
- folded `axis`/C2 decodes phase at **0.53**, versus a proper permutation null of **0.50 ± 0.005**
  — i.e. ~6 SD ABOVE the null, a small but **statistically real residual, not decoder bias**;
- non-folded `parity` decodes at 0.96, far above the null.

**Consequence:** the paper's "the quotient is exact / folds to chance" is an overclaim. It must
become **"near-exact: folded codes sit a few percent above the group-theoretic floor, significantly
above the permutation null yet far below the non-invariant encodings."** This is what the ML-theory
referee predicted. NOT YET EDITED INTO THE PAPER — do this once the full numbers are in.

Full run (5 networks, 500 perms) was launched tonight; its output went to a session-temp dir that
will not survive. **Re-run it tomorrow** (~30–40 min, self-contained):
```
PYTHONPATH=. python3 analysis/perm_null_geometry.py \
  --ckpt-root "/Volumes/Crucial X6/prnn_backup/checkpoints" \
  --data-root /tmp/permdata \
  --out-dir  "Report/data/analysis_pending" --n-perm 500 --n-traj 2000 --n-states 15000
```
Then edit the paper: (a) section title line "The quotient is exact…" → "near-exact"; (b) the
"folded-to-floor" sentence in the Methods orbit-phase paragraph; (c) Table 3 / ceiling text; (d)
the Theorem's "empirical saturation" paragraph. Report each folded value against the 0.50 null.

### #2 Procrustes — is folding literally the quotient map (novel)
`analysis/procrustes_quotient.py` — coded, compiles, `scipy.procrustes` available, NOT yet run.
Tests whether the C2/axis domain-manifold equals the C1/axis one up to a rigid transform (folding =
the quotient map X→X/G), baselined against cross-seed disparity. Given #1's residual, the sharpened
question is whether the fold is *almost* but not perfectly the quotient manifold. Run:
```
PYTHONPATH=. python3 analysis/procrustes_quotient.py \
  --ckpt-root "/Volumes/Crucial X6/prnn_backup/checkpoints" \
  --data-root /tmp/permdata --out Report/data/analysis_pending/procrustes.csv
```
Run #1 and #2 **sequentially** (both are CPU-heavy; don't overlap on a laptop). #1 and #2 share
the regenerated `/tmp/permdata`, so the second is faster.

### #3 Rotation arm of the CA1 test — DONE in text
Already in the paper (same-orientation r=0.23 vs rotation-arm r=−0.06), framed honestly as
consistent-not-diagnostic. No action unless you want to strengthen it.

### #4 Coset-vs-phase panel — TODO (quick)
A clean figure panel: folded codes decode within-domain position at R²≈0.9 while phase is at the
null ("folded, not broken"). Data already in the phase CSVs (`domain_r2`, `phase_acc`). Add a panel
to `analysis/make_paper_figures.py` (e.g. into `fig_population`), regenerate, wire a sentence.

---

## 4. File map (persistent)

- Papers: `Report/elife/` (source), `Report/biorxiv/` (ported, + `port_from_elife.py`).
- Result CSVs (every paper number): `Report/data/*.csv` (+ `README.md` index).
- Figure generator: `analysis/make_paper_figures.py` — `python3 … --data Report/data --figs
  Report/elife/figures` (regenerates all; copy PDFs to `Report/biorxiv/figures/`).
- Analysis scripts: `analysis/perm_null_geometry.py` (#1), `procrustes_quotient.py` (#2),
  `hockeimer_reanalysis.py` (CA1 + controls), `run_phase_decoding.py`, `run_compartments.py`.
- Trained checkpoints (345 final): `/Volumes/Crucial X6/prnn_backup/checkpoints/` (hd_invariance
  s1/s2/s4 × 4 encodings × seeds; horizon; topology; compartments).
- Full backup (papers + CSVs): `/Volumes/Crucial X6/prnn_backup/pod_final/`.
- Reviews: run this session via subagents; the synthesis is in the conversation, not a file.

---

## 5. Needs compute (blocked until GPU returns) — do NOT attempt on CPU
- **Item 8** (weak symmetry-breaking → partial rate remapping): the experiment that would turn the
  "predicted" table row into a result. Env code is in the repo (`weakbreak` group in `run_multi.py`),
  smoke-tested. Currently a written prediction.
- Hidden-size and arena-size sweeps: dropped (arena-size arenas were found non-C2-symmetric; see the
  `arena.py` rotation hardcoded to N=18). Not worth redoing unless a referee asks.

---

## 6. One-line status
Paper is phenomenon-first, formally proved, referee-hardened, both versions compiling. The only
open scientific item is the **"exact → near-exact" correction** (evidence in hand from #1) plus the
**Procrustes result (#2)** — both CPU, ~1–2 hours total, commands above. Then it is submittable.

*(Commits are author-only; nothing was committed this session. Large data stays on the SSD,
gitignored.)*
