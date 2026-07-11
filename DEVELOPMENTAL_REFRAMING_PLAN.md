# Plan: Reframing the Predictive-RNN Framework as a Probe of the Developing Hippocampus

**Author:** Manas Venkata Sai Ravulapalli · **Drafted:** 2026-06-04
**Status:** active plan · **Centerpiece (locked):** developmental ordering spine (topology-before-geometry + emergence order)

> **⚠️ Framing superseded — read `THESIS_FRAME.md` first.** The thesis spine evolved from descriptive
> "developmental ordering" to **causal dissection**: the headline is now the *triple-locked HD→place
> edge* (necessity + objective-specificity + invariance), and the two priority training runs are the
> **HD-freeze lesion-then-resume** and the **autoencoder objective control**, plus a **spontaneous-only
> maturation** run. The execution detail below (data path, file-level spec, metric reuse) remains valid;
> re-prioritize the experiment list per `THESIS_FRAME.md` Tier 1.

---

## 0. TL;DR

We reframe the existing predictive-RNN cognitive-map work as **an in-silico assay of hippocampal
development** that fuses two literatures: **Ulanovsky** (the *geometry* of the mature map) and
**Dragoi** (preplay / preconfigured sequences and their *developmental emergence*). A predictive RNN
is one object read two ways — **drive it with input → geometry**; **let it run autonomously → preplay/replay** —
and the **training trajectory is the development** connecting an "innate" prior to the mature endpoint.

The centerpiece is the **developmental-ordering spine**: within each trained network, topology
crystallizes before metric geometry, and tuning emerges in the biological order. This is a
**within-network** claim, so it sidesteps the n=3–5 between-condition power problem that currently
limits Project 5.

**Compute reality:** ~$5 RunPod now, more pending from Debbayan. Therefore the plan is
**reanalysis-first**: ~90% of the spine is forward-passes + `ripser` on a laptop (effectively free,
using the checkpoints you already have off-repo). Only the non-predictive control needs a GPU.

**One-sentence thesis.** *Reinterpreting a predictive RNN's training trajectory as an in-silico
ontogeny, we show that population-manifold topology (an HD ring, then a 2-sheet) crystallizes before
metric geometry, that representations and their spontaneous (preplay→replay) read-outs emerge in the
empirically observed order, with network initialization as the "innate" preconfigured prior (Dragoi)
and landmark symmetry as the "rearing" that reshapes the maturation schedule toward the mature
geometry Ulanovsky/Moser describe.*

---

## 1. Strategic decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Paper count | **One integrated paper** | None of P3/P4/P5 stands alone (P5 underpowered, P4 unrun, P3 no artifacts). The integration *is* the contribution; the developmental frame gives the weak symmetry result a home. |
| Centerpiece | **Developmental ordering spine** | Within-network ⇒ immune to the n=3–5 between-condition power floor; safest, and differentiates from competitors (see §8). |
| Evidence to lead with | **Power-immune results** | sRSA permutation-blindness (analytic), RA monotonicity (shuffle z=15/8/6), C2 sign-flip (algebraic), HD-ablation dissociation, and *within-net* convergence-ordering. |
| Build order | **Reanalysis-first, compute-gated last** | Checkpoints exist; the spine is CPU-bound. Spend the scarce GPU only on the non-predictive control. |
| Target | **eLife / PLoS Comp Biol** (MVP); workshop/Cosyne to stake priority | Honest "developmental assay + falsifiable predictions" fits these venues; defer Nature Comms to the stretch. |

**Out of scope for v1** (state explicitly as future work — promising them is a rejection risk):
bat-specific 3D signatures (isotropy, multiscale fields, toroidal HD, fragmented replay — no 3D
training stack exists), the cortical-prior transfer experiment (P3 never ran; architecture-transfer
risk), and any grand "one principle explains everything" Perspective.

---

## 2. The intellectual core (why Ulanovsky + Dragoi actually combine)

Not thematic — mechanical. A predictive RNN trained on self-motion→observation prediction is **one
dynamical object with two measurement modes**:

- **Input-driven mode → geometry.** Probe the representation with experienced trajectories: place/HD/
  boundary tuning, manifold dimensionality and topology, metric type. These are **Ulanovsky's mature-map
  signatures** — they define what "adult" looks like and become the *target* of the developmental curve.
- **Autonomous mode → preplay/replay.** Drive the recurrent core with internal noise (no input): it
  emits decodable trajectory sequences. This is **Dragoi's preconfigured repertoire** — preplay if
  sampled before an environment is experienced, replay if after.
- **Development = the training trajectory.** Initialization + the pre-training autonomous repertoire =
  the "innate" prior; training = the experience that selects/sharpens it; the asymptote = the mature map.

The thesis's three projects are the three axes of this one claim:
**P4 = developmental ordering** (topology before geometry) · **P5 = environmental shaping** (symmetry as
rearing) · **P3 = the preconfigured prior** (deferred to future work in v1).

---

## 3. Honest current state (what exists vs what doesn't)

- **Project 5** — the only project with runs. 8 nets (S4 n=5, S2/S1 n=3), checkpoints at
  {5k,10k,20k,40k,60k,80k,final}, online sRSA(euclid/city)+manifold-ID every 1k steps. Strong,
  power-immune results: sRSA-blindness proof, RA monotonicity, C2 sign-flip, HD-ablation dissociation.
  **Underpowered between conditions** (nothing survives Bonferroni; body misreports p=0.036 vs CSV
  p_bonferroni=0.107). DTG confounded by the MiniGrid city-block metric.
- **Project 4** — machinery real (`ConvergenceTracker`, `ripser` Betti, geodesic sRSA, replay decoder)
  but **unrun**: two headline experiments are `NotImplementedError`, all gap values empty.
- **Project 3** — architecture layer, **no run artifacts**.
- **Checkpoints: exist off-repo (you confirmed).** ⇒ the reanalysis tier is genuinely cheap.
- **Spontaneous/replay machinery** exists but `forward_vs_reverse_ratio` (sign of x-displacement) is
  **not** a publishable sequenceness statistic, and raw `pRNN_th` (the P5 model) has no `spontaneous()`
  method of its own — that lives on the `PredictiveNet`/project3 wrappers (see §6, Task D).

---

## 4. Must-pass defenses (the five bars that decide accept vs desk-reject)

These came out of an adversarial Reviewer-2 pass; every framing dies without them.

1. **Fix the stats misreporting** (free, do first). Reconcile every report p-value with
   `Report/stats_results (3).csv`. One body-vs-CSV mismatch and a reviewer distrusts every number.
2. **Real sequenceness statistic** with circular time-shift **and** cell-ID shuffles **and a
   dimensionality-matched random-dynamics null.** No replay/preplay claim survives the "preplay is a
   low-D artifact" critique without the low-D null.
3. **Validate PAA/RA/C2 on synthetic ground truth** (free): hand-built codes with pure rotation /
   pure folding / both / neither; show the metrics separate them. Cheap, currently absent.
4. **Threshold-robustness for the topology gap**: sweep the Betti-stability and sRSA criteria;
   bootstrap CIs on Betti barcodes. Otherwise "topology before geometry" is a thresholding convention.
5. **One non-predictive control** (matched autoencoder / no-rollout). Without it, "*predictive*
   learning generates these phenomena" is unfalsifiable (SR / path-integration / efficient-coding all
   make place/HD/grid units). This is the single most important credibility item — and the only one
   that needs a GPU.

**Two honesty constraints to write into the paper:**
- The HD input is a **noiseless 12-bin oracle**; the global-anchoring result must be shown to survive
  *noisy/drifting* HD, or the claim softened (the real Bjerknes phenomenon is coherence-amid-drift).
- "Critical period" → call it **"early-curriculum sensitivity"** unless emergence is decoupled from the
  loss. And reframe the deepest tension as a *result* (next line).

**Turn the "training ≠ development" objection into a finding.** Farooq & Dragoi's maturation is
experience-*independent*; gradient descent is experience-*driven*. Don't paper over it — make the
precise claim and test it: **the emergence *order* is invariant to the amount and ordering of
experience, even though absolute timing is not, because order is set by the loss landscape /
architecture / input statistics, not experience quantity.** (Test: vary experience amount and
curriculum order; show order conserved, timing shifts.) This engages Dragoi honestly instead of
faking an experience-independent mechanism.

---

## 5. Work plan, phased by compute cost

### Phase 0 — Free wins (now, ~1–2 days, no GPU)
- **0.1** Fix report stats (Bar 1). Reconcile body ↔ CSV; rewrite significance language honestly.
- **0.2** Inventory & stage checkpoints: copy the off-repo `ckpt_*.pt` into a known local layout
  (`results/<cond>/seed_XX/ckpt_<step>.pt`); record which {condition, seed, step} tuples actually exist.
- **0.3** Synthetic-ground-truth metric validation (Bar 3) → becomes a methods figure.

### Phase 1 — Keystone reanalysis (CPU, effectively free) — **this is the paper**
Everything here runs on a laptop from existing checkpoints.
- **1.1** Build the **checkpoint→(hidden, positions) extractor** (§6, Task A). Reuses
  `_build_model` + `_collect_hidden_states`. Fixed held-out eval trajectory set so all checkpoints see
  identical input.
- **1.2** **Developmental metric curves** (§6, Task B): run the full stateless suite (sRSA euclid/
  geodesic, RA, C2, SCI, PAA, decode_error, frac_tuned, manifold_id) on every checkpoint → tidy long
  CSV `condition,seed,step,metric,value`. Define `T_metric` = first checkpoint crossing criterion.
- **1.3** **Topology-before-geometry on P5 checkpoints** (§6, Task C). Two *free* instantiations:
  - **(a) HD-ring (primary, ties to Bjerknes 2015):** Betti-1=1 on HD-conditioned hidden states
    appears at an earlier checkpoint than spatial sRSA crosses 0.4.
  - **(b) Spatial manifold:** intrinsic-D / persistence-gap stabilizes before metric sRSA.
  - Reuse `compute_betti_numbers`, `compute_*_convergence_step`, `compute_convergence_gap`. Per-net
    paired `T_geometry − T_topology` across 8 nets (within-subject design). Threshold-robustness panel (Bar 4).
- **1.4** **Preplay → replay maturation (Dragoi leg)** (§6, Task D): autonomous-rollout helper for
  `pRNN_th` + proper sequenceness statistic (Bar 2) + low-D null. Score spontaneous activity at init &
  each checkpoint; test preplay-before-experience and experience-strengthens (selection).
- **1.5** **Environmental shaping (the moat)** (§6, Task E): show symmetry condition shifts the
  developmental schedule/endpoint; RA monotonicity & C2 sign-flip as developmental curves;
  symmetry-scales-multifield prediction.

### Phase 2 — Compute-gated upgrades (need Debbayan's GPU; raises venue ceiling)
- **2.1 Non-predictive control (Bar 5, highest priority GPU item).** Train matched no-rollout
  (autoencoder, k=0) nets. **Minimal version if compute is tight:** S1 (asymmetric, no symmetry
  confound) only, 3 seeds. Expand to all conditions if compute allows.
- **2.2** *(optional)* Re-run with **denser checkpointing** (e.g. every 2k early) for finer
  convergence-time resolution — current 6 checkpoints limit timing precision.
- **2.3** *(optional, power)* Re-run P5 at **n≥10/condition** so between-condition effects survive
  Bonferroni (report says ≥8 needed). Only needed if a reviewer demands the symmetry significance.
- **2.4** *(optional, de-confound)* **RatInABox** continuous-space replication of DTG to kill the
  MiniGrid city-block confound.
- **2.5** *(stretch, elevates topology claim)* Run **Project 4's aliasing-controlled loop envs**
  (annulus Betti-1=1, figure-8 Betti-1=2) so topology-before-geometry becomes *genuine loop topology*,
  not just HD-ring/manifold structure. Requires finishing the `NotImplementedError` stubs.

---

## 6. Engineering spec (file-level, with functions to reuse)

New code lives under `project5_symmetry/developmental/` (new package). Reuse existing functions; do not
reimplement metrics.

### Task A — `developmental/extract.py` : checkpoint → (hidden, positions)
```
build_eval_dataset(condition) -> TrajectoryDataset      # fixed seed; regen via generate_trajectories.collect_trajectory
load_checkpoint(ckpt_path, obs_size, act_size, k, trunc, device) -> pRNN_th
    model = train._build_model(obs_size, act_size, k, trunc, device, compile_cell=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model']); model.eval()
extract(model, dataset) -> (H, positions)               # REUSE train._collect_hidden_states(model, dataset, n, device)
```
- Reuse: `project5_symmetry/training/train.py::_build_model` (L611), `::_collect_hidden_states` (L96).
- Checkpoint dict format (`train.py::_save_checkpoint`, L635): `{'model','optimizer','step',...meta}`.
- Get `obs_size/act_size/k/trunc` from `arena_meta.json` + checkpoint meta (don't hardcode).
- **Comparability rule:** evaluate every checkpoint on the *same* fixed trajectory set.

### Task B — `developmental/curves.py` : metric curves across checkpoints
- For each (condition, seed, step): `H, pos = extract(...)`, then call:
  - `evaluation/metrics.py`: `srsa(H,pos,space_metric='euclidean')`, `srsa(...,'cityblock')`,
    `manifold_id(H)`, `spatial_evs`/`spatial_information` (via `compute_tuning_curves`),
    `sci(H,pos,symmetry_pairs)` (pairs from `arena.precompute_symmetry_pairs()`),
    `place_field_spatial_coherence(H,pos,arena_size)`.
  - `full_analysis_part1.py`: `compute_ra(H,pos,symmetry_order,grid_size)`,
    `compute_c2_contrast(H,pos)`, `compute_decode_error(H,pos,k=5)`, `compute_frac_tuned(H,pos)`.
  - `experiments/run_hd_ablation.py`: `compute_PAA_gain(...)` (factor it out into a shared module).
- Output: long CSV + `T_metric` table (`first step where metric crosses criterion for ≥1 checkpoint`;
  with only 6 checkpoints, also report the online sRSA-every-1k crossing for fine-grained geometry timing).

### Task C — `developmental/topology.py` : topology-before-geometry
- Spatial: `from project4...topological_metrics import compute_betti_numbers, compute_betti_correct,
  compute_topology_convergence_step, compute_geometry_convergence_step, compute_convergence_gap`.
  Ground truth for L-shape/square = `{betti_0:1, betti_1:0}` (simply connected → weak; use as 2-sheet/
  persistence-gap signal, **not** loop topology).
- **HD-ring (primary, new, free):** condition hidden states by the 12-bin HD one-hot, average per
  (HD-bin), run `compute_betti_numbers` on the 12 ring points (or on HD-binned centroids); ring ⇒
  Betti-1=1. Track the checkpoint where the ring criterion first holds vs `T_geometry`.
- Threshold-robustness: sweep sRSA threshold ∈ {0.35,0.4,0.45,0.5} and persistence-gap ∈ {1.5,2,3};
  bootstrap Betti barcodes (resample points) for CIs.

### Task D — `developmental/preplay.py` : autonomous read-out (Dragoi leg)
- **Autonomous rollout helper** (raw `pRNN_th` has none): drive the recurrent core with zero input +
  internal Gaussian noise for N steps, collect hidden. Mirror `utils/predictiveNet.py::spontaneous`'s
  noise injection but for `pRNN_th`. (Sanity-check against `project3 hippocampal_module.spontaneous`.)
- Decode: `from project4...replay_decoder import fit_position_decoder` (fit on wake H,pos).
- **Replace** `decode_replay_trajectory.forward_vs_reverse_ratio` with a real **sequenceness** stat:
  rank-order / weighted-correlation replay score, forward vs reverse separated, with **(i)** circular
  time-shift shuffle, **(ii)** cell-ID shuffle, **(iii)** dimensionality-matched random-dynamics null.
- Analyses: preplay@init/early-checkpoint vs trajectories the net masters later (above all three nulls);
  experience-strengthens (compare pre/post a single high-LR experience step, à la ExperienceReplayAnalysis).

### Task E — `developmental/shaping.py` : symmetry as rearing
- Reuse B's curves; contrast schedules/endpoints across S1/S2/S4; RA & C2 as developmental curves;
  multifield-fraction (needs a small **multifield extractor** added to `metrics.py`: per-unit #fields).

### Task F — `developmental/control_train.py` : non-predictive control (Phase 2, GPU)
- Clone `train_parallel_seeds` with the rollout objective swapped for reconstruction/autoencoder (k=0).
  Same arenas/seeds/checkpointing so the control is drop-in comparable to B/C/D.

---

## 7. Figures & paper skeleton (MVP)

| Fig | Content | Source |
|---|---|---|
| 1 | Framework: two read-out modes + training-as-development schematic; sRSA permutation-blindness proof; synthetic ground-truth validation of PAA/RA/C2 | analytic + Task 0.3 |
| 2 | **Emergence order**: metric crossing-times across checkpoints; HD/global-frame anchoring (PAA sub-threshold) before place/aliasing structure; rank-order vs Langston/Wills/Bjerknes | Task B |
| 3 | **Topology before geometry**: HD-ring (Betti-1=1) & manifold/persistence convergence (T_topology) vs geodesic-sRSA (T_geometry); within-net paired gap; threshold-robustness | Task C |
| 4 | **Preplay → replay**: sequenceness across checkpoints incl. init; preplay-before-experience vs 3 nulls; experience-strengthens | Task D |
| 5 | **Environmental shaping (moat)**: symmetry shifts schedule/endpoint; RA monotonicity; C2 sign-flip; multifield-fraction prediction | Task E |
| 6 | **Non-predictive control**: predictive objective necessary for the developmental signatures | Task F (Phase 2) |
| Table | Prediction table: model vs Ulanovsky/Dragoi/Moser-Wills, marked testable / confirmatory / future | — |

Narrative arc: *framework → emergence order → topology first → preplay first → symmetry reshapes it →
predictive learning is necessary (control) → falsifiable predictions.*

---

## 8. Venue, timeline, competitor

- **MVP target:** PLOS Comp Biol or eLife (developmental assay + honest power caveats + prediction table).
- **Priority stake-out (parallel, cheap):** NeurReps / "Symmetry & Geometry in Neural Representations"
  workshop or Cosyne abstract — the sRSA-blindness theorem + topology-before-geometry curves fit well.
- **Competitor to cite & differentiate:** **Abrate et al. 2026** (bioRxiv 2025.12.30.696864) already
  trains a predictive RNN that reproduces the developmental emergence order. **Differentiate on:**
  (1) explicit, threshold-robust **topology-before-geometry gap** (HD-ring + manifold); (2)
  **symmetry-as-rearing** (the unique axis nobody else has — *this is the moat, lead with it*); (3) the
  **preplay/autonomous read-out** unifying Dragoi. Move reasonably fast.
- **Higher ceiling (only if Phase-2 control + loop-topology land):** Nature Communications.
- **Do not** target Nat Neuro/Neuron primary or a unifying Perspective yet — data can't carry it.

---

## 9. Risks / limitations to state plainly in the paper

- Training-step ≈ postnatal-day is **ordinal only**; report rank-order agreement, not a day↔step map.
- L-shape/square are simply connected ⇒ the *free* topology signal is HD-ring + manifold structure,
  **not loop topology** (that needs Phase-2 P4 envs). Say so.
- HD oracle is noiseless; show anchoring survives noisy HD or soften the claim.
- Preplay is contested; only the shuffle+low-D-null-controlled, unseen-environment version counts.
- Between-condition symmetry stats remain weak at n=3–5; the paper rests on within-net ordering and
  power-immune results, not between-condition significance.
- 6 checkpoints limit convergence-time resolution; mitigate with online sRSA (1k) for geometry timing,
  denser re-run only if compute appears.

---

## 10. Open decisions (need your input as we go)

1. Confirm which {condition, seed, step} checkpoints actually survived off-repo (drives Phase-1 coverage).
2. When Debbayan's compute lands: spend it first on the **non-predictive control** (Bar 5), or on
   **denser-checkpoint re-run** (finer timing)? (Recommend: control first.)
3. How hard to push the **preplay leg** — full Dragoi treatment (higher risk/reward) or a compact
   "autonomous read-out matures alongside geometry" supporting result?

---

## Appendix — key paths
- Model: `utils/Architectures.py::pRNN_th`
- Train/build/collect/checkpoint: `project5_symmetry/training/train.py` (`_build_model` L611,
  `_collect_hidden_states` L96, `_save_checkpoint` L635, `train_parallel_seeds` L819)
- Stateless metrics: `project5_symmetry/evaluation/metrics.py`
- RA/C2/SCI/decode/frac_tuned: `project5_symmetry/full_analysis_part1.py`
- PAA: `project5_symmetry/experiments/run_hd_ablation.py::compute_PAA_gain`
- Topology: `project4_topology_before_geometry/evaluation/topological_metrics.py`
- Replay/decoder: `project4_topology_before_geometry/evaluation/replay_decoder.py`
- Spontaneous (wrappers): `utils/predictiveNet.py::spontaneous`,
  `project3_generalization/models/hippocampal_module.py::spontaneous`
- Arena: `project5_symmetry/environments/arena.py` (`passable_positions`, `precompute_symmetry_pairs`)
- Trajectories: `project5_symmetry/environments/generate_trajectories.py::collect_trajectory`
- Stats to fix: `project5_symmetry/Report/stats_results (3).csv` vs `Report/r_fixed.tex`
