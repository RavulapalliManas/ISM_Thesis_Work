# Result CSVs behind the paper

Every number in `../biorxiv/main.tex` traces to a file here. Figures are regenerated
from these by `../../analysis/make_paper_figures.py`. Kept in-repo so the results
survive loss of the compute pod.

- `phase_full_n10.csv` -- orbit-phase decoding, n=10 (s1/s2), n=8 (s4). Table 2, abstract.
- `phase_groupB.csv`, `phase_horizon_k{0,1,3}.csv` -- horizon sweep (k=0..5).
- `compartments.csv` -- in-silico Grieves (2-room), n=8 per arrangement.
- `compartments4.csv` -- full Spiers four-in-a-row: 4 translation-related identical rooms fold
  completely (room decode 0.26 vs chance 0.25, repetition 0.99, within-room R2=0.96, n=8).
- `field_stats.csv` -- place-field counts + rate-map symmetry index.
- `map_quality_group{A,B}.csv` -- sRSA and cross-seed correlation.
- `replay_k{0,1,3,5}.csv` -- offline replay coverage vs wake/shuffle.
- `sequenceness.csv` -- replay sequenceness with time-shift and cell-shuffle nulls.
- `tda_topology.csv` -- topology-before-geometry (null result; see Limitations).
- `speed_*.csv`, `spectral_*.csv` -- initialization study.

- `phase_s4_c4.csv` -- C4 four-way phase decoding in the s4 arena (Table 3; n=4, 16 networks).
- `phase_nonlinear.csv` -- nonlinear phase-decoding control (linear/kNN/MLP, C2 arena).
- `phase_learned_c2.csv` -- LEARNED (angular-velocity) compass: folds C2 (0.531) and C4 (0.525)
  arenas, decodes C1 (0.969); the fold arises from an unanchored path-integrated compass, no
  imposed encoding. Retires the oracle-HD limitation.
- `manifold_robustness.csv` -- fold ratio in the full hidden space and under PCA, Isomap and
  t-SNE, showing the fold (ratio < 1 for axis/C2, > 1 otherwise) is not a PCA artefact.
- `remapping.csv` -- population-vector correlation between symmetry-related positions
  (Leutgeb-style remapping metric): ~0.98 under the invariant encodings (failure to remap /
  folded), ~0.25-0.29 under `full`. The matched contrast survives: s2 axis 0.98 vs parity 0.58.
- `cell_types.csv` -- per-network cell-type composition (place / border fraction, Skaggs info,
  fields per place cell rising with the fold). The HD-tuning column reflects the supplied
  encoding, not emergent tuning, and is not used as a claim.
- `prospective.csv` -- prospective-firing control across prediction horizons; a NULL (delta*=0),
  reported as such -- the place fields track current position, not a future one.
- `topology_robustness.csv` -- Betti-1 estimate under PCA-6/PCA-20/Isomap/spectral embeddings,
  testing whether the topology-before-geometry null is a linear-reduction artefact.
- `diffusion_maps.csv` -- diffusion-map geometry: topology loop scores + intrinsic dimension, and
  the C2 fold ratio in diffusion distance (axis 0.22 vs full 3.79). Figure: figS_diffusion.pdf.
- `hockeimer_field_di.csv`, `hockeimer_summary.csv` -- REAL-DATA test of the HD-invariance
  prediction in Hockeimer et al. (2023) CA1 city-block data (JHU doi:10.7281/T15HMQD4, external).
  Repeating fields in same-orientation (translation-related) alleys share directional preference
  (pair r=+0.23, shuffle p=0.007; mixed-model ICC=0.31, 5 rats) -- the fold in real neurons.
  Regenerate with analysis/hockeimer_reanalysis.py from the downloaded dataset.
- `isotypic_symmetry.csv` -- C4 isotypic spectrum (P0..P3, RA, odd) for the clean 17-network
  full-HD sweep; RA vs odd s1-vs-s2 separation ("Two natural readouts fail").
- `isotypic_hd.csv` -- same spectrum for the 112 HD-invariance networks; the confounded
  odd(parity) - odd(axis) drop per arena. Both from `../../analysis/run_spectrum.py`; the
  reported p-values are two-sided Mann--Whitney U on the `RA`/`odd` columns.

- `exp0_learned_phase_v2.csv`, `exp0_axisconst_phase_v2.csv` -- Experiment 0 (within-episode
  symmetry breaking): does the network maintain a coherent internal orbit-frame within a single
  episode, even though absolute phase is at chance across episodes? Target is z(t) = orbit
  phase at t XOR orbit phase at episode start (a per-episode-relative, not absolute, label);
  decoded with GroupKFold-by-episode logistic regression, `class_weight='balanced'`, reported
  as BALANCED accuracy (chance = 0.5 regardless of z's base rate). An earlier version of this
  script reported raw accuracy and was invalidated: P(z=1) rises from ~0.08 near episode start
  to ~0.46 by 100+ steps purely from position continuity under a random walk (independent of
  the hidden state), so a majority-class-only classifier scored ~0.93 at t=0-10 using zero real
  information -- caught by comparing against the majority-class baseline, now reported alongside
  every number in these files. Corrected result, n=10 seeds each of learned/axis/const, C2
  arena: within-episode balanced accuracy is nominally above chance (Wilcoxon signed-rank vs
  0.5, exact two-sided, uncorrected) at every dwell bin for all three encodings, decaying from
  ~0.66-0.74 near episode start toward the across-episode baseline (~0.52-0.53) by ~100+ steps.
  The effect at the longest bin (100+ steps) is far weaker for `learned` (p=0.049, right
  at the edge) than for axis/const (p<=0.0098), so `learned`'s decay reaches the boundary
  of detectability while axis/const remain more robustly above chance there -- but none actually
  drops to non-significance. Decodable-but-drifting (outcome c) throughout, not the clean
  spontaneous-break (a) or instantaneous-fold (b) alternatives. From
  `../../analysis/run_within_episode_phase.py`.

- `weakbreak_snr.csv` -- referee item 1, graded-cue SNR sweep (translation compartment maze,
  fixed pixel noise sigma=0.05, room-B tint eps swept over 8 values spanning single-step
  d'=eps*sqrt(N)/sigma in {0..3}, n=8 seeds/point, from the rebuilt float-precision cue
  mechanism in `environments/arena.py`/`generate_trajectories.py` and the noise injection in
  `experiments/run_multi.py`). Fixes the old amplitude-only sweep, which was a step function in
  the Eq. 1 bound (every prior amplitude sat at the ceiling). room_gen and room_seen both rise
  smoothly and monotonically with d' (Spearman rho=0.92 / 0.99, n=64, p<1e-25; d'=0 vs d'=3
  exact two-sided Mann-Whitney U p=0.000155, n=8 v 8), confirming the sigmoid design works.
  BUT the predicted Spiers-signature ordering (rate discrimination departing at lower d' than
  position, repetition surviving) was NOT observed as designed: field-shape repetition stays
  at ceiling throughout (~0.997, p=0.88 d'=0 vs d'=3, rho=0.17 n.s.) and the rate-asymmetry
  readout (median relative firing-rate difference between rooms, top-100-variance units, no
  position matching) is flat across the whole sweep (p=0.88, rho=0.02 n.s.) -- no evidence this
  measure picks up a rate-based signal at all, let alone one that leads position. Room
  decodability stays well below even the conservative single-observation Bayes ceiling at every
  d' (e.g. d'=3: room_gen=0.59 vs acc_max(T=1)=0.93), consistent with "information present, not
  fully recruited." Honest negative on the specific rate-leads-position prediction, as
  pre-registered; reconciliation with Spiers' partial fold has to come from the residual (the
  d'=0 room_gen=0.510, itself significantly above chance, Wilcoxon p=0.023, n=8) rather than
  from a graded rate/position dissociation in this architecture. From
  `../../analysis/run_weakbreak_snr.py`.

- `isomap_manifold.csv` -- Tier-2 item 7, "the manifold IS the quotient." A genuine Isomap
  embedding (n_neighbors=150, n_components=2, cosine metric, Levenstein et al.'s parameters),
  not PCA -- complements the existing `fig:manifold` panel (PCA, single rotational C2 arena,
  axis vs full), whose "Isomap/t-SNE" claim is numeric only (`manifold_robustness.csv`), not a
  displayed embedding. Here: compartment maze, translation vs rotation, matched at k=3 (the
  Theorem predicts translation folds, rotation lifts). New diagnostic: matched_cell_dist_norm
  -- for every local cell present in both rooms, the distance between room A's and room B's
  per-cell centroid IN THE EMBEDDING, normalised by the embedding's spread. n=8 seeds/condition:
  translation 0.133 +- 0.015 sd, rotation 1.308 +- 0.157 sd (exact two-sided Mann-Whitney U
  p=0.000155) -- room B's copy of a cell lands almost on top of room A's under translation and
  nearly 10x further away under rotation. Figure: `fig_isomap_manifold.pdf`
  (biorxiv/figures), one representative seed per condition, coloured by room identity. From
  `../../analysis/run_isomap_manifold.py`. Checkpoints: horizon_k3 (referee item 2's sweep);
  trajectory data regenerated locally (deterministic, seed-reproducible) after the training pod
  was lost -- nothing scientifically depends on the pod's survival. `isomap_manifold_3d.csv`
  repeats the same matched-cell-distance check at n_components=3 instead of 2 (same n_neighbors,
  cosine metric): translation 0.172 +- 0.019 sd, rotation 1.720 +- 0.221 sd (n=8, exact
  two-sided Mann-Whitney U p=0.000155) -- the separation holds, if anything slightly larger, so
  the 2D panel is not hiding structure that only resolves in higher dimensions.

- `horizon_sweep_compartment.csv` -- referee item 2, horizon sweep on the compartment maze
  (translation, k in {0,1,3}; k=5 training was interrupted when the GPU pod was lost mid-run,
  k=10 never started -- PARTIAL sweep, 3 of 5 planned points). n=8 seeds/k, room decode
  conditioned on steps-since-room-entry (`analysis/run_compartments_horizon.py`), steady-state
  (>10 steps) reported as primary per the existing Methods convention. Steady-state room_gen:
  k=0 0.508+-0.014 (Wilcoxon vs chance p=0.25, n=8), k=1 0.502+-0.011 (p=1.00), k=3 0.510+-0.010
  (p=0.055) -- none individually significant, and NO rise with k across the three points tested
  (Spearman rho=+0.07, p=0.73, n=24 seed-level points). Repetition stays near ceiling throughout
  (0.98-0.99). Supports "the fold is architecture-robust to horizon choice" over "the residual
  is horizon-limited" for k=0..3, but this is NOT yet the full story: k=5 and k=10 (the larger
  horizons, where an objective would have the most reason to recruit a weak corridor cue) are
  still missing, so a late-emerging horizon dependence at higher k cannot be ruled out from this
  data alone.

- `isomap_manifold_k01.csv` -- same matched-cell-distance check as `isomap_manifold.csv`, at
  k=0 and k=1 instead of k=3, n=8 seeds/condition. k=0 (autoencoder control, no rollout):
  translation 0.201+-0.015, rotation 0.213+-0.027, Mann-Whitney p=0.23 (NOT significant) -- the
  geometric fold is itself prediction-dependent: with nothing to predict, translation and
  rotation are not geometrically distinguishable. k=1: translation 0.188+-0.013, rotation
  0.263+-0.043, p=0.000155 (floor) -- already separated with one step of rollout, growing much
  larger by k=3 (0.13 vs 1.31, see isomap_manifold.csv). Independent, decoder-free confirmation
  of "the fold requires prediction," in Report/elife/supplementary.tex Fig. S12's section.

Numbers not backed by a CSV here are deterministic arena enumerations (the Eq. bound's
distinguishable/predicted columns, the 6.6% and 1228/1296 counts, the ODI values) or are
figure-derived (manifold fold ratios, gridness); all are reproducible from the analysis scripts.

- `isometry_quotient.csv` -- IS THE NEURAL MANIFOLD A MAP OF X, OR OF X/G? Kruskal stress-1 (optimal
  scale) of neural geodesic distances (Isomap on position-conditioned means) against three candidate
  metrics: the arena metric d_X, and the quotient metrics d_{X/G}(x,y) = min_g ||x - g.y|| for G = C2
  and C4. 112 networks (3 arenas x 4 encodings x 10 seeds; 8 for s4).
  Headline: the HD encoding determines WHICH SPACE the manifold is a metric map of. By argmin stress,
  110/112 networks: `full` and `parity` -> X; `axis` -> X/C2; `const` -> X/C4. In the C2 arena the
  folded (`axis`) code fits X/C2 at stress 0.190 against 0.468 for the arena itself, the mirror image
  of `full`, which fits X at 0.097 against 0.485 for the quotient.
  Controls: `stress_sham` uses a SHAM order-2 group (translation by half the arena) with the same
  "min over images" compression as C2 but which is not a symmetry of any arena. This is essential,
  because d_{X/G} <= d_X pointwise, so ANY group shrinks long distances and a compressed code would
  prefer a quotient metric for trivial reasons. It does not: s2/axis prefers the true C2 over the
  sham by 0.343. `stress_shuffled` is a position-shuffled ceiling (~0.56).
  `fold_cos_c2/c4` is a metric-free check: median cosine between H(x) and H(g.x). s2/axis = 0.990
  (the orbit is one point) vs s2/parity = 0.741 (same one bit, opposite invariance; U=100, p=9.1e-5).
  NOTE the s4/`axis` row is marginal (X/C2 0.364 vs X/C4 0.347) -- C4 compresses more than C2 and has
  no sham control, so that row should not be over-read.

- `cell_properties.csv` -- single-cell properties across arena symmetry (C1/C2/C4) x HD encoding,
  112 networks. Skaggs spatial information, sparsity, selectivity, field count, field area,
  coherence, place-cell fraction, and a variance decomposition into EV_pos / EV_hd / EV_add (best
  additive model f(x)+g(h)) / EV_conj (full conjunctive f(x,h)), with `mixed` = EV_conj - EV_add,
  i.e. NONLINEAR MIXED SELECTIVITY: the variance position and heading explain jointly but neither
  explains additively.

  HEADLINE. Ablating the compass does two SEPARABLE things, and only one of them is the fold.
  Decomposing the full->const effect into a part present in C1 (no symmetry, so nothing can fold =
  degradation) and the extra part in C2 (the interaction = folding):

      metric          C1 full->const   C2 full->const   FOLD (C2-C1)   p
      spatial_info        -0.329           -0.305          +0.024      0.011
      sparsity            +0.093           +0.088          -0.006      0.98
      selectivity         -0.362           -0.255          +0.107      0.21
      field_area          +7.17            +8.09           +0.92       0.26
      coherence           -0.330           -0.416          -0.086      1.0
      mixed               +0.089           +0.097          +0.008      1.2e-4
      n_fields            +0.584           +0.940          +0.356      9.1e-5   <-- the fold

  FIELD COUNT carries BY FAR the largest symmetry-specific component (0.61 of the degradation
  effect, i.e. the fold adds 61% on top of what the ablation does where nothing can fold). With
  bootstrap CIs over networks, coherence carries a smaller real one (-0.26: a folded map is less
  locally coherent, since a cell now fires in two places), and spatial information (+0.07),
  sparsity (-0.06) and mixed selectivity (+0.09) have small but detectable components. Field area
  and selectivity are not distinguishable from zero. So the bulk of what ablating the compass does
  to a place cell -- less spatial information, higher sparsity, bigger fields, worse coherence --
  happens just as much in the C1 arena, where there is no symmetry to collapse onto. That part is
  information loss, not folding.
  Consequence for reading the animal data: an HD lesion that lowers spatial information and raises
  sparsity (Harland et al. 2017 report exactly this) is showing degradation; only REPETITION
  diagnoses the quotient. This is also why Calton et al. (2003), recording in a cue-controlled
  cylinder where the symmetry is broken, find no change in field number (1.29 -> 1.53, n.s.) --
  there is nothing to fold onto.

  MIXED SELECTIVITY. The units are strongly conjunctive for position-by-direction (90-97% of units
  have mixed > 0.05), and ablating the compass makes them MORE so (0.225 -> 0.332 from full to
  const), while pure directional tuning collapses (EV_hd 0.109 -> 0.003). Removing head direction
  from the ACTION stream does not remove heading from the code: heading then enters only through
  the egocentric view, which binds it to position. This bears directly on the CA1 limitation, where
  a conjunctive place-by-direction code is the alternative we cannot separate from folding.

- `bvc_tuning.csv` -- DOES THE NETWORK LEARN BOUNDARY-VECTOR CELLS? Referee-proofing: Uria et al.
  (2020) showed a purely predictive RNN grows BVCs, so the charge is "your derivation is the BVC
  model of repetition (Grieves, Duvelle & Dudchenko 2018) with extra steps." Every unit's rate map
  is fitted to the best single idealised BVC (Gaussian over distance to the wall at a given
  ALLOCENTRIC bearing; 4 directions x 13 distances x 5 widths) and, for comparison, to the best
  single Gaussian place field (same functional family, same number of free parameters).
  ANSWER: yes, and it helps us. The network grows BVC-like units -- 41.6% of units in C1 under
  `full` are better fitted by a BVC than by any place field, 61.2% in C4. But BVC tuning REQUIRES
  the compass: ablating it drops that to 14.0% (C1) and 0.8% (C4). A BVC's bearing is allocentric
  and undefined without a heading signal.
  DECISIVE: `axis` and `parity` carry the same one bit and support comparable boundary populations
  (C2: 29.2% vs 22.1% BVC-like; parity is if anything the better boundary fit, mean r 0.482 vs
  0.449), yet only `axis` folds (orbit phase 0.552 vs 0.955). Same boundary code, opposite maps.
  No BVC model can produce that, because it has no representation of WHICH bit the compass carries.
  Boundary cells are downstream of the compass, not upstream of the fold.

- `manifold_fold_ratio.csv`, `manifold_coords.csv` -- the geometry figure, which was an ORPHAN.
  `manifold_s2.png` backed a MAIN-TEXT figure and no script in the repo produced it, and it was a
  raster PNG in an otherwise-vector paper. `analysis/export_manifold.py` regenerates it from the
  checkpoints. Fold ratio = d(x, R^2 x) / d(x, neighbour) in the FULL hidden space, so it is not a
  projection artefact; below 1 means orbit partners sit closer than spatial neighbours, i.e. the
  code folded. n = 10 networks per condition (the old figure was a single representative network):
  axis/C2 = 0.457 +/- 0.004 (FOLDED), axis/C1 = 1.872, parity/C2 = 2.083, full/C2 = 2.675.
  The ordering and the sub-1 result for axis/C2 reproduce; the exact values differ from the old
  n=1 figure, whose numbers cannot be checked because it had no generating script.

- `anisotropic_decoding.csv` -- SPIERS' MEASURE: the fold, converted into a cost. Every other
  measure in this paper describes the code; this one describes what the code COSTS an animal that
  has to use it, which is the currency an experimentalist reads in.
  Spiers et al. (2015) decoded position from CA1 in four identical compartments and found the error
  was strongly ANISOTROPIC -- large along the axis on which the compartments repeat, small across
  it -- and that collapsing the compartments onto one frame removed it. Our two-compartment maze has
  the same geometry (room B is room A shifted 13 rows in the same columns), so the repetition axis
  is the row axis. We decode the occupied cell with a classifier (Spiers used maximum-correlation
  template matching; both COMMIT to a cell, which a regression would not -- a regression hedges
  between the rooms and lands in the corridor, whereas a folded code puts the animal CONFIDENTLY in
  the wrong room), and split the error into the two components. n = 8 networks per mode.

      mode          along   across   collapsed   wrong room
      translation    5.99     0.00       -0.00        46%     <- folded
      rotation       0.04     0.00       -0.00         0%

  The folded network's error is ENTIRELY "wrong room, right place within it": 46% of states decode
  to the wrong room, and 0.46 x the 13-row room offset = 6.0, which is the 5.99 observed. Collapsing
  the rooms removes the error completely. Translation vs rotation: U=64, p=4.7e-4 (n=8 v 8).

  COMPARE TO SPIERS WITH CARE. Their reported errors are in "bins", but the bin size is NOT the
  3.3 cm their Methods state for the correlation analysis -- at 3.3 cm their errors would be worse
  than chance, which no working decoder produces; it is ~1 cm. So compare the ANISOTROPY and the
  collapse, not the absolute numbers. Their strongest control is the bin-SWAP: scrambling compartment
  identity cost their decoder nothing (19.84 vs 20.18), i.e. there was no compartment information
  there to begin with. Our 46%-wrong-room (chance = 50%) is the same statement.
