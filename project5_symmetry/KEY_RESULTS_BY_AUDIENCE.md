# Key results, by audience

The manuscript speaks to two readerships at once: computational-neuroscience / ML-theory readers who
care what a *learned* representation converges to, and systems-neuroscience readers who care whether
it matches real brains. This document sorts the paper's results by which readership each serves, so
the rebuild can foreground the shared spine and let each audience find its entry point. Every entry
is keyed to its table/figure; numbers trace to `NUMBER_MAP.md` → the CSVs in `Report/data/`.

Result IDs (H1–H4, plus the fuller R1–R55 inventory) match the audit's Explore-agent inventory.

---

## The shared spine — four headline results (both audiences)

These four carry the paper. Each is a place where the account could have come apart and did not.

- **H1 — Matched-entropy dissociation.** Two one-bit head-direction encodings, identical entropy and
  activation statistics, produce *opposite* maps in a symmetric arena (orbit-phase decoding 0.552
  for the invariant *axis* vs 0.955 for the equivariant *parity*, Mann–Whitney *p* = 1.1×10⁻⁵), yet
  are *indistinguishable* where there is no symmetry (Δ = 0.000, *p* = 0.97). This isolates
  **equivariance, not information**, as the cause. → Table 2, Fig 4.
- **H2 — Folding theorem + measured 1/|G| ceiling.** A proof that a predictive code collapses onto
  the quotient X/G, so a |G|-way decoder is bounded by 1/|G|; the C₄ networks hit it (*const* 0.275
  vs predicted 0.250; *axis* 0.482 vs 0.500). Theory and confirmation in one arc. → Theorem, Table 3.
- **H3 — Objective control.** Reconstruction discards the compass and folds every encoding alike
  (*axis* vs *parity* *p* = 0.59); one step of prediction restores the fold. The effect is
  **objective-driven, not architectural**. → Table 5.
- **H4 — Map = replay.** Offline coverage and sequenceness step up at the same horizon (k=0→1) the
  map resolves and fall in the same encoding order the map folds; the autoencoder loses both at once.
  The map and the replay are **one variable, read two ways**. → Fig 8.

---

## For computational-neuroscience / ML-theory readers

The "what does a learned code represent, and why" thread. Entry point: the quotient principle and
the ceiling theorem.

- **The quotient = bisimulation = successor representation, *derived not assumed*.** Under
  G-equivariance with a G-invariant prior and a belief-measurable policy, the belief process descends
  to a Markov process on X/G — the coarsest MDP homomorphism / bisimulation quotient — arising from
  self-supervised prediction with no reward, no explicit abstraction, and no group given to the
  network. → Discussion "Relation to other models," Table 7.
- **Zero-parameter input bound (Eq 1).** acc_max = ½ + ½·Pr[input distinguishes the orbit pair],
  computable from the arenas before any training; predicts the whole 12-cell pattern at mean absolute
  error 0.031, and rescues the C₁/*axis* 0.971 "shortfall" as exactly the 6.6% accidentally-aliased
  fraction. → Table 4.
- **Two orthogonal fingerprints.** Orbit-phase decoding is blind to information and tracks
  equivariance (axis vs parity Δ = 0.40); nonlinear mixed selectivity is blind to equivariance and
  tracks information (Δ = 0.04, and 88% present in C₁ where nothing folds). Two causes, two
  fingerprints, in the same networks. → Results "two fingerprints."
- **A learned, unanchored compass folds on its own.** Angular-velocity integration is invariant under
  any arena rotation by construction, so it builds the quotient with no imposed encoding (C₂ 0.526,
  C₄ 0.523; domain-R² intact). Removes the "invariant encoding is a modelling artefact" objection. → Fig 9b.
- **Identifiability result.** Tuning curves are neither sufficient (axis/C₁ 93% bidirectional yet map
  does not fold) nor necessary (const/C₄ on the ceiling with *flat* tuning) to infer a map fold. A
  clean statement of what a directional tuning curve can and cannot license. → Discussion.
- **Isometric quotient geometry.** The folded manifold is a faithful map of X/G, not a damaged map of
  X (Kruskal stress: axis fits X/C₂ at 0.190 vs 0.468 for the arena); fold ratio 0.457 < 1; a sham
  order-2 group control shows the preference is for the *symmetry*, not for compression. → Figs 5, 6.
- **Robustness of the ceiling** to observation format, network capacity, encoder identity, and
  equivariance-preserving noise — the proof turns only on the σ-algebra the (observation, action)
  stream generates. → Theorem robustness paragraph.

---

## For systems-neuroscience readers

The "does this match real brains" thread. Entry point: place-field repetition and the lesion.

- **Harland 2017 lesion — the biological load-bearer.** A *signed, selective* prediction, already run
  in animals with no fitted parameter: abolishing the head-direction signal leaves parallel
  (translation) repetition unchanged (sham 65% vs lesion 63%, *p* = 0.31) and *restores* the fold in
  radial (rotation) compartments (interaction *F*(1,10) = 13.60, *p* < 0.005, η² = 0.58). → Discussion.
- **Grieves/Spiers dissociation reproduced, with its behavioural cost.** Translation-related identical
  rooms fold (repetition 0.997, room decode 0.515≈chance); rotation-related ones separate (repetition
  −0.080, room decode 0.998). Scales to the four-room Spiers maze (decode 0.262 vs 0.25 chance). The
  fold carries a learning cost in the animals. → Fig 8c, Fig 10a.
- **Fuhs 2005 within-animal** dissociation (translation 0.71 = within-box; rotation 0.06, complete
  remap) and the **Harland–Calton reconciliation** (same lesion, two worlds: a cue card makes
  Calton's cylinder a C₁ arena with nothing to fold). → Discussion.
- **The cortical invariant compass is real.** Zhang 2022 (bidirectional/tetradirectional retrosplenial
  cells) and LaChance 2024 (cue-duplication makes postrhinal cells bidirectional, *p* = 2×10⁻¹⁰;
  entorhinal null *p* = 0.62) — but the map reads the *classical breaking* compass (the lesion shows
  it), so these cells are the fold's "directional shadow," not its antecedent. → Discussion.
- **Remapping-metric cast.** Population-vector correlation between a position and its symmetry image is
  0.98 under invariant encodings vs 0.28 under *full* — the network "fails to remap" exactly when the
  compass is blind to the symmetry. → Fig(SI).
- **BVCs are downstream of the compass, not upstream of the fold.** Boundary cells are abundant
  (41.6→61.2%) but vanish when the compass is ablated (→0.8% in C₄); axis and parity carry the same
  boundary code yet fold oppositely; no cell class carries the fold (border↔fold *r* = 0.16). → Fig 12.
- **The honest negatives.** Rate remapping (Spiers' partial-fold signature) tested and *not found*
  across a powered cue sweep; no prospective/anticipatory coding; grid cells and the entorhinal torus
  *absent* (architectural). These are marked, not hidden. → Table 6.
- **The decisive future experiment.** {1 cue, 2 cues} × {sham, HD lesion}: repetition predicted in
  exactly one of four cells (two cues, lesioned), with nothing free to fit; two cells already
  observed. → Outlook.

---

## Controls / robustness (support the spine, not headline)

Nonlinear-decoder replication of the chance fold; folded-not-broken within-domain R²; the sham-group
compression control; metric-free orbit cosine; noisy-compass (30% corruption) robustness; three
superseded readouts (rotational autocorrelation, odd power, spatial RSA) and why phase decoding is
primary; cross-seed reproducibility; decoder-unbiased synthetic null (reads 0.498); within-episode
drift; ODI non-confound; corridor-memory and horizon controls; bit-exact ensemble infrastructure;
omnidirectionality and topology-before-geometry caveats. Most relocate to Supplementary in the
rebuild, each with a main-text pointer.
