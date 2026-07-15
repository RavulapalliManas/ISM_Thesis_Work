# Dynamical-systems probe of the fold — internal note

**Status: exploratory, not for the paper.** This records a dynamical-systems (fixed-point /
Jacobian / equivariance) analysis of the symmetry pRNN run on 2026-07-15, so the numbers and the
reasoning are not lost. It is a lab record, not a manuscript section. The conclusion is largely a
set of well-controlled negatives; it does not add a result the paper needs, but it does *rule out* a
reading (continuous-attractor / integrating-mode) that a systems-dynamics reader might reach for.

All scripts live next to this file (`analysis/dynamics/`). Everything is k=1 unless noted, arena
s2 (C2), model `utils.Architectures.pRNN_th` (LN+ReLU cell), checkpoints under
`/Volumes/Crucial X6/prnn_backup/checkpoints/horizon/k{0,1,3}/s2/{full,axis,parity,const}/seed_*`.
IC / rollout pool: the local `_tmp_fund_domain_data/s2` trajectory cache (300 trajectories; the main
corpus is gone but a rollout regenerates a usable pool). Single step used everywhere:

    h' = F(h,u) = ReLU( LN(W_ih u + W_hh h) + b ),   u = [obs(147), speed, onehot(heading)(4)]

Exact 500x500 Jacobians via `torch.autograd.functional.jacobian`. The C2 orbit map reuses the
repo's own `images(cells,'c2',ARENA)` (from `run_phase_decoding`) plus heading -> heading+2, so the
group action is the established one, not reinvented.

---

## The question

The paper shows the learned map represents the quotient X/G (axis folds, parity does not; matched one
bit, p=1.1e-5). This asks: **how is that fold implemented in the RNN's dynamics?** Is there an
integrating attractor? A reconstruction->prediction bifurcation in the flow? An equivariant vector
field?

---

## Probe 1 — Stage 0: does prediction build integrating modes reconstruction lacks? (`fp_stage0.py`, `jac_along_traj.py`)

**Method.** Load C2/full at k=0 (reconstruct) and k=1 (predict). Find fixed/slow points of the
input-clamped map by minimising ||F(h,u)-h|| from a pool of real hidden states; take the Jacobian
eigen-spectrum at each. Discrete-time marginal (integrating) criterion: |lambda| ~ 1.

**Result — the input-clamped picture is a clamp artifact.** It flips with the clamped observation:

    clamp        k0                         k1
    mean-obs     ~225 fixed pts, |lam|max~1.5   ~70 fixed pts, |lam|max~1.3   (k0 "richer")
    zero-obs     1 stable fixed pt, |lam|max~0.9  ~200 fixed pts, ~4-5 marginal (k1 "richer")

Both clamps are OFF the data manifold (a driven net never receives a blurry mean view or a blank
view), so the fixed-point structure under a constant invented input is not meaningful.

**Clamp-free control (`jac_along_traj.py`).** Leading |lambda| of the one-step Jacobian at real
visited states, with the real input, 5 seeds:

    median |lam|max:  k0 = 0.902,  k1 = 0.903  (per-seed Delta in [-0.004,+0.006])
    frac(|lam|<1):    ~0.72-0.79 for both

**Verdict: clean NEGATIVE.** No integrating-mode bifurcation. Reconstruction and prediction have
statistically identical local dynamics on-manifold; both are edge-of-stability everywhere. The
single-seed "prediction contracts" signal did not replicate. Lesson: a driven RNN's input-clamped
fixed points are meaningless unless the clamp is on-manifold; use real visited states.

---

## Probe 2 — Stage 2 anchor: is the fold in the recurrent state? (`stage2_statefold.py`)

**Method.** `fold_coincidence` (repo function): median cosine between a position's mean hidden state
and its 180-degree C2 partner (~1 => represented as one point). k=1, all four encodings.

    encoding   bits   C2 state-fold
    full        2      0.810   (lifts)
    axis        1      0.994   (folds, C2-invariant)
    parity      1      0.836   (lifts)
    const       0      0.993   (folds)

**Verdict.** The matched axis-vs-parity dissociation (0.994 vs 0.836, same one bit) is present in the
recurrent state, not just the readout. Validates the pipeline against the paper's known result.

---

## Probe 3 — Stage 2 flow-fold: does the local vector field carry the fold? (`stage2_flowfold.py`)

**Method.** Jacobian-spectrum similarity for C2 orbit pairs (matched on position AND heading) vs
random pairs, per encoding. 200 pairs, k=1.

    encoding   state-cos   flow-fold (orbit spectral-sim − random)
    full         0.759       0.026
    axis         0.996       0.030
    parity       0.814       0.025
    const        0.994       0.027

**Verdict.** The local Jacobian spectra are homogeneous across the state space. Orbit pairs are only
~0.03 more spectrally similar than random, *uniformly across encodings*, and this does NOT track the
representational fold (axis 0.030 ~= parity 0.025). The flow does not carry the fold.

---

## Probe 4 — the rho(g) conjugacy test: hidden dynamical symmetry? (`stage2_conjugacy.py`)

**Method.** Fit an orthogonal group action rho(g) on the *representation* (Procrustes on centered,
top-48-PC hidden states; train/test split, held out). Two held-out tests:
- representation: does rho map H(x) -> H(g.x)?  error vs identity baseline.
- flow (inverse-free linearised equivariance): does J_d(g.x) rho = rho J_d(x)?  vs identity and vs a
  shuffled-pair null. All in the 48-dim PC subspace where rho is well-determined.
3 seeds. Axis sanity check (states already identified => rho ~ I) passes every seed
(rho^2-I ~ 0.04, flow_rho ~ flow_I ~ 0.10).

    (medians over seeds 0,1,2)   REPRESENTATION       FLOW equivariance error
    encoding                     rep_rho   rep_I       flow_rho  flow_I  flow_shuf
    full                          0.16     0.94         0.42     0.45     0.69
    parity                        0.17     0.86         0.44     0.38     0.69
    axis                          0.07     0.07         0.10     0.10     0.64
    const                         0.12     0.12         0.18     0.13     0.62

**Verdict, two parts, stable across seeds:**
1. REPRESENTATION — positive. Even where the code *lifts* (full/parity), a fitted orthogonal rho
   maps orbit-related states at ~0.17 error vs ~0.90 for identity. The lifted code carries a genuine
   linear C2 group action: the orbit is not collapsed, but the group acts cleanly on state geometry.
2. FLOW — negative. The vector field is not equivariant under that action. rho does not beat the
   identity baseline for full and actively hurts for parity (flow_rho > flow_I every seed). Orbit-
   paired Jacobians are more alike than random (~0.42 vs ~0.69) only from flow homogeneity/proximity,
   not the group action.

---

## Synthesis

**The quotient X/G is implemented representationally, not dynamically.** The group acts cleanly on
*where states sit* (a faithful linear C2 action, even in the lifted full/parity codes); the *vector
field* does not respect it. There is no integrating attractor, no reconstruction->prediction
bifurcation in the flow, and the Jacobian field is a generic edge-of-stability substrate homogeneous
across the state space. This is consistent with the paper's mechanism (input-invariance under the
axis compass drives the fold) and it rules out a continuous-attractor / integrating-mode reading.

## Caveats
- Stage 0 is 5-seed; the conjugacy test is 3-seed; the state/flow-fold anchors are seed_00 (the
  conjugacy multi-seed run covers the same models and is consistent).
- The flow-equivariance test lives in a 48-PC subspace (captures ~0.9 variance); a genuinely
  full-rank rho is not identifiable from the available number of orbit pairs.
- All at k=1, arena s2. Not extended to C4 or to the horizon k-sweep.
- Scripts hardcode the drive checkpoint root and default to the local s2 cache.
