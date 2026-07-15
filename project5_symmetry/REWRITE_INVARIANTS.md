# Rewrite invariant set — the checklist the rebuilt manuscript must pass

Extracted from `main_best.tex` (backup: `scratchpad/main_best_ORIG.tex`) BEFORE the Feynman rebuild.
Per SMVF §12: no edit may change any item below. After the rebuild, every line here is re-verified
against the new text. Floats (12 figures, 7 tables, the Theorem, the Proof, Eq 1) are preserved
**byte-exact** by programmatic extraction from ORIG — not retyped — so their internal numbers cannot
drift; this file guards the *prose* claims that surround them.

## 1. Theorem proposition (must be unchanged, verbatim)
Free finite group action on X; G-invariant start; G-equivariant likelihood; belief-measurable
policy; G-invariant HD encoding φ(h+g)=φ(h) ⟹ position posterior is G-invariant ⟹ any orbit-element
decoder has accuracy ≤ 1/|G|. **It is an upper bound, not a prediction.** The free-action assumption
(no rotation fixes a cell, satisfied by the even grid) gives every orbit exactly |G| members.

## 2. Equation 1 (eq:bound) — semantics unchanged
acc_max = ½ + ½·Pr[(x,h) ≁ (R²x, h+2)] = ½ + ½·(fraction of orbit pairs the input distinguishes). An
upper bound computed from the arenas with no free parameters.

## 3. Causal claims (each rests on a test that could have failed — do not strengthen/weaken)
- Equivariance, **not** information, drives folding (H1: axis vs parity, matched 1 bit).
- The fold is **objective-driven, not architectural** (H3: reconstruction vs prediction).
- Map and replay are **the same variable** read twice (H4), not two coincident phenomena.
- The hippocampal map reads the **classical breaking** compass, not the invariant one (evidence: the
  Harland lesion + the intact-radial null). *Assumption flagged in text:* the invariant channel's
  survival through the lesion is assumed, not measured.
- Information loss ≠ folding: the in-silico lesion degrades orbit variance without regard to symmetry
  (C₁ −37.6% ≈ C₄ −38.1%) and folds phase only where a symmetry exists.
- BVCs are **downstream** of the compass, not upstream of the fold.
- The learned angular-velocity compass folds **by construction** (invariant under any arena rotation).

## 4. Calibrated hedges (must survive with identical strength — these are the most fragile invariants)
- Compartment two/four-room residual (0.515 / 0.262): "small but real, not noise" — **neither rounded
  away nor leaned on**; the incomplete fold is revisited in Discussion.
- CA1 city-block reanalysis (r=0.23): **"consistent with the quotient but not diagnostic of it"**; the
  *model's own* prediction (r=0.62), not the animal reanalysis, is the load-bearing result.
- Zhang invariant compass is **"approximately," not exactly, invariant** ("we say approximately and
  mean it"); cross-box comparison is between cells, within-cell only 1/8 bidirectional survives.
- LaChance entorhinal null (W=207, p=0.62): **"a failure to reject, on a test with no reported power,
  and we treat it as such."**
- C₄ 8/8 axis geometry exception and the 103/112 metric assignment: **flagged, not resolved**; "no
  claim rests on the geometry metric where it and the decoder part company."
- Alexander 2020 egocentric-sampling account: **"untested here rather than refuted."**
- Coverage: a shuffled rollout covers 1.57× (more than any real net); coverage measures extent not
  structure; **sequenceness** carries the structured-replay claim.
- Rate remapping: **"an honest negative on the specific rate-leads-position prediction, as
  pre-registered"**; reconciliation must come from the zero-cue residual (0.510).
- Calton field count: **"not reproduced," "the sign is wrong," "mismatch"** rows — kept verbatim in strength.
- Cross-seed correlation (>0.96): **"descriptive, no null claimed"** — "a statement that the fold is
  reproducible, not a test that it could have failed."
- Grid/torus and HD-ring rows: **"absent," architectural, "we do not claim"** them.
- No forward-vs-reverse replay claim (sequenceness is an absolute value).
- Field-count statistics **do not match** Rich 2014's Poisson (Fano 0.65, under-dispersed) — stated as
  a mismatch.
- Prospective coding: **"tested, not found."**
- Fuhs transient/anchoring and Skaggs 1998 partial remapping: qualifications kept.

## 5. Hypothesis set (closed — no new hypotheses may appear)
The two competing hypotheses are exactly: folding tracks (a) information the compass carries, or
(b) invariance/equivariance. The paper's thesis is (b). No third mechanism is introduced.

## 6. Numbers → source (full map in NUMBER_MAP.md; all verified to the digit at audit)
Every number in the rebuild is re-checked against `NUMBER_MAP.md` / the CSV at the moment of writing.
Load-bearing values that must appear unchanged: phase table (0.981/0.972/0.971/0.966; 0.978/0.955/
0.552/0.552; 0.984/0.933/0.544/0.521); C₄ (0.918/0.860/0.482/0.275, predicted 0.500/0.250); horizon
(k0 spread 0.0052, p=0.59; k1 p=0.00216); bound MAE 0.031, C₁/axis 0.967 (6.6% aliased); dissociation
p=1.1×10⁻⁵ / p=0.97; ceiling residual t(9)=34.8/35.2, p≈6×10⁻¹¹; geometry 103/112, 8/8, stress
0.190/0.468/0.097/0.485, sham 0.343, cosine 0.990/0.741, fold ratio 0.457; lesion −37.6/−30.9/−38.1,
phase 0.986→0.566 / 0.985→0.567 / 0.862, field −11/+16/+20; Harland F(1,10)=13.60 η²=0.58, d values;
replay 1.30/0.80/0.45/0.31, seq 0.53→0.21 floor 0.26/0.17, shuffle 1.57×, AE 0.37×; noisy 0.556/0.922;
learned 0.971/0.526/0.523; mixed 0.235/0.292/0.331/0.333, pos-var 0.505→0.312; population 93.1%/54.3%,
border-r 0.16; BVC 41.6→14.0 / 61.2→0.8, axis/parity 29.2/22.1, r 0.482/0.449; field size step
33.8/40.7/42.1/39.7; corridor 0.56→0.51 / 0.71→0.92→0.98→0.99; cue sweep ρ=0.92/0.99, residual 0.510;
Hockeimer r 0.23, 127 pairs, 40 cells, 5 rats, ICC 0.31; ODI 0.336/0.288/0.262; Fano 0.65; ~480/500
place fields; ~56,000 units; ~76–96% place cells, ~30% border.

## 7. Float inventory that must all remain cited (no orphans)
Figures: overview, setup, dissoc, manifold, isometry, ceiling, function, generality, fold, brain,
lesion, bvc (12). Tables: encodings, phase, c4, bound, horizon, bio, models (7). Plus Theorem, Proof,
Eq 1. SI figures referenced from main: S10 (CA1), S11 (coset/phase), S12 (isomap), S13 (a population,
b/c isotypic). Every one must be `\ref`-cited after the rebuild.
