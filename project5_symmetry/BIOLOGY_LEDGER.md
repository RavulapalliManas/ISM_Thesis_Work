# Biology ledger: Zhang 2022, LaChance 2024, and what the quotient law may claim

Written 2026-07-14. A record of a multi-pass verification of the two papers that ground the
"invariant compass" premise, what survived, what was refuted, and what changed in the manuscript as a
result. Every numbered item is either VERIFIED against the primary source or explicitly marked
otherwise. Nothing here may be cited from this file alone -- this is a ledger, not a source.

---

## 0. The one-line summary

The premise survives, but **we were pointing it at the wrong claim**. Zhang's multidirectional cells
are not the compass the hippocampus reads; they are a *second instance* of the fold. And our Outlook
experiment, as written, contradicted our own central argument. Both are now fixed.

---

## 1. VERIFIED FACTS (primary sources, quoted)

### 1.1 Zhang, Grieves & Jeffery 2022, J Neurosci 42(49):9227-9241
DOI `10.1523/JNEUROSCI.0619-22.2022` (**not** 1150-22; our bib was already correct).
PMC9761682. 18 male Lister hooded rats.

| | cells | classic HD | multidirectional |
|---|---|---|---|
| 4-box (fourfold) | 743 / 8 rats | 75 (10.1%) | 67 TD (9.0%) |
| 2-box (twofold)  | 575 / 14 rats | 27 (4.7%) | 48 BD (8.3%) |

- **The 2-box vs 4-box contrast is BETWEEN CELLS.** "Every cell was recorded in one of the two
  1-boxes and in either the 2-box or the 4-box." No neuron was recorded in both symmetric boxes.
  **Never write "the same cells showed twofold tuning in the twofold box and fourfold in the
  fourfold box."** That sentence would be false.
- **The within-cell contrast is multifold-box vs 1-box, and there the pattern VANISHES**: only 3/8
  BD and 3/65 TD cells still scored above shuffle in the onefold arena. Note this is mostly cells
  going *non-directional* (56-62% lost directional specificity entirely), not becoming cleanly
  unidirectional. Do not describe the 1-box as "onefold tuning".
- **The symmetry tracked is VISUAL-GEOMETRIC, not the environment's total symmetry.** The odour
  (lemon/vanilla) deliberately broke the global symmetry, leaving a *twofold* multimodal
  environment -- and the cells were still *fourfold*. Always say "visual-geometric symmetry".
- Subtypes: **BC (between-compartment)** cells are unidirectional *within* a compartment with the
  preferred direction rotating 90 deg between compartments (9/67 TD, 9/48 BD; V = 7.98, p < 0.001).
  **WC (within-compartment)** cells are multidirectional *inside* one compartment (29/67, 22/48).
- Darkness: 21/28 TD cells held the pattern in dark trial 1 (TD score did drop, F(2,78)=5.26,
  p=0.007). Only **4 cells in the entire dataset** met an egocentric-boundary criterion.
- Coexistence: classic HD cells are intermingled, globally referenced, and **electrophysiologically
  distinct** -- peak-trough 162.75 us (HD) vs 402/406 us (TD/BD), F(2,100.98)=159.73, p<0.001.
- **76% of MD cells are in DYSGRANULAR RSC** (vs 9% of HD cells; chi2=101.89, p<0.001).
- Their only downstream statement runs the WRONG WAY: "inputs coming into the retrosplenial area
  **from** the subicular region". Hippocampus -> RSC. There is no place-cell data in this paper.

### 1.2 LaChance & Hasselmo 2024, Nat Commun 15:8025
DOI `10.1038/s41467-024-52315-4`. PMC11399390. 6 female Long-Evans rats (4 in the AB experiment).
Title: "Distinct codes for environment structure and symmetry in postrhinal **and** retrosplenial
**cortices**".

The AB session: 120 cm square, cue A on the south wall, an identical cue B added on the north wall.
Session order **A1, AB, A2**, 20 min each.

| region | n | bidirectionality AB vs A1 | verdict |
|---|---|---|---|
| POR HD | 34 | W = 1, **p = 2.33e-10** | folds hard |
| RSC HD | 37 | W = 111, **p = 1.41e-4** | folds |
| MEC/PaS HD | 30 | W = 207, **p = 0.62** | **DOES NOT FOLD** |
| MEC/PaS grid | 61 | W = 879, **p = 0.63** (rate-map corr) | unchanged |

- **The fold is PARTIAL.** Modulation by cue B is reliably *smaller* than by cue A: POR W = 49,
  p = 3.53e-5; RSC W = 3, p = 7.28e-11. Always say "approximately" or "partially" invariant.
- **RSC splits by waveform.** Narrow-waveform (<200 us peak-trough) RSC HD cells do **not** become
  bidirectional; broad-waveform ones do. Same split as Zhang's HD/MD electrophysiology.
- **HYSTERESIS.** In A2 (cue B removed) POR cells *remain* bidirectional (W=118, p=1.57e-3), RSC too
  (p=0.012). The code does not un-fold when the symmetry goes. Our law must be stated as a property
  of a *learned* map, never as an instantaneous readout of the current environment.
- **The MEC HD null (W=207, p=0.62) is the load-bearing MEC arm -- NOT the grid null.** A hexagonal
  lattice is inversion-symmetric: a C2 rotation about the arena centre returns a grid of identical
  spacing and orientation, displaced only by a bounded phase shift (<= lambda/4). Their rate-map
  correlation measure is therefore **structurally attenuated** to a C2 fold, and its sensitivity was
  never verified. A referee who knows lattice symmetry can dismantle "grid maps unchanged, therefore
  MEC did not fold" in one sentence. Concede the caveat before anyone else raises it.
- Conjunctivity: **60% of POR EB cells and 33% of RSC EB cells are also HD-tuned.**
- Symmetry analyses (1/2/3/4-fold) were run **ONLY on the single-cue square**, and **only on EB
  cells**. There is **no EB-cell analysis of any kind in the AB session**. Verified across main text,
  all 15 supplementary captions, and the peer review file.
- Order-specificity control (forced into existence by Reviewer #1): in the square, 2-fold and 3-fold
  symmetry are **at chance**; 4-fold is enriched only in RSC (52%), 1-fold only in POR (26%). The
  code picks out 4, not 2 or 3. Worth citing.
- **AB spike trains are NOT public.** The Zenodo/GitHub deposit holds figure-panel derived data plus
  16 raw *square-session* example cells. Any AB reanalysis must come from Hasselmo directly.
- Peer review: only two reviewers. **Neither questioned the AB session or the nulls.** Reviewer #1
  demanded a second, differently-shaped arena (asymmetric triangle / trapezoid); **the authors
  refused and de-scoped**, deleting "centroid" from the abstract.

### 1.3 Anatomy (bears directly on which compass the map reads)
- **Dysgranular RSC (area 30) has NO direct projection to the hippocampal formation.** The RSC->HF
  projection is granular-specific (area 29) and targets **subiculum only** (Sugar, Witter, van Strien
  & Cappaert 2011, Front Neuroinform 5:7; van Groen & Wyss 1992, J Comp Neurol 315:200-216).
  Shortest dysgranular route to CA1 is **2-3 synapses**, best-supported via postsubiculum ->
  MEC L3 -> CA1.
- **POR does project monosynaptically to dorsal CA1** (weakly) -- Agster & Burwell 2013. So if any
  invariant compass reaches the hippocampus, it is the *postrhinal* one, which is also where
  LaChance's fold is strongest.
- STATUS: **PENDING INDEPENDENT RE-VERIFICATION** before these are cited. Keys `sugar2011` and
  `vangroen1992` are cited in main_best.tex but **were not yet in references.bib** as of writing.

---

## 2. WHAT WAS REFUTED

### 2.1 "Zhang's multidirectional cells are the G-invariant compass the hippocampal map reads."
**REFUTED**, by two independent facts.
1. **Harland's intact rats.** In *radially arranged* compartments -- a rotationally symmetric layout,
   precisely where RSC goes multidirectional -- intact place fields did **NOT** repeat. If the
   hippocampus read the invariant channel, that is the one condition where it would have folded with
   no surgery at all.
2. **The anatomy** (1.3): those cells are dysgranular and at least two synapses from a place cell.

**Consequence:** Zhang is re-roled from *antecedent* to *consequent*. It is not the compass that
folds the map; it is a **second instance of the same fold**, in a cortical directional code. This is
both more defensible and more interesting. DONE in the manuscript.

### 2.2 Our own Outlook experiment. (This was an internal contradiction.)
The draft staked the account on: *record hippocampal place cells in LaChance's two-cue square;
predict C2 place-field repetition.*

**Our own argument predicts the opposite.** Harland shows the map reads the compass that *breaks*
the symmetry. LaChance shows that in AB, the classical compass **stays unidirectional**
(MEC HD, W=207, p=0.62) -- the ring attractor carries an absolute reference forward from A1. So in an
intact rat the symmetry is still broken for the hippocampus, and the law predicts **no fold**.

As written, the experiment we said we would stake the account on would have **falsified us if it came
out the way we predicted**. MUST BE REWRITTEN. (Status: pending.)

### 2.3 "A rate-map symmetry analysis in AB would be a novel test."
**REFUTED as novel.** For the 60%-conjunctive EB x HD population it is close to an arithmetic
corollary of their own Fig. 4. Egocentric bearing satisfies b(-x, th+180) = b(x, th); so if the
directional tuning becomes 180-periodic (which Fig. 4 *measures*) and occupancy is C2-symmetric, the
marginal allocentric rate map satisfies R(-x) ~ R(x) **by construction**. The authors state the
underlying premise themselves in the rebuttal. Proposing it to Hasselmo as a test would have been
embarrassing. (It IS genuinely *unrun* -- but unrun and novel are different things.)

---

## 3. THE CORRECTED EXPERIMENT (replaces the Outlook)

Combine the two apparatuses. **{one cue, two cues} x {sham, HD lesion}.**

| | intact | HD-lesioned |
|---|---|---|
| **one cue** (C1) | normal fields | no repetition -- **Calton 2003, already published** |
| **two cues** (C2) | **no repetition; ANCHORING instead** | **C2 repetition -- THE PREDICTION** |

Repetition appears in exactly **one cell of four**, with zero free parameters.

- Intact + two cues predicts **anchoring, not folding**: no within-session repetition, but the map
  should adopt one of two 180-deg-related gauges *at chance* across disoriented sessions. That is
  Keinath et al. 2017's result, in a C2 chamber.
- Lesioned + two cues: the only compass left is the bidirectional one -> the map must fold.
- **This is a collaboration proposal**, not a favour. Hasselmo has the apparatus. Dudchenko / Wood /
  Grieves (the Harland authors) have the lesion *and* the place cells. Hasselmo has no place cells.

---

## 4. THE NEW RESULT (analysis/compass_symmetry.py)

Two theorems, both measured on existing checkpoints.

**Theorem 1.** If the code is G-invariant (= folded) and occupancy is G-symmetric, the
position-marginalised HD tuning curve obeys T(g.h) = T(h). So G = C_n annihilates every harmonic of
the tuning curve that is not a multiple of n. Bidirectional tuning in C2, tetradirectional in C4.

**Theorem 2.** The *within-compartment* tuning obeys T_{j-1}(g.h) = T_j(h): the local tuning curve
**rotates with the compartment**. That is exactly Zhang's **between-compartment (BC)** cell, derived.
Zhang's **WC** cells do NOT follow from invariance (at fixed x, invariance relates u(x, g.h) to
u(g^-1.x, h) -- a different place, so no constraint). We say so.

### The measured dissociation -- and it is the real prize

| | tuning bi_frac | map: phase decode (chance 0.5) |
|---|---|---|
| **s1 / axis** -- invariant compass, **ASYMMETRIC arena** | **0.88 bidirectional** | **0.971 -- NOT FOLDED** |
| s2 / axis -- invariant compass, symmetric arena | 0.95 | **0.552 -- FOLDED** |
| s2 / full -- breaking compass, symmetric arena | 0.37 | 0.978 -- not folded |
| s2 / parity -- breaking compass, *same 1 bit as axis* | -- | 0.955 -- not folded |

**Multidirectional tuning is NECESSARY BUT NOT SUFFICIENT for a fold.** You need the invariant
compass AND the symmetric arena. In s1/axis the compass is bidirectional and the map is perfectly
unfolded.

**IMPORTANT -- this is why the global tuning curve must NOT be sold as a measure of folding.** The
encoding effect dwarfs the arena effect (m1: full->axis in s1 = 0.34->0.05, an encoding effect with
*no symmetry to fold onto*; axis: s1->s2 = 0.051->0.021, the actual fold effect). This is defect D1
of the plan all over again -- do not repeat it. The reachable null (s1) fired, and it must be
reported.

**What it buys:** it explains LaChance's dissociation exactly. A folded compass upstream (POR/RSC)
and an unfolded map downstream (MEC) is not a paradox -- MEC reads its own, still-breaking compass.
And it is a warning to the field: *you cannot infer a folded map from a multidirectional compass.*

---

## 4b. THE REAL THREAT, AND WHY OUR OWN NULL DISARMS IT

**Alexander, Carstensen, Hinman, Raudies, Chapman & Hasselmo (2020), Sci Adv 6(8):eaaz2322** --
*Hasselmo's own lab*, 555 RSC neurons, 7 rats. Verbatim:

> "strongly tuned EBCs commonly exhibited **quad-modal allocentric directional tuning that was
> aligned with the four walls of square environments**"

> "The **bimodal directional tuning** in the latter experiment **may arise from constrained egocentric
> sampling along two axes** as a consequence of the multicompartment environment segmenting two
> opposing walls."

They explicitly reinterpret Jacob's bidirectional retrosplenial cells as an **egocentric-sampling
artefact**. This is the strongest deflationary account in the literature and any referee from the
Hasselmo/Nitz side raises it immediately. STATUS: pending independent re-verification.

**Our s1 null reproduces this mechanism exactly, and that is why we survive it.** In s1 -- an arena
with *no symmetry whatsoever* -- our `axis` networks already show bi_frac = 0.88 and local tuning that
rotates 180 deg with a fictitious half-arena. A unit that fires when facing a nearby wall peaks north
in the north half and south in the south half, in ANY arena. That is Alexander's mechanism, in our
model, quantified.

**The answer is that Alexander explains the TUNING and not the FOLD, and we never use the tuning as
evidence.** Our model separates them, and nothing else can:

    s1/axis:  tuning bidirectional (0.88), map NOT folded (phase 0.971)
    s2/axis:  tuning bidirectional (0.95), map FOLDED     (phase 0.552)

The tuning is the same; the map is not. Egocentric boundary sampling has no account of *Harland's
lesion*, which is a fact about the map, not about tuning. Cite Alexander as the alternative, show the
s1 null, and answer it with the map. Do not omit it.

Two further deflationary alternatives, both to be acknowledged:
- **Long et al. 2024** (Adv Sci 11(40):e2401216): intrinsic *bipolar* HD cells in MEC. **NOT a
  counterexample** -- the authors state the peak separation is "**not confined to a 180 deg angle to
  each other but rather distributed across 180 deg**", clustered at ~90 AND ~180; the box HAD a cue
  card; the cells CO-ROTATE with the unipolar HD population (same reference frame); and there is no
  place-cell or repetition data at all. It is the **intrinsic two-bump ring-attractor alternative**
  we must exclude, not evidence against us.
- **Yan, Burgess & Bicanski 2021** (PLoS Comput Biol 17(9):e1009434): *unimodal* landmark-bearing
  cells reproduce bidirectional RSC firing without any G-invariant compass. Acknowledge.

## 4c. LAURENT ET AL. -- A GIFT, NOT A THREAT

**Laurent, El Mahmoudi, Smith, Sargolini & Jacob**, eLife **Reviewed Preprint** 109951 v1 (10 Feb
2026), doi 10.7554/eLife.109951.1; bioRxiv 2024.08.22.609122. eLife assessment: "valuable" /
**"incomplete"** evidence. Cite as a reviewed preprint, not a journal article.

They recorded RSC *and* hippocampus in connected multi-room environments where the rooms are
**rotated** (opposed 180 deg in the 2-room; orthogonal 90 deg in the 4-room). Result:

- Hippocampus **remapped**, with essentially **no rotationally-repeated fields** (2-room: 91%
  remapped, 9% repeated, "**none showing the opposite or orthogonally repeated activity patterns as
  observed in the RSC**"; 4-room: 99.5% / 100% remapping, "**no orthogonally repeated patterns were
  detected**").
- RSC multidirectional cells DID show geometry-aligned (90/180 deg) tuning, and it **persisted when
  rooms carried distinguishing visual AND tactile cues** -- the control Zhang lacks.

**This is exactly what our law predicts.** The rooms are ROTATED, so the intact global compass is not
rotation-invariant, so it BREAKS G, so there must be NO fold -- and there isn't. And Laurent then
writes our two-condition law out as their own untested prediction:

> "As the MDCs signal varies across connected rooms with different orientations, it may provide
> orientation-specific input to the hippocampus, leading to place cells remapping. Recording MDC
> activity in environments with **parallel** connected rooms would be informative, as the shared
> orientation between rooms may result in a **uniform directional signal, which could support
> repeated coding by place cells**."

An independent group, converging on our mechanism, having not yet run the experiment. Caveats to
carry honestly: **no parallel-room condition** (so their HPC data give only the negative half of the
law); only 4 rats gave hippocampal data; RSC and HPC are **largely different animals**; and reviewers
contested the geometry-vs-cues claim (though not the hippocampal result).

## 4d. A STRONGER CITATION WE ARE NOT USING

**LaChance, Todd & Taube (2022), Sci Adv 8:eabg8404** -- an EARLIER postrhinal two-identical-cue
experiment, reportedly stronger than the 2024 Nat Commun paper on every axis that matters:
within a single arena; 180-deg-symmetric by construction; enormous effect (t86 = -14.66,
P = 1.23e-24); the tuning FOLLOWS THE CUE to 90 deg when the cue is moved to 90 deg (which no
intrinsic-attractor account predicts); **no coexisting unidirectional HD population in POR** to rescue
the map; and POR projects **monosynaptically to subiculum**. The authors write that these cells
"cannot differentiate between north and south, firing in both directions."
Its weakness: the bidirectionality may come from cue *instability* and may decay with exposure.
STATUS: **PENDING VERIFICATION.** If it holds, this -- not the 2024 paper -- is the citation that
carries the invariant-compass premise.

## 5. STANDING HAZARDS -- do not write these sentences

- ~~"LaChance shows the compass's symmetry order tracks the environment's symmetry."~~ One
  environment, one symmetry order. Reviewer #1 asked for more and was refused. **That is Zhang's
  result, not LaChance's.**
- ~~"the compass becomes G-invariant"~~ without "approximately/partially" (cue B < cue A).
- ~~"grid maps in AB are unchanged, therefore MEC did not fold"~~ as the load-bearing MEC arm. Use
  the MEC **HD** null. Concede the hexagonal-lattice caveat.
- ~~"multidirectional tuning implies a G-invariant compass"~~ -- state only the ONE-WAY implication.
  Multi-peaked directional tuning has at least three explanations that do not involve a G-invariant
  compass: egocentric-boundary sampling (Alexander 2020), an intrinsic two-bump ring attractor
  (Long 2024), and unimodal landmark-bearing codes (Yan 2021). **Our own s1 null is a fourth.**
- ~~"multidirectional tuning is evidence that the map has folded"~~ -- **this is the sentence that
  would sink us**, and our s1 result proves it false in our own model (bidirectional tuning, unfolded
  map). The fold must be measured in the MAP. Never in the tuning.
- ~~"the environment's symmetry"~~ for Zhang. Say **visual-geometric symmetry** (the odour broke the
  total symmetry and the cells ignored it).
- ~~"Finkelstein 2015 (bats) shows bidirectional tuning"~~ -- FALSE. Azimuth tuning is unimodal; the
  180 deg figure is a *shift* of a single peak in inverted bats. A referee will catch this.
- ~~"LaChance, Todd & Taube 2019 (Science) reports bidirectional HD cells"~~ -- FALSE. The bimodality
  there is in the population distribution of preferred *egocentric bearings across cells*, not a
  two-peaked tuning curve in any cell. The POR bidirectional result is the **2022 Sci Adv** paper.
- ~~"Spiers 2015 / Grieves 2016 show the HD compass stayed unidirectional"~~ -- FALSE. Neither
  recorded HD cells at all. They are place-cell studies only.

---

## 5b. THE INTERNAL AUDIT (2026-07-14) -- 25 findings, and what was done

A full-manuscript consistency sweep found 25 defects. Fixed so far:

| # | defect | fix |
|---|---|---|
| 4 | "measures that are **zero in the $C_1$ arena by construction**" -- the rate-map symmetry index is **0.61** in C1 under `axis` (vs 0.97 folded). Directly contradicted the paper's own line 884. | Replaced with the UNIFIED PRINCIPLE: *an invariant compass makes the code symmetric whether or not the arena is.* Rate-map symmetry 0.61, odd power 85% of effect, HD tuning 0.88 bidirectional -- ALL in C1 where nothing folds. **Only orbit-phase decoding keeps a clean null** (0.971 C1 vs 0.552 C2). Phase decoding is now the sole primary readout; everything else descriptive, with its C1 baseline printed. |
| 5 | Discussion inferred a folded MAP from multidirectional TUNING -- the exact inference our own s1 null refutes. | Rewritten. Zhang = "directional shadow", not antecedent. Added the s1/s2 dissociation explicitly. |
| 12 | Abstract defined G as "not the environment's symmetry group... the subgroup that self-motion fails to break" -- **falsified by our own Table 2** (`const` is C4-invariant; in the C1 arena it does NOT fold, phase 0.966). | G is now "the subgroup **of** the environment's symmetry group which self-motion fails to break -- both conditions needed". Fixed in abstract + intro. |
| 1/22 | Results said the room "**could not be decoded**" (0.515) and the 4-room "could not be decoded above 0.25" (0.262/0.268). All three are **significantly above chance** (p=0.008, 0.008, 0.008). | Both rewritten to report the residual and its test. The abstract was already right; the Results were wrong. |
| 3 | Calton row: "silencing leaves field count **unchanged** -- **reproduced**". Ours **falls 11.3% (p=0.031)**; Calton's point estimate **rises** (1.29->1.53, n.s.). | Downgraded to "partly: the absence of multiplication is reproduced, the field count is not, and **the sign is wrong**." |
| 7 | "sequenceness falls to the cell-shuffle floor (0.20) under axis". Actual: **0.277 vs floor 0.183** -- 51% ABOVE its floor. 0.20 is neither number. | Corrected to 0.28 vs floor 0.18, "approaching that floor without reaching it." |
| 8 | Replay **coverage has a shuffle null of 1.50x wake -- higher than ANY real network** (best: full, 1.30x). Never reported. | Null now reported beside the numbers. Coverage demoted to a measure of *extent*; sequenceness carries the structured-replay claim. |
| 10 | $p=9.1\times10^{-5}$ and $1.8\times10^{-4}$ both labelled "the exact floor". Both are **normal approximations** (one- and two-sided) of the same complete separation. | Both corrected to the true exact two-sided floor, $1.1\times10^{-5}$ ($U{=}0$, $n{=}10$ vs $10$). |
| 11 | The C4 ceiling -- the parameter-free headline -- rests on **n=4 per encoding**, never stated; elsewhere the paper says n=8 for s4. | n=4 stated in both captions; "sixteen s4 networks" corrected to "the four `const` networks". |
| 17 | "degrade by **indistinguishable amounts**" asserted with no pairwise test, right after conceding the ANOVA rejects. | Test run and reported: C1 vs C4 **U=15, p=0.61** (genuinely indistinguishable); C2 vs C4 **p=0.010** -- the omnibus is reacting to C2 being *least* degraded. Non-monotonic, which is the actual point. |
| 18 | Permutation test was **single-seed, n=40 perms, no p-value reported**, plus "a full multi-seed run **is in progress**" -- in a submitted manuscript. | Replaced with the proper multi-seed one-sample test (which Methods already claimed): 0.552 vs 0.500, **t(9)=34.8, p=6.6e-11, n=10**. |
| 19 | CSCG table row **inverted in both cells** vs the paper's own text (which quotes Raju: "place field repetition **persists**"). | Row corrected: repetition **yes**; dissociation **no** -- it repeats in both without a compass. |
| 20 | "our autoencoder control **confirms it**" -- for the compartment mazes, on which **no autoencoder was ever trained**; the $p=0.59$ comes from a different experiment. | Marked as a prediction, not a measurement. Table cell corrected. |
| 13 | "A bidirectional HD cell **is** a $C_2$-invariant compass" -- asserted exactly, immediately before the paragraph conceding it is only approximate. Also "grid cells did not change **at all**". | Hedged. The grid null now carries its **hexagonal-lattice caveat** (inversion-symmetric, so a rate-map correlation is structurally insensitive to a C2 fold); the **MEC HD null** is named as load-bearing. |
| 14 | "The invariant compass ... is at least two synapses from a place cell" -- true of Zhang's *dysgranular* population, **false for POR**, where LaChance's fold is strongest and which projects monosynaptically to CA1. | Restricted to Zhang's population; POR exception stated outright. |
| 6 | The randomised-compass figure was captioned "**HD-lesion** dose-response" while the Discussion calls that manipulation "spectacular and entirely **spurious**". | Retitled "Compass corruption **during training**"; the train-time/test-time distinction spelled out in the caption. |
| 15 | t-SNE described as "**distance-preserving**". | Deleted. |
| 16 | `fig:fold` panel refs (a)/(b) **swapped** in the text. | Both corrected. |
| 21 | Cross-seed correlation reads 0.96--1.00 in all 112 networks on a y-axis clipped to [0.9, 1.0], **with no null** -- it cannot fail. | Demoted to descriptive in text and caption; explicitly says no null is claimed. |

**STILL OPEN:** #9 (four missing bib keys -- `sugar2011`, `vangroen1992`, `alexander2020`, `agster2013` -- **HARD BUILD BREAK**, verification agent running); #23 (`fig:lesion`c's radial arm "rises" with no number, and the figure is never `\ref`'d); #25 (speed+heading integrates to displacement, which *can* break a translation -- the general law as stated in the abstract does not address this); #2 (the two room-decode residuals come from different measurements -- steady-state vs whole-session -- and the paper never says so); #24 (assorted numeric mismatches).

## 5c. THE ERROR I MADE, AND THE RETRACTION (2026-07-14, second pass)

The user did not believe the result and asked for another pass. They were right. **I asserted a
mechanism I never tested, and it was wrong.**

**WHAT I WROTE (now retracted):** that the bidirectional tuning in `s1/axis` arises from "an invariant
compass combined with ordinary egocentric-boundary tuning -- a unit that fires facing a nearby wall
peaks North in the North half and South in the South half", and that we therefore **reproduce
Alexander et al. (2020)'s deflationary mechanism**.

**THE TEST THAT KILLED IT.** Alexander's mechanism needs NO compass -- it is occupancy plus
egocentric boundary tuning. So `const` (no compass at all) is where it must appear. It does not:

    s1/const   m1 = 0.052   m2 = 0.021   -> the global tuning curve is FLAT. No multidirectionality.
    s2/const   m1 = 0.011   m2 = 0.025   -> flat.
    s4/const   m1 = 0.011   m2 = 0.009   -> flat.

And I checked whether my own occupancy-balancing was *hiding* it. It is not: the RAW,
occupancy-weighted marginal is essentially identical (s1/const raw bi 0.245 vs balanced 0.223).

**WHY.** Our agent samples headings uniformly and does not hug walls. The thigmotaxis that generates
Alexander's effect is simply **absent from our trajectory statistics**. The metric is fine -- the
ground-truth suite (`analysis/test_compass_symmetry.py`, 20/20) shows the occupancy correction DOES
remove a thigmotaxis artefact when one is present. Our policy just never creates one.

**THE REAL MECHANISM** in our networks is the dullest available: the `axis` input **is** a
bidirectional signal, and the units that read it inherit its period. No arena, no boundary, no
occupancy required.

**WHAT SURVIVES** -- and it is the part that matters:

    s1/axis   tuning bidirectional 0.88   map phase 0.971  NOT FOLDED
    s2/axis   tuning bidirectional 0.99   map phase 0.552  FOLDED

Same tuning, opposite maps. Multidirectional tuning remains **necessary but not sufficient**, and
"no multidirectional tuning curve licenses an inference about the map" still stands. Only my
*explanation* of it was invented.

**Alexander et al. is now cited as an alternative we CANNOT rule out**, not one we reproduce. Testing
it would need a thigmotactic policy, which we have not built. Note their own limitation, which is
real: their quad-modal tuning **vanishes in circular arenas** -- it is imposed by the square, not
carried by the cell -- and their reinterpretation of Jacob's bidirectional cells is hedged
("**may** arise") and untested.

### Bib corrections forced by the verification pass
- `sugar2011` **explicitly refuses** the inference we were using it for: *"could it be that this
  projection is present, but perhaps remained undetected or was detected but not reported?"* It may
  **not** be cited for a positive absence. **`shibata1994` is the correct primary citation** for
  granular-specificity (BDA anterograde; subiculum is in the RSG target list and absent from RSA).
- **`li2025` FALSIFIES "no direct projection to the hippocampal formation"** -- a GABAergic RSPd->CA1
  projection exists. The claim is now narrowed to "no substantial direct **excitatory** route", and
  we cite the exception **against ourselves**.
- `joneswitter2007` is a **trap**, not support: their "cingulate cortex" includes areas 29 AND 30, so
  their no-projection claim would also deny the granular->subiculum projection our argument needs.
  **Do not cite.**
- POR->CA1 was wrong; **`naber2001` establishes POR -> SUBICULUM**, not CA1. Corrected.
- The POR paper is **LaChance, Graham, Shapiro, Morris & Taube (2022)**, Sci Adv 8:eabg8404 -- NOT
  "LaChance, Todd & Taube". And its quote *"cannot differentiate between north and south"* is a
  **TRAP**: it is the authors stating an objection they immediately **rebut** (the cells DO
  differentiate, by firing rate). Quoting the first half alone would misrepresent the source.

## 5d. THIRD PASS (2026-07-15): what the re-audit of the EDITED paper found

The user still did not trust it. Two more agents were run against the edited manuscript. Both found
real defects, including in my own new work.

### The slogan was wrong in the OTHER direction too
"Multidirectional tuning is **necessary but not sufficient** for a fold" -- **it is not necessary
either.** `const` in C4 is the most completely folded condition in the paper (it sits ON the 1/4
ceiling) and its tuning curve is **FLAT** (m1 = 0.011, m2 = 0.009). On a four-way compass a
C4-symmetric curve IS a constant: tetradirectional tuning and no tuning are the same object. The
script said so and I did not read my own script.

**This makes the result stronger.** Both arms are now in the data:

    s1/axis    bidirectional (0.93)   map 0.971   NOT folded   -> not SUFFICIENT
    s2/const   FLAT (0.011/0.025)     map 0.552   FOLDED       -> not NECESSARY

**Directional tuning tells you about the compass and nothing whatever about the map, in neither
direction of inference.**

### A real bug in my own metric
`bi_frac` = |F2|^2/(|F1|^2+|F2|^2) is a RATIO. On an untuned unit it is a ratio of two noise terms.
The tuning floor was 0.02 -- far too low -- so `const`/s2 (a FLAT curve) was passing and reporting
**bi_frac = 0.77**, which reads as "77% bidirectional" for a cell that is not directional at all.
**It nearly went into the paper.** Floor raised to 0.10; `n_tuned` now emitted with every row, and it
correctly collapses for const (24/500 in s2, 5/500 in s4) where axis has 357-409.
Consequence: the reported numbers moved. **s1/axis 0.88 -> 0.93; s2/axis 0.99 -> 0.997.** Paper updated.

### Four FALSE n's -- the same error class, again
- lesion "n = 6 networks per arena" -> **C4 is n = 4**. (Stated twice.)
- `fig:fold` caption "n = 10/10/8, 112 networks" -> the panels are **n = 6/6/4, 64 networks**.
- cross-seed "all 112 networks" -> **64**.
- `p = 0.004` labelled exact -> that is the **ONE-SIDED** floor; two-sided at n=5v5 is **0.0079**.

### "110 of 112 networks" is really 103 -- and the failures are systematic
**All 8 s4/axis networks** are fitted better by X/C4 than by the X/C2 their encoding predicts --
which **contradicts our own decoder** (0.482 = the C2 ceiling). Cause: d_{X/G} <= d_X pointwise, so a
LARGER group always fits a compressed code better; the stress metric is biased toward over-folding
exactly where the groups are nested. Now **stated in the paper**, not rounded away.

### Survivors of the earlier fixes (all now closed)
Abstract still said "folds completely"; Introduction still said the fold is "visible directly in
single-cell tuning"; G reverted to its falsified definition in 3 Discussion passages; "undecodable"
survived in 2 captions + a Table cell; "folds only when the classical compass dies" contradicted the
paper's own translation result (intact rats DO fold under translation -- no compass breaks a
translation).

### The Outlook's two hidden assumptions, now named in the text
1. That the hippocampus is dominated by the **classical** compass rather than by the **postrhinal**
   one -- which folds hardest AND reaches subiculum **monosynaptically** (naber2001). The lesion shows
   the map reads *a* breaking compass; it does not say WHICH.
2. That the **disorientation** protocol required by the anchoring prediction does not destroy the very
   carry-forward that produces the entorhinal null we cite as evidence. It does.
Plus **hysteresis** (LaChance A2): the cortical fold does not un-fold when the symmetry is removed, so
the law is about a LEARNED map, and the 2x2 must be counterbalanced, not run in sequence.

---

## 7. STILL OPEN -- DO NOT SUBMIT UNTIL THESE ARE CLOSED

**A. UNTRACEABLE CLAIMS -- numbers in the paper with NO CSV behind them.** Regenerate or cut:
- field-size by horizon (30.6 / 38.1 / 38.5 / 37.4)
- the rotation corridor-conditioning series (0.84, 0.92, 0.98, 0.999, 1.00) and translation 0.63->0.52
- **gridness** (mean -0.35, max +0.08)
- **Fano factor 0.52** + the chi-square against Poisson
- **omnidirectionality** (mean 0.56, median 0.64, 35% of pairs > 0.8)
- the **Isomap orbit distances** (0.17-0.18 vs 0.85-0.89) -- `manifold_robustness.csv` is a DIFFERENT
  metric (fold ratio, n=1 per condition)

**B. `field_stats.csv` and `cell_properties.csv` disagree** on field counts (full/C1: 1.781 vs 1.746;
const/C4: 3.393 vs 3.265). Two files answer the same question differently; only one is cited.

**C. Holes in the validation suite** (`test_compass_symmetry.py`, 20/20 but incomplete):
every synthetic population has exactly ONE unit, so the aggregation over 500 heterogeneous units is
never exercised; `rot_gain` has no calibrated null (the +0.62 for s1/axis has no reference
distribution); `cx` is recomputed with the same formula as the analysis, so a row/column transpose
would be invisible; and `collect_hd` / checkpoint loading are untested.

**D. The paper has now been audited three times and each pass found real errors, including invented
mechanisms and false n's. Assume a fourth pass will find more.** Do not report it as clean.

## 6. THE LAWS THAT PRODUCED THIS

LAW 2 (biology gets MULTIPLE passes) is what caught every one of the above. The extraction pass alone
would have left us citing Zhang for a claim Harland refutes, proposing a "novel" analysis that is an
arithmetic corollary of Hasselmo's own figure, and leaning the MEC arm on a null that a lattice
argument dissolves. The adversarial pass is not optional.
