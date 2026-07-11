# Thesis Frame: A Predictive RNN as a Model System for Dissecting the Causal Structure of Hippocampal Development

**Author:** Manas Venkata Sai Ravulapalli · **Drafted:** 2026-06-04 · **Status:** canonical frame
**Companion:** execution detail in `DEVELOPMENTAL_REFRAMING_PLAN.md` (this doc supersedes its framing).

---

## The thesis in one sentence

> **Predictive learning provides a model system in which the *causal* structure of representational
> development — which spatial representations are necessary scaffolds for which others, what is innate
> versus experience-set, and what is invariant across rearing — can be dissected by stage-resolved
> interventions that are impossible in any living animal.**

## The move that makes it a thesis (not a footnote)

The weak framing is *"a predictive RNN reproduces hippocampal development."* That is a **that-claim** —
it says a representation *appears* when you optimize the right objective. Every normative model
(Banino 2018, Sorscher 2019, Cueva-Wei 2018, and now **Abrate et al. 2026**, who already show the
developmental *emergence order*) is a that-claim. Stacking more phenomena never escapes *"you tuned a
model until it matched — correlation, not mechanism."*

**Your unfair advantage is not more phenomena. It is the one thing in-silico uniquely licenses:
full, stage-resolved, repeatable intervention on a single developing individual.** Abrate built the
developmental sandbox; **nobody has perturbed it.** The thesis is the first to run the *developmental
experiments that are forbidden in animals* — and to convert "X emerges before Y" (descriptive) into
"**Y cannot form unless X formed first**" (causal). Necessity, not sequence. That is a category
difference, not an increment.

Epistemologically: prior work answers *sufficiency* at Marr's computational level (prediction
*suffices* to produce representation R — cheap and non-unique). Interventions operate at the
algorithmic/implementational interface: *given* that R is produced, **what internal structure is R
causally built out of?** No amount of sufficiency results answers a necessity question.

## The signature object, and the one arrow that carries the thesis

**Signature object:** a *typed, weighted causal developmental dependency graph.* Nodes = representational
properties (HD-ring, place tuning, manifold topology, metric geometry, symmetry-disambiguation,
preplay repertoire, consolidated replay), each tagged with an **innateness weight** (present at init /
in pre-task spontaneous activity ↔ built by experience). Edges = **perturbation-verified necessity
relations** ("X is necessary for Y, proven by lesion-then-resume"), not mere timing.

**The headline is a single arrow — the triple-locked HD→place edge.** Show that head-direction coding
is a *causal prerequisite* for spatial tuning, locked three ways:

1. **Necessity** — freezing the HD-ring subspace *during continued training* prevents place coding from
   developing, while freezing a matched-dimension *random* subspace does not.
2. **Objective-specificity** — the edge is present under the predictive objective and **absent/reordered
   under a matched autoencoder** with identical architecture and identical HD input.
3. **Invariance** — the HD-before-place ordering is conserved across experience *amount* and *curriculum
   order* (only absolute timing shifts).

One arrow carrying necessity + objective-specificity + invariance is no longer a normative-model
coincidence and no longer Abrate's descriptive order. It is a **causal law of this model system**, with
a named in-vivo test (developmental silencing of HD input during the place-cell window).
**That single arrow is worth more than a broad battery of weaker results.**

---

## Why it is robust: the three attacks and the experiments that defeat each

Robustness here means *surviving the hostile committee*, and that requires **new experiments**, not
re-runs. The three attacks that can kill the frame, and the move that defeats each:

### Attack 1 — "Your lesion is readout corruption, not a developmental intervention."
Zeroing HD at *evaluation time* on a trained net only shows the mature computation *uses* HD — of
course it breaks; you trained it to. **Defeat:** **lesion-then-resume** — project the HD-ring subspace
out of the *gradient* and let the rest of the network *keep training*; ask whether place coding ever
matures. Matched random-subspace freeze is the control. *Consequence:* the headline moves OFF the free
eval-time ablation and ONTO a cheap training run. Accept that cost — the free version does not survive.

### Attack 2 — "The architecture/input builds the graph, not the learning."
The noiseless HD oracle is wired in by you; "HD-first" may be a trivial consequence of the input
pipeline, identical across seeds because the *wiring* is fixed. **Defeat:** the **autoencoder objective
control** — same architecture, same HD input, swap predictive→reconstruction. If the dependency graph
*collapses or reorders*, the structure is a property of **predictive learning**, not the wiring. This
is **the single most load-bearing experiment in the thesis.** No free experiment defeats this attack.
*If you can afford exactly one new training run, spend it here.*

### Attack 3 — "There is no developmental clock — you have a learning curve called ontogeny by analogy."
Your time axis is gradient steps = the *experience integral*; biology's striking fact (Farooq & Dragoi)
is that maturation is partly experience-*independent*. So by construction you are the opposite of the
phenomenon. **Defeat (two parts):**
- (a) **Order/amount invariance** (Kendall τ≈1 across 0.25×/1×/4× experience and curriculum scrambles,
  timing shifts but order never): this earns the word *development* — ordering is set by
  architecture+objective+input-statistics, not by the experience integral.
- (b) **NEW experiment — spontaneous-only maturation:** between blocks of real experience, train the
  network on its *own* preplay/replay sequences (no environment input). If *any* property advances
  toward maturity during these experience-free blocks, that is a genuine **experience-independent
  maturation mechanism** — and "development" stops being a metaphor. This is the experiment that most
  upgrades the thesis; build it.

> **Triple-lock + spontaneous-only kills all three attacks.** Everything else is supporting evidence.

---

## The experiment program (only what makes it robust)

Deduplicated from a 19-experiment battery to the load-bearing set. Full protocols, falsifiers, and
biological predictions for all 19 are archived in the plan doc's appendix / workflow output.

### TIER 1 — decisive (the thesis is not robust without these)
| # | Experiment | Locks | Compute | What it proves |
|---|---|---|---|---|
| 1 | **Objective necessity** (predictive vs autoencoder vs SR vs PI-supervised) | objective-specificity | **1 training run (priority)** | Escapes "just normative / scooped by Abrate" |
| 2 | **Lesion-then-resume** (HD-ring gradient-freeze + random-subspace control) | necessity | **1 training run** | The real signature edge; defeats Attack 1 |
| 3 | **Order/amount invariance** (τ-conserved, timing shifts) | invariance | mostly free + cheap | Earns the word "development"; defeats Attack 3a |
| 4 | **Init-surgery** (innate-vs-learned decomposition at step 0) | node innateness | free CPU | Separates wiring from development; defeats Attack 2a |
| 3b | **Spontaneous-only maturation** (train on own preplay, no input) | experience-independent clock | cheap resume | Defeats Attack 3b; makes "development" mechanistic |

### TIER 2 — strong, earns novelty (headline *figures*, not load-bearing)
- **Preplay-forecasts-future-map** (early autonomous repertoire predicts the not-yet-learned map),
  with a **dimensionality-matched low-D null** + rigorous sequenceness statistic. Free, power-immune,
  your most quotable result. *Build the null first; if it isn't beaten, this dies quietly.*
- **Within-net precedence matrix + threshold-robust topology-before-geometry** (HD-ring Betti-1, bootstrap
  barcodes). Necessary plumbing for the graph and the invariance test; by itself it's just precedence.

### TIER 3 — defer (breadth, not robustness; invites the "you tuned to match" attack)
Critical-period sliding-window, symmetry-as-rearing training arm, dark-rearing, order-scrambling
attractor, drift-robust motif-identity. Run the *free* ones as bonus; defer all high-GPU ones.

### Compute reality ($5 RunPod now, more pending)
Almost everything is free CPU on the existing checkpoints. The scarce GPU buys **exactly the two runs
that matter: the HD-freeze resume (#2) and the autoencoder control (#1)** — together they triple-lock
one edge and kill two attacks. The spontaneous-only run (#3b) is a cheap resume. *Do not* spend compute
on critical-period or rearing sweeps until the triple-lock exists.

---

## The epistemic contract (state verbatim in the thesis — the firewall against overreach)

**Claims about the biological developing hippocampus:** the model is **hypothesis-generating about
causal structure** — it produces falsifiable directional necessity claims ("HD-stability gates place
formation") and specifies the exact in-vivo experiment (developmental HD silencing during the place
window) that confirms or refutes each. It claims a **structural correspondence at the level of
dependency logic only** — explicitly *not* at the level of mechanism, anatomy, biophysics, cell types,
or real-time timeline. *If* the developing hippocampus is well-approximated as a predictive learner (an
assumption shared with the entire normative-model program), *then* the model's dependency structure is
a prediction about the biological one.

**Explicit disclaimers (write them down to disarm the committee):** the network is not a hippocampus;
units are place/HD/grid-like only in a functional-signature sense; no step↔postnatal-day mapping is
claimed (only ordering/dependency — the power-immune relations); predictive learning is not claimed to
be *the* unique objective (the claim is *conditional*); nothing generalizes beyond 2D discrete
navigation (3D, continuous, multiscale, vectorial-goal, replay-at-scale = the boundary of the claim and
future work). "Innate" ≡ "present at random init or pre-task spontaneous activity" = not-experience-
derived *within this model*, with the caveat that the architecture is a design choice.

**The ambitious half (so it doesn't read as retreat):** *because* it is a model organism and not a
replica, a positive *causal* result is **more** informative than a positive descriptive one — it
converts a vague developmental correlation ("HD matures before place") into a sharp, mechanistically-
committed, experimentally-actionable necessity hypothesis. The contract trades breadth (no 3D, no real
timeline) for **causal depth and falsifiability** — the correct trade for a thesis that wants to
*matter*, not merely *match*.

> ⚠️ **Do NOT use the "Aplysia / model organism" analogy as a *defense* until the autoencoder control
> (#1) has earned it.** Aplysia earned its status because the mechanism *transferred*; you share only an
> abstract objective you haven't yet shown is *the* objective. Using the analogy to assert the
> conclusion the thesis is meant to establish is the most dangerous self-flattery in the frame.

---

## What makes it undeniable, and what would sink it

- **Undeniable:** the triple-locked HD→place arrow (necessity + objective-specificity + invariance).
  Achievable with existing checkpoints + ~two cheap training runs.
- **Most likely to sink it:** collapse of the time axis — a referee arguing "gradient steps ≠
  development" and that invariance merely shows SGD order-robustness (unremarkable), reducing the thesis
  to "some ablations on a spatial RNN." **Mitigation is non-negotiable: the spontaneous-only maturation
  experiment** is the wall that holds this up. Without it, a sharp developmental neurophysiologist sinks
  you on the definition of your own central word.

---

## Headline sentence for the abstract
> *By performing stage-resolved developmental lesions impossible in any animal, we show that
> head-direction coding is a causal prerequisite for spatial tuning — not merely its temporal
> predecessor — that this dependency is specific to predictive learning and invariant to experience
> schedule, establishing predictive learning as a model system in which the causal architecture of
> representational development can be dissected by intervention.*
