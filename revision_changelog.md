SECTION: Abstract
ORIGINAL: Levenstein et al. showed that a predictive recurrent neural network can learn an allocentric map from egocentric observation prediction, but
NEW: Representational similarity analysis (sRSA) is invariant to global map rotation -- permuting the rows of the hidden state matrix preserves
REASON: Replaced the abstract with the requested metric-first framing that foregrounds RA, PAA, and C2 Contrast.

SECTION: Introduction
ORIGINAL: We test three conditions that differ only in landmark symmetry: S4 is C4-symmetric, S2 is C2-symmetric, and S1 is
NEW: A single map-quality score cannot separate these failure modes because sRSA is formally blind to global map rotation. The metric
REASON: Inserted the requested paragraph explaining why sRSA cannot detect global orientational degeneracy and why the new metrics are needed.

SECTION: Results 3.1
ORIGINAL: Figure 1 summarizes the result pattern. S1 has the highest Euclidean sRSA and the lowest decoding error, while S4
NEW: Symmetry systematically degrades spatial precision and metric geometry without producing global orientational degeneracy: sRSA falls from 0.745 in
REASON: Rewrote the subsection opening to present the main empirical pattern directly.

SECTION: Results 3.3
ORIGINAL: The decoding result measures usable spatial precision rather than general map structure. A linear decoder read position from the hidden
NEW: Decoding error increased monotonically with symmetry group order: S4 = 0.524 ± [STAT:decoding_error:S4_sem], S2 = 0.475 ±
REASON: Replaced the opening sentence with the requested statistics-ready finding-first version.

SECTION: Results 3.4
ORIGINAL: Geometry was evaluated by comparing neural distances against Euclidean and city-block spatial distances. The Distance-Topology Gap is
NEW: Neural distance geometry became more city-block-biased as landmark symmetry increased: DTG was −0.052 in S4, −0.011 in S2,
REASON: Replaced the opening sentence with the requested DTG summary and placeholders for inferential statistics.

SECTION: Results 3.5 Opening
ORIGINAL: The original degeneracy concern was not supported. Cross-seed RSA alignment stayed high in every condition: S4 = 0.977±0.002,
NEW: Cross-seed orientational degeneracy did not occur in any condition: PAA gain was 0.006 in S4 and 0.005 in
REASON: Reframed the subsection to foreground the null global-degeneracy result and added shuffle placeholder support.

SECTION: Results 3.5 RA Insert
ORIGINAL: Local aliasing did increase. Rotational autocorrelation compares a unit's tuning map with its rotated copies; it was S4
NEW: Local aliasing did increase. Rotational autocorrelation compares a unit's tuning map with its rotated copies; it was S4
REASON: Added the requested shuffle z-score and p-value placeholders immediately after the RA values.

SECTION: Results 3.6
ORIGINAL: C2 Contrast was S4 = -0.130, S2 = +0.045, and S1 = -0.070. The contrast compares similarity for
NEW: The S2 population code specifically encoded the C2 symmetry group of the observation function: C2 Contrast was +0.045 in
REASON: Rewrote the opening sentence and inserted the requested mechanistic explanation of negative S4 C2 contrast.

SECTION: Results 3.7
ORIGINAL: TwoNN dimensionality was S4 = 5.95±0.29, S2 = 5.78±0.33, and S1 = 5.66±0.37. This is opposite the
NEW: TwoNN dimensionality was S4 = 5.95±0.29, S2 = 5.78±0.33, and S1 = 5.66±0.37. This is opposite the
REASON: Appended the requested significance sentence with placeholder tokens for the S4 versus S1 test.

SECTION: Discussion Insertion 1
ORIGINAL: Landmark symmetry produces precision loss, local aliasing, and a more path-biased geometry, but not global orientational degeneracy. This reframes
NEW: Significance for the Levenstein Framework. The pRNN framework was originally validated in an L-shaped arena that breaks rotational
REASON: Added the requested interpretation paragraph connecting the findings back to the Levenstein framework.

SECTION: Discussion Insertion 2
ORIGINAL: The C2 result addresses a different mechanism. Predictive learning can encode the actual symmetry group of the observation
NEW: The fraction of spatially tuned units in all conditions (~4–8%) falls substantially below Levenstein et al.'s reported ~20–30%.
REASON: Inserted the requested frac_tuned interpretation paragraph before the limitations paragraph.

SECTION: Discussion Limitation
ORIGINAL: The PAA degeneracy threshold of 0.05 is heuristic; a continuous degeneracy measure, such as SCI, would remove the
NEW: A continuous degeneracy measure would strengthen the main conclusion. The Symmetry Collapse Index (SCI) -- the ratio of mean
REASON: Replaced the existing SCI limitation sentence with the requested scalar-degeneracy framing.

SECTION: Conclusion
ORIGINAL: Overall, landmark diversity matters because it anchors predictive transitions that distinguish otherwise equivalent locations. Symmetry does not destroy
NEW: Landmark symmetry in a C4-symmetric arena degraded the pRNN's spatial representation along three measurable dimensions: decoding error increased
REASON: Replaced the conclusion with the requested two-paragraph synthesis and placeholder tokens.

SECTION: Figure 1 Caption
ORIGINAL: Figure. Metric overview and summary statistics by condition. S4 has lower map precision, more negative DTG, and higher
NEW: Figure. Symmetry degrades spatial precision without eliminating map structure. S4 shows lower sRSA (0.647 vs 0.745), higher
REASON: Rewrote the caption in finding-first form.

SECTION: Figure 2 Caption
ORIGINAL: Figure. Decoding error and ΔTG dynamics. Decoding error rises with symmetry, while negative ΔTG indicates stronger city-block than
NEW: Figure. Decoding error rises and DTG grows more negative with symmetry group order. Left: linear decoding error at
REASON: Rewrote the caption in finding-first form and preserved the S2 caveat.

SECTION: Figure 4 Caption
ORIGINAL: Figure. Cross-seed RSA alignment matrices. High off-diagonal correlations in all conditions show that global map geometry is reproducible
NEW: Figure. Global map orientation is reproducible across seeds in all conditions. Off-diagonal cross-seed RSA values exceed 0.97 in
REASON: Rewrote the alignment caption in finding-first form and explicitly referenced PAA gain.

SECTION: Figure 5 Caption
ORIGINAL: Figure. Spatial information and TwoNN dimensionality. S4 ends with higher dimensionality, consistent with path-history structure rather than simple
NEW: Figure. Manifold dimensionality increases with symmetry, opposite to the collapse prediction. S4 TwoNN dimensionality (5.95 ± 0.29)
REASON: Rewrote the caption in finding-first form with the requested interpretation.

SECTION: Hedging Cleanup
ORIGINAL: If four arena quadrants generate identical landmark sequences up to rotation, then the prediction loss may fail to distinguish globally
NEW: If four arena quadrants generate identical landmark sequences up to rotation, then the prediction loss fails to distinguish globally
REASON: Removed broad hedging language where the argument is supported directly by the report's logic and evidence.

SECTION: Head-Direction Citations
ORIGINAL: The 12-bin head-direction input provides the most direct explanation: it breaks rotational equivalence in the transition sequence even when
NEW: The head-direction input provides the most direct explanation: it breaks rotational equivalence in the transition sequence even when the
REASON: Added the requested citation tag wherever head-direction input is invoked mechanistically in the revised report.
