# Result CSVs behind the paper

Every number in `../biorxiv/main.tex` traces to a file here. Figures are regenerated
from these by `../../analysis/make_paper_figures.py`. Kept in-repo so the results
survive loss of the compute pod.

- `phase_full_n10.csv` -- orbit-phase decoding, n=10 (s1/s2), n=8 (s4). Table 2, abstract.
- `phase_groupB.csv`, `phase_horizon_k{0,1,3}.csv` -- horizon sweep (k=0..5).
- `compartments.csv` -- in-silico Grieves, n=8 per arrangement.
- `field_stats.csv` -- place-field counts + rate-map symmetry index.
- `map_quality_group{A,B}.csv` -- sRSA and cross-seed correlation.
- `replay_k{0,1,3,5}.csv` -- offline replay coverage vs wake/shuffle.
- `sequenceness.csv` -- replay sequenceness with time-shift and cell-shuffle nulls.
- `tda_topology.csv` -- topology-before-geometry (null result; see Limitations).
- `speed_*.csv`, `spectral_*.csv` -- initialization study.

- `phase_s4_c4.csv` -- C4 four-way phase decoding in the s4 arena (Table 3; n=4, 16 networks).
- `phase_nonlinear.csv` -- nonlinear phase-decoding control (linear/kNN/MLP, C2 arena).
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
- `isotypic_symmetry.csv` -- C4 isotypic spectrum (P0..P3, RA, odd) for the clean 17-network
  full-HD sweep; RA vs odd s1-vs-s2 separation ("Two natural readouts fail").
- `isotypic_hd.csv` -- same spectrum for the 112 HD-invariance networks; the confounded
  odd(parity) - odd(axis) drop per arena. Both from `../../analysis/run_spectrum.py`; the
  reported p-values are two-sided Mann--Whitney U on the `RA`/`odd` columns.

Numbers not backed by a CSV here are deterministic arena enumerations (the Eq. bound's
distinguishable/predicted columns, the 6.6% and 1228/1296 counts, the ODI values) or are
figure-derived (manifold fold ratios, gridness); all are reproducible from the analysis scripts.
