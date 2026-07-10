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
