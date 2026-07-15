# NUMBER → SOURCE MAP (eLife `main_best.tex`)

Every reported number recomputed from the saved CSV at audit time (2026-07-15), per CLAUDE.md §5.
Recompute script pattern in `AUDIT_PHASE_A.md`. "✓" = reproduced to the digit (or within seed
rounding). Aggregation is mean over seeds per (condition, encoding) unless noted.

## Decoding tables — ALL ✓

| claim | paper | recomputed | source | verdict |
|---|---|---|---|---|
| Table 2 orbit-phase, all 12 cells | see tab:phase | identical | `phase_full_n10.csv` | ✓ exact |
| axis vs parity separation | s2 complete (p=1.1e-5), s1 chance (p=0.97), s4 complete | U=100 (s2), U=49 (s1), U=64 (s4) | `phase_full_n10.csv` | ✓ |
| Table 3 C4 four-way (0.918/0.860/0.482/0.275) | tab:c4 | 0.918/0.860/0.482/0.275 | `phase_s4_c4.csv` | ✓ exact |
| Table 5 horizon k=0,1,3 | tab:horizon | identical | `phase_horizon_k{0,1,3}.csv` | ✓ exact |
| Table 5 k=5 (0.978/0.953/0.553/0.551) | tab:horizon (n=6) | 0.978/0.955/0.552/0.552 (n=10) | `phase_full_n10.csv` | ✓ (n=6 vs n=10 rounding) |
| Learned compass (0.971/0.526/0.523; R²) | Results | 0.971/0.526/0.523; raw −0.077, dom 0.884/−0.473 | `phase_learned_c2.csv` | ✓ |
| Noisy compass (axis 0.556, parity 0.922) | Results | 0.556 / 0.922 | `phase_noisy.csv` | ✓ |

## Geometry — ALL ✓

| claim | paper | recomputed | source |
|---|---|---|---|
| C2 axis stress: quotient/arena | 0.190 / 0.468 | 0.190 / 0.468 | `isometry_quotient.csv` |
| C2 full stress: arena/quotient | 0.097 / 0.485 | 0.097 / 0.485 | `isometry_quotient.csv` |
| sham preference (C2 axis) | 0.343 | 0.343 | `isometry_quotient.csv` |
| C1 axis: sham−quotient / cos | 0.145 / 0.811 | 0.145 / 0.811 | `isometry_quotient.csv` |
| orbit cosine C2 (axis/parity) | 0.990 / 0.741 | 0.990 / 0.741 | `isometry_quotient.csv` |
| **metric assignment by encoding** | 103/112; 8/8 axis-in-C4 | **103/112; 8/8** | `isometry_quotient.csv` |
| fold ratio (axis-C2/axis-C1/parity-C2/full-C2) | 0.457/1.872/2.083/2.675 | identical | `manifold_fold_ratio.csv` |

## Lesion dissociation — ALL ✓ (`lesion_dose.csv`, silence, dose 0→1)

| claim | paper | recomputed |
|---|---|---|
| orbit-variance drop C1/C2/C4 | −37.6% / −30.9% / −38.1% | −37.6 / −30.9 / −38.1 |
| phase C1 hold / C2 / C4 | 0.986→0.862 / →0.566 / →0.567 | 0.862 / 0.566 / 0.567 |
| field count C1/C2/C4 | −11% / +16% / +20% | −11.3 / +15.9 / +20.4 |
| Calton field-count fall | −11.3%, n=6 | −11.3% | ✓ |

## Repetition, remapping, BVC, heterogeneity — ALL ✓

| claim | paper | recomputed | source |
|---|---|---|---|
| 2-room translation rep/room/R² | 0.997 / 0.515 / 0.95 | 0.997 / 0.515(seen) / 0.951 | `compartments.csv` |
| 2-room rotation rep/room/R² | −0.080 / 0.998 / 0.79 | −0.080 / 0.998 / 0.793 | `compartments.csv` |
| 4-room decode/rep/R² | 0.262 / 0.268 / 0.994 / 0.96 | 0.262 / 0.268 / 0.993 / 0.960 | `compartments4.csv` |
| remapping PV (axis/const/full/parity-C2, axis-C1) | 0.98/0.98/0.28/0.58/0.71 | 0.981/0.978/0.287/0.578/0.714 | `remapping.csv` |
| BVC frac (full-C1/full-C4/const-C1/const-C4/axis-C2/parity-C2) | 41.6/61.2/14.0/0.8/29.2/22.1 | identical (`frac_bvc_like`) | `bvc_tuning.csv` |
| BVC mean_r axis/parity | 0.449 / 0.482 | 0.449 / 0.482 | `bvc_tuning.csv` |
| mixed selectivity full/parity/axis/const (C2) | 0.235/0.292/0.331/0.333 | identical | `cell_properties.csv` |
| frac_mixed 91%→97% | 0.914 → 0.973 | ✓ | `cell_properties.csv` |
| unit-fold C2/axis: frac>0.8, median, p5 | 93.1% / 0.94 / 0.77 | 93.1% / 0.944 / 0.768 | `unit_heterogeneity.csv` |
| C2/full frac<0.2, median | 54.3% / 0.09 | 54.3% / 0.088 | `unit_heterogeneity.csv` |
| r(border/SI/mixed, fold) | 0.16 / 0.25 / −0.15 | 0.156 / 0.252 / −0.151 | `unit_heterogeneity.csv` |
| fold most/least boundary quartile | 0.947 / 0.923 | 0.947 / 0.923 | `unit_heterogeneity.csv` |
| replay coverage full/parity/axis/const (k1) | 1.30/0.80/0.45/0.31 | identical | `replay_k1.csv` |
| autoencoder coverage (k0) | 0.37× | 0.37 | `replay_k0.csv` |
| weak-cue Spearman ρ (gen/seen) | 0.92 / 0.99 | 0.91 / 0.98 | `weakbreak_snr.csv` |
| weak-cue d'=0 / d'=3 room, ceiling | 0.510 / 0.59 / 0.93 | 0.510 / 0.593 / 0.933 | `weakbreak_snr.csv` |
| Hockeimer pair_r / ICC / pairs | 0.23 / 0.31 / 127 | 0.231 / 0.306 / 127 | `hockeimer_summary.csv` |
| Fundamental-domain Isomap (axis/full) | 0.17–0.18 / 0.85–0.89 | 0.167–0.179 / 0.851–0.889 | `fundamental_domain_isomap.csv` |
| orphan Fano / omni (s1/full) | 0.65 (edited) / 0.56–0.64 | 0.649 / 0.586–0.632 | `orphan_metrics.csv` |

## MINOR discrepancies (recompute ≠ paper)

| claim | paper | recomputed | source | note |
|---|---|---|---|---|
| sequenceness under axis | 0.28 | **0.205** (s2) | `sequenceness.csv` | full 0.527✓, floor 0.171✓; qualitative "approaches floor" holds; number off |
| shuffled-rollout coverage | 1.50× | **1.57×** | `replay_k1.csv` | claim ("more than any net") holds |
| position-only variance endpoint | 0.505 → 0.360 | full 0.505, **const 0.312** (0.360 = axis) | `cell_properties.csv` | 0.360 is the axis value, not const |

## Previously untraceable — now REGENERATED (§5.2 resolved)

| claim | paper (untraceable) | regenerated | source | status |
|---|---|---|---|---|
| field size vs horizon k (Results, "switch not dial") | 30.6/38.1/38.5/37.4 | **33.8±1.6 / 40.7±1.0 / 42.1±1.0 / 39.7±0.7** | `field_area_horizon.csv` (new script `field_area_horizon.py`) | pattern holds (step at k=1, then plateau); absolute values ~3–4 higher than untraceable originals — **update text** |
| corridor-conditioning, translation (Methods confound control) | 0.63 → 0.52 | **0.56 → 0.51** (0-2 → 10+ steps) | `corridor_dwell.csv` (new script `corridor_dwell.py`) | decays to floor — pattern holds; **update text** |
| corridor-conditioning, rotation (Methods confound control) | 0.84 → 1.00 | **0.71 → 0.92 → 0.98 → 0.99** | `corridor_dwell.csv` | rises to ceiling — pattern holds; **update text** |

Both regenerated from fresh trajectories through the identical env (compartment data absent from the
backup drive); statistically equivalent, not bit-identical. The scientific claims — field size is a
step not a gradient; the two arrangements have opposite dwell signatures so corridor memory does not
explain the rotation result — are reproduced. New CSVs + scripts (`field_area_horizon.py`,
`corridor_dwell.py`) are the traceable sources.
