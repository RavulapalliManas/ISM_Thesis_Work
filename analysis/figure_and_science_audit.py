import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "results" / "figures"
TABLES_DIR = BASE / "results" / "tables"
METRICS_DIR = BASE / "results" / "metrics"
ABLATION_DIR = BASE / "results" / "ablation"


def file_status(path):
    return {"path": str(path.relative_to(BASE)), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def main():
    master = pd.read_csv(TABLES_DIR / "master_metrics.csv")
    with open(METRICS_DIR / "quadrant_distance_matrices.json", "r", encoding="utf-8") as f:
        qdm = json.load(f)
    with open(ABLATION_DIR / "part_e_preflight_status.json", "r", encoding="utf-8") as f:
        preflight = json.load(f)

    figures = {}
    for stem in [
        "fig_training_curves",
        "fig_rate_maps",
        "fig_population_geometry",
        "fig_quadrant_distances",
        "fig_metric_trajectories",
        "fig_coverage_and_heading",
        "fig_hd_ablation",
        "fig_decode_summary",
    ]:
        figures[stem] = {
            "pdf": file_status(FIG_DIR / f"{stem}.pdf"),
            "png": file_status(FIG_DIR / f"{stem}.png"),
        }

    c2 = master.groupby("condition")["C2_Contrast"].agg(["mean", "std", "count"]).to_dict("index")
    audit = {
        "rsa_matrix_diagnostic": {
            "S2_seed00_RSA_vs_Euclidean_spearman_rho": 0.711318,
            "interpretation": "rsa_matrix stores distances. Larger values mean more different.",
        },
        "c2_contrast_definition": {
            "formula": "mean_C2_distance - mean_C4_distance",
            "interpretation": "negative means C2-related positions are more similar than 90-degree-related positions",
            "condition_values": c2,
            "paper_implication": (
                "The S2 negative value supports C2-related compression under this distance-contrast definition. "
                "The prior paper text saying positive supports C2 structure should be corrected."
            ),
        },
        "qdm_status": {
            "expected_pattern_detected": bool(qdm["s2_pattern_confirmed"]),
            "actual_s2_checks": qdm["s2_checks"],
            "interpretation": (
                "QDM does not show the requested quadrant-level S2 relation, even though pointwise C2 distance contrast is negative. "
                "Treat QDM as a separate coarse quadrant diagnostic, not as proof that the pointwise C2 contrast is wrong."
            ),
        },
        "speedhd_status": {
            "encoding": "SpeedHD 5D [speed, hd_0, hd_1, hd_2, hd_3]",
            "ablation_rule": "preserve speed column 0; zero or uniformize heading columns 1:5",
            "preflight_mode_checks": preflight["mode_checks"],
        },
        "cuda_status": {
            "torch_version": preflight["torch_version"],
            "torch_cuda_available": preflight["torch_cuda_available"],
            "nvidia_smi": preflight["nvidia_smi"],
            "status": "blocked until CUDA-enabled torch is installed in .brain",
        },
        "figure_status": figures,
        "figure_limitations": [
            "fig_hd_ablation is missing because Part E training has not run.",
            "fig_metric_trajectories has final-only RA, not checkpoint RA trajectories, because S1/S2 trajectory folders are absent.",
            "fig_quadrant_distances is intentionally marked unexpected for the requested S2 QDM pattern.",
        ],
    }

    out = TABLES_DIR / "figure_and_science_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"{out.relative_to(BASE)} {out.stat().st_size} bytes")
    print("Audit written: C2 sign interpretation, QDM limitation, SpeedHD rule, CUDA blocker, figure completeness.")


if __name__ == "__main__":
    main()
