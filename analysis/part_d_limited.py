import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
METRICS_DIR = BASE / "results" / "metrics"
TABLES_DIR = BASE / "results" / "tables"
FIG_DIR = BASE / "results" / "figures"
SWEEP = BASE / "project5_symmetry" / "results" / "symmetry_sweep"
COLORS = {"S4": "#2166AC", "S2": "#F4A582", "S1": "#D6604D"}
COND_ORDER = ["S4", "S2", "S1"]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def main():
    master = pd.read_csv(TABLES_DIR / "master_metrics.csv")
    limitation = {
        "D1_status": "limited",
        "reason": (
            "Checkpoint hidden-state extraction for all three seed_00 runs requires "
            "the condition trajectory folders or systematic environment observations. "
            "The current sweep contains trajectories only for S4; S1 and S2 trajectory "
            "folders are absent. Therefore no checkpoint H matrices were inferred for "
            "S1/S2. Final H from evaluation.pkl is used for final-step RA/sRSA/DTG, and "
            "training_log.json is used for dense sRSA trajectories."
        ),
        "checkpoint_keys_confirmed": ["step", "model", "optimizer", "meta"],
        "model_load_status": "s4 seed_00 ckpt_final loads into pRNN_th with no missing/unexpected keys",
        "architecture_note": "SpeedHD action input is 5D [speed, hd_0, hd_1, hd_2, hd_3], no separate Wd.",
    }
    with open(METRICS_DIR / "part_d_limited_status.json", "w", encoding="utf-8") as f:
        json.dump(limitation, f, indent=2)

    rows = []
    for cond in COND_ORDER:
        row = master[(master["condition"] == cond) & (master["seed"] == "seed_00")].iloc[0]
        rows.append(
            {
                "Condition": cond,
                "Step": "final",
                "RA": row["RA"],
                "spatial_sRSA": row["spatial_sRSA"],
                "DTG": row["DTG"],
                "source": "evaluation.pkl final H",
            }
        )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "checkpoint_metrics_limited.csv", index=False)
    make_fig(master)

    print("D limited outputs")
    for path in [
        METRICS_DIR / "part_d_limited_status.json",
        TABLES_DIR / "checkpoint_metrics_limited.csv",
        FIG_DIR / "fig_metric_trajectories.pdf",
        FIG_DIR / "fig_metric_trajectories.png",
    ]:
        print(f"{path.relative_to(BASE)} {path.stat().st_size} bytes")
    print("UNEXPECTED RESULT - Figure 5 RA panel is final-only, not a checkpoint trajectory, because S1/S2 trajectory folders are absent.")


def make_fig(master):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    # Panel A: final RA values as endpoint-only evidence.
    for x, cond in enumerate(COND_ORDER):
        vals = master[master["condition"] == cond]["RA"].to_numpy()
        axes[0].bar(x, vals.mean(), yerr=vals.std(ddof=1), color=COLORS[cond], alpha=0.85, edgecolor="black", linewidth=0.5)
        axes[0].scatter(np.full(len(vals), x) + np.linspace(-0.07, 0.07, len(vals)), vals, color="black", s=12, zorder=3)
    axes[0].set_xticks(range(3), COND_ORDER)
    axes[0].set_ylabel("RA (final H only)")

    # Panel B: dense training-log sRSA trajectories.
    for cond in COND_ORDER:
        dfs = [pd.read_csv(p) for p in sorted(TABLES_DIR.glob(f"training_log_{cond.lower()}_seed_*.csv"))]
        for df in dfs:
            axes[1].plot(df["steps"], df["srsa_euclid"], color=COLORS[cond], alpha=0.25, linewidth=0.6)
        merged = pd.concat(dfs, ignore_index=True)
        mean = merged.groupby("steps", as_index=False)["srsa_euclid"].mean()
        axes[1].plot(mean["steps"], mean["srsa_euclid"], color=COLORS[cond], linewidth=2.0, label=cond)
    axes[1].set_ylabel("spatial_sRSA")
    axes[1].set_xlabel("Training step")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_metric_trajectories.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
