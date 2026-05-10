import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "results" / "figures"
METRICS_DIR = BASE / "results" / "metrics"
TABLES_DIR = BASE / "results" / "tables"
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


def load_all_data():
    with open(METRICS_DIR / "all_evaluations.pkl", "rb") as f:
        return pickle.load(f)


def save_both(fig, stem):
    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def report(paths, expected_ok=True, detail=""):
    prefix = "" if expected_ok else "UNEXPECTED RESULT - "
    for path in paths:
        print(f"{prefix}{path.relative_to(BASE)} {path.stat().st_size} bytes. {detail}")


def mean_by_step(dfs, value_col):
    merged = pd.concat(dfs, ignore_index=True)
    return merged.groupby("steps", as_index=False)[value_col].mean()


def fig_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharex=True)
    for cond in COND_ORDER:
        dfs = []
        for csv in sorted(TABLES_DIR.glob(f"training_log_{cond.lower()}_seed_*.csv")):
            df = pd.read_csv(csv)
            dfs.append(df)
            axes[0].plot(df["steps"], df["loss"], color=COLORS[cond], alpha=0.3, linewidth=0.7)
            axes[1].plot(df["steps"], df["srsa_euclid"], color=COLORS[cond], alpha=0.25, linewidth=0.6)
            axes[1].plot(df["steps"], df["srsa_city"], color=COLORS[cond], alpha=0.25, linewidth=0.6, linestyle="--")
            axes[2].plot(df["steps"], df["manifold_id"], color=COLORS[cond], alpha=0.3, linewidth=0.7)
        if dfs:
            axes[0].plot(*mean_by_step(dfs, "loss").T.values, color=COLORS[cond], linewidth=2.0)
            srsa_e = mean_by_step(dfs, "srsa_euclid")
            srsa_c = mean_by_step(dfs, "srsa_city")
            axes[1].plot(srsa_e["steps"], srsa_e["srsa_euclid"], color=COLORS[cond], linewidth=2.0)
            axes[1].plot(srsa_c["steps"], srsa_c["srsa_city"], color=COLORS[cond], linewidth=2.0, linestyle="--")
            axes[2].plot(*mean_by_step(dfs, "manifold_id").T.values, color=COLORS[cond], linewidth=2.0)
    axes[0].set_ylabel("Prediction Loss")
    axes[1].set_ylabel("sRSA (Spearman)")
    axes[2].set_ylabel("Intrinsic Dimensionality (ID)")
    axes[1].axhline(0.40, color="0.5", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    legend = [
        Line2D([0], [0], color=COLORS[c], lw=2, label=c) for c in COND_ORDER
    ] + [
        Line2D([0], [0], color="0.2", lw=1.4, label="Euclidean"),
        Line2D([0], [0], color="0.2", lw=1.4, linestyle="--", label="City-block"),
    ]
    axes[1].legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    paths = save_both(fig, "fig_training_curves")
    report(paths, True, "Expected pattern check: training logs plotted for all complete seeds.")


def fig_rate_maps(all_data):
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 5.4))
    col_labels = ["High RA", "Median RA", "Low RA", "Place-cell"]
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=10)
    for r, cond in enumerate(COND_ORDER):
        key = f"{cond.lower()}_seed_00"
        H = all_data[key]["H"]
        ra = np.load(METRICS_DIR / f"{cond.lower()}_seed00_RA_per_unit.npy")
        evs = np.load(METRICS_DIR / f"{cond.lower()}_seed00_EVS_per_unit.npy")
        valid_ra = np.where(np.isfinite(ra))[0]
        median_unit = valid_ra[np.argsort(ra[valid_ra])[len(valid_ra) // 2]]
        place_candidates = np.where(evs > 0.93)[0]
        if len(place_candidates) == 0:
            place_unit = int(np.nanargmax(evs))
        else:
            place_unit = int(place_candidates[np.argmax(evs[place_candidates])])
        units = [int(np.nanargmax(ra)), int(median_unit), int(np.nanargmin(ra)), place_unit]
        axes[r, 0].set_ylabel(cond, rotation=0, labelpad=18, va="center", fontsize=11)
        for c, unit in enumerate(units):
            T = H[:, unit].reshape(18, 18)
            axes[r, c].imshow(T, cmap="RdYlBu_r", interpolation="nearest", vmin=np.nanmin(T), vmax=np.nanmax(T))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_xlabel(f"unit {unit}, RA={ra[unit]:.3f}", fontsize=7)
    paths = save_both(fig, "fig_rate_maps")
    report(paths, True, "Expected pattern check: high/median/low/place-like units selected from saved RA and EVS arrays.")


def quadrant_labels(pos):
    labels = []
    for r, c in pos:
        if r <= 9 and c <= 9:
            labels.append(0)
        elif r <= 9 and c > 9:
            labels.append(1)
        elif r > 9 and c <= 9:
            labels.append(2)
        else:
            labels.append(3)
    return np.asarray(labels)


def fig_population_geometry(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    q_colors = np.array(["#D6604D", "#2166AC", "#1B9E77", "#F4A582"])
    expected_ok = True
    detail_bits = []
    for ax, cond in zip(axes, COND_ORDER):
        key = f"{cond.lower()}_seed_00"
        H = all_data[key]["H"]
        pos = all_data[key]["pos_coords"]
        labels = quadrant_labels(pos)
        X = PCA(n_components=50, random_state=42).fit_transform(H)
        try:
            emb = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
        except TypeError:
            emb = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
        ax.scatter(emb[:, 0], emb[:, 1], c=q_colors[labels], s=8, alpha=0.7, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel({"S4": "S4 (C4-sym)", "S2": "S2 (C2-sym)", "S1": "S1 (Asymm)"}[cond])
        if cond == "S4":
            centers = np.vstack([emb[labels == q].mean(axis=0) for q in range(4)])
            spread = np.mean([np.linalg.norm(emb[labels == q] - centers[q], axis=1).mean() for q in range(4)])
            separation = np.mean([np.linalg.norm(centers[i] - centers[j]) for i in range(4) for j in range(i + 1, 4)])
            ratio = separation / spread if spread else np.inf
            detail_bits.append(f"S4 quadrant separation/spread={ratio:.3f}")
            if ratio > 2.5:
                expected_ok = False
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=q_colors[i], markersize=5, label=f"Q{i+1}") for i in range(4)]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    paths = save_both(fig, "fig_population_geometry")
    report(paths, expected_ok, "Expected S4 interleaving check: " + "; ".join(detail_bits))


def fig_quadrant_distances(master):
    with open(METRICS_DIR / "quadrant_distance_matrices.json", "r", encoding="utf-8") as f:
        qdm = json.load(f)
    matrices = {c.upper(): np.asarray(qdm["matrices"][c]) for c in ("s4", "s2", "s1")}
    vmin = min(float(m.min()) for m in matrices.values())
    vmax = max(float(m.max()) for m in matrices.values())
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for ax, cond in zip(axes, COND_ORDER):
        im = ax.imshow(matrices[cond], cmap="viridis", vmin=vmin, vmax=vmax)
        c2 = master[master["condition"] == cond]["C2_Contrast"].mean()
        ax.set_xlabel(f"{cond} - C2 dist. contrast: {c2:.3f}")
        ax.set_xticks(range(4), [f"Q{i}" for i in range(1, 5)])
        ax.set_yticks(range(4), [f"Q{i}" for i in range(1, 5)])
        if cond == "S2" and qdm["s2_pattern_confirmed"]:
            ax.text(2, 0, "*", ha="center", va="center", color="white", fontsize=14)
            ax.text(3, 1, "*", ha="center", va="center", color="white", fontsize=14)
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    paths = save_both(fig, "fig_quadrant_distances")
    report(paths, bool(qdm["s2_pattern_confirmed"]), f"S2 QDM check: {qdm['s2_checks']}")


def fig_decode_summary(master):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.7))
    metrics = [
        ("frac_tuned", "Fraction spatially tuned\n(EVS > 0.93)", None),
        ("SCI", "SCI", 1.0),
        ("DecodeErr", "Linear decoding error", None),
    ]
    master = master.copy()
    frac = []
    for _, row in master.iterrows():
        evs = np.load(METRICS_DIR / f"{row['condition'].lower()}_{row['seed'].replace('_', '')}_EVS_per_unit.npy")
        frac.append(float(np.mean(evs > 0.93)))
    master["frac_tuned"] = frac
    for ax, (metric, ylabel, baseline) in zip(axes, metrics):
        xs = np.arange(len(COND_ORDER))
        means = [master[master["condition"] == c][metric].mean() for c in COND_ORDER]
        stds = [master[master["condition"] == c][metric].std(ddof=1) for c in COND_ORDER]
        ax.bar(xs, means, yerr=stds, color=[COLORS[c] for c in COND_ORDER], alpha=0.85, edgecolor="black", linewidth=0.5)
        for x, cond in zip(xs, COND_ORDER):
            vals = master[master["condition"] == cond][metric].to_numpy()
            jitter = np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", s=11, zorder=3)
        if baseline is not None:
            ax.axhline(baseline, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_xticks(xs, COND_ORDER)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    s4_sci = master[master["condition"] == "S4"]["SCI"].mean()
    s1_sci = master[master["condition"] == "S1"]["SCI"].mean()
    s4_dec = master[master["condition"] == "S4"]["DecodeErr"].mean()
    s1_dec = master[master["condition"] == "S1"]["DecodeErr"].mean()
    expected_ok = s4_sci < s1_sci and s4_dec > s1_dec
    paths = save_both(fig, "fig_decode_summary")
    report(paths, expected_ok, f"SCI S4={s4_sci:.3f}, S1={s1_sci:.3f}; DecodeErr S4={s4_dec:.6f}, S1={s1_dec:.6f}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all_data()
    master = pd.read_csv(TABLES_DIR / "master_metrics.csv")
    fig_training_curves()
    fig_rate_maps(all_data)
    fig_population_geometry(all_data)
    fig_quadrant_distances(master)
    fig_decode_summary(master)


if __name__ == "__main__":
    main()
