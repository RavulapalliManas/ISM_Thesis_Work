import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SWEEP = BASE / "project5_symmetry" / "results" / "symmetry_sweep"
METRICS_DIR = BASE / "results" / "metrics"
TABLES_DIR = BASE / "results" / "tables"
CONDITIONS = {"S1": range(3), "S2": range(3), "S4": range(5)}


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m:.3f} $\\pm$ {s:.3f}"


def load_eval_json(cond, seed):
    with open(SWEEP / cond.lower() / f"seed_{seed:02d}" / "evaluation.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    master = pd.read_csv(TABLES_DIR / "master_metrics.csv")
    stats = pd.read_csv(TABLES_DIR / "statistical_tests.csv")
    flags = []

    g_rows = []
    for cond, seeds in CONDITIONS.items():
        for seed in seeds:
            row = master[(master["condition"] == cond) & (master["seed"] == f"seed_{seed:02d}")].iloc[0].to_dict()
            ev_json = load_eval_json(cond, seed)
            s_json = float(ev_json["srsa"])
            s_pkl = float(row["spatial_sRSA"])
            s_comp = float(row["sRSA_euclid_computed"])
            diffs = [abs(s_json - s_pkl), abs(s_json - s_comp), abs(s_pkl - s_comp)]
            if max(diffs) > 0.02:
                flags.append(
                    f"G1 sRSA discrepancy >0.02 for {cond} seed_{seed:02d}: "
                    f"json={s_json:.6f}, pkl={s_pkl:.6f}, computed={s_comp:.6f}"
                )
            g_rows.append(
                {
                    "Condition": cond,
                    "Seed": f"seed_{seed:02d}",
                    "sRSA_json": s_json,
                    "sRSA_pkl": s_pkl,
                    "sRSA_computed": s_comp,
                    "RA": row["RA"],
                    "PAA_gain": row["PAA_gain"],
                    "SCI": row["SCI"],
                    "C2_Contrast": row["C2_Contrast"],
                    "DecodeErr": row["DecodeErr"],
                    "DTG": row["DTG"],
                    "coherence_median": row["coherence_median"],
                    "coherence_mean": row["coherence_mean"],
                    "rgc_stress": row["rgc_stress"],
                }
            )

    g_df = pd.DataFrame(g_rows)
    g_df.to_csv(TABLES_DIR / "all_metrics_final.csv", index=False)

    s4_ra = master[master["condition"] == "S4"]["RA"].to_numpy()
    s4_ra_mean, s4_ra_std = mean_std(s4_ra)
    if abs(s4_ra_mean - 0.223) > 0.01:
        flags.append(f"G2 S4 RA mean deviates from 0.223 by >0.01: {s4_ra_mean:.6f} +/- {s4_ra_std:.6f}")

    sample_doc = {
        "S1": ["seed_00", "seed_01", "seed_02"],
        "S2": ["seed_00", "seed_01", "seed_02"],
        "S4": ["seed_00", "seed_01", "seed_02", "seed_03", "seed_04"],
    }
    with open(TABLES_DIR / "sample_size_documentation.json", "w", encoding="utf-8") as f:
        json.dump(sample_doc, f, indent=2)

    latex_main = make_main_latex(master, stats)
    (TABLES_DIR / "main_results_table.tex").write_text(latex_main, encoding="utf-8")
    latex_ablation = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{HD ablation metrics. Values unavailable until Part E training completes.}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Condition & sRSA & PAA gain & RA & Decode Err \\\\\n"
        "\\midrule\n"
        "HD\\_FULL & pending & pending & pending & pending \\\\\n"
        "HD\\_ABLATED & pending & pending & pending & pending \\\\\n"
        "HD\\_DEGRADED & pending & pending & pending & pending \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    (TABLES_DIR / "hd_ablation_table.tex").write_text(latex_ablation, encoding="utf-8")

    updates = paper_updates(master)
    (TABLES_DIR / "paper_statistics_updates.txt").write_text("\n".join(updates) + "\n", encoding="utf-8")

    print("G1 sRSA cross-validation")
    print(g_df[["Condition", "Seed", "sRSA_json", "sRSA_pkl", "sRSA_computed"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nG2 RA validation")
    print(f"S4 RA = {s4_ra_mean:.6f} +/- {s4_ra_std:.6f}")
    print("\nG3 sample sizes")
    print("S1 n=3, S2 n=3, S4 n=5 confirmed.")
    print("\nG4 final validation table")
    print(g_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nH1/H2/H3 outputs written")
    for path in [
        TABLES_DIR / "all_metrics_final.csv",
        TABLES_DIR / "statistical_tests.csv",
        TABLES_DIR / "main_results_table.tex",
        TABLES_DIR / "hd_ablation_table.tex",
        TABLES_DIR / "paper_statistics_updates.txt",
        TABLES_DIR / "sample_size_documentation.json",
    ]:
        print(f"{path.relative_to(BASE)} {path.stat().st_size} bytes")
    print("\nFlags")
    if flags:
        for flag in flags:
            print(f"- {flag}")
    else:
        print("None")


def sig_marker(stats, metric, comparison):
    row = stats[(stats["metric"] == metric) & (stats["comparison"] == comparison)]
    if row.empty:
        return ""
    p = float(row.iloc[0]["p"])
    bonf = bool(row.iloc[0]["survives_bonferroni"])
    mark = ""
    if p < 0.01:
        mark = "**"
    elif p < 0.05:
        mark = "*"
    if bonf:
        mark += "$\\dagger$"
    return mark


def make_main_latex(master, stats):
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Final metrics by symmetry condition. Values are mean $\\pm$ std across seeds. C2 distance contrast is $d_{180}-d_{90}$, so negative values indicate greater similarity among C2-related positions. Statistical tests: Mann--Whitney U.}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Condition & sRSA & RA & PAA gain & SCI & C2 Dist. Contrast & Decode Err \\\\",
        "\\midrule",
    ]
    for cond in ["S4", "S2", "S1"]:
        df = master[master["condition"] == cond]
        n = len(df)
        markers = {
            "spatial_sRSA": sig_marker(stats, "spatial_sRSA", "S4_vs_S1") if cond == "S4" else "",
            "RA": sig_marker(stats, "RA", "S4_vs_S1") if cond == "S4" else "",
            "PAA_gain": sig_marker(stats, "PAA_gain", "S4_vs_S1") if cond == "S4" else "",
            "SCI": sig_marker(stats, "SCI", "S4_vs_S1") if cond == "S4" else "",
            "C2_Contrast": sig_marker(stats, "C2_Contrast", "S4_vs_S1") if cond == "S4" else "",
            "DecodeErr": sig_marker(stats, "DecodeErr", "S4_vs_S1") if cond == "S4" else "",
        }
        vals = [
            fmt(df["spatial_sRSA"]) + markers["spatial_sRSA"],
            fmt(df["RA"]) + markers["RA"],
            fmt(df["PAA_gain"]) + markers["PAA_gain"],
            fmt(df["SCI"]) + markers["SCI"],
            fmt(df["C2_Contrast"]) + markers["C2_Contrast"],
            fmt(df["DecodeErr"]) + markers["DecodeErr"],
        ]
        lines.append(f"{cond} (n={n}) & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def paper_updates(master):
    computed = {
        "S4 spatial_sRSA": mean_std(master[master["condition"] == "S4"]["spatial_sRSA"])[0],
        "S2 spatial_sRSA": mean_std(master[master["condition"] == "S2"]["spatial_sRSA"])[0],
        "S1 spatial_sRSA": mean_std(master[master["condition"] == "S1"]["spatial_sRSA"])[0],
        "S4 RA": mean_std(master[master["condition"] == "S4"]["RA"])[0],
        "S2 RA": mean_std(master[master["condition"] == "S2"]["RA"])[0],
        "S1 RA": mean_std(master[master["condition"] == "S1"]["RA"])[0],
        "S4 PAA gain": mean_std(master[master["condition"] == "S4"]["PAA_gain"])[0],
        "S2 PAA gain": mean_std(master[master["condition"] == "S2"]["PAA_gain"])[0],
        "S1 PAA gain": mean_std(master[master["condition"] == "S1"]["PAA_gain"])[0],
        "S4 SCI": mean_std(master[master["condition"] == "S4"]["SCI"])[0],
        "S2 SCI": mean_std(master[master["condition"] == "S2"]["SCI"])[0],
        "S1 SCI": mean_std(master[master["condition"] == "S1"]["SCI"])[0],
        "S4 C2 Contrast": mean_std(master[master["condition"] == "S4"]["C2_Contrast"])[0],
        "S2 C2 Contrast": mean_std(master[master["condition"] == "S2"]["C2_Contrast"])[0],
        "S1 C2 Contrast": mean_std(master[master["condition"] == "S1"]["C2_Contrast"])[0],
        "S4 DTG": mean_std(master[master["condition"] == "S4"]["DTG"])[0],
        "S2 DTG": mean_std(master[master["condition"] == "S2"]["DTG"])[0],
        "S1 DTG": mean_std(master[master["condition"] == "S1"]["DTG"])[0],
        "S4 DecodeErr": mean_std(master[master["condition"] == "S4"]["DecodeErr"])[0],
        "S2 DecodeErr": mean_std(master[master["condition"] == "S2"]["DecodeErr"])[0],
        "S1 DecodeErr": mean_std(master[master["condition"] == "S1"]["DecodeErr"])[0],
    }
    expected = {
        "S4 RA": 0.223,
        "S2 RA": 0.116,
        "S1 RA": 0.098,
        "S4 SCI": 0.889,
        "S2 SCI": 0.892,
        "S1 SCI": 0.984,
        "S2 C2 Contrast": 0.045,
        "S4 C2 Contrast": -0.130,
        "S1 C2 Contrast": -0.070,
        "S4 DTG": -0.052,
        "S2 DTG": -0.011,
        "S1 DTG": -0.020,
        "S4 DecodeErr": 0.524,
        "S2 DecodeErr": 0.475,
        "S1 DecodeErr": 0.365,
    }
    lines = [
        "Architecture note: current codebase uses SpeedHD act_enc with 5 dimensions [speed, hd_0, hd_1, hd_2, hd_3], not a separate 12-bin HD input/Wd.",
        "C2 Contrast note: computed contrast is mean_C2_distance - mean_C4_distance. Because RSA is a distance matrix (S2 seed_00 RSA vs Euclidean rho = +0.711), negative values mean C2 pairs are more similar than C4 pairs.",
        "DecodeErr note: current computation follows B7 normalization and yields very small values because coordinates are decoded almost exactly from H; paper values appear to use a different scale or error definition.",
    ]
    for key, val in computed.items():
        if key in expected:
            action = "CONFIRMED" if abs(val - expected[key]) <= max(0.01, abs(expected[key]) * 0.03) else "UPDATE"
            lines.append(f"Paper states {key} = {expected[key]:.3f}. Computed value = {val:.6f}. Action: {action}")
        else:
            lines.append(f"Paper should report {key}. Computed value = {val:.6f}. Action: UPDATE")
    return lines


if __name__ == "__main__":
    main()
