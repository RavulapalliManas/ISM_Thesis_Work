#!/usr/bin/env python3
"""Execute benchmark sweeps for geometry x aliasing environments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
import pandas as pd
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project4_topology_before_geometry.scripts.aliasing_sweep import build_sweeps


def collect_results(log_dir: Path, envs: List[str], seeds: List[int]) -> pd.DataFrame:
    rows = []
    for env in envs:
        for seed in seeds:
            csv_path = log_dir / f"{env}_{seed}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        last_row = df.iloc[-1].to_dict()
                        last_row["env_name"] = env
                        last_row["seed"] = seed
                        rows.append(last_row)
                except Exception as e:
                    print(f"Failed to read {csv_path}: {e}")
            else:
                print(f"Missing {csv_path}")
    
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default="geometry_alias_cross", 
                        help="Sweep name from aliasing_sweep.py to run (e.g. geometry_alias_cross)")
    parser.add_argument("--config", type=str, default="local_config.yaml",
                        help="Config filename located in configs directory")
    parser.add_argument("--seeds", type=str, default="42,123", help="Comma-separated list of seeds")
    args = parser.parse_args()

    sweeps = build_sweeps()
    if args.sweep not in sweeps and args.sweep != "all":
        print(f"Unknown sweep '{args.sweep}'. Available: {list(sweeps.keys())}")
        sys.exit(1)

    selected_envs = []
    if args.sweep == "all":
        for s in sweeps.values():
            selected_envs.extend([req["env_type"] for req in s])
    else:
        selected_envs = [req["env_type"] for req in sweeps[args.sweep]]
        
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    config_path = REPO_ROOT / "project4_topology_before_geometry/configs" / args.config
    log_dir = REPO_ROOT / f"project4_topology_before_geometry/logs"
    
    print(f"Running benchmark sweep '{args.sweep}' with configurator '{args.config}'")
    print(f"Total Environments: {len(selected_envs)}")
    print(f"Seeds: {seeds}")

    run_script = REPO_ROOT / "project4_topology_before_geometry/scripts/run_remote.py"

    # Execute all
    for env in selected_envs:
        for seed in seeds:
            print(f"\n--- Benchmarking Env [{env}] | Seed [{seed}] ---")
            cmd = [
                sys.executable, str(run_script),
                "--env", env,
                "--seed", str(seed),
                "--config", str(config_path)
            ]
            try:
                # We pipe stdout dynamically to console but it's synchronous
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"WARNING: Run failed for {env} with seed {seed}: {e}")

    # Compile Summary
    print("\nCompiling benchmarking aggregation...")
    summary_df = collect_results(log_dir, selected_envs, seeds)
    if not summary_df.empty:
        # Keep metrics matching handoff
        metrics_of_interest = ['env_name', 'seed', 'geodesic_spearman', 'SW_distance', 'tuned_fraction', 'loss']
        existing_cols = [c for c in metrics_of_interest if c in summary_df.columns]
        other_cols = [c for c in summary_df.columns if c not in metrics_of_interest]
        
        summary_df = summary_df[existing_cols + other_cols]
        output_csv = log_dir / f"benchmark_summary_{args.sweep}.csv"
        summary_df.to_csv(output_csv, index=False)
        print(f"\nCompleted! Wrote benchmark results to {output_csv}")
    else:
        print("\nNo logs were found to compile a summary DataFrame.")


if __name__ == "__main__":
    main()
