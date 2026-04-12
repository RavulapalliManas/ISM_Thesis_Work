#!/usr/bin/env python3
"""Batch diagnostics and validation checks for aliasing-controlled MiniGrid environments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure verify_aliasing checks can be used
from project4_topology_before_geometry.scripts.verify_aliasing import run_checks, visualize_connectivity
from project4_topology_before_geometry.environments.aliasing_controlled_envs import list_prebuilt_environments
from project4_topology_before_geometry.environments.env_factory import get_env

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="project4_topology_before_geometry/figures/aliasing_diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = {"seed": args.seed}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    envs_to_check = list_prebuilt_environments()
    print(f"Batch verifying {len(envs_to_check)} environments...")
    
    results = []
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for env_name in tqdm(envs_to_check, desc="Validating Envs"):
            # We catch warnings specific to the environment
            env_warnings = []
            
            try:
                # Capture specific warnings during this check
                with warnings.catch_warnings(record=True) as env_w:
                    warnings.simplefilter("always")
                    result = run_checks(env_name, cfg)
                    
                    # Store any warnings generated in this env run
                    for warn in env_w:
                        env_warnings.append(str(warn.message))
                
                # We can also generate visual representation for connectivity as diagnostics
                env = get_env(env_name, cfg)
                visualize_connectivity(env, output_dir)

                # Add warnings to result mapping
                result["warnings"] = " | ".join(env_warnings) if env_warnings else "None"
                results.append(result)

            except Exception as e:
                # Some envs might fail; record the error instead
                # if there is an error in generation
                results.append({
                    "env_name": env_name,
                    "canonical_name": env_name,
                    "aliasing_score": None,
                    "duplicate_patch_fraction": None,
                    "symmetry_axes": None,
                    "n_bottlenecks": None,
                    "disconnected": None,
                    "warnings": f"ERROR: {str(e)}"
                })

    # Output CSV summary
    summary_path = output_dir / "batch_summary.csv"
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)
    print(f"\nBatch validation complete. Report saved to: {summary_path}")

    # Small output of those that had errors/warnings
    problematic = df[df['warnings'] != 'None']
    if not problematic.empty:
        print(f"\nFound {len(problematic)} environments with warnings/errors. See log for details.")

if __name__ == "__main__":
    main()
