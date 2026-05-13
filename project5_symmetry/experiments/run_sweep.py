#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('PRNN_TRAINER', 'fast')
os.environ.setdefault('PRNN_DEFER_SRSA', '0')

from project5_symmetry.environments.arena import (
    PixelObsWrapper,
    SymmetryArena,
    _rotate90cw,
    _rotate180,
    get_obs_at,
)
from project5_symmetry.environments.generate_trajectories import generate_dataset
from project5_symmetry.evaluation.metrics import (
    aggregate_hidden_by_position,
    cross_seed_cca_alignment,
    cross_seed_rsa_alignment,
    dtg_curve,
    manifold_id,
    observation_discriminability,
    place_field_spatial_coherence,
    representational_geometry_consistency,
    sci,
    srsa,
)
from project5_symmetry.experiments.configs import PHASE0
from project5_symmetry.training.dataset import PackedTrajectoryStore
from project5_symmetry.training.train import (
    HIDDEN_INIT_SIGMA,
    PRED_OFFSET,
    SUBSAMPLE_N,
    _collect_hidden_states,
    train,
    train_parallel_seeds,
)
from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell


CONDITIONS = ['s4', 's2', 's1']
N_SEEDS = 9

RESULTS_DIR = Path('project5_symmetry/results')
SWEEP_DIR = RESULTS_DIR / 'symmetry_sweep'
RUNS_DIR = SWEEP_DIR
RAW_PATH = SWEEP_DIR / 'symmetry_sweep_raw.pkl'
SUMMARY_PATH = SWEEP_DIR / 'symmetry_sweep_summary.pkl'
MANIFEST_PATH = SWEEP_DIR / 'symmetry_sweep_manifest.json'
VALIDATION_DIR = RESULTS_DIR / 'symmetry_sweep_validate'

P0_CFG = PHASE0[0]
VALIDATION_N_TRAJ = max(P0_CFG.B * 4, 32)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_pickle(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_json_safe(payload), f, indent=2)


def _build_env(condition: str, seed: int, use_landmarks: bool = True) -> SymmetryArena:
    return SymmetryArena(
        shape='square',
        size=18,
        U=0,
        F=7,
        seed=seed,
        use_landmarks=use_landmarks,
        symmetry_condition=condition,
    )


def _load_trained_model(seed_dir, obs_size, act_size, device):
    """Load a trained pRNN_th model from a seed directory."""
    model = pRNN_th(
        obs_size=obs_size, act_size=act_size,
        k=5, hidden_size=500, cell=LayerNormRNNCell,
        dropp=0.15, neuralTimescale=2,
    ).to(device)
    ckpt_path = Path(seed_dir) / 'ckpt_final.pt'
    if not ckpt_path.exists():
        alt = sorted(Path(seed_dir).glob('ckpt_*.pt'))
        if not alt:
            raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
        ckpt_path = alt[-1]
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()
    return model


def _compute_metrics_symmetry_sweep(model, dataset, device, condition, seed):
    """Compute all evaluation metrics for one seed."""
    metrics = {}
    hidden_states, positions = _collect_hidden_states(
        model, dataset, n=SUBSAMPLE_N, device=device,
    )
    if len(hidden_states) == 0:
        return metrics
    metrics['srsa'] = srsa(hidden_states, positions)
    metrics['sci'] = sci(hidden_states, positions, condition=condition)
    metrics['dtg'] = dtg_curve(hidden_states, positions)
    metrics['manifold_id'] = manifold_id(hidden_states)
    return metrics


def validate_phase0(
    n_traj: int = 64,
    n_seeds: int = 3,
    parallel_seeds: int = 1,
    dataset_workers: int = 4,
) -> bool:
    """Phase 0 gate check: must reach sRSA_euclid > 0.4."""
    print("=" * 60)
    print("Phase 0 validation gate")
    print("=" * 60)
    PASS_THRESHOLD = 0.4
    scores = []
    for seed in range(n_seeds):
        env = _build_env('s1', seed=seed, use_landmarks=True)
        wrapped = PixelObsWrapper(env, tile_size=1)
        data_dir = VALIDATION_DIR / f'seed_{seed:02d}' / 'trajectories'
        generate_dataset(wrapped, n_traj=n_traj, T=P0_CFG.T, out_dir=str(data_dir),
                         desc=f'P0 validation seed {seed}')
        dataset = PackedTrajectoryStore(str(data_dir), device='cpu')
        obs_size = wrapped.unwrapped.agent_view_size ** 2 * 3
        model = pRNN_th(
            obs_size=obs_size, act_size=5, k=P0_CFG.k,
            hidden_size=500, cell=LayerNormRNNCell,
            dropp=0.15, neuralTimescale=2,
        ).to('cpu')
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, dataset, opt, n_steps=min(10000, P0_CFG.n_steps // 4),
              log_every=250, device='cpu')
        h, p = _collect_hidden_states(model, dataset, n=SUBSAMPLE_N, device='cpu')
        scores.append(srsa(h, p))
    mean_score = float(np.mean(scores))
    passed = mean_score > PASS_THRESHOLD
    print(f"Phase 0 gate: mean sRSA = {mean_score:.4f} "
          f"({'PASSED' if passed else 'FAILED'}, threshold={PASS_THRESHOLD})")
    return passed


def run_condition_sweep(
    condition: str,
    n_seeds: int = N_SEEDS,
    parallel_seeds: int = 1,
    dataset_workers: int = 4,
):
    """Run a single condition sweep across seeds."""
    results = []
    for seed in range(n_seeds):
        env = _build_env(condition, seed=seed, use_landmarks=(condition != 's4'))
        wrapped = PixelObsWrapper(env, tile_size=1)
        data_dir = SWEEP_DIR / condition / f'seed_{seed:02d}' / 'trajectories'
        generate_dataset(wrapped, P0_CFG.n_traj, P0_CFG.T, str(data_dir),
                         desc=f'{condition} seed {seed}')
        dataset = PackedTrajectoryStore(str(data_dir), device='cuda' if torch.cuda.is_available() else 'cpu')
        obs_size = wrapped.unwrapped.agent_view_size ** 2 * 3
        model = pRNN_th(
            obs_size=obs_size, act_size=5, k=P0_CFG.k,
            hidden_size=500, cell=LayerNormRNNCell,
            dropp=0.15, neuralTimescale=2,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, dataset, opt, n_steps=P0_CFG.n_steps,
              log_every=500, device=device)
        metrics = _compute_metrics_symmetry_sweep(model, dataset, device, condition, seed)
        results.append(metrics)
    return results


def run_full_sweep(
    parallel_seeds: int = 1,
    dataset_workers: int = 4,
    n_seeds: int = N_SEEDS,
):
    """Run the complete symmetry condition sweep."""
    all_results = {}
    for condition in CONDITIONS:
        print(f"\n--- Sweeping condition: {condition} ---")
        condition_results = run_condition_sweep(
            condition=condition,
            n_seeds=n_seeds,
            parallel_seeds=parallel_seeds,
            dataset_workers=dataset_workers,
        )
        all_results[condition] = condition_results
    _write_pickle(RAW_PATH, all_results)
    print(f"\nRaw results saved to {RAW_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Symmetry sweep runner')
    parser.add_argument('--validate', action='store_true',
                        help='Run Phase 0 validation gate only')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip the validation gate')
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS,
                        help='Number of seeds per condition')
    parser.add_argument('--parallel-seeds', type=int, default=1,
                        help='Number of seeds to train in parallel')
    parser.add_argument('--dataset-workers', type=int, default=4,
                        help='Workers for trajectory generation')
    args = parser.parse_args()

    if args.validate:
        passed = validate_phase0(n_seeds=args.n_seeds)
        raise SystemExit(0 if passed else 1)

    if not args.skip_validation:
        print("Running Phase 0 validation gate...")
        passed = validate_phase0(n_seeds=min(3, args.n_seeds))
        if not passed:
            raise SystemExit('Phase 0 validation failed. Aborting sweep.')

    run_full_sweep(
        parallel_seeds=args.parallel_seeds,
        dataset_workers=args.dataset_workers,
        n_seeds=args.n_seeds,
    )


if __name__ == '__main__':
    main()
