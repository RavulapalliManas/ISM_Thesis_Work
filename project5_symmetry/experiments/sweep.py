"""
Outer sweep loop for project5_symmetry experiments.

Usage (from repo root, after activating env):
    python -m project5_symmetry.experiments.sweep --phase 0
    python -m project5_symmetry.experiments.sweep --phase 1
    python -m project5_symmetry.experiments.sweep --all
"""
import argparse
import json
import os

import numpy as np
import torch

from project5_symmetry.environments.arena import Arena
from project5_symmetry.environments.generate_trajectories import generate_dataset
from project5_symmetry.experiments.configs import (
    PHASE0, PHASE1, PHASE2A, PHASE2B, PHASE4A, PHASE4B, ALL_CONDITIONS,
    ExperimentConfig,
)
from project5_symmetry.training.train import train
from project5_symmetry.evaluation.metrics import srsa, sci, dtg_curve, manifold_id
from project5_symmetry.training.dataset import TrajectoryDataset

GATE_THRESHOLD = 0.4
BASE_OUT = 'project5_symmetry/results'


def _run_condition(cfg: ExperimentConfig, seed: int, base_dir: str) -> dict:
    cond_dir = os.path.join(base_dir, cfg.condition_id)
    data_dir = os.path.join(cond_dir, 'trajectories')
    run_dir = os.path.join(cond_dir, f'seed_{seed:02d}')
    os.makedirs(run_dir, exist_ok=True)

    # Build arena (deterministic per condition; seed just controls landmark colours)
    arena = Arena(cfg.arena_shape, cfg.arena_size, cfg.U, cfg.F, seed=seed)

    # Generate trajectories once per condition (shared across seeds)
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) < cfg.n_traj:
        print(f"  Generating {cfg.n_traj} trajectories for {cfg.condition_id} …")
        generate_dataset(arena, cfg.n_traj, cfg.T, data_dir, n_workers=12)
        arena.save_metadata(os.path.join(cond_dir, 'arena_meta.json'))

    obs_size = cfg.F * cfg.F * 3
    log = train(
        data_dir=data_dir,
        obs_size=obs_size,
        out_dir=run_dir,
        k=cfg.k,
        T=cfg.T,
        batch_size=cfg.B,
        n_steps=cfg.n_steps,
        seed=seed,
    )

    # Final evaluation on held-out hidden states
    dataset = TrajectoryDataset(data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load final checkpoint
    from utils.Architectures import pRNN_th
    from utils.thetaRNN import LayerNormRNNCell
    ckpt = torch.load(os.path.join(run_dir, 'ckpt_final.pt'), map_location=device)
    model = pRNN_th(obs_size=obs_size, act_size=5, k=cfg.k,
                    hidden_size=500, cell=LayerNormRNNCell, neuralTimescale=2)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    from project5_symmetry.training.train import _collect_hidden_states
    h, pos = _collect_hidden_states(model, dataset, n=4000, device=device)

    sym_pairs = arena.precompute_symmetry_pairs()
    result = {
        'condition_id': cfg.condition_id,
        'seed': seed,
        'final_srsa_euclid': srsa(h, pos, space_metric='euclidean'),
        'final_srsa_city':   srsa(h, pos, space_metric='cityblock'),
        'final_sci':         sci(h, pos, sym_pairs),
        'final_manifold_id': manifold_id(h),
        'dtg_curve':         dtg_curve(log['srsa_euclid'], log['srsa_city']).tolist(),
        'srsa_euclid_curve': log['srsa_euclid'],
        'srsa_city_curve':   log['srsa_city'],
        'steps': log['steps'],
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def run_phase0(base_dir: str) -> bool:
    cfg = PHASE0[0]
    print(f"\n=== Phase 0 gate: {cfg.condition_id} ===")
    result = _run_condition(cfg, seed=0, base_dir=base_dir)
    passed = result['final_srsa_euclid'] > GATE_THRESHOLD
    print(f"Gate result: sRSA={result['final_srsa_euclid']:.3f} "
          f"({'PASS' if passed else 'FAIL — halting sweep'})")
    return passed


def run_sweep(conditions: list, base_dir: str, phases_label: str = ''):
    os.makedirs(base_dir, exist_ok=True)
    all_results = []

    for cfg in conditions:
        print(f"\n=== Condition {cfg.condition_id} ({cfg.n_seeds} seeds) ===")
        for seed in range(cfg.n_seeds):
            print(f"  seed {seed} …")
            result = _run_condition(cfg, seed, base_dir)
            all_results.append(result)

    out_path = os.path.join(base_dir, f'all_results{phases_label}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default=None,
                        choices=['0', '1', '2a', '2b', '4a', '4b', 'all'])
    parser.add_argument('--out', type=str, default=BASE_OUT)
    args = parser.parse_args()

    if args.phase == '0':
        run_phase0(args.out)
        return

    if not run_phase0(args.out):
        return  # gate failed

    phase_map = {
        '1': PHASE1,
        '2a': PHASE2A,
        '2b': PHASE2B,
        '4a': PHASE4A,
        '4b': PHASE4B,
        'all': ALL_CONDITIONS,
    }

    conditions = phase_map.get(args.phase, ALL_CONDITIONS)
    run_sweep(conditions, args.out, phases_label=f'_phase{args.phase}')


if __name__ == '__main__':
    main()
