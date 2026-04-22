"""
Outer sweep loop for project5_symmetry experiments.

Usage (from repo root):
    python run.py --phase 0
    python run.py --phase 1
    python run.py --phase all

Or directly:
    PYTHONPATH=. python -m project5_symmetry.experiments.sweep --phase 0
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from project5_symmetry.environments.arena import (
    make_symmetry_env, save_arena_metadata,
)
from project5_symmetry.environments.generate_trajectories import generate_dataset
from project5_symmetry.experiments.configs import (
    PHASE0, PHASE1, PHASE2A, PHASE2B, PHASE4A, PHASE4B, ALL_CONDITIONS,
    ExperimentConfig,
)
from project5_symmetry.training.train import (
    train, train_parallel_seeds, _collect_hidden_states,
    HIDDEN_INIT_SIGMA, PRED_OFFSET, SUBSAMPLE_N, PARALLEL_SEED_GROUP,
)
from project5_symmetry.evaluation.metrics import srsa, sci, dtg_curve, manifold_id
from project5_symmetry.training.dataset import TrajectoryDataset

GATE_THRESHOLD = 0.4
BASE_OUT       = 'project5_symmetry/results'


# ── Single condition × seed run ───────────────────────────────────────────────

def _condition_paths(cfg: ExperimentConfig, seed: int, base_dir: str) -> tuple[str, str, str]:
    cond_dir = os.path.join(base_dir, cfg.condition_id)
    data_dir = os.path.join(cond_dir, 'trajectories')
    run_dir  = os.path.join(cond_dir, f'seed_{seed:02d}')
    return cond_dir, data_dir, run_dir


def _ensure_condition_data(cfg: ExperimentConfig, seed: int, base_dir: str):
    cond_dir, data_dir, _ = _condition_paths(cfg, seed, base_dir)
    os.makedirs(cond_dir, exist_ok=True)

    env = make_symmetry_env(cfg.arena_shape, cfg.arena_size, cfg.U, cfg.F, seed=seed)

    existing = len([f for f in os.listdir(data_dir) if f.endswith('.npz')]) \
               if os.path.isdir(data_dir) else 0

    if existing < cfg.n_traj:
        generate_dataset(
            env, cfg.n_traj, cfg.T, data_dir,
            desc=f'{cfg.condition_id} trajectories',
        )
        meta_path = os.path.join(cond_dir, 'arena_meta.json')
        if not os.path.exists(meta_path):
            save_arena_metadata(env, meta_path)

    return env


def _evaluate_condition_run(
    cfg: ExperimentConfig,
    seed: int,
    base_dir: str,
    training_log: dict,
) -> dict:
    cond_dir, data_dir, run_dir = _condition_paths(cfg, seed, base_dir)
    env = make_symmetry_env(cfg.arena_shape, cfg.arena_size, cfg.U, cfg.F, seed=seed)
    os.makedirs(run_dir, exist_ok=True)

    # ── Final evaluation ──────────────────────────────────────────────────────
    obs_size = cfg.F * cfg.F * 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TrajectoryDataset(data_dir)

    from utils.Architectures import pRNN_th
    from utils.thetaRNN import LayerNormRNNCell

    ckpt = torch.load(os.path.join(run_dir, 'ckpt_final.pt'), map_location=device)
    model = pRNN_th(obs_size=obs_size, act_size=5, k=cfg.k,
                    hidden_size=500, cell=LayerNormRNNCell, neuralTimescale=2,
                    predOffset=PRED_OFFSET, hidden_init_sigma=HIDDEN_INIT_SIGMA)
    state = ckpt['model']
    fixed_state = {}
    for k, v in state.items():
        fixed_state[k.replace('rnn.cell._orig_mod.', 'rnn.cell.')] = v
    model.load_state_dict(fixed_state, strict=False)
    model.to(device).eval()

    h, pos = _collect_hidden_states(model, dataset, n=SUBSAMPLE_N, device=device)

    inner     = env.unwrapped
    sym_pairs = inner.precompute_symmetry_pairs()
    # Convert sym_pairs (list of pos tuples) → index pairs for sci()
    pos_list  = inner.passable_positions
    pos_index = {p: i for i, p in enumerate(pos_list)}
    sym_idx   = [
        (pos_index[a], pos_index[b])
        for a, b in sym_pairs
        if a in pos_index and b in pos_index
    ]

    result = {
        'condition_id':      cfg.condition_id,
        'seed':              seed,
        'final_srsa_euclid': float(srsa(h, pos, space_metric='euclidean')),
        'final_srsa_city':   float(srsa(h, pos, space_metric='cityblock')),
        'final_sci':         float(sci(h, pos, sym_idx)) if sym_idx else None,
        'final_manifold_id': float(manifold_id(h)),
        'dtg_curve':         dtg_curve(
                                 training_log['srsa_euclid'],
                                 training_log['srsa_city'],
                             ).tolist(),
        'srsa_euclid_curve': training_log['srsa_euclid'],
        'srsa_city_curve':   training_log['srsa_city'],
        'loss_curve':        training_log['loss'],
        'steps':             training_log['steps'],
    }

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def _run_condition(cfg: ExperimentConfig, seed: int, base_dir: str) -> dict:
    _ensure_condition_data(cfg, seed, base_dir)
    cond_dir, data_dir, run_dir = _condition_paths(cfg, seed, base_dir)

    obs_size  = cfg.F * cfg.F * 3
    run_label = f'{cfg.condition_id} seed{seed}'

    training_log = train(
        data_dir=data_dir,
        obs_size=obs_size,
        out_dir=run_dir,
        k=cfg.k,
        n_steps=cfg.n_steps,
        trunc=cfg.T,
        batch_size=cfg.B,
        seed=seed,
        run_label=run_label,
    )
    return _evaluate_condition_run(cfg, seed, base_dir, training_log)


# ── Phase runners ─────────────────────────────────────────────────────────────

def run_phase0(base_dir: str) -> bool:
    cfg = PHASE0[0]
    tqdm.write(f'\n{"="*60}')
    tqdm.write(f'Phase 0 gate: {cfg.condition_id}  (need sRSA_e > {GATE_THRESHOLD})')
    tqdm.write(f'{"="*60}')
    results = run_sweep([cfg], base_dir, label='_phase0')
    seed0_result = next(result for result in results if result['seed'] == 0)
    mean_srsa = float(np.mean([result['final_srsa_euclid'] for result in results]))
    passed = seed0_result['final_srsa_euclid'] > GATE_THRESHOLD
    tqdm.write(
        f'\nGate seed_00: sRSA_euclid = {seed0_result["final_srsa_euclid"]:.3f}  '
        f'(mean across {len(results)} seeds = {mean_srsa:.3f})  '
        f'→  {"✓ PASS" if passed else "✗ FAIL — sweep halted"}'
    )
    return passed


def run_sweep(conditions: list, base_dir: str, label: str = '') -> list:
    os.makedirs(base_dir, exist_ok=True)
    all_results = []
    trainer_mode = os.getenv('PRNN_TRAINER', 'fast')
    parallel_group = max(1, int(os.getenv('PRNN_PARALLEL_SEEDS', PARALLEL_SEED_GROUP)))

    total_runs = sum(cfg.n_seeds for cfg in conditions)
    outer_pbar = tqdm(total=total_runs, desc=f'Sweep{label}',
                      unit='run', dynamic_ncols=True)

    for cfg in conditions:
        tqdm.write(f'\n── {cfg.condition_id}  '
                   f'({cfg.arena_shape} {cfg.arena_size}×{cfg.arena_size}, '
                   f'U={cfg.U}, F={cfg.F}, k={cfg.k}, T={cfg.T}) ──')
        _ensure_condition_data(cfg, seed=0, base_dir=base_dir)

        if trainer_mode != 'legacy' and parallel_group > 1 and cfg.n_seeds > 1:
            for group_start in range(0, cfg.n_seeds, parallel_group):
                seeds = list(range(group_start, min(group_start + parallel_group, cfg.n_seeds)))
                out_dirs = [_condition_paths(cfg, seed, base_dir)[2] for seed in seeds]
                run_labels = [f'{cfg.condition_id} seed{seed}' for seed in seeds]
                obs_size = cfg.F * cfg.F * 3
                logs_by_seed = train_parallel_seeds(
                    data_dir=_condition_paths(cfg, seeds[0], base_dir)[1],
                    obs_size=obs_size,
                    out_dirs=out_dirs,
                    seeds=seeds,
                    k=cfg.k,
                    n_steps=cfg.n_steps,
                    trunc=cfg.T,
                    batch_size=cfg.B,
                    run_labels=run_labels,
                )
                for seed in seeds:
                    result = _evaluate_condition_run(cfg, seed, base_dir, logs_by_seed[seed])
                    all_results.append(result)
                    outer_pbar.update(1)
                    outer_pbar.set_postfix({
                        'cond':   cfg.condition_id,
                        'seed':   seed,
                        'sRSA_e': f'{result["final_srsa_euclid"]:.3f}',
                    })
        else:
            for seed in range(cfg.n_seeds):
                result = _run_condition(cfg, seed, base_dir)
                all_results.append(result)
                outer_pbar.update(1)
                outer_pbar.set_postfix({
                    'cond':   cfg.condition_id,
                    'seed':   seed,
                    'sRSA_e': f'{result["final_srsa_euclid"]:.3f}',
                })

    outer_pbar.close()

    out_path = os.path.join(base_dir, f'all_results{label}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    tqdm.write(f'\n✓ Results written to {out_path}')
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='project5_symmetry experiment sweep'
    )
    parser.add_argument('--phase', type=str, default='0',
                        choices=['0', '1', '2a', '2b', '4a', '4b', 'all'],
                        help='Which phase to run (default: 0 = gate only)')
    parser.add_argument('--out',   type=str, default=BASE_OUT,
                        help='Output directory root')
    parser.add_argument('--no-gate', action='store_true',
                        help='Skip Phase 0 gate check')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.phase == '0':
        run_phase0(args.out)
        return

    if not args.no_gate and not run_phase0(args.out):
        return   # gate failed

    phase_map = {
        '1':   (PHASE1,  '_phase1'),
        '2a':  (PHASE2A, '_phase2a'),
        '2b':  (PHASE2B, '_phase2b'),
        '4a':  (PHASE4A, '_phase4a'),
        '4b':  (PHASE4B, '_phase4b'),
        'all': (ALL_CONDITIONS, '_all'),
    }
    conditions, label = phase_map[args.phase]
    run_sweep(conditions, args.out, label=label)


if __name__ == '__main__':
    main()
