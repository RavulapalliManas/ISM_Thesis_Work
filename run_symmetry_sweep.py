#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
os.environ.setdefault('PRNN_TRAINER', 'fast')

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
    observation_discriminability,
    place_field_spatial_coherence,
    representational_geometry_consistency,
    srsa,
)
from project5_symmetry.experiments.configs import PHASE0
from project5_symmetry.training.dataset import TrajectoryDataset
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


def _wrapped_env(condition: str, seed: int, use_landmarks: bool = True) -> PixelObsWrapper:
    return PixelObsWrapper(_build_env(condition, seed, use_landmarks=use_landmarks), tile_size=1)


def _condition_dir(condition: str, runs_root: Path = RUNS_DIR) -> Path:
    return runs_root / condition


def _condition_data_dir(condition: str, runs_root: Path = RUNS_DIR) -> Path:
    return _condition_dir(condition, runs_root) / 'trajectories'


def _seed_run_dir(condition: str, seed: int, runs_root: Path = RUNS_DIR) -> Path:
    return _condition_dir(condition, runs_root) / f'seed_{seed:02d}'


def _seed_metrics_paths(condition: str, seed: int, runs_root: Path = RUNS_DIR) -> tuple[Path, Path]:
    run_dir = _seed_run_dir(condition, seed, runs_root)
    return run_dir / 'evaluation.pkl', run_dir / 'evaluation.json'


def _condition_summary_paths(condition: str, runs_root: Path = RUNS_DIR) -> tuple[Path, Path]:
    cond_dir = _condition_dir(condition, runs_root)
    return cond_dir / 'condition_summary.pkl', cond_dir / 'condition_summary.json'


def _condition_observation_paths(condition: str, runs_root: Path = RUNS_DIR) -> tuple[Path, Path]:
    cond_dir = _condition_dir(condition, runs_root)
    return cond_dir / 'observation_summary.pkl', cond_dir / 'observation_summary.json'


def _load_trained_model(ckpt_path: Path, obs_size: int, k: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = pRNN_th(
        obs_size=obs_size,
        act_size=5,
        k=k,
        hidden_size=500,
        cell=LayerNormRNNCell,
        neuralTimescale=2,
        predOffset=PRED_OFFSET,
        hidden_init_sigma=HIDDEN_INIT_SIGMA,
    )
    state = ckpt['model']
    fixed_state = {}
    for key, value in state.items():
        fixed_state[key.replace('rnn.cell._orig_mod.', 'rnn.cell.')] = value
    model.load_state_dict(fixed_state, strict=False)
    model.to(device).eval()
    return model


def _ensure_condition_data(
    condition: str,
    n_traj: int,
    runs_root: Path = RUNS_DIR,
    dataset_workers: int = 12,
) -> Path:
    cond_dir = _condition_dir(condition, runs_root)
    data_dir = _condition_data_dir(condition, runs_root)
    cond_dir.mkdir(parents=True, exist_ok=True)

    env = _wrapped_env(condition, seed=0, use_landmarks=True)
    generate_dataset(
        env,
        n_traj=n_traj,
        T=P0_CFG.T,
        out_dir=str(data_dir),
        n_workers=dataset_workers,
        desc=f'symmetry {condition} trajectories',
    )
    return data_dir


def _condition_observation_summary(condition: str) -> dict:
    env = _wrapped_env(condition, seed=0, use_landmarks=True)
    inner = env.unwrapped
    positions = np.array(inner.passable_positions, dtype=np.int32)
    obs_reprs = []
    for pos in inner.passable_positions:
        heading_obs = [get_obs_at(env, pos, heading) for heading in range(4)]
        obs_reprs.append(np.mean(heading_obs, axis=0))
    obs_reprs = np.stack(obs_reprs, axis=0)
    return {
        'positions': positions,
        'observations': obs_reprs,
        'odi': observation_discriminability(obs_reprs, positions),
    }


def _write_condition_observation_summary(condition: str, obs_summary: dict, runs_root: Path = RUNS_DIR):
    pkl_path, json_path = _condition_observation_paths(condition, runs_root)
    _write_pickle(pkl_path, obs_summary)
    _write_json(json_path, {
        'odi': obs_summary['odi'],
        'n_positions': int(obs_summary['positions'].shape[0]),
        'positions_path': str(pkl_path),
    })


def _evaluate_run(condition: str, seed: int, runs_root: Path = RUNS_DIR) -> dict:
    run_dir = _seed_run_dir(condition, seed, runs_root)
    data_dir = _condition_data_dir(condition, runs_root)
    env = _wrapped_env(condition, seed=0, use_landmarks=True)
    inner = env.unwrapped

    obs_size = P0_CFG.F * P0_CFG.F * 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TrajectoryDataset(str(data_dir))
    model = _load_trained_model(run_dir / 'ckpt_final.pt', obs_size=obs_size, k=P0_CFG.k, device=device)
    eval_n = max(SUBSAMPLE_N, len(inner.passable_positions) * 32)
    hidden, positions = _collect_hidden_states(model, dataset, n=eval_n, device=device)

    aggregated = aggregate_hidden_by_position(
        hidden,
        positions,
        passable_positions=inner.passable_positions,
    )
    position_hidden = aggregated['hidden']
    position_array = aggregated['positions']

    srsa_value, rsa_matrix = srsa(
        position_hidden,
        position_array,
        space_metric='euclidean',
        return_matrix=True,
    )
    place_field = place_field_spatial_coherence(
        hidden,
        positions,
        arena_size=inner.arena_size,
    )
    rgc = representational_geometry_consistency(position_hidden)

    result = {
        'condition': condition,
        'seed': seed,
        'srsa': float(srsa_value),
        'rsa_matrix': rsa_matrix,
        'position_hidden': position_hidden,
        'position_array': position_array,
        'position_counts': aggregated['counts'],
        'place_field_coherence': place_field,
        'rgc': rgc,
    }
    pkl_path, json_path = _seed_metrics_paths(condition, seed, runs_root)
    _write_pickle(pkl_path, result)
    _write_json(json_path, {
        'condition': condition,
        'seed': seed,
        'srsa': result['srsa'],
        'n_positions': int(position_array.shape[0]),
        'position_hidden_shape': list(position_hidden.shape),
        'rsa_matrix_shape': list(rsa_matrix.shape),
        'place_field_coherence': {
            'mean_score': result['place_field_coherence']['mean_score'],
            'std_score': result['place_field_coherence']['std_score'],
            'n_valid_units': result['place_field_coherence']['n_valid_units'],
            'evs_threshold': result['place_field_coherence']['evs_threshold'],
        },
        'rgc': result['rgc'],
        'pickle_path': str(pkl_path),
    })
    return result


def _train_single_seed(
    condition: str,
    seed: int,
    runs_root: Path = RUNS_DIR,
) -> dict:
    data_dir = _condition_data_dir(condition, runs_root)
    run_dir = _seed_run_dir(condition, seed, runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=str(data_dir),
        obs_size=P0_CFG.F * P0_CFG.F * 3,
        out_dir=str(run_dir),
        k=P0_CFG.k,
        n_steps=P0_CFG.n_steps,
        trunc=P0_CFG.T,
        batch_size=P0_CFG.B,
        seed=seed,
        run_label=f'symmetry {condition} seed{seed}',
    )
    return _evaluate_run(condition, seed, runs_root=runs_root)


def _train_seed_group(
    condition: str,
    seeds: list[int],
    runs_root: Path = RUNS_DIR,
) -> dict[int, dict]:
    data_dir = _condition_data_dir(condition, runs_root)
    out_dirs = []
    run_labels = []
    for seed in seeds:
        run_dir = _seed_run_dir(condition, seed, runs_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(str(run_dir))
        run_labels.append(f'symmetry {condition} seed{seed}')

    train_parallel_seeds(
        data_dir=str(data_dir),
        obs_size=P0_CFG.F * P0_CFG.F * 3,
        out_dirs=out_dirs,
        seeds=seeds,
        k=P0_CFG.k,
        n_steps=P0_CFG.n_steps,
        trunc=P0_CFG.T,
        batch_size=P0_CFG.B,
        run_labels=run_labels,
    )
    return {seed: _evaluate_run(condition, seed, runs_root=runs_root) for seed in seeds}


def run_validation(parallel_seeds: int, dataset_workers: int) -> bool:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checks_passed = True

    print('CHECK 1: Tile bounds')
    try:
        for condition in CONDITIONS[::-1]:
            env = _build_env(condition, seed=0)
            tiles = env._get_landmark_tiles()
            assert all(1 <= r <= 18 and 1 <= c <= 18 for (r, c) in tiles)
            print(f'  {condition}: {len(tiles)} tiles, all in bounds — PASS')
    except Exception:
        checks_passed = False
        print('  Tile bounds — FAIL')

    print('CHECK 2: S4 rotational invariance')
    try:
        env_s4 = _build_env('s4', seed=0)
        tiles = env_s4._get_landmark_tiles()
        rotated = _rotate90cw(tiles)
        assert set(rotated.keys()) == set(tiles.keys()), 'S4 NOT C4-symmetric — FAIL'
        print('  S4 C4-symmetry geometry — PASS')
    except Exception as exc:
        checks_passed = False
        print(f'  {exc}')

    print('CHECK 3: S2 180° invariance')
    try:
        env_s2 = _build_env('s2', seed=0)
        tiles = env_s2._get_landmark_tiles()
        rotated = _rotate180(tiles)
        assert set(rotated.keys()) == set(tiles.keys()), 'S2 NOT C2-symmetric — FAIL'
        print('  S2 C2-symmetry geometry — PASS')
    except Exception as exc:
        checks_passed = False
        print(f'  {exc}')

    print('CHECK 4: S1 quadrant separation')
    try:
        env_s1 = _build_env('s1', seed=0)
        tiles = env_s1._get_landmark_tiles()
        q1 = {(r, c) for (r, c) in tiles if r <= 9 and c <= 9}
        q2 = {(r, c) for (r, c) in tiles if r <= 9 and c >= 10}
        q3 = {(r, c) for (r, c) in tiles if r >= 10 and c >= 10}
        q4 = {(r, c) for (r, c) in tiles if r >= 10 and c <= 9}
        assert len(q1) > 0 and len(q2) > 0 and len(q3) > 0 and len(q4) > 0
        print('  S1 all four quadrants populated — PASS')
    except Exception:
        checks_passed = False
        print('  S1 quadrant separation — FAIL')

    print('CHECK 5: run_fast_train executes for 10 steps without error')
    try:
        validate_root = VALIDATION_DIR
        validate_root.mkdir(parents=True, exist_ok=True)
        data_dir = _ensure_condition_data(
            condition='s1',
            n_traj=VALIDATION_N_TRAJ,
            runs_root=validate_root,
            dataset_workers=max(1, min(dataset_workers, 1)),
        )
        run_dir = _seed_run_dir('s1', 0, validate_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        train(
            data_dir=str(data_dir),
            obs_size=P0_CFG.F * P0_CFG.F * 3,
            out_dir=str(run_dir),
            k=P0_CFG.k,
            n_steps=10,
            trunc=P0_CFG.T,
            batch_size=P0_CFG.B,
            seed=0,
            run_label='symmetry s1 seed0',
        )
        _evaluate_run('s1', 0, runs_root=validate_root)
        print('  Smoke test (s1, seed=0, 10 steps) — PASS')
    except Exception as exc:
        checks_passed = False
        print(f'  Smoke test (s1, seed=0, 10 steps) — FAIL: {exc}')

    return checks_passed


def _seed_group_iterator(n_seeds: int, parallel_seeds: int):
    for start in range(0, n_seeds, parallel_seeds):
        yield list(range(start, min(start + parallel_seeds, n_seeds)))


def run_full_sweep(parallel_seeds: int, dataset_workers: int, n_seeds: int = N_SEEDS):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    summary = {}
    overall_start = time.time()

    for condition in CONDITIONS:
        print(f'\n=== condition={condition} ===')
        _ensure_condition_data(
            condition=condition,
            n_traj=P0_CFG.n_traj,
            runs_root=RUNS_DIR,
            dataset_workers=dataset_workers,
        )
        obs_summary = _condition_observation_summary(condition)
        _write_condition_observation_summary(condition, obs_summary, runs_root=RUNS_DIR)

        for seeds in _seed_group_iterator(n_seeds, parallel_seeds):
            group_start = time.time()
            if len(seeds) == 1:
                seed = seeds[0]
                result = _train_single_seed(condition, seed, runs_root=RUNS_DIR)
                results[(condition, seed)] = result
                elapsed = int(time.time() - group_start)
                print(f'[condition={condition} seed={seed}] sRSA={result["srsa"]:.3f}  (elapsed: {elapsed}s)')
            else:
                group_results = _train_seed_group(condition, seeds, runs_root=RUNS_DIR)
                for seed in seeds:
                    result = group_results[seed]
                    results[(condition, seed)] = result
                    elapsed = int(time.time() - group_start)
                    print(f'[condition={condition} seed={seed}] sRSA={result["srsa"]:.3f}  (elapsed: {elapsed}s)')

        rsa_matrices = [results[(condition, seed)]['rsa_matrix'] for seed in range(n_seeds)]
        position_hidden = [results[(condition, seed)]['position_hidden'] for seed in range(n_seeds)]
        srsa_values = [results[(condition, seed)]['srsa'] for seed in range(n_seeds)]
        coherence_values = [
            results[(condition, seed)]['place_field_coherence']['mean_score']
            for seed in range(n_seeds)
        ]
        rgc_values = [results[(condition, seed)]['rgc']['stress'] for seed in range(n_seeds)]
        rgc_pca_values = [results[(condition, seed)]['rgc']['stress_pca'] for seed in range(n_seeds)]

        summary[condition] = {
            'srsa_per_seed': srsa_values,
            'srsa_mean': float(np.nanmean(srsa_values)),
            'srsa_std': float(np.nanstd(srsa_values)),
            'alignment': cross_seed_rsa_alignment(rsa_matrices),
            'cca_alignment': cross_seed_cca_alignment(position_hidden),
            'place_field_coherence_per_seed': coherence_values,
            'place_field_coherence_mean': float(np.nanmean(coherence_values)),
            'place_field_coherence_std': float(np.nanstd(coherence_values)),
            'odi': obs_summary['odi'],
            'rgc_stress_per_seed': rgc_values,
            'rgc_stress_mean': float(np.nanmean(rgc_values)),
            'rgc_stress_std': float(np.nanstd(rgc_values)),
            'rgc_pca_stress_per_seed': rgc_pca_values,
            'rgc_pca_stress_mean': float(np.nanmean(rgc_pca_values)),
            'rgc_pca_stress_std': float(np.nanstd(rgc_pca_values)),
            'positions': obs_summary['positions'],
        }
        cond_pkl_path, cond_json_path = _condition_summary_paths(condition, runs_root=RUNS_DIR)
        _write_pickle(cond_pkl_path, summary[condition])
        _write_json(cond_json_path, {
            'condition': condition,
            'n_seeds': n_seeds,
            'srsa_mean': summary[condition]['srsa_mean'],
            'srsa_std': summary[condition]['srsa_std'],
            'cca_alignment': summary[condition]['cca_alignment'],
            'alignment': summary[condition]['alignment'],
            'place_field_coherence_mean': summary[condition]['place_field_coherence_mean'],
            'place_field_coherence_std': summary[condition]['place_field_coherence_std'],
            'odi': summary[condition]['odi'],
            'rgc_stress_mean': summary[condition]['rgc_stress_mean'],
            'rgc_stress_std': summary[condition]['rgc_stress_std'],
            'rgc_pca_stress_mean': summary[condition]['rgc_pca_stress_mean'],
            'rgc_pca_stress_std': summary[condition]['rgc_pca_stress_std'],
            'pickle_path': str(cond_pkl_path),
        })

    _write_pickle(RAW_PATH, results)
    _write_pickle(SUMMARY_PATH, summary)

    manifest = {
        'conditions': CONDITIONS,
        'n_seeds': n_seeds,
        'parallel_seeds': parallel_seeds,
        'dataset_workers': dataset_workers,
        'elapsed_s': int(time.time() - overall_start),
        'results_root': str(SWEEP_DIR),
        'raw_path': str(RAW_PATH),
        'summary_path': str(SUMMARY_PATH),
    }
    _write_json(MANIFEST_PATH, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--parallel-seeds', type=int, default=4)
    parser.add_argument('--dataset-workers', type=int, default=12)
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS)
    parser.add_argument('--skip-validate', action='store_true')
    args = parser.parse_args()

    if not args.skip_validate:
        passed = run_validation(
            parallel_seeds=args.parallel_seeds,
            dataset_workers=args.dataset_workers,
        )
        if args.validate:
            raise SystemExit(0 if passed else 1)
        if not passed:
            raise SystemExit('Validation failed. Full sweep aborted.')
    elif args.validate:
        raise SystemExit(0)

    if args.validate:
        return

    run_full_sweep(
        parallel_seeds=max(1, args.parallel_seeds),
        dataset_workers=max(1, args.dataset_workers),
        n_seeds=max(1, args.n_seeds),
    )


if __name__ == '__main__':
    main()
