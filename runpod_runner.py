"""
RunPod auto-runner for project5_symmetry.

Runs the remaining symmetry sweep seeds and HD ablation seeds on a single-GPU
RunPod instance. The script is intentionally self-contained, but it imports the
existing project training helpers for model construction, optimizer setup,
dataset loading, checkpoint format, and metric aggregation.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import multiprocessing
import os
import pickle
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PER_SEED_VRAM_GB = 2.0
VRAM_HEADROOM_GB = 1.0
MAX_PARALLEL_CAP = 12
RUNPOD_A5000_USD_PER_HOUR = 0.27

SYMMETRY_CHECKPOINTS = [5000, 10000, 20000, 40000, 60000, 80000]
ABLATION_CHECKPOINTS = [10000, 20000, 30000, 40000]
SRSA_LOG_EVERY = 1000
GEOMETRY_LOG_EVERY = 5000
EVS_THRESHOLD = 0.10


@dataclass(frozen=True)
class CodebasePaths:
    launch_dir: Path
    repo_root: Path
    package_dir: Path
    results_root: Path


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def find_codebase_paths() -> CodebasePaths:
    """Find repo/package/results roots for both local and RunPod layouts."""
    launch_dir = Path.cwd().resolve()
    candidates = [launch_dir, *launch_dir.parents]

    for base in candidates:
        nested_train = base / "project5_symmetry" / "training" / "train.py"
        flat_train = base / "training" / "train.py"
        if nested_train.exists():
            repo_root = base
            package_dir = base / "project5_symmetry"
            break
        if flat_train.exists():
            repo_root = base.parent
            package_dir = base
            break
    else:
        print("WARNING: Could not locate project5_symmetry/training/train.py from cwd.")
        repo_root = launch_dir
        package_dir = launch_dir / "project5_symmetry"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(package_dir.parent) not in sys.path:
        sys.path.insert(0, str(package_dir.parent))

    direct_results = launch_dir / "results"
    package_results = package_dir / "results"
    repo_results = repo_root / "results"

    if (direct_results / "symmetry_sweep").exists():
        results_root = direct_results
    elif (package_results / "symmetry_sweep").exists():
        results_root = package_results
    elif (repo_results / "symmetry_sweep").exists():
        results_root = repo_results
    elif (direct_results / "ablation").exists():
        results_root = direct_results
    elif package_results.exists():
        results_root = package_results
    elif (repo_results / "ablation").exists():
        results_root = repo_results
    else:
        results_root = direct_results if (launch_dir / "training" / "train.py").exists() else package_results

    return CodebasePaths(
        launch_dir=launch_dir,
        repo_root=repo_root.resolve(),
        package_dir=package_dir.resolve(),
        results_root=results_root.resolve(),
    )


PATHS = find_codebase_paths()


def detect_hardware() -> dict[str, Any]:
    """Detect GPU, VRAM, RAM, CPU count, and calculate safe parallelism."""
    import psutil
    import torch

    gpu_name = "CPU-only (no CUDA GPU detected)"
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    available_vram_gb = 0.0

    try:
        gm = importlib.import_module("gpu_manager")
        if hasattr(gm, "HardwareProfile"):
            hw = gm.HardwareProfile()
            gpu_name = getattr(hw, "gpu_name", gpu_name)
            total_vram_gb = float(getattr(hw, "vram_total_mb", 0)) / 1024.0
            try:
                _, _used_mb, free_mb = hw.gpu_stats()
                free_vram_gb = float(free_mb) / 1024.0
            except Exception as exc:
                print(f"WARNING: gpu_manager.gpu_stats failed: {exc}")
    except Exception as exc:
        print(f"WARNING: gpu_manager.HardwareProfile unavailable: {exc}")

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            total_vram_gb = props.total_memory / (1024 ** 3)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_vram_gb = free_bytes / (1024 ** 3)
            total_vram_gb = total_bytes / (1024 ** 3)
        except Exception as exc:
            print(f"WARNING: torch CUDA VRAM detection failed: {exc}")

    if free_vram_gb > 0:
        available_vram_gb = free_vram_gb
    else:
        available_vram_gb = max(0.0, total_vram_gb - VRAM_HEADROOM_GB)

    ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_available_gb = psutil.virtual_memory().available / (1024 ** 3)
    cpu_count = os.cpu_count() or multiprocessing.cpu_count()

    # Per-seed VRAM measured at 0.35GB actual on A5000
    # Use measured value not estimated 2.0GB
    max_parallel_gpu = max(1, math.floor((available_vram_gb - VRAM_HEADROOM_GB) / 0.5))
    # 0.5GB per seed with safety margin over measured 0.35GB
    max_parallel_cpu = max(1, cpu_count // 2)
    optimal_parallel = max(1, min(max_parallel_gpu, max_parallel_cpu, MAX_PARALLEL_CAP))

    print("Measured VRAM per seed: 0.35GB")
    print("Safety margin per seed: 0.50GB")
    print(f"Max parallel seeds (VRAM): {max_parallel_gpu}")
    print(f"Selected parallelism: {optimal_parallel}")
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Available VRAM: {available_vram_gb:.2f} GB")
    print(f"System RAM: {ram_total_gb:.2f} GB total, {ram_available_gb:.2f} GB available")
    print(f"CPU count: {cpu_count}")
    print(f"Max parallel seeds (VRAM): {max_parallel_gpu}")
    print(f"Max parallel seeds (CPU): {max_parallel_cpu}")
    print(f"Selected parallelism: {optimal_parallel} seeds in parallel")

    return {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "available_vram_gb": available_vram_gb,
        "system_ram_total_gb": ram_total_gb,
        "system_ram_available_gb": ram_available_gb,
        "cpu_count": cpu_count,
        "max_parallel_gpu": max_parallel_gpu,
        "max_parallel_cpu": max_parallel_cpu,
        "optimal_parallel": optimal_parallel,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }


def _symmetry_run_dir(condition: str, seed: int) -> Path:
    return PATHS.results_root / "symmetry_sweep" / condition / f"seed_{seed:02d}"


def _ablation_run_dir(hd_mode: str, seed: int) -> Path:
    return PATHS.results_root / "ablation" / hd_mode / f"seed_{seed:02d}"


def check_existing(condition: str, seed: int, hd_mode: str | None = None) -> bool:
    """Return True if ckpt_final.pt exists for this run."""
    if hd_mode is None:
        path = _symmetry_run_dir(condition, seed) / "ckpt_final.pt"
    else:
        path = _ablation_run_dir(hd_mode, seed) / "ckpt_final.pt"
    return path.exists()


def apply_hd_mode(d_t, hd_mode: str):
    """
    Apply HD substitution. Speed dimension index 0 is always preserved.
    Heading dimensions indices 1:5 are modified for ablated/degraded modes.
    """
    if hd_mode == "full":
        return d_t
    if hd_mode not in {"ablated", "degraded"}:
        raise ValueError(f"Unexpected hd_mode: {hd_mode}")

    d_t_mod = d_t.clone()
    fill_value = 0.0 if hd_mode == "ablated" else 0.25

    if d_t.dim() == 1:
        d_t_mod[1:] = fill_value
    elif d_t.dim() == 2:
        d_t_mod[:, 1:] = fill_value
    elif d_t.dim() == 3:
        d_t_mod[..., 1:] = fill_value
    else:
        raise ValueError(f"Unexpected d_t dimensions: {d_t.dim()}")
    return d_t_mod


def verify_hd_substitution(d_t_mod, hd_mode: str):
    if hd_mode == "ablated":
        heading_sum = d_t_mod[..., 1:].sum().item()
        assert abs(heading_sum) < 1e-6, f"HD_ABLATED heading not zero: {heading_sum}"
        print(f"HD_ABLATED verified: heading sum = {heading_sum:.6f}", flush=True)
    elif hd_mode == "degraded":
        if d_t_mod.dim() == 1:
            vals = d_t_mod[1:].tolist()
        elif d_t_mod.dim() == 2:
            vals = d_t_mod[0, 1:].tolist()
        else:
            vals = d_t_mod[0, 0, 1:].tolist()
        assert all(abs(v - 0.25) < 1e-6 for v in vals), f"HD_DEGRADED heading not uniform: {vals}"
        print(f"HD_DEGRADED verified: heading = {vals}", flush=True)

    speed_val = d_t_mod[..., 0].mean().item()
    print(f"  Speed preserved, mean = {speed_val:.4f}", flush=True)


def _ensure_condition_data(condition: str, dataset_workers: int) -> tuple[Path, int]:
    """Use run_symmetry_sweep's dataset generation when available."""
    try:
        sweep = importlib.import_module("run_symmetry_sweep")
        old_runs_dir = getattr(sweep, "RUNS_DIR", None)
        if old_runs_dir is not None:
            sweep.RUNS_DIR = PATHS.results_root / "symmetry_sweep"
        data_dir = sweep._ensure_condition_data(
            condition,
            n_traj=sweep.P0_CFG.n_traj,
            runs_root=PATHS.results_root / "symmetry_sweep",
            dataset_workers=dataset_workers,
        )
        obs_size = sweep.P0_CFG.F * sweep.P0_CFG.F * 3
        return Path(data_dir), obs_size
    except Exception as exc:
        print(f"WARNING: run_symmetry_sweep._ensure_condition_data failed for {condition}: {exc}")
        data_dir = PATHS.results_root / "symmetry_sweep" / condition / "trajectories"
        return data_dir, 7 * 7 * 3


def _load_checkpoint_if_present(output_dir: Path, checkpoint_steps: list[int], model, optimizer, device):
    import torch

    for step in sorted(checkpoint_steps, reverse=True):
        ckpt_path = output_dir / f"ckpt_{step}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("model_state")
        opt_state = ckpt.get("optimizer") or ckpt.get("optimizer_state_dict") or ckpt.get("optimizer_state")
        if state is None:
            print(f"WARNING: {ckpt_path} has no model state; ignoring resume checkpoint.")
            continue
        fixed_state = {key.replace("rnn.cell._orig_mod.", "rnn.cell."): value for key, value in state.items()}
        model.load_state_dict(fixed_state, strict=False)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        print(f"  Resuming from step {step}", flush=True)
        return step
    return 0


def _collect_hidden_states_hd(model, dataset, n: int, device, hd_mode: str):
    import torch

    was_training = model.training
    model.eval()
    all_h, all_pos = [], []
    total = 0

    with torch.no_grad():
        for idx in torch.randperm(len(dataset)):
            obs, act, pos, _heading = dataset[int(idx)]
            act_mod = apply_hd_mode(act.to(device), hd_mode)
            _, h, _ = model(obs.unsqueeze(0).to(device), act_mod.unsqueeze(0))
            h0 = h.squeeze(0).detach().cpu().numpy() if h.dim() == 3 else h.detach().cpu().numpy()
            p = pos.numpy()[: h0.shape[0]]
            take = min(h0.shape[0], n - total)
            all_h.append(h0[:take])
            all_pos.append(p[:take])
            total += take
            if total >= n:
                break

    if was_training:
        model.train()
    return np.concatenate(all_h, 0), np.concatenate(all_pos, 0)


def extract_H_matrix_compatible(model, eval_dataset, device, condition: str, hd_mode: str):
    """Use project metrics where possible; fall back when extract_H_matrix is absent."""
    metrics = importlib.import_module("project5_symmetry.evaluation.metrics")
    if hasattr(metrics, "extract_H_matrix"):
        try:
            return metrics.extract_H_matrix(model, device, condition)
        except TypeError:
            try:
                return metrics.extract_H_matrix(model=model, device=device, condition=condition)
            except Exception as exc:
                print(f"WARNING: extract_H_matrix call failed, using hidden aggregation fallback: {exc}")
        except Exception as exc:
            print(f"WARNING: extract_H_matrix failed, using hidden aggregation fallback: {exc}")
    else:
        print("WARNING: evaluation.metrics.extract_H_matrix not found; using aggregate_hidden_by_position fallback.")

    train_mod = importlib.import_module("project5_symmetry.training.train")
    hidden, positions = _collect_hidden_states_hd(
        model,
        eval_dataset,
        n=max(getattr(train_mod, "SUBSAMPLE_N", 5000), 5000),
        device=device,
        hd_mode=hd_mode,
    )
    aggregated = metrics.aggregate_hidden_by_position(hidden, positions)
    return aggregated["hidden"]


def evaluate_srsa_and_geometry_hd(
    model,
    eval_dataset,
    device,
    hd_mode: str,
    need_geometry: bool,
    n: int,
    repeats: int,
) -> tuple[float, float, dict[str, float] | None]:
    """Evaluate sRSA and, optionally, slower geometry metrics with HD substitution applied."""
    metrics = importlib.import_module("project5_symmetry.evaluation.metrics")
    srsa_e = []
    srsa_c = []
    geometry_source = None

    for rep in range(max(1, repeats)):
        hidden, positions = _collect_hidden_states_hd(model, eval_dataset, n=n, device=device, hd_mode=hd_mode)
        srsa_e.append(float(metrics.srsa(hidden, positions, space_metric="euclidean", max_n=n)))
        srsa_c.append(float(metrics.srsa(hidden, positions, space_metric="cityblock", max_n=n)))
        if rep == 0:
            geometry_source = (hidden, positions)

    geometry_metrics = None
    if need_geometry and geometry_source is not None:
        hidden, positions = geometry_source
        arena_size = int(np.max(np.rint(positions))) if positions.size else 0
        aggregated = metrics.aggregate_hidden_by_position(hidden, positions)
        position_hidden = aggregated["hidden"]
        rgc = metrics.representational_geometry_consistency(position_hidden)
        coherence = metrics.place_field_spatial_coherence(
            hidden,
            positions,
            arena_size=arena_size,
        )
        geometry_metrics = {
            "manifold_id": float(metrics.manifold_id(position_hidden)),
            "pca_variance_2d": float(rgc["pca_var_2d"]),
            "mds_stress": float(rgc["stress"]),
            "mean_field_coherence": float(coherence["mean_score"]),
        }

    return float(np.mean(srsa_e)), float(np.mean(srsa_c)), geometry_metrics


def _save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _save_pickle(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_final_model_for_eval(run_dir: Path, data_dir: Path, obs_size: int, device):
    import torch

    train_mod = importlib.import_module("project5_symmetry.training.train")
    model, _compiled = train_mod._build_model(
        obs_size=obs_size,
        act_size=5,
        k=5,
        trunc=200,
        device=device,
        compile_cell=False,
    )
    ckpt_path = run_dir / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise KeyError(f"No model state found in {ckpt_path}")
    fixed_state = {key.replace("rnn.cell._orig_mod.", "rnn.cell."): value for key, value in state.items()}
    model.load_state_dict(fixed_state, strict=False)
    model.to(device).eval()
    return model


def _evaluate_completed_run(
    condition: str,
    seed: int,
    run_dir: Path,
    data_dir: Path,
    obs_size: int,
    hd_mode: str,
) -> dict[str, Any]:
    import torch

    metrics = importlib.import_module("project5_symmetry.evaluation.metrics")
    dataset_mod = importlib.import_module("project5_symmetry.training.dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_mod.TrajectoryDataset(str(data_dir))
    model = _load_final_model_for_eval(run_dir, data_dir, obs_size, device)

    hidden, positions = _collect_hidden_states_hd(model, dataset, n=5000, device=device, hd_mode=hd_mode)
    aggregated = metrics.aggregate_hidden_by_position(hidden, positions)
    position_hidden = aggregated["hidden"]
    position_array = aggregated["positions"]
    srsa_value, rsa_matrix = metrics.srsa(
        position_hidden,
        position_array,
        space_metric="euclidean",
        return_matrix=True,
    )
    arena_size = int(np.max(np.rint(positions))) if positions.size else 18
    place_field = metrics.place_field_spatial_coherence(
        hidden,
        positions,
        arena_size=arena_size,
    )
    rgc = metrics.representational_geometry_consistency(position_hidden)

    result = {
        "condition": condition,
        "seed": seed,
        "hd_mode": hd_mode,
        "srsa": float(srsa_value),
        "rsa_matrix": rsa_matrix,
        "position_hidden": position_hidden,
        "position_array": position_array,
        "position_counts": aggregated["counts"],
        "place_field_coherence": place_field,
        "rgc": rgc,
    }
    _save_pickle(run_dir / "evaluation.pkl", result)
    _save_json(
        run_dir / "evaluation.json",
        {
            "condition": condition,
            "seed": seed,
            "hd_mode": hd_mode,
            "srsa": result["srsa"],
            "n_positions": int(position_array.shape[0]),
            "position_hidden_shape": list(position_hidden.shape),
            "rsa_matrix_shape": list(rsa_matrix.shape),
            "place_field_coherence": {
                "mean_score": result["place_field_coherence"]["mean_score"],
                "std_score": result["place_field_coherence"]["std_score"],
                "n_valid_units": result["place_field_coherence"]["n_valid_units"],
                "evs_threshold": result["place_field_coherence"]["evs_threshold"],
            },
            "rgc": result["rgc"],
            "pickle_path": str(run_dir / "evaluation.pkl"),
        },
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model, dataset
    gc.collect()
    return result


def _discover_symmetry_seed_dirs() -> list[tuple[str, int, Path]]:
    sweep_root = PATHS.results_root / "symmetry_sweep"
    records = []
    for condition in ("s1", "s2", "s4"):
        cond_dir = sweep_root / condition
        if not cond_dir.exists():
            continue
        for seed_dir in sorted(cond_dir.glob("seed_*")):
            if not (seed_dir / "ckpt_final.pt").exists():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            records.append((condition, seed, seed_dir))
    return records


def _discover_ablation_seed_dirs() -> list[tuple[str, int, Path]]:
    ablation_root = PATHS.results_root / "ablation"
    records = []
    for hd_mode in ("full", "ablated", "degraded"):
        mode_dir = ablation_root / hd_mode
        if not mode_dir.exists():
            continue
        for seed_dir in sorted(mode_dir.glob("seed_*")):
            if not (seed_dir / "ckpt_final.pt").exists():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            records.append((hd_mode, seed, seed_dir))
    return records


def _condition_data_dir_for_eval(condition: str) -> tuple[Path, int]:
    return _ensure_condition_data(condition, dataset_workers=1)


def _build_condition_summaries(evaluations: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    metrics = importlib.import_module("project5_symmetry.evaluation.metrics")
    summary = {}
    for condition in sorted({key[0] for key in evaluations}):
        cond_evals = [evaluations[key] for key in sorted(evaluations) if key[0] == condition]
        if not cond_evals:
            continue
        rsa_matrices = [e["rsa_matrix"] for e in cond_evals]
        position_hidden = [e["position_hidden"] for e in cond_evals]
        srsa_values = [float(e["srsa"]) for e in cond_evals]
        coherence_values = [float(e["place_field_coherence"]["mean_score"]) for e in cond_evals]
        rgc_values = [float(e["rgc"]["stress"]) for e in cond_evals]
        rgc_pca_values = [float(e["rgc"]["stress_pca"]) for e in cond_evals]
        cond_summary = {
            "condition": condition,
            "n_seeds": len(cond_evals),
            "seeds": [int(e["seed"]) for e in cond_evals],
            "srsa_per_seed": srsa_values,
            "srsa_mean": float(np.nanmean(srsa_values)),
            "srsa_std": float(np.nanstd(srsa_values)),
            "alignment": metrics.cross_seed_rsa_alignment(rsa_matrices) if len(rsa_matrices) > 1 else None,
            "cca_alignment": metrics.cross_seed_cca_alignment(position_hidden) if len(position_hidden) > 1 else None,
            "place_field_coherence_per_seed": coherence_values,
            "place_field_coherence_mean": float(np.nanmean(coherence_values)),
            "place_field_coherence_std": float(np.nanstd(coherence_values)),
            "rgc_stress_per_seed": rgc_values,
            "rgc_stress_mean": float(np.nanmean(rgc_values)),
            "rgc_stress_std": float(np.nanstd(rgc_values)),
            "rgc_pca_stress_per_seed": rgc_pca_values,
            "rgc_pca_stress_mean": float(np.nanmean(rgc_pca_values)),
            "rgc_pca_stress_std": float(np.nanstd(rgc_pca_values)),
        }
        cond_dir = PATHS.results_root / "symmetry_sweep" / condition
        _save_pickle(cond_dir / "condition_summary.pkl", cond_summary)
        _save_json(cond_dir / "condition_summary.json", cond_summary)
        summary[condition] = cond_summary
    _save_pickle(PATHS.results_root / "symmetry_sweep" / "symmetry_sweep_raw.pkl", evaluations)
    _save_pickle(PATHS.results_root / "symmetry_sweep" / "symmetry_sweep_summary.pkl", summary)
    _save_json(
        PATHS.results_root / "symmetry_sweep" / "symmetry_sweep_manifest.json",
        {
            "conditions": sorted(summary),
            "n_seeds_by_condition": {c: s["n_seeds"] for c, s in summary.items()},
            "results_root": str(PATHS.results_root / "symmetry_sweep"),
        },
    )
    return summary


def _as_position_grid(H: np.ndarray, positions: np.ndarray, arena_size: int = 18) -> np.ndarray:
    grid = np.full((arena_size, arena_size, H.shape[1]), np.nan, dtype=float)
    for h, (col, row) in zip(H, np.rint(positions).astype(int)):
        if 1 <= col <= arena_size and 1 <= row <= arena_size:
            grid[row - 1, col - 1] = h
    return grid


def _compute_ra_from_position_hidden(H: np.ndarray, positions: np.ndarray, evs: np.ndarray) -> tuple[float, np.ndarray]:
    grid = _as_position_grid(H, positions, arena_size=18)
    ra = np.full(H.shape[1], np.nan, dtype=float)
    valid = np.where(np.isfinite(evs) & (evs > EVS_THRESHOLD))[0]
    for unit in valid:
        field = grid[:, :, unit]
        if not np.isfinite(field).all():
            continue
        src = field.reshape(-1)
        rot = np.rot90(field).reshape(-1)
        if np.std(src) <= 1e-8 or np.std(rot) <= 1e-8:
            continue
        ra[unit] = float(np.corrcoef(src, rot)[0, 1])
    return float(np.nanmean(ra)) if np.isfinite(ra).any() else float("nan"), ra


def _rotation_permutation(positions: np.ndarray, rotation: int, arena_size: int = 18) -> np.ndarray:
    pos = np.rint(positions).astype(int)
    mapping = {tuple(p): i for i, p in enumerate(pos)}
    order = []
    for col, row in pos:
        c0, r0 = col - 1, row - 1
        for _ in range((rotation // 90) % 4):
            c0, r0 = arena_size - 1 - r0, c0
        order.append(mapping.get((c0 + 1, r0 + 1), -1))
    return np.asarray(order, dtype=int)


def _compute_paa_by_condition(eval_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    from scipy.stats import spearmanr

    out = {}
    for condition in sorted({row["condition"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["condition"] == condition]
        gains = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                rsa_i = rows[i]["rsa_matrix"]
                rsa_j = rows[j]["rsa_matrix"]
                tri = np.triu_indices(rsa_i.shape[0], k=1)
                base = float(spearmanr(rsa_i[tri], rsa_j[tri]).correlation)
                best = base
                rotations = (180,) if condition in {"s1", "s2"} else (90, 180, 270)
                for rotation in rotations:
                    perm = _rotation_permutation(rows[j]["position_array"], rotation)
                    if np.any(perm < 0):
                        continue
                    rotated = rsa_j[perm, :][:, perm]
                    best = max(best, float(spearmanr(rsa_i[tri], rotated[tri]).correlation))
                gains.append(best - base)
        out[condition] = {
            "PAA_gain_mean": float(np.nanmean(gains)) if gains else float("nan"),
            "PAA_gain_std": float(np.nanstd(gains)) if gains else float("nan"),
            "n_pairs": len(gains),
        }
    return out


def _compute_sci(H: np.ndarray, positions: np.ndarray, condition: str) -> float:
    H_norm = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-8, None)
    n = H_norm.shape[0]
    perm_90 = _rotation_permutation(positions, 90)
    perm_180 = _rotation_permutation(positions, 180)
    perm_270 = _rotation_permutation(positions, 270)
    if condition == "s4":
        pairs = {
            (min(i, int(p)), max(i, int(p)))
            for perm in (perm_90, perm_180, perm_270)
            for i, p in enumerate(perm)
            if int(p) >= 0
        }
    elif condition == "s2":
        pairs = {(min(i, int(p)), max(i, int(p))) for i, p in enumerate(perm_180) if int(p) >= 0}
    else:
        rng = np.random.default_rng(10)
        pairs = set()
        while len(pairs) < 486 and len(pairs) < n * (n - 1) // 2:
            a, b = int(rng.integers(n)), int(rng.integers(n))
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    pairs = sorted((a, b) for a, b in pairs if a != b and a < n and b < n)
    rng = np.random.default_rng(42)
    random_pairs = set()
    while len(random_pairs) < max(1, len(pairs)):
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            random_pairs.add((min(a, b), max(a, b)))
    sym_d = [np.linalg.norm(H_norm[a] - H_norm[b]) for a, b in pairs]
    rand_d = [np.linalg.norm(H_norm[a] - H_norm[b]) for a, b in random_pairs]
    return float(np.mean(sym_d) / np.clip(np.mean(rand_d), 1e-8, None))


def _compute_c2_contrast(H: np.ndarray, positions: np.ndarray) -> float:
    H_norm = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-8, None)
    perm_90 = _rotation_permutation(positions, 90)
    perm_180 = _rotation_permutation(positions, 180)
    perm_270 = _rotation_permutation(positions, 270)

    def mean_cosine_distance(pairs):
        vals = []
        for a, b in pairs:
            if b >= 0 and a != b:
                vals.append(1.0 - float(np.dot(H_norm[a], H_norm[b])))
        return float(np.nanmean(vals)) if vals else float("nan")

    c2_pairs = [(i, int(j)) for i, j in enumerate(perm_180)]
    c4_pairs = [(i, int(j)) for perm in (perm_90, perm_270) for i, j in enumerate(perm)]
    return mean_cosine_distance(c2_pairs) - mean_cosine_distance(c4_pairs)


def _compute_decode_err(H: np.ndarray, positions: np.ndarray) -> float:
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"WARNING: sklearn unavailable for decode error: {exc}")
        return float("nan")

    if H.shape[0] < 3:
        return float("nan")
    X = StandardScaler().fit_transform(H)
    y = np.asarray(positions, dtype=float)
    n_splits = min(5, H.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors = []
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        errors.append(float(np.mean((pred - y[test_idx]) ** 2) / (18 ** 2)))
    return float(np.nanmean(errors))


def _mann_whitney_rows(master_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scipy.stats import mannwhitneyu

    metrics = [
        "srsa_euclid",
        "srsa_city",
        "DTG",
        "manifold_id",
        "RA",
        "PAA_gain",
        "SCI",
        "C2_Contrast",
        "DecodeErr",
    ]
    comparisons = [("s4", "s1"), ("s4", "s2"), ("s2", "s1")]
    rows = []
    for metric in metrics:
        for a, b in comparisons:
            av = np.asarray([r[metric] for r in master_rows if r["condition"] == a], dtype=float)
            bv = np.asarray([r[metric] for r in master_rows if r["condition"] == b], dtype=float)
            av = av[np.isfinite(av)]
            bv = bv[np.isfinite(bv)]
            row = {
                "metric": metric,
                "comparison": f"{a.upper()}_vs_{b.upper()}",
                "n_a": int(av.size),
                "n_b": int(bv.size),
                "mean_a": float(np.nanmean(av)) if av.size else float("nan"),
                "mean_b": float(np.nanmean(bv)) if bv.size else float("nan"),
                "U": float("nan"),
                "p": float("nan"),
            }
            if av.size and bv.size:
                stat = mannwhitneyu(av, bv, alternative="two-sided")
                row["U"] = float(stat.statistic)
                row["p"] = float(stat.pvalue)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_tables(tables_dir: Path, rows: list[dict[str, Any]]):
    def fmt(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return "NA"
        if arr.size == 1:
            return f"{arr[0]:.3f}"
        return f"{arr.mean():.3f} $\\pm$ {arr.std(ddof=1):.3f}"

    metrics = ["srsa_euclid", "RA", "PAA_gain", "SCI", "C2_Contrast", "DecodeErr", "DTG"]
    symmetry = [r for r in rows if r["kind"] == "symmetry"]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & S4 & S2 & S1 \\\\",
        "\\midrule",
    ]
    for metric in metrics:
        vals = {
            cond: [r[metric] for r in symmetry if r["condition"] == cond]
            for cond in ("s4", "s2", "s1")
        }
        lines.append(f"{metric} & {fmt(vals['s4'])} & {fmt(vals['s2'])} & {fmt(vals['s1'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "runpod_main_results_table.tex").write_text("\n".join(lines), encoding="utf-8")

    ablation = [r for r in rows if r["kind"] == "ablation"]
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Metric & HD\\_FULL & HD\\_ABLATED & HD\\_DEGRADED \\\\",
        "\\midrule",
    ]
    for metric in metrics:
        vals = {
            mode: [r[metric] for r in ablation if r["hd_mode"] == mode]
            for mode in ("full", "ablated", "degraded")
        }
        lines.append(
            f"{metric} & {fmt(vals['full'])} & {fmt(vals['ablated'])} & {fmt(vals['degraded'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "runpod_hd_ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")


def run_posthoc_outputs() -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("POST-HOC ANALYSIS - evaluations, statistics, figures")
    print("=" * 60)

    metrics_dir = PATHS.results_root / "metrics"
    figures_dir = PATHS.results_root / "figures"
    tables_dir = PATHS.results_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    symmetry_evals: dict[tuple[str, int], dict[str, Any]] = {}
    for condition, seed, run_dir in _discover_symmetry_seed_dirs():
        try:
            data_dir, obs_size = _condition_data_dir_for_eval(condition)
            print(f"Evaluating symmetry {condition} seed_{seed:02d}")
            symmetry_evals[(condition, seed)] = _evaluate_completed_run(
                condition, seed, run_dir, data_dir, obs_size, hd_mode="full"
            )
        except Exception as exc:
            print(f"WARNING: symmetry evaluation failed for {condition} seed_{seed:02d}: {exc}")

    ablation_evals: dict[tuple[str, int], dict[str, Any]] = {}
    for hd_mode, seed, run_dir in _discover_ablation_seed_dirs():
        try:
            data_dir, obs_size = _condition_data_dir_for_eval("s4")
            print(f"Evaluating HD ablation {hd_mode} seed_{seed:02d}")
            ablation_evals[(hd_mode, seed)] = _evaluate_completed_run(
                "s4", seed, run_dir, data_dir, obs_size, hd_mode=hd_mode
            )
        except Exception as exc:
            print(f"WARNING: ablation evaluation failed for {hd_mode} seed_{seed:02d}: {exc}")

    condition_summaries = _build_condition_summaries(symmetry_evals) if symmetry_evals else {}
    _save_pickle(metrics_dir / "runpod_symmetry_evaluations.pkl", symmetry_evals)
    _save_pickle(metrics_dir / "runpod_ablation_evaluations.pkl", ablation_evals)

    eval_rows = []
    for (condition, seed), ev in sorted(symmetry_evals.items()):
        H = np.asarray(ev["position_hidden"], dtype=float)
        pos = np.asarray(ev["position_array"], dtype=float)
        evs = np.asarray(ev["place_field_coherence"]["evs"], dtype=float)
        ra_mean, ra_per_unit = _compute_ra_from_position_hidden(H, pos, evs)
        np.save(metrics_dir / f"{condition}_seed{seed:02d}_RA_per_unit.npy", ra_per_unit)
        np.save(metrics_dir / f"{condition}_seed{seed:02d}_EVS_per_unit.npy", evs)
        eval_rows.append(
            {
                "kind": "symmetry",
                "condition": condition,
                "seed": seed,
                "hd_mode": "full",
                "srsa_euclid": float(ev["srsa"]),
                "srsa_city": float("nan"),
                "DTG": float("nan"),
                "manifold_id": float("nan"),
                "RA": ra_mean,
                "PAA_gain": float("nan"),
                "SCI": _compute_sci(H, pos, condition),
                "C2_Contrast": _compute_c2_contrast(H, pos),
                "DecodeErr": _compute_decode_err(H, pos),
                "mean_field_coherence": float(ev["place_field_coherence"]["mean_score"]),
                "mds_stress": float(ev["rgc"]["stress"]),
                "pca_variance_2d": float(ev["rgc"]["pca_var_2d"]),
            }
        )

    paa = _compute_paa_by_condition(list(symmetry_evals.values())) if symmetry_evals else {}
    for row in eval_rows:
        if row["kind"] == "symmetry":
            row["PAA_gain"] = paa.get(row["condition"], {}).get("PAA_gain_mean", float("nan"))
        log_path = _symmetry_run_dir(row["condition"], row["seed"]) / "training_log.json"
        if log_path.exists():
            log = _load_json(log_path)
            if log.get("srsa_city"):
                row["srsa_city"] = float(log["srsa_city"][-1])
            if log.get("srsa_euclid"):
                row["srsa_euclid"] = float(log["srsa_euclid"][-1])
            if np.isfinite(row["srsa_euclid"]) and np.isfinite(row["srsa_city"]):
                row["DTG"] = row["srsa_euclid"] - row["srsa_city"]
            for key in ("manifold_id", "mean_field_coherence", "mds_stress", "pca_variance_2d"):
                vals = [float(v) for v in log.get(key, []) if np.isfinite(float(v))]
                if vals:
                    row[key] = vals[-1]

    ablation_rows = []
    for (hd_mode, seed), ev in sorted(ablation_evals.items()):
        H = np.asarray(ev["position_hidden"], dtype=float)
        pos = np.asarray(ev["position_array"], dtype=float)
        evs = np.asarray(ev["place_field_coherence"]["evs"], dtype=float)
        ra_mean, ra_per_unit = _compute_ra_from_position_hidden(H, pos, evs)
        np.save(metrics_dir / f"ablation_{hd_mode}_seed{seed:02d}_RA_per_unit.npy", ra_per_unit)
        np.save(metrics_dir / f"ablation_{hd_mode}_seed{seed:02d}_EVS_per_unit.npy", evs)
        row = {
            "kind": "ablation",
            "condition": "s4",
            "seed": seed,
            "hd_mode": hd_mode,
            "srsa_euclid": float(ev["srsa"]),
            "srsa_city": float("nan"),
            "DTG": float("nan"),
            "manifold_id": float("nan"),
            "RA": ra_mean,
            "PAA_gain": float("nan"),
            "SCI": _compute_sci(H, pos, "s4"),
            "C2_Contrast": _compute_c2_contrast(H, pos),
            "DecodeErr": _compute_decode_err(H, pos),
            "mean_field_coherence": float(ev["place_field_coherence"]["mean_score"]),
            "mds_stress": float(ev["rgc"]["stress"]),
            "pca_variance_2d": float(ev["rgc"]["pca_var_2d"]),
        }
        log_path = _ablation_run_dir(hd_mode, seed) / "training_log.json"
        if log_path.exists():
            log = _load_json(log_path)
            if log.get("srsa_city"):
                row["srsa_city"] = float(log["srsa_city"][-1])
            if log.get("srsa_euclid"):
                row["srsa_euclid"] = float(log["srsa_euclid"][-1])
            if np.isfinite(row["srsa_euclid"]) and np.isfinite(row["srsa_city"]):
                row["DTG"] = row["srsa_euclid"] - row["srsa_city"]
            for key in ("manifold_id", "mean_field_coherence", "mds_stress", "pca_variance_2d"):
                vals = [float(v) for v in log.get(key, []) if np.isfinite(float(v))]
                if vals:
                    row[key] = vals[-1]
        ablation_rows.append(row)

    master_rows = eval_rows + ablation_rows
    _write_csv(tables_dir / "runpod_master_metrics.csv", master_rows)
    _write_csv(tables_dir / "runpod_statistical_tests.csv", _mann_whitney_rows(eval_rows))
    _write_latex_tables(tables_dir, master_rows)
    _save_json(metrics_dir / "runpod_PAA_gains.json", paa)

    try:
        _make_runpod_figures(figures_dir, master_rows)
    except Exception as exc:
        print(f"WARNING: figure generation failed: {exc}")

    output_manifest = {
        "symmetry_evaluations": len(symmetry_evals),
        "ablation_evaluations": len(ablation_evals),
        "condition_summaries": condition_summaries,
        "tables": [
            str(tables_dir / "runpod_master_metrics.csv"),
            str(tables_dir / "runpod_statistical_tests.csv"),
            str(tables_dir / "runpod_main_results_table.tex"),
            str(tables_dir / "runpod_hd_ablation_table.tex"),
        ],
        "figures_dir": str(figures_dir),
        "metrics_dir": str(metrics_dir),
    }
    _save_json(PATHS.results_root / "runpod_posthoc_manifest.json", output_manifest)
    print(f"Post-hoc outputs written under {PATHS.results_root}")
    return output_manifest


def _make_runpod_figures(figures_dir: Path, rows: list[dict[str, Any]]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    colors = {"s4": "#2166ac", "s2": "#f4a582", "s1": "#4daf4a"}
    symmetry = [r for r in rows if r["kind"] == "symmetry"]
    ablation = [r for r in rows if r["kind"] == "ablation"]

    if symmetry:
        metrics_to_plot = [
            ("srsa_euclid", "sRSA Euclid"),
            ("RA", "RA"),
            ("SCI", "SCI"),
            ("C2_Contrast", "C2 Contrast"),
            ("DecodeErr", "Decode Err"),
            ("DTG", "DTG"),
        ]
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 3))
        for ax, (metric, label) in zip(axes, metrics_to_plot):
            conds = ["s4", "s2", "s1"]
            means = []
            sems = []
            for cond in conds:
                vals = np.asarray([r[metric] for r in symmetry if r["condition"] == cond], dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if vals.size else np.nan)
                sems.append(float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
            ax.bar(range(len(conds)), means, yerr=sems, color=[colors[c] for c in conds], capsize=3)
            for idx, cond in enumerate(conds):
                vals = np.asarray([r[metric] for r in symmetry if r["condition"] == cond], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ax.scatter(np.full(vals.size, idx), vals, color="black", s=10, zorder=3)
            ax.set_xticks(range(len(conds)), [c.upper() for c in conds])
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "runpod_final_metric_summary.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / "runpod_final_metric_summary.pdf", bbox_inches="tight")
        plt.close(fig)

    if ablation:
        metrics_to_plot = [
            ("srsa_euclid", "sRSA Euclid"),
            ("RA", "RA"),
            ("SCI", "SCI"),
            ("C2_Contrast", "C2 Contrast"),
            ("DecodeErr", "Decode Err"),
            ("DTG", "DTG"),
        ]
        modes = ["full", "ablated", "degraded"]
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 3))
        for ax, (metric, label) in zip(axes, metrics_to_plot):
            means = []
            sems = []
            for mode in modes:
                vals = np.asarray([r[metric] for r in ablation if r["hd_mode"] == mode], dtype=float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.mean(vals)) if vals.size else np.nan)
                sems.append(float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
            ax.bar(range(len(modes)), means, yerr=sems, color=["#5ab4ac", "#d8b365", "#998ec3"], capsize=3)
            for idx, mode in enumerate(modes):
                vals = np.asarray([r[metric] for r in ablation if r["hd_mode"] == mode], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ax.scatter(np.full(vals.size, idx), vals, color="black", s=10, zorder=3)
            ax.set_xticks(range(len(modes)), ["FULL", "ABLATED", "DEGRADED"], rotation=20)
            ax.set_ylabel(label)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / "runpod_hd_ablation_summary.png", dpi=200, bbox_inches="tight")
        fig.savefig(figures_dir / "runpod_hd_ablation_summary.pdf", bbox_inches="tight")
        plt.close(fig)

    _make_training_curve_figure(figures_dir)


def _make_training_curve_figure(figures_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_records = []
    for condition, seed, run_dir in _discover_symmetry_seed_dirs():
        log_path = run_dir / "training_log.json"
        if log_path.exists():
            log_records.append((condition, "full", _load_json(log_path)))
    for hd_mode, seed, run_dir in _discover_ablation_seed_dirs():
        log_path = run_dir / "training_log.json"
        if log_path.exists():
            log_records.append((f"ablation_{hd_mode}", hd_mode, _load_json(log_path)))
    if not log_records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for label, _mode, log in log_records:
        steps = np.asarray(log.get("steps", []), dtype=float)
        if not steps.size:
            continue
        axes[0].plot(steps, log.get("loss", []), alpha=0.35, linewidth=0.8, label=label)
        axes[1].plot(steps, log.get("srsa_euclid", []), alpha=0.35, linewidth=0.8)
        axes[1].plot(steps, log.get("srsa_city", []), alpha=0.25, linewidth=0.8, linestyle="--")
        dtg = np.asarray(log.get("srsa_euclid", []), dtype=float) - np.asarray(log.get("srsa_city", []), dtype=float)
        axes[2].plot(steps, dtg, alpha=0.35, linewidth=0.8)
    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("sRSA")
    axes[2].set_ylabel("DTG")
    for ax in axes:
        ax.set_xlabel("step")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "runpod_training_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "runpod_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def run_seed_worker(args: dict[str, Any]):
    try:
        return _run_seed_worker_inner(args)
    except Exception as exc:
        is_oom = exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
        if is_oom:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                f'[OOM] {args["condition"]} seed_{args["seed_idx"]:02d} '
                "reduce parallel count and retry",
                flush=True,
            )
            return {
                "condition": args["condition"],
                "seed": args["seed_idx"],
                "hd_mode": args["hd_mode"],
                "final_loss": None,
                "error": str(exc),
                "oom": True,
            }

        print(f'[ERROR] {args["condition"]} seed_{args["seed_idx"]:02d}: {exc}', flush=True)
        traceback.print_exc()
        return {
            "condition": args["condition"],
            "seed": args["seed_idx"],
            "hd_mode": args["hd_mode"],
            "final_loss": None,
            "error": str(exc),
            "oom": False,
        }


def _run_seed_worker_inner(args: dict[str, Any]):
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
    os.environ["PRNN_DEFER_SRSA"] = "1"

    train_mod = importlib.import_module("project5_symmetry.training.train")
    dataset_mod = importlib.import_module("project5_symmetry.training.dataset")

    condition = args["condition"]
    seed_idx = int(args["seed_idx"])
    hd_mode = args["hd_mode"]
    max_steps = int(args["max_steps"])
    checkpoint_steps = [int(x) for x in args["checkpoint_steps"] if int(x) <= max_steps]
    output_dir = Path(args["output_dir"])
    data_dir = Path(args["data_dir"])
    obs_size = int(args["obs_size"])
    run_started = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_mod._enable_tf32(device)

    print(
        f"[Worker] {condition} seed_{seed_idx:02d} hd={hd_mode} on GPU {args['gpu_id']}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"  VRAM at start: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed_idx)
    np.random.seed(seed_idx)

    batch_size = int(os.getenv("PRNN_PER_SEED_BATCH", getattr(train_mod, "FAST_BATCH_SIZE", 16)))
    k = int(args.get("k", 5))
    trunc = int(args.get("trunc", 200))
    anchor_subsample_n = int(os.getenv("PRNN_ANCHOR_SUBSAMPLE", getattr(train_mod, "ANCHOR_SUBSAMPLE_N", 32)))

    eval_dataset = dataset_mod.TrajectoryDataset(str(data_dir))
    packed = dataset_mod.PackedTrajectoryStore(
        str(data_dir),
        device=device,
        act_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model, compiled = train_mod._build_model(
        obs_size=obs_size,
        act_size=packed.act_size,
        k=k,
        trunc=trunc,
        device=device,
        compile_cell=True,
    )
    optimizer = train_mod._build_optimizer(model, batch_size=batch_size)
    latest_step = _load_checkpoint_if_present(output_dir, checkpoint_steps, model, optimizer, device)

    tb_dir = output_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir), comment=f"_{condition}_seed{seed_idx}_{hd_mode}")
    log_dict = train_mod._init_live_metric_log_dict()
    log_dict.update(
        {
            "hd_mode": hd_mode,
            "condition": condition,
            "seed": seed_idx,
            "srsa_log_every": SRSA_LOG_EVERY,
            "geometry_log_every": GEOMETRY_LOG_EVERY,
            "ckpt_paths": [],
            "H_paths": [],
            "runner": "runpod_runner",
            "compiled": compiled,
        }
    )
    train_mod._log_condition_scalar(writer, log_dict)

    T_act = packed.act_seq_len
    T_k = T_act - k
    initial_loss = None
    current_loss = float("nan")
    last_hidden_std = float("nan")
    verified = False
    stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    metric_n = int(getattr(train_mod, "SUBSAMPLE_N", 5000))
    metric_repeats = int(getattr(train_mod, "SRSA_EVAL_RUNS", 3))

    pbar = tqdm(
        total=max_steps - latest_step,
        desc=f"{condition} seed{seed_idx} {hd_mode}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )

    def train_step(step: int):
        nonlocal initial_loss, current_loss, last_hidden_std, verified

        obs_b, act_b = packed.sample_batch(batch_size)
        act_b = apply_hd_mode(act_b, hd_mode)
        if not verified:
            verify_hd_substitution(act_b, hd_mode)
            verified = True

        anchor_idx = train_mod._sample_anchor_idx(T_k, device, anchor_subsample_n)
        n_anchors = T_k if anchor_idx is None else int(anchor_idx.numel())
        pred, hidden_state, target = model(
            obs_b,
            act_b,
            anchor_idx=anchor_idx,
            noise_main=train_mod._sample_main_noise(batch_size, T_act, train_mod.HIDDEN_SIZE, device),
            noise_roll=train_mod._sample_roll_noise(batch_size, k, n_anchors, train_mod.HIDDEN_SIZE, device),
        )

        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = float(loss.item())
        last_hidden_std = float(hidden_state.std().item())
        if initial_loss is None:
            initial_loss = current_loss
        return n_anchors

    model.train()
    for step in range(latest_step + 1, max_steps + 1):
        if stream is None:
            n_anchors = train_step(step)
        else:
            with torch.cuda.stream(stream):
                n_anchors = train_step(step)

        writer.add_scalar("loss/train", current_loss, step)
        pbar.update(1)
        if step % 100 == 0 or step == latest_step + 1:
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "anchors": n_anchors, "B": batch_size})

        if step == latest_step + 500 and initial_loss is not None:
            loss_delta = current_loss - initial_loss
            if abs(loss_delta) < 0.001:
                raise RuntimeError(
                    f"FLAT LOSS at step {step}: loss={current_loss:.6f}, delta={loss_delta:.6f}"
                )
            if last_hidden_std < 0.01:
                raise RuntimeError(f"HIDDEN STATE COLLAPSED at step {step}")

        if step % SRSA_LOG_EVERY == 0:
            need_geometry = step % GEOMETRY_LOG_EVERY == 0
            srsa_e, srsa_c, geometry_metrics = evaluate_srsa_and_geometry_hd(
                model,
                eval_dataset,
                device,
                hd_mode=hd_mode,
                need_geometry=need_geometry,
                n=metric_n,
                repeats=metric_repeats,
            )
            train_mod._write_live_metrics(
                writer,
                step,
                srsa_e,
                srsa_c,
                geometry_metrics=geometry_metrics,
            )
            train_mod._append_live_metrics(
                log_dict,
                step,
                current_loss,
                srsa_e,
                srsa_c,
                geometry_metrics=geometry_metrics,
            )
            if geometry_metrics is not None:
                train_mod._print_live_geometry_metrics(
                    f"{condition} seed{seed_idx} {hd_mode}",
                    step,
                    geometry_metrics,
                )
            print(
                f"  [Metrics step {step}] loss={current_loss:.4f} "
                f"srsa_euclid={srsa_e:.4f} srsa_city={srsa_c:.4f}",
                flush=True,
            )

        if step in checkpoint_steps:
            ckpt_path = output_dir / f"ckpt_{step}.pt"
            train_mod._save_checkpoint(
                str(ckpt_path),
                step,
                model,
                optimizer,
                trunc,
                extra_meta={
                    "trainer_mode": "runpod-runner",
                    "batch_size": batch_size,
                    "hd_mode": hd_mode,
                    "condition": condition,
                    "seed": seed_idx,
                    "loss": current_loss,
                },
            )
            log_dict["ckpt_paths"].append(str(ckpt_path))
            writer.add_text("checkpoint", str(ckpt_path), step)

            H = extract_H_matrix_compatible(model, eval_dataset, device, condition, hd_mode)
            h_path = output_dir / f"H_{step}.npy"
            np.save(h_path, H)
            log_dict["H_paths"].append(str(h_path))

            if device.type == "cuda":
                vram_gb = torch.cuda.memory_allocated() / 1e9
                print(f"  [Step {step}] loss={current_loss:.4f} VRAM={vram_gb:.2f}GB", flush=True)
                if torch.cuda.memory_allocated() > 5.5e9:
                    print(f"WARNING: VRAM {vram_gb:.2f}GB exceeding threshold at step {step}", flush=True)
                    torch.save(
                        {"step": step, "model_state_dict": model.state_dict()},
                        output_dir / f"emergency_ckpt_{step}.pt",
                    )
            else:
                print(f"  [Step {step}] loss={current_loss:.4f}", flush=True)

    final_ckpt = output_dir / "ckpt_final.pt"
    train_mod._save_checkpoint(
        str(final_ckpt),
        max_steps,
        model,
        optimizer,
        trunc,
        extra_meta={
            "trainer_mode": "runpod-runner",
            "batch_size": batch_size,
            "hd_mode": hd_mode,
            "condition": condition,
            "seed": seed_idx,
            "loss": current_loss,
        },
    )
    log_dict["ckpt_paths"].append(str(final_ckpt))

    final_H = extract_H_matrix_compatible(model, eval_dataset, device, condition, hd_mode)
    final_h_path = output_dir / f"H_{max_steps}.npy"
    np.save(final_h_path, final_H)
    log_dict["H_paths"].append(str(final_h_path))

    if max_steps % SRSA_LOG_EVERY != 0:
        srsa_e, srsa_c, geometry_metrics = evaluate_srsa_and_geometry_hd(
            model,
            eval_dataset,
            device,
            hd_mode=hd_mode,
            need_geometry=(max_steps % GEOMETRY_LOG_EVERY == 0),
            n=metric_n,
            repeats=metric_repeats,
        )
        train_mod._write_live_metrics(
            writer,
            max_steps,
            srsa_e,
            srsa_c,
            geometry_metrics=geometry_metrics,
        )
        train_mod._append_live_metrics(
            log_dict,
            max_steps,
            current_loss,
            srsa_e,
            srsa_c,
            geometry_metrics=geometry_metrics,
        )
    _save_json(output_dir / "training_log.json", log_dict)
    writer.close()
    pbar.close()

    elapsed_s = time.time() - run_started
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    del model, optimizer, packed, eval_dataset
    gc.collect()

    print(
        f"[DONE] {condition} seed_{seed_idx:02d} hd={hd_mode} final loss={current_loss:.4f}",
        flush=True,
    )
    return {
        "condition": condition,
        "seed": seed_idx,
        "hd_mode": hd_mode,
        "final_loss": current_loss,
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed_s,
        "error": None,
        "oom": False,
    }


def _build_queue(dataset_workers: int) -> list[dict[str, Any]]:
    new_seeds = [
        ("s1", 3, "full", 80000),
        ("s1", 4, "full", 80000),
        ("s2", 3, "full", 80000),
        ("s2", 4, "full", 80000),
        ("s4", 5, "full", 80000),
        ("s4", 6, "full", 80000),
        ("s4", 7, "full", 80000),
    ]
    ablation_seeds = [
        ("s4", 0, "full", 40000),
        ("s4", 1, "full", 40000),
        ("s4", 0, "ablated", 40000),
        ("s4", 1, "ablated", 40000),
        ("s4", 0, "degraded", 40000),
        ("s4", 1, "degraded", 40000),
    ]

    data_cache: dict[str, tuple[Path, int]] = {}
    queue = []
    for cond, seed_idx, hd_mode, max_steps in new_seeds:
        if check_existing(cond, seed_idx, None):
            print(f"  SKIP (exists): {cond} seed_{seed_idx:02d} hd={hd_mode}")
            continue
        if cond not in data_cache:
            data_cache[cond] = _ensure_condition_data(cond, dataset_workers=dataset_workers)
        data_dir, obs_size = data_cache[cond]
        queue.append(
            {
                "queue": "symmetry_sweep",
                "condition": cond,
                "seed_idx": seed_idx,
                "hd_mode": hd_mode,
                "max_steps": max_steps,
                "checkpoint_steps": SYMMETRY_CHECKPOINTS,
                "output_dir": str(_symmetry_run_dir(cond, seed_idx)),
                "data_dir": str(data_dir),
                "obs_size": obs_size,
                "gpu_id": 0,
                "k": 5,
                "trunc": 200,
            }
        )

    for cond, seed_idx, hd_mode, max_steps in ablation_seeds:
        if check_existing(cond, seed_idx, hd_mode):
            print(f"  SKIP (exists): {cond} seed_{seed_idx:02d} hd={hd_mode}")
            continue
        if cond not in data_cache:
            data_cache[cond] = _ensure_condition_data(cond, dataset_workers=dataset_workers)
        data_dir, obs_size = data_cache[cond]
        queue.append(
            {
                "queue": "ablation",
                "condition": cond,
                "seed_idx": seed_idx,
                "hd_mode": hd_mode,
                "max_steps": max_steps,
                "checkpoint_steps": ABLATION_CHECKPOINTS,
                "output_dir": str(_ablation_run_dir(hd_mode, seed_idx)),
                "data_dir": str(data_dir),
                "obs_size": obs_size,
                "gpu_id": 0,
                "k": 5,
                "trunc": 200,
            }
        )
    return queue


def _run_queue_with_retries(queue: list[dict[str, Any]], parallelism: int):
    ctx = multiprocessing.get_context("spawn")
    pending = list(queue)
    results = []
    current_parallelism = max(1, parallelism)
    oom_reductions = 0

    while pending:
        print(f"\nLaunching {len(pending)} runs with parallelism={current_parallelism}")
        batch_results = []
        with ctx.Pool(processes=current_parallelism) as pool:
            for result in pool.imap_unordered(run_seed_worker, pending):
                batch_results.append(result)
                if result is None:
                    print("Completed: worker returned None")
                elif result.get("error"):
                    print(
                        f'Completed with error: {result["condition"]} seed_{result["seed"]:02d} '
                        f'hd={result["hd_mode"]} error={result["error"]}'
                    )
                else:
                    print(
                        f'Completed: {result["condition"]} seed_{result["seed"]:02d} '
                        f'hd={result["hd_mode"]} loss={result["final_loss"]:.4f}'
                    )

        oom_count = sum(1 for r in batch_results if r and r.get("oom"))
        results.extend(batch_results)

        if oom_count > 2 and current_parallelism > 1:
            current_parallelism -= 1
            oom_reductions += 1
            failed_keys = {
                (r["condition"], r["seed"], r["hd_mode"])
                for r in batch_results
                if r and r.get("oom")
            }
            pending = [
                run
                for run in pending
                if (run["condition"], run["seed_idx"], run["hd_mode"]) in failed_keys
            ]
            print(
                f"More than 2 OOMs detected; reducing parallelism to {current_parallelism} "
                f"and retrying {len(pending)} OOM runs."
            )
        else:
            pending = []

    return results, current_parallelism, oom_reductions


def main():
    start_time = time.time()
    print("=" * 60)
    print("RUNPOD AUTO-RUNNER - Reading hardware and building queue")
    print("=" * 60)
    print(f"Launch dir: {PATHS.launch_dir}")
    print(f"Repo root: {PATHS.repo_root}")
    print(f"Package dir: {PATHS.package_dir}")
    print(f"Results root: {PATHS.results_root}")

    profile = detect_hardware()
    optimal_parallel = int(profile["optimal_parallel"])
    dataset_workers = max(1, min(12, int(profile["cpu_count"]) // 2))

    queue = _build_queue(dataset_workers=dataset_workers)
    print(f"\nQueue built: {len(queue)} runs to execute")
    print(f"Running {optimal_parallel} seeds in parallel")

    total_steps = sum(r["max_steps"] for r in queue)
    if queue:
        steps_per_hour = 40000 / 0.6
        est_hours = total_steps / steps_per_hour / max(1, optimal_parallel)
    else:
        est_hours = 0.0
    estimated_cost = est_hours * RUNPOD_A5000_USD_PER_HOUR
    print(f"Estimated wall time: {est_hours:.1f} hours")
    print(f"Estimated cost: ${estimated_cost:.2f}")

    if not queue:
        results = []
        final_parallelism = optimal_parallel
        oom_reductions = 0
    else:
        results, final_parallelism, oom_reductions = _run_queue_with_retries(queue, optimal_parallel)

    elapsed_hours = (time.time() - start_time) / 3600.0
    successful = [r for r in results if r and not r.get("error") and r.get("final_loss") is not None]
    errors = [r for r in results if r and r.get("error")]

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(queue)}")
    for r in sorted(successful, key=lambda x: (x["condition"], x["hd_mode"], x["seed"])):
        print(f'  {r["condition"]} seed_{r["seed"]:02d} hd={r["hd_mode"]} loss={r["final_loss"]:.4f}')

    if errors:
        print("\nErrors:")
        for r in errors:
            print(f'  {r["condition"]} seed_{r["seed"]:02d} hd={r["hd_mode"]}: {r["error"]}')

    posthoc_manifest = {}
    try:
        posthoc_manifest = run_posthoc_outputs()
    except Exception as exc:
        posthoc_manifest = {"error": str(exc)}
        print(f"WARNING: post-hoc analysis phase failed: {exc}")
        traceback.print_exc()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware_profile": profile,
        "parallelism_selected_initial": optimal_parallel,
        "parallelism_selected_final": final_parallelism,
        "oom_reductions": oom_reductions,
        "paths": {
            "launch_dir": PATHS.launch_dir,
            "repo_root": PATHS.repo_root,
            "package_dir": PATHS.package_dir,
            "results_root": PATHS.results_root,
        },
        "all_runs_attempted": queue,
        "all_runs_completed": successful,
        "errors_encountered": errors,
        "wall_clock_time_per_seed": {
            f'{r["condition"]}/{r["hd_mode"]}/seed_{r["seed"]:02d}': r.get("elapsed_seconds")
            for r in results
            if r
        },
        "elapsed_hours": elapsed_hours,
        "estimated_hours": est_hours,
        "estimated_cost_usd": estimated_cost,
        "actual_cost_usd_at_0_27_per_hour": elapsed_hours * RUNPOD_A5000_USD_PER_HOUR,
        "posthoc_outputs": posthoc_manifest,
    }
    summary_path = PATHS.results_root / "runpod_run_summary.json"
    _save_json(summary_path, summary)
    print(f"\nWrote summary: {summary_path}")

    print("\nNEXT STEPS:")
    print("1. Download results/ directory to local machine")
    print("2. Run post-hoc metrics (PAA, RA, SCI) locally")
    print("3. Terminate RunPod pod to stop billing")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
