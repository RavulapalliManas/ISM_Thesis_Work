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
MAX_PARALLEL_CAP = 6
RUNPOD_A5000_USD_PER_HOUR = 0.27

SYMMETRY_CHECKPOINTS = [5000, 10000, 20000, 40000, 60000, 80000]
ABLATION_CHECKPOINTS = [10000, 20000, 30000, 40000]


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

    max_parallel_gpu = max(1, math.floor((available_vram_gb - VRAM_HEADROOM_GB) / PER_SEED_VRAM_GB))
    max_parallel_cpu = max(1, cpu_count // 2)
    optimal_parallel = max(1, min(max_parallel_gpu, max_parallel_cpu, MAX_PARALLEL_CAP))

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


def _save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


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

    log_dict["steps"].append(max_steps)
    log_dict["loss"].append(float(current_loss))
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
