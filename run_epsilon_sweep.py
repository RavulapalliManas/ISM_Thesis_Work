#!/usr/bin/env python3
"""
Epsilon Sweep Experiment: Observation corruption phase transition analysis.

This script runs the epsilon sweep experiment and performs phase transition analysis.
It can be run standalone or called from runpod_runner.py as Queue 3.
"""

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


EPSILON_CHECKPOINTS = [5000, 10000, 20000, 40000, 60000, 80000]
EPSILON_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 3
EPSILON_BASELINE_S4_SRSA = 0.707
EPSILON_VALIDATION_THRESHOLD = 0.05
SRSA_LOG_EVERY = 1000
GEOMETRY_LOG_EVERY = 5000
EVS_THRESHOLD = 0.10


@dataclass(frozen=True)
class CodebasePaths:
    launch_dir: Path
    repo_root: Path
    package_dir: Path
    results_root: Path


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

    if (direct_results / "epsilon_sweep").exists():
        results_root = direct_results
    elif (package_results / "epsilon_sweep").exists():
        results_root = package_results
    elif (repo_results / "epsilon_sweep").exists():
        results_root = repo_results
    else:
        results_root = direct_results

    return CodebasePaths(
        launch_dir=launch_dir,
        repo_root=repo_root.resolve(),
        package_dir=package_dir.resolve(),
        results_root=results_root.resolve(),
    )


PATHS = find_codebase_paths()


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


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


def _epsilon_run_dir(epsilon: float, seed: int) -> Path:
    return PATHS.results_root / "epsilon_sweep" / f"eps_{epsilon}" / f"seed_{seed:02d}"


def check_existing(epsilon: float, seed: int) -> bool:
    """Return True if ckpt_final.pt exists for this run."""
    path = _epsilon_run_dir(epsilon, seed) / "ckpt_final.pt"
    return path.exists()


def _ensure_s4_data(dataset_workers: int) -> tuple[Path, int]:
    """Get S4 trajectory data for epsilon sweep."""
    try:
        sweep = importlib.import_module("run_symmetry_sweep")
        old_runs_dir = getattr(sweep, "RUNS_DIR", None)
        if old_runs_dir is not None:
            sweep.RUNS_DIR = PATHS.results_root / "symmetry_sweep"
        data_dir = sweep._ensure_condition_data(
            "s4",
            n_traj=sweep.P0_CFG.n_traj,
            runs_root=PATHS.results_root / "symmetry_sweep",
            dataset_workers=dataset_workers,
        )
        obs_size = sweep.P0_CFG.F * sweep.P0_CFG.F * 3
        return Path(data_dir), obs_size
    except Exception as exc:
        print(f"WARNING: run_symmetry_sweep._ensure_condition_data failed: {exc}")
        data_dir = PATHS.results_root / "symmetry_sweep" / "s4" / "trajectories"
        return data_dir, 7 * 7 * 3


def corrupt_observation(obs, epsilon: float, rng):
    """Apply epsilon corruption to observations."""
    import torch
    
    if epsilon == 0.0:
        return obs
    
    obs_min = obs.min().item()
    obs_max = obs.max().item()
    
    obs_random = torch.from_numpy(
        rng.uniform(low=obs_min, high=obs_max, size=obs.shape)
    ).to(obs.device, obs.dtype)
    
    return (1 - epsilon) * obs + epsilon * obs_random


def apply_hd_mode(d_t, hd_mode: str):
    """Apply HD substitution (pass-through for full mode)."""
    if hd_mode == "full":
        return d_t
    return d_t


def verify_hd_substitution(d_t_mod, hd_mode: str):
    """Verify HD substitution (no-op for full mode)."""
    pass


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


def _collect_hidden_states(model, dataset, n: int, device):
    import torch

    was_training = model.training
    model.eval()
    all_h, all_pos = [], []
    total = 0

    with torch.no_grad():
        for idx in torch.randperm(len(dataset)):
            obs, act, pos, _heading = dataset[int(idx)]
            _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
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
                f'[OOM] eps_{args["epsilon"]} seed_{args["seed_idx"]:02d} '
                "reduce parallel count and retry",
                flush=True,
            )
            return {
                "condition": args["condition"],
                "seed": args["seed_idx"],
                "epsilon": args["epsilon"],
                "hd_mode": args["hd_mode"],
                "final_loss": None,
                "error": str(exc),
                "oom": True,
            }

        print(f'[ERROR] eps_{args["epsilon"]} seed_{args["seed_idx"]:02d}: {exc}', flush=True)
        traceback.print_exc()
        return {
            "condition": args["condition"],
            "seed": args["seed_idx"],
            "epsilon": args["epsilon"],
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
    metrics_mod = importlib.import_module("project5_symmetry.evaluation.metrics")

    condition = args["condition"]
    seed_idx = int(args["seed_idx"])
    hd_mode = args["hd_mode"]
    epsilon = float(args["epsilon"])
    max_steps = int(args["max_steps"])
    checkpoint_steps = [int(x) for x in args["checkpoint_steps"] if int(x) <= max_steps]
    output_dir = Path(args["output_dir"])
    data_dir = Path(args["data_dir"])
    obs_size = int(args["obs_size"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_mod._enable_tf32(device)

    print(
        f"[Worker] eps_{epsilon} seed_{seed_idx:02d} hd={hd_mode} on GPU {args['gpu_id']}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"  VRAM at start: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed_idx)
    np.random.seed(seed_idx)
    corruption_rng = np.random.RandomState(seed_idx)

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
    writer = SummaryWriter(log_dir=str(tb_dir), comment=f"_eps{epsilon}_seed{seed_idx}_{hd_mode}")
    log_dict = train_mod._init_live_metric_log_dict()
    log_dict.update(
        {
            "hd_mode": hd_mode,
            "epsilon": epsilon,
            "condition": condition,
            "seed": seed_idx,
            "srsa_log_every": SRSA_LOG_EVERY,
            "geometry_log_every": GEOMETRY_LOG_EVERY,
            "ckpt_paths": [],
            "H_paths": [],
            "runner": "epsilon_sweep",
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
        desc=f"eps{epsilon} seed{seed_idx} {hd_mode}",
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )

    def train_step(step: int):
        nonlocal initial_loss, current_loss, last_hidden_std, verified

        obs_b, act_b = packed.sample_batch(batch_size)
        obs_b = corrupt_observation(obs_b, epsilon, corruption_rng)
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

        do_checkpoint = step in checkpoint_steps
        do_srsa = step % SRSA_LOG_EVERY == 0 and not defer_srsa
        do_geometry = step % GEOMETRY_LOG_EVERY == 0

        if do_checkpoint or do_srsa or do_geometry:
            model.eval()
            with torch.no_grad():
                hidden, positions = _collect_hidden_states(model, eval_dataset, metric_n, device)

            if do_checkpoint:
                ckpt_path = output_dir / f"ckpt_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": current_loss,
                    },
                    ckpt_path,
                )
                log_dict["ckpt_paths"].append(str(ckpt_path))
                H_path = output_dir / f"H_{step}.npy"
                np.save(H_path, hidden)
                log_dict["H_paths"].append(str(H_path))

            if do_srsa:
                srsa_val = float(metrics_mod.srsa(hidden, positions, space_metric="euclidean", max_n=metric_n))
                log_dict["srsa_euclid"].append(srsa_val)
                log_dict["steps"].append(step)
                writer.add_scalar("srsa/euclidean", srsa_val, step)
                srsa_city = float(metrics_mod.srsa(hidden, positions, space_metric="cityblock", max_n=metric_n))
                log_dict["srsa_city"].append(srsa_city)
                writer.add_scalar("srsa/cityblock", srsa_city, step)

            if do_geometry:
                aggregated = metrics_mod.aggregate_hidden_by_position(hidden, positions)
                position_hidden = aggregated["hidden"]
                rgc = metrics_mod.representational_geometry_consistency(position_hidden)
                coherence = metrics_mod.place_field_spatial_coherence(hidden, positions, arena_size=18)
                log_dict["manifold_id"].append(float(metrics_mod.manifold_id(position_hidden)))
                log_dict["mean_field_coherence"].append(float(coherence["mean_score"]))
                log_dict["mds_stress"].append(float(rgc["stress"]))
                log_dict["pca_variance_2d"].append(float(rgc["pca_var_2d"]))
                writer.add_scalar("geometry/manifold_id", log_dict["manifold_id"][-1], step)
                writer.add_scalar("geometry/coherence", log_dict["mean_field_coherence"][-1], step)
                writer.add_scalar("geometry/mds_stress", log_dict["mds_stress"][-1], step)

            model.train()

        if step % 1000 == 0:
            _save_json(output_dir / "training_log.json", log_dict)

    model.eval()
    with torch.no_grad():
        hidden, positions = _collect_hidden_states(model, eval_dataset, metric_n, device)
    final_srsa = float(metrics_mod.srsa(hidden, positions, space_metric="euclidean", max_n=metric_n))

    torch.save(
        {
            "step": max_steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": current_loss,
            "final_srsa": final_srsa,
        },
        output_dir / "ckpt_final.pt",
    )

    _save_json(output_dir / "training_log.json", log_dict)
    writer.close()

    run_duration = time.time() - run_started
    print(
        f"[Worker] Completed eps_{epsilon} seed_{seed_idx:02d} "
        f"final_srsa={final_srsa:.4f} in {run_duration/60:.1f}min",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    del model, optimizer, packed
    gc.collect()

    return {
        "condition": condition,
        "seed": seed_idx,
        "epsilon": epsilon,
        "hd_mode": hd_mode,
        "final_loss": current_loss,
        "final_srsa": final_srsa,
        "duration": run_duration,
        "output_dir": str(output_dir),
    }


def build_queue(dataset_workers: int = 1) -> list[dict[str, Any]]:
    """Build the epsilon sweep queue."""
    epsilon_seeds = [
        (0.0, 0, 80000),
        (0.0, 1, 80000),
        (0.0, 2, 80000),
        (0.1, 0, 80000),
        (0.1, 1, 80000),
        (0.1, 2, 80000),
        (0.2, 0, 80000),
        (0.2, 1, 80000),
        (0.2, 2, 80000),
        (0.3, 0, 80000),
        (0.3, 1, 80000),
        (0.3, 2, 80000),
        (0.5, 0, 80000),
        (0.5, 1, 80000),
        (0.5, 2, 80000),
        (0.7, 0, 80000),
        (0.7, 1, 80000),
        (0.7, 2, 80000),
        (1.0, 0, 80000),
        (1.0, 1, 80000),
        (1.0, 2, 80000),
    ]

    s4_data_dir, s4_obs_size = _ensure_s4_data(dataset_workers)

    queue = []
    for epsilon, seed_idx, max_steps in epsilon_seeds:
        if check_existing(epsilon, seed_idx):
            print(f"  SKIP (exists): eps_{epsilon} seed_{seed_idx:02d}")
            continue
        queue.append(
            {
                "queue": "epsilon_sweep",
                "condition": "s4",
                "seed_idx": seed_idx,
                "hd_mode": "full",
                "epsilon": epsilon,
                "max_steps": max_steps,
                "checkpoint_steps": EPSILON_CHECKPOINTS,
                "output_dir": str(_epsilon_run_dir(epsilon, seed_idx)),
                "data_dir": str(s4_data_dir),
                "obs_size": s4_obs_size,
                "gpu_id": 0,
                "k": 5,
                "trunc": 200,
            }
        )
    return queue


def validate_epsilon_baseline(epsilon_0_results: list[dict]) -> bool:
    """Validate epsilon=0.0 baseline against existing S4 results."""
    if not epsilon_0_results:
        print("WARNING: No epsilon=0.0 results to validate")
        return False
    
    mean_srsa = np.mean([r["final_srsa"] for r in epsilon_0_results])
    delta = abs(mean_srsa - EPSILON_BASELINE_S4_SRSA)
    
    print(f"Epsilon=0.0 validation:")
    print(f"  Mean sRSA: {mean_srsa:.4f}")
    print(f"  Expected S4 sRSA: {EPSILON_BASELINE_S4_SRSA:.4f}")
    print(f"  Delta: {delta:.4f}")
    
    if delta > EPSILON_VALIDATION_THRESHOLD:
        print(f"ERROR: epsilon=0.0 baseline deviates by {delta:.4f} > {EPSILON_VALIDATION_THRESHOLD}")
        print("  Check corruption implementation!")
        return False
    else:
        print(f"OK: epsilon=0.0 baseline matches existing S4 results")
        return True


def estimate_epsilon_c(
    epsilon_levels: list[float],
    mean_srsa_per_level: list[float],
    fraction_tuned_per_level: list[float] | None = None,
) -> dict:
    """Estimate critical epsilon using three methods."""
    
    # Method 1: sRSA threshold crossing (sRSA < 0.40)
    gate = 0.40
    below_gate = [(e, s) for e, s in zip(epsilon_levels, mean_srsa_per_level) if s < gate]
    epsilon_c_threshold = below_gate[0][0] if below_gate else None
    
    # Method 2: Maximum gradient (steepest drop)
    gradients = np.diff(mean_srsa_per_level) / np.diff(epsilon_levels)
    # argmin because gradients are negative (sRSA drops as epsilon rises)
    grad_idx = np.argmin(gradients)
    epsilon_c_gradient = epsilon_levels[grad_idx]
    
    result = {
        "epsilon_c_threshold": epsilon_c_threshold,
        "epsilon_c_gradient": epsilon_c_gradient,
    }
    
    # Method 3: Fraction tuned threshold (< 0.01)
    if fraction_tuned_per_level is not None:
        below_tuned = [(e, f) for e, f in zip(epsilon_levels, fraction_tuned_per_level) if f < 0.01]
        epsilon_c_tuned = below_tuned[0][0] if below_tuned else None
        result["epsilon_c_fraction_tuned"] = epsilon_c_tuned
    
    # Check agreement
    vals = [v for v in [epsilon_c_threshold, epsilon_c_gradient] if v is not None]
    if len(vals) >= 2:
        result["agreement"] = abs(max(vals) - min(vals)) < 0.1
    else:
        result["agreement"] = None
    
    return result


def run_phase_transition_analysis():
    """Analyze phase transition and generate results."""
    print("\n" + "=" * 60)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 60)
    
    results_by_epsilon = {eps: [] for eps in EPSILON_LEVELS}
    
    for epsilon in EPSILON_LEVELS:
        for seed in range(N_SEEDS):
            run_dir = _epsilon_run_dir(epsilon, seed)
            log_path = run_dir / "training_log.json"
            if log_path.exists():
                log = _load_json(log_path)
                final_srsa = log.get("srsa_euclid", [float("nan")])[-1]
                results_by_epsilon[epsilon].append({
                    "seed": seed,
                    "final_srsa": final_srsa,
                    "log": log,
                })
    
    # Compute statistics per epsilon level
    epsilon_stats = []
    for eps in EPSILON_LEVELS:
        results = results_by_epsilon[eps]
        if results:
            srsa_values = [r["final_srsa"] for r in results if np.isfinite(r["final_srsa"])]
            mean_srsa = float(np.mean(srsa_values)) if srsa_values else float("nan")
            std_srsa = float(np.std(srsa_values)) if len(srsa_values) > 1 else 0.0
        else:
            mean_srsa = float("nan")
            std_srsa = float("nan")
        epsilon_stats.append({
            "epsilon": eps,
            "mean_srsa": mean_srsa,
            "std_srsa": std_srsa,
            "n_seeds": len(results),
        })
    
    mean_srsa_per_level = [s["mean_srsa"] for s in epsilon_stats]
    
    # Validate epsilon=0.0 baseline
    epsilon_0_results = results_by_epsilon.get(0.0, [])
    baseline_valid = validate_epsilon_baseline(epsilon_0_results)
    
    # Estimate epsilon_c
    epsilon_c = estimate_epsilon_c(EPSILON_LEVELS, mean_srsa_per_level)
    
    print("\nPhase Transition Results:")
    print(f"  Method 1 (sRSA < 0.40 threshold): epsilon_c = {epsilon_c.get('epsilon_c_threshold')}")
    print(f"  Method 2 (max gradient):          epsilon_c = {epsilon_c.get('epsilon_c_gradient')}")
    if epsilon_c.get('epsilon_c_fraction_tuned'):
        print(f"  Method 3 (fraction_tuned < 0.01): epsilon_c = {epsilon_c['epsilon_c_fraction_tuned']}")
    print(f"  Agreement: {epsilon_c.get('agreement')}")
    
    # Save results
    analysis_dir = PATHS.results_root / "epsilon_sweep"
    _save_json(analysis_dir / "phase_transition_analysis.json", {
        "epsilon_stats": epsilon_stats,
        "epsilon_c": epsilon_c,
        "baseline_valid": baseline_valid,
    })
    
    return epsilon_stats, epsilon_c


def generate_figures():
    """Generate figures for epsilon sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figures")
        return
    
    analysis_dir = PATHS.results_root / "epsilon_sweep"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load analysis results
    analysis_path = analysis_dir / "phase_transition_analysis.json"
    if not analysis_path.exists():
        print("WARNING: No phase_transition_analysis.json found, skipping figures")
        return
    
    analysis = _load_json(analysis_path)
    epsilon_stats = analysis["epsilon_stats"]
    
    epsilons = [s["epsilon"] for s in epsilon_stats]
    mean_srsas = [s["mean_srsa"] for s in epsilon_stats]
    std_srsas = [s["std_srsa"] for s in epsilon_stats]
    
    # Figure A: Phase transition curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(epsilons, mean_srsas, yerr=std_srsas, fmt='o-', capsize=3, color='#2166ac')
    ax.axhline(y=0.40, color='gray', linestyle='--', label='sRSA=0.40 gate')
    if analysis["epsilon_c"].get("epsilon_c_threshold"):
        ax.axvline(x=analysis["epsilon_c"]["epsilon_c_threshold"], color='red', linestyle=':', 
                   label=f'ε_c={analysis["epsilon_c"]["epsilon_c_threshold"]}')
    ax.set_xlabel('Epsilon (observation corruption)')
    ax.set_ylabel('sRSA (Euclidean)')
    ax.set_title('Phase Transition: Map Formation vs Observation Corruption')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_epsilon_phase_transition.pdf", bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figures saved to {figures_dir}")


def run_epsilon_sweep(parallelism: int = 6):
    """Run the full epsilon sweep experiment."""
    print("\n" + "=" * 60)
    print("EPSILON SWEEP EXPERIMENT")
    print("=" * 60)
    
    queue = build_queue(dataset_workers=1)
    
    if not queue:
        print("No runs to execute - all seeds already exist")
        return
    
    print(f"\nQueue built: {len(queue)} runs to execute")
    print(f"Epsilon levels: {EPSILON_LEVELS}")
    print(f"Seeds per level: {N_SEEDS}")
    
    ctx = multiprocessing.get_context("spawn")
    pending = list(queue)
    results = []
    current_parallelism = max(1, parallelism)
    
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
                        f'Completed with error: eps_{result["epsilon"]} seed_{result["seed"]:02d} '
                        f'error={result["error"]}'
                    )
                else:
                    print(
                        f'Completed: eps_{result["epsilon"]} seed_{result["seed"]:02d} '
                        f'sRSA={result["final_srsa"]:.4f}'
                    )
        results.extend(batch_results)
        pending = []
    
    print(f"\nAll runs completed: {len(results)} total")
    
    # Run phase transition analysis
    epsilon_stats, epsilon_c = run_phase_transition_analysis()
    
    # Generate figures
    generate_figures()
    
    print("\n" + "=" * 60)
    print("EPSILON SWEEP COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {PATHS.results_root / 'epsilon_sweep'}")
    
    return results, epsilon_stats, epsilon_c


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run epsilon sweep experiment")
    parser.add_argument("--parallelism", type=int, default=6, help="Number of parallel workers")
    args = parser.parse_args()
    
    run_epsilon_sweep(parallelism=args.parallelism)