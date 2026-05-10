import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from project5_symmetry.evaluation.metrics import aggregate_hidden_by_position, srsa
from project5_symmetry.training.dataset import PackedTrajectoryStore
from project5_symmetry.training.train import (
    ANCHOR_SUBSAMPLE_N,
    CHECKPOINT_STEPS,
    HIDDEN_INIT_SIGMA,
    HIDDEN_SIZE,
    PRED_OFFSET,
    _build_model,
    _build_optimizer,
    _sample_anchor_idx,
    _sample_main_noise,
    _sample_roll_noise,
)


DATA_DIR = BASE / "project5_symmetry" / "results" / "symmetry_sweep" / "s4" / "trajectories"
OUT_BASE = BASE / "results" / "ablation"
CKPT_STEPS = [10_000, 20_000, 30_000, 40_000]
WARN_MB = 5_000
STOP_MB = 5_500


def apply_hd_mode(act, hd_mode):
    """SpeedHD: preserve speed column 0, alter heading columns 1:5."""
    out = act.clone()
    if hd_mode == "full":
        return out
    if hd_mode == "ablated":
        out[..., 1:5] = 0.0
        return out
    if hd_mode == "degraded":
        out[..., 1:5] = 0.25
        return out
    raise ValueError(f"Unknown hd_mode={hd_mode}")


class JsonlDashboard:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "dashboard_events.jsonl"

    def log(self, event, payload):
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        try:
            from analysis.ablation_dashboard import tick

            tick(show_terminal=False)
        except Exception:
            # Training must never fail just because the dashboard renderer failed.
            pass

    def flag(self, message, payload=None):
        self.log("flag", {"message": message, **(payload or {})})


def gpu_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1e6, torch.cuda.memory_reserved() / 1e6


@torch.no_grad()
def extract_H_matrix(model, traj_paths, device, hd_mode, n_hidden=12000):
    model.eval()
    all_h = []
    all_pos = []
    total = 0
    order = np.random.default_rng(12345).permutation(len(traj_paths))
    for idx in order:
        with np.load(traj_paths[int(idx)]) as z:
            obs_np = z["obs"].astype(np.float32, copy=False)
            act_np = z["act_enc"].astype(np.float32, copy=False)
            pos_np = z["pos"].astype(np.float32, copy=False)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device, non_blocking=True)
        act = apply_hd_mode(torch.from_numpy(act_np).unsqueeze(0).to(device, non_blocking=True), hd_mode)
        _, h, _ = model(obs, act)
        h0 = h.squeeze(0).detach().cpu().numpy()
        p0 = pos_np[: h0.shape[0]]
        take = min(h0.shape[0], n_hidden - total)
        all_h.append(h0[:take])
        all_pos.append(p0[:take])
        total += take
        if total >= n_hidden:
            break
    hidden = np.concatenate(all_h, axis=0)
    positions = np.concatenate(all_pos, axis=0)
    agg = aggregate_hidden_by_position(hidden, positions)
    if agg["hidden"].shape[0] != 324:
        raise RuntimeError(f"H extraction covered {agg['hidden'].shape[0]} positions, expected 324")
    return agg["hidden"], agg["positions"], agg["counts"]


def train_one(hd_mode, seed, max_steps, checkpoint_steps):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Do not run Part E ablation on the CPU runtime.")

    out_dir = OUT_BASE / f"hd_{hd_mode}" / f"seed_{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dashboard = JsonlDashboard(out_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    dashboard.log(
        "run_start",
        {
            "hd_mode": hd_mode,
            "seed": seed,
            "max_steps": max_steps,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    )

    traj_paths = sorted(DATA_DIR.glob("traj_*.npz"))
    if not traj_paths:
        raise FileNotFoundError(f"No trajectories found in {DATA_DIR}")
    packed = PackedTrajectoryStore(str(DATA_DIR), device=device, act_dtype=torch.float16)
    model, compiled = _build_model(
        obs_size=packed.obs_size,
        act_size=packed.act_size,
        k=5,
        trunc=200,
        device=device,
        compile_cell=True,
    )
    optimizer = _build_optimizer(model, batch_size=8)

    T_act = packed.act_seq_len
    T_k = T_act - 5
    anchor_n = int(os.getenv("PRNN_ABLATION_ANCHORS", ANCHOR_SUBSAMPLE_N))
    # Existing symmetry_sweep checkpoints were trained with fast-parallel batch_size=32.
    batch_size = int(os.getenv("PRNN_ABLATION_BATCH", 32))

    obs0, act0 = packed.sample_batch(batch_size)
    act0_mod = apply_hd_mode(act0, hd_mode)
    anchor_idx0 = _sample_anchor_idx(T_k, device, anchor_n)
    pred0, h0, target0 = model(
        obs0,
        act0_mod,
        anchor_idx=anchor_idx0,
        noise_main=_sample_main_noise(batch_size, T_act, HIDDEN_SIZE, device),
        noise_roll=_sample_roll_noise(batch_size, 5, int(anchor_idx0.numel()), HIDDEN_SIZE, device),
    )
    loss0 = F.mse_loss(pred0, target0).item()
    alloc, reserved = gpu_mb()
    dashboard.log(
        "hd_check",
        {
            "d_t_dim": int(act0_mod.dim()),
            "d_t_shape": list(act0_mod.shape),
            "heading_sum_mean": float(act0_mod[..., 1:5].sum(-1).mean().item()),
            "heading_max": float(act0_mod[..., 1:5].max().item()),
            "speed_mean": float(act0_mod[..., 0].mean().item()),
            "h_shape": list(h0.shape),
            "h_std": float(h0.std().item()),
            "loss_step0": float(loss0),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "gpu_mb": alloc,
            "gpu_reserved_mb": reserved,
            "compiled_cell": bool(compiled),
        },
    )
    if hd_mode == "ablated" and abs(float(act0_mod[..., 1:5].sum().item())) > 1e-6:
        dashboard.flag("HD_ABLATED heading sum was not zero")
        raise RuntimeError("HD_ABLATED heading sum was not zero")

    step_loss = loss0
    for step in range(1, max_steps + 1):
        obs_b, act_b = packed.sample_batch(batch_size)
        act_b = apply_hd_mode(act_b, hd_mode)
        anchor_idx = _sample_anchor_idx(T_k, device, anchor_n)
        n_anchors = int(anchor_idx.numel())
        pred, h_t, target = model(
            obs_b,
            act_b,
            anchor_idx=anchor_idx,
            noise_main=_sample_main_noise(batch_size, T_act, HIDDEN_SIZE, device),
            noise_roll=_sample_roll_noise(batch_size, 5, n_anchors, HIDDEN_SIZE, device),
        )
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_loss = float(loss.item())

        if step == 500:
            h_std = float(h_t.std().item())
            delta = step_loss - loss0
            dashboard.log("training_check", {"step": step, "loss": step_loss, "loss_delta": delta, "h_std": h_std})
            if abs(delta) < 0.001:
                dashboard.flag("FLAT_LOSS - possible architecture bug", {"loss_delta": delta})
                raise RuntimeError("Flat loss at step 500")
            if h_std < 0.01:
                dashboard.flag("HIDDEN STATE COLLAPSED", {"h_std": h_std})
                raise RuntimeError("Hidden state collapsed")

        progress_interval = int(os.getenv("PRNN_ABLATION_PROGRESS_INTERVAL", 1000))
        if step % progress_interval == 0 or step == 1:
            alloc, reserved = gpu_mb()
            dashboard.log(
                "progress",
                {
                    "step": step,
                    "total": max_steps,
                    "loss": step_loss,
                    "gpu_mb": alloc,
                    "gpu_reserved_mb": reserved,
                },
            )
            if alloc > WARN_MB:
                dashboard.flag("GPU memory warning threshold exceeded", {"gpu_mb": alloc})
            if alloc > STOP_MB:
                dashboard.flag("GPU memory stop threshold exceeded", {"gpu_mb": alloc})
                raise RuntimeError("GPU memory exceeded stop threshold")

        if step in checkpoint_steps:
            ckpt_path = out_dir / f"ckpt_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hd_mode": hd_mode,
                    "loss": step_loss,
                    "encoding": "SpeedHD_5D",
                    "heading_dims": [1, 2, 3, 4],
                    "speed_dim": 0,
                },
                ckpt_path,
            )
            # Stream trajectory files from disk here instead of keeping a full CPU
            # evaluation dataset resident in every parallel training process.
            H, pos, counts = extract_H_matrix(model, traj_paths, device, hd_mode)
            np.save(out_dir / f"H_{step}.npy", H)
            np.save(out_dir / f"H_positions_{step}.npy", pos)
            np.save(out_dir / f"H_counts_{step}.npy", counts)
            spatial_srsa = float(srsa(H, pos, space_metric="euclidean", max_n=5000))
            dashboard.log(
                "checkpoint",
                {
                    "step": step,
                    "ckpt": str(ckpt_path.relative_to(BASE)),
                    "H": str((out_dir / f"H_{step}.npy").relative_to(BASE)),
                    "spatial_sRSA": spatial_srsa,
                    "loss": step_loss,
                },
            )
            model.train()

    dashboard.log("run_complete", {"hd_mode": hd_mode, "seed": seed, "loss": step_loss})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hd-mode", choices=["full", "ablated", "degraded"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=40_000)
    parser.add_argument("--checkpoint-steps", default="10000,20000,30000,40000")
    args = parser.parse_args()
    checkpoint_steps = [] if args.checkpoint_steps.lower() in {"none", "no", "off"} else [
        int(x) for x in args.checkpoint_steps.split(",") if x.strip()
    ]
    train_one(args.hd_mode, args.seed, args.max_steps, checkpoint_steps)


if __name__ == "__main__":
    main()
