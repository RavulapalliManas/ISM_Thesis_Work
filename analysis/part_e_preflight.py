import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from utils.Architectures import pRNN_th
from utils.thetaRNN import LayerNormRNNCell


SWEEP = BASE / "project5_symmetry" / "results" / "symmetry_sweep"
OUT = BASE / "results" / "ablation"


def apply_hd_mode(act, mode):
    out = act.clone()
    if mode == "ablated":
        out[..., 1:5] = 0.0
    elif mode == "degraded":
        out[..., 1:5] = 0.25
    elif mode == "full":
        pass
    else:
        raise ValueError(mode)
    return out


def nvidia_smi():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else res.stderr.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traj = np.load(SWEEP / "s4" / "trajectories" / "traj_00000.npz")
    obs = torch.from_numpy(traj["obs"]).unsqueeze(0).float()
    act = torch.from_numpy(traj["act_enc"]).unsqueeze(0).float()
    model = pRNN_th(obs_size=147, act_size=5, k=5, hidden_size=500, cell=LayerNormRNNCell, neuralTimescale=2, predOffset=0, hidden_init_sigma=0.1)
    with torch.no_grad():
        pred, h_t, target = model(obs, act)

    mode_checks = {}
    for mode in ["full", "ablated", "degraded"]:
        mod = apply_hd_mode(act, mode)
        mode_checks[mode] = {
            "hd_sum_first_step": float(mod[0, 0, 1:5].sum().item()),
            "hd_values_first_step": [float(x) for x in mod[0, 0, 1:5].tolist()],
            "speed_first_step": float(mod[0, 0, 0].item()),
            "total_hd_sum_sequence": float(mod[..., 1:5].sum().item()),
        }

    status = {
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "nvidia_smi": nvidia_smi(),
        "model_param_count": int(sum(p.numel() for p in model.parameters())),
        "forward_h_t_shape": list(h_t.shape),
        "forward_pred_shape": list(pred.shape),
        "forward_target_shape": list(target.shape),
        "input_encoding": "SpeedHD 5D [speed, hd_0, hd_1, hd_2, hd_3]",
        "hd_ablation_target": "act_enc[..., 1:5]",
        "separate_Wd_present": False,
        "mode_checks": mode_checks,
        "training_status": "not_started",
        "training_blocker": (
            "Current .brain PyTorch runtime is CPU-only. Full Part E requires long-running "
            "6-run training and should be launched only from a CUDA-enabled torch runtime."
        ),
    }
    with open(OUT / "part_e_preflight_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    print("Part E preflight")
    print(f"torch {status['torch_version']}, cuda_available={status['torch_cuda_available']}")
    print(f"nvidia-smi: {status['nvidia_smi']}")
    print(f"h_t.shape = {tuple(h_t.shape)}")
    print(f"param count = {status['model_param_count']}")
    for mode, check in mode_checks.items():
        print(f"{mode}: hd first-step sum={check['hd_sum_first_step']:.6f}, values={check['hd_values_first_step']}")
    print(f"{(OUT / 'part_e_preflight_status.json').relative_to(BASE)} {(OUT / 'part_e_preflight_status.json').stat().st_size} bytes")


if __name__ == "__main__":
    main()
