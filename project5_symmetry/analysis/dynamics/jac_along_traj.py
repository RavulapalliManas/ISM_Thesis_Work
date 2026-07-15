#!/usr/bin/env python3
"""Clamp-free hardening of Stage 0: local dynamics ALONG REAL TRAJECTORIES.

The input-clamped fixed-point picture flipped with the clamp because mean-obs and
zero-obs are both OFF the data manifold. This avoids the problem entirely: at each
state the network actually visits, with the real input it actually receives, take
the one-step Jacobian dF/dh and its leading |lambda|. Average over the real state
distribution. No fixed-point finder, no clamp, no tolerance.

Question: is the predictive net (k=1) more contractive along its own trajectories
than the reconstruction net (k=0)?  Reported per seed, so it is a real (multi-seed)
result, not n=1.

|lambda|_max < 1  => locally contracting;  > 1 => locally expanding.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/Volumes/Crucial X6/Thesis_work")
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa
from project5_symmetry.environments.hd_encodings import apply_hd            # noqa
from project5_symmetry.training.dataset import TrajectoryDataset            # noqa

CKROOT = "/Volumes/Crucial X6/prnn_backup/checkpoints/horizon"


@torch.no_grad()
def visited_states(model, dataset, hd_mode, n, device, seed=0):
    """Real (h_t, u_t) pairs: hidden state and the real input that drives it next."""
    H, U, tot = [], [], 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, _ = dataset[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0)                      # (T,500)
        T = h.shape[0]
        u = torch.cat([obs[:T], act[:T]], dim=1)   # (T,152) real input at each step
        H.append(h); U.append(u); tot += T
        if tot >= n:
            break
    return torch.cat(H)[:n], torch.cat(U)[:n]


def leading_abs_eig(cell, h, u):
    """|lambda|_max of dF/dh at (h,u), exact 500x500 autograd Jacobian."""
    Z = torch.zeros(1, h.shape[0])
    U = u.unsqueeze(0)
    def F(x):
        hn, _ = cell(U, Z, (x.unsqueeze(0), 0))
        return hn.squeeze(0)
    J = torch.autograd.functional.jacobian(F, h, vectorize=True)
    ev = np.linalg.eigvals(J.numpy())
    return np.abs(ev)


def analyse(k, seed, dataset, args, device):
    path = f"{CKROOT}/k{k}/s2/full/seed_{seed:02d}/ckpt_final.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(ck, device).requires_grad_(False)
    cell = model.rnn.cell
    H, U = visited_states(model, dataset, args.hd_mode, args.n_states, device, seed=args.seed)
    top = np.empty(H.shape[0]); n_marg = np.empty(H.shape[0])
    for i in range(H.shape[0]):
        a = np.sort(leading_abs_eig(cell, H[i], U[i]))[::-1]
        top[i] = a[0]
        n_marg[i] = ((a >= 0.95) & (a <= 1.05)).sum()
    frac_contract = float((top < 1.0).mean())
    print(f"  k={k} seed{seed:02d}: |lam|max  median={np.median(top):.3f} "
          f"mean={top.mean():.3f}  p90={np.percentile(top,90):.3f}  "
          f"frac(<1)={frac_contract:.2f}  #marg/state={n_marg.mean():.2f}")
    return dict(k=k, seed=seed, med=float(np.median(top)), mean=float(top.mean()),
                p90=float(np.percentile(top, 90)), frac_contract=frac_contract,
                n_marg=float(n_marg.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/Crucial X6/Thesis_work/project5_symmetry/_tmp_fund_domain_data/s2")
    ap.add_argument("--hd-mode", default="full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-states", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    dataset = TrajectoryDataset(a.data)
    print(f"dataset: {len(dataset)} trajectories | n_states/model={a.n_states} | seeds={a.seeds}")

    rows = []
    for s in a.seeds:
        print(f"\n--- seed {s:02d} ---")
        rows.append(analyse(0, s, dataset, a, device))
        rows.append(analyse(1, s, dataset, a, device))

    print("\n============ contraction along real trajectories: k0 vs k1 ============")
    for s in a.seeds:
        r0 = next(r for r in rows if r["k"] == 0 and r["seed"] == s)
        r1 = next(r for r in rows if r["k"] == 1 and r["seed"] == s)
        print(f"  seed{s:02d}:  median|lam|max  k0={r0['med']:.3f} -> k1={r1['med']:.3f} "
              f"(Delta={r1['med']-r0['med']:+.3f})   frac(<1)  k0={r0['frac_contract']:.2f} -> k1={r1['frac_contract']:.2f}")
    k0m = np.mean([r["med"] for r in rows if r["k"] == 0])
    k1m = np.mean([r["med"] for r in rows if r["k"] == 1])
    print(f"\n  mean median|lam|max over seeds:  k0={k0m:.3f}  k1={k1m:.3f}  Delta={k1m-k0m:+.3f}")


if __name__ == "__main__":
    main()
