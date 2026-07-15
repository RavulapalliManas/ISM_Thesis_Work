#!/usr/bin/env python3
"""Stage 0 go/no-go: does prediction (k=1) build slow/integrating modes that
reconstruction (k=0) lacks, in the C2/full pRNN?

Method (Sussillo & Barak 2013, adapted to this discrete-time LN+ReLU cell):
  single step  h' = F(h,u) = ReLU( LN(W_ih u + W_hh h) + b )
  fixed/slow points: minimise q(h) = 0.5 * ||F(h,u) - h||^2 from a pool of real
  hidden states, input u clamped (per heading).  At each converged point take the
  exact 500x500 autograd Jacobian dF/dh and its eigen-spectrum.

Discrete-time marginal-mode criterion: |lambda| ~ 1 (NOT Re~0).

Reports, per (model, heading):
  - full spectrum summary (max|lambda|, count |lambda|>0.9, mean top-5), not a binary
  - n distinct fixed points (clustered) -> the discrete-fold signature, so a code
    that folds via shared point attractors is not misread as "no slow modes".

Memory: 500x500 Jacobian = 2 MB; IC pool ~3000x500x8 = 12 MB. Trivial, CPU-only.
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

HEADINGS = ["E", "S", "W", "N"]   # index 0..3 = one-hot slot (hd_encodings.py)


@torch.no_grad()
def rollout_pool(model, dataset, hd_mode, n_states, device, seed=0):
    """IC pool of real hidden states + the mean obs / mean speed for the clamp."""
    hs, obs_sum, spd_sum, n_in, total = [], None, 0.0, 0, 0
    g = torch.Generator().manual_seed(seed)
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, _ = dataset[int(idx)]
        act = apply_hd(act, hd_mode)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0).cpu().numpy()
        take = min(h.shape[0], n_states - total)
        hs.append(h[:take])
        # obs is (T+1, 147); align to the T action steps
        o = obs[: h.shape[0]].numpy()
        obs_sum = o.sum(0) if obs_sum is None else obs_sum + o.sum(0)
        spd_sum += float(act[: h.shape[0], 0].sum())
        n_in += h.shape[0]
        total += take
        if total >= n_states:
            break
    hidden = np.concatenate(hs, 0)
    mean_obs = (obs_sum / n_in).astype(np.float32)      # (147,)
    mean_speed = spd_sum / n_in
    return hidden, mean_obs, mean_speed


def clamp_input(mean_obs, mean_speed, heading, hd_mode, device, obs_clamp="mean"):
    """u = [obs(147), speed, onehot(heading)] then apply the encoding to the hd block."""
    obs = mean_obs if obs_clamp == "mean" else np.zeros_like(mean_obs)
    act = np.zeros(5, np.float32)
    act[0] = mean_speed
    act[1 + heading] = 1.0
    act = apply_hd(torch.from_numpy(act), hd_mode).numpy()
    u = np.concatenate([obs, act]).astype(np.float32)
    return torch.from_numpy(u).to(device)


def find_fixed_points(cell, u, ic, iters, lr, device):
    """Minimise q(h)=0.5||F(h,u)-h||^2 over a batch of ICs.
    Returns (h*, dh_norm, h_norm): use rel = dh/h as the scale-free slowness."""
    h = torch.tensor(ic, device=device, requires_grad=True)
    U = u.unsqueeze(0).expand(h.shape[0], -1)
    Z = torch.zeros_like(h)
    opt = torch.optim.Adam([h], lr=lr)
    for it in range(iters):
        if it == int(0.6 * iters):
            for g in opt.param_groups:
                g["lr"] = lr * 0.2
        opt.zero_grad()
        hn, _ = cell(U, Z, (h, 0))
        q = 0.5 * ((hn - h) ** 2).sum(1)
        q.sum().backward()
        opt.step()
        with torch.no_grad():
            h.clamp_(min=0.0)   # states live in ReLU orthant; keep ICs feasible
    with torch.no_grad():
        hn, _ = cell(U, Z, (h, 0))
        dh = torch.linalg.norm(hn - h, dim=1)
        hnorm = torch.linalg.norm(h, dim=1)
    return h.detach(), dh.cpu().numpy(), hnorm.cpu().numpy()


def cluster(points, rel_radius):
    """Greedy distinct-point count: merge within an L2 radius scaled to the
    typical point norm, so the count is comparable across nets of different scale."""
    if not len(points):
        return np.array(points)
    scale = np.median(np.linalg.norm(points, axis=1)) + 1e-6
    radius = rel_radius * scale
    reps = []
    for p in points:
        if all(np.linalg.norm(p - r) > radius for r in reps):
            reps.append(p)
    return np.array(reps)


def jac_eig(cell, u, h_star, device):
    Z = torch.zeros(1, h_star.shape[0], device=device)
    U = u.unsqueeze(0)
    def F(h):
        hn, _ = cell(U, Z, (h.unsqueeze(0), 0))
        return hn.squeeze(0)
    J = torch.autograd.functional.jacobian(F, torch.tensor(h_star, device=device))
    ev = np.linalg.eigvals(J.cpu().numpy())
    return np.sort(np.abs(ev))[::-1]


def analyse(ckpt_path, dataset, args, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(ck, device)
    model.requires_grad_(False)
    cell = model.rnn.cell
    k = ck["meta"]["k"]

    pool, mean_obs, mean_speed = rollout_pool(
        model, dataset, args.hd_mode, args.n_pool, device, seed=args.seed)
    print(f"\n=== {ckpt_path}  (k={k}) ===")
    print(f"pool {pool.shape[0]} states | mean_speed={mean_speed:.3f} | "
          f"pool ||h|| median={np.median(np.linalg.norm(pool,axis=1)):.2f}")

    rng = np.random.default_rng(args.seed)
    out = {}
    for hd in range(4):
        u = clamp_input(mean_obs, mean_speed, hd, args.hd_mode, device, args.obs_clamp)
        ic = pool[rng.choice(pool.shape[0], size=args.n_ic, replace=False)]
        h_star, dh, hn = find_fixed_points(cell, u, ic, args.iters, args.lr, device)
        h_star = h_star.cpu().numpy()
        rel = dh / np.maximum(hn, 1e-6)
        slow = rel < args.rel_tol
        pts = h_star[slow]
        reps = cluster(pts, args.cluster_r) if len(pts) else np.empty((0, 500))
        specs = [jac_eig(cell, u, r, device) for r in reps[: args.n_jac]]
        specs = np.array(specs) if specs else np.zeros((0, 500))
        if len(specs):
            top = specs[:, 0]
            marg = ((specs >= 0.95) & (specs <= 1.05)).sum(1)   # integrating band
            out[hd] = dict(n_slow=int(slow.sum()), n_distinct=len(reps),
                           rel_min=float(rel.min()), max_lam=float(top.max()),
                           med_top=float(np.median(top)),
                           n_marg_band=float(marg.mean()),
                           frac_stable=float((top < 1.02).mean()),
                           specs=specs)
        else:
            out[hd] = dict(n_slow=int(slow.sum()), n_distinct=0,
                           rel_min=float(rel.min()), max_lam=np.nan,
                           med_top=np.nan, n_marg_band=np.nan,
                           frac_stable=np.nan, specs=specs)
        o = out[hd]
        print(f"  hd={HEADINGS[hd]}: rel_min={o['rel_min']:.1e}  "
              f"n_fp={o['n_distinct']:3d}  max|lam|={o['max_lam']:.3f}  "
              f"med_top|lam|={o['med_top']:.3f}  #|lam|in[.95,1.05]={o['n_marg_band']:.2f}  "
              f"frac_stable={o['frac_stable']:.2f}")
    return k, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default="/Volumes/Crucial X6/Thesis_work/project5_symmetry/_tmp_fund_domain_data/s2")
    ap.add_argument("--hd-mode", default="full")
    ap.add_argument("--n-pool", type=int, default=3000)
    ap.add_argument("--n-ic", type=int, default=400)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--cluster-r", type=float, default=0.03)   # fraction of state norm
    ap.add_argument("--obs-clamp", default="mean", choices=["mean", "zero"])
    ap.add_argument("--n-jac", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    dataset = TrajectoryDataset(a.data)
    print(f"dataset: {len(dataset)} trajectories from {a.data}")

    results = {}
    for c in a.ckpts:
        k, out = analyse(c, dataset, a, device)
        results[k] = out

    # save raw spectra for offline inspection
    save = {}
    for k in results:
        for hd in range(4):
            sp = results[k][hd].get("specs")
            if sp is not None and len(sp):
                save[f"k{k}_hd{hd}"] = sp
    if save:
        outp = str(Path(__file__).resolve().parent / "fp_stage0_spectra.npz")
        np.savez(outp, **save)
        print(f"\nsaved spectra -> {outp}")

    if len(results) >= 2 and 0 in results and 1 in results:
        print("\n============ k=0 (reconstruct) vs k=1 (predict) ============")
        print("           max|lam|        #|lam| in [.95,1.05]     frac_stable")
        for hd in range(4):
            a, b = results[0][hd], results[1][hd]
            print(f"  hd={HEADINGS[hd]}:  {a['max_lam']:.3f} -> {b['max_lam']:.3f}     "
                  f"{a['n_marg_band']:.2f} -> {b['n_marg_band']:.2f}          "
                  f"{a['frac_stable']:.2f} -> {b['frac_stable']:.2f}")


if __name__ == "__main__":
    main()
