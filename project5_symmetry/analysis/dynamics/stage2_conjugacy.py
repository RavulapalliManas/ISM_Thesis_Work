#!/usr/bin/env python3
"""Stage 2, definitive: is there a HIDDEN dynamical symmetry where the representation lifts?

If the one-step flow F is equivariant under a learned linear group action rho(g) on hidden
state, linearising gives the conjugacy condition  J(g.x) = rho J(x) rho^{-1}.

Protocol (with a reachable null so it can fail):
  1. mean hidden state H(k) and mean input U(k) per (pos, heading) key, k=1.
  2. C2 orbit pairs (k, g.k): g.pos = repo C2 map, g.heading = heading+2. SPLIT train/test.
  3. fit rho (ridge) on TRAIN:  rho = B A^T (A A^T + lam I)^{-1},  A=[H(k)], B=[H(g.k)].
  4. on HELD-OUT TEST pairs measure:
       rep error:   ||rho H(k) - H(g.k)|| / ||H(g.k)||      vs  rho=I baseline
       flow conj:   ||J(g.k) - rho J(k) rho^{-1}||_F / ||J(g.k)||_F
                    vs  rho=I baseline  and  vs  shuffled-pair null (conjugate, wrong partner)
  Diagnostic: ||rho^2 - I|| (C2 => g^2=e => rho^2 should be ~I).

Read: for full/parity, conj_rho << conj_I and << conj_shuf  =>  hidden dynamical symmetry.
      conj_rho ~ conj_I ~ conj_shuf                          =>  lift is genuinely non-equivariant.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, "/Volumes/Crucial X6/Thesis_work")
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint  # noqa
from project5_symmetry.analysis.run_phase_decoding import images, ARENA    # noqa
from project5_symmetry.environments.hd_encodings import apply_hd           # noqa
from project5_symmetry.training.dataset import TrajectoryDataset           # noqa

CKROOT = "/Volumes/Crucial X6/prnn_backup/checkpoints/horizon"
ENCS = ["full", "axis", "parity", "const"]


@torch.no_grad()
def collect_keyed(model, dataset, enc, n_states, device, seed, min_count):
    hacc, uacc, cnt = {}, {}, {}
    g = torch.Generator().manual_seed(seed)
    tot = 0
    for idx in torch.randperm(len(dataset), generator=g):
        obs, act, pos, heading = dataset[int(idx)]
        act = apply_hd(act, enc)
        _, h, _ = model(obs.unsqueeze(0).to(device), act.unsqueeze(0).to(device))
        h = h.squeeze(0); T = h.shape[0]
        u = torch.cat([obs[:T], act[:T]], dim=1)
        p = pos[:T].numpy().astype(int); hd = heading[:T].numpy().astype(int)
        for t in range(T):
            key = (int(p[t, 0]), int(p[t, 1]), int(hd[t]))
            hacc[key] = hacc.get(key, 0.0) + h[t].numpy()
            uacc[key] = uacc.get(key, 0.0) + u[t].numpy()
            cnt[key] = cnt.get(key, 0) + 1
        tot += T
        if tot >= n_states:
            break
    H = {k: hacc[k] / cnt[k] for k in hacc if cnt[k] >= min_count}
    U = {k: uacc[k] / cnt[k] for k in hacc if cnt[k] >= min_count}
    return H, U


def jac(cell, h, u):
    Z = torch.zeros(1, h.shape[0]); U = torch.from_numpy(u).float().unsqueeze(0)
    def F(x):
        hn, _ = cell(U, Z, (x.unsqueeze(0), 0))
        return hn.squeeze(0)
    J = torch.autograd.functional.jacobian(F, torch.from_numpy(h).float(), vectorize=True)
    return J.numpy()


def relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--data", default="/Volumes/Crucial X6/Thesis_work/project5_symmetry/_tmp_fund_domain_data/s2")
    ap.add_argument("--n-states", type=int, default=30000)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--max-test", type=int, default=120)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    dataset = TrajectoryDataset(a.data)
    rng = np.random.default_rng(a.seed)
    print(f"dataset {len(dataset)} traj | k={a.k} | rho(g) conjugacy test (train/test split)\n")
    print(f"  {'enc':7s} {'npair':>5s} | {'rep_rho':>7s} {'rep_I':>6s} | "
          f"{'flow_rho':>8s} {'flow_I':>6s} {'flow_shuf':>9s} | {'rho^2-I':>6s} {'var':>5s}")

    for enc in ENCS:
        path = f"{CKROOT}/k{a.k}/s2/{enc}/seed_{a.seed:02d}/ckpt_final.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(ck, device).requires_grad_(False)
        cell = model.rnn.cell
        H, U = collect_keyed(model, dataset, enc, a.n_states, device, a.seed, a.min_count)

        cells = np.array(sorted({(x, y) for (x, y, _) in H}), dtype=int)
        orb = images(cells, "c2", ARENA).astype(int)
        cmap = {tuple(cells[i]): tuple(orb[i, 1]) for i in range(len(cells))}

        pairs, seen = [], set()
        for key in H:
            x, y, th = key
            if (x, y) not in cmap:
                continue
            gk = (int(cmap[(x, y)][0]), int(cmap[(x, y)][1]), (th + 2) % 4)
            if gk not in H or gk == key:
                continue
            fs = frozenset((key, gk))
            if fs in seen:
                continue
            seen.add(fs); pairs.append((key, gk))
        rng.shuffle(pairs)
        n_test = min(a.max_test, len(pairs) // 3)
        test, train = pairs[:n_test], pairs[n_test:]
        if len(train) < 20 or n_test < 10:
            print(f"  {enc:7s}  too few pairs ({len(pairs)})"); continue

        # Center (ReLU DC offset -> affine action) and restrict to the top-d PC subspace, where
        # the code lives and a d x d orthogonal rho is well-determined by hundreds of pairs. The
        # flow is projected into the same subspace: J_d = V^T J V. Differentiation kills the
        # constant offset, so J_d(g.x) rho = rho J_d(x) is the linearized equivariance in-subspace.
        allH = np.stack(list(H.values()))
        m = allH.mean(0)
        Vfull, S, _ = np.linalg.svd((allH - m).T, full_matrices=False)
        d = min(a.dim, len(train))
        V = Vfull[:, :d]                                # (500, d) PC basis
        var_expl = float((S[:d] ** 2).sum() / (S ** 2).sum())
        Hc = {k: V.T @ (H[k] - m) for k in H}          # d-dim PC coords

        A = np.stack([Hc[a_] for a_, _ in train], 1)   # (d, Ntr)
        B = np.stack([Hc[b_] for _, b_ in train], 1)
        Um, _, Vt = np.linalg.svd(B @ A.T)
        rho = Um @ Vt                                  # d x d orthogonal
        rho2_dev = relerr(rho @ rho, np.eye(d))

        rep_rho = np.median([relerr(rho @ Hc[a_], Hc[b_]) for a_, b_ in test])
        rep_I = np.median([relerr(Hc[a_], Hc[b_]) for a_, b_ in test])

        # FLOW equivariance in-subspace (inverse-free): J_d(g.x) rho == rho J_d(x) ?
        need = {k for pr in test for k in pr}
        Jd = {k: V.T @ jac(cell, H[k], U[k]) @ V for k in need}
        flow_rho = [relerr(Jd[b_] @ rho, rho @ Jd[a_]) for a_, b_ in test]
        flow_I = [relerr(Jd[b_], Jd[a_]) for a_, b_ in test]
        bs = [b_ for _, b_ in test]
        perm = rng.permutation(len(test))
        flow_shuf = [relerr(Jd[bs[perm[i]]] @ rho, rho @ Jd[test[i][0]]) for i in range(len(test))]

        print(f"  {enc:7s} {len(test):>5d} | {rep_rho:>7.3f} {rep_I:>6.3f} | "
              f"{np.median(flow_rho):>8.3f} {np.median(flow_I):>6.3f} {np.median(flow_shuf):>9.3f} | "
              f"{rho2_dev:>6.3f} {var_expl:>5.2f}")


if __name__ == "__main__":
    main()
