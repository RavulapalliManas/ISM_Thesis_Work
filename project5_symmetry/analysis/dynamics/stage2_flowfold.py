#!/usr/bin/env python3
"""Stage 2, payload: does the network implement a G-EQUIVARIANT FLOW?

The state fold says H(x) == H(g.x). This asks the dynamical question the readout
cannot: is the LOCAL FLOW (Jacobian of the one-step map) the same at orbit-related
points -- and does that hold only where the representation folds (axis/const) or
also where it lifts (full/parity)?

Orbit pairs matched on (position, heading) via the repo's C2 map: (p, th) -> (R2 p, th+2).
At each member take the exact 500x500 Jacobian dF/dh (at that state's mean h and mean
input u). Compare the SORTED |lambda| spectra (basis-free) of orbit pairs vs random
pairs. Report a flow-fold index = median orbit-pair spectral similarity minus random.

similarity = 1 - ||sort|lam|_a - sort|lam|_b|| / (||.||_a + ||.||_b)   in [0,1], 1=identical flow.
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
def collect_keyed(model, dataset, enc, n_states, device, seed=0):
    """Mean hidden state and mean input per (position, heading) key."""
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
    H = {k: hacc[k] / cnt[k] for k in hacc}
    U = {k: uacc[k] / cnt[k] for k in uacc}
    return H, U


def jac_spectrum(cell, h, u):
    Z = torch.zeros(1, h.shape[0]); U = torch.from_numpy(u).float().unsqueeze(0)
    def F(x):
        hn, _ = cell(U, Z, (x.unsqueeze(0), 0))
        return hn.squeeze(0)
    J = torch.autograd.functional.jacobian(F, torch.from_numpy(h).float(), vectorize=True)
    return np.sort(np.abs(np.linalg.eigvals(J.numpy())))[::-1]


def sim(a, b):
    return 1.0 - np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def c2_partner(key, cell_orbit):
    (x, y, th) = key
    px, py = cell_orbit.get((x, y), (None, None))
    if px is None:
        return None
    return (int(px), int(py), (th + 2) % 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--data", default="/Volumes/Crucial X6/Thesis_work/project5_symmetry/_tmp_fund_domain_data/s2")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    dataset = TrajectoryDataset(a.data)
    rng = np.random.default_rng(a.seed)
    print(f"dataset {len(dataset)} traj | k={a.k} | flow-fold: Jacobian-spectrum similarity, orbit vs random\n")
    print(f"  {'encoding':8s}  {'state-cos':>9s}  {'orbit-sim':>9s}  {'rand-sim':>8s}  {'flow-fold':>9s}  n_pairs")

    for enc in ENCS:
        path = f"{CKROOT}/k{a.k}/s2/{enc}/seed_{a.seed:02d}/ckpt_final.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(ck, device).requires_grad_(False)
        cell = model.rnn.cell
        H, U = collect_keyed(model, dataset, enc, a.n_states, device, seed=a.seed)

        # position orbit from repo's C2 map
        cells = np.array(sorted({(x, y) for (x, y, _) in H}), dtype=int)
        orb = images(cells, "c2", ARENA).astype(int)
        cell_orbit = {tuple(cells[i]): tuple(orb[i, 1]) for i in range(len(cells))}

        keys = list(H)
        pairs = []
        seen = set()
        for key in keys:
            pk = c2_partner(key, cell_orbit)
            if pk is None or pk not in H or pk == key:
                continue
            fs = frozenset((key, pk))
            if fs in seen:
                continue
            seen.add(fs); pairs.append((key, pk))
        rng.shuffle(pairs)
        pairs = pairs[: a.max_pairs]
        if not pairs:
            print(f"  {enc:8s}  no orbit pairs"); continue

        # spectra cache
        need = {k for pr in pairs for k in pr}
        spec = {k: jac_spectrum(cell, H[k], U[k]) for k in need}
        # state cosine (heading-matched) for reference
        scos = [float(np.dot(H[a_], H[b_]) / (np.linalg.norm(H[a_]) * np.linalg.norm(H[b_]) + 1e-12))
                for a_, b_ in pairs]
        orbit_sim = [sim(spec[a_], spec[b_]) for a_, b_ in pairs]
        # random pairs from the same key pool
        klist = list(need)
        rand_sim = []
        for _ in range(len(pairs)):
            i, j = rng.choice(len(klist), 2, replace=False)
            rand_sim.append(sim(spec[klist[i]], spec[klist[j]]))
        os_, rs_ = np.median(orbit_sim), np.median(rand_sim)
        print(f"  {enc:8s}  {np.median(scos):>9.3f}  {os_:>9.3f}  {rs_:>8.3f}  {os_-rs_:>9.3f}  {len(pairs)}")


if __name__ == "__main__":
    main()
