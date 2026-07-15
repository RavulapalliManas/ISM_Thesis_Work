#!/usr/bin/env python3
"""Stage 2, anchor: is the C2 fold present in the recurrent STATE at k=1, and does
it obey the matched axis-vs-parity dissociation?

Reuses the paper's own machinery (position_means, fold_coincidence, images/ARENA)
so the group action is the established one, not reinvented. This is the state-space
version of the readout dissociation and validates the pipeline before the novel
vector-field (Jacobian) fold is built on top.

fold_coincidence ~ 1  => H(x) == H(g.x): the orbit is one point (folded).
Prediction (matched 1 bit):  axis should fold (C2-invariant), parity should not.
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, "/Volumes/Crucial X6/Thesis_work")
from project5_symmetry.analysis.run_spectrum import model_from_checkpoint, collect  # noqa
from project5_symmetry.analysis.isometry_quotient import position_means, fold_coincidence  # noqa
from project5_symmetry.training.dataset import TrajectoryDataset  # noqa

CKROOT = "/Volumes/Crucial X6/prnn_backup/checkpoints/horizon"
ENCS = ["full", "axis", "parity", "const"]
BITS = {"full": 2.0, "axis": 1.0, "parity": 1.0, "const": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--data", default="/Volumes/Crucial X6/Thesis_work/project5_symmetry/_tmp_fund_domain_data/s2")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    dataset = TrajectoryDataset(a.data)
    print(f"dataset {len(dataset)} traj | k={a.k} | C2 state-fold (median cosine of orbit-paired mean states)\n")
    print(f"  {'encoding':8s} {'bits':>4s}   {'C2 state-fold':>13s}   prediction")
    for enc in ENCS:
        path = f"{CKROOT}/k{a.k}/s2/{enc}/seed_{a.seed:02d}/ckpt_final.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(ck, device)
        H, pos = collect(model, dataset, enc, a.n_states, device, seed=a.seed)
        M, cells = position_means(H, pos)
        fold = fold_coincidence(M, cells, "c2")
        pred = "folds (C2-inv)" if enc in ("axis", "const") else "lifts (breaks C2)"
        print(f"  {enc:8s} {BITS[enc]:>4.0f}   {fold:>13.3f}   {pred}")


if __name__ == "__main__":
    main()
