#!/usr/bin/env python3
"""Every k=5 experiment in ONE vmap ensemble.

The graph is fixed by (F, k, T) alone. Arenas, arena *sizes*, initialisations and seeds all
leave it untouched -- only the data store and the initial weights differ. Running these as
four sequential jobs pays four inductor compiles and leaves the card at ~60% while a single
job of the same total size sits at ~95%.

    topology     4 arenas (b1 = 0,1,2,2) x 4 seeds        16
    init         8 recurrent inits on the annulus x 4      32
    kaiming      the abandoned init, 2 arms x 4 seeds       8
    compartment  translation / rotation x 4 seeds           8
                                                          ---
                                                           64 models, one compile

Checkpoints are log-spaced from step 0. Step 0 is the untrained network and is the control
that decides the topology-before-geometry interpretation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project5_symmetry.environments.compartment_arenas import (MODES as COMP_MODES,  # noqa: E402
                                                               make_compartment_env)
from project5_symmetry.environments.generate_trajectories import generate_dataset  # noqa: E402
from project5_symmetry.environments.topology_arenas import (EXPECTED_B1,  # noqa: E402
                                                            LAYOUTS, betti,
                                                            make_topology_env)
from project5_symmetry.experiments.run_ensemble_sweep import (  # noqa: E402
    ANCHOR_SUBSAMPLE_N, DROPOUT_P, F, HIDDEN_INIT_SIGMA, HIDDEN_SIZE, K, NOISE_STD,
    OBS, PRED_OFFSET, T, _enable_tf32, _sample_anchor_idx)
from project5_symmetry.experiments.run_hd_invariance import build_models_k  # noqa: E402
from project5_symmetry.training import ensemble as ens  # noqa: E402
from project5_symmetry.training.dataset import PackedTrajectoryStore  # noqa: E402
from project5_symmetry.training.inits import VARIANTS, apply_init, spectral_radius  # noqa: E402

CHECKPOINT_STEPS = (0, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 80000)
INIT_VARIANTS = ('tau1', 'baseline', 'tau4', 'tau8', 'gain_lo', 'gain_hi', 'orth', 'zero_rec')
KAIMING_VARIANTS = ('kaiming', 'kaiming_noid')


def build_spec(n_seeds: int, groups: tuple[str, ...], seed0: int = 0) -> list[dict]:
    """Each entry: group, store name, init variant (or None), seed.

    `seed0` shifts the seed block so a second run extends the ensemble instead of retracing
    it: seeds are deterministic, so `--n-seeds 8` would recompute 0..3 bit-for-bit and write
    them over themselves. `--seed-offset 4 --n-seeds 4` adds seeds 4..7 for the same cost.
    """
    seeds = range(seed0, seed0 + n_seeds)
    spec = []
    if 'topology' in groups:
        spec += [{'group': 'topology', 'store': f'topology/{l}', 'name': l,
                  'init': None, 'seed': s} for l in LAYOUTS for s in seeds]
    if 'init' in groups:
        spec += [{'group': 'init', 'store': 'topology/annulus', 'name': v,
                  'init': v, 'seed': s} for v in INIT_VARIANTS for s in seeds]
    if 'kaiming' in groups:
        spec += [{'group': 'kaiming', 'store': 'topology/annulus', 'name': v,
                  'init': v, 'seed': s} for v in KAIMING_VARIANTS for s in seeds]
    if 'compartment' in groups:
        spec += [{'group': 'compartment', 'store': f'compartment/{m}', 'name': m,
                  'init': None, 'seed': s} for m in COMP_MODES for s in seeds]
    return spec


def torch_seed(e: dict) -> int:
    """Seed for the constructor draw (W_in, W_out, bias).

    The init arms must share these within a seed so the sweep is PAIRED: only the recurrent
    matrix differs. The topology and compartment arms index by their own store so different
    arenas get different draws.
    """
    if e['group'] in ('init', 'kaiming'):
        return e['seed']
    return 1000 * (hash_name(e['name'])) + e['seed']


def hash_name(name: str) -> int:
    """Small deterministic index. NOT hash(): that is salted per process."""
    table = list(LAYOUTS) + list(COMP_MODES)
    return table.index(name) + 1 if name in table else 0


def ensure_store(store: str, data_root: str, n_traj: int, workers: int) -> Path:
    kind, name = store.split('/')
    d = Path(data_root) / kind / name
    if kind == 'topology':
        generate_dataset(make_topology_env(name, F=F, seed=0), n_traj=n_traj, T=T,
                         out_dir=str(d), n_workers=workers, desc=name,
                         env_factory=make_topology_env,
                         factory_kwargs={'layout': name, 'F': F, 'seed': 0})
    else:
        generate_dataset(make_compartment_env(name, F=F, seed=0), n_traj=n_traj, T=T,
                         out_dir=str(d), n_workers=workers, desc=name,
                         env_factory=make_compartment_env,
                         factory_kwargs={'mode': name, 'F': F, 'seed': 0})
    return d


def sample_batches(stores, spec, B):
    """Rows of obs must line up with `spec` positionally. `spec` is grouped by store."""
    order = list(dict.fromkeys(e['store'] for e in spec))
    obs_parts, act_parts = [], []
    for st in order:
        n = sum(1 for e in spec if e['store'] == st)
        o, a = stores[st].sample_parallel_batches(n, B)
        obs_parts.append(o)
        act_parts.append(a)
    return torch.cat(obs_parts, 0), torch.cat(act_parts, 0)


def save_checkpoints(params, spec, out_root, step, meta):
    for i, e in enumerate(spec):
        d = Path(out_root) / e['group'] / e['name'] / f"seed_{e['seed']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        name = 'ckpt_final.pt' if step == 'final' else f'ckpt_{step}.pt'
        extra = {'group': e['group'], 'seed': e['seed'], 'store': e['store']}
        if e['group'] in ('topology',):
            extra |= {'layout': e['name'], 'b1': EXPECTED_B1[e['name']]}
        elif e['group'] in ('init', 'kaiming'):
            tau, gain, struct = VARIANTS[e['name']]
            extra |= {'variant': e['name'], 'layout': 'annulus', 'b1': EXPECTED_B1['annulus'],
                      'tau': tau, 'gain': gain, 'struct': struct}
        else:
            extra |= {'mode': e['name']}
        torch.save({'step': step, 'model': ens.unstack_state_dict(params, i),
                    'meta': meta | extra}, d / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/root/runs/multi')
    ap.add_argument('--data-root', default='/root/data')
    ap.add_argument('--n-steps', type=int, default=80_000)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--n-traj', type=int, default=10_000)
    ap.add_argument('--n-seeds', type=int, default=4)
    ap.add_argument('--seed-offset', type=int, default=0,
                    help='train seeds [offset, offset + n_seeds); use to EXTEND an ensemble')
    ap.add_argument('--groups', nargs='+',
                    default=['topology', 'init', 'kaiming', 'compartment'])
    ap.add_argument('--dataset-workers', type=int, default=16)
    ap.add_argument('--compile', default='default', choices=['default', 'none'])
    ap.add_argument('--log-every', type=int, default=200)
    a = ap.parse_args()

    device = torch.device('cuda')
    _enable_tf32(device)
    k, B, T_k = K, a.batch_size, T - K
    spec = sorted(build_spec(a.n_seeds, tuple(a.groups), a.seed_offset), key=lambda e: e['store'])
    S = len(spec)
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)

    print(f'device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}')
    print(f'models : S={S}  (one graph, one compile)')
    for g in a.groups:
        n = sum(1 for e in spec if e['group'] == g)
        print(f'  {g:<12s} {n:>3d} models')
    print(f'config : B={B} T={T} k={k} steps={a.n_steps}')
    print(f'ckpts  : {CHECKPOINT_STEPS}\n', flush=True)

    stores = {}
    for st in dict.fromkeys(e['store'] for e in spec):
        d = ensure_store(st, a.data_root, a.n_traj, a.dataset_workers)
        stores[st] = PackedTrajectoryStore(str(d), device=device)
        print(f'  {st}: {len(stores[st])} trajectories', flush=True)

    models = build_models_k([torch_seed(e) for e in spec], device, k)
    for m, e in zip(models, spec):
        if e['init'] is not None:
            apply_init(m, e['init'], e['seed'])

    # paired design: the init arms share W_in / W_out / bias within a seed
    init_arms = [i for i, e in enumerate(spec) if e['group'] in ('init', 'kaiming')]
    for s in range(a.n_seeds):
        arms = [i for i in init_arms if spec[i]['seed'] == s]
        for i in arms[1:]:
            for nm in ('W_in', 'W_out', 'bias'):
                assert torch.equal(getattr(models[arms[0]], nm), getattr(models[i], nm)), \
                    f'seed {s}: {nm} differs across init variants; design is not paired'

    for e, m in zip(spec, models):
        if e['seed'] == 0 and e['init'] is not None:
            W = m.W.detach().cpu().numpy()
            print(f"  init {e['name']:<13s} diag={W.diagonal().mean():+.3f} "
                  f"rho={spectral_radius(W):.3f}", flush=True)

    obs, act = sample_batches(stores, spec, B)
    state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
    nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
    nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
    aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)
    for i, m in enumerate(models):
        m.eval()
        ens.warm_buffers(m, obs[i], act[i], anchor_idx=aidx, state=state[i],
                         noise_main=nm[i], noise_roll=nr[i])
        m.train()

    params, buffers, base = ens.stack_models(models)
    base.train()
    leaves = ens.as_leaves(params)
    opt = ens.stacked_rmsprop(leaves, batch_size=B)
    gfn = ens.make_loss_and_grad_fn(base, buffers, randomness='different')
    if a.compile != 'none':
        gfn = torch.compile(gfn, mode=a.compile)

    meta = {'obs_size': OBS, 'k': k, 'trunc': T, 'hidden_size': HIDDEN_SIZE, 'F': F,
            'batch_size': B, 'noise_std': NOISE_STD, 'dropout_p': DROPOUT_P,
            'hidden_init_sigma': HIDDEN_INIT_SIGMA, 'pred_offset': PRED_OFFSET,
            'n_steps': a.n_steps, 'runner': 'ensemble_vmap_multi'}
    key = lambda e: f"{e['group']}/{e['name']}/seed_{e['seed']:02d}"
    log = {key(e): {'steps': [], 'loss': []} for e in spec}

    save_checkpoints(leaves, spec, out_root, 0, meta)
    print('  [ckpt] 0  (the untrained network)', flush=True)

    print('compiling + training...', flush=True)
    t0 = time.perf_counter()
    for step in range(1, a.n_steps + 1):
        obs, act = sample_batches(stores, spec, B)
        state = torch.rand(S, B, HIDDEN_SIZE, device=device) * HIDDEN_INIT_SIGMA
        nm = NOISE_STD * torch.randn(S, B, T, HIDDEN_SIZE, device=device)
        nr = NOISE_STD * torch.randn(S, B, k, ANCHOR_SUBSAMPLE_N, HIDDEN_SIZE, device=device)
        aidx = _sample_anchor_idx(T_k, device, ANCHOR_SUBSAMPLE_N)

        grads, losses = gfn(leaves, obs, act, state, nm, nr, aidx)
        grads = {kk: v.clone() for kk, v in grads.items()}
        ens.clip_per_slice(grads, max_norm=1.0)
        opt.zero_grad(set_to_none=True)
        ens.apply_grads(leaves, grads)
        opt.step()

        if step == 1:
            print(f'  first step (incl. compile) {time.perf_counter()-t0:.0f}s  '
                  f'peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
        if step % a.log_every == 0 or step == 1:
            l_ = losses.detach().cpu().tolist()
            for e, v in zip(spec, l_):
                log[key(e)]['steps'].append(step)
                log[key(e)]['loss'].append(v)
            if step % 5000 == 0:
                el = time.perf_counter() - t0
                print(f'  step {step:>6}/{a.n_steps}  mean loss {sum(l_)/len(l_):.5f}  '
                      f'{el/step*1000:.1f} ms/step  eta {(a.n_steps-step)*el/step/60:.0f} min',
                      flush=True)
        if step in CHECKPOINT_STEPS:
            save_checkpoints(leaves, spec, out_root, step, meta)
            print(f'  [ckpt] {step}', flush=True)

    save_checkpoints(leaves, spec, out_root, 'final', meta)
    for e in spec:
        d = out_root / e['group'] / e['name'] / f"seed_{e['seed']:02d}"
        with open(d / 'training_log.json', 'w') as f:
            json.dump(log[key(e)] | {'meta': meta | {'group': e['group'],
                                                     'variant': e['name'],
                                                     'seed': e['seed']}}, f)

    tot = time.perf_counter() - t0
    print(f'\nDONE  {a.n_steps} steps in {tot/60:.1f} min '
          f'({tot/a.n_steps*1000:.1f} ms/ensemble-step, {tot/a.n_steps/S*1000:.2f} ms/model-step)',
          flush=True)


if __name__ == '__main__':
    main()
