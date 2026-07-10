"""Generate every paper figure from the result CSVs. No number is typed by hand.

Each function reads one or more CSVs from --data and writes one vector PDF to --figs. Run it
again whenever the CSVs are refreshed (e.g. n=4 -> n=8); the figures regenerate from whatever
is on disk. A figure whose CSV is missing is skipped with a printed note, never faked.

    python3 make_paper_figures.py --data <csv dir> --figs project5_symmetry/Report/biorxiv/figures
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Colourblind-safe (Wong 2011). Fixed per head-direction encoding across every figure.
HD_COLOR = {'full': '#0072B2', 'parity': '#009E73', 'axis': '#D55E00', 'const': '#7F7F7F'}
HD_ORDER = ['full', 'parity', 'axis', 'const']
COND_MARK = {'s1': 'o', 's2': 's', 's4': '^'}
TWO = {'translation': '#0072B2', 'rotation': '#D55E00'}

plt.rcParams.update({
    'font.size': 9, 'font.family': 'sans-serif', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.8, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'xtick.direction': 'out',
    'ytick.direction': 'out',
})


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _mean_sem(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return np.nan, np.nan
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0)


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


# --------------------------------------------------------------------------- replay + symmetry
def fig_horizon(data: Path, figs: Path):
    """Two panels sharing the prediction horizon k: symmetry resolution (orbit-phase decoding)
    and offline replay (coverage relative to a wake path of equal length). Both step up at
    k = 1 and are graded by how much of the compass each encoding keeps."""
    ph = {0: 'phase_horizon_k0', 1: 'phase_horizon_k1', 3: 'phase_horizon_k3'}
    phase = {}
    for k, name in ph.items():
        p = data / f'{name}.csv'
        if not p.exists():
            print(f'  skip fig_horizon: missing {name}'); return
        phase[k] = _read(p)
    gb = data / 'phase_groupB.csv'
    if gb.exists():
        phase[5] = [r for r in _read(gb) if r['condition'] == 's2' and r['k'] == '5']

    rep = {}
    for k in (0, 1, 3, 5):
        p = data / f'replay_k{k}.csv'
        if p.exists():
            rep[k] = [r for r in _read(p) if r['condition'] == 's2']

    ks_p = sorted(phase)
    ks_r = sorted(rep)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for hd in HD_ORDER:
        m, e = [], []
        for k in ks_p:
            mm, se = _mean_sem([_f(r, 'phase_acc') for r in phase[k] if r['hd_mode'] == hd])
            m.append(mm); e.append(se)
        axL.errorbar(ks_p, m, yerr=e, marker='o', ms=4, lw=1.5, capsize=2,
                     color=HD_COLOR[hd], label=hd)
    axL.axhline(0.5, ls='--', lw=0.8, color='k', alpha=0.5)
    axL.text(5, 0.515, 'chance', fontsize=7, ha='right', color='k', alpha=0.6)
    axL.set_xlabel('prediction horizon $k$'); axL.set_ylabel('orbit-phase accuracy')
    axL.set_title('Symmetry resolution'); axL.set_ylim(0.45, 1.02); axL.set_xticks(ks_p)

    for hd in HD_ORDER:
        m, e = [], []
        for k in ks_r:
            rows = [r for r in rep[k] if r['hd_mode'] == hd]
            ratio = [_f(r, 'off_cov') / _f(r, 'wake_cov') for r in rows
                     if np.isfinite(_f(r, 'wake_cov')) and _f(r, 'wake_cov') > 0]
            mm, se = _mean_sem(ratio); m.append(mm); e.append(se)
        axR.errorbar(ks_r, m, yerr=e, marker='o', ms=4, lw=1.5, capsize=2,
                     color=HD_COLOR[hd], label=hd)
    axR.axhline(1.0, ls='--', lw=0.8, color='k', alpha=0.5)
    axR.text(5, 1.03, 'wake', fontsize=7, ha='right', color='k', alpha=0.6)
    axR.set_xlabel('prediction horizon $k$')
    axR.set_ylabel('offline coverage / wake coverage')
    axR.set_title('Offline replay'); axR.set_xticks(ks_r)
    axR.legend(title='HD encoding', fontsize=7, title_fontsize=7, loc='upper left')

    fig.tight_layout()
    out = figs / 'fig_horizon.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


# ----------------------------------------------------------------------------- map forms, folds
def fig_map_quality(data: Path, figs: Path):
    """A map forms and folds. Spatial RSA collapses as the compass is ablated (left), while the
    position-conditioned code stays reproducible across seeds (right): folding is consistent,
    not degenerate."""
    p = data / 'map_quality_groupB.csv'
    if not p.exists():
        print('  skip fig_map_quality: missing map_quality_groupB'); return
    rows = _read(p)
    conds = sorted({r['condition'] for r in rows})
    x = np.arange(len(HD_ORDER)); w = 0.8 / max(len(conds), 1)
    shades = {c: plt.cm.Greys(0.35 + 0.5 * i / max(len(conds) - 1, 1)) for i, c in enumerate(conds)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for panel, (ax, col, ylab, ttl) in enumerate([
            (axA, 'srsa_e', 'spatial RSA', 'Folding collapses spatial RSA'),
            (axB, 'cross_seed_rho', 'cross-seed correlation', 'The fold is reproducible')]):
        for i, c in enumerate(conds):
            m, e = [], []
            for hd in HD_ORDER:
                mm, se = _mean_sem([_f(r, col) for r in rows
                                    if r['condition'] == c and r['hd_mode'] == hd])
                m.append(mm); e.append(se)
            ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=2,
                   color=shades[c], edgecolor='k', lw=0.5, label=c)
        ax.set_xticks(x); ax.set_xticklabels(HD_ORDER); ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.set_ylim(0, 1.0)
    axB.legend(title='arena', fontsize=7, title_fontsize=7)
    fig.tight_layout()
    out = figs / 'fig_map_quality.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


# ------------------------------------------------------------------------------- compartments
def fig_compartments(data: Path, figs: Path):
    """In-silico Grieves. Parallel (translation) compartments fold: fields repeat and the room
    is undecodable. Radial (rotation) compartments lift. within_r2 stays high in both, so
    neither has stopped coding space."""
    p = data / 'compartments.csv'
    if not p.exists():
        print('  skip fig_compartments: missing compartments'); return
    rows = _read(p)
    modes = [m for m in ('translation', 'rotation') if any(r['mode'] == m for r in rows)]
    metrics = [('repetition', 'field repetition'), ('room_gen', 'room decoding\n(held-out cells)'),
               ('room_seen', 'room decoding\n(seen cells)'), ('within_r2', 'within-room $R^2$')]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, mode in enumerate(modes):
        sub = [r for r in rows if r['mode'] == mode]
        m, e = [], []
        for col, _ in metrics:
            mm, se = _mean_sem([_f(r, col) for r in sub]); m.append(mm); e.append(se)
        xpos = x + (i - 0.5) * w
        ax.bar(xpos, m, w, yerr=e, capsize=3, color=TWO[mode], edgecolor='k', lw=0.5,
               label=f'{mode} (n={len(sub)})')
        for j, (col, _) in enumerate(metrics):
            pts = [_f(r, col) for r in sub]
            ax.scatter(np.full(len(pts), xpos[j]), pts, s=9, color='k', alpha=0.5, zorder=3)
    ax.axhline(0.5, ls='--', lw=0.8, color='k', alpha=0.5)
    ax.text(len(metrics) - 0.5, 0.52, 'chance (room)', fontsize=7, ha='right', alpha=0.6)
    ax.axhline(0.0, lw=0.6, color='k')
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics], fontsize=8)
    ax.set_ylabel('value'); ax.set_ylim(-0.3, 1.1)
    ax.set_title('Place-field repetition across identical compartments')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    out = figs / 'fig_compartments.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


# --------------------------------------------------------------------------------- init study
def fig_init(data: Path, figs: Path):
    """Learning speed by recurrent initialisation, ranked by area under the loss curve (lower is
    faster). zero_rec (pure leak, no random recurrence) is fastest; the abandoned kaiming init
    is slowest but does not diverge."""
    frames = []
    for name in ('speed_init', 'speed_kaiming'):
        p = data / f'{name}.csv'
        if p.exists():
            frames += _read(p)
    if not frames:
        print('  skip fig_init: missing speed_init/speed_kaiming'); return
    variants = sorted({r['variant'] for r in frames},
                      key=lambda v: np.mean([_f(r, 'loss_auc') for r in frames if r['variant'] == v]))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for i, v in enumerate(variants):
        sub = [r for r in frames if r['variant'] == v]
        vals = [_f(r, 'loss_auc') for r in sub]
        m, e = _mean_sem(vals)
        color = '#D55E00' if v.startswith('kaiming') else ('#0072B2' if v == 'baseline' else '#7F7F7F')
        ax.bar(i, m, 0.7, yerr=e, capsize=3, color=color, edgecolor='k', lw=0.5)
        ax.scatter(np.full(len(vals), i), vals, s=10, color='k', alpha=0.5, zorder=3)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('loss AUC over training  (lower = faster)')
    ax.set_title('Learning speed by recurrent initialisation')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#0072B2', '#7F7F7F', '#D55E00')]
    ax.legend(handles, ['paper default', 'other', 'kaiming (abandoned)'], fontsize=7, loc='upper left')
    fig.tight_layout()
    out = figs / 'fig_init.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


# ---------------------------------------------------------------------- topology before geometry
def fig_topology(data: Path, figs: Path):
    """Geometry and topology across training. Requires the shuffle-null TDA (b1_hat can be 0);
    a run where `open` never scores b1_correct is the broken estimator and is refused."""
    p = data / 'tda_topology.csv'
    if not p.exists():
        print('  skip fig_topology: missing tda_topology'); return
    rows = [r for r in _read(p) if r['step'] != 'final']
    openrows = [r for r in rows if r['layout'] == 'open']
    if openrows and not any(r['b1_correct'] in ('1', 'True') for r in openrows):
        print('  skip fig_topology: open never correct -> pre-null (broken) CSV, not plotting')
        return
    steps = sorted({int(r['step']) for r in rows})
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    # geometry: decode_r2 pooled over layouts/seeds, vs step; topology: fraction b1 correct
    g_m, g_e, t = [], [], []
    for s in steps:
        sub = [r for r in rows if int(r['step']) == s]
        mm, se = _mean_sem([_f(r, 'decode_r2') for r in sub]); g_m.append(mm); g_e.append(se)
        t.append(np.mean([1.0 if r['b1_correct'] in ('1', 'True') else 0.0 for r in sub]))
    xs = [max(s, 1) for s in steps]
    ax.errorbar(xs, g_m, yerr=g_e, marker='o', ms=4, lw=1.5, capsize=2,
                color='#0072B2', label='geometry (position $R^2$)')
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    ax2.plot(xs, t, marker='s', ms=4, lw=1.5, color='#D55E00', label='topology (frac. $b_1$ correct)')
    ax.axvline(1, ls=':', lw=0.8, color='k', alpha=0.4)
    ax.text(1.1, ax.get_ylim()[0], 'init', fontsize=7, alpha=0.6)
    ax.set_xscale('log'); ax.set_xlabel('training step')
    ax.set_ylabel('position decode $R^2$', color='#0072B2')
    ax2.set_ylabel('fraction $b_1$ correct', color='#D55E00'); ax2.set_ylim(0, 1)
    ax.set_title('Geometry and topology across training')
    fig.tight_layout()
    out = figs / 'fig_topology.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--figs', required=True)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    data, figs = Path(a.data), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    allfigs = {'horizon': fig_horizon, 'map_quality': fig_map_quality,
               'compartments': fig_compartments, 'init': fig_init, 'topology': fig_topology}
    for name, fn in allfigs.items():
        if a.only and name not in a.only:
            continue
        print(f'[{name}]')
        fn(data, figs)


if __name__ == '__main__':
    main()
