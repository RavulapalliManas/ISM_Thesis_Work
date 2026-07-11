"""Generate the paper's composite figures from the result CSVs. No number is typed by hand.

Four main figures, each one matplotlib figure with internal panels (a, b, ...), built to Nature
figure specs: 183 mm double-column width, <=170 mm tall, Arial, 5-7 pt text, 8 pt bold lowercase
panel letters, thin strokes, Wong colourblind-safe palette, vector PDF. Panel titles are
descriptive, not interpretive; claims live in the captions. Arenas are named by their symmetry
group C1/C2/C4 throughout.

    python3 make_paper_figures.py --data <csv dir> --figs project5_symmetry/Report/biorxiv/figures
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

# Muted, formal palette (still colourblind-distinguishable), fixed per HD encoding everywhere.
HD_COLOR = {'full': '#1F3B5C', 'parity': '#3A6B6B', 'axis': '#9C4A2F', 'const': '#8A8A8A'}
HD_ORDER = ['full', 'parity', 'axis', 'const']
TWO = {'translation': '#1F3B5C', 'rotation': '#9C4A2F'}
COND_CN = {'s1': '$C_1$', 's2': '$C_2$', 's4': '$C_4$'}       # arena naming, everywhere
COND_SHADE = {'s1': '#BBBBBB', 's2': '#6E6E6E', 's4': '#222222'}
INK = '#222222'
RATE_CMAP = plt.get_cmap('magma').copy(); RATE_CMAP.set_bad('#ffffff')
ANG_CMAP = plt.get_cmap('twilight')

# Nature widths at print size (inches): single 89 mm, 1.5-col 136 mm, double 183 mm; <=170 mm tall.
COL, ONEHALF, WIDE, HMAX = 3.50, 5.35, 7.20, 6.69

_PREFERRED = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'TeX Gyre Heros', 'DejaVu Sans']
_have = {f.name for f in font_manager.fontManager.ttflist}
_family = next((f for f in _PREFERRED if f in _have and f != 'Helvetica'), 'DejaVu Sans')

plt.rcParams.update({
    'font.size': 7, 'font.family': _family, 'mathtext.fontset': 'dejavusans',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.titlecolor': INK,
    'axes.titlesize': 7.5, 'axes.titleweight': 'normal', 'axes.titlepad': 4,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.labelpad': 2.5, 'legend.fontsize': 6.5, 'legend.frameon': False,
    'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'lines.solid_capstyle': 'round',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.handlelength': 1.3,
    'legend.handletextpad': 0.5, 'legend.columnspacing': 1.0,
})


def _panel(ax, letter, dx=-0.16, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='right', color=INK, family=_family)


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


def _hd_legend(fig, ncol=4, y=0.0):
    """One shared head-direction key, placed below the figure so it never covers data."""
    handles = [plt.Line2D([0], [0], color=HD_COLOR[h], marker='o', ms=4, lw=1.4) for h in HD_ORDER]
    fig.legend(handles, HD_ORDER, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, y),
               title='head-direction encoding', title_fontsize=6.5, frameon=False)


# ============================================================ panel drawers (one axis each)
def _draw_srsa(ax, rows, conds):
    x = np.arange(len(HD_ORDER)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m, e = zip(*[_mean_sem([_f(r, 'srsa_e') for r in rows
                    if r['condition'] == c and r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=1.5,
               color=COND_SHADE.get(c, '#888'), edgecolor='k', lw=0.4,
               error_kw={'lw': 0.6}, label=COND_CN.get(c, c))
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER); ax.set_ylabel('spatial RSA')
    ax.set_ylim(0, 0.85)


def _draw_crossseed(ax, rows, conds):
    x = np.arange(len(HD_ORDER)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m, e = zip(*[_mean_sem([_f(r, 'cross_seed_rho') for r in rows
                    if r['condition'] == c and r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=1.5,
               color=COND_SHADE.get(c, '#888'), edgecolor='k', lw=0.4,
               error_kw={'lw': 0.6}, label=COND_CN.get(c, c))
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER); ax.set_ylabel('cross-seed correlation')
    ax.set_ylim(0.9, 1.0)          # zoom: the reproducibility is otherwise invisible


def _draw_fieldcount(ax, rows, conds):
    x = np.arange(len(HD_ORDER)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m, e = zip(*[_mean_sem([_f(r, 'mean_fields') for r in rows
                    if r['condition'] == c and r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=1.5,
               color=COND_SHADE.get(c, '#888'), edgecolor='k', lw=0.4,
               error_kw={'lw': 0.6}, label=COND_CN.get(c, c))
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER); ax.set_ylabel('place fields per unit')
    ax.set_ylim(0, 3.8)


def _draw_symidx(ax, rows, conds):
    x = np.arange(len(HD_ORDER)); w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m, e = zip(*[_mean_sem([_f(r, 'sym_c2') for r in rows
                    if r['condition'] == c and r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, m, w, yerr=e, capsize=1.5,
               color=COND_SHADE.get(c, '#888'), edgecolor='k', lw=0.4,
               error_kw={'lw': 0.6}, label=COND_CN.get(c, c))
    ax.axhline(0, lw=0.5, color='k', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER)
    ax.set_ylabel('rate-map $C_2$ symmetry'); ax.set_ylim(-0.1, 1.05)


def _draw_phase_horizon(ax, phase, ks):
    for hd in HD_ORDER:
        m, e = zip(*[_mean_sem([_f(r, 'phase_acc') for r in phase[k] if r['hd_mode'] == hd])
                     for k in ks])
        ax.errorbar(ks, m, yerr=e, marker='o', ms=3.5, lw=1.3, capsize=1.5,
                    color=HD_COLOR[hd], elinewidth=0.6)
    ax.axhline(0.5, ls='--', lw=0.6, color='k', alpha=0.5)
    ax.text(ks[0], 0.51, 'chance', fontsize=6, ha='left', va='bottom', color='k', alpha=0.6)
    ax.set_xlabel('prediction horizon $k$'); ax.set_ylabel('orbit-phase accuracy')
    ax.set_ylim(0.45, 1.02); ax.set_xticks(ks)


def _draw_replay_horizon(ax, rep, ks):
    for hd in HD_ORDER:
        vals = []
        for k in ks:
            rows = [r for r in rep[k] if r['hd_mode'] == hd]
            ratio = [_f(r, 'off_cov') / _f(r, 'wake_cov') for r in rows
                     if np.isfinite(_f(r, 'wake_cov')) and _f(r, 'wake_cov') > 0]
            vals.append(_mean_sem(ratio))
        m, e = zip(*vals)
        ax.errorbar(ks, m, yerr=e, marker='o', ms=3.5, lw=1.3, capsize=1.5,
                    color=HD_COLOR[hd], elinewidth=0.6)
    ax.axhline(1.0, ls='--', lw=0.6, color='k', alpha=0.5)
    ax.text(ks[-1], 1.02, 'wake', fontsize=6, ha='right', va='bottom', color='k', alpha=0.6)
    ax.set_xlabel('prediction horizon $k$')
    ax.set_ylabel('offline / wake coverage'); ax.set_xticks(ks)


def _draw_compartments(ax, rows, modes):
    metrics = [('repetition', 'repeat'), ('room_gen', 'room\n(gen)'),
               ('room_seen', 'room\n(seen)'), ('within_r2', 'within\n$R^2$')]
    x = np.arange(len(metrics)); w = 0.36
    for i, mode in enumerate(modes):
        sub = [r for r in rows if r['mode'] == mode]
        m, e = zip(*[_mean_sem([_f(r, col) for r in sub]) for col, _ in metrics])
        xpos = x + (i - 0.5) * w
        ax.bar(xpos, m, w, yerr=e, capsize=2, color=TWO[mode], edgecolor='k', lw=0.4,
               error_kw={'lw': 0.6}, label=mode)
        for j, (col, _) in enumerate(metrics):
            pts = [_f(r, col) for r in sub]
            ax.scatter(np.full(len(pts), xpos[j]), pts, s=5, color='k', alpha=0.5, zorder=3)
    ax.axhline(0.5, ls='--', lw=0.6, color='k', alpha=0.5)
    ax.axhline(0.0, lw=0.5, color='k')
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics], fontsize=5.5)
    ax.set_ylabel('index / accuracy / $R^2$'); ax.set_ylim(-0.3, 1.1)
    ax.legend(loc='lower center', ncol=2, fontsize=5.5, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.8)


# ================================================================= composite figures
def fig1_setup(data: Path, figs: Path):
    """Fig 1: the manipulation (encoding matrices) and the phenotype (place fields folding)."""
    npz = data / 'rate_maps.npz'
    if not npz.exists():
        print('  skip fig1: missing rate_maps.npz'); return
    d = np.load(npz)
    H = ['E', 'S', 'W', 'N']
    mats = {'full': np.eye(4),
            'parity': np.array([[.5, .5, 0, 0], [.5, .5, 0, 0], [0, 0, .5, .5], [0, 0, .5, .5]]),
            'axis': np.array([[.5, 0, .5, 0], [0, .5, 0, .5], [.5, 0, .5, 0], [0, .5, 0, .5]]),
            'const': np.full((4, 4), .25)}
    bits = {'full': '2 bits', 'parity': '1 bit', 'axis': '1 bit', 'const': '0 bits'}
    inv = {'full': 'none', 'parity': 'none', 'axis': '$C_2$', 'const': '$C_2,C_4$'}
    rows_pc = [('s1_full', '$C_1$, full', 'clean'), ('s2_full', '$C_2$, full', 'clean'),
               ('s2_axis', '$C_2$, axis', 'c2'), ('s4_const', '$C_4$, const', 'c4')]
    rows_pc = [r for r in rows_pc if f'{r[0]}__maps' in d.files]
    ncol = 5

    fig = plt.figure(figsize=(WIDE, 5.9))
    gs = GridSpec(2, 1, height_ratios=[1.0, 2.9], hspace=0.32, figure=fig)
    # -- panel a: encoding matrices + shared colorbar
    gsa = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], width_ratios=[1, 1, 1, 1, 0.08],
                                  wspace=0.35)
    for j, hd in enumerate(HD_ORDER):
        ax = fig.add_subplot(gsa[j])
        im = ax.imshow(mats[hd], cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(H, fontsize=6); ax.set_yticklabels(H if j == 0 else [], fontsize=6)
        ax.set_title(hd, color=HD_COLOR[hd], fontweight='bold', fontsize=7.5, pad=2)
        ax.set_xlabel(f'{bits[hd]} | inv: {inv[hd]}', fontsize=6)
        for s in ax.spines.values():
            s.set_edgecolor('#999999'); s.set_linewidth(0.5)
        if j == 0:
            _panel(ax, 'a', dx=-0.45, dy=1.18)
    cax = fig.add_subplot(gsa[4]); fig.colorbar(im, cax=cax); cax.tick_params(labelsize=5.5)
    cax.set_ylabel('weight', fontsize=6)
    # -- panel b: rate-map grid, magma colorbar
    gsb = GridSpecFromSubplotSpec(len(rows_pc), ncol + 1, subplot_spec=gs[1],
                                  width_ratios=[1] * ncol + [0.10], hspace=0.12, wspace=0.08)
    im2 = None
    for i, (key, label, kind) in enumerate(rows_pc):
        maps, occ = d[f'{key}__maps'], d[f'{key}__occ']
        mask = occ == 0; pick = _select(maps, kind, ncol)
        for j in range(ncol):
            ax = fig.add_subplot(gsb[i, j])
            m = _smooth(maps[pick[j]].astype(float), mask)
            m = m / (m.max() if m.max() > 0 else 1.0)
            im2 = ax.imshow(np.ma.array(m, mask=mask), cmap=RATE_CMAP, origin='lower',
                            vmin=0, vmax=1, interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=6.5, rotation=0, ha='right', va='center', labelpad=12)
            if i == 0 and j == 0:
                _panel(ax, 'b', dx=-1.1, dy=1.12)
    cax2 = fig.add_subplot(gsb[:, ncol]); fig.colorbar(im2, cax=cax2)
    cax2.tick_params(labelsize=5.5); cax2.set_ylabel('normalised rate', fontsize=6)
    out = figs / 'fig1_setup.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig2_fold(data: Path, figs: Path):
    """Fig 2 (2x2): four quantitative read-outs of the fold."""
    fp = data / 'field_stats.csv'; mq = data / 'map_quality_groupB.csv'
    if not (fp.exists() and mq.exists()):
        print('  skip fig2: missing field_stats/map_quality'); return
    fr = _read(fp); mr = _read(mq)
    fconds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in fr)]
    mconds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in mr)]
    fig, ax = plt.subplots(2, 2, figsize=(WIDE, 4.6))
    _draw_fieldcount(ax[0, 0], fr, fconds); _panel(ax[0, 0], 'a')
    _draw_symidx(ax[0, 1], fr, fconds); _panel(ax[0, 1], 'b')
    _draw_srsa(ax[1, 0], mr, mconds); _panel(ax[1, 0], 'c')
    _draw_crossseed(ax[1, 1], mr, mconds); _panel(ax[1, 1], 'd')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_SHADE[c]) for c in fconds]
    fig.legend(handles, [COND_CN[c] for c in fconds], loc='lower center', ncol=3,
               title='arena', title_fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = figs / 'fig2_fold.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig3_function(data: Path, figs: Path):
    """Fig 3: functional consequences -- symmetry resolution, replay, compartment repetition."""
    ph = {0: 'phase_horizon_k0', 1: 'phase_horizon_k1', 3: 'phase_horizon_k3'}
    phase = {}
    for k, name in ph.items():
        p = data / f'{name}.csv'
        if not p.exists():
            print('  skip fig3: missing horizon'); return
        phase[k] = _read(p)
    gb = data / 'phase_groupB.csv'
    if gb.exists():
        phase[5] = [r for r in _read(gb) if r['condition'] == 's2' and r['k'] == '5']
    rep = {k: [r for r in _read(data / f'replay_k{k}.csv') if r['condition'] == 's2']
           for k in (0, 1, 3, 5) if (data / f'replay_k{k}.csv').exists()}
    comp = _read(data / 'compartments.csv') if (data / 'compartments.csv').exists() else None

    fig = plt.figure(figsize=(WIDE, 3.2))
    gs = GridSpec(1, 3, wspace=0.42, figure=fig)
    axA = fig.add_subplot(gs[0]); _draw_phase_horizon(axA, phase, sorted(phase)); _panel(axA, 'a')
    axB = fig.add_subplot(gs[1]); _draw_replay_horizon(axB, rep, sorted(rep)); _panel(axB, 'b')
    if comp is not None:
        modes = [m for m in ('translation', 'rotation') if any(r['mode'] == m for r in comp)]
        axC = fig.add_subplot(gs[2]); _draw_compartments(axC, comp, modes); _panel(axC, 'c')
    fig.subplots_adjust(bottom=0.30, top=0.90, left=0.07, right=0.985, wspace=0.55)
    _hd_legend(fig, ncol=4, y=0.03)
    out = figs / 'fig3_function.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig4_geometry(data: Path, figs: Path):
    """Fig 4: population manifolds of the topology arenas (PCA-3D, coloured by arena angle)."""
    npz = data / 'manifold_pc.npz'
    if not npz.exists():
        print('  skip fig4: missing manifold_pc.npz'); return
    d = np.load(npz)
    order = ['open', 'annulus', 'theta', 'figure8']
    labels = {'open': 'open ($b_1{=}0$)', 'annulus': 'annulus ($b_1{=}1$)',
              'theta': 'theta ($b_1{=}2$)', 'figure8': 'figure-8 ($b_1{=}2$)'}
    lays = [l for l in order if f'{l}__Y' in d.files]
    fig = plt.figure(figsize=(WIDE, 2.8))
    gs = GridSpec(1, len(lays) + 1, width_ratios=[1] * len(lays) + [0.32], wspace=0.28, figure=fig)
    for i, lay in enumerate(lays):
        Y, ang, evr = d[f'{lay}__Y'], d[f'{lay}__angle'], d[f'{lay}__evr']
        ax = fig.add_subplot(gs[i], projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=ang, cmap=ANG_CMAP, s=6, depthshade=False,
                   edgecolors='none', vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'{labels.get(lay, lay)}\n{100 * float(evr[:3].sum()):.0f}% var', fontsize=6.5, pad=-1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel('PC1', fontsize=5.5, labelpad=-12); ax.set_ylabel('PC2', fontsize=5.5, labelpad=-12)
        ax.set_zlabel('PC3', fontsize=5.5, labelpad=-12)
        for p in (ax.xaxis, ax.yaxis, ax.zaxis):
            p.pane.set_edgecolor('#dddddd'); p.pane.set_alpha(0.25)
        ax.grid(False)
    # angular colour wheel key
    caw = fig.add_subplot(gs[len(lays)], projection='polar')
    th = np.linspace(-np.pi, np.pi, 256)
    caw.scatter(th, np.ones_like(th), c=th, cmap=ANG_CMAP, s=3, vmin=-np.pi, vmax=np.pi)
    caw.set_yticks([]); caw.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    caw.set_xticklabels([]); caw.set_title('angle', fontsize=6, pad=1); caw.set_ylim(0, 1.2)
    caw.spines['polar'].set_visible(False)
    out = figs / 'fig4_geometry.pdf'; fig.savefig(out, dpi=350); plt.close(fig)
    print(f'  wrote {out.name}')


# ===================================================================== helpers for rate maps
def _border_frac(m, ring=2):
    tot = m.sum()
    return 1.0 if tot <= 0 else 1.0 - m[ring:-ring, ring:-ring].sum() / tot


def _fold_corr(m, order):
    a = (m - m.mean()).ravel()
    js = (2,) if order == 2 else (1, 2, 3)
    return float(np.mean([np.corrcoef(a, (np.rot90(m, j) - m.mean()).ravel())[0, 1] for j in js]))


def _smooth(m, mask, sigma=0.9):
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return m
    num = gaussian_filter(np.where(mask, 0.0, m), sigma)
    den = gaussian_filter((~mask).astype(float), sigma)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)


def _select(maps, kind, n):
    idx = list(range(len(maps)))
    if kind == 'clean':
        cand = [i for i in idx if _border_frac(maps[i]) < 0.45] or idx
        return cand[:n]
    order = 2 if kind == 'c2' else 4
    cand = [i for i in idx if _border_frac(maps[i]) < 0.65] or idx
    cand.sort(key=lambda i: -_fold_corr(maps[i], order))
    return cand[:n]


# ===================================================================== supplementary figures
def figS_init(data: Path, figs: Path):
    frames = []
    for name in ('speed_init', 'speed_kaiming'):
        p = data / f'{name}.csv'
        if p.exists():
            frames += _read(p)
    if not frames:
        print('  skip figS_init: missing speed_*'); return
    nice = {'zero_rec': 'pure leak', 'gain_lo': 'low gain', 'gain_hi': 'high gain',
            'tau1': r'$\tau{=}1$', 'tau4': r'$\tau{=}4$', 'tau8': r'$\tau{=}8$',
            'orth': 'orthogonal', 'baseline': 'default', 'kaiming': 'Kaiming',
            'kaiming_noid': 'Kaiming, no leak'}
    variants = sorted({r['variant'] for r in frames},
                      key=lambda v: np.mean([_f(r, 'loss_auc') for r in frames if r['variant'] == v]))
    fig, ax = plt.subplots(figsize=(ONEHALF, 2.6))
    for i, v in enumerate(variants):
        sub = [r for r in frames if r['variant'] == v]; vals = [_f(r, 'loss_auc') for r in sub]
        m, e = _mean_sem(vals)
        col = '#D55E00' if v.startswith('kaiming') else ('#0072B2' if v == 'baseline' else '#7F7F7F')
        ax.bar(i, m, 0.7, yerr=e, capsize=2, color=col, edgecolor='k', lw=0.4, error_kw={'lw': 0.6})
        ax.scatter(np.full(len(vals), i), vals, s=6, color='k', alpha=0.5, zorder=3)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([nice.get(v, v) for v in variants], rotation=40, ha='right', fontsize=6)
    ax.set_ylabel('loss AUC over training')
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#0072B2', '#7F7F7F', '#D55E00')]
    ax.legend(h, ['default', 'other', 'Kaiming'], fontsize=6, loc='upper left')
    fig.tight_layout()
    out = figs / 'figS_init.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def figS_units(data: Path, figs: Path):
    npz = data / 'rate_maps.npz'
    if not npz.exists():
        print('  skip figS_units: missing rate_maps.npz'); return
    d = np.load(npz)
    order = [('s1_full', '$C_1$, full HD'), ('s2_full', '$C_2$, full HD'),
             ('s2_axis', '$C_2$, axis HD'), ('s4_const', '$C_4$, const HD')]
    for key, label in order:
        if f'{key}__maps' not in d.files:
            continue
        maps, occ = d[f'{key}__maps'], d[f'{key}__occ']; mask = occ == 0
        n = min(len(maps), 40); ncol = 8; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(WIDE, 0.9 * nrow + 0.4),
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        for k, ax in enumerate(axes.flat):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k >= n:
                ax.axis('off'); continue
            m = _smooth(maps[k].astype(float), mask); m = m / (m.max() if m.max() > 0 else 1.0)
            ax.imshow(np.ma.array(m, mask=mask), cmap=RATE_CMAP, origin='lower', vmin=0, vmax=1,
                      interpolation='bilinear')
        fig.suptitle(f'{label}: {n} most spatially informative units', fontsize=8, y=0.995)
        out = figs / f'figS_units_{key}.pdf'; fig.savefig(out); plt.close(fig)
        print(f'  wrote {out.name}')


def figS_celltypes(data: Path, figs: Path):
    p = data / 'cell_types.csv'
    if not p.exists():
        print('  skip figS_celltypes: missing cell_types.csv'); return
    rows = _read(p)
    conds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in rows)]
    fig, ax = plt.subplots(1, 2, figsize=(5.35, 2.5))
    # (a) place- and border-cell fraction by HD encoding, pooled across arenas (composition stable)
    x = np.arange(len(HD_ORDER)); w = 0.38
    for off, key, lab, col in [(-w / 2, 'frac_place', 'place cells', '#1F3B5C'),
                               (w / 2, 'frac_border', 'border-modulated', '#9C4A2F')]:
        m, e = zip(*[_mean_sem([_f(r, key) for r in rows if r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax[0].bar(x + off, m, w, yerr=e, capsize=1.5, color=col, edgecolor='k', lw=0.4,
                  error_kw={'lw': 0.6}, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(HD_ORDER, rotation=20)
    ax[0].set_ylabel('fraction of units'); ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=6, loc='lower left'); _panel(ax[0], 'a')
    # (b) fields per place cell by encoding x arena -- the fold multiplies fields
    ww = 0.8 / len(conds)
    for i, c in enumerate(conds):
        m, e = zip(*[_mean_sem([_f(r, 'mean_fields_place') for r in rows
                    if r['condition'] == c and r['hd_mode'] == hd]) for hd in HD_ORDER])
        ax[1].bar(x + (i - (len(conds) - 1) / 2) * ww, m, ww, yerr=e, capsize=1.5,
                  color=COND_SHADE.get(c, '#888'), edgecolor='k', lw=0.4,
                  error_kw={'lw': 0.6}, label=COND_CN.get(c, c))
    ax[1].set_xticks(x); ax[1].set_xticklabels(HD_ORDER, rotation=20)
    ax[1].set_ylabel('fields per place cell'); ax[1].legend(fontsize=6, loc='upper left')
    _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'figS_celltypes.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def _phase(data, name):
    p = data / f'{name}.csv'
    return _read(p) if p.exists() else []


def _acc(rows, cond, hd):
    return _mean_sem([_f(r, 'phase_acc') for r in rows
                      if r['condition'] == cond and r['hd_mode'] == hd])


def fig5_generality(data: Path, figs: Path):
    """Generality of the fold: arena size, hidden size, HD-lesion dose, learned compass."""
    base = _phase(data, 'phase_full_n10')          # baseline: hidden 500, arena 18, noise 0
    fig, ax = plt.subplots(2, 2, figsize=(WIDE, 4.7))

    def _sweep(a_, points, xlabel, marker, hds=('axis', 'parity')):
        for hd in hds:
            xs, ms, es = [], [], []
            for x, nm in points:
                rows = base if nm is None else _phase(data, nm)
                if not rows:
                    continue
                m, e = _acc(rows, 's2', hd)
                if np.isfinite(m):
                    xs.append(x); ms.append(m); es.append(e)
            if xs:
                a_.errorbar(xs, ms, yerr=es, marker=marker, ms=4, color=HD_COLOR[hd],
                            label=hd, capsize=2, lw=1.3, elinewidth=0.7)
        a_.axhline(0.5, ls='--', color='#999', lw=0.8)
        a_.set_xlabel(xlabel); a_.set_ylabel('orbit-phase accuracy'); a_.set_ylim(0.45, 1.02)

    _sweep(ax[0, 0], [(12, 'phase_a12'), (18, None), (24, 'phase_a24'), (30, 'phase_a30')],
           'arena size', 'o')
    _panel(ax[0, 0], 'a'); ax[0, 0].legend(fontsize=6, loc='center right')
    _sweep(ax[0, 1], [(250, 'phase_h250'), (500, None), (1000, 'phase_h1000')], 'hidden units', 's')
    _panel(ax[0, 1], 'b')
    # (c) HD-lesion dose-response, all encodings
    levels = [(0.0, None), (0.15, 'phase_noise015'), (0.30, 'phase_noisy'),
              (0.50, 'phase_noise050'), (0.70, 'phase_noise070')]
    for hd in HD_ORDER:
        xs, ms, es = [], [], []
        for nz, nm in levels:
            rows = base if nm is None else _phase(data, nm)
            if not rows:
                continue
            m, e = _acc(rows, 's2', hd)
            if np.isfinite(m):
                xs.append(nz); ms.append(m); es.append(e)
        if xs:
            ax[1, 0].errorbar(xs, ms, yerr=es, marker='o', ms=4, color=HD_COLOR[hd],
                              label=hd, capsize=2, lw=1.3, elinewidth=0.7)
    ax[1, 0].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[1, 0].set_xlabel('head-direction corruption'); ax[1, 0].set_ylabel('orbit-phase accuracy')
    ax[1, 0].set_ylim(0.45, 1.02); _panel(ax[1, 0], 'c'); ax[1, 0].legend(fontsize=6, ncol=2)
    # (d) learned (angular-velocity) compass, by arena
    lrn = _phase(data, 'phase_learned_c2')
    conds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in lrn)]
    ms, es = zip(*[_mean_sem([_f(r, 'phase_acc') for r in lrn if r['condition'] == c])
                   for c in conds]) if conds else ([], [])
    ax[1, 1].bar(range(len(conds)), ms, 0.62, yerr=es, capsize=2,
                 color=[COND_SHADE[c] for c in conds], edgecolor='k', lw=0.4, error_kw={'lw': 0.7})
    ax[1, 1].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[1, 1].set_xticks(range(len(conds))); ax[1, 1].set_xticklabels([COND_CN[c] for c in conds])
    ax[1, 1].set_ylabel('orbit-phase accuracy'); ax[1, 1].set_ylim(0.45, 1.02)
    ax[1, 1].set_title('learned compass', fontsize=7); _panel(ax[1, 1], 'd')
    fig.tight_layout()
    out = figs / 'fig5_generality.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig6_brain(data: Path, figs: Path):
    """To the brain: 4-room repetition, real CA1 directional correlation, city-block model."""
    fig, ax = plt.subplots(1, 3, figsize=(WIDE, 2.7))
    # (a) four-room Spiers
    c4 = _read(data / 'compartments4.csv') if (data / 'compartments4.csv').exists() else []
    mets = [('room_gen', 'room\ndecode'), ('room_seen', 'room\nkNN'),
            ('repetition', 'field\nrepeat'), ('within_r2', 'within\n$R^2$')]
    if c4:
        ms, es = zip(*[_mean_sem([_f(r, k) for r in c4]) for k, _ in mets])
        ax[0].bar(range(4), ms, 0.66, yerr=es, capsize=2,
                  color=['#1F3B5C', '#1F3B5C', '#3A6B6B', '#9C4A2F'], edgecolor='k', lw=0.4,
                  error_kw={'lw': 0.7})
        ax[0].axhline(0.25, ls='--', color='#999', lw=0.8)
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels([m[1] for m in mets], fontsize=6)
        ax[0].set_ylabel('value'); ax[0].set_ylim(-0.1, 1.05)
    ax[0].set_title('four-room maze (model)', fontsize=7); _panel(ax[0], 'a')
    # (b) real CA1 directional-index correlation, same-orientation field pairs
    fd = _read(data / 'hockeimer_field_di.csv') if (data / 'hockeimer_field_di.csv').exists() else []
    xs, ys = [], []
    if fd:
        by = {}
        for r in fd:
            if r['rep'] in ('True', 'TRUE', True):
                by.setdefault((r['cell'], r['orient']), []).append(_f(r, 'di'))
        for v in by.values():
            for i in range(len(v)):
                for j in range(len(v)):
                    if i != j:
                        xs.append(v[i]); ys.append(v[j])
    if xs:
        ax[1].scatter(xs, ys, s=7, color='#1F3B5C', alpha=0.4, edgecolors='none')
        xr = np.array([-1, 1]); b = np.polyfit(xs, ys, 1)
        ax[1].plot(xr, b[0] * xr + b[1], color='#9C4A2F', lw=1.5)
        r = np.corrcoef(xs, ys)[0, 1]
        ax[1].text(0.05, 0.92, f'$r={r:.2f}$', transform=ax[1].transAxes, fontsize=7)
        ax[1].set_xlim(-1, 1); ax[1].set_ylim(-1, 1)
        ax[1].set_xlabel('field $i$ directional index'); ax[1].set_ylabel('field $j$ directional index')
    ax[1].set_title('real CA1 (Hockeimer)', fontsize=7); _panel(ax[1], 'b')
    # (c) city-block model directional correlation (if available)
    cb = _read(data / 'cityblock.csv') if (data / 'cityblock.csv').exists() else []
    cx, cy = [], []
    if cb:
        by = {}
        for r in cb:
            by.setdefault((r['seed'], r['unit'], r['orient']), []).append(_f(r, 'di'))
        for v in by.values():
            for i in range(len(v)):
                for j in range(len(v)):
                    if i != j:
                        cx.append(v[i]); cy.append(v[j])
    if cx:
        ax[2].scatter(cx, cy, s=6, color='#3A6B6B', alpha=0.35, edgecolors='none')
        xr = np.array([-1, 1]); b = np.polyfit(cx, cy, 1)
        ax[2].plot(xr, b[0] * xr + b[1], color='#9C4A2F', lw=1.5)
        rc = np.corrcoef(cx, cy)[0, 1]
        ax[2].text(0.05, 0.92, f'$r={rc:.2f}$', transform=ax[2].transAxes, fontsize=7)
        ax[2].set_xlim(-1, 1); ax[2].set_ylim(-1, 1); ax[2].set_xlabel('field $i$ DI'); ax[2].set_ylabel('field $j$ DI')
    else:
        ax[2].text(0.5, 0.5, 'city-block\n(training)', ha='center', va='center',
                   transform=ax[2].transAxes, fontsize=7, color='#999')
        ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title('city-block model', fontsize=7); _panel(ax[2], 'c')
    fig.tight_layout()
    out = figs / 'fig6_brain.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--figs', required=True)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    data, figs = Path(a.data), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    allfigs = {'fig1': fig1_setup, 'fig2': fig2_fold, 'fig3': fig3_function,
               'fig4': fig4_geometry, 'fig5': fig5_generality, 'fig6': fig6_brain,
               'init': figS_init, 'units': figS_units, 'celltypes': figS_celltypes}
    for name, fn in allfigs.items():
        if a.only and name not in a.only:
            continue
        print(f'[{name}]')
        fn(data, figs)


if __name__ == '__main__':
    main()
