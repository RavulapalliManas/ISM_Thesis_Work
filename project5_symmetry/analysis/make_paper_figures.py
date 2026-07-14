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
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # noqa: E402
from project5_symmetry.environments.arena import PixelObsWrapper, SymmetryArena  # noqa: E402
from project5_symmetry.environments.compartment4 import Compartment4Arena  # noqa: E402
from project5_symmetry.environments.compartment_arenas import CompartmentArena  # noqa: E402

# Muted, formal palette (still colourblind-distinguishable), fixed per HD encoding everywhere.
HD_COLOR = {'full': '#1F3B5C', 'parity': '#3A6B6B', 'axis': '#9C4A2F', 'const': '#8A8A8A'}
HD_ORDER = ['full', 'parity', 'axis', 'const']
TWO = {'translation': '#1F3B5C', 'rotation': '#9C4A2F'}
COND_CN = {'s1': '$C_1$', 's2': '$C_2$', 's4': '$C_4$'}       # arena naming, everywhere
# fig1_overview only: card-style rendering (soft shadow, floor texture) for the schematic
# panels. Foreground colours reuse TWO/HD_COLOR above; these are the supporting greys.
OV_GREY_FLOOR = '#EDEDED'
OV_GRID_LINE = '#DBDBDB'
OV_WALL = '#6A6A6A'
OV_LABEL_GREY = '#8A8A8A'
OV_ARROW_GREY = '#767676'
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
    # No savefig.bbox='tight': it CROPS the output and silently changes the figure width,
    # which defeats the exact column sizing below. fig.tight_layout() already packs the
    # axes inside the declared figsize, so the exported width is the width we asked for.
    'figure.dpi': 150, 'savefig.dpi': 400,
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
def _boot_ci(v, n_boot=2000, seed=0):
    """Mean and 95% bootstrap CI over networks. The unit of observation is the network, so the
    interval must be computed over networks, not over pooled units."""
    v = np.asarray([t for t in v if np.isfinite(t)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _draw_dots(ax, rows, conds, key, ylabel, ylim=None, zero_line=False):
    """One dot per network, with the mean and a 95% bootstrap CI.

    Replaces the bar+SEM design these panels used to have. Bars summarising ~10 networks hide the
    distribution, SEM understates uncertainty at that n, and a bar drawn on a non-zero baseline
    (cross-seed correlation was zoomed to 0.9-1.0) misstates the effect size outright. Dots carry
    no baseline claim, so the zoom is legitimate once the bar is gone.
    """
    x = np.arange(len(HD_ORDER))
    w = 0.8 / len(conds)
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * w
        for j, hd in enumerate(HD_ORDER):
            v = [_f(r, key) for r in rows
                 if r['condition'] == c and r['hd_mode'] == hd]
            v = [t for t in v if np.isfinite(t)]
            if not v:
                continue
            xx = x[j] + off
            jit = (rng.random(len(v)) - 0.5) * w * 0.5
            ax.scatter(xx + jit, v, s=3.0, color=COND_SHADE.get(c, '#888'),
                       edgecolor='k', linewidths=0.2, zorder=3, alpha=0.95,
                       label=COND_CN.get(c, c) if j == 0 else None)
            m, lo, hi = _boot_ci(v)
            ax.plot([xx - w * 0.32, xx + w * 0.32], [m, m], color='k', lw=1.0, zorder=4)
            ax.plot([xx, xx], [lo, hi], color='k', lw=0.7, zorder=4)
    if zero_line:
        ax.axhline(0, lw=0.5, color='k', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(HD_ORDER)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _draw_srsa(ax, rows, conds):
    _draw_dots(ax, rows, conds, 'srsa_e', 'spatial RSA', (0, 0.85))


def _draw_crossseed(ax, rows, conds):
    _draw_dots(ax, rows, conds, 'cross_seed_rho', 'cross-seed correlation', (0.9, 1.0))


def _draw_fieldcount(ax, rows, conds):
    _draw_dots(ax, rows, conds, 'mean_fields', 'place fields per unit', (0, 3.8))


def _draw_symidx(ax, rows, conds):
    _draw_dots(ax, rows, conds, 'sym_c2', 'rate-map $C_2$ symmetry', (-0.1, 1.05),
               zero_line=True)


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


# ======================================================= fig1_overview panel drawers
def _ov_mute(hexcolor, amt=0.42):
    """Blend a saturated colour 42% toward white -- softer than the raw landmark RGB."""
    hexcolor = hexcolor.lstrip('#')
    a = tuple(int(hexcolor[i:i + 2], 16) for i in (0, 2, 4))
    m = tuple(round(x * (1 - amt) + 255 * amt) for x in a)
    return f'#{m[0]:02x}{m[1]:02x}{m[2]:02x}'


_OV_DISPLAY = {
    (0.0, 0.0, 0.45): _ov_mute('2255AA'), (0.45, 0.0, 0.0): _ov_mute('B23A2E'),
    (0.45, 0.45, 0.0): _ov_mute('C79A1E'), (0.0, 0.35, 0.0): _ov_mute('3A7D3A'),
}


def _ov_card_shadow(ax, x0, y0, w, h, n_layers=14, dx=0.30, dy=-0.34, a=0.022, zorder=-1):
    for k in range(n_layers, 0, -1):
        f = k / n_layers
        ax.add_patch(Rectangle((x0 + dx * f, y0 + dy * f), w, h, facecolor='#000000',
                                edgecolor='none', alpha=a, zorder=zorder))


def _ov_circle_shadow(ax, cx, cy, r, n_layers=14, dx=0.10, dy=-0.12, a=0.024, zorder=-1):
    """Same soft-shadow treatment as _ov_card_shadow but circular -- a square shadow behind
    a round object shows its corners past the object's edge in every renderer, not a bug
    specific to one (confirmed: poppler just renders it sharply enough to be obvious)."""
    for k in range(n_layers, 0, -1):
        f = k / n_layers
        ax.add_patch(Circle((cx + dx * f, cy + dy * f), r, facecolor='#000000',
                             edgecolor='none', alpha=a, zorder=zorder))


def _ov_rotation_icon(ax, cx, cy, size, label, color=TWO['translation'], label_dy=1.1):
    """Standard clockwise-rotation glyph -- reused for panels b and d."""
    ax.text(cx, cy, '⟳', fontsize=size, color=color, ha='center', va='center',
            family='DejaVu Sans', zorder=6)
    ax.text(cx, cy - label_dy, label, fontsize=6.6, color=color, ha='center', va='top')


# ---- panel a: pipeline (agent -> egocentric patch -> pRNN -> prediction) ----------
def _ov_pipeline_data():
    N = 18
    env = SymmetryArena(shape='square', size=N, U=4, symmetry_condition='s4',
                         use_landmarks=True, F=7)
    penv = PixelObsWrapper(env)
    obs, _ = penv.reset()
    patch_t0 = obs['image'].copy()
    path = [tuple(env.agent_pos)]
    for a in (2, 2, 2):
        obs, *_ = penv.step(a)
        path.append(tuple(env.agent_pos))
    patch_t3 = obs['image'].copy()
    return N, env._get_landmark_tiles(), path, patch_t0, patch_t3


def _ov_draw_arena_agent(ax, N, tiles, path):
    _ov_card_shadow(ax, 0, 0, N, N, dx=0.35, dy=-0.42)
    ax.add_patch(Rectangle((0, 0), N, N, facecolor=OV_GREY_FLOOR, edgecolor='none', zorder=0))
    for (r, c), rgb in tiles.items():
        color = _OV_DISPLAY.get(tuple(rgb), '#999999')
        ax.add_patch(Rectangle((c - 1, N - r), 1, 1, facecolor=color, edgecolor='none', zorder=1))
    for i in range(N + 1):
        ax.plot([i, i], [0, N], color=OV_GRID_LINE, lw=0.35, zorder=0.5)
        ax.plot([0, N], [i, i], color=OV_GRID_LINE, lw=0.35, zorder=0.5)
    ax.add_patch(Rectangle((0, 0), N, N, facecolor='none', edgecolor=OV_WALL, lw=1.1, zorder=4))

    xs = [p[0] + 0.5 for p in path]
    ys = [N - p[1] - 0.5 for p in path]
    ax.plot(xs, ys, color=TWO['translation'], lw=1.1, zorder=5, solid_capstyle='round', alpha=0.85)
    for x, y in zip(xs[:-1], ys[:-1]):
        ax.add_patch(Circle((x, y), 0.11, facecolor=TWO['translation'], edgecolor='none',
                             zorder=5, alpha=0.85))

    vx0, vy0 = xs[-1] - 3.5, ys[-1]
    ax.add_patch(Rectangle((vx0, vy0), 7, 3.6, facecolor='#F2D98A', edgecolor='none',
                            alpha=0.38, zorder=3))
    ax.add_patch(Polygon([(xs[-1], ys[-1] + 0.42), (xs[-1] - 0.30, ys[-1] - 0.24),
                           (xs[-1] + 0.30, ys[-1] - 0.24)], closed=True,
                          facecolor=TWO['translation'], edgecolor='white', lw=0.6, zorder=6))

    ax.set_xlim(-0.7, N + 0.7); ax.set_ylim(-0.7, N + 0.7)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _ov_draw_patch(ax, patch, accent, label):
    w, h = patch.shape[1], patch.shape[0]
    _ov_card_shadow(ax, 0, 0, w, h, dx=0.22, dy=-0.30, a=0.030)
    ax.imshow(patch, interpolation='nearest', extent=(0, w, 0, h), zorder=1)
    ax.add_patch(Rectangle((0, 0), w, h, facecolor='none', edgecolor='#FFFFFF', lw=0.8, zorder=2))
    ax.add_patch(Rectangle((0, h + 0.35), 0.9, 0.16, facecolor=accent, edgecolor='none'))
    ax.set_xlim(-0.3, w + 0.3); ax.set_ylim(-0.4, h + 0.9)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(w / 2, -0.9, label, fontsize=6.6, color=OV_LABEL_GREY, ha='center', va='top')


def _ov_draw_rnn(ax, cx, cy, r):
    _ov_circle_shadow(ax, cx, cy, r, dx=0.10 * r, dy=-0.12 * r, a=0.024, zorder=0)
    # radial gradient via many thin concentric vector circles, not a raster image with an
    # alpha soft-mask -- image clip-paths/soft-masks render inconsistently across PDF
    # interpreters (confirmed: poppler shows a hard square halo where gs/matplotlib don't).
    # Solid-fill circles at flat per-layer alpha are a standard PDF construct every viewer
    # handles the same way.
    n_rings = 44
    for k in range(n_rings, 0, -1):
        f = k / n_rings
        ax.add_patch(Circle((cx, cy), r * f, facecolor=(0.75, 0.80, 0.87), edgecolor='none',
                             alpha=0.028, zorder=1))
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor='#9FB0C2', lw=0.8, zorder=2))

    rng = np.random.default_rng(3)
    rel = [(-0.235, 0.559), (0.088, 0.618), (0.529, 0.324), (0.588, -0.118),
           (0.324, -0.529), (-0.147, -0.588), (-0.559, -0.265), (-0.529, 0.176),
           (0.059, 0.088), (0.0, 0.0), (-0.36, 0.10), (0.30, 0.62), (-0.62, -0.55),
           (0.62, -0.42)]
    sizes = [0.040, 0.030, 0.045, 0.028, 0.038, 0.026, 0.042, 0.030, 0.055, 0.048,
             0.024, 0.020, 0.022, 0.026]
    nodes = [(cx + dx * r, cy + dy * r) for dx, dy in rel]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
             (8, 0), (8, 3), (8, 5), (9, 1), (9, 6), (10, 6), (10, 0),
             (11, 1), (12, 5), (13, 3)]
    for i, j in edges:
        p0, p1 = nodes[i], nodes[j]
        rad = rng.uniform(-0.28, 0.28)
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                                      arrowstyle='-', lw=0.9, color='#4C5F73',
                                      alpha=0.8, zorder=3))
    for (x, y), s in zip(nodes, sizes):
        rad = s * r
        ax.add_patch(Circle((x + rad * 0.28, y - rad * 0.32), rad * 1.05,
                             facecolor='#000000', alpha=0.13, edgecolor='none', zorder=4))
        ax.add_patch(Circle((x, y), rad, facecolor='#5E7089', edgecolor='none', zorder=5))
        ax.add_patch(Circle((x - rad * 0.32, y + rad * 0.32), rad * 0.42,
                             facecolor='#9FB0C2', edgecolor='none', alpha=0.80, zorder=6))


def _ov_draw_actions(ax, heading='N'):
    """The quantity actually fed to the network: the 5-D SpeedHD vector [speed, onehot(h)]
    (Methods), not the raw turn/forward/stop environment actions -- drawn in the same
    greyscale-matrix convention as the encoding matrices in fig1_setup panel a."""
    cols = ['speed', 'E', 'S', 'W', 'N']
    vals = [1.0, 0.0, 0.0, 0.0, 0.0]
    vals[cols.index(heading)] = 1.0
    CELL = 0.62
    x0 = -2.5 * CELL
    _ov_card_shadow(ax, x0, 0, 5 * CELL, CELL * 0.88, n_layers=10, dx=0.05, dy=-0.06, a=0.022)
    for i, (name, v) in enumerate(zip(cols, vals)):
        x = x0 + i * CELL
        shade = 1 - v
        ax.add_patch(Rectangle((x, 0), CELL * 0.88, CELL * 0.88,
                                facecolor=(shade, shade, shade), edgecolor='#9A9A9A', lw=0.5))
        ax.text(x + CELL * 0.44, -0.10, name, fontsize=5.5, color=OV_LABEL_GREY,
                ha='center', va='top', style=('normal' if name == 'speed' else 'italic'))
    ax.set_xlim(x0 - 0.15, x0 + 5 * CELL + 0.15); ax.set_ylim(-0.42, 0.98)
    ax.set_aspect('equal'); ax.axis('off')
    ax.text(0, 0.86, r'$a_t = [\,\mathrm{speed},\ \mathrm{onehot}(h)\,]$', fontsize=6.6,
            color=OV_LABEL_GREY, ha='center', va='bottom')


# ---- panel b: the three arenas, C1/C2/C4, real landmark geometry ------------------
_OV_CONDS = [
    ('s1', '$C_1$ -- no symmetry', None, '4 distinct landmarks'),
    ('s2', '$C_2$ -- 180° rotation', 'top_half', '2 landmarks, ×2 copies each'),
    ('s4', '$C_4$ -- 90° rotation', 'q1', '1 landmark, ×4 copies'),
]


def _ov_domain_path(kind, N):
    if kind is None:
        return []
    if kind == 'top_half':
        return [(1, 1, N // 2, N)]
    if kind == 'q1':
        return [(1, 1, N // 2, N // 2)]
    raise ValueError(kind)


def _ov_draw_symarena(ax, cond, domain_kind, title, N=18):
    arena = SymmetryArena(shape='square', size=N, U=4, symmetry_condition=cond,
                           use_landmarks=True)
    tiles = arena._get_landmark_tiles()

    _ov_card_shadow(ax, 0, 0, N, N)
    ax.add_patch(Rectangle((0, 0), N, N, facecolor=OV_GREY_FLOOR, edgecolor='none', zorder=0))
    for (r, c), rgb in tiles.items():
        color = _OV_DISPLAY.get(tuple(rgb), '#999999')
        ax.add_patch(Rectangle((c - 1, N - r), 1, 1, facecolor=color, edgecolor='none', zorder=1))
    ax.add_patch(Rectangle((0, 0), N, N, facecolor='none', edgecolor=OV_WALL, lw=1.4, zorder=4))

    for (r0, c0, r1, c1) in _ov_domain_path(domain_kind, N):
        x0, y0 = c0 - 1, N - r1
        w, h = (c1 - c0 + 1), (r1 - r0 + 1)
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none', edgecolor=TWO['translation'],
                                lw=1.3, ls=(0, (3, 2)), zorder=5))

    if cond == 's2':
        _ov_rotation_icon(ax, N + 2.3, N - 1.9, 19, '180°')
    elif cond == 's4':
        _ov_rotation_icon(ax, N + 2.3, N - 1.9, 19, '90°')

    ax.set_xlim(-0.6, N + 4.6); ax.set_ylim(-0.6, N + 0.6)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=7, color=INK, pad=5)


# ---- panel c: the animal phenomenon, real compartment-arena geometry --------------
def _ov_bbox(cells):
    rs = [c[0] for c in cells]; cs = [c[1] for c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _ov_draw_footprint(ax, passable, tiles, pad=1):
    r0, r1, c0, c1 = _ov_bbox(passable)
    r0 -= pad; r1 += pad; c0 -= pad; c1 += pad
    W, H = c1 - c0 + 1, r1 - r0 + 1
    _ov_card_shadow(ax, 0, 0, W, H, dx=0.35, dy=-0.4)
    for r, c in passable:
        x, y = c - c0, r1 - r
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=OV_GREY_FLOOR, edgecolor=OV_GRID_LINE,
                                lw=0.3, zorder=0))
    for (r, c), rgb in tiles.items():
        color = _OV_DISPLAY.get(tuple(rgb), '#999999')
        x, y = c - c0, r1 - r
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='none', zorder=1))
    ax.set_xlim(-0.3, W + 0.3); ax.set_ylim(-0.3, H + 0.3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _ov_animal_data():
    a4 = Compartment4Arena()
    a_trans = CompartmentArena('translation')
    a_rot = CompartmentArena('rotation')
    return {
        'spiers': (a4.passable, a4._get_landmark_tiles()),
        'translation': (a_trans.passable, a_trans._get_landmark_tiles()),
        'rotation': (a_rot.passable, a_rot._get_landmark_tiles()),
    }


# ---- panel d: the quotient map X -> X/G, purely geometric -------------------------
_OV_S = 10.0
_OV_REPS = [(1.8, 8.4), (3.6, 6.7), (2.6, 7.6)]
_OV_ORBIT_COLORS = [TWO['translation'], TWO['rotation'], '#3A6B6B']


def _ov_rot4(pt, k, cx, cy):
    x, y = pt[0] - cx, pt[1] - cy
    for _ in range(k):
        x, y = -y, x
    return x + cx, y + cy


def _ov_draw_X(ax):
    S = _OV_S; cx = cy = S / 2
    _ov_card_shadow(ax, 0, 0, S, S, dx=0.10, dy=-0.12, a=0.024)
    ax.add_patch(Rectangle((0, 0), S, S, facecolor=OV_GREY_FLOOR, edgecolor=OV_WALL, lw=1.2))
    for i in range(int(S) + 1):
        ax.plot([i, i], [0, S], color=OV_GRID_LINE, lw=0.3, zorder=0.5)
        ax.plot([0, S], [i, i], color=OV_GRID_LINE, lw=0.3, zorder=0.5)
    ax.plot([cx, cx], [0, S], color='#BBBBBB', lw=0.6, zorder=1)
    ax.plot([0, S], [cy, cy], color='#BBBBBB', lw=0.6, zorder=1)

    for i, rep in enumerate(_OV_REPS):
        color = _OV_ORBIT_COLORS[i]
        pts = [_ov_rot4(rep, k, cx, cy) for k in range(4)]
        radius = np.hypot(rep[0] - cx, rep[1] - cy)
        ax.add_patch(Circle((cx, cy), radius, facecolor='none', edgecolor=color,
                             lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=2))
        a0 = np.arctan2(rep[1] - cy, rep[0] - cx)
        a1 = a0 + np.radians(24)
        p_from = (cx + radius * np.cos(a0 + np.radians(6)), cy + radius * np.sin(a0 + np.radians(6)))
        p_to = (cx + radius * np.cos(a1), cy + radius * np.sin(a1))
        ax.add_patch(FancyArrowPatch(p_from, p_to, connectionstyle='arc3,rad=0.15',
                                      arrowstyle='-|>', mutation_scale=7, lw=1.0,
                                      color=color, alpha=0.85, zorder=3))
        for p in pts:
            ax.add_patch(Circle(p, 0.22, facecolor=color, edgecolor='white', lw=0.4, zorder=4))

    ax.annotate('orbit of $x$:\n$\\{x, gx, g^2x, g^3x\\}$', xy=_OV_REPS[0], xytext=(0.25, 9.9),
                fontsize=6.2, color=_OV_ORBIT_COLORS[0], ha='left', va='top',
                arrowprops=dict(arrowstyle='-', color=_OV_ORBIT_COLORS[0], lw=0.6, alpha=0.7,
                                 shrinkA=0, shrinkB=4, connectionstyle='arc3,rad=-0.15'))
    _ov_rotation_icon(ax, S + 1.6, S - 1.0, 17, '$G=C_4$', label_dy=1.15)

    ax.set_xlim(-0.4, S + 3.2); ax.set_ylim(-0.4, S + 0.6)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('state space $X$', fontsize=7, pad=4)
    ax.text(cx, -1.0, '(every point has 4 physically\ndistinct images under $G$)',
            fontsize=5.6, color=OV_LABEL_GREY, ha='center', va='top')


def _ov_draw_quotient(ax):
    S = _OV_S; cy = S / 2; d = S / 2
    _ov_card_shadow(ax, 0, 0, d, d, dx=0.10, dy=-0.12, a=0.024)
    ax.add_patch(Rectangle((0, 0), d, d, facecolor=OV_GREY_FLOOR, edgecolor=TWO['translation'],
                             lw=1.6, ls=(0, (3, 2))))
    for i in range(int(d) + 1):
        ax.plot([i, i], [0, d], color=OV_GRID_LINE, lw=0.3, zorder=0.5)
        ax.plot([0, d], [i, i], color=OV_GRID_LINE, lw=0.3, zorder=0.5)
    for i, rep in enumerate(_OV_REPS):
        color = _OV_ORBIT_COLORS[i]
        p = (rep[0], rep[1] - cy)
        ax.add_patch(Circle(p, 0.22, facecolor=color, edgecolor='white', lw=0.4, zorder=3))
    ax.annotate('$\\pi(x)$ -- the coset,\nnot the orbit element', xy=(_OV_REPS[0][0], _OV_REPS[0][1] - cy),
                xytext=(0.4, 1.5), fontsize=6.2, color=_OV_ORBIT_COLORS[0], ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=_OV_ORBIT_COLORS[0], lw=0.5, alpha=0.6))
    ax.set_xlim(-0.4, d + 0.4); ax.set_ylim(-0.4, d + 0.6)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('quotient $X/G$', fontsize=7, pad=4)
    ax.text(d / 2, -1.0, '(one point per orbit --\nthe fold)', fontsize=5.6, color=OV_LABEL_GREY,
            ha='center', va='top')


def fig1_overview(data: Path, figs: Path):
    """Fig 1 (NEW, overview): the pipeline (agent -> egocentric patch -> pRNN -> prediction),
    the three arenas with real landmark geometry, the Spiers/Grieves animal phenomenon this
    explains, and the X -> X/G quotient-map cartoon. Purely illustrative/geometric -- every
    arena, patch, and footprint is pulled live from the actual environment classes, nothing
    hand-drawn. The quantitative manipulation + phenotype panels are fig1_setup (now Fig 2)."""
    N, tiles, path, patch_t0, patch_t3 = _ov_pipeline_data()
    animal = _ov_animal_data()

    fig = plt.figure(figsize=(WIDE, 6.7))
    gs = GridSpec(3, 1, height_ratios=[2.05, 1.95, 1.9], hspace=0.45, figure=fig)

    # -- row 1: panel a (pipeline)
    gsa = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0],
                                   width_ratios=[1.35, 0.65, 1.15, 0.65], wspace=0.55)
    axArena = fig.add_subplot(gsa[0]); axObs = fig.add_subplot(gsa[1])
    axRNN = fig.add_subplot(gsa[2]); axPred = fig.add_subplot(gsa[3])
    _ov_draw_arena_agent(axArena, N, tiles, path)
    _ov_draw_patch(axObs, patch_t0, TWO['translation'], 'observation $o_t$')
    _ov_draw_rnn(axRNN, 0.5, 0.55, 0.40)
    axRNN.set_xlim(0, 1); axRNN.set_ylim(0, 1.05); axRNN.axis('off')
    _ov_draw_patch(axPred, patch_t3, TWO['rotation'], r'prediction $\hat{o}_{t+k}$')
    _panel(axArena, 'a', dx=-0.10, dy=1.06)

    for (ax0, ax1, rad) in [(axArena, axObs, -0.25), (axObs, axRNN, -0.2), (axRNN, axPred, -0.2)]:
        p0 = ax0.get_position(); p1 = ax1.get_position()
        y = (p0.y0 + p0.y1) / 2
        fig.patches.append(FancyArrowPatch((p0.x1 + 0.005, y), (p1.x0 - 0.005, y),
                                            transform=fig.transFigure,
                                            connectionstyle=f'arc3,rad={rad}',
                                            arrowstyle='-|>', mutation_scale=8, lw=1.0,
                                            color=OV_ARROW_GREY, shrinkA=1, shrinkB=1))
    p_rnn = axRNN.get_position()
    axAct = fig.add_axes([p_rnn.x0 + 0.01, p_rnn.y0 - 0.115, p_rnn.width - 0.02, 0.09])
    _ov_draw_actions(axAct, heading='N')

    # -- row 2: panel b (arenas) + panel d (quotient map)
    gsbd = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1.55, 1.0], wspace=0.12)
    gsb = GridSpecFromSubplotSpec(1, 3, subplot_spec=gsbd[0], wspace=0.30)
    for j, (cond, title, domain_kind, caption) in enumerate(_OV_CONDS):
        axb = fig.add_subplot(gsb[j])
        _ov_draw_symarena(axb, cond, domain_kind, title, N=N)
        axb.text(0.5, -0.09, caption, transform=axb.transAxes, fontsize=5.6,
                 color=OV_LABEL_GREY, ha='center', va='top')
        if j == 0:
            _panel(axb, 'b', dx=-0.22, dy=1.10)

    gsd = GridSpecFromSubplotSpec(1, 3, subplot_spec=gsbd[1], width_ratios=[1.0, 0.35, 0.62],
                                   wspace=0.05)
    axDX = fig.add_subplot(gsd[0]); axDArrow = fig.add_subplot(gsd[1]); axDQ = fig.add_subplot(gsd[2])
    _ov_draw_X(axDX)
    _ov_draw_quotient(axDQ)
    axDArrow.set_xlim(0, 1); axDArrow.set_ylim(0, 1); axDArrow.axis('off')
    axDArrow.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                       arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3, mutation_scale=10))
    axDArrow.text(0.5, 0.62, r'$\pi$', fontsize=9, color=INK, ha='center', va='bottom', style='italic')
    _panel(axDX, 'd', dx=-0.16, dy=1.10)

    # -- row 3: panel c (Spiers / Grieves)
    gsc = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1.05, 1.4, 0.85],
                                   wspace=0.25)
    axSpiers = fig.add_subplot(gsc[0])
    _ov_draw_footprint(axSpiers, *animal['spiers'])
    axSpiers.set_title('Spiers et al. 2015', fontsize=7, pad=4, color=INK)
    _panel(axSpiers, 'c', dx=-0.14, dy=1.06)

    gsg = GridSpecFromSubplotSpec(2, 1, subplot_spec=gsc[1], hspace=0.55)
    axTrans = fig.add_subplot(gsg[0]); axRot = fig.add_subplot(gsg[1])
    _ov_draw_footprint(axTrans, *animal['translation'])
    axTrans.set_title('parallel (translation)', fontsize=6.5, pad=3, color=TWO['translation'])
    _ov_draw_footprint(axRot, *animal['rotation'])
    axRot.set_title('radial (rotation)', fontsize=6.5, pad=3, color=TWO['rotation'])
    p_trans = axTrans.get_position()
    fig.text(p_trans.x0, p_trans.y1 + 0.035, 'Grieves et al. 2016', fontsize=7, color=INK, ha='left')

    axCap = fig.add_subplot(gsc[2]); axCap.axis('off')
    axCap.set_xlim(0, 1); axCap.set_ylim(0, 1)
    axCap.text(0.0, 0.92, 'same compass reading', fontsize=6.0, color=TWO['translation'], ha='left')
    axCap.text(0.0, 0.82, r'$\Rightarrow$ code folds $\Rightarrow$ repeats', fontsize=6.0,
               color=TWO['translation'], ha='left', fontweight='bold')
    axCap.text(0.0, 0.62, 'different compass reading', fontsize=6.0, color=TWO['rotation'], ha='left')
    axCap.text(0.0, 0.52, r'$\Rightarrow$ code lifts $\Rightarrow$ remaps', fontsize=6.0,
               color=TWO['rotation'], ha='left', fontweight='bold')

    for ax in fig.axes:
        ax.patch.set_visible(False)

    out = figs / 'fig1_overview.pdf'
    fig.savefig(out, pad_inches=0.10)
    plt.close(fig)
    print(f'  wrote {out.name}')


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
    """Fig 2: the fold-specific readout (symmetry index) leads; field count, spatial RSA and
    cross-seed correlation are smaller supporting panels, matching their role in the text
    (symidx is fold-specific; fieldcount is confounded; srsa/crossseed are sanity checks)."""
    fp = data / 'field_stats.csv'; mq = data / 'map_quality_groupB.csv'
    if not (fp.exists() and mq.exists()):
        print('  skip fig2: missing field_stats/map_quality'); return
    fr = _read(fp); mr = _read(mq)
    fconds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in fr)]
    mconds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in mr)]
    fig = plt.figure(figsize=(WIDE, 5.2))
    gs = GridSpec(2, 3, height_ratios=[1.55, 1.0], hspace=0.55, wspace=0.42, figure=fig)
    axB = fig.add_subplot(gs[0, :])
    _draw_symidx(axB, fr, fconds); _panel(axB, 'a', dx=-0.06, dy=1.05)
    axB.axhline(1.0, ls=':', lw=0.7, color='k', alpha=0.6)
    axB.text(len(HD_ORDER) - 0.55, 1.0, 'perfect symmetry', fontsize=6, ha='right', va='bottom',
             color='k', alpha=0.7)
    axA = fig.add_subplot(gs[1, 0])
    _draw_fieldcount(axA, fr, fconds); _panel(axA, 'b')
    axC = fig.add_subplot(gs[1, 1])
    _draw_srsa(axC, mr, mconds); _panel(axC, 'c')
    axD = fig.add_subplot(gs[1, 2])
    _draw_crossseed(axD, mr, mconds); _panel(axD, 'd')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_SHADE[c]) for c in fconds]
    fig.legend(handles, [COND_CN[c] for c in fconds], loc='lower center', ncol=3,
               title='arena', title_fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = figs / 'fig2_fold.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig_compartments_solo(data: Path, figs: Path):
    """Standalone version of fig3_function panel c, for use outside the composite figure
    (e.g. slides): the two-room translation-vs-rotation result on its own."""
    comp = _read(data / 'compartments.csv') if (data / 'compartments.csv').exists() else None
    if comp is None:
        print('  skip fig_compartments_solo: missing compartments.csv'); return
    modes = [m for m in ('translation', 'rotation') if any(r['mode'] == m for r in comp)]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    _draw_compartments(ax, comp, modes)
    fig.tight_layout()
    out = figs / 'fig_compartments_solo.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig_horizon_solo(data: Path, figs: Path):
    """Standalone version of fig3_function panels a,b, for use outside the composite figure
    (e.g. slides): horizon sweep and replay coverage, without the compartments panel."""
    ph = {0: 'phase_horizon_k0', 1: 'phase_horizon_k1', 3: 'phase_horizon_k3'}
    phase = {}
    for k, name in ph.items():
        p = data / f'{name}.csv'
        if not p.exists():
            print('  skip fig_horizon_solo: missing horizon'); return
        phase[k] = _read(p)
    gb = data / 'phase_groupB.csv'
    if gb.exists():
        phase[5] = [r for r in _read(gb) if r['condition'] == 's2' and r['k'] == '5']
    rep = {k: [r for r in _read(data / f'replay_k{k}.csv') if r['condition'] == 's2']
           for k in (0, 1, 3, 5) if (data / f'replay_k{k}.csv').exists()}
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.7))
    _draw_phase_horizon(ax[0], phase, sorted(phase)); _panel(ax[0], 'a')
    _draw_replay_horizon(ax[1], rep, sorted(rep)); _panel(ax[1], 'b')
    fig.subplots_adjust(bottom=0.32, top=0.90, left=0.12, right=0.98, wspace=0.5)
    _hd_legend(fig, ncol=4, y=0.02)
    out = figs / 'fig_horizon_solo.pdf'; fig.savefig(out); plt.close(fig)
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


def fig2_dissociation(data: Path, figs: Path):
    """Headline: orbit-phase decoding per network -- invariance, not information, folds the code."""
    base = _phase(data, 'phase_full_n10')
    if not base:
        print('  skip fig2_dissociation: missing phase_full_n10'); return
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(5.35, 2.6), gridspec_kw={'width_ratios': [1.7, 1]})
    conds = ['s1', 's2', 's4']; x = np.arange(len(HD_ORDER)); w = 0.26
    for i, c in enumerate(conds):
        off = (i - 1) * w
        for j, hd in enumerate(HD_ORDER):
            v = [_f(r, 'phase_acc') for r in base if r['condition'] == c and r['hd_mode'] == hd]
            if not v:
                continue
            xj = x[j] + off
            ax[0].scatter(np.full(len(v), xj) + rng.uniform(-0.05, 0.05, len(v)), v, s=5,
                          color=COND_SHADE[c], alpha=0.75, edgecolors='none', zorder=3)
            ax[0].plot([xj - 0.11, xj + 0.11], [np.mean(v)] * 2, color=COND_SHADE[c], lw=1.6)
    ax[0].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(HD_ORDER)
    ax[0].set_ylabel('orbit-phase accuracy'); ax[0].set_ylim(0.45, 1.02)
    hleg = [plt.Line2D([0], [0], marker='o', ls='', ms=4, color=COND_SHADE[c], label=COND_CN[c])
            for c in conds]
    ax[0].legend(handles=hleg, fontsize=6, loc='center left', title='arena', title_fontsize=6)
    _panel(ax[0], 'a')
    # (b) matched-information contrast: axis vs parity, C1 vs C2
    xb = np.arange(2)
    for k, hd in enumerate(['parity', 'axis']):
        for i, c in enumerate(['s1', 's2']):
            v = [_f(r, 'phase_acc') for r in base if r['condition'] == c and r['hd_mode'] == hd]
            xi = xb[i] + (k - 0.5) * 0.32
            ax[1].scatter(np.full(len(v), xi) + rng.uniform(-0.04, 0.04, len(v)), v, s=6,
                          color=HD_COLOR[hd], alpha=0.85, edgecolors='none', zorder=3)
            ax[1].plot([xi - 0.12, xi + 0.12], [np.mean(v)] * 2, color=HD_COLOR[hd], lw=1.6)
    ax[1].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(['$C_1$', '$C_2$']); ax[1].set_ylim(0.45, 1.02)
    ax[1].set_title('matched: 1 bit each', fontsize=7)
    h2 = [plt.Line2D([0], [0], marker='o', ls='', ms=4, color=HD_COLOR[hd], label=hd)
          for hd in ['parity', 'axis']]
    ax[1].legend(handles=h2, fontsize=6, loc='center left'); _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'fig2_dissociation.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def _dots_by_arena(ax, rows, key, conds=('s1', 's2', 's4'), w=0.26, seed=0):
    """Grouped dot-and-mean plot: HD encoding on x, one shaded cluster per arena."""
    rng = np.random.default_rng(seed); x = np.arange(len(HD_ORDER))
    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * w
        for j, hd in enumerate(HD_ORDER):
            v = [_f(r, key) for r in rows if r['condition'] == c and r['hd_mode'] == hd]
            v = [x for x in v if np.isfinite(x)]
            if not v:
                continue
            xj = x[j] + off
            ax.scatter(np.full(len(v), xj) + rng.uniform(-0.05, 0.05, len(v)), v, s=5,
                       color=COND_SHADE[c], alpha=0.75, edgecolors='none', zorder=3)
            ax.plot([xj - 0.11, xj + 0.11], [np.mean(v)] * 2, color=COND_SHADE[c], lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER)
    return [plt.Line2D([0], [0], marker='o', ls='', ms=4, color=COND_SHADE[c], label=COND_CN[c])
            for c in conds]


def fig_population(data: Path, figs: Path):
    """The fold in the population code. Remapping (a) is the load-bearing population-level
    readout and gets the large panel; the isotypic spectrum (b) and odd-power (c) are the
    secondary, descriptive readouts the text demotes them to, and are sized to match."""
    rm = _read(data / 'remapping.csv') if (data / 'remapping.csv').exists() else []
    iso = _read(data / 'isotypic_hd.csv') if (data / 'isotypic_hd.csv').exists() else []
    if not rm or not iso:
        print('  skip fig_population: missing remapping/isotypic'); return
    fig = plt.figure(figsize=(WIDE, 4.9))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1.0], hspace=0.5, wspace=0.38, figure=fig)
    # (a) remapping: population-vector correlation between symmetry-related positions -- the
    # load-bearing readout, spanning the full width
    axA = fig.add_subplot(gs[0, :])
    hleg = _dots_by_arena(axA, rm, 'pv_orbit')
    axA.axhline(0.5, ls='--', color='#ccc', lw=0.7)
    axA.axhline(1.0, ls=':', lw=0.7, color='k', alpha=0.6)
    axA.text(len(HD_ORDER) - 0.55, 1.0, 'written as one place', fontsize=6, ha='right',
             va='bottom', color='k', alpha=0.7)
    axA.set_ylabel('remapping corr. (orbit PV)'); axA.set_ylim(0, 1.05)
    axA.legend(handles=hleg, fontsize=6, loc='center left', title='arena', title_fontsize=6)
    _panel(axA, 'a', dx=-0.06, dy=1.05)
    # (b) isotypic spectrum in the C2 arena: where each encoding places its power (descriptive)
    axB = fig.add_subplot(gs[1, 0])
    comps = ['P0', 'P1', 'P2', 'P3']; cshade = ['#C7CCD6', '#8FA0B3', '#5A7192', '#2E4763']
    x = np.arange(len(HD_ORDER)); w = 0.19
    for ci, comp in enumerate(comps):
        off = (ci - 1.5) * w
        m = [np.mean([_f(r, comp) for r in iso if r['condition'] == 's2' and r['hd_mode'] == hd] or [np.nan])
             for hd in HD_ORDER]
        axB.bar(x + off, m, w * 0.92, color=cshade[ci], label=comp, edgecolor='none')
    axB.set_xticks(x); axB.set_xticklabels(HD_ORDER)
    axB.set_ylabel('isotypic power ($C_2$ arena)'); axB.set_ylim(0, 0.45)
    axB.legend(fontsize=5.5, ncol=4, loc='upper center', columnspacing=0.8, handlelength=0.9)
    _panel(axB, 'b')
    # (c) odd-power (C2-odd content) collapses under the invariant encodings (descriptive)
    axC = fig.add_subplot(gs[1, 1])
    hleg = _dots_by_arena(axC, iso, 'odd')
    axC.set_ylabel('$C_2$-odd power ($P_1{+}P_3$)'); axC.set_ylim(0.25, 0.55)
    axC.legend(handles=hleg, fontsize=6, loc='lower left', title='arena', title_fontsize=6)
    _panel(axC, 'c')
    fig.tight_layout()
    out = figs / 'fig_population.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def figS_robustness(data: Path, figs: Path):
    """The fold is not an artifact: survives nonlinear embeddings and nonlinear decoders."""
    mr = _read(data / 'manifold_robustness.csv') if (data / 'manifold_robustness.csv').exists() else []
    nl = _read(data / 'phase_nonlinear.csv') if (data / 'phase_nonlinear.csv').exists() else []
    if not mr or not nl:
        print('  skip figS_robustness: missing inputs'); return
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    # (a) fold ratio (<1 = folded) across embeddings, C2 arena
    embs = [('fold_ratio_full', 'raw'), ('fold_ratio_pca3', 'PCA-3'),
            ('fold_ratio_isomap3', 'Isomap-3'), ('fold_ratio_tsne2', 't-SNE-2')]
    modes = ['full', 'parity', 'axis']; x = np.arange(len(embs)); w = 0.25
    for mi, hd in enumerate(modes):
        row = next((r for r in mr if r['condition'] == 's2' and r['hd_mode'] == hd), None)
        if not row:
            continue
        vals = [_f(row, k) for k, _ in embs]
        ax[0].bar(x + (mi - 1) * w, vals, w * 0.9, color=HD_COLOR[hd], label=hd, edgecolor='none')
    ax[0].axhline(1.0, ls='--', color='#999', lw=0.8)
    ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels([e[1] for e in embs], rotation=20)
    ax[0].set_ylabel('fold ratio  (<1 = folded)')
    ax[0].legend(fontsize=6, loc='upper left'); _panel(ax[0], 'a')
    # (b) nonlinear decoders agree with the linear readout
    decs = ['linear', 'knn', 'mlp']; dshade = ['#2E4763', '#6E6E6E', '#9C4A2F']
    xb = np.arange(len(HD_ORDER)); wb = 0.26
    for di, dec in enumerate(decs):
        m = [np.mean([_f(r, dec) for r in nl if r['hd_mode'] == hd] or [np.nan]) for hd in HD_ORDER]
        ax[1].bar(xb + (di - 1) * wb, m, wb * 0.9, color=dshade[di], label=dec, edgecolor='none')
    ax[1].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(HD_ORDER)
    ax[1].set_ylabel('orbit-phase accuracy ($C_2$)'); ax[1].set_ylim(0.45, 1.02)
    ax[1].legend(fontsize=6, loc='center left'); _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'figS_robustness.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def figS_prospective(data: Path, figs: Path):
    """Prospective-firing null: place fields track current position, not a future one."""
    pr = _read(data / 'prospective.csv') if (data / 'prospective.csv').exists() else []
    if not pr:
        print('  skip figS_prospective: missing prospective'); return
    offs = list(range(-3, 4)); keys = [f'si_{o:+d}' for o in offs]
    ks = sorted({int(r['k']) for r in pr})
    kcol = {k: plt.get_cmap('viridis')(i / max(1, len(ks) - 1)) for i, k in enumerate(ks)}
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for k in ks:
        rows = [r for r in pr if int(r['k']) == k]
        m = [np.mean([_f(r, key) for r in rows]) for key in keys]
        s = [(_mean_sem([_f(r, key) for r in rows])[1]) for key in keys]
        ax.errorbar(offs, m, yerr=s, color=kcol[k], lw=1.2, marker='o', ms=3,
                    capsize=1.5, label=f'k={k}')
    ax.axvline(0, ls=':', color='#999', lw=0.8)
    ax.set_xlabel('position offset (steps)'); ax.set_ylabel('spatial information (bits)')
    ax.set_xticks(offs); ax.legend(fontsize=6, title='horizon', title_fontsize=6, ncol=2)
    fig.tight_layout()
    out = figs / 'figS_prospective.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def fig_ceiling(data: Path, figs: Path):
    """The C4 four-way ceiling: invariant encodings fall to the group-theoretic 1/|G|."""
    r4 = _phase(data, 'phase_s4_c4')
    if not r4:
        print('  skip fig_ceiling: missing phase_s4_c4'); return
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(HD_ORDER))
    for j, hd in enumerate(HD_ORDER):
        v = [_f(r, 'phase_acc') for r in r4 if r['hd_mode'] == hd]
        if not v:
            continue
        ax.scatter(np.full(len(v), j) + rng.uniform(-0.09, 0.09, len(v)), v, s=8,
                   color=HD_COLOR[hd], alpha=0.85, edgecolors='none', zorder=3)
        ax.plot([j - 0.16, j + 0.16], [np.mean(v)] * 2, color=HD_COLOR[hd], lw=1.6)
    ax.axhline(0.25, ls=':', color='#9C4A2F', lw=1.1, label='$1/4$  ($C_4$-invariant)')
    ax.axhline(0.50, ls=':', color='#3A6B6B', lw=1.1, label='$1/2$  ($C_2$-invariant)')
    ax.set_xticks(x); ax.set_xticklabels(HD_ORDER)
    ax.set_ylabel('$C_4$ four-way phase accuracy'); ax.set_ylim(0.15, 1.02)
    ax.legend(fontsize=6, loc='center right')
    fig.tight_layout()
    out = figs / 'fig_ceiling.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def _acc(rows, cond, hd):
    return _mean_sem([_f(r, 'phase_acc') for r in rows
                      if r['condition'] == cond and r['hd_mode'] == hd])


def fig5_generality(data: Path, figs: Path):
    """Generality of the fold: HD-lesion dose-response and a learned compass."""
    base = _phase(data, 'phase_full_n10')          # baseline: hidden 500, arena 18, noise 0
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.6))
    # (a) HD-lesion dose-response, all encodings
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
            ax[0].errorbar(xs, ms, yerr=es, marker='o', ms=4, color=HD_COLOR[hd],
                           label=hd, capsize=2, lw=1.3, elinewidth=0.7)
    ax[0].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[0].set_xlabel('head-direction corruption'); ax[0].set_ylabel('orbit-phase accuracy ($C_2$)')
    ax[0].set_ylim(0.45, 1.02); _panel(ax[0], 'a'); ax[0].legend(fontsize=6, ncol=2)
    # (b) learned (angular-velocity) compass, by arena
    lrn = _phase(data, 'phase_learned_c2')
    conds = [c for c in ('s1', 's2', 's4') if any(r['condition'] == c for r in lrn)]
    ms, es = zip(*[_mean_sem([_f(r, 'phase_acc') for r in lrn if r['condition'] == c])
                   for c in conds]) if conds else ([], [])
    ax[1].bar(range(len(conds)), ms, 0.62, yerr=es, capsize=2,
              color=[COND_SHADE[c] for c in conds], edgecolor='k', lw=0.4, error_kw={'lw': 0.7})
    ax[1].axhline(0.5, ls='--', color='#999', lw=0.8)
    ax[1].set_xticks(range(len(conds))); ax[1].set_xticklabels([COND_CN[c] for c in conds])
    ax[1].set_xlabel('arena'); ax[1].set_ylabel('orbit-phase accuracy'); ax[1].set_ylim(0.45, 1.02)
    ax[1].set_title('learned compass', fontsize=7); _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'fig5_generality.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def _di_pairs(rows, key_fields, di='di', rep_only=False):
    """Within-group same-orientation ordered DI pairs; returns (xs, ys)."""
    by = {}
    for r in rows:
        if rep_only and r.get('rep') not in ('True', 'TRUE', True):
            continue
        by.setdefault(tuple(r[k] for k in key_fields), []).append(_f(r, di))
    xs, ys = [], []
    for v in by.values():
        for i in range(len(v)):
            for j in range(len(v)):
                if i != j:
                    xs.append(v[i]); ys.append(v[j])
    return np.asarray(xs), np.asarray(ys)


def _di_scatter(ax, xs, ys, color, subsample=None, seed=0):
    if len(xs) < 3:
        return
    r = np.corrcoef(xs, ys)[0, 1]; b = np.polyfit(xs, ys, 1)
    px, py = xs, ys
    if subsample and len(xs) > subsample:
        idx = np.random.default_rng(seed).choice(len(xs), subsample, replace=False)
        px, py = xs[idx], ys[idx]
    ax.scatter(px, py, s=5, color=color, alpha=0.28, edgecolors='none')
    xr = np.array([-1, 1]); ax.plot(xr, b[0] * xr + b[1], color='#9C4A2F', lw=1.5)
    ax.text(0.05, 0.92, f'$r={r:.2f}$', transform=ax.transAxes, fontsize=7)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)


def fig6_brain(data: Path, figs: Path):
    """To the brain (model): four-room complete fold, city-block directional repetition."""
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.7))
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
    ax[0].set_title('four-room maze', fontsize=7); _panel(ax[0], 'a')
    # (b) city-block model directional-repetition correlation
    cb = _read(data / 'cityblock.csv') if (data / 'cityblock.csv').exists() else []
    cx, cy = _di_pairs(cb, ('seed', 'unit', 'orient'))
    _di_scatter(ax[1], cx, cy, '#3A6B6B', subsample=2500)
    ax[1].set_xlabel('field $i$ DI'); ax[1].set_ylabel('field $j$ DI')
    ax[1].set_title('city-block model', fontsize=7); _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'fig6_brain.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def figS_coset_phase(data: Path, figs: Path):
    """Supplementary: folded, not broken. Folded codes (axis/const in symmetric arenas) keep
    within-domain position (domain_r2, high) while losing orbit phase (phase_acc, at the null);
    non-folded codes have both. The two readouts move independently, which is the signature that
    distinguishes a fold from a dead code."""
    base = _phase(data, 'phase_full_n10')
    if not base:
        print('  skip figS_coset_phase: missing phase_full_n10'); return
    fig, ax = plt.subplots(1, 1, figsize=(COL, 2.9))
    conds = ['s1', 's2', 's4']
    markers = {'s1': 'o', 's2': 's', 's4': '^'}
    for c in conds:
        for hd in HD_ORDER:
            xs = [_f(r, 'domain_r2') for r in base if r['condition'] == c and r['hd_mode'] == hd]
            ys = [_f(r, 'phase_acc') for r in base if r['condition'] == c and r['hd_mode'] == hd]
            if not xs:
                continue
            ax.scatter(xs, ys, s=14, marker=markers[c], color=HD_COLOR[hd], alpha=0.8,
                      edgecolors='none')
    ax.axhline(0.5, ls='--', lw=0.7, color='#999')
    ax.text(0.02, 0.51, 'phase at chance', fontsize=6, color='#666')
    ax.set_xlabel('within-domain position ($R^2_{\\mathrm{domain}}$)')
    ax.set_ylabel('orbit-phase accuracy')
    ax.set_ylim(0.45, 1.02)
    hleg = [plt.Line2D([0], [0], marker='o', ls='', ms=4, color=HD_COLOR[hd], label=hd)
            for hd in HD_ORDER]
    aleg = [plt.Line2D([0], [0], marker=markers[c], ls='', ms=4, color='#555', label=COND_CN[c])
            for c in conds]
    leg1 = ax.legend(handles=hleg, fontsize=6, loc='center left', title='encoding',
                     title_fontsize=6, bbox_to_anchor=(1.01, 0.7))
    ax.add_artist(leg1)
    ax.legend(handles=aleg, fontsize=6, loc='center left', title='arena', title_fontsize=6,
              bbox_to_anchor=(1.01, 0.3))
    fig.tight_layout()
    out = figs / 'figS_coset_phase.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


def figS_ca1(data: Path, figs: Path):
    """Supplementary: exploratory CA1 reanalysis (consistent with the fold, not diagnostic)."""
    fd = _read(data / 'hockeimer_field_di.csv') if (data / 'hockeimer_field_di.csv').exists() else []
    if not fd:
        print('  skip figS_ca1: missing hockeimer_field_di'); return
    fig, ax = plt.subplots(1, 2, figsize=(ONEHALF, 2.7))
    xr, yr = _di_pairs(fd, ('cell', 'orient'), rep_only=True)          # repeating (claim)
    _di_scatter(ax[0], xr, yr, '#1F3B5C')
    ax[0].set_xlabel('field $i$ DI'); ax[0].set_ylabel('field $j$ DI')
    ax[0].set_title('CA1 repeating cells', fontsize=7); _panel(ax[0], 'a')
    xn, yn = _di_pairs(fd, ('cell', 'orient'), rep_only=False)         # all (control)
    # non-repeating subset for the control comparison
    xnr, ynr = _di_pairs([r for r in fd if r.get('rep') not in ('True', 'TRUE', True)],
                         ('cell', 'orient'))
    _di_scatter(ax[1], xnr, ynr, '#8A8A8A')
    ax[1].set_xlabel('field $i$ DI'); ax[1].set_ylabel('field $j$ DI')
    ax[1].set_title('CA1 non-repeating (control)', fontsize=7); _panel(ax[1], 'b')
    fig.tight_layout()
    out = figs / 'figS_ca1.pdf'; fig.savefig(out); plt.close(fig)
    print(f'  wrote {out.name}')


# ============================================================ the three new results
# Figure grammar follows Levenstein et al.: one dot is one network, never a summary bar; a
# reachable FLOOR and a reachable CEILING are drawn in the same axes as the data, so the reader can
# see how much of the phenomenon is explained rather than only that it is non-zero; and every
# geometric claim carries its scalar on the panel.

def _dots_by_hd(ax, rows, key, conds=('s1', 's2', 's4'), seed=0):
    """One dot per network, grouped by HD encoding, split by arena. Mean and 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    x = np.arange(len(HD_ORDER))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        off = (i - (len(conds) - 1) / 2) * w
        for j, hd in enumerate(HD_ORDER):
            v = [_f(r, key) for r in rows if r['condition'] == c and r['hd_mode'] == hd]
            v = [t for t in v if np.isfinite(t)]
            if not v:
                continue
            xx = x[j] + off
            ax.scatter(xx + (rng.random(len(v)) - 0.5) * w * 0.5, v, s=3.0,
                       color=COND_SHADE.get(c, '#888'), edgecolor='k', linewidths=0.2,
                       zorder=3, label=COND_CN.get(c, c) if j == 0 else None)
            m, lo, hi = _boot_ci(v)
            ax.plot([xx - w * 0.32, xx + w * 0.32], [m, m], color='k', lw=1.0, zorder=4)
            ax.plot([xx, xx], [lo, hi], color='k', lw=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(HD_ORDER)


def fig_isometry(data: Path, figs: Path):
    """The map IS the quotient. The central result was a negative (a decoder at chance); this is
    the positive form of it: what space is the manifold a metric map of?"""
    rows = _read(data / 'isometry_quotient.csv')
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.35))
    aA, aB, aC = axes

    # (a) the money plane: fit to the arena vs fit to the quotient. Below the diagonal = the
    # network is better described as a map of X/G than of X.
    for hd in HD_ORDER:
        rs = [r for r in rows if r['hd_mode'] == hd]
        aA.scatter([_f(r, 'stress_X') for r in rs], [_f(r, 'stress_XG_c2') for r in rs],
                   s=7, color=HD_COLOR[hd], edgecolor='k', linewidths=0.2, label=hd, zorder=3)
    lim = (0.05, 0.62)
    aA.plot(lim, lim, ls='--', lw=0.7, color=INK, zorder=2)
    # Above the diagonal, stress_X < stress_{X/G}: the network is a better map of the ARENA.
    # Below it, the quotient wins. Both labels sit in their own region.
    aA.text(0.215, 0.545, 'maps $X$ better', fontsize=5.2, color=INK, ha='center')
    aA.text(0.175, 0.105, 'maps $X/G$ better', fontsize=5.2, color=INK, ha='center')
    # reachable ceiling: what the metric returns when the code carries no spatial information
    shuf = np.mean([_f(r, 'stress_shuffled') for r in rows])
    aA.axhline(shuf, color='#AAAAAA', lw=0.7, ls=':', zorder=1)
    aA.text(lim[1] - 0.01, shuf + 0.008, 'shuffled', fontsize=5, color='#777777', ha='right')
    aA.set_xlim(*lim); aA.set_ylim(*lim)
    aA.set_xlabel('stress vs the arena $d_X$')
    aA.set_ylabel('stress vs the quotient $d_{X/C_2}$')
    aA.legend(fontsize=5, frameon=False, loc='lower right', handletextpad=0.2, borderpad=0.2)
    _panel(aA, 'a')

    # (b) the sham control. d_{X/G} <= d_X pointwise, so ANY group compresses long distances and a
    # merely compressed code would prefer a quotient metric for trivial reasons. A sham order-2
    # group (half-arena translation, not a symmetry of any arena) has the same compression and is
    # the null this measure needs.
    x = np.arange(len(HD_ORDER))
    for i, (key, col, lab) in enumerate([('stress_XG_c2', '#9C4A2F', 'true $C_2$'),
                                         ('stress_sham', '#8A8A8A', 'sham order-2')]):
        m = [np.mean([_f(r, key) for r in rows
                      if r['condition'] == 's2' and r['hd_mode'] == hd]) for hd in HD_ORDER]
        aB.plot(x, m, 'o-', ms=3.5, lw=1.0, color=col, label=lab, zorder=3)
    aB.set_xticks(x); aB.set_xticklabels(HD_ORDER)
    aB.set_ylabel('stress ($C_2$ arena)')
    aB.set_ylim(0.1, 0.65)
    aB.legend(fontsize=5, frameon=False, loc='upper left', handletextpad=0.3)
    aB.annotate('', xy=(2, 0.19), xytext=(2, 0.534),
                arrowprops=dict(arrowstyle='<->', lw=0.6, color=INK))
    aB.text(2.08, 0.35, 'the fold is\nsymmetry-specific', fontsize=5, color=INK, va='center')
    _panel(aB, 'b')

    # (c) the matched test, metric-free: is H(x) = H(g.x)?  axis and parity carry the same one bit.
    _dots_by_hd(aC, rows, 'fold_cos_c2')
    aC.axhline(1.0, color='#AAAAAA', lw=0.7, ls=':', zorder=1)
    aC.text(3.35, 1.005, 'orbit is\none point', fontsize=5, color='#777777', ha='right')
    aC.set_ylabel(r'fold coincidence  $\cos(H(x), H(R^2x))$')
    aC.set_ylim(0.2, 1.06)
    aC.legend(fontsize=5, frameon=False, loc='center left', handletextpad=0.2, borderpad=0.2)
    _panel(aC, 'c')

    fig.tight_layout()
    out = figs / 'fig_isometry.pdf'
    fig.savefig(out); plt.close(fig)
    print(f'  {out}')


def fig_bvc(data: Path, figs: Path):
    """Boundary-vector cells are downstream of the compass, not upstream of the fold."""
    rows = _read(data / 'bvc_tuning.csv')
    ph = _read(data / 'phase_full_n10.csv')
    fig, axes = plt.subplots(1, 2, figsize=(ONEHALF, 2.35))
    aA, aB = axes

    # (a) the network DOES grow boundary cells -- and they need the compass.
    _dots_by_hd(aA, rows, 'frac_bvc_like')
    aA.set_ylabel('units better fit by a BVC\nthan by any place field')
    aA.set_ylim(-0.02, 0.78)
    aA.legend(fontsize=5, frameon=False, loc='upper right', handletextpad=0.2, borderpad=0.2)
    _panel(aA, 'a')

    # (b) the decisive plane. axis and parity carry the same one bit and support comparable
    # boundary populations, yet only axis folds. Same boundary code, opposite maps.
    pmap = {}
    for r in ph:
        pmap.setdefault((r['condition'], r['hd_mode']), []).append(_f(r, 'phase_acc'))
    for hd in HD_ORDER:
        for c in ('s1', 's2', 's4'):
            rs = [r for r in rows if r['hd_mode'] == hd and r['condition'] == c]
            if not rs:
                continue
            xv = np.mean([_f(r, 'frac_bvc_like') for r in rs])
            yv = np.nanmean(pmap.get((c, hd), [np.nan]))
            mk = {'s1': 'o', 's2': 's', 's4': '^'}[c]
            aB.scatter(xv, yv, s=26, marker=mk, color=HD_COLOR[hd], edgecolor='k',
                       linewidths=0.35, zorder=3)
    # the two reachable references: the group-theoretic floor and the non-invariant ceiling
    aB.axhline(0.5, color=INK, lw=0.8, zorder=1)
    aB.text(0.66, 0.515, r'$1/|G|$ floor', fontsize=5, color=INK)
    aB.axhspan(0.93, 0.99, color='#DDDDDD', alpha=0.6, zorder=0)
    aB.text(0.66, 0.945, 'non-invariant\nceiling', fontsize=5, color='#666666')
    aB.annotate('axis', xy=(0.292, 0.552), xytext=(0.33, 0.68), fontsize=5.5,
                color=HD_COLOR['axis'],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=HD_COLOR['axis']))
    aB.annotate('parity', xy=(0.221, 0.955), xytext=(0.06, 0.86), fontsize=5.5,
                color=HD_COLOR['parity'],
                arrowprops=dict(arrowstyle='-', lw=0.5, color=HD_COLOR['parity']))
    aB.set_xlabel('fraction of units that are BVC-like')
    aB.set_ylabel('orbit-phase accuracy')
    aB.set_xlim(-0.02, 0.78); aB.set_ylim(0.44, 1.02)
    hs = [plt.Line2D([0], [0], marker=m, ls='', color='#555555', ms=3.5)
          for m in ('o', 's', '^')]
    aB.legend(hs, [COND_CN[c] for c in ('s1', 's2', 's4')], fontsize=5, frameon=False,
              loc='center right', handletextpad=0.2, borderpad=0.2)
    _panel(aB, 'b')

    fig.tight_layout()
    out = figs / 'fig_bvc.pdf'
    fig.savefig(out); plt.close(fig)
    print(f'  {out}')


def fig_cellprops(data: Path, figs: Path):
    """Ablating the compass degrades the map everywhere; only field COUNT also folds.

    For each cell property, the effect of removing the compass is measured twice: in the C1 arena,
    where there is no symmetry and so nothing can fold (that part is pure degradation), and in the
    C2 arena. The difference is the symmetry-specific component. Each is normalised by the C1
    effect, so the units are "fraction of the degradation effect", and each carries a 95% bootstrap
    CI over networks, so a reader can see which components are real rather than being told.
    """
    rows = _read(data / 'cell_properties.csv')
    METRICS = [('spatial_info', 'spatial info'), ('sparsity', 'sparsity'),
               ('selectivity', 'selectivity'), ('field_area', 'field area'),
               ('coherence', 'coherence'), ('mixed', 'mixed selectivity'),
               ('n_fields', 'fields per cell')]
    fig, ax = plt.subplots(figsize=(COL, 2.7))

    def vals(cond, hd, key):
        return np.array([_f(r, key) for r in rows
                         if r['condition'] == cond and r['hd_mode'] == hd], float)

    rng = np.random.default_rng(0)
    for i, (key, lab) in enumerate(METRICS):
        f1, k1 = vals('s1', 'full', key), vals('s1', 'const', key)
        f2, k2 = vals('s2', 'full', key), vals('s2', 'const', key)
        deg = k1.mean() - f1.mean()                      # the C1 (degradation) effect
        if abs(deg) < 1e-9:
            continue
        # bootstrap the interaction over networks
        B = np.empty(4000)
        for b in range(B.size):
            e1 = rng.choice(k1, k1.size).mean() - rng.choice(f1, f1.size).mean()
            e2 = rng.choice(k2, k2.size).mean() - rng.choice(f2, f2.size).mean()
            B[b] = (e2 - e1) / abs(deg)
        m, lo, hi = B.mean(), np.percentile(B, 2.5), np.percentile(B, 97.5)
        real = (lo > 0) or (hi < 0)
        col = '#9C4A2F' if real else '#9A9A9A'
        ax.plot([lo, hi], [i, i], color=col, lw=1.1, zorder=3, solid_capstyle='round')
        ax.plot(m, i, 'o', ms=4.2, color=col, mec='k', mew=0.35, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([m[1] for m in METRICS])
    ax.set_xlabel('symmetry-specific component\n(fraction of the degradation effect)')
    ax.text(0.60, 6.42, 'the fold', fontsize=6, color='#9C4A2F', fontweight='bold',
            ha='center')
    ax.text(-0.42, 3.9, 'zero = the compass\nablation does the same\nthing where nothing\ncan fold',
            fontsize=5.0, color='#777777', va='center', ha='left')
    ax.set_xlim(-0.45, 0.95)
    ax.set_ylim(-0.7, len(METRICS) - 0.15)
    fig.tight_layout()
    out = figs / 'fig_cellprops.pdf'
    fig.savefig(out); plt.close(fig)
    print(f'  {out}')


def fig_lesion(data: Path, figs: Path):
    """The in-silico head-direction lesion: Harland et al. (2017), with a dose.

    This is the paper's causal figure, and the only one whose prediction was tested in an animal
    before we made it. Panels a-b take the compass away from an ADULT network (trained with one, as
    Harland's rats developed with one) and show that the damage has two separable parts: a
    DEGRADATION that happens everywhere, and a FOLD that happens only where a symmetry exists for the
    map to fold onto. Panel c is Harland's own 2x2, in silico. Panel d is the quantitative match.

    Panel c carries the reachable null, and it is the whole reason the figure is not circular. If our
    lesion were merely a generalised insult to the code, repetition would rise in the TRANSLATION arm
    too. It must not: a compass is translation-invariant by construction, so it was never the thing
    holding the parallel compartments apart, and destroying it cannot change them. Harland measured
    exactly this and found exactly nothing (65% -> 63%, p = 0.31), while the radial arm folded
    (p = 0.021), interaction F(1,10) = 13.60. A flat translation line is the prediction; a rising one
    would falsify us.
    """
    dose = [r for r in _read(data / 'lesion_dose.csv')
            if r.get('lesion_mode', 'silence') == 'silence']   # a lying compass is not a lesion
    comp_p = data / 'lesion_compartments_silence.csv'
    comp = _read(comp_p) if comp_p.exists() else []

    fig, axes = plt.subplots(2, 2, figsize=(ONEHALF, 4.3))
    (a, b), (c, d) = axes

    def curve(ax, rows, group_key, group, ykey, color, label, norm=False):
        xs = sorted({_f(r, 'dose') for r in rows})
        m, lo, hi = [], [], []
        base = None
        for x in xs:
            v = [_f(r, ykey) for r in rows if r[group_key] == group and _f(r, 'dose') == x]
            mm, ll, hh = _boot_ci(v)
            if norm:
                if base is None:
                    base = mm
                mm, ll, hh = 100 * (mm / base - 1), 100 * (ll / base - 1), 100 * (hh / base - 1)
            m.append(mm); lo.append(ll); hi.append(hh)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.16, lw=0)
        ax.plot(xs, m, '-o', color=color, ms=2.6, mec='k', mew=0.25, lw=1.2, label=label)
        return xs, m

    # (a) the fold: orbit-phase decoding collapses to chance ONLY where a symmetry exists
    for cond in ['s1', 's2', 's4']:
        curve(a, dose, 'condition', cond, 'phase_acc', COND_SHADE[cond], COND_CN.get(cond, cond))
    a.axhline(0.5, ls=':', lw=0.8, color='#B03A2E')
    a.text(0.99, 0.462, 'chance', fontsize=5.5, color='#B03A2E', ha='right')
    a.axhline(1.0, ls=':', lw=0.8, color='#7A7A7A')
    a.text(0.99, 1.012, 'ceiling', fontsize=5.5, color='#7A7A7A', ha='right')
    a.set_ylim(0.44, 1.08); a.set_ylabel('orbit-phase decoding')
    a.set_xlabel('lesion dose (fraction of steps)')
    a.text(0.5, 1.035, 'equivariance loss', fontsize=6, color='#333333', ha='center',
           style='italic')
    a.legend(loc='lower left', fontsize=5.8, handlelength=1.0, bbox_to_anchor=(0.0, 0.06))
    _panel(a, 'a')

    # (b) THE OTHER HALF OF THE SAME LESION. The quotient map -- position in the fundamental domain --
    # degrades by about a third in EVERY arena, and the degradation is NOT ordered by symmetry: C1 and
    # C4, at opposite ends of the symmetry range, lose indistinguishable amounts. This is the
    # information loss, and panel (a) is the equivariance loss. Whatever else the compass does, what
    # it does to the QUOTIENT does not depend on whether a symmetry exists -- which is exactly why the
    # fold in (a) cannot be degradation. Plotting them side by side IS the argument.
    for cond in ['s1', 's2', 's4']:
        curve(b, dose, 'condition', cond, 'ev_orbit', COND_SHADE[cond], COND_CN.get(cond, cond),
              norm=True)
    b.axhline(0, lw=0.6, color='k')
    b.set_ylim(-46, 8)
    b.set_ylabel('quotient map, orbit variance\n(% vs sham)')
    b.set_xlabel('lesion dose (fraction of steps)')
    # Honest wording: the drops are NOT statistically identical (ANOVA F = 10.78, p = 0.002). What
    # they are is NOT ORDERED BY SYMMETRY -- C1 (-37.6%) and C4 (-38.1%), at opposite ends of the
    # symmetry range, are indistinguishable, and C2 (-30.9%) degrades least. That is what rules
    # degradation out as the cause of the fold, and it is a weaker claim than "matched".
    b.text(0.03, -41.5, 'C$_1$ and C$_4$ lose the same third.\nThe damage does not track symmetry.',
           fontsize=5.2, color='#555555')
    b.text(0.5, 3.0, 'information loss', fontsize=6, color='#333333', ha='center',
           style='italic')
    _panel(b, 'b')

    # (c) Harland's 2x2 in silico. The translation arm is the reachable null.
    if comp:
        COMP_COL = {'translation': '#9A9A9A', 'rotation': '#B03A2E'}
        for mode in ['translation', 'rotation']:
            curve(c, comp, 'mode', mode, 'repetition', COMP_COL[mode],
                  {'translation': 'parallel (translation)',
                   'rotation': 'radial (rotation)'}[mode])
        # the control: a folded code still knows where in the room it is
        xs, r2 = curve(c, comp, 'mode', 'rotation', 'within_r2', '#5B84B1', 'within-room $R^2$')
        c.set_ylim(-0.32, 1.08); c.set_ylabel('field repetition, A vs B')
        c.set_xlabel('lesion dose (fraction of steps)')
        c.legend(loc='center left', fontsize=5.5, handlelength=1.0)
        c.text(0.3, -0.27, 'Harland: parallel 65%$\\to$63% n.s.; radial folds, p = 0.021',
               fontsize=5.0, color='#777777', ha='center')
    else:
        c.text(0.5, 0.5, 'lesion_compartments.csv\nnot yet generated', ha='center', va='center',
               fontsize=6, color='#999999'); c.set_axis_off()
    _panel(c, 'c')

    # (d) Two head-direction lesion studies that appear to contradict each other, reconciled.
    #
    # Calton et al. (2003) lesioned ADN/postsubiculum and found NO field multiplication (1.29 ->
    # 1.53, n.s.) and no significant spatial-information loss. Harland et al. (2017) lesioned the
    # LMN and found repetition RETURNING (p = 0.021) and information falling (p < 0.002). Read as
    # claims about "what an HD lesion does", these disagree.
    #
    # They do not disagree. Calton's cylinder carried a white cue card over ~100 deg of arc -- a
    # polarising landmark that BREAKS the rotational symmetry. Harland's compartments were identical,
    # behind a black curtain, with no polarising cue -- the symmetry STANDS. The quotient law says a
    # compass lesion can only fold a map onto a symmetry that exists, so it must multiply fields in
    # Harland's world and leave Calton's alone. Same lesion, two different worlds.
    #
    # We show the model's own interaction, one dot per network, and leave the rat numbers to the
    # text. A bar chart here would be dishonest twice over: Harland reports a between-compartment
    # CORRELATION, not a field count, so his cell has no number to plot, and Calton's field count
    # (+18.6%, n.s., n = 17 cells) is a point estimate whose interval spans zero -- drawing it as a
    # bar beside ours would invite a comparison of point estimates that neither study supports.
    # The claim is the INTERACTION, so the interaction is what we draw.
    rng2 = np.random.default_rng(1)
    for i, cond in enumerate(['s1', 's2', 's4']):
        rows_c = [r for r in dose if r['condition'] == cond]
        base = {int(r['seed']): _f(r, 'n_fields') for r in rows_c if _f(r, 'dose') == 0.0}
        v = [100 * (_f(r, 'n_fields') / base[int(r['seed'])] - 1)
             for r in rows_c if _f(r, 'dose') == 1.0 and int(r['seed']) in base]
        col = COND_SHADE[cond]
        jit = (rng2.random(len(v)) - 0.5) * 0.28
        d.scatter(i + jit, v, s=9, color=col, edgecolor='k', linewidths=0.3, zorder=3)
        m, lo, hi = _boot_ci(v)
        d.plot([i - 0.22, i + 0.22], [m, m], color='k', lw=1.1, zorder=4)
        d.plot([i, i], [lo, hi], color='k', lw=0.8, zorder=4)
    d.axhline(0, lw=0.7, color='k')
    d.set_xticks([0, 1, 2])
    d.set_xticklabels(['C$_1$\nnone', 'C$_2$\n180$\\degree$', 'C$_4$\n90$\\degree$'], fontsize=6)
    d.set_xlabel('symmetry available to fold onto')
    d.set_ylabel('fields per cell after\nfull lesion (% vs sham)')
    d.set_ylim(-22, 40)
    d.set_xlim(-0.55, 2.55)
    d.text(0.02, 0.94, "Calton's world", fontsize=5.4, color='#8A8A8A', transform=d.transAxes)
    d.text(0.02, 0.87, '(cue card breaks it)', fontsize=5.0, color='#A5A5A5',
           transform=d.transAxes)
    d.text(0.98, 0.94, "Harland's world", fontsize=5.4, color='#333333', ha='right',
           transform=d.transAxes)
    d.text(0.98, 0.87, '(identical rooms)', fontsize=5.0, color='#777777', ha='right',
           transform=d.transAxes)
    _panel(d, 'd')

    fig.tight_layout()
    out = figs / 'fig_lesion.pdf'
    fig.savefig(out); plt.close(fig)
    print(f'  {out}')


def fig_manifold(data: Path, figs: Path):
    """The fold, seen geometrically. Replaces manifold_s2.png, which was an orphan raster: it
    backed a main-text figure and no script produced it, and a raster in an otherwise-vector paper
    is an automatic reject at several journals. Emitted as vector, with only the dense 3-D scatter
    rasterized, so axes and text stay editable and the file stays small."""
    ratio = _read(data / 'manifold_fold_ratio.csv')
    coords = _read(data / 'manifold_coords.csv')
    PANELS = [('s2', 'axis'), ('s1', 'axis'), ('s2', 'parity'), ('s2', 'full')]
    lab = {('s2', 'axis'): 'axis / $C_2$', ('s1', 'axis'): 'axis / $C_1$',
           ('s2', 'parity'): 'parity / $C_2$', ('s2', 'full'): 'full / $C_2$'}

    fig = plt.figure(figsize=(WIDE, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.34)
    aA = fig.add_subplot(gs[0, 0])

    # (a) the fold ratio, one dot per network, with the meaningful reference at 1.0
    rng = np.random.default_rng(0)
    for i, key in enumerate(PANELS):
        v = [_f(r, 'fold_ratio') for r in ratio
             if (r['condition'], r['hd_mode']) == key]
        v = [t for t in v if np.isfinite(t)]
        col = HD_COLOR[key[1]]
        aA.scatter(i + (rng.random(len(v)) - 0.5) * 0.30, v, s=5, color=col,
                   edgecolor='k', linewidths=0.2, zorder=3)
        m, lo, hi = _boot_ci(v)
        aA.plot([i - 0.22, i + 0.22], [m, m], color='k', lw=1.0, zorder=4)
        aA.plot([i, i], [lo, hi], color='k', lw=0.7, zorder=4)
    aA.axhline(1.0, color=INK, lw=0.8, zorder=2)
    aA.text(3.42, 1.06, 'orbit partners as far apart\nas spatial neighbours',
            fontsize=4.6, color='#666666', ha='right')
    aA.text(0.0, 0.80, 'FOLDED', fontsize=5.4, color=HD_COLOR['axis'], fontweight='bold',
            ha='center')
    aA.set_xticks(range(len(PANELS)))
    aA.set_xticklabels([lab[k] for k in PANELS], rotation=20, ha='right')
    aA.set_ylabel(r'fold ratio  $d(x,\,R^2x)\ /\ d(x,\,\mathrm{neighbour})$')
    aA.set_ylim(0, 3.1)
    _panel(aA, 'a')

    # (b, c) the manifold itself, for the most- and the least-folded condition
    for pi, key in enumerate([('s2', 'axis'), ('s2', 'full')]):
        ax = fig.add_subplot(gs[0, pi + 1], projection='3d')
        rs = [r for r in coords if (r['condition'], r['hd_mode']) == key]
        P = np.array([[_f(r, 'pc1'), _f(r, 'pc2'), _f(r, 'pc3')] for r in rs])
        ph = np.array([int(r['phase']) for r in rs])
        for p, c in ((0, '#1F3B5C'), (1, '#9C4A2F')):
            s = ph == p
            # rasterize ONLY the scatter: the panel stays a vector PDF
            ax.scatter(P[s, 0], P[s, 1], P[s, 2], s=5, color=c, alpha=0.85,
                       edgecolor='none', rasterized=True,
                       label=f'phase {p}' if pi == 0 else None)
        fr = float(np.mean([_f(r, 'fold_ratio') for r in rs]))
        # never a manifold without its scalar printed under it
        ax.set_title(f'{lab[key]}\nfold ratio {fr:.2f}', fontsize=6, pad=-2)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel('PC1', fontsize=5, labelpad=-12)
        ax.set_ylabel('PC2', fontsize=5, labelpad=-12)
        ax.set_zlabel('PC3', fontsize=5, labelpad=-12)
        ax.tick_params(length=0)
        ax.view_init(elev=20, azim=-58)
        # Axes3D.text takes (x, y, z, s); the panel letter needs the 2-D overlay.
        ax.text2D(0.02, 0.94, 'bc'[pi], transform=ax.transAxes, fontsize=8,
                  fontweight='bold', va='bottom', ha='right', color=INK, family=_family)
        if pi == 0:
            ax.legend(fontsize=5, frameon=False, loc='upper left',
                      bbox_to_anchor=(-0.06, 0.98), handletextpad=0.1)

    out = figs / 'fig_manifold.pdf'
    fig.savefig(out, dpi=450)
    plt.close(fig)
    print(f'  {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--figs', required=True)
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    data, figs = Path(a.data), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    allfigs = {'overview': fig1_overview, 'fig1': fig1_setup, 'dissociation': fig2_dissociation,
               'ceiling': fig_ceiling,
               'population': fig_population, 'fig2': fig2_fold, 'fig3': fig3_function,
               'isometry': fig_isometry, 'bvc': fig_bvc, 'cellprops': fig_cellprops,
               'manifold': fig_manifold, 'lesion': fig_lesion,
               'fig4': fig4_geometry, 'fig5': fig5_generality, 'fig6': fig6_brain,
               'init': figS_init, 'units': figS_units, 'celltypes': figS_celltypes,
               'robustness': figS_robustness, 'prospective': figS_prospective, 'ca1': figS_ca1,
               'coset_phase': figS_coset_phase, 'compartments_solo': fig_compartments_solo,
               'horizon_solo': fig_horizon_solo}
    for name, fn in allfigs.items():
        if a.only and name not in a.only:
            continue
        print(f'[{name}]')
        fn(data, figs)


if __name__ == '__main__':
    main()
