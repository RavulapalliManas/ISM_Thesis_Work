"""
Environment visualiser for project5_symmetry.

Saves individual images AND a combined overview figure into an output folder.

Output folder layout
--------------------
<out_dir>/
  arenas/
    arena_lshape_18.png        — top-down render of each shape × size
    arena_square_12.png
    arena_square_18.png
    arena_square_24.png
    arena_square_30.png
  landmarks/
    landmark_U0.png            — 18×18 square at each U value
    landmark_U1.png
    ...
    landmark_U4.png
  pov/
    pov_F3_hd_E.png            — agent egocentric view at each F × heading
    pov_F5_hd_S.png
    ...  (12 images total: 3 F-values × 4 headings)
  h2/
    h2_distributions.png       — aliasing count histograms
    h2_heatmap_<condition>.png — per-state H2 heat-map overlaid on arena
  overview.png                 — all panels combined

Run from repo root:
    PYTHONPATH=. python project5_symmetry/visualize_environments.py
    PYTHONPATH=. python project5_symmetry/visualize_environments.py --out my_folder
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from tqdm import tqdm

from project5_symmetry.environments.arena import (
    make_symmetry_env, get_obs_at, compute_H2,
)

DEFAULT_OUT = 'project5_symmetry/env_viz'


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_arena(wrapped_env, tile_px: int = 12) -> np.ndarray:
    """Top-down RGB render — does not need render_mode or Pyglet."""
    inner = wrapped_env.unwrapped
    inner.reset()
    return inner.get_frame(highlight=False, tile_size=tile_px)


def render_pov(wrapped_env, pos_xy, heading, F) -> np.ndarray:
    """Agent egocentric F×F×3 observation at (col,row), heading."""
    wrapped_env.unwrapped.reset()
    flat = get_obs_at(wrapped_env, pos_xy, heading)   # float32 [0,1]
    return flat.reshape(F, F, 3)


def _centre_pos(wrapped_env):
    """Passable tile closest to the arena centroid."""
    positions = wrapped_env.unwrapped.passable_positions
    arr = np.array(positions, dtype=float)
    dists = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
    return tuple(positions[int(dists.argmin())])


# ─────────────────────────────────────────────────────────────────────────────
# Section A — arena shapes & sizes
# ─────────────────────────────────────────────────────────────────────────────

ARENA_CONFIGS = [
    ('l_shape', 18, 'L-shape 18×18\n(baseline)'),
    ('square',  12, 'Square 12×12'),
    ('square',  18, 'Square 18×18'),
    ('square',  24, 'Square 24×24'),
    ('square',  30, 'Square 30×30'),
]


def save_arenas(out_dir: str):
    folder = os.path.join(out_dir, 'arenas')
    for shape, size, label in tqdm(ARENA_CONFIGS, desc='Arenas', unit='env'):
        env = make_symmetry_env(shape, size, U=3, F=7, seed=0)
        img = render_arena(env, tile_px=12)
        n   = len(env.unwrapped.passable_positions)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(f'{n} passable tiles', fontsize=9)
        ax.axis('off')
        fname = f'arena_{shape}_{size}.png'
        _save(fig, os.path.join(folder, fname))

    tqdm.write(f'  ✓ {len(ARENA_CONFIGS)} arena images → {folder}/')


# ─────────────────────────────────────────────────────────────────────────────
# Section B — landmark density U = 0 … 4
# ─────────────────────────────────────────────────────────────────────────────

U_LABELS = {
    0: 'U=0  (uniform grey)',
    1: 'U=1  (+red)',
    2: 'U=2  (+blue)',
    3: 'U=3  (+yellow)  [paper]',
    4: 'U=4  (+green)',
}


def save_landmarks(out_dir: str):
    folder = os.path.join(out_dir, 'landmarks')
    for U in tqdm(range(5), desc='Landmark densities', unit='U'):
        env = make_symmetry_env('square', 18, U, F=7, seed=42)
        img = render_arena(env, tile_px=12)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_title(U_LABELS[U], fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')
        _save(fig, os.path.join(folder, f'landmark_U{U}.png'))

    tqdm.write(f'  ✓ 5 landmark images → {folder}/')


# ─────────────────────────────────────────────────────────────────────────────
# Section C — egocentric POV at F = 3 / 5 / 7  ×  4 headings
# ─────────────────────────────────────────────────────────────────────────────

F_VALUES   = [3, 5, 7]
HD_NAMES   = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}


def save_pov(out_dir: str):
    folder = os.path.join(out_dir, 'pov')
    total  = len(F_VALUES) * len(HD_NAMES)
    pbar   = tqdm(total=total, desc='POV frames', unit='img')

    for F in F_VALUES:
        env = make_symmetry_env('l_shape', 18, U=3, F=F, seed=0)
        env.reset()
        pos = _centre_pos(env)

        for hd, hd_name in HD_NAMES.items():
            pov = render_pov(env, pos, hd, F)

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(pov, interpolation='nearest')
            ax.set_title(f'F={F}  heading={hd_name}', fontsize=10,
                         fontweight='bold', pad=6)
            ax.set_xlabel(f'{F*F*3}-dim input', fontsize=8)
            ax.axis('off')
            _save(fig, os.path.join(folder, f'pov_F{F}_hd_{hd_name}.png'))
            pbar.update(1)

    pbar.close()
    tqdm.write(f'  ✓ {total} POV images → {folder}/')


# ─────────────────────────────────────────────────────────────────────────────
# Section D — H2 aliasing
# ─────────────────────────────────────────────────────────────────────────────

H2_CONFIGS = [
    ('l_shape', 18, 3, 7, 'L-shape  U=3 F=7'),
    ('square',  18, 0, 7, 'Square   U=0 F=7'),
    ('square',  18, 1, 7, 'Square   U=1 F=7'),
    ('square',  18, 3, 7, 'Square   U=3 F=7'),
    ('square',  18, 3, 5, 'Square   U=3 F=5'),
    ('square',  18, 3, 3, 'Square   U=3 F=3'),
]
H2_COLORS = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#795548']


def save_h2(out_dir: str):
    folder = os.path.join(out_dir, 'h2')

    # ── 1. Histogram of aliasing counts ──────────────────────────────────────
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))

    for (shape, size, U, F, label), color in tqdm(
            zip(H2_CONFIGS, H2_COLORS), total=len(H2_CONFIGS),
            desc='H2 histograms', unit='env'):

        env = make_symmetry_env(shape, size, U, F=F, seed=0)
        h2  = compute_H2(env)
        dist = np.array(h2['distribution'])
        mean = h2['mean']
        bins = np.arange(dist.max() + 2) - 0.5
        ax_hist.hist(dist, bins=bins, density=True, alpha=0.55,
                     color=color, label=f'{label}  μ={mean:.2f}',
                     edgecolor='none')

    ax_hist.set_xlabel('Aliasing count (# other states with identical obs)', fontsize=10)
    ax_hist.set_ylabel('Density', fontsize=10)
    ax_hist.set_title('H2 aliasing distributions', fontsize=12, fontweight='bold')
    ax_hist.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax_hist.spines[['top', 'right']].set_visible(False)
    _save(fig_hist, os.path.join(folder, 'h2_distributions.png'))

    # ── 2. Per-state H2 heat-maps overlaid on arena ──────────────────────────
    heatmap_configs = [
        ('l_shape', 18, 3, 7),
        ('square',  18, 0, 7),
        ('square',  18, 3, 7),
        ('square',  18, 3, 3),
    ]

    for shape, size, U, F in tqdm(heatmap_configs, desc='H2 heatmaps', unit='env'):
        env   = make_symmetry_env(shape, size, U, F=F, seed=0)
        h2    = compute_H2(env)
        inner = env.unwrapped

        # Build 2-D aliasing grid (average over 4 headings per tile)
        grid = np.zeros((size + 2, size + 2), dtype=float)
        dist = np.array(h2['distribution'])
        states = [(pos, hd) for pos in inner.passable_positions for hd in range(4)]
        for idx, ((col, row), _) in enumerate(states):
            grid[row, col] += dist[idx] / 4.0   # mean over headings

        # Arena background
        arena_img = render_arena(env, tile_px=10)
        h_px, w_px = arena_img.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                                 gridspec_kw={'width_ratios': [1, 1]})
        axes[0].imshow(arena_img)
        axes[0].set_title('Arena (top-down)', fontsize=9)
        axes[0].axis('off')

        im = axes[1].imshow(grid, cmap='hot_r', interpolation='nearest',
                            origin='upper')
        plt.colorbar(im, ax=axes[1], label='Mean H2 aliasing')
        axes[1].set_title('H2 aliasing per tile\n(mean over 4 headings)', fontsize=9)
        axes[1].axis('off')

        fig.suptitle(f'{shape}  {size}×{size}  U={U}  F={F}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        fname = f'h2_heatmap_{shape}_{size}_U{U}_F{F}.png'
        _save(fig, os.path.join(folder, fname))

    tqdm.write(f'  ✓ H2 images → {folder}/')


# ─────────────────────────────────────────────────────────────────────────────
# Combined overview figure
# ─────────────────────────────────────────────────────────────────────────────

def save_overview(out_dir: str):
    """One big figure with panels A-D side by side for quick inspection."""
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('project5_symmetry — Environment Overview',
                 fontsize=14, fontweight='bold', y=0.99)

    outer = gridspec.GridSpec(4, 1, figure=fig, hspace=0.5,
                              height_ratios=[1, 1, 1.2, 1.2])

    # ── Panel A ──────────────────────────────────────────────────────────────
    inner_a = gridspec.GridSpecFromSubplotSpec(
        1, len(ARENA_CONFIGS), subplot_spec=outer[0], wspace=0.06)
    for j, (shape, size, label) in enumerate(
            tqdm(ARENA_CONFIGS, desc='Overview A', leave=False)):
        ax = fig.add_subplot(inner_a[0, j])
        env = make_symmetry_env(shape, size, U=3, F=7, seed=0)
        ax.imshow(render_arena(env, tile_px=10))
        ax.set_title(label, fontsize=8, pad=3)
        ax.axis('off')
    fig.add_subplot(outer[0]).set_title(
        'A  —  Arena shapes & sizes  (U=3, F=7)',
        fontsize=10, fontweight='bold', loc='left', pad=10, color='#333')
    fig.axes[-1].axis('off')

    # ── Panel B ──────────────────────────────────────────────────────────────
    inner_b = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1], wspace=0.06)
    for U in tqdm(range(5), desc='Overview B', leave=False):
        ax = fig.add_subplot(inner_b[0, U])
        env = make_symmetry_env('square', 18, U, F=7, seed=42)
        ax.imshow(render_arena(env, tile_px=10))
        ax.set_title(U_LABELS[U], fontsize=7, pad=3)
        ax.axis('off')
    fig.add_subplot(outer[1]).set_title(
        'B  —  Landmark density  (18×18 square, F=7)',
        fontsize=10, fontweight='bold', loc='left', pad=10, color='#333')
    fig.axes[-1].axis('off')

    # ── Panel C ──────────────────────────────────────────────────────────────
    n_rows, n_cols = len(F_VALUES), len(HD_NAMES)
    inner_c = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer[2], wspace=0.04, hspace=0.12)
    for r, F in enumerate(tqdm(F_VALUES, desc='Overview C', leave=False)):
        env = make_symmetry_env('l_shape', 18, U=3, F=F, seed=0)
        env.reset()
        pos = _centre_pos(env)
        for c, (hd, hd_name) in enumerate(HD_NAMES.items()):
            ax = fig.add_subplot(inner_c[r, c])
            ax.imshow(render_pov(env, pos, hd, F), interpolation='nearest')
            ax.axis('off')
            if r == 0:
                ax.set_title(f'HD={hd_name}', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'F={F}', fontsize=8, rotation=0,
                              labelpad=28, va='center')
    fig.add_subplot(outer[2]).set_title(
        'C  —  Agent POV  (L-shape, U=3, centre tile)',
        fontsize=10, fontweight='bold', loc='left', pad=10, color='#333')
    fig.axes[-1].axis('off')

    # ── Panel D ──────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(outer[3])
    for (shape, size, U, F, label), color in tqdm(
            zip(H2_CONFIGS, H2_COLORS), total=len(H2_CONFIGS),
            desc='Overview D', leave=False):
        env  = make_symmetry_env(shape, size, U, F=F, seed=0)
        h2   = compute_H2(env)
        dist = np.array(h2['distribution'])
        bins = np.arange(dist.max() + 2) - 0.5
        ax_d.hist(dist, bins=bins, density=True, alpha=0.55,
                  color=color, label=f'{label}  μ={h2["mean"]:.2f}',
                  edgecolor='none')
    ax_d.set_xlabel('H2 aliasing count', fontsize=9)
    ax_d.set_ylabel('Density', fontsize=9)
    ax_d.set_title('D  —  H2 aliasing distributions',
                   fontsize=10, fontweight='bold', loc='left', pad=6, color='#333')
    ax_d.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax_d.spines[['top', 'right']].set_visible(False)

    out_path = os.path.join(out_dir, 'overview.png')
    _save(fig, out_path, dpi=150)
    tqdm.write(f'  ✓ Combined overview → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate environment visualisation images'
    )
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'Output folder (default: {DEFAULT_OUT})')
    parser.add_argument('--skip-h2', action='store_true',
                        help='Skip H2 computation (slow for large arenas)')
    parser.add_argument('--skip-overview', action='store_true',
                        help='Skip combined overview figure')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'\nSaving all images to:  {os.path.abspath(args.out)}/\n')

    save_arenas(args.out)
    save_landmarks(args.out)
    save_pov(args.out)

    if not args.skip_h2:
        save_h2(args.out)
    else:
        print('  (H2 skipped)')

    if not args.skip_overview:
        print('\nBuilding combined overview figure...')
        save_overview(args.out)

    # ── Print file tree ───────────────────────────────────────────────────────
    print(f'\n{"─"*50}')
    print(f'Done.  Output folder: {os.path.abspath(args.out)}/')
    for root, dirs, files in os.walk(args.out):
        dirs.sort()
        level = root.replace(args.out, '').count(os.sep)
        indent = '  ' * level
        rel = os.path.relpath(root, args.out)
        if rel != '.':
            print(f'{indent}{os.path.basename(root)}/')
        for f in sorted(files):
            print(f'{"  "*(level+1)}{f}')
    print(f'{"─"*50}\n')


if __name__ == '__main__':
    main()
