"""
Phase 0 arena visualiser — generates all images for the P0 condition.

Phase 0 arena: L-shape 18×18, U=3, F=7

Images saved to  project5_symmetry/env_viz/phase0/
  01_topdown.png           — full top-down arena (landmark colours visible)
  02_passable_mask.png     — which tiles the agent can visit (L-shape cutout)
  03_pov_grid.png          — F=7 egocentric obs from 9 representative positions
                             × 4 headings (E/S/W/N)
  04_pov_centre_*.png      — individual POV images from arena centre, each heading
  05_h2_heatmap.png        — mean H2 aliasing count per tile (how confusable
                             is each location?)
  06_h2_histogram.png      — distribution of aliasing counts across all states

Run from repo root:
    PYTHONPATH=. python3 project5_symmetry/visualize_phase0.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm

from project5_symmetry.environments.arena import (
    make_symmetry_env, get_obs_at, compute_H2,
)
from project5_symmetry.experiments.configs import PHASE0

OUT_DIR   = 'project5_symmetry/env_viz/phase0'
TILE_PX   = 18          # pixels per tile for top-down renders
HD_NAMES  = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}
HD_ARROWS = {0: '→', 1: '↓', 2: '←', 3: '↑'}


def _save(fig, name: str, dpi: int = 150):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f'  saved  {name}')


def _make_env():
    cfg = PHASE0[0]
    return make_symmetry_env(cfg.arena_shape, cfg.arena_size, cfg.U, cfg.F, seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Top-down full arena
# ─────────────────────────────────────────────────────────────────────────────

def plot_topdown(env):
    inner = env.unwrapped
    inner.reset()
    frame = inner.get_frame(highlight=False, tile_size=TILE_PX)   # (H, W, 3) uint8

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(frame)
    ax.set_title('Phase 0 arena — L-shape 18×18  (U=3, F=7)\nTop-down view',
                 fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')

    # Legend for landmark colours
    landmark_patches = [
        mpatches.Patch(color=np.array([50, 50, 50])/255,    label='Grey  (U=0 base)'),
        mpatches.Patch(color=np.array([127, 0,   0])/255,   label='Red   (U≥1)'),
        mpatches.Patch(color=np.array([0,   0,   127])/255, label='Blue  (U≥2)'),
        mpatches.Patch(color=np.array([127, 127, 0])/255,   label='Yellow (U≥3) ← paper default'),
    ]
    ax.legend(handles=landmark_patches, loc='lower left',
              fontsize=8, framealpha=0.85, title='Landmark colours')

    n = len(inner.passable_positions)
    ax.text(0.98, 0.02, f'{n} passable tiles\n{n*4} states (×4 headings)',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    _save(fig, '01_topdown.png')


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Passable / impassable mask
# ─────────────────────────────────────────────────────────────────────────────

def plot_passable_mask(env):
    inner = env.unwrapped
    s     = inner.arena_size        # 18
    mask  = np.zeros((s + 2, s + 2), dtype=np.uint8)
    for col, row in inner.passable_positions:
        mask[row, col] = 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap='Blues', vmin=0, vmax=1, interpolation='nearest',
              origin='upper')

    # Grid lines
    for x in range(s + 3):
        ax.axvline(x - 0.5, color='white', lw=0.4)
    for y in range(s + 3):
        ax.axhline(y - 0.5, color='white', lw=0.4)

    ax.set_title('Passable tiles (L-shape mask)\nBlue = passable, White = wall',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(range(s + 2))
    ax.set_yticks(range(s + 2))
    ax.tick_params(labelsize=6)

    n = len(inner.passable_positions)
    ax.text(0.98, 0.02, f'{n} / {s*s} tiles passable\n(¾ of interior)',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    _save(fig, '02_passable_mask.png')


# ─────────────────────────────────────────────────────────────────────────────
# 03 — POV grid: 9 positions × 4 headings
# ─────────────────────────────────────────────────────────────────────────────

def _pick_positions(inner, n: int = 9):
    """
    Pick n representative passable positions spread across the L-shape:
    corners of the bottom-left quadrant, top of the left column,
    and the centre — sorted for a clean grid layout.
    """
    positions = inner.passable_positions
    arr = np.array(positions, dtype=float)
    centre = arr.mean(axis=0)

    # Divide the arena into a rough n-grid by k-means–style quantiles
    cols = np.percentile(arr[:, 0], np.linspace(0, 100, int(n**0.5) + 2)[1:-1])
    rows = np.percentile(arr[:, 1], np.linspace(0, 100, int(n**0.5) + 2)[1:-1])

    chosen = []
    for c_target in cols:
        for r_target in rows:
            dists = np.abs(arr[:, 0] - c_target) + np.abs(arr[:, 1] - r_target)
            chosen.append(tuple(positions[int(dists.argmin())]))

    # Fill up to n using spread-out extras
    if len(chosen) < n:
        dists_to_chosen = np.min(
            [np.linalg.norm(arr - np.array(p), axis=1) for p in chosen], axis=0)
        while len(chosen) < n:
            chosen.append(tuple(positions[int(dists_to_chosen.argmax())]))
            dists_to_chosen = np.min(
                [np.linalg.norm(arr - np.array(p), axis=1) for p in chosen], axis=0)

    return chosen[:n]


def plot_pov_grid(env):
    inner = env.unwrapped
    F     = inner.agent_view_size     # 7
    positions = _pick_positions(inner, n=9)

    n_pos = len(positions)
    n_hd  = 4
    fig, axes = plt.subplots(n_pos, n_hd, figsize=(n_hd * 2.2, n_pos * 2.2))
    fig.suptitle(f'Agent POV  —  F={F} (egocentric {F}×{F}×3)\n'
                 f'9 positions × 4 headings  (L-shape 18×18, U=3)',
                 fontsize=12, fontweight='bold', y=1.01)

    for r, pos in enumerate(positions):
        for c, (hd, hd_name) in enumerate(HD_NAMES.items()):
            ax  = axes[r][c]
            pov = get_obs_at(env, pos, hd).reshape(F, F, 3)
            ax.imshow(pov, interpolation='nearest')
            ax.axis('off')
            if r == 0:
                ax.set_title(f'HD={hd_name} {HD_ARROWS[hd]}', fontsize=9,
                             fontweight='bold')
            if c == 0:
                ax.set_ylabel(f'({pos[0]},{pos[1]})', fontsize=7,
                              rotation=0, labelpad=34, va='center')

    plt.tight_layout()
    _save(fig, '03_pov_grid.png', dpi=130)


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Individual centre-position POV images, one per heading
# ─────────────────────────────────────────────────────────────────────────────

def plot_pov_centre(env):
    inner = env.unwrapped
    inner.reset()
    F   = inner.agent_view_size
    arr = np.array(inner.passable_positions, dtype=float)
    ctr = tuple(inner.passable_positions[
        int(np.linalg.norm(arr - arr.mean(0), axis=1).argmin())])

    for hd, hd_name in HD_NAMES.items():
        pov = get_obs_at(env, ctr, hd).reshape(F, F, 3)

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(pov, interpolation='nearest')
        ax.set_title(
            f'Centre-tile POV  HD={hd_name} {HD_ARROWS[hd]}\n'
            f'pos=({ctr[0]},{ctr[1]})  F={F}  ({F*F*3}-dim input)',
            fontsize=10, fontweight='bold', pad=6)
        ax.axis('off')
        _save(fig, f'04_pov_centre_hd{hd_name}.png')


# ─────────────────────────────────────────────────────────────────────────────
# 05 — H2 aliasing heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_h2_heatmap(env, h2: dict):
    inner  = env.unwrapped
    s      = inner.arena_size
    states = [(pos, hd) for pos in inner.passable_positions for hd in range(4)]
    dist   = np.array(h2['distribution'])

    # Mean aliasing per tile (averaged over 4 headings)
    grid = np.full((s + 2, s + 2), np.nan)
    for idx, ((col, row), _) in enumerate(states):
        if np.isnan(grid[row, col]):
            grid[row, col] = 0.0
        grid[row, col] += dist[idx] / 4.0

    arena_img = inner.get_frame(highlight=False, tile_size=TILE_PX)
    H, W      = arena_img.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'H2 aliasing — L-shape 18×18  U=3  F=7\n'
                 f'mean={h2["mean"]:.3f}  n_states={h2["n_states"]}',
                 fontsize=12, fontweight='bold')

    # Left: arena backdrop
    axes[0].imshow(arena_img)
    axes[0].set_title('Arena (landmark colours)', fontsize=10)
    axes[0].axis('off')

    # Right: heat-map
    vmax = np.nanmax(grid) if np.nanmax(grid) > 0 else 1
    im   = axes[1].imshow(grid, cmap='YlOrRd', vmin=0, vmax=vmax,
                          interpolation='nearest', origin='upper')
    plt.colorbar(im, ax=axes[1], label='Mean aliasing count\n(avg over 4 headings)',
                 fraction=0.046, pad=0.04)
    axes[1].set_title('H2 aliasing per tile\n(0 = fully unique obs)', fontsize=10)
    axes[1].axis('off')

    # Annotate max-aliasing tile
    if vmax > 0:
        r_max, c_max = np.unravel_index(np.nanargmax(grid), grid.shape)
        axes[1].scatter(c_max, r_max, s=120, marker='*',
                        color='blue', zorder=5, label=f'max={vmax:.1f}')
        axes[1].legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    _save(fig, '05_h2_heatmap.png')


# ─────────────────────────────────────────────────────────────────────────────
# 06 — H2 aliasing histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_h2_histogram(h2: dict):
    dist = np.array(h2['distribution'])
    mean = h2['mean']
    n_zero_pct = 100.0 * (dist == 0).sum() / len(dist)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(dist.max() + 2) - 0.5
    ax.hist(dist, bins=bins, color='#2196F3', edgecolor='white', linewidth=0.5)
    ax.axvline(mean, color='red', lw=1.8, linestyle='--',
               label=f'mean = {mean:.3f}')
    ax.set_xlabel('Aliasing count  (# other states with identical observation)',
                  fontsize=10)
    ax.set_ylabel('Number of states', fontsize=10)
    ax.set_title('H2 aliasing distribution  —  L-shape 18×18  U=3  F=7',
                 fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95, f'{n_zero_pct:.1f}% of states\nhave unique obs (aliasing=0)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, '06_h2_histogram.png')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = PHASE0[0]
    print(f'\nPhase 0 arena:  {cfg.arena_shape}  {cfg.arena_size}×{cfg.arena_size}'
          f'  U={cfg.U}  F={cfg.F}  k={cfg.k}  T={cfg.T}')
    print(f'Output folder:  {os.path.abspath(OUT_DIR)}/\n')

    steps = [
        ('01 top-down',       lambda: plot_topdown(env)),
        ('02 passable mask',  lambda: plot_passable_mask(env)),
        ('03 POV grid',       lambda: plot_pov_grid(env)),
        ('04 POV centre',     lambda: plot_pov_centre(env)),
        ('05+06 H2',          None),          # handled separately (needs h2 dict)
    ]

    env = _make_env()
    env.reset()

    for label, fn in tqdm(steps[:-1], desc='Generating images', unit='plot'):
        fn()

    # H2 is computed once and shared between heatmap + histogram
    tqdm.write('  computing H2  (enumerating all state observations)…')
    h2 = compute_H2(env)
    tqdm.write(f'  H2 mean={h2["mean"]:.3f}  n_states={h2["n_states"]}')
    plot_h2_heatmap(env, h2)
    plot_h2_histogram(h2)

    # ── Print file list ───────────────────────────────────────────────────────
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith('.png'))
    print(f'\n{"─"*52}')
    print(f'Phase 0 images ({len(files)} files):')
    for f in files:
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f'  {f:40s}  {size_kb:4d} KB')
    print(f'{"─"*52}')
    print(f'Folder: {os.path.abspath(OUT_DIR)}/')


if __name__ == '__main__':
    main()
