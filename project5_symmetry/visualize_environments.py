"""
Visualise all environment types used in project5_symmetry.

Produces a 4-panel figure:
  Panel A  — Arena shapes & sizes (top-down grid view)
  Panel B  — Landmark density sweep U=0..4  (18x18 square)
  Panel C  — Visual-field size comparison F=3/5/7 (sample obs)
  Panel D  — H2 aliasing distributions for key conditions

Run from repo root:
    python -m project5_symmetry.visualize_environments
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from project5_symmetry.environments.arena import (
    make_symmetry_env, get_obs_at, compute_H2,
)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_arena(wrapped_env, tile_px: int = 12) -> np.ndarray:
    """
    Return an RGB numpy array of the full top-down arena.
    Uses MiniGridEnv.get_frame() which does NOT require render_mode or Pyglet.
    """
    inner = wrapped_env.env
    inner.reset()
    # highlight=False keeps agent tile the same colour as floor
    frame = inner.get_frame(highlight=False, tile_size=tile_px)
    return frame          # (H*tile_px, W*tile_px, 3)  uint8


def render_agent_pov(wrapped_env, pos_xy, heading, F) -> np.ndarray:
    """
    Return the F×F×3 egocentric observation at (col, row), heading.
    Reshaped to (F, F, 3) for imshow.
    """
    wrapped_env.env.reset()
    flat = get_obs_at(wrapped_env, pos_xy, heading)   # (F*F*3,) float32 [0,1]
    return flat.reshape(F, F, 3)


def _centre_pos(wrapped_env):
    """Return a passable position near the centre of the arena."""
    positions = wrapped_env.env.passable_positions
    arr = np.array(positions)
    centre = arr.mean(axis=0)
    dists = np.linalg.norm(arr - centre, axis=1)
    return tuple(positions[dists.argmin()])


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — arena shapes and sizes
# ─────────────────────────────────────────────────────────────────────────────

PANEL_A_CONFIGS = [
    ('l_shape', 18, 3, 'L-shape 18×18\n(baseline)'),
    ('square',  12, 3, 'Square 12×12'),
    ('square',  18, 3, 'Square 18×18'),
    ('square',  24, 3, 'Square 24×24'),
    ('square',  30, 3, 'Square 30×30'),
]


def draw_panel_a(axes):
    for ax, (shape, size, U, label) in zip(axes, PANEL_A_CONFIGS):
        env = make_symmetry_env(shape, size, U, F=7, seed=0)
        img = render_arena(env, tile_px=10)
        ax.imshow(img)
        ax.set_title(label, fontsize=8, pad=3)
        ax.axis('off')
        # passable tile count
        n = len(env.env.passable_positions)
        ax.set_xlabel(f'{n} passable tiles', fontsize=7)
        ax.xaxis.set_label_position('top')


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — landmark density U=0..4 on 18×18 square
# ─────────────────────────────────────────────────────────────────────────────

PANEL_B_U = [0, 1, 2, 3, 4]
U_LABELS = [
    'U=0\n(uniform grey)',
    'U=1\n(+red)',
    'U=2\n(+blue)',
    'U=3\n(+yellow)\n[paper default]',
    'U=4\n(+green)',
]


def draw_panel_b(axes):
    for ax, U, label in zip(axes, PANEL_B_U, U_LABELS):
        env = make_symmetry_env('square', 18, U, F=7, seed=42)
        img = render_arena(env, tile_px=10)
        ax.imshow(img)
        ax.set_title(label, fontsize=7, pad=3)
        ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — visual field sizes F=3/5/7 (same position, 4 headings)
# ─────────────────────────────────────────────────────────────────────────────

PANEL_C_F = [3, 5, 7]
HEADINGS   = [0, 1, 2, 3]
HD_LABELS  = ['E', 'S', 'W', 'N']


def draw_panel_c(axes_grid):
    # axes_grid: (3 rows = F values) × (4 cols = headings)
    for row, F in enumerate(PANEL_C_F):
        env = make_symmetry_env('l_shape', 18, U=3, F=F, seed=0)
        env.reset()
        pos = _centre_pos(env)
        for col, (hd, hd_label) in enumerate(zip(HEADINGS, HD_LABELS)):
            ax = axes_grid[row][col]
            pov = render_agent_pov(env, pos, hd, F)
            ax.imshow(pov, interpolation='nearest')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'HD={hd_label}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'F={F}\n({F*F*3}d)', fontsize=8, rotation=0,
                              labelpad=35, va='center')


# ─────────────────────────────────────────────────────────────────────────────
# Panel D — H2 aliasing distributions for selected conditions
# ─────────────────────────────────────────────────────────────────────────────

PANEL_D_CONFIGS = [
    ('l_shape', 18, 3, 7, 'L-shape\nU=3, F=7'),
    ('square',  18, 0, 7, 'Square\nU=0, F=7'),
    ('square',  18, 1, 7, 'Square\nU=1, F=7'),
    ('square',  18, 3, 7, 'Square\nU=3, F=7'),
    ('square',  18, 3, 5, 'Square\nU=3, F=5'),
    ('square',  18, 3, 3, 'Square\nU=3, F=3'),
]

COLORS_D = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#795548']


def draw_panel_d(ax):
    for i, (shape, size, U, F, label) in enumerate(PANEL_D_CONFIGS):
        env = make_symmetry_env(shape, size, U, F=F, seed=0)
        env.reset()
        h2 = compute_H2(env)
        dist = np.array(h2['distribution'])
        mean = h2['mean']
        # KDE-style histogram (normalised)
        bins = np.arange(dist.max() + 2) - 0.5
        ax.hist(dist, bins=bins, density=True, alpha=0.55,
                color=COLORS_D[i], label=f'{label}  (μ={mean:.1f})',
                edgecolor='none')

    ax.set_xlabel('H2 aliasing count per state', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('H2 aliasing distributions', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Compose full figure
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('project5_symmetry — Environment Overview', fontsize=14,
                 fontweight='bold', y=0.98)

    outer = gridspec.GridSpec(4, 1, figure=fig,
                              hspace=0.45,
                              height_ratios=[1, 1, 1, 1])

    # ── Panel A: arena shapes/sizes ──────────────────────────────────────────
    ax_a_label = fig.add_subplot(outer[0])
    ax_a_label.axis('off')
    ax_a_label.set_title('A  —  Arena shapes & sizes (U=3, F=7, top-down view)',
                          fontsize=10, loc='left', pad=8, fontweight='bold')
    inner_a = gridspec.GridSpecFromSubplotSpec(
        1, len(PANEL_A_CONFIGS), subplot_spec=outer[0], wspace=0.08)
    axes_a = [fig.add_subplot(inner_a[0, j]) for j in range(len(PANEL_A_CONFIGS))]
    draw_panel_a(axes_a)

    # ── Panel B: landmark density ─────────────────────────────────────────────
    ax_b_label = fig.add_subplot(outer[1])
    ax_b_label.axis('off')
    ax_b_label.set_title('B  —  Landmark density sweep  (18×18 square, F=7)',
                          fontsize=10, loc='left', pad=8, fontweight='bold')
    inner_b = gridspec.GridSpecFromSubplotSpec(
        1, len(PANEL_B_U), subplot_spec=outer[1], wspace=0.08)
    axes_b = [fig.add_subplot(inner_b[0, j]) for j in range(len(PANEL_B_U))]
    draw_panel_b(axes_b)

    # ── Panel C: egocentric obs at F=3/5/7 ───────────────────────────────────
    ax_c_label = fig.add_subplot(outer[2])
    ax_c_label.axis('off')
    ax_c_label.set_title('C  —  Agent POV at F=3/5/7  (L-shape, U=3, centre position)',
                          fontsize=10, loc='left', pad=8, fontweight='bold')
    inner_c = gridspec.GridSpecFromSubplotSpec(
        len(PANEL_C_F), len(HEADINGS), subplot_spec=outer[2],
        wspace=0.05, hspace=0.15)
    axes_c = [[fig.add_subplot(inner_c[r, c])
               for c in range(len(HEADINGS))]
              for r in range(len(PANEL_C_F))]
    draw_panel_c(axes_c)

    # ── Panel D: H2 distributions ─────────────────────────────────────────────
    ax_d = fig.add_subplot(outer[3])
    ax_d.set_title('D  —  H2 aliasing distributions for key conditions',
                   fontsize=10, loc='left', pad=8, fontweight='bold')
    draw_panel_d(ax_d)

    plt.savefig('project5_symmetry/environment_overview.png',
                dpi=150, bbox_inches='tight')
    print('Saved: project5_symmetry/environment_overview.png')
    plt.show()


if __name__ == '__main__':
    main()
