#!/usr/bin/env python3
"""
Part 3: Supplementary figures for ISM thesis.
Refactored to meet high-end journal standards (Nature, Science, Cell).
"""

import pickle, json, warnings, os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, gaussian_kde, binned_statistic_2d
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# 0. GLOBAL SETTINGS & RC PARAMS
# ═══════════════════════════════════════════════════════════════════════════
RESULTS_DIR = Path('results2/analysis_output')
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load saved data
try:
    df = pd.read_pickle(RESULTS_DIR / '_df.pkl')
    stat_df = pd.read_pickle(RESULTS_DIR / '_stat_df.pkl')
    with open(RESULTS_DIR / '_conditions.pkl', 'rb') as f:
        conditions = pickle.load(f)
except FileNotFoundError:
    print("Warning: Data files not found. Please run Part 1 first.")
    df = pd.DataFrame()
    stat_df = pd.DataFrame()
    conditions = {}

# Aesthetic North Star Typography & Global Style
mpl.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size'        : 9,
    'axes.labelsize'   : 9,
    'axes.titlesize'   : 10,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8,
    'figure.dpi'       : 300,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 0.8,
    'axes.edgecolor'   : '#333333',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})

# Qualitative Palette
PALETTE = {
    'S1': '#4878CF',
    'S2': '#C4AD66',
    'S4': '#D65F5F',
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. REUSABLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def style_ax(ax, despine=True, grid=False):
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
    ax.tick_params(direction='out', length=3, width=0.8, colors='#333333')
    if grid:
        ax.grid(True, alpha=0.15, lw=0.5, ls='--', zorder=0)

def add_panel_label(ax, label, x=-0.1, y=1.1):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='semibold', va='top', ha='right')

def save_fig(fig, name):
    fig.savefig(FIGURES_DIR / f'{name}.pdf', dpi=300)
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, facecolor='white', transparent=False)
    plt.close(fig)
    print(f"  Saved: {name}")

def make_rate_map(H_unit, positions, grid_size=20, sigma=0.8):
    pos_min, pos_max = positions.min(0), positions.max(0)
    pn = (positions - pos_min) / (pos_max - pos_min + 1e-8)
    xi = (pn[:,0]*(grid_size-1)).astype(int).clip(0, grid_size-1)
    yi = (pn[:,1]*(grid_size-1)).astype(int).clip(0, grid_size-1)
    rm, cm = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size))
    for i in range(len(positions)):
        rm[yi[i], xi[i]] += H_unit[i]
        cm[yi[i], xi[i]] += 1
    mask = cm > 0; rm[mask] /= cm[mask]
    return gaussian_filter(rm, sigma=sigma)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S1: Sorted RDMs (7.0 in wide)
# ═══════════════════════════════════════════════════════════════════════════
if conditions:
    print("\n=== Figure S1: RDMs (sorted) ===")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    for ax, cond in zip(axes, ['S1', 'S2', 'S4']):
        rec_list = conditions.get(cond, [])
        rec = rec_list[0] if rec_list else None
        if not rec or rec['H'] is None:
            ax.axis('off'); continue
        H, pos = rec['H'], rec['positions']
        angles = np.arctan2(pos[:,1]-pos[:,1].mean(), pos[:,0]-pos[:,0].mean())
        idx = np.argsort(angles)
        D = squareform(pdist(H[idx] - H.mean(0), 'cosine'))
        im = ax.imshow(D, cmap='viridis', aspect='equal', vmin=0, vmax=1.2, interpolation='nearest')
        ax.set_title(cond, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if cond == 'S4':
            n = len(D)
            for f in [0.25, 0.5, 0.75]:
                ax.axhline(f*n, color='w', lw=0.6, alpha=0.5)
                ax.axvline(f*n, color='w', lw=0.6, alpha=0.5)
    cax = fig.add_axes([1.01, 0.25, 0.012, 0.5])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Distance', fontsize=8)
    cbar.outline.set_visible(False)
    save_fig(fig, 'fig_s1_rdms')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S2: Decoding Error Maps (7.0 in wide)
# ═══════════════════════════════════════════════════════════════════════════
if conditions:
    print("=== Figure S2: Decoding Maps ===")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    for ax, cond in zip(axes, ['S1', 'S2', 'S4']):
        rec_list = conditions.get(cond, [])
        rec = rec_list[0] if rec_list else None
        if not rec or rec['H'] is None:
            ax.axis('off'); continue
        H, pos = rec['H'], rec['positions']
        knn = KNeighborsRegressor(n_neighbors=5).fit(H, pos)
        preds = knn.predict(H)
        errs = np.linalg.norm(preds - pos, axis=1)
        stat, x, y, _ = binned_statistic_2d(pos[:,0], pos[:,1], errs, bins=16)
        im = ax.imshow(stat.T, origin='lower', cmap='magma', interpolation='bilinear', aspect='equal')
        ax.set_title(cond, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    save_fig(fig, 'fig_s2_decoding')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S3: Intrinsic Dim (3.5 in wide)
# ═══════════════════════════════════════════════════════════════════════════
if conditions:
    print("=== Figure S3: Intrinsic Dim ===")
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for cond in ['S1', 'S2', 'S4']:
        curves = []
        for rec in conditions.get(cond, []):
            log = rec.get('training_log')
            if log and 'manifold_id' in log:
                curves.append((np.array(log['steps']), np.array(log['manifold_id'])))
        if not curves: continue
        common = np.linspace(0, 80000, 200)
        interp = np.array([np.interp(common, c[0], c[1]) for c in curves])
        ax.plot(common/1000, interp.mean(0), color=PALETTE[cond], label=cond, lw=1.8)
        ax.fill_between(common/1000, interp.mean(0)-interp.std(0)/np.sqrt(5), 
                        interp.mean(0)+interp.std(0)/np.sqrt(5), color=PALETTE[cond], alpha=0.15, lw=0)
    ax.set_xlabel('Steps ($10^3$)'); ax.set_ylabel('Intrinsic Dim')
    ax.legend(frameon=False); style_ax(ax)
    save_fig(fig, 'fig_s3_intrinsic_dim')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S4: Spectral Gap (4.5 in wide)
# ═══════════════════════════════════════════════════════════════════════════
if os.path.exists(RESULTS_DIR / 'spectral_gap_data.csv'):
    print("=== Figure S4: Spectral Gap ===")
    spec_df = pd.read_csv(RESULTS_DIR / 'spectral_gap_data.csv')
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.5), constrained_layout=True)
    
    # Gap comparison
    ax = axes[0]
    for i, cond in enumerate(['S1', 'S2', 'S4']):
        vals = spec_df[spec_df['condition']==cond]['spectral_gap'].values
        ax.bar(i, np.mean(vals), color=PALETTE[cond], alpha=0.3)
        ax.scatter([i]*len(vals), vals, color=PALETTE[cond], s=10, alpha=0.7)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['S1', 'S2', 'S4'])
    ax.set_ylabel('Spectral Gap'); style_ax(ax)
    
    # Gap vs sRSA
    ax = axes[1]
    for cond in ['S1', 'S2', 'S4']:
        sub = spec_df[spec_df['condition']==cond]
        ax.scatter(sub['spectral_gap'], sub['srsa'], color=PALETTE[cond], s=15, alpha=0.8, label=cond)
    ax.set_xlabel('Spectral Gap'); ax.set_ylabel('sRSA')
    style_ax(ax)
    save_fig(fig, 'fig_s4_spectral')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S5: Field Size (3.5 in wide)
# ═══════════════════════════════════════════════════════════════════════════
if conditions:
    print("=== Figure S5: Field Size ===")
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for cond in ['S1', 'S2', 'S4']:
        rec = conditions[cond][0] if conditions[cond] else None
        if not rec: continue
        H, pos = rec['H'], rec['positions']
        sizes = []
        for u in range(min(H.shape[1], 100)):
            rm = make_rate_map(H[:, u], pos)
            peak = rm.max()
            if peak > 0:
                area = (rm > 0.5 * peak).sum()
                sizes.append(np.sqrt(area))
        if sizes:
            kde = gaussian_kde(sizes)
            x = np.linspace(min(sizes), max(sizes), 100)
            ax.plot(x, kde(x), color=PALETTE[cond], label=cond, lw=1.5)
            ax.fill_between(x, kde(x), color=PALETTE[cond], alpha=0.1)
    ax.set_xlabel('Field width (normalized)'); ax.set_ylabel('Density')
    ax.legend(frameon=False); style_ax(ax)
    save_fig(fig, 'fig_s5_field_size')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE S6: Schematic (7.0 in wide)
# ═══════════════════════════════════════════════════════════════════════════
print("=== Figure S6: Schematic ===")
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
for i, ax in enumerate(axes):
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    circle = plt.Circle((0,0), 1, color='#dddddd', fill=False, lw=1)
    ax.add_artist(circle)
    if i == 0: # S1
        ax.scatter([0.7], [0], color=PALETTE['S1'], s=40)
        ax.set_title('Asymmetric (S1)')
    elif i == 1: # S4
        pts = [(0.7,0), (0,0.7), (-0.7,0), (0,-0.7)]
        for p in pts: ax.scatter(p[0], p[1], color=PALETTE['S4'], s=40)
        ax.set_title('Symmetric (S4)')
    else: # Folding
        ax.scatter([0.7, 0, -0.7, 0], [0, 0.7, 0, -0.7], color=PALETTE['S4'], s=40, alpha=0.3)
        ax.annotate('', xy=(0,0), xytext=(0.7,0), arrowprops=dict(arrowstyle='->', color='#333333'))
        ax.set_title('Folding Hypothesis')
    ax.axis('off')
save_fig(fig, 'fig_s6_schematic')

print("\n" + "="*60 + "\nSUPP FIX COMPLETE\n" + "="*60)
