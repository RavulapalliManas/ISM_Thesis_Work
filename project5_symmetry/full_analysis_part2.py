#!/usr/bin/env python3
"""
Part 2: Publication-quality figures.
Refactored to meet high-end journal standards (Nature, Science, Cell).
"""

import pickle, json, warnings, os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

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
    with open(RESULTS_DIR / '_ablation.pkl', 'rb') as f:
        ablation_data = pickle.load(f)
except FileNotFoundError:
    print("Warning: Data files not found. Please run Part 1 first.")
    df = pd.DataFrame()
    stat_df = pd.DataFrame()
    conditions = {}
    ablation_data = {}

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
    'HD_FULL': '#4878CF',
    'HD_DEGRADED': '#C4AD66',
    'HD_ABLATED': '#D65F5F'
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
    print(f"Saved: {name}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Training Dynamics
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("\n--- Figure 1: Training Curves ---")
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    for cond in ['S1', 'S2', 'S4']:
        all_steps, all_srsa = [], []
        if cond in conditions:
            for rec in conditions[cond]:
                log = rec.get('training_log')
                if not log: continue
                steps = np.array(log.get('steps', []))
                srsa = np.array(log.get('srsa_euclid', log.get('srsa', [])))
                if len(steps) == 0 or len(srsa) == 0: continue
                n = min(len(steps), len(srsa))
                all_steps.append(steps[:n]); all_srsa.append(srsa[:n])
        if all_steps:
            max_step = max(s[-1] for s in all_steps)
            common = np.linspace(0, max_step, 200)
            interp = np.array([np.interp(common, s, v) for s, v in zip(all_steps, all_srsa)])
            mean, sem = interp.mean(0), interp.std(0)/np.sqrt(len(interp))
            ax.plot(common/1000, mean, color=PALETTE[cond], label=cond, lw=1.8)
            ax.fill_between(common/1000, mean-sem, mean+sem, color=PALETTE[cond], alpha=0.15, lw=0)
    ax.axhline(0.40, color='#aaaaaa', ls='--', lw=0.6, alpha=0.5, zorder=0)
    ax.set_xlabel('Training steps ($10^3$)'); ax.set_ylabel('Spatial sRSA')
    ax.set_title('Neural Map Formation', pad=12); ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 1.0); style_ax(ax)
    save_fig(fig, 'fig1_training')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Metric Summary
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("--- Figure 2: Metric Summary ---")
    metrics_info = [('srsa', 'sRSA', (0, 1.0)), ('ra', 'RA', None), ('sci', 'SCI', (0, 0.5)),
                    ('c2_contrast', 'C2 Contrast', None), ('decode_error', 'Error (m)', None)]
    sym_df = df[df['experiment'] == 'symmetry_sweep']
    cond_order = ['S1', 'S2', 'S4']
    fig, axes = plt.subplots(1, len(metrics_info), figsize=(7.0, 2.2), constrained_layout=True)
    for i, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, metrics_info)):
        all_vals = [sym_df[sym_df['condition']==c][metric].dropna().values for c in cond_order]
        means = [np.mean(v) if len(v) else 0 for v in all_vals]
        sems = [np.std(v)/np.sqrt(len(v)) if len(v) > 1 else 0 for v in all_vals]
        ax.bar(range(3), means, color=[PALETTE[c] for c in cond_order], alpha=0.3, width=0.7, edgecolor='none')
        ax.errorbar(range(3), means, yerr=sems, fmt='none', color='#333333', lw=0.8)
        for xi, vals in zip(range(3), all_vals):
            ax.scatter(xi + np.random.uniform(-0.12, 0.12, size=len(vals)), vals, color=PALETTE[cond_order[xi]], s=10, alpha=0.7, lw=0)
        ax.set_xticks(range(3)); ax.set_xticklabels(cond_order)
        ax.set_title(ylabel, pad=8)
        if ylim: ax.set_ylim(ylim)
        style_ax(ax)
        if i == 0: add_panel_label(ax, 'A', x=-0.2)
    save_fig(fig, 'fig2_metrics')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Neural Manifolds
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("--- Figure 3: Population Geometry ---")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    for i, (ax, cond) in enumerate(zip(axes, ['S1', 'S2', 'S4'])):
        rec_list = conditions.get(cond, [])
        rec = rec_list[0] if rec_list else None
        if not rec or rec['H'] is None: 
            ax.axis('off')
            continue
        H, pos = rec['H'], rec['positions']
        try: H_2d = Isomap(n_components=2, n_neighbors=10).fit_transform(H)
        except: H_2d = PCA(n_components=2).fit_transform(H)
        color_vals = np.arctan2(pos[:,1]-pos[:,1].mean(), pos[:,0]-pos[:,0].mean())
        sc = ax.scatter(H_2d[:,0], H_2d[:,1], c=color_vals, cmap='twilight', s=4, alpha=0.7, lw=0)
        ax.set_title(cond, pad=6); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        style_ax(ax)
        if i == 0: add_panel_label(ax, 'A', x=-0.05)
    cax = fig.add_axes([1.01, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Heading (rad)', fontsize=8)
    cbar.set_ticks([-np.pi, 0, np.pi]); cbar.set_ticklabels(['-π', '0', 'π'])
    cbar.outline.set_visible(False)
    save_fig(fig, 'fig3_manifolds')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Rate Maps
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("--- Figure 4: Rate Maps ---")
    def get_rate_map(H_unit, positions, grid_size=20, sigma=0.8):
        pos_min, pos_max = positions.min(0), positions.max(0)
        pn = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        xi = (pn[:,0]*(grid_size-1)).astype(int).clip(0,grid_size-1)
        yi = (pn[:,1]*(grid_size-1)).astype(int).clip(0,grid_size-1)
        rm, cm = np.zeros((grid_size,grid_size)), np.zeros((grid_size,grid_size))
        for i in range(len(positions)):
            rm[yi[i],xi[i]] += H_unit[i]; cm[yi[i],xi[i]] += 1
        mask = cm > 0; rm[mask] /= cm[mask]
        return gaussian_filter(rm, sigma=sigma)
    N_EX = 5
    fig, axes = plt.subplots(3, N_EX, figsize=(5.0, 3.2), constrained_layout=True)
    for row, cond in enumerate(['S1', 'S2', 'S4']):
        rec_list = conditions.get(cond, [])
        rec = rec_list[0] if rec_list else None
        if not rec or rec['H'] is None:
            for ci in range(N_EX): axes[row,ci].axis('off')
            continue
        H, pos = rec['H'], rec['positions']
        vars = [np.var(get_rate_map(H[:,u], pos)) for u in range(H.shape[1])]
        top_units = np.argsort(vars)[-N_EX:][::-1]
        for ci, unit in enumerate(top_units):
            ax = axes[row, ci]
            rm = get_rate_map(H[:,unit], pos)
            ax.imshow(rm, cmap='magma', interpolation='nearest', origin='lower', aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0: ax.set_ylabel(cond, rotation=0, ha='right', va='center', labelpad=10)
            for spine in ax.spines.values(): spine.set_visible(False)
    save_fig(fig, 'fig4_rate_maps')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Ablation
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("--- Figure 5: HD Ablation ---")
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.5), constrained_layout=True)
    abl_df = df[df['experiment'] == 'hd_ablation']
    abl_order = ['HD_FULL', 'HD_DEGRADED', 'HD_ABLATED']
    ax = axes[0]
    sub_df = abl_df[abl_df['condition'].isin(abl_order)]
    if not sub_df.empty:
        means = [abl_df[abl_df['condition']==c]['srsa'].mean() for c in abl_order]
        sems = [abl_df[abl_df['condition']==c]['srsa'].std()/np.sqrt(5) for c in abl_order]
        ax.bar(range(3), means, color=[PALETTE[c] for c in abl_order], alpha=0.3, width=0.7, edgecolor='none')
        ax.errorbar(range(3), means, yerr=sems, fmt='none', color='#333333', lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(['Full', 'Deg.', 'None'])
        ax.set_ylabel('sRSA'); style_ax(ax)
    ax = axes[1]
    for cond in abl_order:
        curves = []
        for rec in ablation_data.get(cond, []):
            log = rec.get('training_log')
            if log and 'srsa_euclid' in log: curves.append((np.array(log['steps']), np.array(log['srsa_euclid'])))
        if not curves: continue
        common = np.linspace(0, 80000, 200)
        interp = np.array([np.interp(common, c[0], c[1]) for c in curves])
        ax.plot(common/1000, interp.mean(0), color=PALETTE[cond], lw=1.5)
    ax.set_xlabel('Steps ($10^3$)'); ax.set_ylim(0, 1); style_ax(ax)
    save_fig(fig, 'fig5_ablation')

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Robustness
# ═══════════════════════════════════════════════════════════════════════════
if not df.empty:
    print("--- Figure 6: Epsilon Sweep ---")
    eps_df = df[df['experiment'] == 'epsilon_sweep']
    if not eps_df.empty:
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        stats = eps_df.groupby('epsilon')['srsa'].agg(['mean', 'sem']).reset_index()
        ax.plot(stats['epsilon'], stats['mean'], '-o', color='#4878CF', lw=1.8, markersize=4)
        ax.fill_between(stats['epsilon'], stats['mean']-stats['sem'], stats['mean']+stats['sem'], color='#4878CF', alpha=0.15, lw=0)
        ax.set_xlabel('Path Integration Noise ($\epsilon$)'); ax.set_ylabel('Spatial sRSA')
        ax.set_ylim(0, 1); style_ax(ax)
        save_fig(fig, 'fig6_robustness')

print("\n" + "="*60 + "\nREFAC COMPLETE\n" + "="*60)
