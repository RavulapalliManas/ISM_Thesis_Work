"""Does topology lock in before geometry, during training?

For each model we find two saturation steps from its checkpoint series:

    topo_step  the first checkpoint at which b1_hat == b1_true and STAYS correct thereafter
    geom_step  the first checkpoint at which metric fidelity reaches `frac` of its final
               value and stays there

"and stays" matters. A binary indicator that flickers correct-wrong-correct has not locked
in; taking the first hit would score noise as an early answer.

The comparison is PAIRED within a model -- the same network, the same run, two readouts --
so a Wilcoxon signed-rank test on (geom_step - topo_step) is the right test. An unpaired
test across models would be swamped by seed-to-seed variation in overall learning speed.

Step 0 is included, and is the whole interpretation:
    topo_step == 0  ->  the topology was never learned; the initialisation already had it
    topo_step >  0  ->  the topology emerged from training
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _series(g):
    """Steps and values, sorted, with the duplicate 'final' checkpoint dropped."""
    g = g[g.step.astype(str) != 'final'].copy()
    g['step'] = g.step.astype(int)
    return g.sort_values('step')


def first_stable(steps, ok):
    """First step from which `ok` is True for that step and every later one. nan if never."""
    steps, ok = np.asarray(steps), np.asarray(ok, dtype=bool)
    stable = np.cumprod(ok[::-1])[::-1].astype(bool)      # True where ok holds from here on
    idx = np.where(stable)[0]
    return float(steps[idx[0]]) if len(idx) else float('nan')


def saturation_step(steps, values, frac=0.9):
    """First step from which `values` stays at or above `frac` * final value. nan if never.

    Guards the degenerate case: if the final value is <= 0 the metric never rose, so there
    is nothing to saturate and the answer is nan, not step 0.
    """
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    final = values[-1]
    if not np.isfinite(final) or final <= 0:
        return float('nan')
    return first_stable(steps, values >= frac * final)


def per_model(df, frac=0.9):
    keys = [c for c in ('layout', 'variant', 'seed') if c in df.columns]
    rows = []
    for key, g in df.groupby(keys):
        g = _series(g)
        if len(g) < 2:
            continue
        r = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        r['b1_true'] = int(g.b1_true.iloc[0])
        r['topo_step'] = first_stable(g.step, g.b1_hat == g.b1_true)
        r['geom_step'] = saturation_step(g.step, g.metric, frac)
        r['decode_step'] = saturation_step(g.step, g.decode_r2, frac)
        r['b1_at_init'] = int(g.b1_hat.iloc[0] == g.b1_true.iloc[0])
        r['metric_final'] = float(g.metric.iloc[-1])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tda', required=True)
    ap.add_argument('--frac', type=float, default=0.9)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.tda)
    pm = per_model(df, a.frac)
    if a.out:
        pm.to_csv(a.out, index=False)

    print(f'{len(pm)} models\n')
    cols = [c for c in ('layout', 'variant') if c in pm.columns]
    if cols:
        print(pm.groupby(cols)[['topo_step', 'geom_step', 'b1_at_init', 'metric_final']]
              .mean().round(3).to_string())

    ok = pm.dropna(subset=['topo_step', 'geom_step'])
    if len(ok) >= 6:
        d = ok.geom_step - ok.topo_step
        stat, p = wilcoxon(ok.geom_step, ok.topo_step, alternative='greater')
        print(f'\npaired Wilcoxon, geom_step > topo_step:  W={stat:.1f}  p={p:.5f}  '
              f'(n={len(ok)}, median lag {np.median(d):.0f} steps)')
    dropped = len(pm) - len(ok)
    if dropped:
        print(f'  {dropped} model(s) never satisfied one of the criteria and were excluded '
              f'(reported, not silently dropped)')

    n_init = int(pm.b1_at_init.sum())
    print(f'\nb1 already correct at step 0 (random init): {n_init}/{len(pm)} models')
    if n_init:
        print('  -> for those, topology is INHERITED from the initialisation, not learned.')


if __name__ == '__main__':
    main()
