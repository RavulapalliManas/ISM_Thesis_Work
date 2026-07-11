"""How fast does each initialisation learn, and does a preconfigured manifold predict it?

Speed is `steps_to_threshold`: the first training step at which a model's loss drops below
a target. The target must be one EVERY model reaches, or "never reached" masquerades as
"slow" -- so it is set to the worst final loss across the ensemble, i.e. the easiest common
target. Losses are noisy, so the curve is smoothed and the crossing is linearly interpolated
between logged steps rather than snapped to the logging grid.

Joined against `run_tda.py`'s step-0 row, this answers the mechanistic form of the
Guardamagna et al. (2026) claim: does an initialisation whose manifold ALREADY carries the
arena's loop (b1 correct at step 0, before any training) learn the geometry faster?
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

_trapz = getattr(np, 'trapezoid', None) or np.trapz   # renamed in numpy 2.0


def smooth(y, w=5):
    if len(y) < w:
        return np.asarray(y, dtype=float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(y, dtype=float), k, mode='same')


def steps_to_threshold(steps, loss, thresh, w=5):
    """First step whose smoothed loss is <= thresh, linearly interpolated. nan if never."""
    steps = np.asarray(steps, dtype=float)
    y = smooth(loss, w)
    below = np.where(y <= thresh)[0]
    if len(below) == 0:
        return float('nan')
    i = below[0]
    if i == 0:
        return float(steps[0])
    y0, y1 = y[i - 1], y[i]
    if y0 == y1:
        return float(steps[i])
    f = (y0 - thresh) / (y0 - y1)
    return float(steps[i - 1] + f * (steps[i] - steps[i - 1]))


def load_logs(root):
    out = []
    for p in sorted(Path(root).rglob('training_log.json')):
        d = json.loads(p.read_text())
        meta = d.get('meta', {})
        out.append({'variant': meta.get('variant', p.parent.parent.name),
                    'seed': meta.get('seed', int(p.parent.name.split('_')[1])),
                    'steps': d['steps'], 'loss': d['loss']})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    logs = load_logs(a.runs)
    if not logs:
        raise SystemExit(f'no training_log.json under {a.runs}')

    finals = [smooth(r['loss'])[-3:].mean() for r in logs]
    thresh = float(max(finals))       # the easiest target every model reaches
    print(f'{len(logs)} models; common threshold = worst final loss = {thresh:.6f}')

    rows = []
    for r in logs:
        rows.append({'variant': r['variant'], 'seed': r['seed'],
                     'final_loss': float(smooth(r['loss'])[-3:].mean()),
                     'threshold': thresh,
                     'steps_to_threshold': steps_to_threshold(r['steps'], r['loss'], thresh),
                     'loss_auc': float(_trapz(smooth(r['loss']), r['steps']))})
    rows.sort(key=lambda x: (x['variant'], x['seed']))

    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by.setdefault(r['variant'], []).append(r['steps_to_threshold'])
    print(f'\n{"variant":<10s} {"steps_to_thresh":>18s} {"final_loss":>12s}')
    for v in sorted(by, key=lambda v: np.nanmean(by[v])):
        fl = np.mean([r['final_loss'] for r in rows if r['variant'] == v])
        print(f'{v:<10s} {np.nanmean(by[v]):>18.0f} {fl:>12.6f}')
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
