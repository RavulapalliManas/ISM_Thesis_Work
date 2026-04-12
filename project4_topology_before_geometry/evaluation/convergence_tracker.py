"""Training-time convergence tracking for topology and geometry."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from project4_topology_before_geometry.evaluation.drift_tracker import compute_rdm_delta
from project4_topology_before_geometry.evaluation.geometric_metrics import (
    compute_decoding_error,
    compute_explained_variance_spatial,
    compute_participation_ratio,
    compute_place_field_coverage_map,
    compute_spatial_information,
    compute_srsa,
    compute_sw_dist,
)
from project4_topology_before_geometry.evaluation.replay_decoder import decode_replay_trajectory, fit_position_decoder
from project4_topology_before_geometry.evaluation.topological_metrics import (
    compute_betti_correct,
    compute_betti_numbers,
    compute_convergence_gap,
    compute_geometry_convergence_step,
    compute_topology_convergence_step,
)
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.sensory.aliasing_control import (
    compute_aliasing_score,
    compute_geo_euclidean_discrepancy,
)


class ConvergenceTracker:
    """Compute and persist geometry and topology summaries during training."""

    def __init__(self, env, model, ground_truth: dict[str, int], cfg: dict[str, Any]):
        self.env = env
        self.model = model
        self.ground_truth = dict(ground_truth)
        self.cfg = dict(cfg)
        self.log_dir = Path(self.cfg.get("log_dir", "project4_topology_before_geometry/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        seed_value = self.cfg.get("seed", 42)
        if isinstance(seed_value, list):
            seed_value = seed_value[0]
        self.seed = int(seed_value)
        self.run_name = f"{env.env_name}_{self.seed}"
        self.log_path = self.log_dir / f"{self.run_name}.csv"
        self.summary_path = self.log_dir / f"{self.run_name}_summary.csv"
        self.geometry_log: list[dict[str, Any]] = []
        self.topology_log: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.hidden_snapshots: dict[int, np.ndarray] = {}
        self._last_hidden: np.ndarray | None = None
        self._action_encoder = ActionEncoder(backend=env.backend)
        self._aliasing_score: float | None = None
        discrepancy = compute_geo_euclidean_discrepancy(env)
        env.geo_euclidean_discrepancy = float(discrepancy["mean"])

    def _rollout_hidden(self, seed: int = 42):
        eval_steps = int(self.cfg.get("eval_rollout_steps", max(5 * int(self.cfg.get("seq_duration", 100)), 600)))
        rollout = self.env.sample_rollout(eval_steps, seed=seed)
        actions = self._action_encoder.encode(rollout.actions, rollout.headings)
        hidden = self.model.get_hidden_states(rollout.observations[None, ...], actions[None, ...])
        positions = rollout.positions[: len(hidden)]
        return rollout, hidden, positions, actions

    def _append_row(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))
        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def evaluate(self, trial: int, loss: float, trajectories_seen: int | None = None) -> dict[str, Any]:
        rollout, hidden, positions, _ = self._rollout_hidden(seed=42)
        _, hidden_sleep = self.model.spontaneous(int(self.cfg.get("sleep_rollout_steps", 1000)))
        hidden_sleep_np = hidden_sleep.detach().cpu().numpy()
        if hidden_sleep_np.ndim > 2:
            hidden_sleep_np = hidden_sleep_np.reshape(-1, hidden_sleep_np.shape[-1])

        spatial_info = compute_spatial_information(hidden, positions)
        evs = compute_explained_variance_spatial(hidden, positions)
        decoder = fit_position_decoder(hidden, positions)
        replay = decode_replay_trajectory(hidden_sleep_np, decoder, self.env)

        row = {
            "trial": int(trial),
            "update": int(trial),
            "trajectories_seen": int(trajectories_seen) if trajectories_seen is not None else np.nan,
            "loss": float(loss),
            "srsa_euclidean": compute_srsa(hidden, positions, self.env, distance_type="euclidean"),
            "srsa_geodesic": compute_srsa(hidden, positions, self.env, distance_type="geodesic"),
            "decode_error": compute_decoding_error(
                hidden,
                positions,
                device=self.cfg.get("device"),
                n_updates=int(self.cfg.get("decoder_updates", 250)),
                batch_size=int(self.cfg.get("decoder_batch_size", 1024)),
            ),
            "participation_ratio": compute_participation_ratio(hidden),
            "spatial_info": float(spatial_info["mean_spatial_information"]),
            "fraction_tuned": float(evs["fraction_tuned"]),
            "sw_dist": compute_sw_dist(hidden, hidden_sleep_np),
            "rdm_frobenius_delta": compute_rdm_delta(self._last_hidden, hidden) if self._last_hidden is not None else 0.0,
            "aliasing_score": float(self._aliasing_score if self._aliasing_score is not None else compute_aliasing_score(self.env)),
            "geo_euclidean_discrepancy": float(self.env.geo_euclidean_discrepancy),
            "betti_0": np.nan,
            "betti_1": np.nan,
            "persistence_gap_ratio_dim1": np.nan,
            "betti_correct": np.nan,
            "replay_fraction_consistent": float(replay["fraction_geometrically_consistent"]),
            "replay_forward_reverse_ratio": float(replay["forward_vs_reverse_ratio"]),
        }
        if self._aliasing_score is None:
            self._aliasing_score = float(row["aliasing_score"])
        self._last_hidden = hidden
        self.hidden_snapshots[int(trial)] = hidden
        self.geometry_log.append(dict(row))
        self._append_row(row)
        return row

    def evaluate_topology(self, trial: int, trajectories_seen: int | None = None) -> dict[str, Any]:
        if int(trial) in self.hidden_snapshots:
            hidden = self.hidden_snapshots[int(trial)]
        else:
            _, hidden, _, _ = self._rollout_hidden(seed=42)
            self.hidden_snapshots[int(trial)] = hidden

        betti = compute_betti_numbers(hidden, seed=42)
        topo_row = {
            "trial": int(trial),
            "update": int(trial),
            "trajectories_seen": int(trajectories_seen) if trajectories_seen is not None else np.nan,
            "betti_0": int(betti.get("betti_0", 0)),
            "betti_1": int(betti.get("betti_1", 0)),
            "persistence_gap_ratio_dim1": float(betti.get("persistence_gap_ratio_dim1", 0.0)),
            "betti_correct": bool(compute_betti_correct(betti, self.ground_truth)),
        }
        self.topology_log.append(dict(topo_row))
        if self.rows and self.rows[-1]["trial"] == int(trial):
            self.rows[-1].update(topo_row)
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[-1].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
        else:
            filler = {
                "trial": int(trial),
                "update": int(trial),
                "trajectories_seen": int(trajectories_seen) if trajectories_seen is not None else np.nan,
                "loss": np.nan,
                "srsa_euclidean": np.nan,
                "srsa_geodesic": np.nan,
                "decode_error": np.nan,
                "participation_ratio": np.nan,
                "spatial_info": np.nan,
                "fraction_tuned": np.nan,
                "sw_dist": np.nan,
                "rdm_frobenius_delta": np.nan,
                "aliasing_score": float(self._aliasing_score or 0.0),
                "geo_euclidean_discrepancy": float(self.env.geo_euclidean_discrepancy),
                "replay_fraction_consistent": np.nan,
                "replay_forward_reverse_ratio": np.nan,
            }
            filler.update(topo_row)
            self._append_row(filler)
        return topo_row

    def maybe_update(self, trial: int, loss: float, trajectories_seen: int | None = None) -> dict[str, Any] | None:
        row = None
        if trial % int(self.cfg.get("eval_every_trials", 100)) == 0:
            row = self.evaluate(trial, loss, trajectories_seen=trajectories_seen)
        if trial % int(self.cfg.get("topo_eval_every_trials", 500)) == 0:
            self.evaluate_topology(trial, trajectories_seen=trajectories_seen)
        return row

    def get_hidden_snapshot(self, trial: int, **_) -> np.ndarray | None:
        return self.hidden_snapshots.get(int(trial))

    def finalize(self) -> dict[str, Any]:
        T_topology = compute_topology_convergence_step(self.topology_log, self.ground_truth)
        T_geometry = compute_geometry_convergence_step(self.geometry_log)
        gap = compute_convergence_gap(T_topology, T_geometry)
        coverage_map = None
        if self.geometry_log:
            _, hidden, positions, _ = self._rollout_hidden(seed=99)
            coverage_map = compute_place_field_coverage_map(hidden, positions, self.env)

        results = {
            "env_name": self.env.env_name,
            "T_topology": T_topology,
            "T_geometry": T_geometry,
            "gap": gap,
            "n_accumulate": int(self.cfg.get("n_accumulate", 1)),
            "complexity_index": float(self.env.complexity_index),
            "geo_euclidean_discrepancy": float(self.env.geo_euclidean_discrepancy),
            "aliasing_score": float(self._aliasing_score or 0.0),
            "coverage_map_shape": None if coverage_map is None else tuple(np.asarray(coverage_map).shape),
        }
        with self.summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
        return results
