"""Training-time convergence tracking for topology and geometry."""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import queue
import threading
import time
import warnings
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
from project4_topology_before_geometry.sensory.action_encoder import ActionEncoder
from project4_topology_before_geometry.sensory.aliasing_control import (
    compute_aliasing_score,
    compute_geo_euclidean_discrepancy,
)


def _topology_worker_fn(input_queue, output_queue, disable_cuda_devices: bool = True):
    """Compute persistent homology in a background CPU-only process."""
    if disable_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from project4_topology_before_geometry.evaluation.topological_metrics import (
        compute_betti_correct,
        compute_betti_numbers,
    )

    while True:
        item = input_queue.get()
        if item is None:
            break

        hidden_states = np.asarray(item["hidden_states"], dtype=np.float32)
        betti = compute_betti_numbers(hidden_states, seed=int(item.get("seed", 42)))
        result = {
            "trial": int(item["update_idx"]),
            "update": int(item["update_idx"]),
            "trajectories_seen": int(item.get("trajectories_seen", -1)),
            "betti_0": int(betti.get("betti_0", 0)),
            "betti_1": int(betti.get("betti_1", 0)),
            "persistence_gap_ratio_dim1": float(betti.get("persistence_gap_ratio_dim1", 0.0)),
            "betti_correct": bool(compute_betti_correct(betti, item["ground_truth_betti"])),
            "diagrams": betti.get("diagrams", []),
        }
        output_queue.put(result)


class AsyncTopologyWorker:
    """Persistent background process for ripser-based topology evaluation."""

    def __init__(self):
        self._thread_fallback = False
        try:
            ctx = mp.get_context("spawn")
            self.input_queue = ctx.Queue(maxsize=4)
            self.output_queue = ctx.Queue()
            self.process = ctx.Process(
                target=_topology_worker_fn,
                args=(self.input_queue, self.output_queue, True),
                daemon=True,
            )
            self.process.start()
        except Exception as exc:  # pragma: no cover - fallback for restricted local environments
            warnings.warn(
                f"AsyncTopologyWorker could not start a process ({exc}). Falling back to a background thread for this environment.",
                stacklevel=2,
            )
            self._thread_fallback = True
            self.input_queue = queue.Queue(maxsize=4)
            self.output_queue = queue.Queue()
            self.process = threading.Thread(
                target=_topology_worker_fn,
                args=(self.input_queue, self.output_queue, False),
                daemon=True,
            )
            self.process.start()
        self._pending: dict[int, dict[str, Any]] = {}

    def submit(
        self,
        hidden_states: np.ndarray,
        update_idx: int,
        ground_truth_betti: dict[str, int],
        *,
        trajectories_seen: int | None = None,
        seed: int = 42,
    ) -> bool:
        if update_idx in self._pending:
            return False

        hidden_states = np.asarray(hidden_states, dtype=np.float32)
        if len(hidden_states) > 300:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(hidden_states), size=300, replace=False))
            hidden_states = hidden_states[keep]

        payload = {
            "hidden_states": hidden_states,
            "update_idx": int(update_idx),
            "trajectories_seen": int(trajectories_seen) if trajectories_seen is not None else -1,
            "ground_truth_betti": dict(ground_truth_betti),
            "seed": int(seed),
        }
        try:
            self.input_queue.put_nowait(payload)
        except queue.Full:
            return False
        self._pending[int(update_idx)] = payload
        return True

    def poll(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while True:
            try:
                result = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._pending.pop(int(result["update"]), None)
            results.append(result)
        return results

    def wait_for_all(self, timeout_s: float = 300.0) -> list[dict[str, Any]]:
        deadline = time.perf_counter() + float(timeout_s)
        results: list[dict[str, Any]] = []
        while self._pending and time.perf_counter() < deadline:
            results.extend(self.poll())
            if self._pending:
                time.sleep(0.05)
        results.extend(self.poll())
        return results

    def shutdown(self):
        try:
            self.input_queue.put(None, timeout=1.0) if hasattr(self.input_queue, "put") else None
        except Exception:
            pass
        if hasattr(self.process, "join"):
            self.process.join(timeout=10)


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
        self._logger = None
        self._topology_worker = AsyncTopologyWorker() if bool(self.cfg.get("async_topology", False)) else None

        discrepancy = compute_geo_euclidean_discrepancy(env)
        env.geo_euclidean_discrepancy = float(discrepancy["mean"])

    def attach_logger(self, logger) -> None:
        self._logger = logger

    def _empty_row(self, update: int, trajectories_seen: int | None = None) -> dict[str, Any]:
        return {
            "trial": int(update),
            "update": int(update),
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
            "geo_euclidean_discrepancy": float(getattr(self.env, "geo_euclidean_discrepancy", np.nan)),
            "betti_0": np.nan,
            "betti_1": np.nan,
            "persistence_gap_ratio_dim1": np.nan,
            "betti_correct": np.nan,
            "replay_fraction_consistent": np.nan,
            "replay_forward_reverse_ratio": np.nan,
        }

    def _append_row(self, row: dict[str, Any]) -> None:
        row = dict(row)
        self.rows.append(row)
        if self._logger is not None:
            self._logger.log(row)
            return

        write_header = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _drain_topology_results(self) -> None:
        if self._topology_worker is None:
            return
        for result in self._topology_worker.poll():
            topo_entry = {
                "trial": int(result["trial"]),
                "update": int(result["update"]),
                "trajectories_seen": int(result["trajectories_seen"]) if int(result["trajectories_seen"]) >= 0 else np.nan,
                "betti_0": int(result["betti_0"]),
                "betti_1": int(result["betti_1"]),
                "persistence_gap_ratio_dim1": float(result["persistence_gap_ratio_dim1"]),
                "betti_correct": bool(result["betti_correct"]),
            }
            self.topology_log.append(dict(topo_entry))
            row = self._empty_row(int(result["update"]), trajectories_seen=None)
            row.update(topo_entry)
            self._append_row(row)

    def _rollout_hidden(self, seed: int = 42):
        eval_steps = int(self.cfg.get("eval_rollout_steps", max(5 * int(self.cfg.get("seq_duration", 100)), 600)))
        rollout = self.env.sample_rollout(eval_steps, seed=seed)
        actions = self._action_encoder.encode(rollout.actions, rollout.headings)
        hidden = self.model.get_hidden_states(rollout.observations[None, ...], actions[None, ...])
        positions = rollout.positions[: len(hidden)]
        return rollout, hidden, positions, actions

    def evaluate(self, trial: int, loss: float, trajectories_seen: int | None = None) -> dict[str, Any]:
        self._drain_topology_results()

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
            "decode_error": compute_decoding_error(hidden, positions),
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
        self._drain_topology_results()
        if int(trial) in self.hidden_snapshots:
            hidden = self.hidden_snapshots[int(trial)]
        else:
            _, hidden, _, _ = self._rollout_hidden(seed=42)
            self.hidden_snapshots[int(trial)] = hidden

        if self._topology_worker is not None:
            submitted = self._topology_worker.submit(
                hidden,
                update_idx=int(trial),
                trajectories_seen=trajectories_seen,
                ground_truth_betti=self.ground_truth,
                seed=42,
            )
            return {
                "trial": int(trial),
                "update": int(trial),
                "trajectories_seen": int(trajectories_seen) if trajectories_seen is not None else np.nan,
                "betti_0": np.nan,
                "betti_1": np.nan,
                "persistence_gap_ratio_dim1": np.nan,
                "betti_correct": np.nan,
                "submitted": bool(submitted),
            }

        from project4_topology_before_geometry.evaluation.topological_metrics import (
            compute_betti_correct,
            compute_betti_numbers,
        )

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
        row = self._empty_row(int(trial), trajectories_seen=trajectories_seen)
        row.update(topo_row)
        self._append_row(row)
        return topo_row

    def maybe_update(self, trial: int, loss: float, trajectories_seen: int | None = None) -> dict[str, Any] | None:
        self._drain_topology_results()
        row = None
        if trial % int(self.cfg.get("eval_every_trials", 100)) == 0:
            row = self.evaluate(trial, loss, trajectories_seen=trajectories_seen)
        if trial % int(self.cfg.get("topo_eval_every_trials", 500)) == 0:
            self.evaluate_topology(trial, trajectories_seen=trajectories_seen)
        return row

    def get_hidden_snapshot(self, trial: int, **_) -> np.ndarray | None:
        return self.hidden_snapshots.get(int(trial))

    def finalize(self) -> dict[str, Any]:
        if self._topology_worker is not None:
            for result in self._topology_worker.wait_for_all(timeout_s=float(self.cfg.get("topology_wait_timeout_s", 300.0))):
                topo_entry = {
                    "trial": int(result["trial"]),
                    "update": int(result["update"]),
                    "trajectories_seen": int(result["trajectories_seen"]) if int(result["trajectories_seen"]) >= 0 else np.nan,
                    "betti_0": int(result["betti_0"]),
                    "betti_1": int(result["betti_1"]),
                    "persistence_gap_ratio_dim1": float(result["persistence_gap_ratio_dim1"]),
                    "betti_correct": bool(result["betti_correct"]),
                }
                self.topology_log.append(dict(topo_entry))
                row = self._empty_row(int(result["update"]), trajectories_seen=None)
                row.update(topo_entry)
                self._append_row(row)
            self._topology_worker.shutdown()
            self._topology_worker = None

        from project4_topology_before_geometry.evaluation.topological_metrics import (
            compute_convergence_gap,
            compute_geometry_convergence_step,
            compute_topology_convergence_step,
        )

        self.topology_log.sort(key=lambda row: int(row.get("update", row.get("trial", 0))))
        self.geometry_log.sort(key=lambda row: int(row.get("update", row.get("trial", 0))))
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

        if self._logger is not None:
            self._logger.close()
            self._logger = None
        return results

    def close(self) -> None:
        self._drain_topology_results()
        if self._topology_worker is not None:
            self._topology_worker.shutdown()
            self._topology_worker = None
        if self._logger is not None:
            self._logger.close()
            self._logger = None
