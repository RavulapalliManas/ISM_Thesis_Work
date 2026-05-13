from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from reasoning_geometry.common import (
    DistanceBundle,
    ReasoningReasoningExperimentConfig,
    GateFailedError,
    GateResult,
    ReasoningExample,
    TrajectoryBundle,
    ensure_dirs,
    pairwise_euclidean,
    save_json,
)
from reasoning_geometry.data.graph_utils import dependency_like_distance, logical_hop_distance
from reasoning_geometry.data.halueval import load_halueval
from reasoning_geometry.data.prontoqa import load_prontoqa
from reasoning_geometry.metrics.manifold import bootstrap_twonn, mds_stress, pca_variance_summary
from reasoning_geometry.metrics.srsa import batched_spearman_rsa, spearman_rsa
from reasoning_geometry.metrics.statistics import symmetric_permutation_test
from reasoning_geometry.models.embeddings import compute_surface_distances


def load_dataset_split(config: ReasoningExperimentConfig) -> Dict[str, List[ReasoningExample]]:
    if config.dataset == "prontoqa":
        return load_prontoqa(config)
    if config.dataset == "halueval":
        return load_halueval(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def prepare_output_dirs(config: ReasoningExperimentConfig) -> Dict[str, Path]:
    dirs = {
        "output": config.resolve_dir(config.output_dir),
        "trajectories": config.resolve_dir(config.trajectories_dir),
        "matrices": config.resolve_dir(config.matrices_dir),
        "figures": config.resolve_dir(config.figures_dir),
        "checkpoints": config.resolve_dir(config.checkpoints_dir),
        "tensorboard": config.resolve_dir(config.tensorboard_dir),
    }
    ensure_dirs(dirs.values())
    return dirs


def logical_distance_for_example(example: ReasoningExample, gpu_mode: bool = False) -> np.ndarray:
    if gpu_mode:
        return dependency_like_distance(example.steps)
    return logical_hop_distance(example.steps)


def build_distance_bundles(
    examples: Sequence[ReasoningExample],
    tokenizer=None,
    model=None,
    config: ReasoningExperimentConfig | None = None,
    gpu_mode: bool = False,
) -> Dict[str, DistanceBundle]:
    config = config or ReasoningExperimentConfig(
        model="phi-2",
        dataset="prontoqa",
        n_examples=0,
        n_val=0,
        pRNN_hidden=256,
        pRNN_layers=2,
        pRNN_steps=1,
        subspace_k=16,
        permutation_n=100,
        batch_size=1,
        device="cpu",
        dtype="float32",
    )
    bundles: Dict[str, DistanceBundle] = {}
    for example in examples:
        bundles[example.example_id] = DistanceBundle(
            logical=logical_distance_for_example(example, gpu_mode=gpu_mode),
            surface=compute_surface_distances(
                example.steps,
                tokenizer=tokenizer,
                model=model,
                device=config.device,
                dtype=config.torch_dtype,
                modes=config.surface_embeddings,
            ),
        )
    return bundles


def save_distance_bundles(path: str | Path, bundles: Mapping[str, DistanceBundle]) -> None:
    payload = {
        example_id: {"logical": bundle.logical, "surface": bundle.surface}
        for example_id, bundle in bundles.items()
    }
    save_json(path, payload)


def save_trajectories(directory: str | Path, bundles: Sequence[TrajectoryBundle]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for bundle in bundles:
        np.savez_compressed(
            directory / f"{bundle.example_id}.npz",
            hidden_states=bundle.hidden_states,
            label=np.int32(bundle.label),
            step_text=np.array(bundle.step_text, dtype=object),
            **{f"layer_{key}": value for key, value in bundle.layer_hidden_states.items()},
        )


def summarize_srsa(
    trajectories: Sequence[TrajectoryBundle],
    distances: Mapping[str, DistanceBundle],
) -> Dict[str, object]:
    hidden_matrices = [pairwise_euclidean(bundle.hidden_states) for bundle in trajectories]
    logical = [distances[bundle.example_id].logical for bundle in trajectories]
    logical_result = batched_spearman_rsa(hidden_matrices, logical)

    surface_results = {}
    for mode in next(iter(distances.values())).surface.keys():
        mats = [distances[bundle.example_id].surface[mode] for bundle in trajectories]
        surface_results[mode] = batched_spearman_rsa(hidden_matrices, mats)

    per_example = []
    for bundle, hidden_matrix in zip(trajectories, hidden_matrices):
        logical_rho, _ = spearman_rsa(hidden_matrix, distances[bundle.example_id].logical)
        surface_rhos = {
            mode: spearman_rsa(hidden_matrix, distances[bundle.example_id].surface[mode])[0]
            for mode in distances[bundle.example_id].surface
        }
        per_example.append(
            {
                "example_id": bundle.example_id,
                "logical": logical_rho,
                **surface_rhos,
            }
        )

    return {
        "logical": logical_result,
        "surface": surface_results,
        "per_example": per_example,
    }


def summarize_manifold(trajectories: Sequence[TrajectoryBundle]) -> Dict[str, object]:
    ids = []
    stresses = []
    pca_stats = []
    embeddings = {}
    for bundle in trajectories:
        ids.append(bootstrap_twonn(bundle.hidden_states))
        stress = mds_stress(bundle.hidden_states)
        stresses.append(stress["stress"])
        embeddings[bundle.example_id] = stress["embedding"]
        pca_stats.append(pca_variance_summary(bundle.hidden_states))
    return {"twonn": ids, "stress": stresses, "pca": pca_stats, "mds_embeddings": embeddings}


def gate1_result(srsa_summary: Mapping[str, object], example_distance: Mapping[str, DistanceBundle], trajectories: Sequence[TrajectoryBundle], n_permutations: int) -> GateResult:
    logical_mean = float(srsa_summary["logical"]["mean"])
    surface_means = [float(payload["mean"]) for payload in srsa_summary["surface"].values()]
    best_surface = max(surface_means) if surface_means else 0.0
    hidden_matrices = [pairwise_euclidean(bundle.hidden_states) for bundle in trajectories]
    logical_mats = [example_distance[bundle.example_id].logical for bundle in trajectories]
    observed = []
    null_values = []
    for hidden_matrix, logical_mat in zip(hidden_matrices, logical_mats):
        rho, _ = spearman_rsa(hidden_matrix, logical_mat)
        observed.append(rho)
    for _ in range(n_permutations):
        permuted_scores = []
        for logical_mat, hidden_matrix in zip(logical_mats, hidden_matrices):
            perm = np.random.permutation(logical_mat.shape[0])
            permuted = logical_mat[perm][:, perm]
            rho, _ = spearman_rsa(hidden_matrix, permuted)
            permuted_scores.append(rho)
        null_values.append(float(np.mean(permuted_scores)))
    observed_mean = float(np.mean(observed)) if observed else 0.0
    null_arr = np.asarray(null_values, dtype=np.float64)
    p_value = float(np.mean(null_arr >= observed_mean)) if null_values else 1.0
    passed = logical_mean > best_surface + 0.1 and p_value < 0.01
    return GateResult(
        name="Gate 1",
        passed=passed,
        statistic=logical_mean - best_surface,
        threshold=0.1,
        p_value=p_value,
        message=(
            "PASS: logical geometry exceeds surface similarity."
            if passed
            else "FAIL: logical geometry does not sufficiently exceed surface similarity."
        ),
    )


def gate3_result(best_val_srsa: float) -> GateResult:
    passed = best_val_srsa > 0.3
    return GateResult(
        name="Gate 3",
        passed=passed,
        statistic=best_val_srsa,
        threshold=0.3,
        p_value=None,
        message=(
            "PASS: pRNN learned logical structure."
            if passed
            else "FAIL: pRNN validation sRSA stayed below 0.3."
        ),
    )


def enforce_gates(gates: Sequence[GateResult], results_path: str | Path, payload: Mapping[str, object]) -> None:
    save_json(results_path, payload)
    failed = [gate for gate in gates if not gate.passed]
    if failed:
        joined = "; ".join(gate.message for gate in failed)
        raise GateFailedError(joined)
