from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from reasoning_geometry.analysis.baselines import (
    aggregate_detector,
    iti_probe_scores,
    nearest_neighbour_centroid_scores,
)
from reasoning_geometry.analysis.shared import (
    build_distance_bundles,
    enforce_gates,
    gate1_result,
    gate3_result,
    load_dataset_split,
    prepare_output_dirs,
    save_distance_bundles,
    save_trajectories,
    summarize_manifold,
    summarize_srsa,
)
from reasoning_geometry.common import GateFailedError, GateResult, ExperimentConfig, save_json, set_global_seed
from reasoning_geometry.logging.tb_logger import TensorBoardLogger
from reasoning_geometry.metrics.statistics import grassmann_permutation_test
from reasoning_geometry.metrics.subspace import canonical_angles, fit_pca_subspace, projection_dynamics_statistics, projection_ratio, signed_velocity
from reasoning_geometry.models.coconut import CoconutWrapper
from reasoning_geometry.models.extractor import extract_teacher_forcing_trajectory, load_causal_lm
from reasoning_geometry.training.prnn import (
    apply_normalization,
    prnn_hidden_trajectory,
    save_prnn_checkpoint,
    train_prnn,
)
from reasoning_geometry.visualization.plots import figure4_gpu, figure5_gpu, save_figure, set_publication_style


def gate2_result(observed: float, null_95: float, p_value: float) -> GateResult:
    passed = observed > null_95
    return GateResult(
        name="Gate 2",
        passed=passed,
        statistic=observed,
        threshold=null_95,
        p_value=p_value,
        message=(
            "PASS: factual and hallucination subspaces are separable."
            if passed
            else "FAIL: subspace separability was not established."
        ),
    )


def run(config_path: str) -> Dict[str, object]:
    config = ExperimentConfig.from_yaml(config_path)
    set_global_seed(config.seed)
    set_publication_style()
    dirs = prepare_output_dirs(config)
    tb_logger = TensorBoardLogger(dirs["tensorboard"] / "gpu")

    splits = load_dataset_split(config)
    loaded_lm = load_causal_lm(config)
    coconut = CoconutWrapper(loaded_lm, extract_layers=config.extract_layers)

    distances_train = build_distance_bundles(splits["train"], loaded_lm.tokenizer, loaded_lm.model, config, gpu_mode=True)
    distances_val = build_distance_bundles(splits["val"], loaded_lm.tokenizer, loaded_lm.model, config, gpu_mode=True)

    trajectories_train_teacher = [
        extract_teacher_forcing_trajectory(example, loaded_lm, config) for example in splits["train"]
    ]
    trajectories_val_teacher = [
        extract_teacher_forcing_trajectory(example, loaded_lm, config) for example in splits["val"]
    ]
    trajectories_train_cont = [coconut.extract_continuous_trajectory(example, config) for example in splits["train"]]
    trajectories_val_cont = [coconut.extract_continuous_trajectory(example, config) for example in splits["val"]]

    save_trajectories(dirs["trajectories"] / "gpu_train_teacher", trajectories_train_teacher)
    save_trajectories(dirs["trajectories"] / "gpu_val_teacher", trajectories_val_teacher)
    save_trajectories(dirs["trajectories"] / "gpu_train_cont", trajectories_train_cont)
    save_trajectories(dirs["trajectories"] / "gpu_val_cont", trajectories_val_cont)
    save_distance_bundles(dirs["matrices"] / "gpu_train_distances.json", distances_train)
    save_distance_bundles(dirs["matrices"] / "gpu_val_distances.json", distances_val)

    srsa_train = summarize_srsa(trajectories_train_cont, distances_train)
    manifold_train = summarize_manifold(trajectories_train_cont[:20])
    gate1 = gate1_result(srsa_train, distances_train, trajectories_train_cont, config.permutation_n)

    factual_states = np.concatenate([bundle.hidden_states for bundle in trajectories_train_cont if bundle.label == 0], axis=0)
    hall_states = np.concatenate([bundle.hidden_states for bundle in trajectories_train_cont if bundle.label == 1], axis=0)
    subspace_perm = grassmann_permutation_test(
        factual_states,
        hall_states,
        config.subspace_k,
        config.permutation_n,
        fit_pca_subspace,
        canonical_angles,
    )
    factual_basis = fit_pca_subspace(factual_states, config.subspace_k)
    hall_basis = fit_pca_subspace(hall_states, config.subspace_k)
    angle_bundle = canonical_angles(factual_basis, hall_basis)
    gate2 = gate2_result(
        float(subspace_perm["observed"]),
        float(subspace_perm["null_95"]),
        float(subspace_perm["p_value"]),
    )

    hall_ratios = [projection_ratio(bundle.hidden_states, factual_basis, hall_basis) for bundle in trajectories_val_cont if bundle.label == 1]
    fact_ratios = [projection_ratio(bundle.hidden_states, factual_basis, hall_basis) for bundle in trajectories_val_cont if bundle.label == 0]
    phi_stats = projection_dynamics_statistics(hall_ratios, fact_ratios)

    prnn_result = train_prnn(
        trajectories_train_cont,
        trajectories_val_cont,
        {k: v.logical for k, v in distances_val.items()},
        config,
        tb_logger,
    )
    gate3 = gate3_result(prnn_result["best_val_srsa"])

    prnn_model = prnn_result["model"]
    normalization = prnn_result["normalization"]
    gates = [gate1, gate2, gate3]
    partial_results = {
        "config": config.to_dict(),
        "gates": [gate.__dict__ for gate in gates],
        "srsa_train": srsa_train,
        "manifold_train": manifold_train,
        "subspace": {
            "permutation": subspace_perm,
            "canonical_angles": angle_bundle,
            "factual_basis_shape": list(factual_basis.shape),
            "hall_basis_shape": list(hall_basis.shape),
        },
        "projection_ratio": phi_stats,
        "prnn": {"best_val_srsa": prnn_result["best_val_srsa"]},
    }
    results_path = dirs["output"] / "gpu_results.json"
    enforce_gates(gates, results_path, partial_results)

    save_prnn_checkpoint(
        dirs["checkpoints"] / "gpu_prnn.pt",
        prnn_model,
        normalization,
        {"best_val_srsa": prnn_result["best_val_srsa"], "config": config.to_dict()},
    )

    y_true = [bundle.label for bundle in trajectories_val_cont]
    phi_min_scores = []
    phi_delta_scores = []
    entropy_scores = []
    stepwise_phi_delta = []
    for bundle in trajectories_val_cont:
        phi = projection_ratio(bundle.hidden_states, factual_basis, hall_basis)
        delta_phi = signed_velocity(phi)
        phi_min_scores.append(float(phi.min()))
        phi_delta_scores.append(float(delta_phi.mean()))
        stepwise_phi_delta.append(delta_phi.tolist())
        entropy_scores.append(float(np.mean(bundle.metadata.get("entropies", [0.0]))))

    iti_scores = iti_probe_scores(trajectories_train_cont, trajectories_val_cont, layer="16")
    nn_scores = nearest_neighbour_centroid_scores(trajectories_train_cont, trajectories_val_cont)

    detectors = {
        "mean_delta_phi": aggregate_detector("mean_delta_phi", y_true, [-score for score in phi_delta_scores], stepwise_phi_delta, greater_is_positive=False),
        "min_phi": aggregate_detector("min_phi", y_true, [-score for score in phi_min_scores]),
        "token_entropy": aggregate_detector("token_entropy", y_true, entropy_scores),
        "iti_probe": aggregate_detector("iti_probe", y_true, iti_scores),
        "nn_centroid": aggregate_detector("nn_centroid", y_true, nn_scores),
    }

    results = {
        "config": config.to_dict(),
        "gates": [gate.__dict__ for gate in gates],
        "srsa_train": srsa_train,
        "manifold_train": manifold_train,
        "subspace": {
            "permutation": subspace_perm,
            "canonical_angles": angle_bundle,
            "factual_basis_shape": list(factual_basis.shape),
            "hall_basis_shape": list(hall_basis.shape),
        },
        "projection_ratio": phi_stats,
        "detectors": {name: detector.__dict__ for name, detector in detectors.items()},
        "prnn": {"best_val_srsa": prnn_result["best_val_srsa"]},
    }

    np.save(dirs["output"] / "U_F.npy", factual_basis)
    np.save(dirs["output"] / "U_H.npy", hall_basis)

    detector_curves = {name: detector.curves for name, detector in detectors.items()}
    max_fact_len = max(map(len, fact_ratios)) if fact_ratios else 1
    max_hall_len = max(map(len, hall_ratios)) if hall_ratios else 1
    phi_factual = (
        np.nanmean(
            np.stack(
                [
                    np.pad(phi, (0, max(0, max_fact_len - len(phi))), constant_values=np.nan)
                    for phi in fact_ratios
                ]
            ),
            axis=0,
        )
        if fact_ratios
        else np.zeros(1, dtype=np.float32)
    )
    phi_hall = (
        np.nanmean(
            np.stack(
                [
                    np.pad(phi, (0, max(0, max_hall_len - len(phi))), constant_values=np.nan)
                    for phi in hall_ratios
                ]
            ),
            axis=0,
        )
        if hall_ratios
        else np.zeros(1, dtype=np.float32)
    )
    fig4 = figure4_gpu(angle_bundle["angles"], subspace_perm["null_95"], np.nan_to_num(phi_factual), np.nan_to_num(phi_hall), detector_curves)
    save_figure(fig4, dirs["figures"] / "figure4_gpu")

    delta_t_minus_2 = [scores[-2] if len(scores) >= 2 else scores[-1] for scores in stepwise_phi_delta]
    fig5 = figure5_gpu(delta_t_minus_2, y_true, {name: detector.detection_latency for name, detector in detectors.items()})
    save_figure(fig5, dirs["figures"] / "figure5_gpu")

    tb_logger.close()
    save_json(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "gpu_config.yaml"),
    )
    args = parser.parse_args()
    try:
        results = run(args.config)
        for gate in results["gates"]:
            print(f"{gate['name']}: {'PASS' if gate['passed'] else 'FAIL'} | {gate['message']}")
    except GateFailedError as exc:
        print(f"Validation failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
