from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

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
from reasoning_geometry.common import GateFailedError, ExperimentConfig, save_json, set_global_seed
from reasoning_geometry.logging.tb_logger import TensorBoardLogger
from reasoning_geometry.metrics.manifold import bootstrap_twonn
from reasoning_geometry.models.extractor import extract_teacher_forcing_trajectory, load_causal_lm
from reasoning_geometry.training.prnn import (
    apply_normalization,
    prnn_hidden_trajectory,
    save_prnn_checkpoint,
    train_prnn,
)
from reasoning_geometry.visualization.plots import (
    figure1_mds,
    figure2_srsa,
    figure3_intrinsic_dimension,
    save_figure,
    set_publication_style,
)


def run(config_path: str) -> Dict[str, object]:
    config = ExperimentConfig.from_yaml(config_path)
    set_global_seed(config.seed)
    set_publication_style()
    dirs = prepare_output_dirs(config)
    tb_logger = TensorBoardLogger(dirs["tensorboard"] / "cpu")

    splits = load_dataset_split(config)
    loaded_lm = load_causal_lm(config)
    distances_train = build_distance_bundles(splits["train"], loaded_lm.tokenizer, loaded_lm.model, config, gpu_mode=False)
    distances_val = build_distance_bundles(splits["val"], loaded_lm.tokenizer, loaded_lm.model, config, gpu_mode=False)

    trajectories_train = [
        extract_teacher_forcing_trajectory(example, loaded_lm, config) for example in splits["train"]
    ]
    trajectories_val = [
        extract_teacher_forcing_trajectory(example, loaded_lm, config) for example in splits["val"]
    ]
    save_trajectories(dirs["trajectories"] / "cpu_train", trajectories_train)
    save_trajectories(dirs["trajectories"] / "cpu_val", trajectories_val)
    save_distance_bundles(dirs["matrices"] / "cpu_train_distances.json", distances_train)
    save_distance_bundles(dirs["matrices"] / "cpu_val_distances.json", distances_val)

    srsa_train = summarize_srsa(trajectories_train, distances_train)
    manifold_train = summarize_manifold(trajectories_train[:20])

    gate1 = gate1_result(srsa_train, distances_train, trajectories_train, config.permutation_n)

    prnn_result = train_prnn(trajectories_train, trajectories_val, {k: v.logical for k, v in distances_val.items()}, config, tb_logger)
    gate3 = gate3_result(prnn_result["best_val_srsa"])
    prnn_model = prnn_result["model"]
    normalization = prnn_result["normalization"]
    gates = [gate1, gate3]

    prnn_trajectories = []
    for bundle in trajectories_val:
        latent = prnn_hidden_trajectory(
            prnn_model,
            apply_normalization(bundle.hidden_states, normalization),
            config.device,
        )
        prnn_trajectories.append(
            {
                "example_id": bundle.example_id,
                "hidden_states": latent,
                "label": bundle.label,
            }
        )

    results = {
        "config": config.to_dict(),
        "gates": [gate.__dict__ for gate in gates],
        "srsa_train": srsa_train,
        "manifold_train": manifold_train,
        "prnn": {"best_val_srsa": prnn_result["best_val_srsa"]},
        "prnn_trajectories": prnn_trajectories,
    }
    results_path = dirs["output"] / "cpu_results.json"
    enforce_gates(gates, results_path, results)

    save_prnn_checkpoint(
        dirs["checkpoints"] / "cpu_prnn.pt",
        prnn_model,
        normalization,
        {"best_val_srsa": prnn_result["best_val_srsa"], "config": config.to_dict()},
    )

    tb_logger.add_histogram(
        "srsa/logical",
        srsa_train["logical"]["values"],
        step=0,
    )
    for mode, payload in srsa_train["surface"].items():
        tb_logger.add_histogram(f"srsa/{mode}", payload["values"], step=0)

    embeddings = [manifold_train["mds_embeddings"][bundle.example_id] for bundle in trajectories_train[:20]]
    hop_colors = [np.arange(embedding.shape[0]) for embedding in embeddings]
    cluster_colors = []
    for bundle in trajectories_train[:20]:
        tfidf_distance = distances_train[bundle.example_id].surface["tfidf"]
        n_clusters = min(3, tfidf_distance.shape[0])
        if n_clusters <= 1:
            cluster_colors.append(np.zeros(tfidf_distance.shape[0], dtype=np.int32))
            continue
        clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=config.seed).fit_predict(tfidf_distance)
        cluster_colors.append(clusters)
    fig1 = figure1_mds(embeddings, hop_colors, cluster_colors)
    save_figure(fig1, dirs["figures"] / "figure1_cpu")

    surface_summary = {mode: payload["mean"] for mode, payload in srsa_train["surface"].items()}
    summary = {"logical": srsa_train["logical"]["mean"], **surface_summary}
    scatter = [
        (
            item["logical"],
            max(item[mode] for mode in surface_summary),
        )
        for item in srsa_train["per_example"]
    ]
    fig2 = figure2_srsa(summary, scatter)
    save_figure(fig2, dirs["figures"] / "figure2_cpu")

    prnn_id = [
        bootstrap_twonn(np.asarray(item["hidden_states"], dtype=np.float32), n_bootstrap=min(200, config.bootstrap_n))
        for item in prnn_trajectories[:20]
    ]
    random_walk_id = []
    for bundle in trajectories_train[:20]:
        walk = np.cumsum(np.random.randn(*bundle.hidden_states.shape).astype(np.float32), axis=0)
        random_walk_id.append(bootstrap_twonn(walk, n_bootstrap=min(200, config.bootstrap_n)))
    id_mean = np.mean([entry["estimate"] for entry in manifold_train["twonn"]])
    id_low = np.mean([entry["ci_low"] for entry in manifold_train["twonn"]])
    id_high = np.mean([entry["ci_high"] for entry in manifold_train["twonn"]])
    prnn_mean = np.mean([entry["estimate"] for entry in prnn_id]) if prnn_id else 0.0
    prnn_low = np.mean([entry["ci_low"] for entry in prnn_id]) if prnn_id else 0.0
    prnn_high = np.mean([entry["ci_high"] for entry in prnn_id]) if prnn_id else 0.0
    rw_mean = np.mean([entry["estimate"] for entry in random_walk_id]) if random_walk_id else 0.0
    rw_low = np.mean([entry["ci_low"] for entry in random_walk_id]) if random_walk_id else 0.0
    rw_high = np.mean([entry["ci_high"] for entry in random_walk_id]) if random_walk_id else 0.0
    fig3 = figure3_intrinsic_dimension(
        {
            "Reasoning": (id_mean, id_low, id_high),
            "pRNN": (prnn_mean, prnn_low, prnn_high),
            "Random walk": (rw_mean, rw_low, rw_high),
        }
    )
    save_figure(fig3, dirs["figures"] / "figure3_cpu")

    tb_logger.close()
    save_json(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "cpu_config.yaml"),
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
