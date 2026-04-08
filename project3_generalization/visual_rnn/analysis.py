from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def plot_loss_curves(history: Sequence[Mapping[str, float]]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [entry["epoch"] for entry in history]
    ax.plot(epochs, [entry["train_prediction_loss"] for entry in history], label="Train Prediction", linewidth=2)
    ax.plot(epochs, [entry["val_prediction_loss"] for entry in history], label="Val Prediction", linewidth=2)
    ax.plot(epochs, [entry["val_rollout_loss"] for entry in history], label="Val Rollout", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Prediction and Rollout Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_decoding_error(history: Sequence[Mapping[str, float]]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [entry["epoch"] for entry in history]
    ax.plot(epochs, [entry["decoder_normalized_rmse"] for entry in history], color="tab:red", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized RMSE")
    ax.set_title("Decoded Trajectory Error")
    fig.tight_layout()
    return fig, ax


def plot_visual_prediction(true_patch: np.ndarray, pred_patch: np.ndarray) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    error_patch = np.abs(pred_patch - true_patch)
    axes[0].imshow(true_patch)
    axes[0].set_title("True")
    axes[1].imshow(pred_patch)
    axes[1].set_title("Predicted")
    axes[2].imshow(error_patch / max(float(error_patch.max()), 1e-6))
    axes[2].set_title("Abs Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig, axes


def plot_trajectory_reconstruction(
    true_positions: np.ndarray,
    decoded_positions: np.ndarray,
    replay_positions: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(true_positions[:, 0], true_positions[:, 1], color="black", linewidth=2, label="True")
    axes[0].plot(decoded_positions[:, 0], decoded_positions[:, 1], color="tab:blue", linewidth=2, alpha=0.85, label="Decoded")
    axes[0].set_title("Wake Trajectory")
    axes[0].legend(frameon=False)

    axes[1].plot(replay_positions[:, 0], replay_positions[:, 1], color="tab:green", linewidth=2)
    axes[1].set_title("Replay Trajectory")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, axes


def plot_hidden_embedding(
    embedding: np.ndarray,
    positions: np.ndarray,
    *,
    method: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = np.linspace(0.0, 1.0, len(embedding))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="viridis", s=10, alpha=0.8)
    ax.set_title(f"Hidden State {method.upper()}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig, ax


def plot_sample_tuning_curves(
    tuning_curves: np.ndarray,
    spatial_information: np.ndarray,
    *,
    max_units: int = 6,
) -> tuple[plt.Figure, np.ndarray]:
    n_units = min(max_units, tuning_curves.shape[0])
    ranked = np.argsort(spatial_information)[::-1][:n_units]
    cols = 3
    rows = int(np.ceil(n_units / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, unit_idx in zip(axes.ravel(), ranked):
        axis.imshow(tuning_curves[unit_idx], cmap="magma")
        axis.set_title(f"Unit {unit_idx}")
        axis.axis("off")
    fig.tight_layout()
    return fig, axes


def summarize_run(history: Sequence[Mapping[str, float]], final_metrics: Mapping[str, Any]) -> dict[str, Any]:
    last_entry = history[-1]
    stability_notes: list[str] = []
    if last_entry["val_prediction_loss"] > 1.25 * last_entry["train_prediction_loss"]:
        stability_notes.append("validation loss is notably above training loss")
    if last_entry["hidden_norm_mean"] > 10.0:
        stability_notes.append("hidden-state norms are elevated")
    if last_entry["trajectory_smoothness"] < 0.5:
        stability_notes.append("decoded trajectories are somewhat jagged")
    if not stability_notes:
        stability_notes.append("training remained numerically stable across the final epochs")

    return {
        "final_loss": float(last_entry["val_prediction_loss"]),
        "final_rollout_loss": float(last_entry["val_rollout_loss"]),
        "decoding_accuracy": float(last_entry["decoder_r2"]),
        "decoding_error": float(last_entry["decoder_normalized_rmse"]),
        "fraction_spatially_tuned": float(last_entry["fraction_spatially_tuned"]),
        "mean_spatial_information": float(last_entry["mean_spatial_information"]),
        "variance_explained": float(last_entry["pca_variance_explained"]),
        "evidence_of_spatial_selectivity": bool(last_entry["fraction_spatially_tuned"] > 0.05),
        "stability_notes": stability_notes,
        "best_checkpoint": final_metrics.get("best_checkpoint"),
    }


def generate_post_run_analysis(
    run_dir: str | Path,
    history: Sequence[Mapping[str, float]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate plots and a summary for a completed visual-input experiment."""

    run_path = Path(run_dir)
    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    loss_fig, _ = plot_loss_curves(history)
    loss_fig.savefig(plots_dir / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(loss_fig)

    decode_fig, _ = plot_decoding_error(history)
    decode_fig.savefig(plots_dir / "decoding_error.png", dpi=180, bbox_inches="tight")
    plt.close(decode_fig)

    visual_fig, _ = plot_visual_prediction(evaluation["true_patch"], evaluation["pred_patch"])
    visual_fig.savefig(plots_dir / "visual_prediction_quality.png", dpi=180, bbox_inches="tight")
    plt.close(visual_fig)

    trajectory_fig, _ = plot_trajectory_reconstruction(
        evaluation["true_positions"],
        evaluation["decoded_positions"],
        evaluation["replay_positions"],
    )
    trajectory_fig.savefig(plots_dir / "trajectory_reconstruction.png", dpi=180, bbox_inches="tight")
    plt.close(trajectory_fig)

    tuning_fig, _ = plot_sample_tuning_curves(
        evaluation["tuning_curves"],
        evaluation["spatial_information"],
    )
    tuning_fig.savefig(plots_dir / "spatial_tuning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(tuning_fig)

    embedding_fig, _ = plot_hidden_embedding(
        evaluation["hidden_embedding"],
        evaluation["true_positions"][: len(evaluation["hidden_embedding"])],
        method=evaluation["embedding_method"],
    )
    embedding_fig.savefig(plots_dir / "hidden_state_embedding.png", dpi=180, bbox_inches="tight")
    plt.close(embedding_fig)

    summary = summarize_run(history, evaluation)
    _write_json(run_path / "summary.json", summary)
    np.savez_compressed(
        run_path / "evaluation_snapshot.npz",
        true_patch=evaluation["true_patch"],
        pred_patch=evaluation["pred_patch"],
        true_positions=evaluation["true_positions"],
        decoded_positions=evaluation["decoded_positions"],
        replay_positions=evaluation["replay_positions"],
        hidden_embedding=evaluation["hidden_embedding"],
        tuning_curves=evaluation["tuning_curves"],
        spatial_information=evaluation["spatial_information"],
    )
    return summary


__all__ = [
    "generate_post_run_analysis",
    "plot_decoding_error",
    "plot_hidden_embedding",
    "plot_loss_curves",
    "plot_sample_tuning_curves",
    "plot_trajectory_reconstruction",
    "plot_visual_prediction",
    "summarize_run",
]
