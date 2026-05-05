from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from reasoning_geometry.common import ExperimentConfig, TrajectoryBundle, save_json
from reasoning_geometry.metrics.srsa import spearman_rsa


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.gru(x, hidden)
        output = self.norm(output)
        return output, hidden


class PredictiveRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [LayerNormGRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        layer_output = x
        final_hidden = None
        for layer in self.layers:
            layer_output, final_hidden = layer(layer_output, final_hidden)
        pred = self.output_head(layer_output)
        return pred, layer_output


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class NextStepDataset(Dataset):
    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.pairs[idx]


def fit_normalization(trajectories: Sequence[np.ndarray]) -> NormalizationStats:
    stacked = np.concatenate(trajectories, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_normalization(hidden_states: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return ((hidden_states - stats.mean) / stats.std).astype(np.float32)


def build_next_step_pairs(trajectories: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for trajectory in trajectories:
        if trajectory.shape[0] < 2:
            continue
        for left, right in zip(trajectory[:-1], trajectory[1:]):
            pairs.append((left.astype(np.float32), right.astype(np.float32)))
    return pairs


def collate_pairs(batch: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32).unsqueeze(1)
    return inputs, targets


def prnn_hidden_trajectory(model: PredictiveRNN, hidden_states: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(hidden_states, dtype=torch.float32, device=device).unsqueeze(0)
        _, latent = model(tensor)
    return latent[0].detach().cpu().numpy().astype(np.float32)


def validation_srsa(
    model: PredictiveRNN,
    trajectories: Sequence[TrajectoryBundle],
    logical_matrices: Dict[str, np.ndarray],
    stats: NormalizationStats,
    config: ExperimentConfig,
) -> float:
    scores: List[float] = []
    for bundle in trajectories[: config.validation_metric_examples]:
        normed = apply_normalization(bundle.hidden_states, stats)
        latent = prnn_hidden_trajectory(model, normed, config.device)
        rho, _ = spearman_rsa(
            logical_matrices[bundle.example_id],
            np.linalg.norm(latent[:, None, :] - latent[None, :, :], axis=-1),
        )
        scores.append(rho)
    return float(np.mean(scores)) if scores else 0.0


def train_prnn(
    trajectories_train: Sequence[TrajectoryBundle],
    trajectories_val: Sequence[TrajectoryBundle],
    logical_matrices_val: Dict[str, np.ndarray],
    config: ExperimentConfig,
    tb_logger=None,
) -> Dict[str, object]:
    stats = fit_normalization([bundle.hidden_states for bundle in trajectories_train])
    train_pairs = build_next_step_pairs(
        [apply_normalization(bundle.hidden_states, stats) for bundle in trajectories_train]
    )
    val_pairs = build_next_step_pairs(
        [apply_normalization(bundle.hidden_states, stats) for bundle in trajectories_val]
    )
    if not train_pairs:
        raise ValueError("No pRNN training pairs could be created from the trajectories.")

    dataset_train = NextStepDataset(train_pairs)
    dataset_val = NextStepDataset(val_pairs) if val_pairs else NextStepDataset(train_pairs[:1])
    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_pairs, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, collate_fn=collate_pairs, drop_last=False)

    input_dim = trajectories_train[0].hidden_states.shape[-1]
    model = PredictiveRNN(
        input_dim=input_dim,
        hidden_dim=config.pRNN_hidden,
        num_layers=config.pRNN_layers,
        output_dim=input_dim,
    ).to(config.device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.pRNN_steps)

    best_srsa = float("-inf")
    best_state = None
    global_step = 0
    train_iter = iter(loader_train)
    val_iter = iter(loader_val)

    while global_step < config.pRNN_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(loader_train)
            inputs, targets = next(train_iter)
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)

        model.train()
        preds, _ = model(inputs)
        loss = F.mse_loss(preds, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if global_step % config.logging_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                for val_inputs, val_targets in loader_val:
                    val_inputs = val_inputs.to(config.device)
                    val_targets = val_targets.to(config.device)
                    val_preds, _ = model(val_inputs)
                    val_losses.append(F.mse_loss(val_preds, val_targets).item())
            if tb_logger is not None:
                tb_logger.add_scalars(
                    "prnn/loss",
                    {"train": float(loss.item()), "val": float(np.mean(val_losses))},
                    global_step,
                )

        if global_step % config.validation_every == 0:
            model.eval()
            score = validation_srsa(model, trajectories_val, logical_matrices_val, stats, config)
            if tb_logger is not None:
                tb_logger.add_scalars("prnn/val_srsa", {"value": score}, global_step)
            if score > best_srsa:
                best_srsa = score
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        global_step += 1

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "normalization": stats,
        "best_val_srsa": best_srsa,
    }


def save_prnn_checkpoint(
    path: str | Path,
    model: PredictiveRNN,
    normalization: NormalizationStats,
    metadata: Dict[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "metadata": metadata,
        },
        path,
    )

