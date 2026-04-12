"""Losses used by the topology-before-geometry experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from utils.lossFuns import predMSE


def _require_matching_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise AssertionError(
            f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}. Fix pipeline."
        )


class RolloutMSE(nn.Module):
    """Paper-primary rollout loss backed by the legacy MSE module."""

    def __init__(self):
        super().__init__()
        self._legacy = predMSE()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor | None = None, **_) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        dummy_hidden = hidden if hidden is not None else torch.zeros_like(pred[..., :1])
        total, _ = self._legacy(pred, target, dummy_hidden)
        return total


class WeightedRolloutMSE(nn.Module):
    """Gamma-weighted variant of the paper's rollout loss."""

    def __init__(self, gamma: float = 0.9):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        if pred.ndim < 2:
            return F.mse_loss(pred, target)
        horizon = pred.shape[0]
        weights = pred.new_tensor([self.gamma ** idx for idx in range(horizon)])
        weights = weights / weights.sum()
        per_step = ((pred - target) ** 2).mean(dim=tuple(range(1, pred.ndim)))
        return (weights * per_step).sum()


class LatentConsistencyLoss(nn.Module):
    """Observation-space and latent-space consistency loss."""

    def __init__(self):
        super().__init__()
        self.rollout = RolloutMSE()
        self.encoder = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        base = self.rollout(pred, target)
        pred_latent = self.encoder(pred.reshape(-1, pred.shape[-1]))
        target_latent = self.encoder(target.reshape(-1, target.shape[-1]))
        return base + F.mse_loss(pred_latent, target_latent)


class InfoNCELoss(nn.Module):
    """Contrastive rollout loss expected to enlarge topology-geometry gaps."""

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = float(tau)
        self.hidden_projector = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 64))
        self.obs_projector = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor | None = None, **_) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        if hidden is None:
            raise ValueError("InfoNCELoss requires hidden states.")
        hidden_flat = self.hidden_projector(hidden.reshape(-1, hidden.shape[-1]))
        obs_flat = self.obs_projector(target.reshape(-1, target.shape[-1]))
        hidden_flat = F.normalize(hidden_flat, dim=-1)
        obs_flat = F.normalize(obs_flat, dim=-1)
        logits = hidden_flat @ obs_flat.T / self.tau
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)


class SpectralLoss(nn.Module):
    """Match leading singular values of predicted and target observation matrices."""

    def __init__(self, k: int = 20):
        super().__init__()
        self.k = int(k)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        pred_matrix = pred.reshape(-1, pred.shape[-1])
        target_matrix = target.reshape(-1, target.shape[-1])
        pred_vals = torch.linalg.svdvals(pred_matrix)[: self.k]
        target_vals = torch.linalg.svdvals(target_matrix)[: self.k]
        min_len = min(len(pred_vals), len(target_vals))
        return F.mse_loss(pred_vals[:min_len], target_vals[:min_len])


class ManifoldRegularizationLoss(nn.Module):
    """Spatially supervised triplet-style upper-bound baseline."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = float(margin)
        self.rollout = RolloutMSE()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        hidden: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:
        _require_matching_shapes(pred, target)
        base = self.rollout(pred, target, hidden=hidden)
        if hidden is None or positions is None:
            return base

        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        positions_flat = positions.reshape(-1, positions.shape[-1])
        if hidden_flat.shape[0] < 3:
            return base

        anchor = hidden_flat[:-2]
        positive = hidden_flat[1:-1]
        negative = hidden_flat[2:]
        pos_dist = torch.linalg.norm(positions_flat[:-2] - positions_flat[1:-1], dim=-1)
        neg_dist = torch.linalg.norm(positions_flat[:-2] - positions_flat[2:], dim=-1)
        valid = neg_dist > pos_dist
        if not torch.any(valid):
            return base
        triplet = F.triplet_margin_loss(anchor[valid], positive[valid], negative[valid], margin=self.margin)
        return base + triplet


class LossFactory:
    """Registry for the main rollout loss and supplementary ablations."""

    @staticmethod
    def get_loss(loss_type: str, rollout_k: int, device) -> nn.Module:
        registry = {
            "rollout_mse": RolloutMSE,
            "weighted_rollout_mse": WeightedRolloutMSE,
            "latent_consistency": LatentConsistencyLoss,
            "infonce": InfoNCELoss,
            "spectral": SpectralLoss,
            "manifold_reg": ManifoldRegularizationLoss,
        }
        if loss_type not in registry:
            raise ValueError(f"Unknown loss: {loss_type}. Options: {list(registry)}")
        return registry[loss_type]().to(device)

