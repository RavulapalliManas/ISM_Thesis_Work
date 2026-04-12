"""Reuse-first wrapper around the legacy rollout predictive RNN architectures."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from project3_generalization.models.hippocampal_module import (
    HippocampalConfig,
    HippocampalPredictiveRNN,
)


class RolloutPRNN(nn.Module):
    """Thin wrapper exposing the rollout pRNN used for the paper-faithful baseline."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        rollout_k: int = 5,
        sigma: float = 0.03,
        time_mode: str = "continuous",
        recurrence_scale: float = 1.0,
        dropout: float = 0.15,
        neural_timescale: float = 2.0,
        lr: float = 2e-3,
        weight_decay: float = 3e-3,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.rollout_k = int(rollout_k)
        self.time_mode = str(time_mode)
        self.sigma = float(sigma)
        self.device_ = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        architecture = "thcycRNN_5win_fullc" if self.time_mode == "continuous" else "thcycRNN_5win_full"
        config = HippocampalConfig(
            obs_size=self.obs_dim,
            action_size=self.act_dim,
            hidden_size=self.hidden_dim,
            pRNNtype=architecture,
            learning_rate=lr,
            weight_decay=weight_decay,
            neural_timescale=neural_timescale,
            dropout=dropout,
            spontaneous_noise_std=self.sigma,
            device=str(self.device_),
            use_amp=False,
            gradient_checkpointing=False,
            encoder_type="identity",
            rollout_steps=self.rollout_k + 1,
            rollout_loss_weight=0.0,
            latent_loss_weight=0.0,
            rollout_mode="autoregressive",
            visual_patch_size=self.obs_dim,
            head_direction_size=0,
            recurrence_scale=recurrence_scale,
        )
        self.core_model = HippocampalPredictiveRNN(config)
        self.to(self.device_)

    @property
    def optimizer(self):
        return self.core_model.optimizer

    def _sample_initial_state(self) -> torch.Tensor:
        return self.sigma * torch.randn((1, 1, self.hidden_dim), dtype=torch.float32, device=self.device_)

    def zero_grad(self) -> None:
        self.core_model.zero_grad()

    def optimizer_step(self, gradient_clip: float | None = None) -> None:
        self.core_model.optimizer_step(gradient_clip=gradient_clip)

    def _to_tensor(self, value: torch.Tensor | np.ndarray | Any) -> torch.Tensor:
        return self.core_model._to_tensor(value)

    def forward_sequence(
        self,
        observations: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
        *,
        training: bool = True,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        obs, act = self.core_model._ensure_batched_inputs(observations, actions)
        original_initial_state = self.core_model._initial_state
        self.core_model._initial_state = self._sample_initial_state
        try:
            outputs = self.core_model._forward_teacher_forced(obs, act, training=training)
        finally:
            self.core_model._initial_state = original_initial_state
        decoded = outputs.get("decoded_predictions")
        raw_targets = outputs.get("raw_targets")
        encoded_targets = outputs.get("encoded_targets")
        if (
            isinstance(decoded, torch.Tensor)
            and isinstance(raw_targets, torch.Tensor)
            and isinstance(encoded_targets, torch.Tensor)
            and decoded.shape != raw_targets.shape
            and decoded.shape == encoded_targets.shape
        ):
            outputs["raw_targets"] = self.core_model.observation_adapter.decode(encoded_targets).float()
        return outputs

    def forward_rollout(
        self,
        h_prev: torch.Tensor | None,
        obs_t: torch.Tensor | np.ndarray,
        actions_t_to_t_plus_k: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Run one rollout block using the legacy theta-cycle implementation."""
        obs_t = self._to_tensor(obs_t)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(1)
        actions = self._to_tensor(actions_t_to_t_plus_k)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)

        pad_obs = torch.zeros(
            (obs_t.shape[0], actions.shape[1], obs_t.shape[-1]),
            dtype=obs_t.dtype,
            device=obs_t.device,
        )
        pad_obs[:, 0, :] = obs_t[:, 0, :]
        outputs = self.forward_sequence(pad_obs, actions, training=self.training)
        hidden = outputs["hidden"]
        decoded = outputs["decoded_predictions"]
        if not isinstance(hidden, torch.Tensor) or not isinstance(decoded, torch.Tensor):
            raise RuntimeError("Legacy rollout wrapper returned non-tensor outputs.")
        return {"hidden": hidden, "predictions": decoded}

    def get_hidden_states(
        self,
        obs_seq: torch.Tensor | np.ndarray,
        act_seq: torch.Tensor | np.ndarray,
        reduce: str = "last",
    ) -> np.ndarray:
        """Return hidden states as a CPU numpy array for evaluation code."""
        with torch.no_grad():
            outputs = self.forward_sequence(obs_seq, act_seq, training=False)
        hidden = outputs["hidden"]
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Legacy wrapper returned non-tensor hidden states.")
        if hidden.ndim == 3:
            hidden = self.core_model.collapse_hidden(hidden, reduce=reduce)
        return hidden.detach().cpu().numpy()

    def spontaneous(self, timesteps: int, noise_std: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core_model.spontaneous(timesteps, noise_std=noise_std)
