from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

import utils.Architectures as architectures
from utils.lossFuns import predMSE


@dataclass
class HippocampalConfig:
    obs_size: int = 76
    action_size: int = 3
    hidden_size: int = 500
    pRNNtype: str = "thRNN_5win"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    neural_timescale: float = 2.0
    dropout: float = 0.0
    sparsity: float = 0.5
    truncation: int = 50
    recurrence_scale: float = 1.0
    spontaneous_noise_std: float = 0.03


class HippocampalPredictiveRNN(nn.Module):
    """Thin wrapper around the Levenstein predictive-RNN architectures."""

    def __init__(self, config: HippocampalConfig | None = None):
        super().__init__()
        self.config = config or HippocampalConfig()
        if not hasattr(architectures, self.config.pRNNtype):
            raise ValueError(f"Unknown predictive architecture `{self.config.pRNNtype}`.")

        architecture_cls = getattr(architectures, self.config.pRNNtype)
        self.core = architecture_cls(
            self.config.obs_size,
            self.config.action_size,
            hidden_size=self.config.hidden_size,
            trunc=self.config.truncation,
            neuralTimescale=self.config.neural_timescale,
            dropp=self.config.dropout,
            f=self.config.sparsity,
        )
        self.loss_fn = predMSE()
        self.recurrence_scale = float(self.config.recurrence_scale)
        self._recurrence_scaled = False
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        rootk_h = (1.0 / self.config.hidden_size) ** 0.5
        rootk_i = (1.0 / self.core.rnn.cell.input_size) ** 0.5
        return torch.optim.RMSprop(
            [
                {
                    "params": self.core.W,
                    "name": "recurrent",
                    "lr": self.config.learning_rate * rootk_h,
                    "weight_decay": self.config.weight_decay * self.config.learning_rate * rootk_h,
                },
                {
                    "params": self.core.W_out,
                    "name": "readout",
                    "lr": self.config.learning_rate * rootk_h,
                    "weight_decay": self.config.weight_decay * self.config.learning_rate * rootk_h,
                },
                {
                    "params": self.core.W_in,
                    "name": "input",
                    "lr": self.config.learning_rate * rootk_i,
                    "weight_decay": self.config.weight_decay * self.config.learning_rate * rootk_i,
                },
            ],
            alpha=0.95,
            eps=1e-7,
        )

    @property
    def device(self) -> torch.device:
        return self.core.W.device

    def set_recurrence_scale(self, scale: float) -> None:
        if self._recurrence_scaled:
            self._remove_recurrence_scale()
        self.recurrence_scale = float(scale)

    def _apply_recurrence_scale(self) -> None:
        if self.recurrence_scale == 1.0 or self._recurrence_scaled:
            return
        with torch.no_grad():
            self.core.W.mul_(self.recurrence_scale)
        self._recurrence_scaled = True

    def _remove_recurrence_scale(self) -> None:
        if self.recurrence_scale == 1.0 or not self._recurrence_scaled:
            return
        with torch.no_grad():
            self.core.W.div_(self.recurrence_scale)
        self._recurrence_scaled = False

    def _to_tensor(self, value: torch.Tensor | Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _ensure_batched_inputs(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self._to_tensor(observations)
        act = self._to_tensor(actions)
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        if act.ndim == 2:
            act = act.unsqueeze(0)
        return obs, act

    @staticmethod
    def collapse_hidden(hidden: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden states of rank 2 or 3, got {hidden.shape}.")
        if reduce == "mean":
            return hidden.mean(dim=0)
        if reduce == "last":
            return hidden[-1]
        raise ValueError(f"Unknown hidden-state reduction `{reduce}`.")

    def predict_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        reduce_hidden: str | None = None,
        no_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, act = self._ensure_batched_inputs(observations, actions)
        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            self._apply_recurrence_scale()
            try:
                obs_pred, hidden, obs_target = self.core(obs, act)
            finally:
                self._remove_recurrence_scale()

        if reduce_hidden is not None:
            hidden = self.collapse_hidden(hidden, reduce=reduce_hidden)
        return obs_pred, obs_target, hidden

    def train_on_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        ewc_penalty: torch.Tensor | None = None,
        gradient_clip: float | None = None,
    ) -> dict[str, float]:
        obs, act = self._ensure_batched_inputs(observations, actions)
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._apply_recurrence_scale()
        try:
            obs_pred, hidden, obs_target = self.core(obs, act)
            total_loss, pred_loss = self.loss_fn(obs_pred, obs_target, hidden)
            if ewc_penalty is not None:
                total_loss = total_loss + ewc_penalty
            total_loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
            self.optimizer.step()
        finally:
            self._remove_recurrence_scale()

        return {
            "loss": float(total_loss.detach().cpu()),
            "prediction_loss": float(pred_loss.detach().cpu()),
        }

    def spontaneous(
        self,
        timesteps: int,
        *,
        noise_std: float | None = None,
        noise_mean: float = 0.0,
        reduce_hidden: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_std = self.config.spontaneous_noise_std if noise_std is None else noise_std
        noise_t = noise_mean + noise_std * torch.randn(
            (1, timesteps, self.config.hidden_size),
            device=self.device,
        )
        self.eval()
        with torch.no_grad():
            self._apply_recurrence_scale()
            try:
                obs_pred, hidden = self.core.internal(noise_t)
            finally:
                self._remove_recurrence_scale()
        if reduce_hidden is not None:
            hidden = self.collapse_hidden(hidden, reduce=reduce_hidden)
        return obs_pred, hidden

    def recurrent_matrix(self) -> torch.Tensor:
        return self.core.W.detach().clone()

    def apply_recurrent_matrix(self, matrix: torch.Tensor | Any) -> None:
        with torch.no_grad():
            self.core.W.copy_(torch.as_tensor(matrix, dtype=self.core.W.dtype, device=self.device))

    def freeze_recurrent_weights(self) -> None:
        self.core.W.requires_grad_(False)

    def freeze_all_but_readout(self) -> None:
        self.core.W.requires_grad_(False)
        self.core.W_in.requires_grad_(False)
        self.core.W_out.requires_grad_(True)
        if hasattr(self.core, "bias"):
            self.core.bias.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def named_regularizable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, param) for name, param in self.named_parameters() if param.requires_grad]

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.named_parameters()}

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "recurrence_scale": self.recurrence_scale,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "HippocampalPredictiveRNN":
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(HippocampalConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.set_recurrence_scale(checkpoint.get("recurrence_scale", 1.0))
        return model


__all__ = [
    "HippocampalConfig",
    "HippocampalPredictiveRNN",
]
