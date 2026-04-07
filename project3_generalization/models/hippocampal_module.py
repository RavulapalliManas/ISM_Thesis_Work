from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import utils.Architectures as architectures
from project3_generalization.hardware import gpu_memory_snapshot
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
    truncation: int = 64
    chunk_length: int = 64
    recurrence_scale: float = 1.0
    spontaneous_noise_std: float = 0.03
    device: str | None = None
    use_amp: bool = True
    amp_dtype: str = "float16"
    gradient_checkpointing: bool = True


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

        self.to(self._resolve_device())
        self.optimizer = self._build_optimizer()
        self._amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        self._amp_dtype = torch.float16 if self.config.amp_dtype == "float16" else torch.float32
        self.grad_scaler = torch.amp.GradScaler(device="cuda", enabled=self._amp_enabled)

    def _resolve_device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def _autocast_context(self):
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def _initial_state(self) -> torch.Tensor:
        return torch.zeros((1, 1, self.config.hidden_size), dtype=torch.float32, device=self.device)

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
            return value.to(self.device, dtype=torch.float32)
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

    def _sequence_layout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        x_t, obs_target, outmask = self.core.restructure_inputs(observations, actions)
        if isinstance(outmask, (bool, np.bool_)):
            mask = np.full(obs_target.shape[1], bool(outmask), dtype=bool)
        else:
            mask = np.asarray(outmask, dtype=bool)
        return x_t, obs_target, mask

    def _run_recurrent_chunk(self, x_chunk: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.rnn(x_chunk, state=state)

    def _forward_chunk(
        self,
        x_chunk: torch.Tensor,
        state: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_checkpoint = bool(training and self.config.gradient_checkpointing)
        if use_checkpoint:
            if not x_chunk.requires_grad:
                x_chunk = x_chunk.detach().requires_grad_(True)
            return checkpoint(self._run_recurrent_chunk, x_chunk, state, use_reentrant=False)
        return self._run_recurrent_chunk(x_chunk, state)

    @staticmethod
    def _apply_output_mask(predictions: torch.Tensor, mask_chunk: np.ndarray) -> torch.Tensor:
        if mask_chunk.all():
            return predictions
        masked = torch.zeros_like(predictions)
        masked[:, mask_chunk, :] = predictions[:, mask_chunk, :]
        return masked

    def _compute_sequence_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        ewc_penalty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_t, obs_target, outmask = self._sequence_layout(observations, actions)
        total_length = max(int(x_t.shape[1]), 1)
        chunk_length = max(1, min(int(self.config.chunk_length), total_length))
        state = self._initial_state()
        weighted_total = torch.zeros((), dtype=torch.float32, device=self.device)
        weighted_pred = torch.zeros((), dtype=torch.float32, device=self.device)

        self._apply_recurrence_scale()
        try:
            for start in range(0, total_length, chunk_length):
                end = min(start + chunk_length, total_length)
                x_chunk = x_t[:, start:end, :]
                target_chunk = obs_target[:, start:end, :]
                mask_chunk = outmask[start:end]
                with self._autocast_context():
                    hidden_chunk, state = self._forward_chunk(x_chunk, state, training=self.training)
                    pred_chunk = self.core.outlayer(hidden_chunk)
                    masked_pred = self._apply_output_mask(pred_chunk, mask_chunk)
                    chunk_total, chunk_pred = self.loss_fn(masked_pred, target_chunk, hidden_chunk)
                weight = float(end - start) / float(total_length)
                weighted_total = weighted_total + chunk_total.float() * weight
                weighted_pred = weighted_pred + chunk_pred.float() * weight
                if self.training:
                    state = state.detach()
            if ewc_penalty is not None:
                weighted_total = weighted_total + ewc_penalty.float()
        finally:
            self._remove_recurrence_scale()
        return weighted_total, weighted_pred

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def backward_on_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        ewc_penalty: torch.Tensor | None = None,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        obs, act = self._ensure_batched_inputs(observations, actions)
        self.train()
        total_loss, pred_loss = self._compute_sequence_losses(obs, act, ewc_penalty=ewc_penalty)
        scaled_total = total_loss * float(loss_scale)
        if self._amp_enabled:
            self.grad_scaler.scale(scaled_total).backward()
        else:
            scaled_total.backward()
        return {
            "loss": float(total_loss.detach().cpu()),
            "prediction_loss": float(pred_loss.detach().cpu()),
        }

    def optimizer_step(self, *, gradient_clip: float | None = None) -> None:
        if gradient_clip is not None:
            if self._amp_enabled:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
        if self._amp_enabled:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

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
            x_t, obs_target, outmask = self._sequence_layout(obs, act)
            total_length = max(int(x_t.shape[1]), 1)
            chunk_length = max(1, min(int(self.config.chunk_length), total_length))
            state = self._initial_state()
            predictions: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            hidden_chunks: list[torch.Tensor] = []

            self._apply_recurrence_scale()
            try:
                for start in range(0, total_length, chunk_length):
                    end = min(start + chunk_length, total_length)
                    x_chunk = x_t[:, start:end, :]
                    target_chunk = obs_target[:, start:end, :]
                    mask_chunk = outmask[start:end]
                    with self._autocast_context():
                        hidden_chunk, state = self._run_recurrent_chunk(x_chunk, state)
                        pred_chunk = self.core.outlayer(hidden_chunk)
                    predictions.append(self._apply_output_mask(pred_chunk.float(), mask_chunk))
                    targets.append(target_chunk.float())
                    hidden_chunks.append(hidden_chunk.float())
                    state = state.detach()
            finally:
                self._remove_recurrence_scale()

        pred = torch.cat(predictions, dim=1)
        target = torch.cat(targets, dim=1)
        hidden = torch.cat(hidden_chunks, dim=1)
        if reduce_hidden is not None:
            hidden = self.collapse_hidden(hidden, reduce=reduce_hidden)
        return pred, target, hidden

    def train_on_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        ewc_penalty: torch.Tensor | None = None,
        gradient_clip: float | None = None,
    ) -> dict[str, float]:
        self.zero_grad()
        metrics = self.backward_on_batch(observations, actions, ewc_penalty=ewc_penalty)
        self.optimizer_step(gradient_clip=gradient_clip)
        return metrics

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
                with self._autocast_context():
                    obs_pred, hidden = self.core.internal(noise_t)
            finally:
                self._remove_recurrence_scale()
        obs_pred = obs_pred.float()
        hidden = hidden.float()
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

    def memory_stats(self) -> dict[str, float]:
        return gpu_memory_snapshot(self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "grad_scaler_state_dict": self.grad_scaler.state_dict() if self._amp_enabled else None,
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
        checkpoint_blob = torch.load(path, map_location=map_location)
        model = cls(HippocampalConfig(**checkpoint_blob["config"]))
        model.load_state_dict(checkpoint_blob["state_dict"])
        model.optimizer.load_state_dict(checkpoint_blob["optimizer_state_dict"])
        scaler_state = checkpoint_blob.get("grad_scaler_state_dict")
        if scaler_state is not None and model._amp_enabled:
            model.grad_scaler.load_state_dict(scaler_state)
        model.set_recurrence_scale(checkpoint_blob.get("recurrence_scale", 1.0))
        return model


__all__ = [
    "HippocampalConfig",
    "HippocampalPredictiveRNN",
]
