"""
File: project3_generalization/models/hippocampal_module.py

Description:
Wraps the legacy predictive-RNN architectures in a Project 3-friendly module
that supports configurable observation encoders, rollout losses, checkpointing,
and hardware-aware training.

Role in system:
This is the central model abstraction for the new codebase. Training loops use
it instead of directly touching the older `utils.Architectures` classes, which
makes experiment code easier to configure and reason about.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import utils.Architectures as architectures
from project3_generalization.hardware import gpu_memory_snapshot
from utils.lossFuns import predMSE


@dataclass
class HippocampalConfig:
    """Hyperparameters for the hippocampal predictive-RNN wrapper."""

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
    encoder_type: str = "identity"
    encoder_hidden_size: int = 256
    encoder_output_size: int | None = None
    visual_patch_size: int = 147
    visual_patch_width: int = 7
    visual_channels: int = 3
    head_direction_size: int = 0
    rollout_steps: int = 1
    rollout_loss_weight: float = 0.0
    latent_loss_weight: float = 0.0
    rollout_mode: str = "autoregressive"


class _ObservationAdapter(nn.Module):
    """Base class for observation encoders/decoders wrapped around the recurrent core."""

    def __init__(self, input_size: int, output_size: int, visual_size: int, aux_size: int):
        """Store input/output dimensionality for a concrete observation adapter."""
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.visual_size = int(max(visual_size, 0))
        self.aux_size = int(max(aux_size, 0))

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode raw observations into the latent space consumed by the recurrent core."""
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent predictions back into the original observation space."""
        raise NotImplementedError

    def decoder_parameters(self):
        """Return decoder parameters that should remain trainable in readout-only mode."""
        return []

    def _reshape_flat(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten leading dimensions so adapters can process batches and sequences uniformly."""
        shape = value.shape[:-1]
        return value.reshape(-1, value.shape[-1]), shape

    def _restore_shape(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """Restore a flattened tensor back to its original leading dimensions."""
        return value.reshape(*shape, value.shape[-1])

    def _constrain_output(self, decoded: torch.Tensor) -> torch.Tensor:
        """Apply modality-appropriate output nonlinearities after decoding."""
        if self.visual_size <= 0:
            return decoded
        visual = torch.sigmoid(decoded[:, : self.visual_size])
        if self.aux_size > 0:
            aux = torch.tanh(decoded[:, self.visual_size : self.visual_size + self.aux_size])
            return torch.cat([visual, aux], dim=-1)
        return visual


class _IdentityObservationAdapter(_ObservationAdapter):
    """Adapter that leaves observations unchanged."""

    def __init__(self, input_size: int):
        """Configure an identity mapping for already-suitable observation vectors."""
        super().__init__(input_size=input_size, output_size=input_size, visual_size=input_size, aux_size=0)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the input observation tensor unchanged."""
        return observations

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the latent tensor unchanged."""
        return latent


class _MLPObservationAdapter(_ObservationAdapter):
    """Multi-layer perceptron adapter for vector observations."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        visual_size: int,
        aux_size: int,
    ):
        """Build an MLP encoder/decoder pair around the recurrent core."""
        super().__init__(input_size=input_size, output_size=output_size, visual_size=visual_size, aux_size=aux_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode raw observations with a feedforward MLP."""
        flat, shape = self._reshape_flat(observations)
        encoded = self.encoder(flat)
        return self._restore_shape(encoded, shape)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent predictions with the MLP decoder."""
        flat, shape = self._reshape_flat(latent)
        decoded = self.decoder(flat)
        return self._restore_shape(self._constrain_output(decoded), shape)

    def decoder_parameters(self):
        """Expose decoder parameters for readout-only fine-tuning."""
        return self.decoder.parameters()


class _CNNObservationAdapter(_ObservationAdapter):
    """Convolutional adapter for egocentric RGB patch observations."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        visual_size: int,
        aux_size: int,
        patch_width: int,
        channels: int,
    ):
        """Build a CNN encoder/MLP decoder pair for flattened image-like observations."""
        if visual_size != patch_width * patch_width * channels:
            raise ValueError(
                "CNN encoder expects the visual observation to match patch_width * patch_width * channels."
            )
        super().__init__(input_size=input_size, output_size=output_size, visual_size=visual_size, aux_size=aux_size)
        self.patch_width = int(patch_width)
        self.channels = int(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_size = 32 * self.patch_width * self.patch_width
        self.encoder_head = nn.Sequential(
            nn.Linear(conv_size + self.aux_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode a flattened visual patch and optional auxiliary channels."""
        flat, shape = self._reshape_flat(observations)
        visual = flat[:, : self.visual_size].reshape(-1, self.channels, self.patch_width, self.patch_width)
        conv_features = self.conv(visual)
        if self.aux_size > 0:
            aux = flat[:, self.visual_size : self.visual_size + self.aux_size]
            conv_features = torch.cat([conv_features, aux], dim=-1)
        encoded = self.encoder_head(conv_features)
        return self._restore_shape(encoded, shape)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent predictions back into flattened visual observations."""
        flat, shape = self._reshape_flat(latent)
        decoded = self.decoder(flat)
        return self._restore_shape(self._constrain_output(decoded), shape)

    def decoder_parameters(self):
        """Expose decoder parameters for readout-only fine-tuning."""
        return self.decoder.parameters()


class HippocampalPredictiveRNN(nn.Module):
    """Predictive RNN wrapper with optional encoders, rollout loss, and checkpointing."""

    def __init__(self, config: HippocampalConfig | None = None):
        """Initialize the wrapped predictive RNN and its optimizer state."""
        super().__init__()
        self.config = config or HippocampalConfig()
        if not hasattr(architectures, self.config.pRNNtype):
            raise ValueError(f"Unknown predictive architecture `{self.config.pRNNtype}`.")

        self.visual_obs_size = int(min(self.config.visual_patch_size, max(self.config.obs_size - self.config.head_direction_size, 0)))
        self.aux_obs_size = int(max(self.config.obs_size - self.visual_obs_size, 0))
        self.observation_adapter = self._build_observation_adapter()
        self.encoded_obs_size = int(self.observation_adapter.output_size)

        architecture_cls = getattr(architectures, self.config.pRNNtype)
        self.core = architecture_cls(
            self.encoded_obs_size,
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

    def _build_observation_adapter(self) -> _ObservationAdapter:
        """Instantiate the observation adapter selected by the configuration."""
        encoder_type = self.config.encoder_type.lower()
        output_size = int(self.config.encoder_output_size or max(self.config.encoder_hidden_size // 2, 64))
        if encoder_type == "identity":
            return _IdentityObservationAdapter(self.config.obs_size)
        if encoder_type == "mlp":
            return _MLPObservationAdapter(
                input_size=self.config.obs_size,
                output_size=output_size,
                hidden_size=self.config.encoder_hidden_size,
                visual_size=self.visual_obs_size,
                aux_size=self.aux_obs_size,
            )
        if encoder_type == "cnn":
            return _CNNObservationAdapter(
                input_size=self.config.obs_size,
                output_size=output_size,
                hidden_size=self.config.encoder_hidden_size,
                visual_size=self.visual_obs_size,
                aux_size=self.aux_obs_size,
                patch_width=self.config.visual_patch_width,
                channels=self.config.visual_channels,
            )
        raise ValueError(f"Unsupported encoder type `{self.config.encoder_type}`.")

    def _resolve_device(self) -> torch.device:
        """Choose the explicit device from config or fall back to CUDA when available."""
        if self.config.device is not None:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct the optimizer with legacy-style layerwise learning-rate scaling."""
        rootk_h = (1.0 / self.config.hidden_size) ** 0.5
        rootk_i = (1.0 / self.core.rnn.cell.input_size) ** 0.5
        bias_params = []
        if hasattr(self.core, "bias") and isinstance(self.core.bias, nn.Parameter):
            bias_params.append(self.core.bias)
        parameter_groups = [
            {
                "params": [self.core.W],
                "name": "recurrent",
                "lr": self.config.learning_rate * rootk_h,
                "weight_decay": 0.0,
            },
            {
                "params": [self.core.W_out],
                "name": "readout",
                "lr": self.config.learning_rate * rootk_h,
                "weight_decay": 0.0,
            },
            {
                "params": [self.core.W_in],
                "name": "input",
                "lr": self.config.learning_rate * rootk_i,
                "weight_decay": 0.0,
            },
        ]
        if bias_params:
            parameter_groups.append(
                {
                    "params": bias_params,
                    "name": "bias",
                    "lr": self.config.learning_rate * rootk_h,
                    "weight_decay": self.config.weight_decay * self.config.learning_rate * rootk_h,
                }
            )
        core_params = {id(self.core.W), id(self.core.W_out), id(self.core.W_in), *(id(param) for param in bias_params)}
        extra_params = [param for param in self.parameters() if param.requires_grad and id(param) not in core_params]
        if extra_params:
            parameter_groups.append(
                {
                    "params": extra_params,
                    "name": "adapter",
                    "lr": self.config.learning_rate,
                    "weight_decay": 0.0,
                }
            )
        return torch.optim.RMSprop(parameter_groups, alpha=0.95, eps=1e-7)

    @property
    def device(self) -> torch.device:
        """Return the device on which the recurrent core currently lives."""
        return self.core.W.device

    def _autocast_context(self):
        """Return an AMP autocast context when mixed precision is enabled."""
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def _initial_state(self) -> torch.Tensor:
        """Create a zero recurrent state with the configured hidden width."""
        return torch.zeros((1, 1, self.config.hidden_size), dtype=torch.float32, device=self.device)

    def set_recurrence_scale(self, scale: float) -> None:
        """Update the multiplicative scale applied to the recurrent matrix during forward passes."""
        if self._recurrence_scaled:
            self._remove_recurrence_scale()
        self.recurrence_scale = float(scale)

    def _apply_recurrence_scale(self) -> None:
        """Temporarily scale recurrent weights for ablation experiments."""
        if self.recurrence_scale == 1.0 or self._recurrence_scaled:
            return
        with torch.no_grad():
            self.core.W.mul_(self.recurrence_scale)
        self._recurrence_scaled = True

    def _remove_recurrence_scale(self) -> None:
        """Undo a temporary recurrent-weight scaling after a forward pass."""
        if self.recurrence_scale == 1.0 or not self._recurrence_scaled:
            return
        with torch.no_grad():
            self.core.W.div_(self.recurrence_scale)
        self._recurrence_scaled = False

    def _to_tensor(self, value: torch.Tensor | Any) -> torch.Tensor:
        """Convert arbitrary array-like input into a float tensor on the model device."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _ensure_batched_inputs(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure observations and actions have an explicit batch dimension."""
        obs = self._to_tensor(observations)
        act = self._to_tensor(actions)
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
        if act.ndim == 2:
            act = act.unsqueeze(0)
        return obs, act

    @staticmethod
    def collapse_hidden(hidden: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
        """Reduce a theta/batch dimension in hidden states for downstream analyses."""
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden states of rank 2 or 3, got {hidden.shape}.")
        if reduce == "mean":
            return hidden.mean(dim=0)
        if reduce == "last":
            return hidden[-1]
        raise ValueError(f"Unknown hidden-state reduction `{reduce}`.")

    def _mask_from_layout(self, outmask: Any, length: int) -> np.ndarray:
        """Normalize the architecture-specific output mask into a boolean array."""
        if isinstance(outmask, (bool, np.bool_)):
            return np.full(length, bool(outmask), dtype=bool)
        return np.asarray(outmask, dtype=bool)

    def _apply_output_mask(self, predictions: torch.Tensor, mask_chunk: np.ndarray) -> torch.Tensor:
        """Zero out time steps that should not contribute predictions or losses."""
        if mask_chunk.all():
            return predictions
        masked = torch.zeros_like(predictions)
        masked[:, mask_chunk, :] = predictions[:, mask_chunk, :]
        return masked

    def _align_raw_targets(self, observations: torch.Tensor, actions: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """Align raw observation targets with the legacy architecture's temporal offsets."""
        action_inputs = self.core.actpad(actions)
        raw_target = observations[:, self.core.predOffset :, :]
        minsize = min(observations.size(1), action_inputs.size(1), raw_target.size(1))
        raw_target = raw_target[:, :minsize, :]
        return self._apply_output_mask(raw_target, mask[:minsize])

    def _sequence_layout(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """Prepare encoded inputs, encoded targets, raw targets, and masks for one sequence."""
        encoded_observations = self.observation_adapter.encode(observations)
        x_t, encoded_target, outmask = self.core.restructure_inputs(encoded_observations, actions)
        mask = self._mask_from_layout(outmask, encoded_target.shape[1])
        raw_target = self._align_raw_targets(observations, actions, mask)
        return x_t, encoded_target, raw_target, mask, encoded_observations

    def _run_recurrent_chunk(self, x_chunk: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward one chunk through the recurrent core."""
        return self.core.rnn(x_chunk, state=state)

    def _forward_chunk(
        self,
        x_chunk: torch.Tensor,
        state: torch.Tensor,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally checkpoint one recurrent chunk to trade compute for memory."""
        use_checkpoint = bool(training and self.config.gradient_checkpointing)
        if use_checkpoint:
            if not x_chunk.requires_grad:
                x_chunk = x_chunk.detach().requires_grad_(True)
            return checkpoint(self._run_recurrent_chunk, x_chunk, state, use_reentrant=False)
        return self._run_recurrent_chunk(x_chunk, state)

    def _forward_teacher_forced(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        training: bool,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Run teacher-forced prediction over a sequence and collect intermediate tensors."""
        x_t, encoded_target, raw_target, outmask, encoded_observations = self._sequence_layout(observations, actions)
        total_length = max(int(x_t.shape[1]), 1)
        chunk_length = max(1, min(int(self.config.chunk_length), total_length))
        state = self._initial_state()
        latent_predictions: list[torch.Tensor] = []
        decoded_predictions: list[torch.Tensor] = []
        hidden_chunks: list[torch.Tensor] = []

        self._apply_recurrence_scale()
        try:
            for start in range(0, total_length, chunk_length):
                end = min(start + chunk_length, total_length)
                x_chunk = x_t[:, start:end, :]
                mask_chunk = outmask[start:end]
                with self._autocast_context():
                    hidden_chunk, state = self._forward_chunk(x_chunk, state, training=training)
                    latent_chunk = self.core.outlayer(hidden_chunk)
                    decoded_chunk = self.observation_adapter.decode(latent_chunk)
                latent_predictions.append(self._apply_output_mask(latent_chunk.float(), mask_chunk))
                decoded_predictions.append(self._apply_output_mask(decoded_chunk.float(), mask_chunk))
                hidden_chunks.append(hidden_chunk.float())
                if training:
                    state = state.detach()
        finally:
            self._remove_recurrence_scale()

        return {
            "latent_predictions": torch.cat(latent_predictions, dim=1),
            "decoded_predictions": torch.cat(decoded_predictions, dim=1),
            "encoded_targets": encoded_target.float(),
            "raw_targets": raw_target.float(),
            "hidden": torch.cat(hidden_chunks, dim=1),
            "mask": outmask,
            "encoded_observations": encoded_observations.float(),
        }

    def _compute_rollout_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        teacher_outputs: dict[str, torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        """Compute multi-step rollout loss beyond the teacher-forced one-step objective."""
        if self.config.rollout_loss_weight <= 0.0 or self.config.rollout_steps <= 1:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        teacher_hidden = teacher_outputs["hidden"]
        teacher_latent = teacher_outputs["latent_predictions"]
        if not isinstance(teacher_hidden, torch.Tensor) or not isinstance(teacher_latent, torch.Tensor):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        rollout_terms: list[torch.Tensor] = []
        base_offset = max(int(getattr(self.core, "predOffset", 0)), 1)
        horizon = int(self.config.rollout_steps)
        max_start = max(int(actions.shape[1]) - horizon + 1, 0)
        for start in range(max_start):
            state = teacher_hidden[:, start : start + 1, :].detach()
            latent_pred = teacher_latent[:, start : start + 1, :]
            for step_ahead in range(2, horizon + 1):
                action_index = start + step_ahead - 1
                target_index = start + base_offset + step_ahead - 1
                if action_index >= actions.shape[1] or target_index >= observations.shape[1]:
                    break
                # In masked mode the rollout must rely purely on internal dynamics and actions.
                obs_input = torch.zeros_like(latent_pred) if self.config.rollout_mode == "masked" else latent_pred
                x_input = torch.cat([obs_input, actions[:, action_index : action_index + 1, :]], dim=-1)
                with self._autocast_context():
                    hidden_roll, state = self._run_recurrent_chunk(x_input, state)
                    latent_pred = self.core.outlayer(hidden_roll)
                    decoded_pred = self.observation_adapter.decode(latent_pred)
                target = observations[:, target_index : target_index + 1, :]
                rollout_terms.append(F.mse_loss(decoded_pred.float(), target.float()))
        if not rollout_terms:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return torch.stack(rollout_terms).mean()

    def _compute_sequence_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        ewc_penalty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble prediction, rollout, latent-alignment, and optional EWC losses."""
        teacher_outputs = self._forward_teacher_forced(observations, actions, training=self.training)
        decoded_predictions = teacher_outputs["decoded_predictions"]
        raw_targets = teacher_outputs["raw_targets"]
        hidden = teacher_outputs["hidden"]
        encoded_predictions = teacher_outputs["latent_predictions"]
        encoded_targets = teacher_outputs["encoded_targets"]
        if not all(isinstance(item, torch.Tensor) for item in (decoded_predictions, raw_targets, hidden, encoded_predictions, encoded_targets)):
            raise RuntimeError("Teacher-forced pass did not return tensor outputs.")

        pred_total, pred_loss = self.loss_fn(decoded_predictions, raw_targets, hidden)
        latent_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.config.latent_loss_weight > 0.0:
            latent_loss = F.mse_loss(encoded_predictions, encoded_targets)
        rollout_loss = self._compute_rollout_loss(observations, actions, teacher_outputs)
        total = pred_total.float()
        total = total + float(self.config.latent_loss_weight) * latent_loss.float()
        total = total + float(self.config.rollout_loss_weight) * rollout_loss.float()
        if ewc_penalty is not None:
            total = total + ewc_penalty.float()
        return {
            "loss": total.float(),
            "prediction_loss": pred_loss.float(),
            "rollout_loss": rollout_loss.float(),
            "latent_loss": latent_loss.float(),
        }

    def zero_grad(self) -> None:
        """Clear optimizer gradients using PyTorch's set-to-none convention."""
        self.optimizer.zero_grad(set_to_none=True)

    def backward_on_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        ewc_penalty: torch.Tensor | None = None,
        loss_scale: float = 1.0,
    ) -> dict[str, float]:
        """Backpropagate sequence losses for one batch without stepping the optimizer."""
        obs, act = self._ensure_batched_inputs(observations, actions)
        self.train()
        losses = self._compute_sequence_losses(obs, act, ewc_penalty=ewc_penalty)
        scaled_total = losses["loss"] * float(loss_scale)
        if self._amp_enabled:
            self.grad_scaler.scale(scaled_total).backward()
        else:
            scaled_total.backward()
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def optimizer_step(self, *, gradient_clip: float | None = None) -> None:
        """Apply one optimizer step, including optional gradient clipping and AMP handling."""
        if gradient_clip is not None:
            if self._amp_enabled:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
        if self._amp_enabled:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    def evaluate_on_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
    ) -> dict[str, float]:
        """Evaluate loss terms for one batch without updating parameters."""
        obs, act = self._ensure_batched_inputs(observations, actions)
        self.eval()
        with torch.no_grad():
            losses = self._compute_sequence_losses(obs, act)
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def predict_batch(
        self,
        observations: torch.Tensor | Any,
        actions: torch.Tensor | Any,
        *,
        reduce_hidden: str | None = None,
        no_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return decoded predictions, aligned targets, and hidden states for one batch."""
        obs, act = self._ensure_batched_inputs(observations, actions)
        context = torch.no_grad() if no_grad else torch.enable_grad()
        with context:
            outputs = self._forward_teacher_forced(obs, act, training=False)
        pred = outputs["decoded_predictions"]
        target = outputs["raw_targets"]
        hidden = outputs["hidden"]
        if not all(isinstance(item, torch.Tensor) for item in (pred, target, hidden)):
            raise RuntimeError("Predict batch expected tensor outputs.")
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
        """Run a full train step for one batch and return scalar loss components."""
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
        """Generate spontaneous activity by driving the recurrent core with internal noise."""
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
                    encoded_pred, hidden = self.core.internal(noise_t)
                    obs_pred = self.observation_adapter.decode(encoded_pred)
            finally:
                self._remove_recurrence_scale()
        obs_pred = obs_pred.float()
        hidden = hidden.float()
        if reduce_hidden is not None:
            hidden = self.collapse_hidden(hidden, reduce=reduce_hidden)
        return obs_pred, hidden

    def recurrent_matrix(self) -> torch.Tensor:
        """Return a detached copy of the recurrent weight matrix."""
        return self.core.W.detach().clone()

    def apply_recurrent_matrix(self, matrix: torch.Tensor | Any) -> None:
        """Overwrite the recurrent matrix with externally supplied weights."""
        with torch.no_grad():
            self.core.W.copy_(torch.as_tensor(matrix, dtype=self.core.W.dtype, device=self.device))

    def freeze_recurrent_weights(self) -> None:
        """Freeze only the recurrent matrix parameters."""
        self.core.W.requires_grad_(False)

    def freeze_all_but_readout(self) -> None:
        """Freeze the model except for readout and observation-decoder parameters."""
        for param in self.parameters():
            param.requires_grad_(False)
        self.core.W_out.requires_grad_(True)
        for param in self.observation_adapter.decoder_parameters():
            param.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Mark all parameters as trainable again."""
        for param in self.parameters():
            param.requires_grad_(True)

    def named_regularizable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return trainable parameters eligible for regularization terms such as EWC."""
        return [(name, param) for name, param in self.named_parameters() if param.requires_grad]

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        """Capture a detached clone of every parameter tensor."""
        return {name: param.detach().clone() for name, param in self.named_parameters()}

    def memory_stats(self) -> dict[str, float]:
        """Return current CUDA memory statistics for the model device."""
        return gpu_memory_snapshot(self.device)

    def save_checkpoint(self, path: str | Path) -> None:
        """Serialize the model, optimizer, scaler, and recurrence settings to disk."""
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
        """Load a checkpoint and reconstruct a configured model instance."""
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
