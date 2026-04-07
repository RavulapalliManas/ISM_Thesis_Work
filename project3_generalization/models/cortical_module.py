from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from project3_generalization.models.hippocampal_module import HippocampalPredictiveRNN


@dataclass
class CorticalModuleConfig:
    hippocampal_hidden_size: int = 500
    cortical_hidden_size: int = 100
    decoder_rank: int = 32
    learning_rate: float = 1e-4
    update_every_n_calls: int = 2


class CorticalRNNPrior(nn.Module):
    """Slow-learning cortical module that builds a transferable prior over hippocampal dynamics."""

    def __init__(self, config: CorticalModuleConfig | None = None):
        super().__init__()
        self.config = config or CorticalModuleConfig()
        hidden_size = self.config.hippocampal_hidden_size
        cortical_size = self.config.cortical_hidden_size
        rank = self.config.decoder_rank

        self.rnn = nn.GRU(hidden_size, cortical_size, batch_first=True)
        self.transition_head = nn.Linear(cortical_size, hidden_size)
        self.decoder_left = nn.Linear(cortical_size, hidden_size * rank)
        self.decoder_right = nn.Linear(cortical_size, rank * hidden_size)
        self.activation = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.last_state: torch.Tensor | None = None
        self.update_calls = 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_sequence(
        self,
        hidden_sequence: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_sequence.ndim == 2:
            hidden_sequence = hidden_sequence.unsqueeze(0)
        outputs, state = self.rnn(hidden_sequence.to(self.device), initial_state)
        return outputs, state

    def infer_recurrent_matrix(self, cortical_state: torch.Tensor) -> torch.Tensor:
        if cortical_state.ndim == 3:
            cortical_state = cortical_state[-1]
        cortical_state = cortical_state.reshape(cortical_state.shape[0], -1)
        rank = self.config.decoder_rank
        hidden_size = self.config.hippocampal_hidden_size
        left = self.decoder_left(self.activation(cortical_state)).reshape(-1, hidden_size, rank)
        right = self.decoder_right(self.activation(cortical_state)).reshape(-1, rank, hidden_size)
        matrix = torch.matmul(left, right) / max(rank, 1)
        return matrix.squeeze(0)

    def initialize_hippocampus(
        self,
        hippocampal_module: HippocampalPredictiveRNN,
        *,
        blend: float = 1.0,
    ) -> None:
        if self.last_state is None:
            return
        decoded = self.infer_recurrent_matrix(self.last_state)
        existing = hippocampal_module.recurrent_matrix().to(decoded.device)
        blended = (1.0 - blend) * existing + blend * decoded
        hippocampal_module.apply_recurrent_matrix(blended)

    def train_on_hidden_sequence(self, hidden_sequence: torch.Tensor) -> dict[str, float]:
        if hidden_sequence.ndim == 2:
            hidden_sequence = hidden_sequence.unsqueeze(0)
        hidden_sequence = hidden_sequence.to(self.device)
        self.update_calls += 1
        if self.config.update_every_n_calls > 1 and (self.update_calls % self.config.update_every_n_calls) != 0:
            self.update_state_only(hidden_sequence)
            return {"loss": 0.0, "skipped_optimizer_step": 1.0}
        self.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs, state = self.rnn(hidden_sequence[:, :-1, :], self.last_state)
        predictions = self.transition_head(outputs)
        target = hidden_sequence[:, 1:, :]
        loss = nn.functional.mse_loss(predictions, target)
        loss.backward()
        self.optimizer.step()
        self.last_state = state.detach()
        return {"loss": float(loss.detach().cpu())}

    def update_state_only(self, hidden_sequence: torch.Tensor) -> None:
        if hidden_sequence.ndim == 2:
            hidden_sequence = hidden_sequence.unsqueeze(0)
        _, state = self.rnn(hidden_sequence.to(self.device), self.last_state)
        self.last_state = state.detach()


__all__ = [
    "CorticalModuleConfig",
    "CorticalRNNPrior",
]
