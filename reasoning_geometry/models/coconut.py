from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from reasoning_geometry.common import ReasoningReasoningExperimentConfig, ReasoningExample, TrajectoryBundle
from reasoning_geometry.models.extractor import LoadedLM, _last_token_hidden, next_token_entropy


@dataclass
class CoconutOutputs:
    hidden_states: np.ndarray
    layer_hidden_states: Dict[str, np.ndarray]
    entropies: np.ndarray


class CoconutWrapper:
    """
    Continuous thought chaining using direct `inputs_embeds` injection.

    The wrapper takes the final hidden state from step t, projects it into the model
    embedding space when needed, and appends it as the next "thought token" for step t+1.
    """

    def __init__(self, loaded_lm: LoadedLM, extract_layers: Optional[Sequence[int]] = None):
        self.tokenizer = loaded_lm.tokenizer
        self.model = loaded_lm.model
        self.extract_layers = list(extract_layers or [])
        hidden_size = self.model.config.hidden_size
        embed_dim = self.model.get_input_embeddings().weight.shape[1]
        self.project_to_embed = None
        if hidden_size != embed_dim:
            self.project_to_embed = torch.nn.Linear(hidden_size, embed_dim, bias=False).to(
                self.model.device, dtype=self.model.dtype
            )
            with torch.no_grad():
                if hidden_size == embed_dim:
                    self.project_to_embed.weight.copy_(torch.eye(hidden_size, device=self.model.device))

    def _hidden_to_embed(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.project_to_embed is None:
            return hidden
        return self.project_to_embed(hidden)

    def extract_continuous_trajectory(
        self,
        example: ReasoningExample,
        config: ReasoningExperimentConfig,
    ) -> TrajectoryBundle:
        prompt = example.prompt.strip()
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded_prompt = {key: value.to(config.device) for key, value in encoded_prompt.items()}
        input_embeds = self.model.get_input_embeddings()(encoded_prompt["input_ids"])
        attention_mask = encoded_prompt["attention_mask"]

        final_states: List[np.ndarray] = []
        layer_states: Dict[str, List[np.ndarray]] = {str(layer): [] for layer in self.extract_layers}
        entropies: List[float] = []

        with torch.no_grad():
            for step_text in example.steps:
                step_tokens = self.tokenizer(step_text, return_tensors="pt", add_special_tokens=False)
                step_tokens = {key: value.to(config.device) for key, value in step_tokens.items()}
                step_embeds = self.model.get_input_embeddings()(step_tokens["input_ids"])
                current_embeds = torch.cat([input_embeds, step_embeds], dim=1)
                current_mask = torch.cat([attention_mask, step_tokens["attention_mask"]], dim=1)

                outputs = self.model(
                    inputs_embeds=current_embeds.to(dtype=self.model.dtype),
                    attention_mask=current_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                final_hidden = _last_token_hidden(outputs.hidden_states[-1], current_mask)[0]
                final_states.append(final_hidden.detach().cpu().to(torch.float32).numpy())
                for layer in self.extract_layers:
                    state = _last_token_hidden(outputs.hidden_states[layer], current_mask)[0]
                    layer_states[str(layer)].append(state.detach().cpu().to(torch.float32).numpy())
                entropies.append(next_token_entropy(outputs.logits[:, -1, :])[0].detach().cpu().item())

                injected = self._hidden_to_embed(final_hidden.unsqueeze(0).unsqueeze(0))
                input_embeds = torch.cat([current_embeds, injected], dim=1)
                pad = torch.ones((attention_mask.shape[0], 1), device=config.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([current_mask, pad], dim=1)

        return TrajectoryBundle(
            example_id=example.example_id,
            hidden_states=np.stack(final_states).astype(np.float32),
            layer_hidden_states={
                key: np.stack(values).astype(np.float32) for key, values in layer_states.items() if values
            },
            step_text=example.steps,
            label=example.label,
            metadata={"entropies": entropies, "trajectory_type": "continuous", **example.metadata},
        )
