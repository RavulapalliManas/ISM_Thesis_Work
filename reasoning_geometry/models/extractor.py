from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_geometry.common import ReasoningReasoningExperimentConfig, ReasoningExample, TrajectoryBundle


MODEL_NAME_MAP = {
    "phi-2": "microsoft/phi-2",
    "gemma-2b": "google/gemma-2b",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
}


@dataclass
class LoadedLM:
    tokenizer: any
    model: AutoModelForCausalLM
    name: str


def resolve_model_name(name: str) -> str:
    return MODEL_NAME_MAP.get(name, name)


def load_causal_lm(config: ReasoningExperimentConfig) -> LoadedLM:
    model_name = resolve_model_name(config.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config.torch_dtype,
        output_hidden_states=True,
    )
    model.to(config.device)
    model.eval()
    return LoadedLM(tokenizer=tokenizer, model=model, name=model_name)


def _last_token_hidden(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_index = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[batch_indices, last_index]


def next_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)


def teacher_forcing_hidden_states(
    example: ReasoningExample,
    tokenizer,
    model,
    device: str,
    extract_layers: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    extract_layers = list(extract_layers or [])
    final_layer_states: List[np.ndarray] = []
    intermediate_states: Dict[str, List[np.ndarray]] = {str(layer): [] for layer in extract_layers}
    entropies: List[float] = []

    prefix = example.prompt.strip()
    with torch.no_grad():
        for step in example.steps:
            prefix = f"{prefix}\n{step}".strip()
            encoded = tokenizer(prefix, return_tensors="pt", truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            final_hidden = _last_token_hidden(outputs.hidden_states[-1], encoded["attention_mask"])
            final_layer_states.append(final_hidden[0].detach().cpu().to(torch.float32).numpy())
            for layer in extract_layers:
                layer_hidden = _last_token_hidden(outputs.hidden_states[layer], encoded["attention_mask"])
                intermediate_states[str(layer)].append(
                    layer_hidden[0].detach().cpu().to(torch.float32).numpy()
                )
            step_entropy = next_token_entropy(outputs.logits[:, -1, :])[0].detach().cpu().item()
            entropies.append(step_entropy)

    layer_arrays = {
        key: np.stack(values).astype(np.float32) for key, values in intermediate_states.items() if values
    }
    return {
        "hidden_states": np.stack(final_layer_states).astype(np.float32),
        "layer_hidden_states": layer_arrays,
        "entropies": np.asarray(entropies, dtype=np.float32),
    }


def extract_teacher_forcing_trajectory(
    example: ReasoningExample,
    loaded_lm: LoadedLM,
    config: ReasoningExperimentConfig,
) -> TrajectoryBundle:
    payload = teacher_forcing_hidden_states(
        example=example,
        tokenizer=loaded_lm.tokenizer,
        model=loaded_lm.model,
        device=config.device,
        extract_layers=config.extract_layers,
    )
    return TrajectoryBundle(
        example_id=example.example_id,
        hidden_states=payload["hidden_states"],
        layer_hidden_states=payload["layer_hidden_states"],
        step_text=example.steps,
        label=example.label,
        metadata={"entropies": payload["entropies"].tolist(), **example.metadata},
    )


def batch_centroid(trajectories: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate([np.asarray(traj, dtype=np.float32) for traj in trajectories], axis=0)
    return stacked.mean(axis=0)

