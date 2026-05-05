from __future__ import annotations

from typing import Dict, List, Tuple

from reasoning_geometry.common import ExperimentConfig, ReasoningExample
from reasoning_geometry.data.graph_utils import filter_steps, split_reasoning_steps


def _load_hf_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for ProntoQA loading. Install `datasets`."
        ) from exc
    return load_dataset(name, split=split)


def _extract_fields(record: Dict) -> Tuple[str, str, int]:
    context = (
        record.get("context")
        or record.get("premises")
        or record.get("question")
        or record.get("input")
        or ""
    )
    reasoning = (
        record.get("chain_of_thought")
        or record.get("reasoning")
        or record.get("proof")
        or record.get("cot")
        or record.get("explanation")
        or ""
    )
    label_raw = record.get("label", record.get("answer", 0))
    if isinstance(label_raw, bool):
        label = int(label_raw)
    elif isinstance(label_raw, str):
        label = int(label_raw.strip().lower() in {"true", "yes", "1", "entails"})
    else:
        label = int(label_raw)
    return str(context), str(reasoning), label


def load_prontoqa(config: ExperimentConfig) -> Dict[str, List[ReasoningExample]]:
    dataset = _load_hf_dataset("longface/prontoqa", config.prontoqa_split)
    examples: List[ReasoningExample] = []
    for idx, record in enumerate(dataset):
        context, reasoning, label = _extract_fields(record)
        steps = filter_steps(
            split_reasoning_steps(reasoning),
            min_len=config.min_chain_length,
            max_len=config.max_chain_length,
        )
        if not steps:
            continue
        examples.append(
            ReasoningExample(
                example_id=str(record.get("id", f"prontoqa_{idx:06d}")),
                prompt=context,
                response=reasoning,
                label=label,
                steps=steps,
                metadata={"source_split": config.prontoqa_split},
            )
        )
        if len(examples) >= config.n_examples + config.n_val:
            break

    if len(examples) < config.n_examples + config.n_val:
        raise ValueError(
            f"Only collected {len(examples)} ProntoQA examples after filtering; "
            f"expected at least {config.n_examples + config.n_val}."
        )

    train = examples[: config.n_examples]
    val = examples[config.n_examples : config.n_examples + config.n_val]
    return {"train": train, "val": val}

