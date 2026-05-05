from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from reasoning_geometry.common import ExperimentConfig, ReasoningExample
from reasoning_geometry.data.graph_utils import filter_steps, split_reasoning_steps


def _load_hf_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for HaluEval loading. Install `datasets`."
        ) from exc
    return load_dataset(name, split=split)


def _extract_label(record: Dict) -> int:
    raw = record.get("hallucination", record.get("label", record.get("is_hallucinated", 0)))
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, str):
        return int(raw.strip().lower() in {"hallucinated", "yes", "true", "1"})
    return int(raw)


def _extract_domain(record: Dict) -> str:
    return str(
        record.get("domain")
        or record.get("category")
        or record.get("topic")
        or record.get("source")
        or "unknown"
    )


def load_halueval(config: ExperimentConfig) -> Dict[str, List[ReasoningExample]]:
    dataset = _load_hf_dataset("openkg/HaluEval", config.halueval_split)
    grouped: Dict[tuple[str, int], List[ReasoningExample]] = defaultdict(list)
    for idx, record in enumerate(dataset):
        question = str(record.get("question") or record.get("input") or "")
        answer = str(
            record.get("answer")
            or record.get("model_answer")
            or record.get("response")
            or record.get("output")
            or ""
        )
        label = _extract_label(record)
        steps = filter_steps(
            split_reasoning_steps(answer),
            min_len=config.min_chain_length,
            max_len=max(config.max_chain_length, 16),
        )
        if not steps:
            continue
        domain = _extract_domain(record)
        grouped[(domain, label)].append(
            ReasoningExample(
                example_id=str(record.get("id", f"halueval_{idx:06d}")),
                prompt=question,
                response=answer,
                label=label,
                steps=steps,
                metadata={"domain": domain},
            )
        )

    total_target = config.n_examples + config.n_val
    per_class_target = total_target // 2
    sampled: List[ReasoningExample] = []
    domain_names = sorted({domain for domain, _ in grouped})
    for label in [0, 1]:
        class_examples: List[ReasoningExample] = []
        round_robin = True
        while round_robin and len(class_examples) < per_class_target:
            round_robin = False
            for domain in domain_names:
                bucket = grouped.get((domain, label), [])
                if bucket:
                    class_examples.append(bucket.pop(0))
                    round_robin = True
                    if len(class_examples) >= per_class_target:
                        break
        if len(class_examples) < per_class_target:
            raise ValueError(
                f"Insufficient HaluEval examples for label {label}: "
                f"{len(class_examples)} < {per_class_target}"
            )
        sampled.extend(class_examples)

    train_per_class = config.n_examples // 2
    val_per_class = config.n_val // 2
    by_label = {
        0: [example for example in sampled if example.label == 0],
        1: [example for example in sampled if example.label == 1],
    }
    train = by_label[0][:train_per_class] + by_label[1][:train_per_class]
    val = (
        by_label[0][train_per_class : train_per_class + val_per_class]
        + by_label[1][train_per_class : train_per_class + val_per_class]
    )
    if len(train) != config.n_examples or len(val) != config.n_val:
        raise ValueError(
            f"Unable to create the requested HaluEval split sizes: "
            f"train={len(train)} val={len(val)}."
        )
    return {"train": train, "val": val}
