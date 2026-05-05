from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np


STEP_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:step\s*\d+[:.)-]|\d+[.)-])\s*", re.IGNORECASE),
    re.compile(r"\s*(?:therefore|thus|so|next|finally)\s+", re.IGNORECASE),
]


def split_reasoning_steps(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    for pattern in STEP_PATTERNS[:1]:
        spans = list(pattern.finditer(text))
        if spans:
            starts = [match.start() for match in spans] + [len(text)]
            steps: List[str] = []
            for left, right in zip(starts[:-1], starts[1:]):
                chunk = text[left:right].strip()
                chunk = re.sub(r"^(step\s*\d+[:.)-]|\d+[.)-])\s*", "", chunk, flags=re.IGNORECASE)
                if chunk:
                    steps.append(chunk.strip())
            if steps:
                return steps

    sentence_splits = re.split(r"(?<=[.!?])\s+", text)
    steps = [chunk.strip() for chunk in sentence_splits if chunk.strip()]
    return steps


def filter_steps(steps: Iterable[str], min_len: int = 3, max_len: int = 8) -> List[str]:
    filtered = [step.strip() for step in steps if step and step.strip()]
    if min_len <= len(filtered) <= max_len:
        return filtered
    return []


def logical_hop_distance(steps: List[str]) -> np.ndarray:
    n_steps = len(steps)
    indices = np.arange(n_steps)
    return np.abs(indices[:, None] - indices[None, :]).astype(np.float32)


def dependency_like_distance(steps: List[str]) -> np.ndarray:
    """Fallback dependency structure: sequential hops when no parser is available."""
    return logical_hop_distance(steps)

