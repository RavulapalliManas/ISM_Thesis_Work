from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent


@dataclass
class ReasoningExperimentConfig:
    model: str
    dataset: str
    n_examples: int
    n_val: int
    pRNN_hidden: int
    pRNN_layers: int
    pRNN_steps: int
    subspace_k: int
    permutation_n: int
    batch_size: int
    device: str
    dtype: str
    output_dir: str = "reasoning_geometry/results"
    trajectories_dir: str = "reasoning_geometry/artifacts/trajectories"
    matrices_dir: str = "reasoning_geometry/artifacts/matrices"
    figures_dir: str = "reasoning_geometry/artifacts/figures"
    checkpoints_dir: str = "reasoning_geometry/artifacts/checkpoints"
    tensorboard_dir: str = "reasoning_geometry/artifacts/tensorboard"
    seed: int = 7
    max_chain_length: int = 8
    min_chain_length: int = 3
    bootstrap_n: int = 1000
    validation_metric_examples: int = 50
    validation_every: int = 500
    logging_every: int = 100
    teacher_forcing_batch_size: int = 1
    extract_layers: List[int] = field(default_factory=list)
    use_coconut: bool = False
    surface_embeddings: List[str] = field(
        default_factory=lambda: ["tfidf", "sbert", "model_first_token"]
    )
    halueval_split: str = "qa"
    prontoqa_split: str = "train"
    holdout_name: str = "validation"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReasoningExperimentConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolve_dir(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return ROOT.parent / path

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return mapping[self.dtype]


@dataclass
class ReasoningExample:
    example_id: str
    prompt: str
    response: str
    label: int
    steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistanceBundle:
    logical: np.ndarray
    surface: Dict[str, np.ndarray]


@dataclass
class TrajectoryBundle:
    example_id: str
    hidden_states: np.ndarray
    layer_hidden_states: Dict[str, np.ndarray]
    step_text: List[str]
    label: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    name: str
    passed: bool
    statistic: float
    threshold: float
    p_value: Optional[float]
    message: str


class GateFailedError(RuntimeError):
    """Raised when a validation gate fails."""


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if dataclass_isinstance(obj):
        return {key: json_ready(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(value) for value in obj]
    return obj


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = x / norms
    return normalized @ normalized.T


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    distances = sq + sq.T - 2.0 * x @ x.T
    np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances, dtype=np.float64)


def dataclass_isinstance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))

