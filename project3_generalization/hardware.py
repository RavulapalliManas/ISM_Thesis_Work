from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(6, cpu_count - 2))


@dataclass
class MemoryBudget:
    vram_budget_gb: float = 5.5
    ram_budget_gb: float = 14.0
    gpu_buffer_gb: float = 0.5


@dataclass
class ExecutionBudget:
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    gradient_checkpointing: bool = True
    max_workers: int = field(default_factory=_default_max_workers)
    keep_cpu_free: int = 2
    rollout_batch_size: int = 2
    min_rollout_batch_size: int = 1
    slow_batch_threshold_seconds: float = 8.0


@dataclass
class SimilarityBudget:
    num_steps: int = 25_000
    grid_size: int = 30
    gamma: float = 0.9
    temporal_horizon: int = 10
    cg_tolerance: float = 1e-5
    cg_max_iter: int = 1_500
    use_memmap: bool = True


@dataclass
class BaselineBudget:
    total_steps: int = 100_000
    sequence_length: int = 64
    min_sequence_length: int = 32
    evaluation_interval: int = 10_000
    evaluation_rollout_steps: int = 2_000
    replay_rollout_steps: int = 400
    hidden_size: int = 500
    store_hidden_states: bool = True
    hidden_state_sample_limit: int = 2_000


@dataclass
class CurriculumBudget:
    steps_per_environment: int = 50_000
    max_environments: int = 3
    evaluation_interval: int = 5_000
    reexposure_steps: int = 1_000
    rolling_window_size: int = 3


@dataclass
class MetricBudget:
    srsa_max_samples: int = 2_000
    cka_batch_size: int = 256
    betti_max_points: int = 1_000
    sr_grid_size: int = 30


@dataclass
class TwoModuleBudget:
    cortical_hidden_size: int = 100
    update_every_n_calls: int = 2


@dataclass
class AblationBudget:
    recurrence_scales: tuple[float, ...] = (0.3, 0.7, 1.0, 1.5)
    target_env_id: str = "B1_l_shape"


@dataclass
class ThreeDBudget:
    enabled_by_default: bool = False
    hidden_size: int = 600
    max_steps: int = 150_000
    preferred_navigator: str = "surface"


@dataclass
class HardwareConfig:
    output_root: str = "project3_generalization/results/hardware_constrained"
    memory: MemoryBudget = field(default_factory=MemoryBudget)
    execution: ExecutionBudget = field(default_factory=ExecutionBudget)
    similarity: SimilarityBudget = field(default_factory=SimilarityBudget)
    baseline: BaselineBudget = field(default_factory=BaselineBudget)
    curriculum: CurriculumBudget = field(default_factory=CurriculumBudget)
    metrics: MetricBudget = field(default_factory=MetricBudget)
    two_module: TwoModuleBudget = field(default_factory=TwoModuleBudget)
    ablation: AblationBudget = field(default_factory=AblationBudget)
    three_d: ThreeDBudget = field(default_factory=ThreeDBudget)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "HardwareConfig":
        return cls(
            output_root=str(mapping.get("output_root", cls.output_root)),
            memory=MemoryBudget(**mapping.get("memory", {})),
            execution=ExecutionBudget(**mapping.get("execution", {})),
            similarity=SimilarityBudget(**mapping.get("similarity", {})),
            baseline=BaselineBudget(**mapping.get("baseline", {})),
            curriculum=CurriculumBudget(**mapping.get("curriculum", {})),
            metrics=MetricBudget(**mapping.get("metrics", {})),
            two_module=TwoModuleBudget(**mapping.get("two_module", {})),
            ablation=AblationBudget(
                recurrence_scales=tuple(mapping.get("ablation", {}).get("recurrence_scales", (0.3, 0.7, 1.0, 1.5))),
                target_env_id=str(mapping.get("ablation", {}).get("target_env_id", "B1_l_shape")),
            ),
            three_d=ThreeDBudget(**mapping.get("three_d", {})),
        )


def load_hardware_config(path: str | Path) -> HardwareConfig:
    raw = json.loads(Path(path).read_text())
    return HardwareConfig.from_mapping(raw)


def gpu_memory_snapshot(device: torch.device | None = None) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "memory_allocated_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
        }
    if device is None:
        device = torch.device("cuda")
    index = 0 if device.index is None else int(device.index)
    return {
        "memory_allocated_mb": float(torch.cuda.memory_allocated(index) / (1024 ** 2)),
        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024 ** 2)),
        "memory_reserved_mb": float(torch.cuda.memory_reserved(index) / (1024 ** 2)),
        "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(index) / (1024 ** 2)),
    }


class PhaseLogger:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.records: dict[str, dict[str, Any]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        start_memory = gpu_memory_snapshot(self.device)
        try:
            yield
        finally:
            end = time.perf_counter()
            self.records[name] = {
                "runtime_seconds": float(end - start),
                "gpu_memory": gpu_memory_snapshot(self.device),
                "gpu_memory_start": start_memory,
            }


def make_output_directory(
    root: str | Path,
    *,
    mode: str,
    env_ids: list[str],
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "_".join(env_ids[:3]) if env_ids else "no_env"
    output_dir = Path(root) / f"{timestamp}_{mode}_seed{seed}_{stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "AblationBudget",
    "BaselineBudget",
    "CurriculumBudget",
    "ExecutionBudget",
    "HardwareConfig",
    "MemoryBudget",
    "MetricBudget",
    "PhaseLogger",
    "SimilarityBudget",
    "ThreeDBudget",
    "TwoModuleBudget",
    "gpu_memory_snapshot",
    "load_hardware_config",
    "make_output_directory",
    "write_json",
]
