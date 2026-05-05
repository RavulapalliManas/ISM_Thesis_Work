from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def add_scalars(self, tag: str, scalars: Mapping[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(f"{tag}/{key}", float(value), step)

    def add_histogram(self, tag: str, values: Sequence[float], step: int) -> None:
        values = np.asarray(values, dtype=np.float32)
        self.writer.add_histogram(tag, values, step)

    def add_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        self.writer.add_figure(tag, figure, global_step=step)
        plt.close(figure)

    def add_image_from_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        buffer.seek(0)
        image = plt.imread(buffer)
        self.writer.add_image(tag, image, step, dataformats="HWC")
        plt.close(figure)

    def add_curve(self, tag: str, x: Iterable[float], y: Iterable[float], step: int) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(list(x), list(y), linewidth=2)
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        self.add_figure(tag, fig, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

