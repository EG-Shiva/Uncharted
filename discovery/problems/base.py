from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Evaluation:
    valid: bool
    score: float
    message: str
    metrics: dict = field(default_factory=dict)


class Problem:
    name: str = ""
    title: str = ""
    maximize: bool = True
    best_known: float | None = None
    best_known_source: str = ""
    timeout_seconds: float = 120.0
    picture_hint: str = ("The attached image is the parent's solution drawn to scale. Study it before coding: where is space wasted, "
                         "which elements are cramped or could grow, what symmetry or structure is missing?")

    def description(self) -> str:
        raise NotImplementedError

    def seed_code(self) -> str:
        raise NotImplementedError

    def input_files(self, current_best: np.ndarray | None) -> dict[str, np.ndarray]:
        return {}

    def evaluate(self, output: np.ndarray) -> Evaluation:
        raise NotImplementedError

    def render(self, output: np.ndarray, evaluation: Evaluation, path: Path) -> None:
        raise NotImplementedError

    def better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    def gap_to_record(self, score: float) -> float | None:
        if self.best_known is None:
            return None
        return (self.best_known - score) if self.maximize else (score - self.best_known)
