import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Program:
    id: int
    parent_id: int | None
    generation: int
    idea: str
    code: str
    valid: bool
    score: float
    message: str
    metrics: dict = field(default_factory=dict)
    image: str = ""
    model: str = ""
    llm_cost: float = 0.0
    run_seconds: float = 0.0
    direction: str = ""
    created: float = field(default_factory=time.time)


class ProgramDatabase:
    def __init__(self, path: Path, maximize: bool = True):
        self.path = path
        self.maximize = maximize
        self.programs: list[Program] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def reserve_id(self) -> int:
        with self._lock:
            pid = self._next_id
            self._next_id += 1
            return pid

    def add(self, p: Program) -> None:
        with self._lock:
            self.programs.append(p)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(p)) + "\n")

    def valid(self) -> list[Program]:
        return [p for p in self.programs if p.valid]

    def ranked(self) -> list[Program]:
        return sorted(self.valid(), key=lambda p: p.score, reverse=self.maximize)

    def best(self) -> Program | None:
        r = self.ranked()
        return r[0] if r else None

    def select_parent(self, rng: random.Random) -> Program:
        ranked = self.ranked()
        if not ranked:
            return self.programs[0]
        if rng.random() < 0.75:
            top = ranked[: max(3, len(ranked) // 5)]
            weights = [1.0 / (i + 1) for i in range(len(top))]
            return rng.choices(top, weights=weights, k=1)[0]
        return rng.choice(ranked)

    def inspirations(self, exclude_id: int, k: int = 2) -> list[Program]:
        seen_scores = set()
        out = []
        for p in self.ranked():
            key = round(p.score, 6)
            if p.id == exclude_id or key in seen_scores:
                continue
            seen_scores.add(key)
            out.append(p)
            if len(out) == k:
                break
        return out

    def recent_failures(self, k: int = 6) -> list[Program]:
        return [p for p in reversed(self.programs) if not p.valid][:k]
