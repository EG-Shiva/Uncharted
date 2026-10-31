from .base import Evaluation, Problem
from .circle_packing import CirclePackingSquare

REGISTRY = {
    "circle_packing_26": lambda: CirclePackingSquare(26),
    "circle_packing_32": lambda: CirclePackingSquare(32),
}


def get_problem(name: str) -> Problem:
    if name not in REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]()
