import re

from .base import Evaluation, Problem
from .circle_packing import CirclePackingSquare
from .spherical_code import SphericalCode

REGISTRY = {
    "circle_packing_26": lambda: CirclePackingSquare(26),
    "circle_packing_32": lambda: CirclePackingSquare(32),
}


def get_problem(name: str) -> Problem:
    m = re.fullmatch(r"spherical_code_(\d+)d_(\d+)", name)
    if m:
        return SphericalCode(int(m.group(1)), int(m.group(2)))
    if name not in REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {', '.join(REGISTRY)}, spherical_code_<d>d_<n>")
    return REGISTRY[name]()
