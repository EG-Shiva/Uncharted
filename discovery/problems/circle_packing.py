from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .base import Evaluation, Problem


class CirclePackingSquare(Problem):
    """Pack n circles of free radii in the unit square, maximising the sum of radii."""

    def __init__(self, n: int = 26):
        self.n = n
        self.name = f"circle_packing_{n}"
        self.title = f"{n} circles in a unit square: maximise the sum of radii"
        self.maximize = True
        if n == 26:
            self.best_known = 2.6359830849
            self.best_known_source = "Public 2026 record (OpenEvolve community / ClaudeEvolve / active-model); AlphaEvolve reported 2.635863"
        self.timeout_seconds = 150.0

    def description(self) -> str:
        return f"""Place exactly {self.n} circles inside the unit square [0,1] x [0,1].
Each circle i has centre (x_i, y_i) and radius r_i > 0.
Constraints (checked strictly, tolerance 1e-12):
  - every circle lies fully inside the square: r_i <= x_i <= 1 - r_i and r_i <= y_i <= 1 - r_i
  - no two circles overlap: distance(centre_i, centre_j) >= r_i + r_j
Objective: MAXIMISE the sum of all radii.
Output contract: define a function `solve()` that returns a numpy array of shape ({self.n}, 3) with rows (x, y, r).
`solve()` must finish within {int(self.timeout_seconds) - 20} seconds on one CPU core. numpy and scipy are available."""

    def seed_code(self) -> str:
        return f'''import numpy as np

def solve():
    n = {self.n}
    k = int(np.ceil(np.sqrt(n)))
    r = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2 * i + 1) * r, (2 * j + 1) * r, r * 0.999])
    return np.array(pts)
'''

    def evaluate(self, out: np.ndarray, tol: float = 1e-12) -> Evaluation:
        if out.shape != (self.n, 3):
            return Evaluation(False, 0.0, f"Expected shape ({self.n}, 3), got {out.shape}")
        if not np.all(np.isfinite(out)):
            return Evaluation(False, 0.0, "Output contains NaN or infinity")
        x, y, r = out[:, 0], out[:, 1], out[:, 2]
        if np.any(r <= 0):
            return Evaluation(False, 0.0, f"{int(np.sum(r <= 0))} circles have non-positive radius")
        wall = np.minimum.reduce([x - r, 1 - x - r, y - r, 1 - y - r])
        worst_wall = float(wall.min())
        if worst_wall < -tol:
            return Evaluation(False, 0.0, f"Circle {int(wall.argmin())} crosses the square boundary by {-worst_wall:.3e}")
        d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
        gap = d - (r[:, None] + r[None, :])
        np.fill_diagonal(gap, np.inf)
        worst_gap = float(gap.min())
        if worst_gap < -tol:
            i, j = np.unravel_index(np.argmin(gap), gap.shape)
            return Evaluation(False, 0.0, f"Circles {i} and {j} overlap by {-worst_gap:.3e}")
        score = float(r.sum())
        return Evaluation(True, score, "valid", {
            "sum_radii": score,
            "min_radius": float(r.min()),
            "max_radius": float(r.max()),
            "min_wall_gap": worst_wall,
            "min_pair_gap": worst_gap,
        })

    def render(self, out: np.ndarray, evaluation: Evaluation, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=100)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=2, color="black"))
        if out is not None and out.ndim == 2 and out.shape[1] == 3:
            r = out[:, 2]
            norm = (r - r.min()) / (np.ptp(r) + 1e-12)
            for idx, (cx, cy, cr) in enumerate(out):
                ax.add_patch(plt.Circle((cx, cy), cr, facecolor=plt.cm.viridis(norm[idx]), edgecolor="black", lw=0.6, alpha=0.85))
                ax.text(cx, cy, str(idx), ha="center", va="center", fontsize=7, color="white")
        status = f"sum of radii = {evaluation.score:.8f}" if evaluation.valid else f"INVALID: {evaluation.message[:60]}"
        ax.set_title(status, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
