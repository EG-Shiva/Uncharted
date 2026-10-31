import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .base import Evaluation, Problem

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TABLE_URL = "https://spherical-codes.org/"


def load_record(d: int, n: int) -> tuple[float, np.ndarray]:
    table = json.loads((DATA_DIR / "spherical_table.json").read_text())
    entry = next(e for e in table if e["d"] == d and e["n"] == n)
    path = DATA_DIR / "codes" / f"{d}_{n}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(urllib.request.urlopen(f"{TABLE_URL}data/{d}/{n}", timeout=60).read().decode())
    return entry["cosine"], np.loadtxt(path, delimiter=",")


def gram_max(x: np.ndarray) -> tuple[float, np.ndarray]:
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    g = u @ u.T
    np.fill_diagonal(g, -2.0)
    return float(g.max()), g


class SphericalCode(Problem):
    """Place n points on the unit sphere in R^d, minimising the largest pairwise cosine."""

    def __init__(self, d: int, n: int):
        self.d, self.n = d, n
        self.name = f"spherical_code_{d}d_{n}"
        self.title = f"{n} points on the sphere in {d} dimensions: minimise the largest pairwise cosine"
        self.maximize = False
        self.best_known, self.record = load_record(d, n)
        self.best_known_source = f"spherical-codes.org table (Henry Cohn), entry d={d}, n={n}"
        self.timeout_seconds = 200.0
        self.picture_hint = (
            "The attached image diagnoses the parent's code. Left: slack of the tightest pairs (points below the red line are "
            "contacts at the maximum cosine). Middle: contacts per point; rattlers (0 contacts) and over-loaded points show where "
            "the structure is loose or jammed. Right: the contact graph projected to 2D. Use it to decide which points to move, "
            "remove and re-insert, or which region to re-arrange before coding.")

    def description(self) -> str:
        return f"""Find {self.n} unit vectors in R^{self.d} whose largest pairwise cosine (inner product) is as SMALL as possible.
Equivalently: maximise the minimum angle between any two points on the sphere S^{self.d - 1}.
The published world record (spherical-codes.org, Henry Cohn's table) is cosine {self.best_known:.12f}.
That entry is marked "needs more optimization", so the record is believed improvable. Any valid code below it is a new record.
Scoring: rows are normalised to unit length, then score = max over i<j of <x_i, x_j>. LOWER is better.
Output contract: define `solve()` returning a numpy array of shape ({self.n}, {self.d}). No zero rows, no NaN.
Files in the working directory: `np.load("best.npy")` is the best code found so far in this search (initially the published record),
and `np.load("record.npy")` is the published record itself. You may start from them, perturb them, or ignore them.
`solve()` must finish within {int(self.timeout_seconds) - 20} seconds on one CPU core. numpy and scipy are available.
Useful facts: at a local optimum many pairs are tied at the maximum cosine ("contacts"); some points ("rattlers") have no contacts.
A linear program over tangent displacements (minimise t subject to linearised <x_i,x_j> <= t for near-maximal pairs, inside a
trust region) converges to the nearest local optimum, so real progress needs moving to a different basin (topology changes)."""

    def seed_code(self) -> str:
        return '''import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max()


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    x = np.load("best.npy")
    return lp_polish(x, 30)
'''

    def input_files(self, current_best: np.ndarray | None) -> dict[str, np.ndarray]:
        best = current_best if current_best is not None else self.record
        return {"best.npy": best, "record.npy": self.record}

    def evaluate(self, out: np.ndarray) -> Evaluation:
        if out.shape != (self.n, self.d):
            return Evaluation(False, 1.0, f"Expected shape ({self.n}, {self.d}), got {out.shape}")
        if not np.all(np.isfinite(out)):
            return Evaluation(False, 1.0, "Output contains NaN or infinity")
        norms = np.linalg.norm(out, axis=1)
        if norms.min() < 1e-9:
            return Evaluation(False, 1.0, f"Row {int(norms.argmin())} is (near) zero")
        score, g = gram_max(out)
        contacts = (g > score - 1e-9).sum(1)
        return Evaluation(True, score, "valid", {
            "max_cosine": score,
            "min_angle_deg": float(np.degrees(np.arccos(min(1.0, score)))),
            "contact_pairs": int((np.triu(g, 1) > score - 1e-9).sum()),
            "rattlers": int((contacts == 0).sum()),
            "max_contacts_per_point": int(contacts.max()),
        })

    def render(self, out: np.ndarray, evaluation: Evaluation, path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=90)
        if not evaluation.valid:
            for ax in axes:
                ax.axis("off")
            axes[1].text(0.5, 0.5, f"INVALID: {evaluation.message[:80]}", ha="center", va="center")
            fig.savefig(path)
            plt.close(fig)
            return
        score, g = gram_max(out)
        iu = np.triu_indices(self.n, 1)
        slack = score - g[iu]
        tight = np.sort(slack)[: min(len(slack), 6 * self.n)]

        ax = axes[0]
        ax.semilogy(np.arange(len(tight)), np.maximum(tight, 1e-16), ".", ms=3)
        ax.axhline(1e-9, color="red", lw=0.8, ls="--", label="contact threshold 1e-9")
        ax.set_title("Pair slack (max cosine - cosine), tightest pairs", fontsize=9)
        ax.set_xlabel("pair rank")
        ax.legend(fontsize=7)

        contacts = (g > score - 1e-9).sum(1)
        ax = axes[1]
        ax.bar(np.arange(self.n), contacts, width=1.0, color=plt.cm.plasma(contacts / max(1, contacts.max())))
        ax.set_title(f"Contacts per point ({int((contacts == 0).sum())} rattlers with 0 contacts)", fontsize=9)
        ax.set_xlabel("point index")

        u = out / np.linalg.norm(out, axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(u - u.mean(0), full_matrices=False)
        p = u @ vt[:2].T
        ax = axes[2]
        I, J = np.where(np.triu(g, 1) > score - 1e-9)
        for i, j in zip(I, J):
            ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="grey", lw=0.3, alpha=0.5)
        sc = ax.scatter(p[:, 0], p[:, 1], c=contacts, cmap="plasma", s=14, edgecolors="black", linewidths=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="contacts")
        ax.set_title("Contact graph, projected onto top-2 principal axes", fontsize=9)
        ax.set_aspect("equal")

        gap = score - self.best_known
        fig.suptitle(f"{self.n} points in {self.d}D   max cosine = {score:.12f}   vs record {self.best_known:.12f}  ({gap:+.2e})",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
