"""Baseline global search: random starts, Riesz-energy descent, LP polish. Reports gap to the table record."""
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lp_polish import polish  # noqa: E402
from polish_sphere import DATA, max_cos  # noqa: E402


def energy_descent(x, s, steps=3000):
    lr = 0.01
    for _ in range(steps):
        diff = x[:, None, :] - x[None, :, :]
        r2 = (diff ** 2).sum(-1) + np.eye(len(x))
        f = (diff * (r2 ** (-(s + 2) / 2))[..., None]).sum(1)
        f -= (f * x).sum(1, keepdims=True) * x
        x = x + lr * f / (np.linalg.norm(f, axis=1).max() + 1e-300)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= 0.999
    return x


def one_start(args):
    d, n, seed, polish_seconds = args
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    for s in (6, 24, 80):
        x = energy_descent(x, s, 800)
    best, _ = polish(x, polish_seconds)
    return best


def main():
    table = {(e["d"], e["n"]): e["cosine"] for e in json.loads((DATA.parent / "spherical_table.json").read_text())}
    starts = int(sys.argv[1])
    for arg in sys.argv[2:]:
        d, n = map(int, arg.split("x"))
        t0 = time.time()
        with ProcessPoolExecutor(6) as ex:
            results = list(ex.map(one_start, [(d, n, s, 20.0) for s in range(starts)]))
        best = min(results)
        rec = table[(d, n)]
        print(f"d={d} n={n} record={rec:.12f} baseline_best={best:.12f} gap={best-rec:+.3e} "
              f"median={np.median(results):.6f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
