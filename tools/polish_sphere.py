"""Download a spherical code and try to lower its maximum pairwise cosine by local optimisation."""
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "codes"


def fetch(d, n):
    path = DATA / f"{d}_{n}.csv"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        text = urllib.request.urlopen(f"https://spherical-codes.org/data/{d}/{n}", timeout=60).read().decode()
        path.write_text(text)
    return np.loadtxt(path, delimiter=",")


def max_cos(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max()


def polish(x, seconds=60.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best, best_x = max_cos(x), x.copy()
    n = len(x)
    iu = np.triu_indices(n, 1)
    start = time.time()
    beta = 200.0 / max(1e-3, 1 - best)
    lr = 1e-3 * (1 - best)
    while time.time() - start < seconds:
        g = x @ x.T
        c = g[iu]
        m = c.max()
        w = np.exp(beta * (c - m))
        w /= w.sum()
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        grad = W @ x
        grad -= np.sum(grad * x, axis=1, keepdims=True) * x
        x_new = x - lr * grad / (np.abs(grad).max() + 1e-300)
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        v = max_cos(x_new)
        if v < m:
            x = x_new
            lr *= 1.1
            if v < best:
                best, best_x = v, x.copy()
        else:
            lr *= 0.5
            if lr < 1e-16:
                beta *= 1.5
                lr = 1e-4 * (1 - best)
    return best, best_x


def main():
    table = {(e["d"], e["n"]): e for e in json.loads((DATA.parent / "spherical_table.json").read_text())}
    targets = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]]
    seconds = 45.0
    for d, n in targets:
        record = table[(d, n)]["cosine"]
        x = fetch(d, n)
        start = max_cos(x)
        best, bx = polish(x, seconds)
        gain = record - best
        print(f"d={d} n={n} record={record:.12f} file={start:.12f} polished={best:.12f} gain={gain:.3e}"
              + ("  IMPROVED" if gain > 1e-10 else ""), flush=True)
        if gain > 1e-10:
            np.savetxt(DATA / f"{d}_{n}_improved.csv", bx, delimiter=",", fmt="%.17g")


if __name__ == "__main__":
    main()
