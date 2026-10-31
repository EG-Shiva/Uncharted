"""Trust-region linear-programming polish for spherical codes (minimise the maximum pairwise cosine)."""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from polish_sphere import DATA, fetch, max_cos  # noqa: E402


def lp_step(x, margin, radius):
    n, d = x.shape
    g = x @ x.T
    iu = np.triu_indices(n, 1)
    c = g[iu]
    m = c.max()
    sel = c > m - margin
    I, J = iu[0][sel], iu[1][sel]
    p = len(I)
    nv = n * d + 1
    rows, cols, vals = [], [], []
    for k in range(p):
        i, j = I[k], J[k]
        rows += [k] * (2 * d + 1)
        cols += list(range(i * d, i * d + d)) + list(range(j * d, j * d + d)) + [n * d]
        vals += list(x[j]) + list(x[i]) + [-1.0]
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    b_ub = -c[sel]
    er, ec, ev = [], [], []
    for i in range(n):
        er += [i] * d
        ec += list(range(i * d, i * d + d))
        ev += list(x[i])
    A_eq = coo_matrix((ev, (er, ec)), shape=(n, nv)).tocsr()
    obj = np.zeros(nv)
    obj[-1] = 1.0
    bounds = [(-radius, radius)] * (n * d) + [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
    if res.status != 0:
        return None, m, p
    delta = res.x[: n * d].reshape(n, d)
    y = x + delta
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y, m, p


def polish(x, seconds=60.0, log=False):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    it = 0
    while time.time() - start < seconds and radius > 1e-14:
        it += 1
        margin = max(20 * radius, 1e-9)
        y, m, p = lp_step(x, margin, radius)
        if y is None:
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if log and it % 10 == 0:
            print(f"   it={it} best={best:.14f} radius={radius:.1e} active_pairs={p}", flush=True)
    return best, x


def main():
    table = {(e["d"], e["n"]): e for e in json.loads((DATA.parent / "spherical_table.json").read_text())}
    seconds = float(sys.argv[1])
    for arg in sys.argv[2:]:
        d, n = map(int, arg.split("x"))
        record = table[(d, n)]["cosine"]
        x = fetch(d, n)
        t0 = time.time()
        best, bx = polish(x, seconds)
        gain = record - best
        flag = "  IMPROVED" if gain > 1e-9 else ""
        print(f"d={d} n={n} record={record:.12f} polished={best:.14f} gain={gain:.3e} ({time.time()-t0:.0f}s){flag}", flush=True)
        if gain > 1e-9:
            np.savetxt(DATA / f"{d}_{n}_improved.csv", bx, delimiter=",", fmt="%.17g")


if __name__ == "__main__":
    main()
