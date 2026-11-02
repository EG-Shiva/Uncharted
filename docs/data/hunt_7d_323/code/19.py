import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        # cap constraints to keep linprog fast
        if cnt > 2500:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -2500)[-2500:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[part]] = True
            sel = sel2
            cnt = 2500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        crow = cols.reshape(p, 2 * d + 1)
        vrow = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        crow[:, :d] = I[:, None] * d + ar
        crow[:, d:2 * d] = J[:, None] * d + ar
        crow[:, 2 * d] = n * d
        vrow[:, :d] = x[J]
        vrow[:, d:2 * d] = x[I]
        vrow[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # keep global best even if drift
        if time.time() - start >= seconds:
            break
    return best_x


def solve():
    rng = np.random.default_rng(0)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                # guard nan
                if not np.all(np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            continue
    if x0 is None:
        x0 = rng.standard_normal(size=(323, 7))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    # polish with safe time budget (must finish <180s)
    # use ~100s max for LP
    try:
        x1 = _lp_polish(x0, seconds=100)
    except Exception:
        x1 = x0
    # pick better of polished vs original
    try:
        v0 = _max_cos(x0)
        v1 = _max_cos(x1)
        x = x1 if v1 < v0 else x0
    except Exception:
        x = x0
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x
