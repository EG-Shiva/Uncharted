import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _pt_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _lp_polish(x, deadline, radius_init=8e-4, max_iter=30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iter:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thr = max(20.0 * radius, 2e-9)
        sel = c > m - thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap constraints for speed: keep worst 2500
        if p > 2500:
            idx = np.argpartition(csel, p - 2500)[p - 2500:]
            I, J, csel = I[idx], J[idx], csel[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x, best

def _soft(x, steps, a0, a1, l0, l1, deadline):
    n, d = x.shape
    for s in range(steps):
        if s % 50 == 0 and time.time() > deadline:
            break
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # avoid overflow: exp(alpha*(G-m)), <=1
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        # clip per-row norm to 1 for stability
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / np.maximum(1.0, nr)
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_fill(x, idx_list, rng, M, deadline):
    n, d = x.shape
    for idx in idx_list:
        if time.time() > deadline:
            break
        R = rng.standard_normal((M, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ x.T
        D[:, idx] = -2.0
        vals = np.max(D, axis=1)
        j = int(np.argmin(vals))
        x[idx] = R[j]
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(42)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (189, 12):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n, d = 189, 12

    trial = 0
    # reserve ~12s for final polish
    while time.time() < deadline - 14.0 and trial < 80:
        if time.time() > deadline - 14.0:
            break
        x = best_x.copy()
        pm = _pt_max(x)
        order = np.argsort(pm)  # loose first
        mode = trial % 4
        if mode == 0:
            k = 4 + (trial % 4)  # 4..7 loose/rattler reinsert
            loose = order[:k].tolist()
            x = _hole_fill(x, loose, rng, 20000, deadline - 14.0)
            tight = order[-(3 + (trial % 3)):].tolist()
            x[tight] += 0.012 * rng.standard_normal((len(tight), d))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            x = _soft(x, 600, 30.0, 250.0, 0.20, 0.015, deadline - 14.0)
            lp_dl = min(deadline - 14.0, time.time() + 1.6)
            if time.time() < lp_dl:
                x, v = _lp_polish(x, lp_dl)
            else:
                v = _max_cos(x)
        elif mode == 1:
            worst = order[-2:].tolist()
            x = _hole_fill(x, worst, rng, 25000, deadline - 14.0)
            sel = rng.choice(n, size=12, replace=False)
            x[sel] += 0.02 * rng.standard_normal((12, d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            x = _soft(x, 800, 25.0, 300.0, 0.22, 0.01, deadline - 14.0)
            lp_dl = min(deadline - 14.0, time.time() + 1.6)
            if time.time() < lp_dl:
                x, v = _lp_polish(x, lp_dl)
            else:
                v = _max_cos(x)
        elif mode == 2:
            sel = rng.choice(n, size=20, replace=False)
            x[sel] += 0.08 * rng.standard_normal((20, d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            x = _soft(x, 700, 20.0, 200.0, 0.25, 0.02, deadline - 14.0)
            lp_dl = min(deadline - 14.0, time.time() + 1.2)
            if time.time() < lp_dl:
                x, v = _lp_polish(x, lp_dl)
            else:
                v = _max_cos(x)
        else:
            # global small shake + targeted hub kick
            x += 0.008 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            tight = order[-6:].tolist()
            x[tight] += 0.03 * rng.standard_normal((len(tight), d))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            x = _soft(x, 500, 35.0, 220.0, 0.18, 0.012, deadline - 14.0)
            lp_dl = min(deadline - 14.0, time.time() + 1.2)
            if time.time() < lp_dl:
                x, v = _lp_polish(x, lp_dl)
            else:
                v = _max_cos(x)
        trial += 1
        if v < best_val - 1e-12:
            best_val, best_x = v, x.copy()
    # final polish of best
    try:
        best_x, best_val = _lp_polish(best_x, deadline - 1.0, radius_init=6e-4, max_iter=40)
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
