import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, max_solves=100):
    """Trust-region LP polish until deadline."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_anneal(x, steps, T0, T1, lrfac):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _find_void(x_keep, rng, ns=12000):
    R = rng.standard_normal(size=(ns, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x_keep.T
    m = S.max(axis=1)
    return R[np.argmin(m)]


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 138.0
    rng = np.random.default_rng(12345)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = vals[bi]

    cur_x = best_x.copy()
    cur_val = best_val

    trial = 0
    T_accept = 0.0015
    while time.time() < explore_end:
        trial += 1
        # periodically restart from best
        if trial % 10 == 0:
            base = best_x
        else:
            base = cur_x
        # reset wandering if too bad
        if cur_val > best_val + 0.012 and trial % 7 == 0:
            base = best_x
        mode = trial % 6
        y = base.copy()
        n = 153
        if mode == 0:
            y = y + 0.015 * rng.standard_normal(y.shape)
        elif mode == 1:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            idx = np.argsort(-per)[:12]
            y[idx] = y[idx] + 0.08 * rng.standard_normal((12, 4))
            y = y + 0.003 * rng.standard_normal(y.shape)
        elif mode == 2:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            keep = np.delete(base, w, axis=0)
            v = _find_void(keep, rng)
            y[w] = v + 0.01 * rng.standard_normal(4)
            y = y + 0.004 * rng.standard_normal(y.shape)
        elif mode == 3:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            ws = np.argsort(-per)[:2]
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=8000)
                y[w] = v + 0.008 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y = y + 0.004 * rng.standard_normal(y.shape)
        elif mode == 4:
            y = y + (0.10 if (trial % 12 == 4) else 0.05) * rng.standard_normal(y.shape)
        else:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            worst = np.argsort(-per)[:6]
            iso = np.argsort(per)[:4]
            y[worst] = y[worst] + 0.06 * rng.standard_normal((6, 4))
            y[iso] = y[iso] + 0.03 * rng.standard_normal((4, 4))
            y = y + 0.002 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)

        if mode in (4,):
            y = _smooth_anneal(y, steps=550, T0=0.028, T1=0.003, lrfac=1.4)
        elif mode in (1, 5):
            y = _smooth_anneal(y, steps=450, T0=0.018, T1=0.0022, lrfac=1.2)
        else:
            y = _smooth_anneal(y, steps=420, T0=0.015, T1=0.002, lrfac=1.2)
        # short minimax tail
        y = _smooth_anneal(y, steps=120, T0=0.004, T1=0.0015, lrfac=1.0)
        v = _max_cos(y)

        improved_best = False
        if v < best_val - 1e-12:
            # confirm with quick LP if close to record (avoid noise)
            if v < best_val - 5e-6:
                y2, v2 = _lp_polish(y, min(deadline - 8.0, time.time() + 2.5), max_solves=25)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                    improved_best = True
                elif v < best_val:
                    best_x, best_val = y, v
                    improved_best = True
            else:
                best_x, best_val = y, v
                improved_best = True
            cur_x, cur_val = best_x.copy(), best_val
            continue

        # basin-hopping acceptance
        if v < cur_val:
            cur_x, cur_val = y, v
        else:
            if rng.random() < np.exp((cur_val - v) / T_accept):
                cur_x, cur_val = y, v

        # occasional LP probe of promising wandering state
        if time.time() < explore_end - 6.0 and (trial % 6 == 0) and v < best_val + 0.0025:
            y2, v2 = _lp_polish(y, time.time() + 1.2, max_solves=12)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                cur_x, cur_val = y2.copy(), v2
            elif v2 < cur_val:
                cur_x, cur_val = y2, v2

    # final squeeze of best
    if time.time() < deadline - 1.0:
        best_x, best_val = _lp_polish(best_x, deadline - 0.5, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
