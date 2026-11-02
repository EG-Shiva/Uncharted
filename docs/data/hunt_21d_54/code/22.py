import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    y = x / nrm
    y = np.nan_to_num(y)
    n2 = np.linalg.norm(y, axis=1, keepdims=True)
    n2[n2 == 0] = 1.0
    return y / n2


def _lp_polish(x, seconds, r0=1e-3):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        csv = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            k = np.argpartition(-csv, 2500)[:2500]
            I, J = I[k], J[k]
            csv = csv[k]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                           shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_quench(y, steps=600, T0=0.02, T1=0.003, lr=0.06):
    y = _proj(y)
    n, d = y.shape
    # geometric T schedule
    logr = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for s in range(steps):
        T = T0 * np.exp(logr * (s / max(1, steps - 1)))
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: (G-m)<=0
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / S
        grad -= (grad * y).sum(axis=1, keepdims=True) * y
        step = lr * (1.0 - 0.5 * s / steps)
        y = y - step * grad
        y = _proj(y)
    return y


def solve():
    T_start = time.time()
    LIMIT = 150.0  # hard budget to stay <180s
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _proj(np.asarray(x0, dtype=np.float64))
    n, d = x0.shape
    # safety reshape if needed
    if (n, d) != (54, 21):
        # try to adapt: truncate/pad deterministically
        nn, dd = 54, 21
        y = np.zeros((nn, dd))
        rnd = np.random.default_rng(1)
        tmp = rnd.normal(size=(nn, dd))
        # copy overlap
        a = min(n, nn); b = min(d, dd)
        y[:a, :b] = x0[:a, :b]
        # fill rest with random then project
        mask = np.ones((nn, dd), bool)
        mask[:a, :b] = False
        y[mask] = tmp[mask]
        x0 = _proj(y)
        n, d = nn, dd

    best = x0.copy()
    best_val = _max_cos(best)

    def elapsed():
        return time.time() - T_start

    # initial squeeze (already near optimum, so short)
    if elapsed() < LIMIT - 20:
        tb = min(8.0, max(1.0, LIMIT - 20 - elapsed()))
        nb = _lp_polish(best, tb)
        nv = _max_cos(nb)
        if nv < best_val:
            best, best_val = nb, nv

    strengths = [0.008, 0.015, 0.03, 0.06, 0.12, 0.02, 0.05, 0.10, 0.005, 0.04]
    it = 0
    # basin hopping
    while elapsed() < LIMIT - 12:
        it += 1
        s = strengths[(it - 1) % len(strengths)]
        # shake best
        noise = rng.normal(size=(n, d)) * s
        # tangent part
        noise -= (noise * best).sum(axis=1, keepdims=True) * best
        y = best + noise
        # extra kick to a few random points on some iterations
        if it % 3 == 0:
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            extra = rng.normal(size=(k, d)) * (s * 2.0 + 1e-3)
            extra -= (extra * best[idx]).sum(axis=1, keepdims=True) * best[idx]
            y[idx] = best[idx] + extra
        y = _proj(y)
        # smooth quench, cheaper than LP
        steps = 500 if s < 0.05 else 700
        T0 = 0.018 if s < 0.05 else 0.025
        y = _smooth_quench(y, steps=steps, T0=T0, T1=0.0025, lr=0.06)
        vq = _max_cos(y)
        # only LP-polish if promising (within margin)
        if vq > best_val + 0.008 and it % 5 != 0:
            continue
        if elapsed() >= LIMIT - 10:
            # no time for LP, just check raw
            if vq < best_val:
                best, best_val = y, vq
            continue
        tb2 = min(3.5, LIMIT - 10 - elapsed())
        if tb2 < 0.5:
            if vq < best_val:
                best, best_val = y, vq
            break
        yp = _lp_polish(y, tb2)
        vp = _max_cos(yp)
        if vp < best_val:
            best, best_val = yp, vp
        # also accept equal to move basin? keep deterministic drift
        # if close, continue from polished candidate sometimes
        if it % 7 == 0 and vp < best_val + 0.002:
            # exploratory move: perturb from this new basin next time
            # by temporarily using it as base for one step (without losing best)
            # do one extra small shake from yp
            if elapsed() < LIMIT - 12:
                noise2 = rng.normal(size=(n, d)) * 0.01
                noise2 -= (noise2 * yp).sum(axis=1, keepdims=True) * yp
                y2 = _proj(yp + noise2)
                y2 = _smooth_quench(y2, steps=300, T0=0.012, T1=0.002, lr=0.05)
                v2 = _max_cos(y2)
                if v2 < best_val + 0.004:
                    tb3 = min(2.5, LIMIT - 10 - elapsed())
                    if tb3 > 0.5:
                        y2p = _lp_polish(y2, tb3)
                        v2p = _max_cos(y2p)
                        if v2p < best_val:
                            best, best_val = y2p, v2p
    best = _proj(best)
    return best
