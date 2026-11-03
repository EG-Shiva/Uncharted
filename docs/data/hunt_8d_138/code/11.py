import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _anneal(x, steps, beta0, beta1, lr0, lr1, mu=0.82):
    n, d = x.shape
    vel = np.zeros_like(x)
    inv = 1.0 - mu
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        avg = W.sum() / n
        if avg > 1.0:
            Tan /= avg
        vel *= mu
        vel += inv * Tan
        rn = np.sqrt((vel * vel).sum(axis=1, keepdims=True))
        # clip per-point step to 1
        fac = np.ones_like(rn)
        msk = rn > 1.0
        fac[msk] = 1.0 / (rn[msk] + 1e-12)
        x = x - lr * (vel * fac)
        x /= np.sqrt((x * x).sum(axis=1, keepdims=True))
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
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
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(7)
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # quick ensure polished baseline (short)
    try:
        if time.time() < deadline - 10:
            b0 = _lp_polish(best, min(deadline - 8, time.time() + 2.0))
            v0 = _max_cos(b0)
            if v0 < best_val:
                best, best_val = b0, v0
    except Exception:
        pass

    it = 0
    iu_all = np.triu_indices(n, 1)
    while time.time() < deadline - 8.0 and it < 500:
        it += 1
        mode = it % 10
        if mode < 6:
            sigmas = (0.10, 0.30, 0.60, 0.15, 0.40, 0.05)
            sigma = sigmas[mode]
            y = best + sigma * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if it % 3 == 0:
                k = int(rng.integers(4, 16))
                idx = rng.choice(n, size=k, replace=False)
                if rng.random() < 0.55:
                    # hole-guided re-insertion
                    try:
                        cand = rng.normal(size=(6000, d))
                        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                        keep_mask = np.ones(n, dtype=bool)
                        keep_mask[idx] = False
                        rem = y[keep_mask]
                        S = cand @ rem.T
                        scores = S.max(axis=1)
                        order = np.argsort(scores)[:k]
                        y[idx] = cand[order]
                    except Exception:
                        rp = rng.normal(size=(k, d))
                        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                        y[idx] = rp
                else:
                    rp = rng.normal(size=(k, d))
                    rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                    y[idx] = rp
            steps = 650 if (it % 2 == 0) else 950
            b0 = 12.0 + rng.random() * 15.0
            b1 = 280.0 + rng.random() * 220.0
            y = _anneal(y, steps=steps, beta0=b0, beta1=b1,
                        lr0=0.030, lr1=0.0025, mu=0.82)
        elif mode < 8:
            y = rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=1100, beta0=10.0, beta1=380.0,
                        lr0=0.035, lr1=0.002, mu=0.85)
        else:
            y = best.copy()
            k = 30
            idx = rng.choice(n, size=k, replace=False)
            try:
                A = rng.normal(size=(d, d))
                Q, _ = np.linalg.qr(A)
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1
                y[idx] = y[idx] @ Q.T
            except Exception:
                pass
            y = y + 0.06 * rng.normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=750, beta0=18.0, beta1=320.0,
                        lr0=0.025, lr1=0.002, mu=0.82)

        v_fast = _max_cos(y)
        if v_fast > best_val + 0.030:
            continue
        # cheap focused refinement (high beta anneal)
        try:
            y = _anneal(y, steps=220, beta0=200.0, beta1=800.0,
                        lr0=0.008, lr1=0.001, mu=0.7)
        except Exception:
            pass
        v_med = _max_cos(y)
        if v_med > best_val + 0.008:
            continue
        try:
            y = _lp_polish(y, min(deadline - 8.0, time.time() + 2.2))
        except Exception:
            continue
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()

    try:
        best = _lp_polish(best, min(deadline, time.time() + 6.5))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # strict finite check
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
