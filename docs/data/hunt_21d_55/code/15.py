import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _smooth(X0, steps, b0, b1, lr0, lr1, rng, noise=0.0, mu=0.85):
    X = X0.copy()
    V = np.zeros_like(X)
    for t in range(1, steps + 1):
        f = t / max(steps, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            continue
        Gw = (W @ X) / S
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        if noise > 0:
            nz = rng.standard_normal(X.shape)
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
        V = mu * V + (1.0 - mu) * Gt
        X = X - lr * V
        X = _norm(X)
    return X


def _deep_hole(Y, rng, n_cand=8000, refine=120):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    h = cand[np.argmin(md)].copy()
    # refine single vector against fixed Y with LSE gradient
    beta = 200.0
    for _ in range(refine):
        dv = Y @ h
        m = float(np.max(dv))
        w = np.exp(beta * (dv - m))
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            break
        g = (w @ Y) / s
        gt = g - float(g @ h) * h
        n = float(np.linalg.norm(gt))
        if n < 1e-12:
            break
        gt /= n
        h = h - 0.02 * gt
        h /= max(float(np.linalg.norm(h)), 1e-12)
        beta = min(beta * 1.02, 800.0)
    return h


def _lp_polish(x, deadline, radius_init=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1000:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1000]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    LIM = 168.0
    rng = np.random.default_rng(7)
    # load start
    best = None
    best_s = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_s:
                    best_s = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((55, 21)))
        best_s = _max_cos(best)
    n, d = 55, 21
    cur = best.copy()
    cur_s = best_s
    it = 0
    T = 0.0008
    # main basin hopping, leave ~10s for final polish
    while time.time() - t0 < LIM - 10.0:
        it += 1
        if it % 30 == 0:
            cur = best.copy()
            cur_s = best_s
        base = best if rng.random() < 0.65 else cur
        prop = base.copy()
        r = rng.random()
        if r < 0.45:
            # hub hole re-insertion (1 or 2 points)
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            mx = float(np.max(G))
            cnt = np.sum(G > mx - 1e-3, axis=1)
            k = 1 if rng.random() < 0.6 else 2
            victims = np.argsort(-cnt)[:k].copy()
            if rng.random() < 0.25:
                victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Y = prop[keep]
            for vi in victims:
                h = _deep_hole(Y, rng, n_cand=6000, refine=80)
                prop[vi] = h
                Y = np.vstack([Y, h[None, :]])
            prop = prop + 0.004 * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            steps = 700
            b0, b1, lr0, lr1, nz = 150.0, 1000.0, 0.008, 0.001, 0.02
        elif r < 0.75:
            sigma = 0.02 + 0.08 * rng.random()
            prop = prop + sigma * rng.standard_normal(prop.shape)
            prop = _norm(prop)
            steps = 550
            b0, b1, lr0, lr1, nz = 150.0, 800.0, 0.008, 0.001, 0.02
        else:
            k = int(rng.integers(3, 10))
            idx = rng.choice(n, size=k, replace=False)
            prop[idx] = prop[idx] + 0.35 * rng.standard_normal((k, d))
            prop = _norm(prop)
            steps = 650
            b0, b1, lr0, lr1, nz = 180.0, 900.0, 0.009, 0.0012, 0.03
        prop = _smooth(prop, steps, b0, b1, lr0, lr1, rng, noise=nz)
        s = _max_cos(prop)
        if s < best_s - 1e-12:
            # quick LP refine
            dl = min(time.time() + 2.0, t0 + LIM - 10.0)
            if dl > time.time() + 0.3:
                prop2, s2 = _lp_polish(prop, dl, radius_init=8e-4)
            else:
                prop2, s2 = prop, s
            if s2 < best_s:
                best = prop2.copy()
                best_s = s2
                cur = prop2.copy()
                cur_s = s2
                T *= 0.98
            continue
        # SA acceptance for cur
        if s < cur_s or rng.random() < np.exp((cur_s - s) / max(T, 1e-6)):
            cur = prop.copy()
            cur_s = s
        # slow cooling
        T = 0.0008 * (0.995 ** (it / 5.0)) + 0.00015
        if time.time() - t0 >= LIM - 10.0:
            break
    # final polish of best with remaining time
    try:
        deadline = t0 + 176.0
        best, best_s = _lp_polish(best, deadline, radius_init=1e-3)
    except Exception:
        pass
    best = _norm(best)
    return best
