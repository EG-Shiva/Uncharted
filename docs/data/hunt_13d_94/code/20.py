import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_refine(x, steps, k0, k1, lr0, mu, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = float(np.max(G - np.eye(n) * 2.0))
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = E @ x / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _riesz_refine(x, steps, s0, s1, lr0, mu, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        sexp = s0 + (s1 - s0) * frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        D = 2.0 - 2.0 * G
        # clip to avoid div0; D in (0,4]
        np.clip(D, 1e-6, 4.0, out=D)
        # power: D^{-s/2 -1}
        # cap exponent to avoid overflow
        # use log formulation with clipping
        pw = -sexp * 0.5 - 1.0
        # D^pw = exp(pw*log D)
        W = np.exp(pw * np.log(D))
        np.fill_diagonal(W, 0.0)
        # cap huge values
        # normalize per-row to keep step stable
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        Grad = (W @ x) / rs
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip tangent norm
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _reinsert_one(others, rng, n_cand=4000, steps=120):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(scores)].copy()
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        kk = 50.0
        w = np.exp(kk * (dots - m))
        w /= w.sum()
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.2 * (1.0 - 0.5 * it / steps)
        c = c - step * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(1)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(x_best):
            x_best = xr.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    def point_info(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        per = G.max(axis=1)
        cnt = (G > mx - 1e-3).sum(axis=1)
        return per, cnt, G

    HOP_END = T0 + 112.0
    it = 0
    while time.time() - T0 < HOP_END and time.time() - T0 < BUDGET - 55:
        it += 1
        per, cnt, G = point_info(cur_x)
        y = cur_x.copy()
        mode = rng.integers(0, 6)
        if mode == 0:
            # kill 1-2 worst points, farthest reinsert + smooth
            order = np.argsort(-per)
            kkill = 1 if rng.random() < 0.5 else 2
            victims = list(order[:kkill])
            # occasionally also kill a low-contact rattler-ish point
            if rng.random() < 0.3:
                order_lo = np.argsort(cnt)
                victims.append(int(order_lo[0]))
                victims = list(dict.fromkeys(victims))
            for idx in victims:
                others = np.delete(y, idx, axis=0)
                y[idx] = _reinsert_one(others, rng, n_cand=4000, steps=120)
            y = _smooth_refine(y, steps=550, k0=25, k1=320, lr0=0.25, mu=0.86, rng=rng)
            # short riesz tighten
            if rng.random() < 0.5:
                y = _riesz_refine(y, steps=200, s0=8, s1=30, lr0=0.15, mu=0.85, rng=rng)
        elif mode == 1:
            sigma = float(rng.choice([0.06, 0.10, 0.16, 0.25]))
            base = x_best if rng.random() < 0.5 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.5:
                y = _smooth_refine(y, steps=700, k0=20, k1=350, lr0=0.28, mu=0.9, rng=rng)
            else:
                y = _riesz_refine(y, steps=700, s0=6, s1=40, lr0=0.25, mu=0.88, rng=rng)
                y = _smooth_refine(y, steps=250, k0=40, k1=350, lr0=0.15, mu=0.88, rng=rng)
        elif mode == 2:
            # cluster move: pick center + nearest neighbors, joint shift
            c0 = int(rng.integers(0, n))
            dots = (cur_x @ cur_x[c0])
            dots[c0] = -10
            k = int(rng.integers(4, 10))
            neigh = np.argsort(-dots)[:k].tolist() + [c0]
            neigh = np.unique(neigh)
            common = rng.standard_normal(size=(d,)) * 0.18
            y[neigh] += common + 0.10 * rng.standard_normal(size=(len(neigh), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=650, k0=25, k1=330, lr0=0.22, mu=0.87, rng=rng)
        elif mode == 3:
            # riesz annealing from best with random exponent range
            sigma = 0.03 + 0.05 * rng.random()
            y = x_best + sigma * rng.standard_normal(size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            s0 = float(rng.choice([4, 8, 12]))
            s1 = float(rng.choice([25, 40, 60]))
            y = _riesz_refine(y, steps=800, s0=s0, s1=s1, lr0=0.25, mu=0.9, rng=rng)
            y = _smooth_refine(y, steps=300, k0=30, k1=400, lr0=0.15, mu=0.88, rng=rng)
        elif mode == 4:
            # targeted shake of overloaded + loose
            idx_lo = np.argsort(cnt)[:6]
            idx_hi = np.argsort(-cnt)[:6]
            idx_w = np.argsort(-per)[:4]
            idx = np.unique(np.concatenate([idx_lo, idx_hi, idx_w]))
            y[idx] += 0.28 * rng.standard_normal(size=(len(idx), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=600, k0=28, k1=320, lr0=0.22, mu=0.88, rng=rng)
        else:
            # small exploitation around best
            kpick = 10
            idx = rng.choice(n, size=kpick, replace=False)
            y[idx] += 0.05 * rng.standard_normal(size=(kpick, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=450, k0=45, k1=450, lr0=0.12, mu=0.9, rng=rng)

        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        else:
            elapsed = time.time() - T0
            T = 0.002 * max(0.05, 1.0 - elapsed / 115.0)
            T = max(T, 0.0002)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy()
                cur_val = new_val

    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 5:
        try:
            xt = _smooth_refine(x_best, steps=300, k0=80, k1=600, lr0=0.08, mu=0.9, rng=rng)
            nv = _max_cos(xt)
            if nv < best_val:
                x_best = xt
                best_val = nv
        except Exception:
            pass
        try:
            xp = _lp_polish(x_best, seconds=remain, t0=time.time())
            pv = _max_cos(xp)
            if pv < best_val and np.all(np.isfinite(xp)):
                x_best = xp
                best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0):
            x_best = x0 if _max_cos(x0) < _max_cos(x_best) else x_best
    except Exception:
        pass
    return x_best
