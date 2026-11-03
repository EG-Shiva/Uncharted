import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_refine(x, steps, k0, k1, lr0, mu, rng, jitter=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    if jitter > 0:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = np.max(G - np.eye(n) * 2.0)
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

def _refine_single(c, others, steps, k=50.0, step0=0.15):
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(k * (dots - m))
        sw = w.sum()
        if sw == 0 or not np.isfinite(sw):
            break
        w /= sw
        grad = w @ others
        gt = grad - (grad @ c) * c
        st = step0 * (1.0 - 0.5 * it / max(1, steps))
        c = c - st * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _greedy_point(others, rng, n_cand=3000, steps=80):
    n_o, d = others.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(scores)].copy()
    c = _refine_single(c, others, steps=steps)
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 170.0
    HOP_END = T0 + 118.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    x_best = xb.copy()
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = x_best.shape

    def contact_counts(x, tol=1e-4):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        cnt = (G > mx - tol).sum(axis=1)
        worst = G.max(axis=1)
        return cnt, worst, G, mx

    trial = 0
    while time.time() - T0 < HOP_END and time.time() - T0 < BUDGET - 55:
        trial += 1
        r = rng.random()
        y = None
        if r < 0.55:
            # block regrow around jammed / random center
            cnt, worst, Gf, mx = contact_counts(cur_x)
            if rng.random() < 0.6:
                # overloaded center among top-12
                order = np.argsort(-cnt - 0.5 * (worst - worst.min()) / (worst.max() - worst.min() + 1e-12))
                top = order[:12]
                center = int(top[rng.integers(0, len(top))])
            else:
                center = int(rng.integers(0, n))
            K = int(rng.integers(4, 13))  # 4..12
            dots_c = cur_x @ cur_x[center]
            order_nn = np.argsort(-dots_c)
            # exclude center? include it: closest K include center itself (dot=1)
            block = order_nn[:K].copy()
            # randomize insertion order
            rng.shuffle(block)
            keep = np.ones(n, dtype=bool)
            keep[block] = False
            base = cur_x[keep].copy()
            new_pts = []
            pool = base.copy()
            for b in range(K):
                c_new = _greedy_point(pool, rng, n_cand=3000, steps=80)
                new_pts.append(c_new)
                pool = np.vstack([pool, c_new.reshape(1, -1)])
            y = cur_x.copy()
            for idx, pt in zip(block, new_pts):
                y[idx] = pt
            y = _smooth_refine(y, steps=700, k0=25, k1=350, lr0=0.25, mu=0.87, rng=rng, jitter=0.003)
        elif r < 0.82:
            # large shake to leave basin
            sigma = float(rng.choice([0.25, 0.35, 0.45, 0.15]))
            base = x_best if rng.random() < 0.5 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=800, k0=20, k1=380, lr0=0.30, mu=0.90, rng=rng)
        else:
            # fresh random restart, long relax
            y = rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_refine(y, steps=1500, k0=15, k1=350, lr0=0.30, mu=0.90, rng=rng)
            # second stage sharper
            y = _smooth_refine(y, steps=400, k0=60, k1=500, lr0=0.12, mu=0.90, rng=rng)

        if y is None or not np.all(np.isfinite(y)):
            continue
        new_val = _max_cos(y)
        if new_val < best_val - 1e-12:
            # squeeze with quick LP
            t_lp = time.time()
            try:
                y2 = _lp_polish(y, seconds=2.5, t0=t_lp)
                v2 = _max_cos(y2)
                if v2 < new_val and np.all(np.isfinite(y2)):
                    y, new_val = y2, v2
            except Exception:
                pass
            x_best = y.copy()
            best_val = new_val
            cur_x = y.copy()
            cur_val = new_val
        elif new_val < cur_val or True:
            # SA acceptance for cur
            elapsed = time.time() - T0
            T = 0.0025 * max(0.05, 1.0 - elapsed / 120.0)
            T = max(T, 0.0002)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy()
                cur_val = new_val
            # also track best even if tiny gain
            if new_val < best_val:
                x_best = y.copy()
                best_val = new_val

    # final tightening + LP polish with remaining time
    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 5:
        try:
            x_best = _smooth_refine(x_best, steps=350, k0=80, k1=700, lr0=0.08, mu=0.90, rng=rng)
            best_val = _max_cos(x_best)
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=remain, t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0) + 1e-12:
            # return better of the two to avoid regression
            if _max_cos(x0) < _max_cos(x_best):
                x_best = x0
    except Exception:
        pass
    return x_best
