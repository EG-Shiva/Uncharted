import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.85, iso=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        if iso > 0:
            # encourage isotropy: penalize X^T X deviation
            C = (X.T @ X) / n  # d x d
            # gradient of ||C - (1/d)I||^2 w.r.t X ~ (4/n) X (C - I/d)
            E = C - np.eye(d) / d
            Grad = Grad + iso * (X @ E)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize step scale
        V = mu * V + Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _best_hole(others, rng, n_samples=4000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.1:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    t0 = time.time()
    DEADLINE = t0 + 170.0
    EXPL_END = t0 + 132.0
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val
    n, d = 95, 13

    def row_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G.max()

    trial = 0
    sigmas = [0.01, 0.03, 0.07, 0.15, 0.30, 0.02, 0.05, 0.12]
    while time.time() < EXPL_END and time.time() < DEADLINE - 35:
        trial += 1
        mode = trial % 10
        Xt = cur.copy()
        if mode in (0, 1, 2, 3):
            sigma = sigmas[(trial // 2) % len(sigmas)]
            if rng.random() < 0.35:
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps = 600
            T0 = 0.015 if sigma > 0.08 else 0.010
            T1 = 0.0008
            lr0 = 0.12 if sigma > 0.08 else 0.08
            lr1 = 0.008
            iso = 0.02 if rng.random() < 0.3 else 0.0
            Xt = _gd_refine(Xt, steps, T0, T1, lr0, lr1, iso=iso)
        elif mode in (4, 5, 6):
            # hole reinsertion: remove 1-3 points, reinsert at best hole
            rm, mc = row_max(cur)
            order = np.argsort(-rm)
            if rng.random() < 0.5:
                k = 1
            elif rng.random() < 0.7:
                k = 2
            else:
                k = 3
            # mostly worst hubs, sometimes random / low-contact
            if rng.random() < 0.65:
                idx = order[:k].copy()
                # occasionally swap one for rattler/low
                if rng.random() < 0.3:
                    low = np.argsort(rm)[:10]
                    idx[-1] = rng.choice(low)
            else:
                idx = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            others = Xt[keep]
            for j in idx:
                nh = _best_hole(others, rng, n_samples=4000)
                # small jitter
                nh = nh + 0.005 * rng.standard_normal(d)
                nh /= np.linalg.norm(nh)
                Xt[j] = nh
                others = np.vstack([others, nh.reshape(1, -1)]) if False else Xt[np.ones(n, dtype=bool)]
                # update others for next hole: use current Xt without remaining to-fill?
                # simpler: recompute others excluding not-yet-filled? keep approx
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _gd_refine(Xt, 600, 0.012, 0.0008, 0.10, 0.008)
        elif mode == 7:
            # fresh random start, short anneal (diversity)
            Xt = rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _gd_refine(Xt, 900, 0.020, 0.001, 0.15, 0.010)
        else:
            # centroid-balancing shake: push anisotropic cluster apart
            sigma = 0.05
            Xt += sigma * rng.standard_normal((n, d))
            # subtract mean to encourage zero centroid
            Xt = Xt - 0.2 * Xt.mean(axis=0, keepdims=True)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _gd_refine(Xt, 600, 0.012, 0.0008, 0.10, 0.008, iso=0.05)

        v = _max_cos(Xt)
        if v < best_val:
            x_best = Xt.copy(); best_val = v
            cur = Xt.copy(); cur_val = v
        elif v < cur_val:
            cur = Xt.copy(); cur_val = v
        else:
            # simulated-annealing acceptance to wander to new topology
            gap = v - cur_val
            # allow ~0.005-0.02 uphill with small prob
            if gap < 0.015 and rng.random() < 0.12:
                cur = Xt.copy(); cur_val = v
            elif trial % 25 == 0:
                # periodic restart from best to avoid drift
                cur = x_best.copy(); cur_val = best_val
        if best_val < 0.235:
            break

    # final squeeze: LP polish best with remaining time
    remain = DEADLINE - time.time()
    if remain > 5:
        # short GD cool-down first (cheap, no LP)
        try:
            x_best = _gd_refine(x_best, 400, 0.003, 0.0003, 0.03, 0.004)
            nv = _max_cos(x_best)
            if nv < best_val:
                best_val = nv
        except Exception:
            pass
        remain = DEADLINE - time.time()
        try:
            x_best = _lp_polish(x_best, max(4.0, remain - 1.0), t_deadline=DEADLINE - 0.5)
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
