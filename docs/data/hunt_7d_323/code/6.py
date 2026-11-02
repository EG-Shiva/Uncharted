import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, start_ref=None):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    if start_ref is None:
        t0 = time.time()
    else:
        t0 = start_ref
    # allow caller to control total time via seconds from now
    t_start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < seconds and radius > 1e-14:
        if time.time() - t0 > 168:
            break
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = iu_c.max()
        thr = m - max(20 * radius, 1e-9)
        sel = iu_c > thr
        # cap constraints for speed
        p0 = int(sel.sum())
        I = iu[0][sel]; J = iu[1][sel]; csel = iu_c[sel]
        if p0 > 4000:
            # keep top 4000
            idx = np.argpartition(-csel, 4000)[:4000]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p0 = 4000
        if p0 == 0:
            break
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d+k, J*d+k, last
        # build cols
        ar = np.arange(d)
        cI = (I[:, None] * d + ar).ravel()
        cJ = (J[:, None] * d + ar).ravel()
        # interleave per row: need [cI_row, cJ_row, last] per row
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2*d+1] = np.empty(p, dtype=np.int64)  # placeholder
        # faster: build via concatenate per block using repeat logic
        # simply construct:
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p)], axis=0) if False else None
        # straightforward construction with loops in numpy:
        base = np.empty((p, 2*d+1), dtype=np.int64)
        base[:, :d] = I[:, None]*d + ar
        base[:, d:2*d] = J[:, None]*d + ar
        base[:, 2*d] = n*d
        cols = base.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            # keep x as is for next smaller trust region
    return xb

def _smooth(x, steps=500, tau0=0.02, tau1=0.002, mu=0.9, ms0=0.004, ms1=0.0004):
    n, d = x.shape
    vel = np.zeros_like(x)
    lr = 8.0
    for s in range(steps):
        frac = s / max(1, steps-1)
        tau = tau0 * (tau1/tau0) ** frac
        ms = ms0 * (ms1/ms0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        # exp in stable way
        E = np.exp((G - gm) / tau)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        step = lr * vel
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-18
        sc = np.minimum(1.0, ms / nm)
        step = step * sc
        x = x - step
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((323, 7))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (323, 7):
        # fallback: tile / trim
        tmp = rng.standard_normal((323, 7))
        m = min(x0.shape[0], 323)
        tmp[:m] = x0[:m, :7] if x0.shape[1] >= 7 else x0[:m]
        x0 = tmp
    best = _norm(x0)
    best_val = _max_cos(best)
    # quick stats for perturbation
    it = 0
    # main basin-hopping loop, reserve time for final polish
    while time.time() - t0 < 140:
        # compute contact density of current best
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        # dense counts within 2e-3 for targeting
        cnt = (G > mx - 0.002).sum(axis=1).astype(np.float64)
        order = np.argsort(cnt)  # ascending
        mode = it % 6
        y = best.copy()
        n, d = y.shape
        if mode == 0:
            sig = 0.015
            y = y + sig * rng.standard_normal((n, d))
            y = _norm(y)
        elif mode == 1:
            # cluster around overloaded point
            w = cnt**2 + 1e-6
            w = w / w.sum()
            c = rng.choice(n, p=w)
            neigh = np.where(G[c] > mx - 0.015)[0]
            if len(neigh) < 3:
                neigh = np.argsort(-G[c])[:8]
            # include center
            neigh = np.unique(np.append(neigh, c))
            sig = 0.06
            y[neigh] += sig * rng.standard_normal((len(neigh), d))
            y = _norm(y)
        elif mode == 2:
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            sig = 0.12
            y[idx] += sig * rng.standard_normal((k, d))
            y = _norm(y)
        elif mode == 3:
            # teleport rattlers (lowest counts)
            k = 2
            idx = order[:k]
            for j in idx:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                y[j] = v
            # tiny global shake
            y = y + 0.005 * rng.standard_normal((n, d))
            y = _norm(y)
        elif mode == 4:
            w = cnt**2 + 1e-6
            w = w / w.sum()
            c = rng.choice(n, p=w)
            neigh = np.where(G[c] > mx - 0.01)[0]
            if len(neigh) < 3:
                neigh = np.argsort(-G[c])[:12]
            neigh = np.unique(np.append(neigh, c))
            sig = 0.10
            y[neigh] += sig * rng.standard_normal((len(neigh), d))
            y = _norm(y)
        else:
            sig = 0.03
            y = y + sig * rng.standard_normal((n, d))
            y = _norm(y)
        # occasional bigger kick
        if it % 23 == 22:
            idx = rng.choice(n, size=20, replace=False)
            y[idx] += 0.15 * rng.standard_normal((20, d))
            y = _norm(y)
        # smooth relax
        y2 = _smooth(y, steps=500)
        v2 = _max_cos(y2)
        # only LP if promising
        if v2 < best_val + 0.005:
            remain = 140 - (time.time() - t0)
            if remain < 2:
                break
            budget = min(3.5, remain)
            y3 = _lp_polish(y2, seconds=budget, start_ref=t0)
            v3 = _max_cos(y3)
            if v3 < best_val - 1e-12:
                best = y3
                best_val = v3
        it += 1
        if time.time() - t0 > 140:
            break
    # final polish of best with remaining time
    remain = 172 - (time.time() - t0)
    if remain > 3:
        best = _lp_polish(best, seconds=min(remain, 25), start_ref=t0)
    best = _norm(best)
    return best
