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
    n[n == 0] = 1.0
    x = x / n
    x = np.where(np.isfinite(x), x, 0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu0]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1.0 - b1) * g
        v *= b2
        v += (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _void(others, rng, n_cand=8000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j].copy(), float(sc[j])
            step *= 1.05
        else:
            step *= 0.96
        if step < 5e-6:
            break
    return b

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 174.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = x0
    x0 = _norm(np.asarray(x0, dtype=float))
    xr = _norm(np.asarray(xr, dtype=float))
    if x0.shape != (272, 4):
        x0 = _norm(rng.standard_normal((272, 4)))
    if xr.shape != (272, 4):
        xr = x0.copy()
    v0, vr = _max_cos(x0), _max_cos(xr)
    if vr < v0:
        best, best_val = xr.copy(), vr
    else:
        best, best_val = x0.copy(), v0

    sigmas = [0.025, 0.012, 0.055, 0.018, 0.085, 0.008, 0.04]
    taus = [(0.014, 0.0012), (0.022, 0.0022), (0.008, 0.0007), (0.030, 0.003)]
    lrs = [(0.009, 0.0009), (0.013, 0.0011), (0.006, 0.0005)]

    trial = 0
    # basin hopping, reserve ~45s for final polish
    while time.time() < t0 + 126.0 and time.time() < deadline - 45.0:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[trial % len(taus)]
        lr0, lr1 = lrs[trial % len(lrs)]
        y = best + sigma * rng.standard_normal((272, 4))
        y = _norm(y)
        mode = trial % 5
        if mode == 1 or mode == 2:
            nkill = (2 + (trial % 3)) if mode == 1 else (4 + (trial % 3))
            nkill = min(nkill, 6)
            for _ in range(nkill):
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                order = np.argsort(-pw)
                pool = order[:10]
                wi = int(pool[rng.integers(0, len(pool))])
                others = np.delete(y, wi, axis=0)
                y[wi] = _void(others, rng, n_cand=7000)
        elif mode == 3:
            # break random jammed edges
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            iu = np.triu_indices(272, 1)
            cc = G[iu]
            sel = np.where(cc > mcur - 1e-4)[0]
            if sel.size > 0:
                nb = 2 + (trial % 4)
                for _ in range(nb):
                    e = int(sel[rng.integers(0, sel.size)])
                    ii, jj = int(iu[0][e]), int(iu[1][e])
                    # push apart tangentially
                    vi = y[jj] - (y[jj] @ y[ii]) * y[ii]
                    vj = y[ii] - (y[ii] @ y[jj]) * y[jj]
                    a = 0.02 + 0.02 * rng.random()
                    y[ii] = _norm((y[ii] - a * vi).reshape(1, -1))[0]
                    y[jj] = _norm((y[jj] - a * vj).reshape(1, -1))[0]
        elif mode == 4:
            # relocate rattlers / low-contact points to voids
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            cnt = (G > mcur - 1e-7).sum(axis=1)
            idx = np.argsort(cnt)
            nrel = 1 + (trial % 3)
            for k in range(nrel):
                wi = int(idx[k])
                # only move if truly loose, else random worst
                if cnt[wi] > 2 and rng.random() < 0.5:
                    continue
                others = np.delete(y, wi, axis=0)
                y[wi] = _void(others, rng, n_cand=7000)
        # annealed smoothing
        steps = 380 + (trial % 3) * 80
        y = _adam(y, steps, tau0, tau1, lr0, lr1)
        v = _max_cos(y)
        if v < best_val:
            # snap immediately
            ys = _lp_polish(y, seconds=2.0, deadline=deadline - 30.0)
            vs = _max_cos(ys)
            if vs < best_val:
                best, best_val = ys, vs
            elif v < best_val:
                best, best_val = y, v
        elif v < best_val + 0.0025 and (trial % 4 == 0):
            ys = _lp_polish(y, seconds=1.6, deadline=deadline - 30.0)
            vs = _max_cos(ys)
            if vs < best_val:
                best, best_val = ys, vs
        if time.time() >= deadline - 45.0:
            break
    remain = deadline - time.time() - 1.5
    if remain > 3.0:
        best = _lp_polish(best, seconds=min(remain, 40.0), deadline=deadline - 1.0)
    return _norm(best)
