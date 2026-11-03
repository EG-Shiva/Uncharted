import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - tol).sum(axis=1)
    nn = G.max(axis=1)
    return m, cnt, nn, G

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            idx = np.argpartition(-cf, 3500)[:3500]
            I = iuI[idx]; J = iuJ[idx]
            c = cf[idx]
        else:
            I = iuI[sel]; J = iuJ[sel]
            c = cf[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _balanced_smooth(X, rng, taus=(0.005, 0.002, 0.0007), iters=200, lr0=0.018, lr1=0.003):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    mu = 0.8
    for tau in taus:
        V = np.zeros_like(X)
        for t in range(iters):
            f = t / max(1, iters - 1)
            lr = lr0 * ((lr1 / lr0) ** f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            Gmax = float(G.max())
            E = np.exp((G - Gmax) / tau)
            np.fill_diagonal(E, 0.0)
            rs = E.sum(axis=1, keepdims=True)
            rs = np.maximum(rs, 1e-300)
            W = E / rs
            Sym = W + W.T
            Gb = Sym @ X
            dotb = (Gb * X).sum(axis=1, keepdims=True)
            Gb_t = Gb - dotb * X
            Sg = float(E.sum())
            if np.isfinite(Sg) and Sg > 0:
                Pg = E / Sg
                Gg = (Pg + Pg.T) @ X
                dotg = (Gg * X).sum(axis=1, keepdims=True)
                Gg_t = Gg - dotg * X
                nb = float(np.linalg.norm(Gb_t) / np.sqrt(max(1, n))) + 1e-12
                ng = float(np.linalg.norm(Gg_t) / np.sqrt(max(1, n))) + 1e-12
                Grad = 0.7 * Gb_t / nb + 0.3 * Gg_t / ng
                mx = float(np.linalg.norm(Grad, axis=1).max()) + 1e-12
                Grad = Grad / mx
            else:
                mx = float(np.linalg.norm(Gb_t, axis=1).max()) + 1e-12
                Grad = Gb_t / mx
            if not np.all(np.isfinite(Grad)):
                break
            V = mu * V + Grad
            X = X - lr * V
            X = _norm(X)
    return X

def _largest_hole(Xf, rng, n_cand=20000):
    C = _norm(rng.randn(n_cand, Xf.shape[1]))
    dots = C @ Xf.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))].copy()

def _refine_hole(c, Xf, iters=150, lr0=0.08, lr1=0.005, tau=0.01):
    c = c / max(1e-12, np.linalg.norm(c))
    for t in range(iters):
        f = t / max(1, iters - 1)
        lr = lr0 * ((lr1 / lr0) ** f)
        d = Xf @ c
        m = float(d.max())
        w = np.exp((d - m) / tau)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            break
        w = w / s
        g = w @ Xf
        gt = g - float(g @ c) * c
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14:
            break
        c = c - lr * gt / nrm * 2.0
        c = c / max(1e-12, np.linalg.norm(c))
    return c

def _pick_victim(X, mode, rng):
    m, cnt, nn, G = _contacts(X, tol=1e-7)
    n = X.shape[0]
    zero = np.where(cnt == 0)[0]
    if len(zero) > 0:
        return int(zero[0])
    if mode == "low":
        mn = int(cnt.min())
        cand = np.where(cnt == mn)[0]
        if len(cand) == 1:
            return int(cand[0])
        # most isolated among them (smallest nn)
        return int(cand[np.argmin(nn[cand])])
    elif mode == "high":
        mx = int(cnt.max())
        cand = np.where(cnt == mx)[0]
        if len(cand) == 1:
            return int(cand[0])
        return int(cand[np.argmax(nn[cand])])
    elif mode == "random":
        return int(rng.randint(n))
    else:
        return int(rng.randint(n))

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    final_reserve = 20.0
    rng_global = np.random.RandomState(12345)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s
                    Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng_global.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy()
    s_best = float(sb)
    Kdown = 10  # 162 -> 152
    modes = ["low", "high", "random", "low"]
    seeds = [101, 202, 303, 404]
    # quick initial LP on best if time allows (small)
    # deletion-regrowth chains
    for mi, mode in enumerate(modes):
        if time.time() > deadline - final_reserve - 8:
            break
        rng = np.random.RandomState(seeds[mi])
        X_cur = X_best.copy() if mi == 0 else Xb.copy()
        # down phase
        ok = True
        for k in range(Kdown):
            if time.time() > deadline - final_reserve - 6:
                ok = False
                break
            vic = _pick_victim(X_cur, mode, rng)
            mask = np.ones(X_cur.shape[0], dtype=bool)
            mask[vic] = False
            X_cur = X_cur[mask]
            # polish smaller code (under-full -> lower max)
            budget = 0.7
            remain = deadline - final_reserve - time.time() - 2
            if remain < 0.5:
                ok = False
                break
            budget = min(budget, max(0.3, remain / 8.0))
            X_cur = _lp_polish(X_cur, budget, deadline=deadline - final_reserve)
        if not ok:
            continue
        # up phase: 152 -> 162
        for up in range(Kdown):
            if time.time() > deadline - final_reserve - 4:
                ok = False
                break
            hole = _largest_hole(X_cur, rng, n_cand=20000)
            hole = _refine_hole(hole, X_cur, iters=120)
            X_cur = np.vstack([X_cur, hole[None, :]])
            X_cur = _norm(X_cur)
            # balanced smooth phase tau=0.005,0.002,0.0007
            X_cur = _balanced_smooth(X_cur, rng, taus=(0.005, 0.002, 0.0007), iters=160, lr0=0.016, lr1=0.003)
            budget = 0.7
            remain = deadline - final_reserve - time.time() - 1
            if remain < 0.4:
                # no LP, continue with smooth only
                continue
            budget = min(budget, max(0.3, remain / 6.0))
            X_cur = _lp_polish(X_cur, budget, deadline=deadline - final_reserve)
        if X_cur.shape[0] != 162:
            continue
        try:
            s_cur = _max_cos(_norm(X_cur))
        except Exception:
            continue
        if s_cur < s_best and np.all(np.isfinite(X_cur)):
            X_best = _norm(X_cur.copy())
            s_best = float(s_cur)
    # final 20-second LP polish on best completed 162 code
    remain = deadline - time.time() - 1.0
    if remain > 2.0:
        budget = min(final_reserve, remain)
        Xp = _lp_polish(X_best, budget, deadline=deadline - 0.5)
        try:
            sp = _max_cos(Xp)
        except Exception:
            sp = s_best
        if np.isfinite(sp) and sp < s_best:
            X_best, s_best = Xp, sp
    X_best = _norm(X_best)
    return X_best
