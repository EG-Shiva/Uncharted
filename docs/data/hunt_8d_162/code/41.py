import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_opt(X, steps, b0, b1, lr0, lr1, noise0, noise1, rng, mu=0.7):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(1, steps - 1)
        b = b0 * ((b1 / b0) ** f) if b1 != b0 else b0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Gm = float(G.max())
        W = np.exp(b * (G - Gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Gt = (H - WG[:, None] * X) / S
        # project to tangent (already ~tangent but enforce)
        dot = (Gt * X).sum(axis=1, keepdims=True)
        Gt = Gt - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            # still add noise
            if nm > 0:
                N = rng.randn(n, d)
                nd = (N * X).sum(axis=1, keepdims=True)
                N = N - nd * X
                X = _norm(X + nm * N)
            continue
        V = mu * V + Gt / mx
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-cf, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask if False else mask
            cnt = 2500
        I = iuI[sel]
        J = iuJ[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X = Y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return X

def _pick_victims(X, k, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    per_max = G.max(axis=1)
    tight = (G > m - 1e-9).sum(axis=1)
    ratt = np.where(tight == 0)[0].tolist()
    order = np.argsort(-per_max)
    pool = order[:max(18, k + 12)].tolist()
    rng.shuffle(pool)
    victims = []
    for r in ratt:
        if len(victims) < k:
            victims.append(int(r))
    for c in pool:
        if len(victims) >= k:
            break
        c = int(c)
        if c not in victims:
            victims.append(c)
    while len(victims) < k:
        v = int(rng.randint(0, n))
        if v not in victims:
            victims.append(v)
    return np.array(victims[:k], dtype=int)

def _holes(X_keep, k, rng, n_samp=24000):
    d = X_keep.shape[1]
    C = rng.randn(n_samp, d)
    C = _norm(C)
    S = C @ X_keep.T
    sc = S.max(axis=1)
    order = np.argsort(sc)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.60:
                ok = False
                break
        if ok:
            picked.append(v.copy())
        if len(picked) >= k:
            break
        # allow denser packing after scanning half
        if idx > n_samp // 2 and len(picked) < k:
            # relax separation
            pass
    while len(picked) < k:
        v = rng.randn(d)
        v = v / max(1e-12, np.linalg.norm(v))
        picked.append(v)
    return np.array(picked)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(20261103)
    X_best = None
    s_best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < s_best:
                    s_best = s
                    X_best = v.copy()
        except Exception:
            pass
    if X_best is None:
        X_best = _norm(rng.randn(162, 8))
        s_best = _max_cos(X_best)
    n, d = X_best.shape
    trial = 0
    # exploration phase, reserve ~42s for final polish
    explore_end = t0 + 122.0
    if deadline - explore_end < 42:
        explore_end = deadline - 42
    while time.time() < explore_end and time.time() < deadline - 42:
        trial += 1
        mode = trial % 6
        Xt = X_best.copy()
        if mode == 0:
            k = 1
            vic = _pick_victims(Xt, k, rng)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[vic] = False
            Xk = Xt[keep_mask]
            nh = _holes(Xk, k, rng)
            Xt[vic] = nh
            sig = 0.004
            Xt = _norm(Xt + rng.randn(n, d) * sig)
            Xt = _smooth_opt(Xt, 420, 28, 160, 0.010, 0.003, 0.0022, 0.0, rng, mu=0.75)
            Xt = _smooth_opt(Xt, 420, 160, 700, 0.0045, 0.0012, 0.0, 0.0, rng, mu=0.5)
        elif mode == 1:
            k = 2
            vic = _pick_victims(Xt, k, rng)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[vic] = False
            Xk = Xt[keep_mask]
            nh = _holes(Xk, k, rng)
            Xt[vic] = nh
            sig = 0.007
            Xt = _norm(Xt + rng.randn(n, d) * sig)
            Xt = _smooth_opt(Xt, 450, 25, 150, 0.011, 0.0025, 0.0028, 0.0, rng, mu=0.78)
            Xt = _smooth_opt(Xt, 450, 150, 800, 0.0045, 0.0010, 0.0, 0.0, rng, mu=0.5)
        elif mode == 2:
            k = 3 + (trial % 2)
            vic = _pick_victims(Xt, k, rng)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[vic] = False
            Xk = Xt[keep_mask]
            nh = _holes(Xk, k, rng)
            Xt[vic] = nh
            sig = 0.010
            Xt = _norm(Xt + rng.randn(n, d) * sig)
            Xt = _smooth_opt(Xt, 500, 22, 140, 0.012, 0.0025, 0.0032, 0.0, rng, mu=0.8)
            Xt = _smooth_opt(Xt, 400, 140, 650, 0.005, 0.0012, 0.0, 0.0, rng, mu=0.5)
        elif mode == 3:
            sig = 0.016 + 0.008 * rng.rand()
            Xt = _norm(Xt + rng.randn(n, d) * sig)
            Xt = _smooth_opt(Xt, 500, 25, 160, 0.011, 0.0025, 0.0025, 0.0, rng, mu=0.75)
            Xt = _smooth_opt(Xt, 400, 160, 750, 0.0045, 0.0010, 0.0, 0.0, rng, mu=0.5)
        elif mode == 4:
            sig = 0.030 + 0.015 * rng.rand()
            Xt = _norm(Xt + rng.randn(n, d) * sig)
            Xt = _smooth_opt(Xt, 600, 20, 120, 0.013, 0.003, 0.004, 0.0, rng, mu=0.82)
            Xt = _smooth_opt(Xt, 400, 120, 600, 0.005, 0.0012, 0.0, 0.0, rng, mu=0.5)
        else:
            # neighbourhood shake around a random overloaded point
            G0 = Xt @ Xt.T
            np.fill_diagonal(G0, -2.0)
            pm = G0.max(axis=1)
            c0 = int(np.argmax(pm + 0.002 * rng.randn(n)))
            neigh = np.where(G0[c0] > 0.45)[0]
            if len(neigh) == 0:
                neigh = np.array([c0])
            sig = 0.05
            Xt[neigh] = _norm(Xt[neigh] + rng.randn(len(neigh), d) * sig)
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _smooth_opt(Xt, 450, 24, 150, 0.011, 0.0025, 0.003, 0.0, rng, mu=0.78)
            Xt = _smooth_opt(Xt, 450, 150, 800, 0.0045, 0.0010, 0.0, 0.0, rng, mu=0.5)
        s_sm = _max_cos(Xt)
        # filter: only LP-polish promising ones
        if s_sm < s_best + 0.0035:
            # short LP, 1.1s
            sec = 1.1
            if time.time() + sec > deadline - 40:
                sec = max(0.3, deadline - 40 - time.time())
            if sec > 0.25:
                Xt2 = _lp_polish(Xt, sec, t_deadline=deadline - 38)
            else:
                Xt2 = Xt
            s2 = _max_cos(Xt2)
            if s2 < s_best - 1e-12:
                X_best = Xt2.copy()
                s_best = float(s2)
        if time.time() > deadline - 40:
            break
    # final long polish of best
    remain = deadline - time.time() - 2.0
    if remain > 5:
        sec = min(remain, 38.0)
        X_best = _lp_polish(X_best, sec, t_deadline=deadline - 1.0)
    X_best = _norm(X_best)
    return X_best
