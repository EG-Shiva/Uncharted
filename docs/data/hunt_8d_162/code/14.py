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

def _degrees(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > m - thr
    deg = C.sum(axis=1).astype(int)
    return m, deg, G

def _smooth(X, iters, b0, b1, lr0, lr1, n0, n1, rng, mu=0.8):
    X = _norm(X)
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(iters):
        f = t / max(1, iters - 1)
        beta = b0 * ((b1 / b0) ** f) if b0 != b1 else b0
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        if n0 <= 0:
            nm = 0.0
        else:
            nm = n0 * ((n1 / n0) ** f) if n1 > 0 else n0 * (1.0 - f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        P = W / S
        Grad = 2.0 * (P @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        if nm > 0:
            N = rng.randn(n, d)
            nd = np.sum(N * X, axis=1, keepdims=True)
            N = N - nd * X
            X = X + nm * N
        X = _norm(X)
    return X

def _lp(X, seconds, deadline=None, r0=0.004, rmax=0.05):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[I0, J0]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            # keep tightest 3500
            idx = np.argpartition(-cf, 3500)[:3500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            # ensure max included; argpartition already keeps top
            sel = mask
        I = I0[sel]; J = J0[sel]
        c = cf[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            X = y; best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return X

def _holes(X_keep, k, rng, n_samp=30000, sep=0.55):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ X_keep.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= np.linalg.norm(v)
        picked.append(v)
    return np.array(picked)

def solve():
    t0 = time.time()
    deadline = t0 + 173.0
    rng = np.random.RandomState(0)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8)); sb = _max_cos(Xb)
    X_best = Xb.copy(); s_best = float(sb)
    n, d = X_best.shape

    trial = 0
    noimp = 0
    # exploration until ~135s, reserve rest for final polish
    while time.time() < t0 + 132.0:
        if time.time() > deadline - 45:
            break
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        # contact info of best
        try:
            _, deg, Gfull = _degrees(X_best, thr=1e-7)
        except Exception:
            deg = np.ones(n, dtype=int) * 10
        order_deg = np.argsort(deg)
        order_deg_desc = order_deg[::-1]
        grow = 1.0 + min(noimp / 25.0, 1.5)

        if mode == 0:
            sigma = 0.008 * grow
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth(Xt, 600, 30, 400, 0.014, 0.0015, 0.0025, 0.0, rng)
            Xt = _smooth(Xt, 350, 100, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            sigma = 0.028 * grow
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth(Xt, 600, 25, 350, 0.016, 0.0015, 0.003, 0.0, rng)
            Xt = _smooth(Xt, 350, 100, 1000, 0.006, 0.0007, 0.0, 0.0, rng)
        elif mode == 2:
            sigma = 0.060 * min(grow, 1.6)
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth(Xt, 800, 20, 300, 0.018, 0.0015, 0.004, 0.0, rng)
            Xt = _smooth(Xt, 350, 120, 1100, 0.005, 0.0006, 0.0, 0.0, rng)
        elif mode == 3:
            # overload shake: top hubs + neighbours
            k_hub = 3 + (trial % 3)
            hubs = order_deg_desc[:8].copy()
            rng.shuffle(hubs)
            hubs = hubs[:k_hub]
            # neighbours with large dot
            G = X_best @ X_best.T
            for h in hubs:
                row = G[h].copy(); row[h] = -2
                nb = np.argsort(-row)[:12]
                idx = np.concatenate([[h], nb])
                Xt[idx] = _norm(Xt[idx] + rng.randn(len(idx), d) * (0.05 * grow))
            # small global noise
            Xt = _norm(Xt + rng.randn(n, d) * 0.004)
            Xt = _smooth(Xt, 650, 30, 450, 0.014, 0.0012, 0.002, 0.0, rng)
            Xt = _smooth(Xt, 350, 120, 1000, 0.005, 0.0006, 0.0, 0.0, rng)
        elif mode == 4:
            # hole replacement of worst points
            k = 4 + (trial % 3)  # 4..6
            # worst = highest max-dot
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            worst = np.argsort(-G.max(axis=1))[:k]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            X_keep = Xt[keep]
            newp = _holes(X_keep, k, rng, n_samp=25000, sep=0.5)
            Xt[worst] = newp
            Xt = _smooth(Xt, 700, 25, 400, 0.015, 0.0012, 0.0025, 0.0, rng)
            Xt = _smooth(Xt, 350, 120, 1000, 0.005, 0.0006, 0.0, 0.0, rng)
        elif mode == 5:
            # relocate rattlers / low-degree
            k = 3
            lows = order_deg[:10].copy()
            rng.shuffle(lows)
            lows = lows[:k]
            keep = np.ones(n, dtype=bool); keep[lows] = False
            X_keep = Xt[keep]
            newp = _holes(X_keep, k, rng, n_samp=25000, sep=0.5)
            Xt[lows] = newp
            Xt = _smooth(Xt, 650, 30, 450, 0.014, 0.0012, 0.0015, 0.0, rng)
            Xt = _smooth(Xt, 350, 150, 1100, 0.005, 0.0006, 0.0, 0.0, rng)
        else:
            # targeted pair split: pick random contact pair
            G = X_best @ X_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            iu = np.triu_indices(n, 1)
            c = G[iu]
            tight = np.where(c > m - 1e-6)[0]
            if len(tight) > 0:
                pick = tight[rng.randint(len(tight))]
                i = int(iu[0][pick]); j = int(iu[1][pick])
                diff = Xt[i] - Xt[j]
                diff = diff - (diff @ Xt[i]) * Xt[i]
                dn = np.linalg.norm(diff)
                if dn > 1e-12:
                    diff /= dn
                    Xt[i] = _norm((Xt[i] + diff * 0.02 * grow).reshape(1, -1))[0]
                    Xt[j] = _norm((Xt[j] - diff * 0.02 * grow).reshape(1, -1))[0]
            Xt = _norm(Xt + rng.randn(n, d) * 0.010 * grow)
            Xt = _smooth(Xt, 600, 40, 500, 0.013, 0.0012, 0.002, 0.0, rng)
            Xt = _smooth(Xt, 350, 150, 1100, 0.005, 0.0006, 0.0, 0.0, rng)

        s = _max_cos(Xt)
        # decide LP: if promising
        tol = 0.004 if noimp < 40 else 0.008
        if s < s_best + tol:
            # short polish
            budget = 1.6 if s < s_best + 0.001 else 1.0
            # don't exceed exploration window
            if time.time() + budget > t0 + 133.0:
                budget = max(0.4, (t0 + 133.0) - time.time())
            Xp = _lp(Xt, budget, deadline=deadline - 30)
            sp = _max_cos(Xp)
            if sp < s_best - 1e-12:
                X_best = Xp.copy(); s_best = float(sp)
                noimp = 0
            else:
                noimp += 1
        else:
            noimp += 1

    # final intensive polish of best
    rem = deadline - time.time() - 2.0
    if rem > 5:
        # alternate smooth + LP
        Xc = X_best.copy()
        # light smooth to escape tiny basin then LP
        half = rem * 0.25
        try:
            Xc = _smooth(Xc, 400, 200, 1500, 0.003, 0.0005, 0.0, 0.0, rng)
        except Exception:
            pass
        if _max_cos(Xc) < s_best:
            X_best = Xc; s_best = _max_cos(Xc)
        X_best = _lp(X_best, rem * 0.75, deadline=deadline - 1.0, r0=0.003)
        s_best = _max_cos(X_best)
    X_best = _norm(X_best)
    return X_best
