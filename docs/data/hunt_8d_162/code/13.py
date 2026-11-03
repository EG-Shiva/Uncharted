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

def _smooth_norm(X, iters, b0, b1, lr0, lr1, n0, n1, rng, mu=0.8):
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
        H = W @ X
        dot = (H * X).sum(axis=1, keepdims=True)
        Gt = H - dot * X
        mx = float(np.sqrt((Gt * Gt).sum(axis=1)).max())
        if not np.isfinite(mx) or mx < 1e-14:
            break
        Gt = Gt / mx
        V = mu * V + Gt
        # limit velocity to avoid explosion
        vm = float(np.sqrt((V * V).sum(axis=1)).max())
        if vm > 3.0:
            V = V / vm * 3.0
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
    iuI, iuJ = iu0[0], iu0[1]
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
        sel = cf > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        I = iuI[sel]; J = iuJ[sel]; c = cf[sel]
        if cnt > 3000:
            idx = np.argpartition(-c, 3000)[:3000]
            I = I[idx]; J = J[idx]; c = c[idx]
            cnt = 3000
        p = cnt
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
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _best_hole(X_keep, rng, n_samp=15000, repel=None):
    d = X_keep.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ X_keep.T
    sc = S.max(axis=1)
    if repel is not None and len(repel) > 0:
        R = np.array(repel)
        sc = np.maximum(sc, (C @ R.T).max(axis=1))
    return C[int(np.argmin(sc))]

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
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
    n, d = 162, 8

    trial = 0
    # exploration until ~135s, reserve ~35s for final polish
    while time.time() < t0 + 132.0:
        if time.time() > deadline - 40:
            break
        trial += 1
        Xt = X_best.copy()
        mode = trial % 7
        if mode == 0:  # cluster shake
            c = rng.randint(n)
            Gc = Xt @ Xt[c]
            nn = np.argpartition(-Gc, 12)[:12]
            sigma = 0.05 + 0.07 * rng.rand()
            Xt[nn] = _norm(Xt[nn] + rng.randn(len(nn), d) * sigma)
            Xt = _smooth_norm(Xt, 350, 60, 350, 0.014, 0.003, 0.003, 0.0, rng)
            Xt = _smooth_norm(Xt, 350, 350, 900, 0.006, 0.001, 0.0, 0.0, rng)
        elif mode == 1:  # kill overloaded + holes
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.006).sum(axis=1)
            k = int(rng.randint(2, 5))
            vict = np.argsort(-cnt)[:k + 2]
            # keep one random among top to add diversity
            vict = vict[rng.permutation(len(vict))][:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[vict] = False
            Xk = Xt[keep_mask]
            newpts = []
            for _ in range(k):
                Xall = Xk if len(newpts) == 0 else np.vstack([Xk] + newpts)
                h = _best_hole(Xall, rng, n_samp=15000, repel=newpts if len(newpts) else None)
                newpts.append(h[None, :])
            Xt[ vict] = np.vstack(newpts)
            Xt = _smooth_norm(Xt, 400, 60, 400, 0.014, 0.003, 0.002, 0.0, rng)
            Xt = _smooth_norm(Xt, 400, 400, 1000, 0.006, 0.0009, 0.0, 0.0, rng)
        elif mode == 2:  # kill random + holes
            k = int(rng.randint(3, 7))
            vict = rng.choice(n, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool); keep_mask[vict] = False
            Xk = Xt[keep_mask]
            newpts = []
            for _ in range(k):
                Xall = Xk if len(newpts) == 0 else np.vstack([Xk] + newpts)
                h = _best_hole(Xall, rng, n_samp=12000, repel=newpts if len(newpts) else None)
                newpts.append(h[None, :])
            Xt[vict] = np.vstack(newpts)
            Xt = _smooth_norm(Xt, 400, 50, 350, 0.016, 0.003, 0.003, 0.0, rng)
            Xt = _smooth_norm(Xt, 400, 350, 900, 0.006, 0.001, 0.0, 0.0, rng)
        elif mode == 3:  # global medium shake
            sigma = 0.02 + 0.02 * rng.rand()
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth_norm(Xt, 350, 80, 400, 0.012, 0.002, 0.002, 0.0, rng)
            Xt = _smooth_norm(Xt, 350, 400, 1000, 0.005, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:  # rattler / weakest relocation
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            k = int(rng.randint(1, 4))
            vict = np.argsort(worst)[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[vict] = False
            Xk = Xt[keep_mask]
            newpts = []
            for _ in range(k):
                Xall = Xk if len(newpts) == 0 else np.vstack([Xk] + newpts)
                h = _best_hole(Xall, rng, n_samp=15000, repel=newpts if len(newpts) else None)
                newpts.append(h[None, :])
            Xt[vict] = np.vstack(newpts)
            # small shake of neighbours of holes to make room
            Xt = _smooth_norm(Xt, 700, 70, 800, 0.010, 0.001, 0.001, 0.0, rng)
        elif mode == 5:  # single worst-point hole move + local shake
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            worst = G.max(axis=1)
            c = int(np.argmax(worst))
            mask = np.ones(n, dtype=bool); mask[c] = False
            h = _best_hole(Xt[mask], rng, n_samp=18000)
            Xt[c] = h
            # shake its new neighbours a bit
            Gn = Xt @ Xt[c]
            Gn[c] = -2
            nn = np.argpartition(-Gn, 8)[:8]
            Xt[nn] = _norm(Xt[nn] + rng.randn(len(nn), d) * 0.03)
            Xt = _smooth_norm(Xt, 700, 80, 900, 0.009, 0.0009, 0.001, 0.0, rng)
        else:  # small global shake fine search
            sigma = 0.008 + 0.008 * rng.rand()
            Xt = _norm(Xt + rng.randn(n, d) * sigma)
            Xt = _smooth_norm(Xt, 300, 100, 500, 0.008, 0.002, 0.001, 0.0, rng)
            Xt = _smooth_norm(Xt, 300, 500, 1200, 0.004, 0.0007, 0.0, 0.0, rng)

        s = _max_cos(Xt)
        # direct win
        if s < s_best - 1e-12:
            # lock in with short LP
            t_rem = deadline - time.time()
            if t_rem > 8:
                Xt2 = _lp_polish(Xt, min(4.0, t_rem - 4.0), t_deadline=deadline - 2.0)
                s2 = _max_cos(Xt2)
                if s2 < s:
                    Xt, s = Xt2, s2
            if s < s_best:
                X_best, s_best = Xt, float(s)
            continue
        # promising basin? polish briefly to test true potential
        thresh = 0.008 if mode in (1, 2) else 0.004
        if s < s_best + thresh and time.time() < deadline - 30:
            Xt2 = _lp_polish(Xt, 1.2, t_deadline=deadline - 2.0)
            s2 = _max_cos(Xt2)
            if s2 < s_best - 1e-12:
                X_best, s_best = Xt2, float(s2)

    # final long polish of best
    t_rem = deadline - time.time()
    if t_rem > 3:
        X_best = _lp_polish(X_best, t_rem - 1.0, t_deadline=deadline - 0.5)
    X_best = _norm(X_best)
    return X_best
