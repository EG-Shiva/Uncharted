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

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _smooth(X, iters, a0, a1, lr0, lr1, nz0, nz1, rng):
    n, d = X.shape
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a0 != a1 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
        if nz0 <= 0:
            nm = 0.0
        else:
            nm = nz0 * ((nz1 / nz0) ** f) if nz1 > 0 else nz0 * (1.0 - f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        W = np.exp(a * (G - gm))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            break
        H = W @ X
        WG = (W * G).sum(axis=1)
        Grad = (H - WG[:, None] * X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            if nm <= 0:
                continue
            mx = 1.0
        else:
            X = X - (lr / mx) * Gt
        if nm > 0:
            N = rng.randn(n, d)
            nd = (N * X).sum(axis=1, keepdims=True)
            N = N - nd * X
            nn = np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
            N = N / nn
            X = X + nm * N
        X = _norm(X)
    return X

def _lp_polish(X, seconds, deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
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
        thresh = m - max(20.0 * radius, 1e-9)
        # indices of tight pairs
        sel_idx = np.where(cf > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > 2800:
            # keep tightest 2800
            part = np.argpartition(-cf[sel_idx], 2800)[:2800]
            sel_idx = sel_idx[part]
        I = iuI[sel_idx]; J = iuJ[sel_idx]
        c = cf[sel_idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

def _largest_holes(Xkeep, k, rng, n_samp=25000):
    d = Xkeep.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ Xkeep.T
    sc = S.max(axis=1)
    order = np.argsort(sc)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.6:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d); v /= (np.linalg.norm(v) + 1e-12)
        picked.append(v)
    return np.array(picked)

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
    n, d = X_best.shape

    trial = 0
    # exploration until deadline-30
    while time.time() < deadline - 30.0:
        trial += 1
        mode = trial % 7
        Xt = X_best.copy()
        if mode == 0:
            Xt = _norm(Xt + rng.randn(n, d) * 0.012)
            Xt = _smooth(Xt, 400, 30, 300, 0.012, 0.002, 0.002, 0.0, rng)
            Xt = _smooth(Xt, 300, 150, 900, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 1:
            Xt = _norm(Xt + rng.randn(n, d) * 0.035)
            Xt = _smooth(Xt, 500, 25, 250, 0.015, 0.002, 0.004, 0.0, rng)
            Xt = _smooth(Xt, 300, 150, 1000, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 2:
            # delete 2 worst, reinsert in holes
            rm, G = _row_max(Xt)
            k = 2
            del_idx = np.argsort(-rm)[:k]
            keep = np.ones(n, dtype=bool); keep[del_idx] = False
            holes = _largest_holes(Xt[keep], k, rng)
            Xt[del_idx] = holes
            Xt = _smooth(Xt, 500, 25, 300, 0.015, 0.002, 0.003, 0.0, rng)
            Xt = _smooth(Xt, 300, 150, 1000, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 3:
            rm, G = _row_max(Xt)
            k = int(rng.choice([4, 5, 6]))
            # mix worst + random
            worst = list(np.argsort(-rm)[:k + 4])
            rng.shuffle(worst)
            del_idx = np.array(worst[:k])
            keep = np.ones(n, dtype=bool); keep[del_idx] = False
            holes = _largest_holes(Xt[keep], k, rng)
            Xt[del_idx] = holes
            Xt = _smooth(Xt, 600, 20, 250, 0.016, 0.002, 0.004, 0.0, rng)
            Xt = _smooth(Xt, 350, 150, 1100, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 4:
            # cluster shake around random center
            ci = rng.randint(n)
            G0 = Xt @ Xt[ci]
            nn = np.argsort(-G0)[:12]
            Xt[nn] = _norm(Xt[nn] + rng.randn(len(nn), d) * 0.09)
            Xt = _smooth(Xt, 500, 25, 300, 0.014, 0.002, 0.003, 0.0, rng)
            Xt = _smooth(Xt, 300, 150, 1000, 0.006, 0.0008, 0.0, 0.0, rng)
        elif mode == 5:
            Xt = _norm(Xt + rng.randn(n, d) * 0.075)
            Xt = _smooth(Xt, 700, 18, 200, 0.018, 0.002, 0.006, 0.0, rng)
            Xt = _smooth(Xt, 350, 150, 1100, 0.006, 0.0008, 0.0, 0.0, rng)
        else:
            # rattler/low-contact relocation + shake
            Gf = Xt @ Xt.T
            np.fill_diagonal(Gf, -2.0)
            m0 = float(Gf.max())
            cnt = (Gf > m0 - 1e-7).sum(axis=1)
            # pick low-contact points and one worst
            order_low = np.argsort(cnt)
            k = 3
            del_idx = np.concatenate([order_low[:2], np.argsort(-Gf.max(axis=1))[:1]])
            # unique
            del_idx = np.unique(del_idx)[:k]
            # add random to reach k
            while len(del_idx) < k:
                r = rng.randint(n)
                if r not in del_idx:
                    del_idx = np.append(del_idx, r)
            keep = np.ones(n, dtype=bool); keep[del_idx] = False
            holes = _largest_holes(Xt[keep], len(del_idx), rng)
            Xt[del_idx] = holes
            Xt = _norm(Xt + rng.randn(n, d) * 0.008)
            Xt = _smooth(Xt, 500, 25, 300, 0.014, 0.002, 0.003, 0.0, rng)
            Xt = _smooth(Xt, 300, 150, 1000, 0.006, 0.0008, 0.0, 0.0, rng)

        s = _max_cos(Xt)
        # polish if promising
        if s < s_best + 0.004 and time.time() < deadline - 28:
            Xt = _lp_polish(Xt, 2.5, deadline - 26)
            s = _max_cos(Xt)
        if s < s_best:
            s_best = s
            X_best = Xt.copy()
            if time.time() < deadline - 26:
                # bonus short extra polish on improvement
                X_best = _lp_polish(X_best, 2.0, deadline - 26)
                s_best = _max_cos(X_best)
        if time.time() > deadline - 30:
            break

    # final long polish
    remain = deadline - time.time() - 1.0
    if remain > 3:
        X_best = _lp_polish(X_best, min(remain, 25.0), deadline - 0.5)
    X_best = _norm(X_best)
    return X_best
