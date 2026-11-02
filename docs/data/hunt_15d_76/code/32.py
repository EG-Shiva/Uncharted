import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal(X, n_iter, T0, T1, lr0, lr1, t_end=1e30):
    n, d = X.shape
    lT0 = np.log(T0); lT1 = np.log(T1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    for k in range(n_iter):
        if k % 500 == 0 and time.time() > t_end:
            break
        f = k / max(1, n_iter - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(llr0 + (llr1 - llr0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        # renorm
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X = X / nn
    return X

def _refine_hole(v, Y, n_iter=150, T0=0.008, lr0=0.08):
    v = v / np.linalg.norm(v)
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        Tk = T0 * (0.15 ** f)
        lrk = lr0 * (0.2 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if sw == 0 or not np.isfinite(sw):
            break
        w = w / sw
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            break
        v = v / nv
    return v

def _lp_polish(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(x.astype(float))
    n, d = x.shape
    best = _maxcos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.6 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 900:
            idx = np.argpartition(-c, 900)[:900]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    t_end = t0 + BUDGET
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = np.load("record.npy").astype(float)
    if Xb.shape != (76, 15):
        rng0 = np.random.default_rng(0)
        Xb = rng0.normal(size=(76, 15))
    Xb = _norm(Xb)
    best_val = _maxcos(Xb)
    Xbest = Xb.copy()
    n, d = 76, 15

    def contact_counts(X, thr=0.002):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        C = (G > m - thr).sum(axis=1)
        return C, m

    sigmas = [0.05, 0.09, 0.14, 0.03, 0.07, 0.11, 0.025, 0.16]
    trial = 0
    # reserve ~22s for final LP
    while time.time() < t_end - 25 and trial < 250:
        trial += 1
        rng = np.random.default_rng(1000 + trial)
        sigma = sigmas[(trial - 1) % len(sigmas)]
        cnt, mcur = contact_counts(Xbest)
        order = np.argsort(-cnt)
        # number to teleport
        if trial % 3 == 0:
            k_re = 3
        elif trial % 2 == 0:
            k_re = 2
        else:
            k_re = 1
        Xt = Xbest.copy()
        # choose indices: mostly overloaded, sometimes random
        cand = []
        cand.append(int(order[(trial - 1) % n]))
        if k_re >= 2:
            cand.append(int(order[(trial * 7) % n]))
        if k_re >= 3:
            cand.append(int(rng.integers(0, n)))
        cand = list(dict.fromkeys(cand))
        for idx in cand:
            if time.time() > t_end - 26:
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Y = Xt[mask]
            NS = 2500
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = _refine_hole(C[bi], Y, n_iter=150, T0=0.008, lr0=0.08)
            Xt[idx] = v0
        # global shake
        Xt = Xt + sigma * rng.normal(size=(n, d))
        Xt = _norm(Xt)
        # anneal params by sigma
        if sigma < 0.04:
            Xt = _anneal(Xt, n_iter=2200, T0=0.012, T1=0.0010, lr0=4.0, lr1=0.15, t_end=t_end - 25)
        elif sigma < 0.10:
            Xt = _anneal(Xt, n_iter=2800, T0=0.016, T1=0.0009, lr0=5.0, lr1=0.12, t_end=t_end - 25)
        else:
            Xt = _anneal(Xt, n_iter=3200, T0=0.022, T1=0.0012, lr0=6.0, lr1=0.15, t_end=t_end - 25)
        v = _maxcos(Xt)
        if v < best_val - 1e-12:
            # extra low-T squeeze
            Xt2 = _anneal(Xt, n_iter=1200, T0=0.0010, T1=0.00025, lr0=0.8, lr1=0.05, t_end=t_end - 25)
            v2 = _maxcos(Xt2)
            if v2 < v:
                Xt, v = Xt2, v2
            best_val = float(v)
            Xbest = Xt.copy()
    # final LP squeeze if time remains
    if time.time() < t_end - 2:
        try:
            Xbest = _lp_polish(Xbest, t_end)
        except Exception:
            pass
        best_val = _maxcos(Xbest)
    # safety: if we somehow got worse than loaded best, return loaded best
    if best_val > _maxcos(Xb) + 1e-12:
        return _norm(Xb)
    return _norm(Xbest)
