import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_opt(X_init, steps, T0, T1, lr0, lr1):
    X = X_init.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    # precompute log ratio for geometric interp
    lt0, lt1 = np.log(T0), np.log(T1)
    lrlt0, lrlt1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(lrlt0 + (lrlt1 - lrlt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mv = float(np.max(G))
        # softmax weights over ordered off-diag pairs
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S == 0.0 or not np.isfinite(S):
            break
        E /= S
        Gr = E @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        upd = m1h / (np.sqrt(m2h) + eps)
        X = X - lr * upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # load seeds
    cands = []
    try:
        cands.append(np.load("best.npy"))
    except Exception:
        pass
    try:
        cands.append(np.load("record.npy"))
    except Exception:
        pass
    if len(cands) == 0:
        Xb = rng.standard_normal(size=(82, 14))
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    else:
        vals = []
        for c in cands:
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (82, 14):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            vals.append((max_cos(c), c))
        vals.sort(key=lambda t: t[0])
        Xb = vals[0][1].copy()
    best_val = max_cos(Xb)
    n, d = 82, 14

    def hole_candidate(X, idx, n_cand=2000):
        others = np.delete(X, idx, axis=0)
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = np.max(dots, axis=1)
        return cand[np.argmin(worst)]

    trial = 0
    # Phase A: basin hopping until ~130s elapsed
    while time.time() < t0 + 128.0 and time.time() < deadline - 30.0:
        trial += 1
        mode = trial % 5
        Y = Xb.copy()
        if mode == 1:
            sig = 0.02
            Y = Y + sig * rng.standard_normal(size=Y.shape)
        elif mode == 2:
            sig = 0.07
            Y = Y + sig * rng.standard_normal(size=Y.shape)
        elif mode == 3:
            k = int(rng.integers(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = rng.standard_normal(size=(k, d))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y[mask] += 0.01 * rng.standard_normal(size=(n - k, d))
        elif mode == 4:
            # rattler / low-contact reinsertion
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            mv = float(np.max(G))
            cnt = np.sum(G > mv - 0.005, axis=1)
            order = np.argsort(cnt)
            k = int(rng.integers(1, 4))
            targets = order[:k]
            # small shake others first
            Y = Xb + 0.008 * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            for ti in targets:
                Y[ti] = hole_candidate(Y, int(ti), n_cand=2000)
        else:  # mode 0: single hole + medium shake
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            mv = float(np.max(G))
            cnt = np.sum(G > mv - 0.005, axis=1)
            ti = int(np.argmin(cnt)) if rng.random() < 0.7 else int(rng.integers(0, n))
            Y = Xb + 0.015 * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y[ti] = hole_candidate(Y, ti, n_cand=2000)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # annealed softmax Adam: short run
        # vary schedule slightly by trial for diversity
        if trial % 2 == 0:
            Yo = smooth_opt(Y, steps=900, T0=0.015, T1=0.0012, lr0=0.008, lr1=0.0004)
        else:
            Yo = smooth_opt(Y, steps=900, T0=0.008, T1=0.0008, lr0=0.006, lr1=0.0003)
        v = max_cos(Yo)
        if v < best_val:
            # extra refine at low T before accepting
            Yo2 = smooth_opt(Yo, steps=400, T0=0.002, T1=0.0005, lr0=0.002, lr1=0.0002)
            v2 = max_cos(Yo2)
            if v2 < v:
                Yo, v = Yo2, v2
            Xb, best_val = Yo, v
        # safety: stop if time low
        if time.time() > deadline - 30.0:
            break

    # Phase B: longer refine of best
    if time.time() < deadline - 25.0:
        Xb = smooth_opt(Xb, steps=1200, T0=0.005, T1=0.0005, lr0=0.004, lr1=0.0002)
        best_val = max_cos(Xb)
    # Phase C: LP polish with remaining time
    rem = deadline - time.time() - 1.0
    if rem > 3.0:
        Xb = lp_polish(Xb, seconds=min(rem, 28.0), t_deadline=deadline - 0.5)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb
