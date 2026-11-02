import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_opt(X_init, steps, T0, T1, lr0, lr1):
    X = X_init
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mv = float(np.max(G))
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
        X /= nr
    return X

def lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > t_deadline:
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
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        Xb = rng.standard_normal(size=(n, d))
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
        best_val = max_cos(Xb)
    else:
        cands.sort(key=lambda t: t[0])
        best_val, Xb = cands[0][0], cands[0][1].copy()
    # quick hole helper inline
    def hole_replace(Y, idx, nc=1500):
        others = np.delete(Y, idx, axis=0)
        cand = rng.standard_normal(size=(nc, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = np.max(dots, axis=1)
        return cand[np.argmin(worst)]

    trial = 0
    # basin hopping, stop early enough for final polish
    while time.time() < t0 + 132.0 and time.time() < deadline - 32.0:
        trial += 1
        mode = trial % 6
        Y = Xb.copy()
        if mode == 1:
            Y = Y + 0.005 * rng.standard_normal(size=Y.shape)
        elif mode == 2:
            Y = Y + 0.018 * rng.standard_normal(size=Y.shape)
        elif mode == 3:
            # shake max pair harder
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(np.max(G, axis=1)))
            # find partner
            j = int(np.argmax(G[k]))
            Y[k] += 0.02 * rng.standard_normal(size=d)
            Y[j] += 0.02 * rng.standard_normal(size=d)
            Y += 0.004 * rng.standard_normal(size=Y.shape)
        elif mode == 4:
            # worst point hole reinsert
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            w = int(np.argmax(pm))
            if rng.random() < 0.5:
                # second worst too
                w2 = int(np.argsort(pm)[-2])
                Y = Xb + 0.006 * rng.standard_normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y[w] = hole_replace(Y, w)
                Y[w2] = hole_replace(Y, w2)
            else:
                Y = Xb + 0.006 * rng.standard_normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                Y[w] = hole_replace(Y, w)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Yo = smooth_opt(Y, steps=800, T0=0.010, T1=0.0009, lr0=0.007, lr1=0.0004)
            v = max_cos(Yo)
            if v < best_val:
                Yo2 = smooth_opt(Yo, steps=500, T0=0.002, T1=0.0004, lr0=0.002, lr1=0.0002)
                v2 = max_cos(Yo2)
                if v2 < v:
                    Yo, v = Yo2, v2
                Xb, best_val = Yo, v
            continue
        elif mode == 5:
            # rattler reinsert: points with few contacts
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            mv = float(np.max(G))
            cnt = np.sum(G > mv - 0.005, axis=1)
            order = np.argsort(cnt)
            k = int(rng.integers(1, 3))
            targets = order[:k]
            Y = Xb + 0.008 * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            for ti in targets:
                Y[int(ti)] = hole_replace(Y, int(ti))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Yo = smooth_opt(Y, steps=800, T0=0.010, T1=0.0009, lr0=0.007, lr1=0.0004)
            v = max_cos(Yo)
            if v < best_val:
                Yo2 = smooth_opt(Yo, steps=500, T0=0.002, T1=0.0004, lr0=0.002, lr1=0.0002)
                v2 = max_cos(Yo2)
                if v2 < v:
                    Yo, v = Yo2, v2
                Xb, best_val = Yo, v
            continue
        else:
            Y = Y + 0.009 * rng.standard_normal(size=Y.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        if trial % 2 == 0:
            Yo = smooth_opt(Y, steps=800, T0=0.012, T1=0.0010, lr0=0.008, lr1=0.0004)
        else:
            Yo = smooth_opt(Y, steps=800, T0=0.007, T1=0.0007, lr0=0.006, lr1=0.0003)
        v = max_cos(Yo)
        if v < best_val:
            Yo2 = smooth_opt(Yo, steps=500, T0=0.002, T1=0.0004, lr0=0.002, lr1=0.0002)
            v2 = max_cos(Yo2)
            if v2 < v:
                Yo, v = Yo2, v2
            Xb, best_val = Yo, v
        if time.time() > deadline - 32.0:
            break
    if time.time() < deadline - 28.0:
        # medium refine
        Xb = smooth_opt(Xb, steps=1200, T0=0.004, T1=0.0004, lr0=0.004, lr1=0.0002)
        best_val = max_cos(Xb)
    rem = deadline - time.time() - 1.0
    if rem > 3.0:
        Xb = lp_polish(Xb, seconds=min(rem, 26.0), t_deadline=deadline - 0.5)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb
