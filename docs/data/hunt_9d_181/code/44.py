import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    return m, cnt, G

def _anneal(X, steps, T0, T1, lr0, lr1, mu=0.75, rng=None):
    n, d = X.shape
    Y = X.copy()
    V = np.zeros_like(Y)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        # softmax weights
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        Dir = WY - r[:, None] * Y
        mn = np.mean(np.linalg.norm(Dir, axis=1)) + 1e-12
        Dir /= mn
        V *= mu
        V += Dir
        Y -= lr * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y

def _max_descent(Y, steps=250, lr=0.004, tol=5e-4):
    # unweighted repulsion from near-max pairs only (fast LP surrogate)
    n, d = Y.shape
    for _ in range(steps):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        M = G > m - tol
        # symmetric mask, no diag
        W = M.astype(np.float64)
        np.fill_diagonal(W, 0.0)
        # degree normalize to avoid hub dominance
        deg = W.sum(axis=1, keepdims=True) + 1.0
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        Dir = WY - r[:, None] * Y
        Dir /= deg
        mn = np.mean(np.linalg.norm(Dir, axis=1)) + 1e-12
        if mn < 1e-12:
            break
        Dir /= mn
        Y = Y - lr * Dir
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # decay tol to focus
        tol = max(tol * 0.995, 1e-5)
    return Y

def _void_for(Xkeep, rng, n_cand=4000):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30
    dots = cand @ Xkeep.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    # few refinement steps pushing away from neighbours
    for _ in range(30):
        dv = Xkeep @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xkeep
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02 * g
        v /= np.linalg.norm(v) + 1e-30
    return v / np.linalg.norm(v)

def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - t0 < budget and radius > 1e-14 and it < 60:
        it += 1
        if budget - (time.time() - t0) < 0.4 and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            # cap to 2800 tightest
            idx = np.where(sel)[0]
            if len(idx) > 2800:
                order = np.argsort(-c_all[idx])[:2800]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    gbest = cands[bi].copy()
    gval = vals[bi]
    # quick contact stats for surgery targeting
    def overloaded_idx(X, k):
        _, cnt, _ = _contacts(X)
        return np.argsort(-cnt)[:k], cnt
    # ---- iterated basin hopping with fast anneal, ~100s ----
    HOP_BUDGET = 105.0
    it = 0
    while time.time() - t0 < HOP_BUDGET:
        it += 1
        rem = BUDGET - (time.time() - t0)
        if rem < 15:
            break
        mode = it % 6
        y = gbest.copy()
        if mode == 1:
            sig = 0.025
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=500, T0=0.008, T1=0.0008, lr0=0.02, lr1=0.004)
        elif mode == 2:
            sig = 0.07
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=700, T0=0.012, T1=0.001, lr0=0.025, lr1=0.004)
        elif mode == 3:
            # targeted shake of overloaded hubs
            idx, _ = overloaded_idx(y, 24)
            noise = rng.standard_normal(y.shape) * 0.01
            noise[idx] = rng.standard_normal((len(idx), d)) * 0.16
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=700, T0=0.010, T1=0.0008, lr0=0.022, lr1=0.003)
        elif mode == 4:
            # void surgery: replace k worst by voids
            k = 3 + (it % 3)  # 3-5
            idx, _ = overloaded_idx(y, 30)
            # pick k among top30 randomly
            pick = rng.choice(idx, size=k, replace=False)
            keep_mask = np.ones(n, dtype=bool); keep_mask[pick] = False
            Xkeep = y[keep_mask]
            for p_ in pick:
                y[p_] = _void_for(Xkeep, rng, n_cand=3000)
                Xkeep = np.concatenate([Xkeep, y[p_:p_+1]], axis=0)
            y = _anneal(y, steps=800, T0=0.014, T1=0.001, lr0=0.028, lr1=0.004)
        elif mode == 5:
            sig = 0.13
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=800, T0=0.016, T1=0.0012, lr0=0.03, lr1=0.005)
        else:
            sig = 0.045
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, steps=600, T0=0.009, T1=0.0009, lr0=0.022, lr1=0.004)
        # fast max-only squeeze before scoring
        y = _max_descent(y, steps=200, lr=0.004, tol=5e-4)
        v = _max_cos(y)
        if v < gval:
            gbest, gval = y, v
        # time guard for anneal loops
        if time.time() - t0 > HOP_BUDGET:
            break
    # ---- final LP squeeze with remaining time ----
    rem = BUDGET - (time.time() - t0) - 2.0
    if rem > 5:
        gbest = _lp_polish(gbest, budget=min(rem, 55.0))
        gval = _max_cos(gbest)
        # one more tiny max-descent + micro LP if time left
        rem2 = BUDGET - (time.time() - t0) - 1.0
        if rem2 > 5:
            gbest = _max_descent(gbest, steps=150, lr=0.002, tol=2e-4)
            gbest = _lp_polish(gbest, budget=min(rem2 - 1.0, 20.0))
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    # shrink margin: ensure no NaN/zero
    return gbest
