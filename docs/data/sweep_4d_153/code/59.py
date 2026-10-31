import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _edge_set(X, tol=1e-3):
    n = X.shape[0]
    G = X @ X.T
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    m = float(c.max())
    sel = c > m - tol
    I = iu0[0][sel]
    J = iu0[1][sel]
    s = set((int(i) * n + int(j)) for i, j in zip(I, J))
    return s, m

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union

def _anneal_full(X0, iters, k0, k1, lr0, lr1, deadline):
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    for t in range(iters):
        if t % 50 == 0 and time.time() > deadline:
            break
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # clip extreme grads for stability
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 1.0 / gn)
        # clip only when huge
        gt = gt * np.where(gn > 2.0, 2.0 / gn, 1.0)
        X = X - lr * gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _anneal_frozen(X0, frozen_mask, iters, k0, k1, lr0, lr1, deadline):
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    mov = np.where(~frozen_mask)[0]
    if len(mov) == 0:
        return X
    for t in range(iters):
        if t % 50 == 0 and time.time() > deadline:
            break
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(k * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt * np.where(gn > 2.0, 2.0 / gn, 1.0)
        Xm = X[mov] - lr * gt[mov]
        Xm /= np.linalg.norm(Xm, axis=1, keepdims=True)
        X[mov] = Xm
    return X

def _lp_polish(X, deadline, r0=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _quasi_hopf(n, offset):
    idx = np.arange(n, dtype=float)
    a1 = 0.7548776662466927
    a2 = 0.5698402909980532
    a3 = 0.3671564005865769
    s = (idx * a1 + offset) % 1.0
    p1 = 2.0 * np.pi * ((idx * a2 + offset * 1.3) % 1.0)
    p2 = 2.0 * np.pi * ((idx * a3 + offset * 2.1) % 1.0)
    c = np.sqrt(1.0 - s)
    sr = np.sqrt(s)
    X = np.stack([c * np.cos(p1), c * np.sin(p1), sr * np.cos(p2), sr * np.sin(p2)], axis=1)
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    n, d = 153, 4
    rng = np.random.default_rng(12345)
    # load incumbent
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cand.append(a)
        except Exception:
            pass
    if cand:
        vals = [_max_cos(a) for a in cand]
        incumbent = cand[int(np.argmin(vals))].copy()
    else:
        incumbent = rng.standard_normal((n, d))
        incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    inc_val = _max_cos(incumbent)
    try:
        E_inc, _ = _edge_set(incumbent, tol=1e-3)
    except Exception:
        E_inc = set()
    best_X = incumbent.copy()
    best_val = inc_val

    # build 8 starts: 4 random + 4 Hopf/Fibonacci quasi-uniform
    starts = []
    for s in (11, 22, 33, 44):
        r = np.random.default_rng(s)
        X = r.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        starts.append(X)
    for off in (0.0, 0.25, 0.5, 0.75):
        X = _quasi_hopf(n, off)
        # tiny jitter to break exact lattice, deterministic
        r = np.random.default_rng(int(off * 100) + 999)
        j = r.standard_normal((n, d)) * 1e-3
        X = X + j
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        starts.append(X)

    pool = []  # (val, X, edge, jacc)
    # fixed short schedule: k up, lr down
    ITERS = 420
    K0, K1 = 15.0, 500.0
    LR0, LR1 = 0.20, 0.01
    for X0 in starts:
        if time.time() > deadline - 60.0:
            break
        X = _anneal_full(X0, ITERS, K0, K1, LR0, LR1, deadline)
        try:
            v = _max_cos(X)
            E, _ = _edge_set(X, tol=1e-3)
            j = _jaccard(E, E_inc)
        except Exception:
            v = 1e9
            E = set()
            j = 0.0
        pool.append((v, X, E, j))
        if v < best_val:
            best_val = v
            best_X = X.copy()

    # rank by value, deprioritise near-identical to incumbent
    # penalise high Jaccard slightly: sort by v + small penalty
    pool_sorted = sorted(pool, key=lambda t: t[0] + 0.002 * t[3])
    # pick best two mutually diverse (Jaccard between them < 0.5)
    chosen = []
    for item in pool_sorted:
        ok = True
        for c in chosen:
            try:
                if _jaccard(item[2], c[2]) > 0.55:
                    ok = False
                    break
            except Exception:
                pass
        # also require diversity vs incumbent unless clearly better
        if item[3] > 0.75 and item[0] >= inc_val - 1e-6:
            ok = False
        if ok:
            chosen.append(item)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        # fill with top values regardless
        for item in pool_sorted:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) >= 2:
                break
    # LP-polish only best two diverse basins
    for item in chosen[:2]:
        if time.time() > deadline - 35.0:
            break
        X = item[1].copy()
        Xp = _lp_polish(X, min(deadline - 20.0, time.time() + 12.0))
        try:
            v = _max_cos(Xp)
        except Exception:
            continue
        if v < best_val:
            best_val = v
            best_X = Xp.copy()

    # if still worse than record, one graph-guided transplant
    rec_val = inc_val
    if best_val >= rec_val - 1e-12 and time.time() < deadline - 25.0 and len(pool) > 0:
        # source: most diverse (smallest Jaccard) among decent basins
        pool_by_div = sorted(pool, key=lambda t: (t[3], t[0]))
        src = pool_by_div[0][1]
        # cap center: use src point 0 direction (elementary, no alignment)
        # choose K=48 retained patch = top dots with center
        K = 48
        center = src[0] / np.linalg.norm(src[0])
        dots = src @ center
        order = np.argsort(-dots)
        keep_idx = order[:K]
        frozen = np.zeros(n, dtype=bool)
        # transplant keeps patch coordinates, complement re-optimised
        # build new config: patch from src, complement random outside cap
        cap_cos = float(np.sort(dots)[n - K])
        Xn = np.zeros((n, d))
        Xn[keep_idx] = src[keep_idx]
        Xn[keep_idx] /= np.linalg.norm(Xn[keep_idx], axis=1, keepdims=True)
        fr = np.zeros(n, dtype=bool)
        fr[keep_idx] = True
        # init complement outside cap
        r2 = np.random.default_rng(777)
        need = n - K
        comp = []
        guard = 0
        while len(comp) < need and guard < 20000:
            guard += 1
            y = r2.standard_normal((need - len(comp), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            dd = y @ center
            keep = dd < cap_cos - 0.05
            for k2 in np.where(keep)[0]:
                comp.append(y[k2])
                if len(comp) >= need:
                    break
        # fallback fill
        while len(comp) < need:
            y = r2.standard_normal(d)
            y /= np.linalg.norm(y)
            comp.append(y)
        mov_idx = np.where(~fr)[0]
        Xn[mov_idx] = np.array(comp[:len(mov_idx)])
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        # re-optimise complement against fixed patch
        Xn = _anneal_frozen(Xn, fr, 420, 15.0, 500.0, 0.20, 0.01, deadline - 12.0)
        # short full relaxation
        if time.time() < deadline - 15.0:
            Xn = _anneal_full(Xn, 150, 60.0, 600.0, 0.05, 0.005, deadline - 10.0)
        if time.time() < deadline - 8.0:
            Xn = _lp_polish(Xn, min(deadline - 2.0, time.time() + 8.0))
        try:
            v = _max_cos(Xn)
            if v < best_val:
                best_val = v
                best_X = Xn.copy()
        except Exception:
            pass

    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # tiny shrink guard not needed; just ensure finite
    best_X = np.where(np.isfinite(best_X), best_X, 0.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_X = best_X / nr
    return best_X
