import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _single_point_opt(v, Y, steps=100, b0=80.0, b1=800.0, lr0=0.08, lr1=0.005):
    # minimize max_j <v,Yj> with Y fixed, v on sphere
    v = v / (np.linalg.norm(v) + 1e-12)
    m = Y.shape[0]
    if m == 0:
        return v
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ v  # (m,)
        dm = float(dots.max())
        w = np.exp(beta * (dots - dm))
        sw = float(w.sum()) + 1e-30
        w /= sw
        grad = w @ Y  # (d,)
        gdot = float(grad @ v)
        gt = grad - gdot * v
        v = v - lr * gt
        nrm = np.linalg.norm(v)
        if nrm == 0 or not np.isfinite(nrm):
            break
        v = v / nrm
    return v

def _softmax_quench(X, steps=110, b0=150.0, b1=2500.0, lr0=0.03, lr1=0.004):
    n, d = X.shape
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # per-row softmax (stable)
        M = G.max(axis=1, keepdims=True)
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        W /= rs
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip per-point step to 0.05
        sn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # scale lr*Gt
        step = lr * Gt
        s_n = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = s_n > 0.05
        step = np.where(over, step * (0.05 / s_n), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _lp_polish(X, seconds, deadline, max_iter=30, max_pairs=600, r0=5e-4):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > max_iter:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < max_pairs:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= max_pairs:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list, dtype=np.int64)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    EXPLORE_END = t0 + 130.0
    FINAL_BUDGET = 20.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (n, d):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val

    ks = [3, 4, 5, 6, 8]
    trial = 0
    # quick estimate of LP cost: use short LP only for promising quenches
    while time.time() < EXPLORE_END and time.time() < HARD_END - 25:
        trial += 1
        try:
            k = int(rng.choice(np.array(ks)))
        except Exception:
            k = ks[trial % len(ks)]
        mode = trial % 3  # 0 random, 1 hubs, 2 cluster
        # --- choose removal set ---
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
        except Exception:
            m0 = best_val
            G0 = None
        rem = None
        try:
            if mode == 1:
                gap = 0.003
                cnt = (G0 > m0 - gap).sum(axis=1)
                # add tiny random tie-break for determinism via rng
                jitter = rng.standard_normal(n) * 1e-6
                order = np.argsort(-(cnt.astype(float) + jitter))
                rem = order[:k]
            elif mode == 2:
                seed = int(rng.integers(n))
                dots = best[seed] @ best.T
                order = np.argsort(-dots)
                rem = order[:k]
            else:
                rem = rng.choice(n, size=k, replace=False)
        except Exception:
            rem = rng.choice(n, size=k, replace=False)
        rem = np.array(rem, dtype=int).ravel()
        mask = np.ones(n, dtype=bool)
        mask[rem] = False
        try:
            Y = best[mask].copy()  # (n-k, d)
        except Exception:
            continue
        # --- sequential reinsertion into largest holes ---
        ok = True
        for _ii in range(k):
            if time.time() >= EXPLORE_END or time.time() >= HARD_END - 25:
                ok = False
                break
            # sample >=8192 candidates
            NC = 8192
            try:
                C = rng.standard_normal((NC, d))
                C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                dots = C @ Y.T  # (NC, m_cur)
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                v0 = C[bi].copy()
            except Exception:
                v0 = rng.standard_normal(d)
                v0 /= (np.linalg.norm(v0) + 1e-12)
            try:
                v0 = _single_point_opt(v0, Y, steps=80)
            except Exception:
                pass
            try:
                Y = np.vstack([Y, v0.reshape(1, d)])
            except Exception:
                ok = False
                break
        if not ok:
            break
        # Y is now (n, d)
        Xtr = Y
        # --- short all-point soft-max quench ---
        try:
            Xtr = _softmax_quench(Xtr, steps=100, b0=150.0, b1=2500.0, lr0=0.03, lr1=0.004)
        except Exception:
            continue
        try:
            vq = _max_cos(Xtr)
        except Exception:
            continue
        # only LP-polish if promising (saves time, avoids timeouts)
        # threshold: allow LP if within 0.015 of incumbent
        if vq > inc_val + 0.015:
            continue
        # capped-pair LP polish, short budget
        lp_budget = 1.5
        # don't exceed explore end
        remain_explore = EXPLORE_END - time.time()
        if remain_explore < 0.5:
            break
        lp_budget = min(lp_budget, remain_explore - 0.05, HARD_END - time.time() - 22)
        if lp_budget < 0.3:
            continue
        try:
            Xtr2 = _lp_polish(Xtr, seconds=lp_budget, deadline=min(EXPLORE_END, HARD_END - 22),
                              max_iter=8, max_pairs=500, r0=5e-4)
            v2 = _max_cos(Xtr2)
        except Exception:
            continue
        if v2 < inc_val - 1e-12:
            best = Xtr2 / np.linalg.norm(Xtr2, axis=1, keepdims=True)
            inc_val = v2
            best_val = v2
            # keep monotonic; continue searching from new incumbent
        # loop
    # --- reserve final 20s for LP polishing of incumbent best ---
    try:
        now = time.time()
        fb = min(FINAL_BUDGET, HARD_END - now - 2.0)
        if fb > 1.0 and best is not None:
            best = _lp_polish(best, seconds=fb, deadline=HARD_END - 2.0,
                              max_iter=30, max_pairs=700, r0=5e-4)
            best_val = _max_cos(best)
    except Exception:
        pass
    # fallback guarantee
    try:
        if best is None or not np.all(np.isfinite(best)) or best.shape != (58, 22):
            best = np.load("best.npy").astype(float)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        # if our search somehow got worse, return original incumbent
        bv = _max_cos(best)
        iv = _max_cos(incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True))
        if bv > iv + 1e-12:
            best = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
