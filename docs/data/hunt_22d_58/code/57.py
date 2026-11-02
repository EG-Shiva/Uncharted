import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _refine_single(x0, Y, steps=120, b0=60.0, b1=600.0, lr0=0.25, lr1=0.02):
    x = x0 / (np.linalg.norm(x0) + 1e-16)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ x
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = float(w.sum()) + 1e-16
        w /= sw
        g = w @ Y
        gt = g - float(g @ x) * x
        nrm = float(np.linalg.norm(gt)) + 1e-12
        # normalized tangent step
        x = x - (lr / max(1.0, nrm * 5.0)) * gt
        # alternative stable: fixed lr tangent
        # x = x - lr * gt
        n = float(np.linalg.norm(x))
        if not np.isfinite(n) or n < 1e-12:
            break
        x = x / n
    return x

def _quench_short(X, steps, a0=120.0, a1=1200.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt, axis=1).mean()) + 1e-12
        X = X - (lr / gn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _lp_capped(X, deadline, max_iter=4, cap=550, radius=7e-4):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_iter and time.time() < deadline:
        it += 1
        c = (X @ X.T)[iu]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        # keep top-cap pairs within window
        order = np.argsort(-c)
        sel_idx = []
        for idx in order[:cap + 200]:
            if c[idx] > m - window and len(sel_idx) < cap:
                sel_idx.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_idx) >= cap:
                break
        if len(sel_idx) == 0:
            break
        sel = np.array(sel_idx)
        I = iu[0][sel]
        J = iu[1][sel]
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
                          method="highs", options={"time_limit": 1.2})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-13:
                break
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-13:
                break
    return X

def _lp_final(X, seconds, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 850:
            idx = np.argpartition(-c, 850)[:850]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 150.0
    FINAL_RESERVE = 20.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)
    incumbent = best.copy()
    incumbent_val = float(best_val)

    ks = [3, 4, 5, 6, 8]
    trial = 0
    NCAND = 8192
    while time.time() < HOP_END and time.time() < T_END - FINAL_RESERVE - 2.0:
        # stop if very little time for a trial
        if time.time() > HOP_END - 1.0:
            break
        trial += 1
        k = ks[trial % len(ks)]
        mode = trial % 3  # 0 random, 1 hubs, 2 cluster
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
        except Exception:
            continue
        victims = None
        if mode == 0:
            victims = rng.choice(n, size=k, replace=False)
        elif mode == 1:
            cnt = (G0 > m0 - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            # mix: sometimes top-k, sometimes random among top-12
            if (trial // 3) % 2 == 0:
                victims = order[:k]
            else:
                pool = order[:min(12, n)]
                victims = rng.choice(pool, size=k, replace=False)
        else:
            seed = int(rng.integers(0, n))
            row = G0[seed].copy()
            row[seed] = -2.0
            nn = np.argsort(-row, kind="stable")[:k - 1]
            victims = np.concatenate(([seed], nn)).astype(int)
        victims = np.array([int(v) for v in victims], dtype=int)
        # random reinsertion order
        order_ins = victims.copy()
        rng.shuffle(order_ins)
        y = best.copy()
        # sequential reinsertion
        ok = True
        for vidx in order_ins:
            if time.time() >= HOP_END:
                ok = False
                break
            mask = np.ones(n, dtype=bool)
            mask[int(vidx)] = False
            Y = y[mask]
            try:
                C = rng.standard_normal((NCAND, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-16
                dots = C @ Y.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                cand = C[bi]
            except Exception:
                ok = False
                break
            try:
                cand = _refine_single(cand, Y, steps=100)
            except Exception:
                pass
            y[int(vidx)] = cand / (np.linalg.norm(cand) + 1e-16)
        if not ok:
            continue
        # short all-point soft-max quench
        try:
            # vary quench slightly by trial
            if trial % 2 == 0:
                y = _quench_short(y, 220, 120.0, 1200.0, 0.02, 0.002)
            else:
                y = _quench_short(y, 180, 200.0, 2000.0, 0.015, 0.0015)
        except Exception:
            continue
        try:
            vq = _max_cos(y)
        except Exception:
            continue
        # only LP-polish promising ones to save time
        if vq < best_val + 0.003:
            try:
                dl = min(HOP_END, T_END - FINAL_RESERVE - 1.0)
                if time.time() < dl - 0.5:
                    y = _lp_capped(y, dl, max_iter=3, cap=550, radius=7e-4)
            except Exception:
                pass
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best = y / np.linalg.norm(y, axis=1, keepdims=True)
            best_val = float(v)
        # safety: never exceed hop deadline
        if time.time() >= HOP_END:
            break

    # fallback guarantee
    if best is None or not np.all(np.isfinite(best)):
        best = incumbent.copy()
        best_val = incumbent_val
    # if no improvement, keep incumbent (already best)
    # final LP polish reserve ~20s
    try:
        remain = T_END - time.time() - 1.0
        budget = min(FINAL_RESERVE, max(0.0, remain))
        if budget > 2.0:
            best = _lp_final(best, budget, T_END - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    for i in range(n):
        nr = float(np.linalg.norm(best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v) + 1e-16
            best[i] = v
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
