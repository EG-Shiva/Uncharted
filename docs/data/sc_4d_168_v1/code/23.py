import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X, steps, tau0, tau1, lr0, lr1, mu=0.9):
    V = np.zeros_like(X)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True) + 1e-30
        Grad = (W @ X) / rs
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        V = mu * V + (1.0 - mu) * Gt
        bc = 1.0 - mu ** (s + 1)
        Vn = V / bc
        X = X - lr * Vn
        # project
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _hard_refine(X, iters, step0, step1, tol=0.004):
    best = _max_cos(X)
    Xb = X.copy()
    for it in range(iters):
        f = it / max(1, iters - 1)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        window = tol * (1.0 - f * 0.7)
        Mask = G > (m - window)
        cnt = np.sum(Mask, axis=1, keepdims=True) + 1e-12
        Grad = (Mask.astype(float) @ X) / cnt
        d = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - d * X
        lr = step0 * (step1 / step0) ** f if step0 != step1 else step0
        Y = _norm(X - lr * Gt)
        v = _max_cos(Y)
        if v < m:
            X = Y
            if v < best:
                best = v
                Xb = Y.copy()
        else:
            X = _norm(X - (lr * 0.3) * Gt)
    return Xb

def _refine_single(x0, Xref, steps=100, beta0=50.0, lr0=0.08):
    x = x0 / (np.linalg.norm(x0) + 1e-30)
    beta = beta0
    lr = lr0
    for _ in range(steps):
        d = Xref @ x
        dm = float(np.max(d))
        w = np.exp(beta * (d - dm))
        g = w @ Xref
        g = g - np.dot(g, x) * x
        gn = np.linalg.norm(g)
        if gn > 0:
            g = g / gn
        x = x - lr * g
        n = np.linalg.norm(x)
        if n == 0:
            x = x0 / (np.linalg.norm(x0) + 1e-30)
        else:
            x = x / n
        lr *= 0.995
        beta = min(beta * 1.005, 220.0)
    return x / (np.linalg.norm(x) + 1e-30)

def _greedy_reinsert(Y, victims, rng, n_sample=6000):
    n = Y.shape[0]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    X_keep = Y[keep_mask]
    new_pts = []
    for s in range(len(victims)):
        if len(new_pts) == 0:
            Xref = X_keep
        else:
            Xref = np.vstack([X_keep, np.array(new_pts)])
        C = rng.randn(n_sample, 4)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xref.T
        mc = np.max(D, axis=1)
        bi = int(np.argmin(mc))
        x_new = _refine_single(C[bi], Xref, steps=100, beta0=50.0, lr0=0.08)
        new_pts.append(x_new)
    Y2 = Y.copy()
    Y2[victims] = np.array(new_pts)
    return Y2

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    iu0_0 = iu0[0]
    iu0_1 = iu0[1]
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        iu_vals = Gfull[iu0_0, iu0_1]
        m = float(np.max(iu_vals))
        sel = iu_vals > m - max(20 * radius, 1e-9)
        if np.sum(sel) < 1:
            break
        I = iu0_0[sel]
        J = iu0_1[sel]
        c = iu_vals[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if _max_cos(x) > best + 1e-12:
            x = xb.copy()
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    BUDGET = 175.0
    t_end_global = t0 + BUDGET - 1.0
    SEARCH_END = t0 + BUDGET - 45.0
    rng = np.random.RandomState(42)
    n = 168
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = rng.randn(n, 4)
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = Xb.copy()
    Xb = _norm(Xb)
    Xr = _norm(Xr)
    vb = _max_cos(Xb)
    vr = _max_cos(Xr)
    if vr < vb:
        Xb = Xr.copy()
        vb = vr
    best_val = vb
    best_X = Xb.copy()
    x_cur = best_X.copy()
    v_cur = best_val
    pool = [(best_val, best_X.copy())]

    def push_pool(v, X):
        pool.append((v, X.copy()))
        pool.sort(key=lambda z: z[0])
        # keep top 5
        del pool[5:]

    trial = 0
    T0, T1 = 0.0015, 0.00002
    while time.time() < SEARCH_END and time.time() < t_end_global - 40.0:
        trial += 1
        elapsed = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        elapsed = min(max(elapsed, 0.0), 1.0)
        Temp = T0 * (T1 / T0) ** elapsed
        mode = trial % 8
        Y = x_cur.copy()
        # default smooth params
        if mode == 0:
            Y = _norm(Y + 0.015 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 600, 0.012, 0.002, 0.06, 0.008, 0.9
            nsamp = 0
        elif mode == 1:
            Y = _norm(Y + 0.065 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 900, 0.018, 0.0015, 0.10, 0.008, 0.9
            nsamp = 0
        elif mode == 2:
            Y = _norm(Y + 0.18 * rng.randn(n, 4))
            k = 4
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = _norm(rng.randn(k, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1200, 0.025, 0.001, 0.12, 0.006, 0.88
            nsamp = 0
        elif mode == 3:
            G = Y @ Y.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.004), axis=1)
            k = 3 + (trial % 3)  # 3..5
            victims = np.argsort(-deg)[:k]
            Y = _greedy_reinsert(Y, victims, rng, n_sample=6000)
            Y = _norm(Y + 0.015 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1100, 0.02, 0.001, 0.10, 0.005, 0.9
            nsamp = 1
        elif mode == 4:
            k = 5 + (trial % 3)
            victims = rng.choice(n, size=k, replace=False)
            Y = _greedy_reinsert(Y, victims, rng, n_sample=6000)
            Y = _norm(Y + 0.015 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1100, 0.02, 0.001, 0.10, 0.005, 0.9
            nsamp = 1
        elif mode == 5:
            G = Y @ Y.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > (m - 0.004), axis=1)
            idx_lo = np.argsort(deg)[:2]
            idx_hi = np.argsort(-deg)[:3]
            victims = np.unique(np.concatenate([idx_lo, idx_hi]))
            Y = _greedy_reinsert(Y, victims, rng, n_sample=6000)
            Y = _norm(Y + 0.012 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1100, 0.016, 0.0008, 0.09, 0.005, 0.9
            nsamp = 1
        elif mode == 6:
            k = 10 + (trial % 5)
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = _norm(rng.randn(k, 4))
            Y = _norm(Y + 0.03 * rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1300, 0.025, 0.001, 0.12, 0.006, 0.88
            nsamp = 0
        else:
            Y = _norm(rng.randn(n, 4))
            steps, tau0, tau1, lr0, lr1, mu = 1500, 0.03, 0.0012, 0.13, 0.006, 0.88
            nsamp = 0

        remain_search = SEARCH_END - time.time()
        if remain_search < 8:
            steps = min(steps, 350)
        Y = _smooth_opt(Y, steps, tau0, tau1, lr0, lr1, mu=mu)
        Y = _hard_refine(Y, 100, 0.006, 0.0008, tol=0.004)
        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best_val = v
            best_X = Y.copy()
            push_pool(v, Y)
            x_cur = Y.copy()
            v_cur = v
        else:
            if v < len(pool) * 0 + best_val + 0.004:
                # keep near-best for diversity (limit size)
                if len(pool) < 5:
                    push_pool(v, Y)
            # Metropolis move from current
            if v <= v_cur:
                x_cur = Y.copy()
                v_cur = v
            else:
                if Temp > 0 and rng.rand() < np.exp((v_cur - v) / Temp):
                    x_cur = Y.copy()
                    v_cur = v
                # occasional restart to best if stuck very high
                if v_cur > best_val + 0.008 and rng.rand() < 0.15:
                    x_cur = best_X.copy()
                    v_cur = best_val
        if time.time() > SEARCH_END - 1.0:
            break

    pool = sorted(pool, key=lambda z: z[0])[:3]
    total_lp = max(5.0, t_end_global - time.time() - 1.0)
    if total_lp >= 5.0:
        shares = np.array([0.6, 0.25, 0.15][:len(pool)], dtype=float)
        shares = shares / np.sum(shares) * total_lp
        for (v0, X0), sec in zip(list(pool), shares):
            if time.time() >= t_end_global - 0.5:
                break
            sec = min(float(sec), t_end_global - time.time() - 0.5)
            if sec < 1.0:
                continue
            Xp = _lp_polish(X0, time.time() + sec)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp
                best_X = Xp.copy()
    if time.time() < t_end_global - 2.0:
        Y = _smooth_opt(best_X.copy(), 300, 0.004, 0.0006, 0.02, 0.003, mu=0.9)
        Y = _hard_refine(Y, 60, 0.003, 0.0005, tol=0.003)
        v = _max_cos(Y)
        if v < best_val:
            best_val = v
            best_X = Y.copy()
    return _norm(best_X)
