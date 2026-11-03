import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_refine(X, steps, T0, T1, lr0, lr1, mom=0.85, deadline=None):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        if deadline is not None and (k % 200 == 0) and time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        S = W @ X
        dot = np.sum(S * X, axis=1, keepdims=True)
        Gr = S - dot * X
        scale = float(np.sum(W)) / n + 1e-12
        Gr = Gr / scale
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X = _norm(X)
    return X

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _degrees(X, mv, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - tol)
    deg = C.sum(axis=1).astype(float)
    return deg, G

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=200, T=0.007, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        s = float(W.sum()) + 1e-12
        W = W / s
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _recenter_rattlers(X, mv, rng, tol=1e-6):
    n = X.shape[0]
    rm, _ = _row_max(X)
    ratt = np.where(rm < mv - tol)[0]
    for i in ratt:
        Xk = np.delete(X, i, axis=0)
        z, _ = _find_hole(Xk, rng, n_samples=8000)
        z = _refine_hole(z, Xk, steps=150, T=0.008, lr=0.03)
        # keep only if it does not worsen global max
        trial = X.copy()
        trial[i] = z
        v = _max_cos(trial)
        if v <= mv + 1e-12:
            X = trial
    return X

def _lp_polish(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    LP_RESERVE = 32.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best = _norm(best)
        best_val = _max_cos(best)

    def probs_from(X, mv):
        deg, _ = _degrees(X, mv, tol=1e-7)
        w = (deg + 1.0) ** 1.5
        return w / w.sum(), deg

    probs, _ = probs_from(best, best_val)

    it = 0
    # exploration deadline
    EXPLORE_END = HARD_END - LP_RESERVE
    while time.time() < EXPLORE_END:
        it += 1
        if it % 8 == 1:
            probs, _ = probs_from(best, best_val)
        Y = best.copy()
        cyc = it % 6
        if cyc == 1:
            Y = Y + 0.015 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=800, T0=0.010, T1=0.002, lr0=0.030, lr1=0.010, deadline=EXPLORE_END)
        elif cyc == 2:
            Y = Y + 0.035 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=1100, T0=0.014, T1=0.002, lr0=0.035, lr1=0.010, deadline=EXPLORE_END)
        elif cyc == 3:
            k = 2
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.20 * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=1300, T0=0.016, T1=0.002, lr0=0.035, lr1=0.010, deadline=EXPLORE_END)
        elif cyc == 4:
            # worst-point hole reinsertion (1 point)
            rm, _ = _row_max(Y)
            worst = int(np.argmax(rm))
            Xk = np.delete(Y, worst, axis=0)
            z, _ = _find_hole(Xk, rng, n_samples=15000)
            z = _refine_hole(z, Xk, steps=200, T=0.007, lr=0.03)
            Y[worst] = z
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=1300, T0=0.012, T1=0.0018, lr0=0.030, lr1=0.008, deadline=EXPLORE_END)
        elif cyc == 5:
            # two worst -> two holes + recenter rattlers occasionally
            rm, _ = _row_max(Y)
            order = np.argsort(-rm)[:2]
            kept = np.delete(Y, order, axis=0)
            for ii, widx in enumerate(order):
                z, _ = _find_hole(kept, rng, n_samples=12000)
                z = _refine_hole(z, kept, steps=180, T=0.007, lr=0.03)
                kept = np.vstack([kept, z.reshape(1, -1)])
                Y[widx] = z
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=1400, T0=0.014, T1=0.002, lr0=0.032, lr1=0.009, deadline=EXPLORE_END)
        else:
            # cyc 0: strong hub kick + global shake, high T to change topology
            k = 1
            idx = rng.choice(n, k, replace=False, p=probs)
            Y[idx] += 0.50 * rng.standard_normal((k, d))
            Y += 0.010 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=1600, T0=0.022, T1=0.002, lr0=0.040, lr1=0.010, deadline=EXPLORE_END)
            # opportunistically recenter rattlers on this branch
            v_tmp = _max_cos(Y)
            if v_tmp < best_val + 0.002:
                Y = _recenter_rattlers(Y, v_tmp, rng, tol=5e-4)

        v = _max_cos(Y)
        if v < best_val - 1e-12:
            best = Y.copy()
            best_val = v
            probs, _ = probs_from(best, best_val)
        # early exit if big win? continue to squeeze
        if time.time() > EXPLORE_END:
            break

    # final squeeze: short smooth at low T then LP
    try:
        best = _smooth_refine(best, steps=600, T0=0.005, T1=0.0015, lr0=0.020, lr1=0.008, deadline=HARD_END - LP_RESERVE + 5.0)
    except Exception:
        pass
    try:
        best = _lp_polish(best, HARD_END - 1.0)
    except Exception:
        pass
    best = _norm(best)
    return best
