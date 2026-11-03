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

def _smooth_refine(X, steps, T0, T1, lr0, lr1, momentum=0.8):
    X = X.copy()
    V = np.zeros_like(X)
    n = X.shape[0]
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * (1 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # exp weights, max weight =1
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        rs = np.sum(W, axis=1, keepdims=True) + 1e-12
        grad = grad / rs
        V = momentum * V + (1.0 - momentum) * grad
        X = X - lr * V
        X = _norm(X)
    return X

def _find_hole(Xk, rng, n_samples=15000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])

def _refine_hole(z, Xk, steps=80, T=0.008, lr=0.05):
    z = z.copy()
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

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
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
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)

    # time budgets: main loop until t0+130, final polish rest
    loop_end = t0 + 130.0
    final_budget = 35.0
    it = 0
    # precompute helper for hub probs
    while time.time() < loop_end:
        it += 1
        # degrees
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        deg_strict = np.sum(G >= best_val - 1e-7, axis=1)
        rattlers = np.where(deg_strict == 0)[0]
        # dense degree for hub selection
        deg_dense = np.sum(G >= best_val - 0.005, axis=1).astype(float)
        probs = (deg_dense + 1.0) ** 2.0
        probs = probs / probs.sum()
        cycle = it % 4
        remove = None
        if cycle == 0:
            # all rattlers + 3 hubs
            h = 3
            order = np.argsort(-deg_dense)
            hubs = [o for o in order if o not in set(rattlers.tolist())][:h]
            remove = list(rattlers.tolist()) + hubs
            if len(remove) == 0:
                remove = order[:6].tolist()
        elif cycle == 1:
            # 3 random rattlers + 4 hubs sampled
            hubs = []
            if len(rattlers) > 0:
                k_r = min(3, len(rattlers))
                sel_r = rng.choice(rattlers, k_r, replace=False).tolist()
            else:
                sel_r = []
            need = 7 - len(sel_r)
            # sample hubs by probs excluding already chosen
            avail = np.array([i for i in range(n) if i not in set(sel_r)])
            pw = probs[avail]; pw = pw / pw.sum()
            sel_h = rng.choice(avail, min(need, len(avail)), replace=False, p=pw).tolist()
            remove = sel_r + sel_h
        elif cycle == 2:
            # 6 hubs only
            avail = np.arange(n)
            remove = rng.choice(avail, 6, replace=False, p=probs).tolist()
        else:
            # global shake without ruin
            Y = best + 0.015 * rng.standard_normal((n, d))
            Y = _norm(Y)
            Y = _smooth_refine(Y, steps=350, T0=0.012, T1=0.0015, lr0=0.25, lr1=0.03, momentum=0.82)
            v = _max_cos(Y)
            if v < best_val + 0.002:
                # short squeeze
                Y2 = _lp_polish(Y, 1.2)
                v2 = _max_cos(Y2)
                if v2 < best_val:
                    best, best_val = Y2, v2
            continue

        remove = sorted(set(remove))
        k = len(remove)
        if k == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[remove] = False
        keep = best[mask]
        cur = keep.copy()
        # sequential hole filling
        for _ in range(k):
            if time.time() >= loop_end:
                break
            z, _ = _find_hole(cur, rng, n_samples=12000)
            z = _refine_hole(z, cur, steps=70, T=0.008, lr=0.05)
            cur = np.vstack([cur, z])
        if cur.shape[0] != n:
            continue
        # annealed smoothing, vary params by iteration
        if cycle == 0:
            cur = _smooth_refine(cur, steps=450, T0=0.015, T1=0.0012, lr0=0.30, lr1=0.03, momentum=0.80)
        elif cycle == 1:
            cur = _smooth_refine(cur, steps=400, T0=0.010, T1=0.0015, lr0=0.25, lr1=0.03, momentum=0.82)
        else:
            cur = _smooth_refine(cur, steps=350, T0=0.008, T1=0.0010, lr0=0.22, lr1=0.025, momentum=0.85)
        v = _max_cos(cur)
        if v < best_val + 0.002 and time.time() < loop_end - 2:
            cur2 = _lp_polish(cur, 1.5)
            v2 = _max_cos(cur2)
            if v2 < best_val:
                best, best_val = cur2, v2
        elif v < best_val:
            best, best_val = cur, v

    # final squeeze of best
    remain = max(1.0, (t0 + 175.0) - time.time())
    remain = min(remain, final_budget)
    try:
        best = _lp_polish(best, remain)
    except Exception:
        pass
    best = _norm(best)
    return best
