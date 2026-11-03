import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish_cap(x, seconds, cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        I_all = iu[0][sel]
        J_all = iu[1][sel]
        c_sel_all = c_all[sel]
        if p0 > cap:
            idx = np.argpartition(-c_sel_all, cap)[:cap]
            I = I_all[idx]; J = J_all[idx]; c_sel = c_sel_all[idx]
            p = cap
        else:
            I = I_all; J = J_all; c_sel = c_sel_all
            p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = I[:, None] * d + ar
        cols_J = J[:, None] * d + ar
        cols = np.concatenate([cols_I, cols_J, np.full((p, 1), n * d, dtype=int)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def anneal_adam(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(steps):
        f = it / (steps - 1) if steps > 1 else 1.0
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        # normalize to keep Adam stable
        rms = np.sqrt(np.mean(grad * grad)) + 1e-12
        grad = grad / rms * 0.1
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * grad * grad
        t = it + 1
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = m1h / (np.sqrt(m2h) + eps)
        # tangent projection of step
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def spectral_partition(x, m, k, rng):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    A = (G > m - 1e-6).astype(float)
    # symmetrize (already symmetric)
    deg = A.sum(axis=1)
    # unnormalized Laplacian
    L = np.diag(deg) - A
    try:
        w, V = np.linalg.eigh(L)
    except Exception:
        return rng.integers(0, k, size=n)
    # take k smallest eigenvectors
    U = V[:, :k]
    # row normalize (skip zero rows)
    nr = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    U = U / nr
    # k-means Lloyd, 20 iters
    idx = rng.choice(n, size=k, replace=False)
    C = U[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(20):
        d2 = ((U[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        nl = np.argmin(d2, axis=1)
        if np.array_equal(nl, labels) and _ > 0:
            labels = nl
            break
        labels = nl
        for c in range(k):
            sel = labels == c
            if np.any(sel):
                C[c] = U[sel].mean(axis=0)
            else:
                C[c] = U[rng.integers(0, n)]
    return labels

def solve():
    t0 = time.time()
    HARD = 168.0
    SEARCH_END = t0 + 130.0
    np.random.seed(1)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and getattr(c, "shape", None) == (180, 9):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append(c)
    if not cands:
        r0 = np.random.default_rng(0)
        c = r0.standard_normal((180, 9))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > HARD - 20.0:
            break
        rng = np.random.default_rng(5000 + trial * 977)
        # contact max
        Gtmp = x_best @ x_best.T
        np.fill_diagonal(Gtmp, -2.0)
        m0 = float(Gtmp.max())
        k = int(rng.integers(6, 11))  # 6..10
        try:
            labels = spectral_partition(x_best, m0, k, rng)
        except Exception:
            continue
        # pick a community (prefer medium size)
        uniq, counts = np.unique(labels, return_counts=True)
        # choose random community
        ci = uniq[int(rng.integers(0, len(uniq)))]
        idx_c = np.where(labels == ci)[0]
        if len(idx_c) == 0 or len(idx_c) >= 179:
            continue
        # random 2-plane and angle 0.03-0.25
        a, b = rng.choice(9, size=2, replace=False)
        ang = float(rng.uniform(0.03, 0.25)) * (1.0 if rng.random() < 0.5 else -1.0)
        ca, sa = np.cos(ang), np.sin(ang)
        x_try = x_best.copy()
        # rigid rotation of community in plane (a,b)
        Va = x_try[idx_c, a].copy()
        Vb = x_try[idx_c, b].copy()
        x_try[idx_c, a] = ca * Va - sa * Vb
        x_try[idx_c, b] = sa * Va + ca * Vb
        # optionally counter-rotate neighbouring community
        if rng.random() < 0.5:
            # neighbour = community with most inter edges to idx_c
            A = (Gtmp > m0 - 1e-6).astype(float)
            scores = {}
            for u in uniq:
                if u == ci:
                    continue
                idx_u = np.where(labels == u)[0]
                s = float(A[np.ix_(idx_c, idx_u)].sum())
                scores[u] = s
            if scores:
                u2 = max(scores, key=lambda z: scores[z])
                idx_n = np.where(labels == u2)[0]
                ang2 = -ang * float(rng.uniform(0.3, 0.7))
                c2, s2 = np.cos(ang2), np.sin(ang2)
                Wa = x_try[idx_n, a].copy()
                Wb = x_try[idx_n, b].copy()
                x_try[idx_n, a] = c2 * Wa - s2 * Wb
                x_try[idx_n, b] = s2 * Wa + c2 * Wb
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # annealed LogSumExp Adam, at most 300 steps
        steps = int(rng.integers(150, 301))
        k0 = float(rng.uniform(20.0, 40.0))
        k1 = float(rng.uniform(120.0, 220.0))
        lr0 = float(rng.uniform(0.008, 0.02))
        lr1 = lr0 * 0.08
        try:
            x_sm = anneal_adam(x_try, steps, k0, k1, lr0, lr1)
        except Exception:
            continue
        try:
            v_sm = max_cos(x_sm)
        except Exception:
            continue
        if v_sm > best_val + 0.012:
            continue
        # short capped LP
        remain_search = SEARCH_END - time.time()
        remain_hard = HARD - time.time() - 14.0
        budget = min(2.5, remain_search - 0.5, remain_hard)
        if budget < 0.7:
            if v_sm < best_val:
                x_best, best_val = x_sm, v_sm
            continue
        try:
            x_pol = lp_polish_cap(x_sm, budget, cap=1500)
            v_pol = max_cos(x_pol)
        except Exception:
            continue
        if v_pol < best_val:
            x_best = x_pol
            best_val = v_pol
            # early stop if clearly improved? keep going to use time
        if trial > 120:
            break
    # final polish with remaining time
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 2.0:
        try:
            x_best = lp_polish_cap(x_best, min(rem, 12.0), cap=2000)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
