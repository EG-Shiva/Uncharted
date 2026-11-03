import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _renorm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _adam_lse(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        W = np.exp(k * (G - mval))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _quantile_crush(x, steps=250, Q=600, kk=220.0, lr0=0.02, lr1=0.003, deadline=None):
    import time as _t
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    for s in range(steps):
        if deadline is not None and _t.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        vals = G[iu0, iu1]
        mval = float(vals.max())
        if len(vals) > Q:
            thr = float(np.partition(vals, len(vals) - Q)[len(vals) - Q])
        else:
            thr = float(vals.min())
        # weights: exp(k*(G-m)) but zero below thr
        W = np.exp(kk * (G - mval))
        W[G < thr - 1e-15] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish(x, deadline):
    import time as _t
    x = _renorm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while _t.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(-cmat[sel], 2500 - 1)[:2500]
            I, J = I[idx], J[idx]
            p = 2500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel] if p != 2500 else -((x @ x.T)[iu][sel][idx] if False else (x @ x.T)[iu[0][sel][idx], iu[1][sel][idx]]),
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _renorm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _align_procrustes(A, B, iters=4):
    n, d = A.shape
    R = np.eye(d)
    perm = np.arange(n)
    Bp = B.copy()
    for _ in range(iters):
        BR = B @ R
        C = A @ BR.T
        r, c = linear_sum_assignment(-C)
        p = np.empty(n, dtype=int)
        p[r] = c
        perm = p
        Bp = B[perm]
        M = Bp.T @ A
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
        except Exception:
            break
    # final matching with final R
    BR = B @ R
    C = A @ BR.T
    r, c = linear_sum_assignment(-C)
    p = np.empty(n, dtype=int)
    p[r] = c
    perm = p
    Bp = B[perm]
    M = Bp.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
    except Exception:
        pass
    return Bp @ R, perm, R

def _slerp_child(A, Ba, alpha, rng, jitter=0.03):
    n, d = A.shape
    X = np.empty_like(A)
    for i in range(n):
        a = A[i]; b = Ba[i]
        dot = float(np.clip(a @ b, -1.0, 1.0))
        if dot > 0.9999999:
            v = (1 - alpha) * a + alpha * b
        elif dot < -0.9999999:
            v = (1 - alpha) * a + alpha * b
        else:
            om = float(np.arccos(dot))
            so = float(np.sin(om))
            if abs(so) < 1e-12:
                v = (1 - alpha) * a + alpha * b
            else:
                w1 = float(np.sin((1 - alpha) * om)) / so
                w2 = float(np.sin(alpha * om)) / so
                v = w1 * a + w2 * b
        if jitter > 0:
            v = v + jitter * rng.standard_normal(d)
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            v = a.copy()
            nv = np.linalg.norm(v)
        X[i] = v / nv
    return X

def solve():
    import time as _t
    T0 = _t.time()
    BUDGET = 178.0
    RECOMB_END = T0 + 158.0
    rng = np.random.default_rng(12345)
    incumbent = None; inc_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = _renorm(a)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < inc_val:
                    inc_val = v; incumbent = a.copy()
        except Exception:
            pass
    if incumbent is None:
        incumbent = rng.standard_normal(size=(94, 13))
        incumbent = _renorm(incumbent)
        inc_val = _max_cos(incumbent)
    global_best = incumbent.copy()
    global_val = float(inc_val)
    # --- pool generation ---
    sigmas = [0.15, 0.15, 0.3, 0.3, 0.5, 0.5, 0.8, 0.8, 0.15, 0.3]
    pool = []
    pool_vals = []
    # include incumbent as parent 0 (unpolished)
    pool.append(incumbent.copy()); pool_vals.append(float(inc_val))
    for si, sig in enumerate(sigmas):
        if _t.time() > T0 + 60:
            break
        y = incumbent + sig * rng.standard_normal(size=incumbent.shape)
        y = _renorm(y)
        y = _adam_lse(y, steps=400, k0=30, k1=300, lr0=0.02, lr1=0.003)
        v = _max_cos(y)
        pool.append(y); pool_vals.append(float(v))
        if v < global_val - 1e-15:
            global_val = float(v); global_best = y.copy()
    for _r in range(2):
        if _t.time() > T0 + 70:
            break
        y = rng.standard_normal(size=incumbent.shape)
        y = _renorm(y)
        y = _adam_lse(y, steps=400, k0=30, k1=300, lr0=0.02, lr1=0.003)
        v = _max_cos(y)
        pool.append(y); pool_vals.append(float(v))
        if v < global_val - 1e-15:
            global_val = float(v); global_best = y.copy()
    # retain best few as elites (e.g., 6) but never discard incumbent for final return
    E = min(6, len(pool))
    order = np.argsort(np.array(pool_vals))
    elites = [pool[i].copy() for i in order[:E]]
    elite_vals = [float(pool_vals[i]) for i in order[:E]]
    # ensure diversity: if elites all same basin, keep one random far parent
    # (keep worst random if far)
    alphas = [0.3, 0.5, 0.7]
    # precompute pair list
    import itertools
    pair_cycle = list(itertools.combinations(range(E), 2))
    if len(pair_cycle) == 0:
        pair_cycle = [(0, 0)]
    rng.shuffle(pair_cycle)
    pi = 0
    ai = 0
    # recombination loop
    while _t.time() < RECOMB_END:
        i, j = pair_cycle[pi % len(pair_cycle)]
        pi += 1
        alpha = alphas[ai % len(alphas)]
        ai += 1
        if i >= len(elites) or j >= len(elites):
            continue
        A = elites[i]; B = elites[j]
        worse = max(elite_vals[i], elite_vals[j])
        try:
            Ba, _, _ = _align_procrustes(A, B, iters=3)
        except Exception:
            continue
        try:
            child = _slerp_child(A, Ba, alpha, rng, jitter=0.03)
        except Exception:
            continue
        # quantile-crush polish
        crush_dead = min(RECOMB_END, _t.time() + 3.0)
        child = _quantile_crush(child, steps=220, Q=600, kk=220.0, lr0=0.018, lr1=0.003, deadline=crush_dead)
        if not np.all(np.isfinite(child)):
            continue
        cv = _max_cos(child)
        if cv >= worse - 1e-12:
            # not better than worse parent -> discard (occasionally keep if global best? no)
            continue
        # short LP polish (at most ~1.5s)
        lp_dead = min(RECOMB_END + 2.0, _t.time() + 1.5)
        # only LP if promising or time allows
        try:
            child2, cv2 = _lp_polish(child, lp_dead)
        except Exception:
            child2, cv2 = child, cv
        # final short high-k tighten
        if _t.time() < RECOMB_END:
            try:
                child2 = _adam_lse(child2, steps=120, k0=150, k1=600, lr0=0.008, lr1=0.002)
                cv2 = _max_cos(child2)
            except Exception:
                pass
        if cv2 < worse - 1e-15:
            # replace worse elite
            worst_idx = int(np.argmax(np.array(elite_vals)))
            # only replace if better than worst elite
            if cv2 < elite_vals[worst_idx] - 1e-15:
                elites[worst_idx] = child2.copy()
                elite_vals[worst_idx] = float(cv2)
            else:
                # replace one of the parents (the worse of the two)
                rep = i if elite_vals[i] >= elite_vals[j] else j
                elites[rep] = child2.copy()
                elite_vals[rep] = float(cv2)
            if cv2 < global_val - 1e-15:
                global_val = float(cv2); global_best = child2.copy()
        if _t.time() > T0 + BUDGET - 18:
            break
        # refresh pair cycle occasionally to include new elites
        if pi % len(pair_cycle) == 0:
            # re-sort elites to keep best E (already size E) and reshuffle pairs
            o = np.argsort(np.array(elite_vals))
            elites = [elites[k] for k in o]
            elite_vals = [elite_vals[k] for k in o]
            pair_cycle = list(itertools.combinations(range(len(elites)), 2))
            rng.shuffle(pair_cycle)
    # final polish of global best with remaining time
    try:
        if _t.time() < T0 + BUDGET - 2:
            global_best = _quantile_crush(global_best, steps=200, Q=600, kk=260.0, lr0=0.012, lr1=0.002, deadline=T0 + BUDGET - 8)
            gv = _max_cos(global_best)
            if gv < global_val:
                global_val = gv
            else:
                pass
    except Exception:
        pass
    try:
        if _t.time() < T0 + BUDGET - 1:
            gb2, gv2 = _lp_polish(global_best, T0 + BUDGET - 0.5)
            if gv2 < global_val:
                global_best, global_val = gb2, gv2
    except Exception:
        pass
    # never discard original incumbent
    if global_val > inc_val:
        global_best = incumbent.copy()
    global_best = _renorm(global_best)
    return global_best
