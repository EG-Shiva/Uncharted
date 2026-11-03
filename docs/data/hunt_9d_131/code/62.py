import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(X0, steps, a0=20.0, a1=400.0, lr0=0.02, lr1=0.003, deadline=None):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu0 = np.triu_indices(n, 1)
    for k in range(steps):
        if deadline is not None and (k % 50 == 0) and time.time() > deadline:
            break
        frac = k / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        # weights exp(alpha*(c-m))
        E = np.exp(alpha * (c - m))
        S = float(E.sum()) + 1e-300
        # build grad: Wfull @ X / S
        Wf = np.zeros((n, n))
        Wf[iu0] = E
        Wf = Wf + Wf.T
        grad = (Wf @ X) / S
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        t = k + 1
        m1 = b1 * m1 + (1 - b1) * grad
        v1 = b2 * v1 + (1 - b2) * (grad * grad)
        mh = m1 / (1 - b1 ** t)
        vh = v1 / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _soft_steps(X0, steps=400, thresh=0.42245, lam=0.008, alpha=120.0,
                lr0=0.012, lr1=0.002, deadline=None):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0
    for k in range(steps):
        if deadline is not None and (k % 50 == 0) and time.time() > deadline:
            break
        frac = k / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        c = G[I0, J0]
        m = float(c.max())
        # hinge weights
        ex = c - thresh
        ex = np.where(ex > 0, ex, 0.0)
        H = 2.0 * ex  # d/dc (c-th)^2
        # weak lse weights
        E = np.exp(alpha * (c - m))
        S = float(E.sum()) + 1e-300
        W = E / S
        # combined pair weight: H + lam*W*scale? scale to match magnitudes:
        # H ~ O(0.02), W sum=1 so per-pair ~1/P; multiply lam accordingly already small
        # Use combined = H + lam*W*P? No, keep lam*W normalized form via separate grads:
        # grad = Hfull@X + lam*(Wfull@X)
        # Hfull built from H, Wfull from W
        # To avoid building two matrices, combine coeff C = H + lam*W
        C = H + lam * W
        Cf = np.zeros((n, n))
        Cf[I0, J0] = C
        Cf = Cf + Cf.T
        grad = Cf @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - dot * X
        t = k + 1
        m1 = b1 * m1 + (1 - b1) * grad
        v1 = b2 * v1 + (1 - b2) * (grad * grad)
        mh = m1 / (1 - b1 ** t)
        vh = v1 / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _refine_holes(Zcand, Xk, steps=150, T=0.008, lr=0.02):
    # Zcand: (K,d), refine each against Xk, return refined + values
    K = Zcand.shape[0]
    Z = Zcand.copy()
    for it in range(steps):
        D = Z @ Xk.T  # K x m
        m = D.max(axis=1, keepdims=True)
        W = np.exp((D - m) / T)
        W = W / (W.sum(axis=1, keepdims=True) + 1e-12)
        Gv = W @ Xk  # K x d
        dot = np.sum(Gv * Z, axis=1, keepdims=True)
        Gt = Gv - dot * Z
        st = lr * (1.0 - 0.5 * it / max(steps, 1))
        Z = Z - st * Gt
        Z = _norm(Z)
        if it == 79:
            T *= 0.7
    vals = (Z @ Xk.T).max(axis=1)
    return Z, vals

def _lp_polish(x, seconds, deadline=None, r0=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        Gfull = x @ x.T
        np.fill_diagonal(Gfull, -2.0)
        m = float(Gfull.max())
        cfull = Gfull[iu]
        thresh = m - max(20 * radius, 1e-9)
        sel = cfull > thresh
        if not np.any(sel):
            radius *= 0.5
            continue
        I = iu[0][sel]; J = iu[1][sel]; csel = cfull[sel]
        p = len(I)
        if p > 2500:
            idx = np.argpartition(csel, p - 2500)[p - 2500:]
            I = I[idx]; J = J[idx]; csel = csel[idx]; p = 2500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    LP_RESERVE = 30.0
    GROW_DEADLINE = HARD - LP_RESERVE - 1.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = xb
    xb = _norm(xb); xr = _norm(xr)
    # pick better as reference for pruning
    if _max_cos(xr) < _max_cos(xb):
        x0 = xr
    else:
        x0 = xb
    n, d = x0.shape  # 131,9
    # contact analysis
    G = x0 @ x0.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    C = G > (m0 - 1e-9)
    cnt = C.sum(axis=1).astype(int)
    per_max = G.max(axis=1)
    ratt = np.where(cnt == 0)[0].tolist()
    # ensure 9 rattlers to remove: if fewer, add lowest-count points
    if len(ratt) < 9:
        order_cnt = np.argsort(cnt, kind="stable")
        for ii in order_cnt:
            if ii not in ratt:
                ratt.append(int(ii))
            if len(ratt) >= 9:
                break
    else:
        ratt = ratt[:9]
    # 3-13 highest per-point max / degree among remainder
    rem = [i for i in range(n) if i not in set(ratt)]
    # sort remainder by per_max desc, then cnt desc
    rem_sorted = sorted(rem, key=lambda i: (float(per_max[i]), int(cnt[i])), reverse=True)
    k_remove = 7  # gives N=115, within 109-122
    extra = rem_sorted[:k_remove]
    drop = set(ratt) | set(extra)
    keep = [i for i in range(n) if i not in drop]
    Xcur = _norm(x0[keep].copy())
    # hard optimise smaller code with annealed LSE Adam 20->400
    # choose steps based on remaining time (3000-6000)
    remain_grow = GROW_DEADLINE - time.time()
    if remain_grow < 20:
        steps_core = 1500
    else:
        steps_core = 4000
    Xcur = _adam_lse(Xcur, steps=steps_core, a0=20.0, a1=400.0,
                     lr0=0.02, lr1=0.003, deadline=GROW_DEADLINE)
    # greedy grow back to 131
    THRESH = 0.42245
    target = 131
    # precompute random pool reuse? generate fresh each size for diversity (deterministic via rng)
    while Xcur.shape[0] < target and time.time() < GROW_DEADLINE - 2.0:
        m_cur = Xcur.shape[0]
        # generate 20000 candidates
        Z = rng.normal(size=(20000, d))
        Z = _norm(Z)
        Dmax = (Z @ Xcur.T).max(axis=1)
        idx30 = np.argpartition(Dmax, 30)[:30]
        # sort those 30 by Dmax
        o = np.argsort(Dmax[idx30])
        idx30 = idx30[o]
        Z30 = Z[idx30]
        Zref, vref = _refine_holes(Z30, Xcur, steps=150, T=0.008, lr=0.02)
        order = np.argsort(vref, kind="stable")
        Zref = Zref[order]; vref = vref[order]
        # try insertions with backtrack on >0.43
        inserted = False
        failed = []
        # try up to 6 distinct holes
        tried = 0
        pi = 0
        best_fallback = None
        best_fallback_val = 1e9
        while pi < len(Zref) and tried < 6:
            ztry = Zref[pi]
            # distinctness vs failed
            skip = False
            for f in failed:
                if float(ztry @ f) > 0.9:
                    skip = True
                    break
            if skip:
                pi += 1
                continue
            tried += 1
            Xtry = np.vstack([Xcur, ztry[None, :]])
            Xtry = _soft_steps(Xtry, steps=400, thresh=THRESH, lam=0.008,
                               alpha=120.0, lr0=0.012, lr1=0.002,
                               deadline=GROW_DEADLINE)
            v = _max_cos(Xtry)
            if v < best_fallback_val:
                best_fallback_val = v
                best_fallback = Xtry
            if v <= 0.43:
                Xcur = Xtry
                inserted = True
                break
            else:
                failed.append(ztry.copy())
                pi += 1
        if not inserted:
            # take best fallback (least bad) to keep cardinality progress
            if best_fallback is not None:
                Xcur = best_fallback
            else:
                Xcur = np.vstack([Xcur, Zref[0][None, :]])
        # quick check: if time very low, reduce remaining soft cost by continuing
    # ensure exactly 131 (trim/pad if needed due to time exit)
    if Xcur.shape[0] > 131:
        Xcur = Xcur[:131]
    elif Xcur.shape[0] < 131:
        need = 131 - Xcur.shape[0]
        Zp = rng.normal(size=(need * 200, d))
        Zp = _norm(Zp)
        # greedy pick far points
        for _ in range(need):
            Dm = (Zp @ Xcur.T).max(axis=1)
            bi = int(np.argmin(Dm))
            Xcur = np.vstack([Xcur, Zp[bi][None, :]])
    Xcur = _norm(Xcur)
    # final LP polish with remaining time (reserve 30s)
    remain = HARD - time.time() - 0.5
    if remain > 2:
        Xcur = _lp_polish(Xcur, seconds=remain, deadline=HARD - 0.3, r0=8e-4)
    Xcur = _norm(Xcur)
    return Xcur
