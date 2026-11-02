import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_global=None, global_budget=175.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_global is not None and (time.time() - t_global) > global_budget:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_adam(X, steps, beta0, beta1, lr, rng=None):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        D = G - gm
        np.fill_diagonal(D, -2.0 - gm if False else -5.0)
        # diag already -2 above; ensure very negative
        # recompute: G diag -2 => D diag -2-gm ~ -2; exp tiny, ok
        # fix: set diag to -5
        np.fill_diagonal(D, -5.0)
        W = np.exp(beta * D)
        S = W.sum()
        if S == 0 or not np.isfinite(S):
            break
        Wn = W / S
        Gr = Wn @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1 ** (s + 1))
        vh = v / (1 - b2 ** (s + 1))
        upd = mh / (np.sqrt(vh) + eps)
        # project update to tangent
        ud = np.sum(upd * x, axis=1, keepdims=True)
        upd_t = upd - ud * x
        x = x - lr * upd_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.RandomState(0)
    n, d = 54, 23
    # fallback best
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    except Exception:
        xb0 = rng.randn(n, d); xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    if xb0.shape != (n, d):
        tmp = rng.randn(n, d); tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb0 = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb0):
            xb0 = xr.copy()
    except Exception:
        pass
    best_val = _max_cos(xb0)
    x_best = xb0.copy()

    # ---- Phase 1: symmetry-restricted search ----
    B = 32
    STEPS = 2000
    beta0, beta1 = 50.0, 800.0
    lr_seed = 0.006
    # init seeds
    A = rng.randn(B, d); A /= np.linalg.norm(A, axis=1, keepdims=True)
    Bb = rng.randn(B, d); Bb /= np.linalg.norm(Bb, axis=1, keepdims=True)
    Cc = rng.randn(B, d); Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
    shifts = np.stack([rng.choice(23, 8, replace=False) for _ in range(B)], axis=0)  # (B,8)
    idx_full = (np.arange(d)[None, :] - np.arange(23)[:, None]) % d  # (23,d)
    kvec = np.arange(d)
    idx_part = (kvec[None, None, :] - shifts[:, :, None]) % d  # (B,8,d)
    mA = np.zeros_like(A); vA = np.zeros_like(A)
    mB = np.zeros_like(Bb); vB = np.zeros_like(Bb)
    mC = np.zeros_like(Cc); vC = np.zeros_like(Cc)
    b1, b2, eps = 0.9, 0.999, 1e-8
    N = 54
    # batched loop
    for step in range(STEPS):
        if time.time() - t0 > 110.0:
            break
        frac = step / max(STEPS - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac
        # forward
        XA = np.take(A, idx_full, axis=1)  # (B,23,d)
        XB = np.take(Bb, idx_full, axis=1)
        # partial gather
        br = np.arange(B)[:, None, None]
        XP = Cc[br, idx_part]  # (B,8,d)
        X = np.concatenate([XA, XB, XP], axis=1)  # (B,54,d)
        # gram batched
        Xt = np.transpose(X, (0, 2, 1))
        G = np.matmul(X, Xt)  # (B,54,54)
        # kill diag
        ridx = np.arange(N)
        G[:, ridx, ridx] = -2.0
        gm = G.max(axis=(1, 2))  # (B,)
        D = G - gm[:, None, None]
        # diag of D is -2-gm ~ -2, already very negative for beta>=50
        W = np.exp(beta * D)
        S = W.sum(axis=(1, 2))  # (B,)
        # guard
        S = np.maximum(S, 1e-300)
        Wn = W / S[:, None, None]
        GrX = np.matmul(Wn, X)  # (B,54,d)
        GrA_full = GrX[:, 0:23, :]
        GrB_full = GrX[:, 23:46, :]
        GrP = GrX[:, 46:54, :]
        # seed grads full via roll-sum
        gA = np.zeros_like(A)
        gB = np.zeros_like(Bb)
        for s in range(23):
            gA += np.roll(GrA_full[:, s, :], -s, axis=1)
            gB += np.roll(GrB_full[:, s, :], -s, axis=1)
        gC = np.zeros_like(Cc)
        for bb in range(B):
            gC[bb] = np.bincount(idx_part[bb].ravel(), weights=GrP[bb].ravel(), minlength=d)
        # Riemannian projection
        def proj(g, s_):
            dot = np.sum(g * s_, axis=1, keepdims=True)
            return g - dot * s_
        gA = proj(gA, A); gB = proj(gB, Bb); gC = proj(gC, Cc)
        # Adam update
        t = step + 1
        for g_, s_, m_, v_ in [(gA, A, mA, vA), (gB, Bb, mB, vB), (gC, Cc, mC, vC)]:
            pass
        # manual to keep references
        mA = b1 * mA + (1 - b1) * gA
        vA = b2 * vA + (1 - b2) * (gA * gA)
        mB = b1 * mB + (1 - b1) * gB
        vB = b2 * vB + (1 - b2) * (gB * gB)
        mC = b1 * mC + (1 - b1) * gC
        vC = b2 * vC + (1 - b2) * (gC * gC)
        mhA = mA / (1 - b1 ** t); vhA = vA / (1 - b2 ** t)
        mhB = mB / (1 - b1 ** t); vhB = vB / (1 - b2 ** t)
        mhC = mC / (1 - b1 ** t); vhC = vC / (1 - b2 ** t)
        updA = mhA / (np.sqrt(vhA) + eps)
        updB = mhB / (np.sqrt(vhB) + eps)
        updC = mhC / (np.sqrt(vhC) + eps)
        updA = proj(updA, A); updB = proj(updB, Bb); updC = proj(updC, Cc)
        A = A - lr_seed * updA
        Bb = Bb - lr_seed * updB
        Cc = Cc - lr_seed * updC
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        Bb /= np.linalg.norm(Bb, axis=1, keepdims=True)
        Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
    # evaluate batch
    try:
        XA = np.take(A, idx_full, axis=1)
        XB = np.take(Bb, idx_full, axis=1)
        br = np.arange(B)[:, None, None]
        XP = Cc[br, idx_part]
        Xb = np.concatenate([XA, XB, XP], axis=1)
        vals = []
        for bb in range(B):
            Xc = Xb[bb] / np.linalg.norm(Xb[bb], axis=1, keepdims=True)
            vals.append(_max_cos(Xc))
        vals = np.array(vals)
        order = np.argsort(vals)
        cands = []
        for k in range(min(6, B)):
            bb = int(order[k])
            Xc = Xb[bb] / np.linalg.norm(Xb[bb], axis=1, keepdims=True)
            cands.append((float(vals[k]), Xc.copy()))
    except Exception:
        cands = []
    # keep distinct best few + global best
    cands = sorted(cands, key=lambda z: z[0])[:5]
    # ---- Phase 2: gradual symmetry breaking ----
    rng2 = np.random.RandomState(1)
    released = []
    for (v, Xc) in cands:
        # cheap smooth before noise
        if time.time() - t0 > BUDGET - 25.0:
            break
        try:
            Xs = _smooth_adam(Xc, steps=250, beta0=120, beta1=1500, lr=0.008)
            vs = _max_cos(Xs)
            if vs < v:
                v, Xc = vs, Xs
        except Exception:
            pass
        released.append((v, Xc))
        # noise stages
        for sigma in [1e-3, 3e-3, 1e-2]:
            if time.time() - t0 > BUDGET - 25.0:
                break
            try:
                Xn = Xc + rng2.randn(n, d) * sigma
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xn = _smooth_adam(Xn, steps=250, beta0=100, beta1=2000, lr=0.008)
                vn = _max_cos(Xn)
                released.append((vn, Xn.copy()))
                if vn < v:
                    v, Xc = vn, Xn.copy()
            except Exception:
                continue
    # merge with global best
    pool = [(best_val, x_best)] + released
    pool = sorted(pool, key=lambda z: z[0])
    # deduplicate roughly keep top 3 distinct
    best_val, x_best = pool[0]
    # ---- Phase 3: single trust-region LP polish in final 20s ----
    remain_lp = BUDGET - (time.time() - t0)
    lp_time = min(max(remain_lp - 1.0, 2.0), 20.0)
    if lp_time > 1.0:
        try:
            x_pol = _lp_polish(x_best, seconds=lp_time, t_global=t0, global_budget=BUDGET)
            v_pol = _max_cos(x_pol)
            if v_pol < best_val:
                best_val, x_best = v_pol, x_pol
        except Exception:
            pass
    # final safety: compare with loaded best
    try:
        if _max_cos(xb0) < best_val:
            x_best = xb0.copy()
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
