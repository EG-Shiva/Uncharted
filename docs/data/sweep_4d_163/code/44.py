import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _row_info(x, thr_eps=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    rm = g.max(axis=1)
    m = float(rm.max())
    C = (g >= m - thr_eps).sum(axis=1)
    return rm, m, C, g

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, r0=2e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 600:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_pool_vec(pool, base, iters=40, beta=50.0, lr=0.06):
    # pool (P,4), base (M,4): minimize max dot to base
    P = pool.shape[0]
    v = pool.copy()
    for _ in range(iters):
        D = v @ base.T  # P x M
        dm = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - dm))
        Ws = W.sum(axis=1, keepdims=True)
        # weighted gradient: (W @ base)/Ws
        G = (W @ base) / np.maximum(Ws, 1e-12)  # P x 4
        dot = np.sum(G * v, axis=1, keepdims=True)
        Gt = G - dot * v
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / np.maximum(nrm, 1e-12)
        v = v - lr * Gt
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _joint_refine_inserted(ins, base, iters=30, beta=50.0, lr=0.05):
    # ins (k,4) refined against base + each other, base fixed
    k = ins.shape[0]
    if k <= 1:
        return _refine_pool_vec(ins, base, iters=iters, beta=beta, lr=lr)
    v = ins.copy()
    for _ in range(iters):
        Db = v @ base.T  # k x M
        Di = v @ v.T  # k x k
        np.fill_diagonal(Di, -2.0)
        # combine: for gradient use softmax over concatenated
        # do separate weights then combine
        mb = Db.max(axis=1, keepdims=True)
        Wb = np.exp(beta * (Db - mb))
        Gb = (Wb @ base) / np.maximum(Wb.sum(axis=1, keepdims=True), 1e-12)
        mi = Di.max(axis=1, keepdims=True)
        Wi = np.exp(beta * (Di - mi))
        # zero self already -2 so weight ~0
        Gi = (Wi @ v) / np.maximum(Wi.sum(axis=1, keepdims=True), 1e-12)
        # weight inter vs base: balance by current max values
        # if inter max dominates, emphasize it
        maxb = Db.max(axis=1, keepdims=True)
        maxi = Di.max(axis=1, keepdims=True)
        # soft switch
        wb = np.exp(beta * (maxb - np.maximum(maxb, maxi)))
        wi = np.exp(beta * (maxi - np.maximum(maxb, maxi)))
        s = wb + wi
        G = (wb * Gb + wi * Gi) / np.maximum(s, 1e-12)
        dot = np.sum(G * v, axis=1, keepdims=True)
        Gt = G - dot * v
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt / np.maximum(nrm, 1e-12)
        v = v - lr * Gt
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def solve():
    t0 = time.time()
    DEAD = 165.0  # strict internal deadline
    rng = np.random.default_rng(7)
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
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append(c)
    if not cands:
        z = rng.normal(size=(163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    REC = 0.856415430312
    n = 163
    trial = 0
    # precompute: we iterate deletion sets
    while time.time() - t0 < 148 and trial < 60:
        if time.time() - t0 > DEAD - 18:
            break
        # analyse current best for rattlers / high-contact
        rm, m, C, G = _row_info(best, thr_eps=1e-9)
        # rattlers: zero contacts (or far from max)
        ratt_idx = np.where(C == 0)[0]
        if len(ratt_idx) == 0:
            # fallback: points with rm < m-5e-4
            ratt_idx = np.where(rm < m - 5e-4)[0]
            # take up to 2 lowest
            if len(ratt_idx) > 2:
                o = np.argsort(rm[ratt_idx])[:2]
                ratt_idx = ratt_idx[o]
        else:
            if len(ratt_idx) > 2:
                o = np.argsort(rm[ratt_idx])[:2]
                ratt_idx = ratt_idx[o]
        # high-contact / high-dual-stress: sort by (C, rm)
        order = np.argsort(C + 0.3 * (rm - rm.min()) / max(m - rm.min(), 1e-12))
        # number of extra deletions 2..8 cycling
        k_extra = 2 + (trial % 7)
        # occasionally vary with trial//7 to shift ranking (stress vs count)
        if (trial // 7) % 2 == 1:
            # sort purely by rm (dual stress proxy)
            order2 = np.argsort(rm)
            top = [i for i in order2[::-1] if i not in set(ratt_idx.tolist())][:k_extra]
        else:
            top = [i for i in order[::-1] if i not in set(ratt_idx.tolist())][:k_extra]
        # occasionally add random perturbation to deletion set
        if trial % 9 == 8:
            r = int(rng.integers(0, n))
            if r not in set(ratt_idx.tolist()) and r not in set(top):
                top[-1] = r
        D = list(ratt_idx.tolist()) + top
        D = np.array(sorted(set(D)), dtype=int)
        k = len(D)
        keep = np.array([i for i in range(n) if i not in set(D.tolist())], dtype=int)
        m_keep = len(keep)
        Xk = best[keep].copy()
        # aggressively optimize remaining 163-k points
        # annealed log-sum-exp repulsion
        # scale iters with remaining budget
        Xk = _smooth(Xk, iters=380, T0=0.028, T1=0.002, lr0=0.011, lr1=0.001)
        # short LP polish on subset if time allows
        if time.time() - t0 < DEAD - 14:
            Xk = _lp_polish(Xk, seconds=0.7, r0=1.8e-3, cap=2000)
        else:
            # no time for reinsertion polish, skip
            pass
        sub_max = _max_cos(Xk)
        # explicitly check whether subset maximum falls below record
        # (and below current best, otherwise full cannot improve)
        if sub_max >= best_val:
            # subset already worse than best -> cannot win; try different deletion
            # do one light shake variant: still try next trial quickly
            trial += 1
            continue
        # ---- reinsertion: sample at least 50,000 candidate directions ----
        n_samp = 50000
        # generate in chunks to avoid huge memory spike (50000x4 fine at once)
        cand = rng.normal(size=(n_samp, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = (cand @ Xk.T).max(axis=1)
        # promising: lowest max-dot
        P = 500
        P = min(P, n_samp)
        order_c = np.argsort(sc)[:P]
        pool = cand[order_c]
        # locally optimize each promising candidate to minimize max dot with fixed points
        if time.time() - t0 > DEAD - 12:
            trial += 1
            continue
        pool = _refine_pool_vec(pool, Xk, iters=40, beta=50.0, lr=0.06)
        sc2 = (pool @ Xk.T).max(axis=1)
        # greedy insertion into largest voids, accounting for mutual dots
        o2 = np.argsort(sc2)
        pool = pool[o2]; sc2 = sc2[o2]
        sel_idx = []
        rem = list(range(len(pool)))
        # start with best
        sel_idx.append(rem.pop(0))
        while len(sel_idx) < k and rem:
            S = pool[sel_idx] @ pool[rem].T
            mm = S.max(axis=0) if len(sel_idx) > 0 else np.zeros(len(rem))
            crit = np.maximum(sc2[rem], mm)
            # add small diversity bonus already in max; pick min
            j = int(np.argmin(crit))
            sel_idx.append(rem.pop(j))
        ins = pool[np.array(sel_idx)]
        # joint refine inserted among themselves + base
        ins = _joint_refine_inserted(ins, Xk, iters=25, beta=50.0, lr=0.05)
        # assemble full
        Xfull = np.empty((163, 4))
        Xfull[keep] = Xk
        Xfull[D] = ins
        Xfull /= np.linalg.norm(Xfull, axis=1, keepdims=True)
        # finish with LP polish
        if time.time() - t0 < DEAD - 8:
            Xfull = _lp_polish(Xfull, seconds=1.0, r0=1.5e-3, cap=2000)
        vfull = _max_cos(Xfull)
        if vfull < best_val - 1e-12:
            best = Xfull.copy(); best_val = vfull
        trial += 1
    # final polish of best within remaining time (respect 180s total)
    elapsed = time.time() - t0
    budget = max(2.0, min(20.0, 172.0 - elapsed))
    t1 = time.time()
    # short smooth then LP
    if budget > 4:
        best = _smooth(best, iters=220, T0=0.010, T1=0.0015, lr0=0.005, lr1=0.0008)
        left = budget - (time.time() - t1)
        if left > 1.0:
            best = _lp_polish(best, seconds=left, r0=1.2e-3, cap=2200)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
