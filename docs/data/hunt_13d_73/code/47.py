import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    # count per point (mask symmetric, diag false)
    cnt = np.count_nonzero(mask, axis=1)
    iu0, iu1 = np.triu_indices(X.shape[0], 1)
    c = (X @ X.T)[iu0, iu1]
    return m, cnt, iu0, iu1, c

def _refine_mobile(Y0, F, steps=600, s0=60.0, s1=500.0, lr0=0.02, lr1=0.002):
    k, d = Y0.shape
    nf = F.shape[0]
    Y = Y0.copy()
    m1 = np.zeros((k, d)); v1 = np.zeros((k, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    l0, l1 = np.log(lr0), np.log(lr1)
    iu_a, iu_b = np.triu_indices(k, 1)
    has_pair = len(iu_a) > 0
    for it in range(steps):
        frac = it / (steps - 1) if steps > 1 else 1.0
        s = s0 + (s1 - s0) * frac
        lr = np.exp(l0 + (l1 - l0) * frac)
        C1 = Y @ F.T
        if has_pair:
            C2u = (Y @ Y.T)[iu_a, iu_b]
            mx = C1.max()
            if C2u.size:
                mx = max(mx, float(C2u.max()))
            e1 = np.exp(s * (C1 - mx))
            e2 = np.exp(s * (C2u - mx))
            se = float(e1.sum() + e2.sum())
            if not np.isfinite(se) or se <= 0:
                w1 = np.full_like(C1, 1.0 / (C1.size + C2u.size))
                w2 = np.full_like(C2u, 1.0 / (C1.size + C2u.size))
            else:
                w1 = e1 / se
                w2 = e2 / se
            G1 = w1 @ F
            W2 = np.zeros((k, k))
            W2[iu_a, iu_b] = w2
            W2[iu_b, iu_a] = w2
            G2 = W2 @ Y
            WX = G1 + G2
        else:
            mx = float(C1.max())
            e1 = np.exp(s * (C1 - mx))
            se = float(e1.sum())
            w1 = e1 / se if se > 0 and np.isfinite(se) else np.full_like(C1, 1.0 / C1.size)
            WX = w1 @ F
        sprod = np.sum(WX * Y, axis=1)
        Gt = WX - sprod[:, None] * Y
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
        b1p *= b1; b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        Y = Y - lr * step
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y /= nr
    return Y

def _adam_full(X0, s0, s1, steps, lr0, lr1, t_end=None):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros((n, d)); v = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    K = len(iu0)
    l0, l1 = np.log(lr0), np.log(lr1)
    for k in range(steps):
        if t_end is not None and time.time() > t_end:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = s0 + (s1 - s0) * frac
        lr = np.exp(l0 + (l1 - l0) * frac)
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            w = np.ones(K) / K
        else:
            w = e / se
        WX = np.zeros((n, d))
        np.add.at(WX, iu0, w[:, None] * X[iu1])
        np.add.at(WX, iu1, w[:, None] * X[iu0])
        sprod = np.sum(WX * X, axis=1)
        Gt = WX - sprod[:, None] * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        b1p *= b1; b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        mm = float(c.max())
        sel = c > mm - max(20.0 * radius, 2e-7)
        if np.count_nonzero(sel) > 2000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng = np.random.default_rng(12345)
    n, d = 73, 13
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc
                        Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbase = Xrec.copy()
    best = float(best_file)
    Xbest = Xbase.copy()

    # ---- targeted surgery phase ----
    # identify rattler + hubs on current base
    cur = Xbase.copy()
    cur_best = best
    fallback_X = cur.copy()
    fallback_mob = 1e9
    n_trials = 55
    for trial in range(n_trials):
        if time.time() > T_END - 85.0:
            break
        mval, cnt, _, _, _ = _contacts(cur)
        rattlers = np.where(cnt == 0)[0]
        r = int(rattlers[0]) if len(rattlers) > 0 else int(np.argmin(cnt))
        order = np.argsort(-cnt)
        order = [int(o) for o in order if int(o) != r]
        top = order[:14]
        # choose k: rattler + 2..4 hubs
        k_extra = [2, 3, 3, 4][trial % 4]
        # diverse hub choice: random subset of top
        if len(top) < k_extra:
            hubs = top
        else:
            # mix deterministic top slice + random to diversify
            if trial % 3 == 0:
                hubs = top[:k_extra]
            else:
                hubs = list(rng.choice(top, size=k_extra, replace=False))
        removal = [r] + [int(h) for h in hubs]
        removal = sorted(set(removal))
        k = len(removal)
        keep = np.array([i for i in range(n) if i not in removal])
        F = cur[keep]
        # 20k hole samples
        pool = rng.standard_normal(size=(20000, d))
        pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        S = pool @ F.T
        smax = S.max(axis=1)
        # greedy diverse pick from pool
        chosen = []
        chosen_idx = []
        avail_mask = np.ones(20000, dtype=bool)
        Y0 = np.zeros((k, d))
        for a in range(k):
            if chosen:
                Pl = np.stack(chosen, axis=0)
                cross = pool @ Pl.T
                cmax = cross.max(axis=1)
                comb = np.maximum(smax, cmax)
            else:
                comb = smax
            # avoid re-pick
            comb = np.where(avail_mask, comb, np.inf)
            bi = int(np.argmin(comb))
            chosen_idx.append(bi)
            avail_mask[bi] = False
            Y0[a] = pool[bi]
            chosen.append(pool[bi])
        Y0 = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
        # refine with hinge-loss (exact-max) Adam
        Yref = _refine_mobile(Y0, F, steps=500, s0=60.0, s1=600.0, lr0=0.02, lr1=0.002)
        Xnew = cur.copy()
        for a, idx in enumerate(removal):
            Xnew[idx] = Yref[a]
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        gnew = _max_cos(Xnew)
        # mobile-involved max
        C1 = Yref @ F.T
        C2 = Yref @ Yref.T
        np.fill_diagonal(C2, -2.0)
        mob = float(max(C1.max(), C2.max()))
        if mob < fallback_mob:
            fallback_mob = mob
            fallback_X = Xnew.copy()
        # accept only if lowers unpolished global max
        if gnew < cur_best - 1e-12:
            cur = Xnew.copy()
            cur_best = gnew
            if gnew < best:
                best = gnew
                Xbest = Xnew.copy()
    # if surgery improved, keep it; else keep fallback topology for polish
    if cur_best < best_file - 1e-12:
        Xbest = cur.copy(); best = cur_best
    else:
        # use fallback (different topology, equal global) as starting point for polish
        # only if it is not worse
        fg = _max_cos(fallback_X)
        if fg <= best + 2e-6:
            # prefer the fallback with lowest mobile max; it relieves hubs
            cur = fallback_X.copy()
            cur_best = fg

    # ---- cheap full polish: annealed Adam then one short LP ----
    if time.time() < T_END - 40:
        try:
            Xa = _adam_full(cur, 80.0, 400.0, 900, 0.015, 0.002, t_end=T_END - 35)
            va = _max_cos(Xa)
            if va < best and np.all(np.isfinite(Xa)):
                best = va; Xbest = Xa.copy()
                cur = Xa.copy(); cur_best = va
            if time.time() < T_END - 30:
                Xb = _adam_full(cur, 300.0, 900.0, 700, 0.008, 0.0012, t_end=T_END - 28)
                vb = _max_cos(Xb)
                if vb < best and np.all(np.isfinite(Xb)):
                    best = vb; Xbest = Xb.copy()
                    cur = Xb.copy(); cur_best = vb
        except Exception:
            pass
    # one short LP polish on best accepted configuration
    if time.time() < T_END - 8:
        try:
            secs = min(22.0, T_END - 5.0 - time.time())
            if secs > 4:
                Xp, vp = _lp_polish(cur if cur_best <= best + 1e-12 else Xbest, secs, T_END - 2.0)
                if vp < best and np.all(np.isfinite(Xp)):
                    best = vp; Xbest = Xp.copy()
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
