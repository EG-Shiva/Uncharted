import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-9):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    cnt = mask.sum(axis=1)
    rowmax = G.max(axis=1)
    return m, cnt, rowmax, G

def _lp_polish(X, seconds, t_deadline=None):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        cf = Gfull[iu0]
        m = float(cf.max())
        if m < best:
            best = m
        sel = cf > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        c = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argpartition(-c, 3000)[:3000]
            I = I[idx]; J = J[idx]; c = c[idx]
            p = 3000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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

def _hole_seeds(F, k, rng, n_samp=50000):
    d = F.shape[1]
    C = _norm(rng.randn(n_samp, d))
    S = C @ F.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.55:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = _norm(rng.randn(1, d))[0]
        picked.append(v)
    return np.array(picked)

def _hole_local_refine(H, F, iters=200, T=0.005, lr=0.08):
    H = _norm(H.copy())
    for _ in range(iters):
        D = H @ F.T
        dm = D.max(axis=1, keepdims=True)
        W = np.exp((D - dm) / T)
        W /= W.sum(axis=1, keepdims=True) + 1e-30
        G = W @ F
        dot = (G * H).sum(axis=1, keepdims=True)
        Gt = G - dot * H
        H = _norm(H - lr * Gt)
    return H

def _joint_lbfgs(H0, F, temps=(0.01, 0.003, 0.001), maxiter=150):
    k, d = H0.shape
    Nf = F.shape[0]
    Y = H0.copy().reshape(-1)
    # precompute triu indices for free-free
    iu = np.triu_indices(k, 1)
    nff = len(iu[0])
    for T in temps:
        def fun(yflat):
            Ym = yflat.reshape(k, d)
            r = np.linalg.norm(Ym, axis=1)
            r = np.maximum(r, 1e-12)
            Z = Ym / r[:, None]
            Dfz = Z @ F.T  # k x Nf
            if nff > 0:
                Gff = Z @ Z.T
                dff = Gff[iu]
            else:
                dff = np.zeros(0)
            dmax = float(Dfz.max())
            if nff > 0 and dff.size > 0:
                dmax = max(dmax, float(dff.max()))
            Efz = np.exp((Dfz - dmax) / T)
            Sfz = Efz.sum()
            if nff > 0:
                Eff = np.exp((dff - dmax) / T)
                Sff = Eff.sum()
            else:
                Sff = 0.0
            S = Sfz + Sff
            f = dmax + T * np.log(max(S, 1e-300))
            Wfz = Efz / S
            Gz = Wfz @ F  # k x d
            if nff > 0:
                Wff = Eff / S
                Wsym = np.zeros((k, k))
                Wsym[iu] = Wff
                Wsym = Wsym + Wsym.T
                Gz = Gz + Wsym @ Z
            # chain to Y
            dot = (Gz * Z).sum(axis=1)
            Gy = (Gz - dot[:, None] * Z) / r[:, None]
            return float(f), Gy.reshape(-1)
        try:
            res = minimize(fun, Y, method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter*2})
            Y = res.x
        except Exception:
            break
    Ym = Y.reshape(k, d)
    Z = _norm(Ym)
    return Z

def _smooth_refine(X, steps=200, T=0.004, lr=0.02):
    X = _norm(X.copy())
    n, d = X.shape
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        P = E / S
        Grad = P @ X
        # tangent projection (factor 2 absorbed in lr)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        mx = float(np.linalg.norm(Gt, axis=1).max())
        if mx < 1e-14:
            break
        cur = lr * (1.0 - 0.5 * it / steps)
        X = _norm(X - (cur / (mx + 1e-12)) * Gt * mx if False else X - (Gt / mx * cur))
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(12345)
    Xb = None
    sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s
                    Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    X_best = Xb.copy()
    s_best = float(sb)
    n, d = X_best.shape
    ks = [3, 4, 6, 8, 10]
    it = 0
    # time for cavity loop until ~150s
    while time.time() - t0 < 150.0:
        if time.time() - t0 > BUDGET - 20:
            break
        it += 1
        k = ks[(it - 1) % len(ks)]
        mode = (it - 1) % 3  # 0 rattler+top, 1 cluster, 2 random mix
        m, cnt, rowmax, G = _contacts(X_best, tol=1e-9)
        # rattler: zero-contact points
        zero = np.where(cnt == 0)[0]
        if len(zero) > 0:
            # pick lowest rowmax among zeros
            rattler = int(zero[np.argmin(rowmax[zero])])
        else:
            rattler = int(np.argmin(cnt))
        order_hi = np.argsort(-cnt)
        tied = np.where(rowmax > m - 1e-12)[0]
        if len(tied) == 0:
            tied = order_hi[:20]
        delset = set([rattler])
        if mode == 0:
            for idx in order_hi:
                if len(delset) >= k:
                    break
                delset.add(int(idx))
        elif mode == 1:
            # spatial cluster: random seed among tied/high
            cand_pool = tied if len(tied) >= 5 else order_hi[:30]
            seed = int(cand_pool[rng.randint(len(cand_pool))])
            dots_to_seed = X_best @ X_best[seed]
            # nearest k-1 to seed
            nn = np.argsort(-dots_to_seed)
            for idx in nn:
                if len(delset) >= k:
                    break
                # ensure rattler kept: add nearest, rattler already in
                delset.add(int(idx))
            # delset already contains rattler; if seed cluster didn't include rattler neighbours, fine
            # if size exceeds, it is k
        else:
            # random mixture: rattler + random, bias to high contacts
            others = list(range(n))
            others.remove(rattler)
            # mix: half biased to high-contact, half uniform
            hi_list = [int(i) for i in order_hi[:40]]
            while len(delset) < k:
                if rng.rand() < 0.5:
                    c = hi_list[rng.randint(len(hi_list))]
                    delset.add(c)
                else:
                    delset.add(int(others[rng.randint(len(others))]))
        del_idx = sorted(list(delset))[:k]
        # in case duplicates caused smaller, fill random
        while len(del_idx) < k:
            c = int(rng.randint(n))
            if c not in del_idx:
                del_idx.append(c)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        F = X_best[keep_mask]
        # holes
        H0 = _hole_seeds(F, k, rng, n_samp=50000)
        H0 = _hole_local_refine(H0, F, iters=150, T=0.005, lr=0.08)
        # joint L-BFGS on free points
        try:
            Zfree = _joint_lbfgs(H0, F, temps=(0.01, 0.003, 0.001), maxiter=150)
        except Exception:
            continue
        Xnew = np.zeros_like(X_best)
        Xnew[keep_mask] = F
        # map del_idx order to Zfree rows
        for a, gi in enumerate(del_idx):
            Xnew[gi] = Zfree[a]
        Xnew = _norm(Xnew)
        # short smooth refinement unfreezed
        try:
            Xnew = _smooth_refine(Xnew, steps=150, T=0.004, lr=0.015)
        except Exception:
            pass
        # trust-region LP polish 3-5 s
        remain = BUDGET - (time.time() - t0)
        if remain < 8:
            break
        lp_sec = 4.0
        if remain < lp_sec + 5:
            lp_sec = max(1.0, remain - 5)
        try:
            Xpol = _lp_polish(Xnew, seconds=lp_sec)
        except Exception:
            continue
        try:
            s = _max_cos(Xpol)
        except Exception:
            continue
        if s < s_best and np.all(np.isfinite(Xpol)):
            X_best = Xpol.copy()
            s_best = float(s)
        # also allow equal to diversify? keep best only
        if time.time() - t0 > 150.0:
            break
    return _norm(X_best)
