import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, m=None, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    C = G > (m - tol)
    cnt = C.sum(axis=1)
    # summed stress
    S = np.where(C, G - (m - tol), 0.0).sum(axis=1)
    return G, m, cnt, S

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 30:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        # cap to worst pairs for speed
        if p > 1500:
            csel = c[sel]
            idx = np.argpartition(-csel, 1500)[:1500]
            I = I[idx]; J = J[idx]
            csel2 = csel[idx]
            p = 1500
        else:
            csel2 = c[sel]
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel2, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_full(X, steps, k0=40.0, k1=250.0, lr0=0.02, lr1=0.002):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _relax_keep(K, steps=150, k0=50.0, k1=200.0):
    K = K.copy()
    m = K.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = 0.015 * (1.0 - frac) + 0.002 * frac
        G = K @ K.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ K
        dot = np.einsum('ij,ij->i', Gr, K)[:, None]
        Gt = Gr - dot * K
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        K = K - (lr / nrm) * Gt
        K /= np.linalg.norm(K, axis=1, keepdims=True)
    return K

def _init_deleted(K, kdel, rng, n_cand=2000):
    d = K.shape[1]
    D = np.empty((kdel, d))
    for j in range(kdel):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ K.T
        mx = S.max(axis=1)
        if j > 0:
            S2 = C @ D[:j].T
            mx2 = S2.max(axis=1)
            mx = np.maximum(mx, mx2)
        # bias to deepest holes: pick best, with small random among top few
        if kdel > 1 and j > 0:
            idx = np.argpartition(mx, 8)[:8]
            # random pick among top-8 weighted to best
            pick = idx[int(rng.integers(0, len(idx)))] if rng.random() < 0.35 else int(np.argmin(mx))
        else:
            pick = int(np.argmin(mx))
        D[j] = C[pick]
    return D

def _joint_gd(D, K, steps=1200, k0=40.0, k1=220.0, lr0=0.025, lr1=0.002, rng=None, kick=True):
    D = D.copy()
    kd = D.shape[0]
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        S = D @ K.T
        Td = D @ D.T
        np.fill_diagonal(Td, -2.0)
        m = float(S.max()) if S.size else -2.0
        m2 = float(Td.max()) if Td.size else -2.0
        if m2 > m:
            m = m2
        WS = np.exp(np.clip(k * (S - m), -50, 0))
        WT = np.exp(np.clip(k * (Td - m), -50, 0))
        np.fill_diagonal(WT, 0.0)
        Gr = WS @ K + WT @ D
        dot = np.einsum('ij,ij->i', Gr, D)[:, None]
        Gt = Gr - dot * D
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        D = D - (lr / nrm) * Gt
        # occasional DE-style kick of worst point
        if kick and rng is not None and (s % 400 == 399) and s < steps - 100:
            # find worst deleted point (max dot to K or D)
            mxS = S.max(axis=1) if S.size else np.full(kd, -2.0)
            mxT = Td.max(axis=1)
            mx = np.maximum(mxS, mxT)
            w = int(np.argmax(mx))
            D[w] = D[w] + 0.05 * rng.standard_normal(D.shape[1])
        D /= np.linalg.norm(D, axis=1, keepdims=True)
    return D

def solve():
    rng = np.random.default_rng(42)
    t0 = time.time()
    T_BUD = 172.0
    t_dead = t0 + T_BUD
    n, d = 181, 9
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (n, d):
        flat = xb.ravel()
        need = n * d
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    X_best = xb.copy()

    del_sizes = [14, 22, 8, 14, 22, 8, 14, 22, 8, 14, 22, 8]
    trial = 0
    for kdel in del_sizes:
        if time.time() > t_dead - 25.0:
            break
        trial += 1
        X = X_best.copy()
        G, m, cnt, stress = _contacts(X, tol=1e-7)
        # order: rattlers first, then largest contact count, tie by stress/per-max
        per_max = G.max(axis=1)
        # score for excision: cnt primary, stress secondary, per_max tertiary
        order = sorted(range(n), key=lambda i: (cnt[i] == 0, cnt[i], stress[i], per_max[i]), reverse=True)
        # ensure rattlers included: they have cnt 0 so sorted last; force include them
        ratt = [i for i in range(n) if cnt[i] == 0]
        Didx = []
        for r in ratt:
            if len(Didx) < kdel and r not in Didx:
                Didx.append(r)
        for i in order:
            if len(Didx) >= kdel:
                break
            if i not in Didx:
                Didx.append(i)
        Didx = np.array(Didx[:kdel], dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[Didx] = False
        K = X[keep_mask].copy()
        # briefly relax remainder to open genuine holes (cap: don't let it blow up)
        m_keep0 = _max_cos(K) if K.shape[0] > 1 else -2.0
        K2 = _relax_keep(K, steps=150, k0=50.0, k1=200.0)
        m_keep1 = _max_cos(K2) if K2.shape[0] > 1 else -2.0
        # only keep relaxed if not much worse than before / threshold
        if m_keep1 > min(m_keep0 + 0.002, 0.46247 + 0.004):
            K2 = K  # revert
        # jointly optimize deleted points against frozen remainder
        D0 = _init_deleted(K2, kdel, rng, n_cand=2000)
        D1 = _joint_gd(D0, K2, steps=1200, k0=40.0, k1=220.0, lr0=0.025, lr1=0.002, rng=rng, kick=True)
        # extra joint pass with higher exponent for squeeze
        D1 = _joint_gd(D1, K2, steps=400, k0=120.0, k1=260.0, lr0=0.008, lr1=0.001, rng=None, kick=False)
        Xnew = np.empty_like(X)
        Xnew[keep_mask] = K2
        Xnew[~keep_mask] = D1
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        # unfreeze: smooth exponential-repulsion anneal 40->250
        Xnew = _smooth_full(Xnew, steps=400, k0=40.0, k1=250.0, lr0=0.02, lr1=0.002)
        v = _max_cos(Xnew)
        if v < best_val + 0.006:
            if time.time() > t_dead - 24.0:
                if v < best_val:
                    X_best, best_val = Xnew, v
                continue
            Xp = _lp_polish(Xnew, budget=0.7, t_dead=t_dead - 20.0)
            vp = _max_cos(Xp)
            if vp < best_val:
                X_best, best_val = Xp, vp
            elif v < best_val:
                X_best, best_val = Xnew, v
        else:
            if v < best_val:
                X_best, best_val = Xnew, v

    # final polish
    remain = t_dead - time.time() - 0.8
    if remain > 2.0:
        X_best = _lp_polish(X_best, budget=min(16.0, remain * 0.7), t_dead=t_dead - 0.4)
        if time.time() < t_dead - 5.0:
            X_best = _smooth_full(X_best, steps=250, k0=200.0, k1=500.0, lr0=0.005, lr1=0.0005)
            X_best = _lp_polish(X_best, budget=max(1.0, t_dead - time.time() - 0.3), t_dead=t_dead - 0.15)
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(X_best)):
        X_best = xb
    return X_best
