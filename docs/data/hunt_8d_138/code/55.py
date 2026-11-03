import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm == 0] = 1.0
    return X / nrm

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _score_candidates(C, P):
    # max cosine of each candidate to partial P
    n = C.shape[0]
    chunk = 5000
    out = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s+chunk, n)
        M = C[s:e] @ P.T
        out[s:e] = M.max(axis=1)
    return out

def _refine_single(c, P, beta=300.0, steps=50, lr=0.08):
    c = c / (np.linalg.norm(c)+1e-30)
    for _ in range(steps):
        d = P @ c
        mv = float(d.max())
        E = np.exp(beta*(d-mv))
        S = float(E.sum())+1e-300
        g = (E @ P)/S
        gt = g - float(g @ c)*c
        c = c - lr*gt
        n = np.linalg.norm(c)
        if not np.isfinite(n) or n == 0:
            break
        c = c/n
    return c

def _relax_partial(X, steps=25, beta=250.0, lr=0.05):
    X = _norm(X)
    n = X.shape[0]
    for _ in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        # per-row softmax
        mv = G.max(axis=1, keepdims=True)
        E = np.exp(beta*(G-mv))
        # zero diag contribution
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)+1e-300
        GX = (E @ X)/rs  # n x d, row-normalized
        dot = np.sum(GX*X, axis=1, keepdims=True)
        Gt = GX - dot*X
        X = _norm(X - lr*Gt)
    return X

def _lp_polish(X, deadline, radius0=1e-2, active=1e-3, cap=3000):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        thr = m - active
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method='highs')
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
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _rebuild_once(X0, k, rng, target, deadline, n_samp=30000, n_top=20):
    n, d = X0.shape
    pm = _per_point_max(X0)
    mn = float(pm.min())
    w = (pm - mn + 1e-9)
    w = w*w + 0.03
    w = w / w.sum()
    rem = rng.choice(n, size=k, replace=False, p=w)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[rem] = False
    partial = X0[keep_mask].copy()
    # insert one at a time
    for s in range(k):
        if time.time() > deadline - 12:
            break
        # sample
        C = rng.standard_normal((n_samp, d))
        C = _norm(C)
        scores = _score_candidates(C, partial)
        idx = np.argpartition(scores, n_top)[:n_top]
        best_c = None
        best_v = 1e9
        for ii in idx:
            c0 = C[ii]
            cr = _refine_single(c0, partial, beta=300.0, steps=50, lr=0.08)
            v = float(np.max(partial @ cr))
            if v < best_v:
                best_v = v
                best_c = cr
        # backtrack if stalled above target: second sampling
        if best_v > target:
            try:
                C2 = rng.standard_normal((n_samp, d))
                C2 = _norm(C2)
                scores2 = _score_candidates(C2, partial)
                idx2 = np.argpartition(scores2, n_top)[:n_top]
                for ii in idx2:
                    c0 = C2[ii]
                    cr = _refine_single(c0, partial, beta=300.0, steps=50, lr=0.08)
                    v = float(np.max(partial @ cr))
                    if v < best_v:
                        best_v = v
                        best_c = cr
            except Exception:
                pass
        partial = np.vstack([partial, best_c.reshape(1, d)])
        partial = _norm(partial)
        if (s+1) % 4 == 0:
            try:
                partial = _relax_partial(partial, steps=20, beta=250.0, lr=0.05)
            except Exception:
                pass
    # brief global relax at end
    try:
        partial = _relax_partial(partial, steps=30, beta=300.0, lr=0.04)
    except Exception:
        pass
    return _norm(partial)

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
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
    incumbent = best.copy()
    inc_val = best_val
    archive = [(inc_val, incumbent.copy())]
    blocks = [16, 24, 32, 48, 24, 16, 32, 48]
    noises = [0.0, 0.008, 0.015, 0.005, 0.02, 0.01, 0.012, 0.007]
    attempt = 0
    while time.time() < deadline - 12:
        k = blocks[attempt % len(blocks)]
        nz = noises[attempt % len(noises)]
        # perturbed copy start
        if nz == 0.0 and attempt == 0:
            X0 = incumbent.copy()
        else:
            X0 = incumbent + nz * rng.standard_normal((n, d))
            X0 = _norm(X0)
            # also occasionally start from archive member
            if attempt % 5 == 4 and len(archive) > 1:
                X0 = archive[rng.integers(len(archive))][1].copy()
                X0 = _norm(X0 + 0.01*rng.standard_normal((n, d)))
        target = inc_val - 2e-4
        try:
            Xr = _rebuild_once(X0, int(k), rng, float(target), deadline)
        except Exception:
            attempt += 1
            continue
        try:
            pre = _max_cos(Xr)
        except Exception:
            attempt += 1
            continue
        # keep archive distinct (by value gap)
        if len(archive) < 6:
            archive.append((pre, Xr.copy()))
            archive.sort(key=lambda x: x[0])
            archive = archive[:6]
        else:
            if pre < archive[-1][0] and all(abs(pre-v) > 1e-7 for v,_ in archive):
                archive[-1] = (pre, Xr.copy())
                archive.sort(key=lambda x: x[0])
        # decide polish budget
        polished = False
        if pre < inc_val + 2e-3 and time.time() < deadline - 10:
            budget = 3.0 if pre < inc_val + 5e-4 else 1.5
            try:
                Xp = _lp_polish(Xr, min(deadline-1.0, time.time()+budget))
                vp = _max_cos(Xp)
                polished = True
            except Exception:
                Xp, vp = Xr, pre
            if vp < inc_val and np.all(np.isfinite(Xp)):
                inc_val = vp
                incumbent = Xp.copy()
                # update archive best
                archive.append((vp, Xp.copy()))
                archive.sort(key=lambda x: x[0])
                archive = archive[:6]
            elif pre < inc_val and not polished:
                inc_val = pre
                incumbent = Xr.copy()
        attempt += 1
        if attempt > 30:
            break
    # final polish of incumbent
    try:
        if time.time() < deadline - 2:
            Xf = _lp_polish(incumbent, deadline-0.5)
            vf = _max_cos(Xf)
            if vf < inc_val and np.all(np.isfinite(Xf)):
                incumbent = Xf
                inc_val = vf
    except Exception:
        pass
    # fallback
    try:
        if inc_val > best_val + 1e-12:
            return _norm(best)
    except Exception:
        pass
    return _norm(incumbent)
