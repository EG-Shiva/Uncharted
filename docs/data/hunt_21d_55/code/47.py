import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _tangent_bases(X):
    n, d = X.shape
    Tb = np.empty((n, d, d - 1))
    for i in range(n):
        # nullspace of 1xd row vector: deterministic via SVD
        _, _, Vh = np.linalg.svd(X[i:i+1, :], full_matrices=True)
        Tb[i] = Vh[1:, :].T
    return Tb

def _build_J(X, Tb, pairs):
    n, d = X.shape
    dt = d - 1
    N = n * dt
    p = len(pairs[0])
    J = np.zeros((p, N), dtype=np.float64)
    for k in range(p):
        i = int(pairs[0][k]); j = int(pairs[1][k])
        # row: x_j^T T_i  and x_i^T T_j
        J[k, i*dt:(i+1)*dt] = X[j] @ Tb[i]
        J[k, j*dt:(j+1)*dt] = X[i] @ Tb[j]
    return J

def _stresses(J, iters=1500, eta=80.0):
    p = J.shape[0]
    if p == 0:
        return np.ones(0)
    G = J @ J.T
    w = np.ones(p, dtype=np.float64) / p
    for _ in range(iters):
        grad = G @ w
        gm = grad.mean()
        # exponentiated gradient on simplex
        w = w * np.exp(-eta * (grad - gm))
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            w = np.ones(p) / p
            break
        w /= s
    return w

def _lp_polish(X, seconds, radius_init=1e-3):
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (X @ X.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    LIMIT = 172.0
    # load best
    xb = None; best_load = 1e9
    for fn in ["best.npy","record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (55,21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_load:
                    best_load = s; xb = a.copy()
        except Exception:
            pass
    if xb is None:
        rng0 = np.random.default_rng(0)
        xb = _norm_rows(rng0.standard_normal((55,21)))
        best_load = _max_cos(xb)
    X0 = xb.copy()
    m0 = float(best_load)
    n, d = 55, 21
    dt = d - 1
    N = n * dt

    # active set within 1e-7 of max (computed once)
    G0 = X0 @ X0.T
    iu_all = np.triu_indices(n,1)
    c_all = G0[iu_all]
    sel_act = c_all > m0 - 1e-7
    # cap to at most 1200 most maximal
    if np.count_nonzero(sel_act) > 1200:
        idx = np.where(sel_act)[0]
        order = np.argsort(-c_all[idx])[:1200]
        ns = np.zeros_like(sel_act); ns[idx[order]]=True
        sel_act = ns
    AI = iu_all[0][sel_act]; AJ = iu_all[1][sel_act]
    p = len(AI)
    if p < 10:
        return _norm_rows(X0)

    Tb = _tangent_bases(X0)
    J = _build_J(X0, Tb, (AI, AJ))

    # contact stresses via regularised dual / exponentiated gradient (LP dual proxy)
    w = _stresses(J, iters=1500, eta=80.0)

    # hub scores: weighted degree
    deg = np.zeros(n)
    for k in range(p):
        wk = float(w[k])
        deg[int(AI[k])] += wk
        deg[int(AJ[k])] += wk
    hub_order = np.argsort(-deg)  # deterministic

    # build 20-40 high-stress contact subsets concentrated on hubs
    order_w = np.argsort(-w)
    subsets = []
    def top_k_global(k):
        k = min(k, p)
        return order_w[:k]
    subsets.append(top_k_global(30))
    subsets.append(top_k_global(20))
    subsets.append(top_k_global(40))
    # hub-concentrated
    for hubs in [hub_order[:2], hub_order[:3], hub_order[:4]]:
        hset = set(int(h) for h in hubs)
        mask = np.array([(int(AI[k]) in hset or int(AJ[k]) in hset) for k in range(p)])
        idx = np.where(mask)[0]
        if len(idx)==0:
            continue
        # sort those by w
        o = idx[np.argsort(-w[idx])]
        subsets.append(o[:min(30,len(o))])
    # single top hub
    h0 = int(hub_order[0])
    idx0 = np.where((AI==h0)|(AJ==h0))[0]
    if len(idx0)>0:
        o0 = idx0[np.argsort(-w[idx0])]
        subsets.append(o0[:min(25,len(o0))])
    # deduplicate subsets (by tuple)
    uniq = []
    seen = set()
    for s in subsets:
        t = tuple(sorted(int(v) for v in s))
        if len(t)<10:
            continue
        if t not in seen:
            seen.add(t); uniq.append(np.array(sorted(t),dtype=int))
    subsets = uniq[:8]
    if len(subsets)==0:
        subsets=[order_w[:min(30,p)]]

    # logarithmic line search grid
    steps = np.logspace(np.log10(1e-4), np.log10(0.5), 26)

    best_X = X0.copy(); best_m = m0

    # inexpensive SVD / line-search from several hub subsets
    for del_idx in subsets:
        if time.time()-t0 > LIMIT-25.0:
            break
        k = len(del_idx)
        keep_mask = np.ones(p, dtype=bool); keep_mask[del_idx]=False
        J_rem = J[keep_mask]          # (p-k) x N
        J_del = J[del_idx]            # k x N
        g_del = J_del.T @ np.ones(k)  # N, most-decrease direction for deleted
        gn = np.linalg.norm(g_del)
        if gn < 1e-12:
            continue
        # SVD nullspace of remaining Jacobian (economy SVD, capped)
        try:
            t_svd = time.time()
            U, S, Vt = np.linalg.svd(J_rem, full_matrices=False)
            # guard time: if too slow skip (should be < few sec)
            if time.time()-t_svd > 4.0:
                continue
        except Exception:
            continue
        # nullspace projection of -g_del: y = -g + Vt^T Vt g
        Vg = Vt @ g_del
        y = -g_del + Vt.T @ Vg
        ny = np.linalg.norm(y)
        cands = []
        if ny > 1e-9:
            cands.append(y/ny)
        # also regularised LS fallback: (J_rem^T J_rem + mu I)^{-1} (-g)
        # one mu value, cheap solve
        try:
            mu = 0.02
            A = J_rem.T @ J_rem + mu*np.eye(N)
            y2 = np.linalg.solve(A, -g_del)
            n2 = np.linalg.norm(y2)
            if n2 > 1e-12:
                # avoid duplicate
                cands.append(y2/n2)
        except Exception:
            pass
        for ydir in cands:
            if time.time()-t0 > LIMIT-25.0:
                break
            # map to ambient displacements
            D = np.empty((n,d))
            for i in range(n):
                D[i] = Tb[i] @ ydir[i*dt:(i+1)*dt]
            # log line search after retracting to sphere
            for s in steps:
                Xs = _norm_rows(X0 + s*D)
                v = _max_cos(Xs)
                if v < best_m - 1e-12:
                    best_m = v; best_X = Xs.copy()
            # also try negative? no, only decreasing direction; but keep check
        # early exit if huge gain? continue to try all hubs cheaply

    # accept only true decrease; else return original
    if best_m < m0 - 1e-12:
        # final short LP polish only for accepted improvement
        remain = LIMIT - (time.time()-t0)
        pol_sec = min(max(remain-2.0, 1.0), 14.0)
        if pol_sec > 0.8:
            best_X = _lp_polish(best_X, seconds=pol_sec, radius_init=8e-4)
            v2 = _max_cos(best_X)
            if v2 < best_m:
                best_m = v2
            elif v2 > best_m + 1e-9:
                # keep unpolished if polish hurt (should not)
                pass
        best_X = _norm_rows(best_X)
        # safety: ensure improvement else revert
        if _max_cos(best_X) >= m0 and best_m < m0:
            # numerical revert guard
            pass
        return best_X
    else:
        return _norm_rows(X0)
