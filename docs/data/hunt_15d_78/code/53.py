import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_masked(X, deadline, exclude=None, r_init=2e-3, r_max=0.03):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bx = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)
    if exclude is None:
        exclude = np.zeros(P, dtype=bool)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = max(20.0 * radius, 1e-9)
        sel = (c_all > m - thr) & (~exclude)
        if int(np.count_nonzero(sel)) == 0:
            radius *= 0.5
            continue
        # cap to largest 1200 for speed
        nnz = int(np.count_nonzero(sel))
        if nnz > 1200:
            idx = np.where(sel)[0]
            ord_ = np.argsort(-c_all[idx])[:1200]
            m2 = np.zeros_like(sel)
            m2[idx[ord_]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)):
            radius *= 0.5
            continue
        y = y / np.maximum(yn, 1e-12)
        v = _max_cos(y)
        if v < best:
            X, best, bx = y, v, y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return bx, best

def _lp_duals(X, radius=2e-3):
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    G = X @ X.T
    c_all = G[iu0, iu1]
    m = float(c_all.max())
    thr = max(20.0 * radius, 1e-9)
    sel = c_all > m - thr
    I = iu0[sel]; J = iu1[sel]
    p = len(I)
    c_sel = c_all[sel]
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
    rows = np.repeat(np.arange(p), 2 * d + 1)
    vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None, sel, None
    if res.status != 0:
        return res, sel, None
    w = None
    try:
        # scipy highs marginals
        if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals"):
            w = np.asarray(res.ineqlin.marginals, dtype=float)
            w = -w  # active => positive weight
        elif isinstance(res, dict) and "marginals" in res:
            w = np.asarray(res["marginals"], dtype=float)[:p]
            w = -w
        elif hasattr(res, "marginals"):
            mm = np.asarray(res.marginals, dtype=float)
            if mm.size >= p:
                w = -mm[:p]
    except Exception:
        w = None
    if w is None or w.shape[0] != p or not np.all(np.isfinite(w)):
        w = None
    return res, sel, w

def _lse_exempt_refine(X0, exclude_mask, beta, delta, m_ref, lam, maxiter):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    inc = ~exclude_mask
    # if all excluded or none, fallback
    if int(np.count_nonzero(inc)) == 0:
        inc = np.ones_like(inc)
    Xc = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xf = Xc.ravel().copy()
    thr_ex = m_ref - delta
    def fun(g):
        X = g.reshape(n, d)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Y = X / nr
        G = Y @ Y.T
        gv = G[iu0, iu1]
        g_ne = gv[inc]
        m = float(g_ne.max())
        e = np.exp(beta * (g_ne - m))
        S = float(e.sum())
        F = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        # map back to full pair ordering
        full_w = np.zeros_like(gv)
        full_w[inc] = w
        W[iu0, iu1] = full_w
        W = W + W.T
        GY = W @ Y
        dot = (GY * Y).sum(axis=1, keepdims=True)
        GX = (GY - dot * Y) / nr
        # hinge on exempt pairs
        if int(np.count_nonzero(exclude_mask)) > 0 and lam > 0:
            ge = gv[exclude_mask]
            over = ge - thr_ex
            pos = over > 0
            if np.any(pos):
                F = F + lam * float(np.sum(over[pos] ** 2))
                # gradient of lam*over^2 = 2*lam*over * d(ge)/dX
                hw = np.zeros_like(gv)
                hov = np.zeros_like(ge)
                hov[pos] = 2.0 * lam * over[pos]
                hw[exclude_mask] = hov
                HW = np.zeros((n, n), dtype=np.float64)
                HW[iu0, iu1] = hw
                HW = HW + HW.T
                HY = HW @ Y
                dot2 = (HY * Y).sum(axis=1, keepdims=True)
                GX = GX + (HY - dot2 * Y) / nr
        return float(F), GX.ravel()
    try:
        res = minimize(fun, xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-6})
        xf = res.x
    except Exception:
        pass
    Xo = xf.reshape(n, d)
    Xo = Xo / np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
    return Xo

def _fingerprint(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    top = np.sort(c)[-250:][::-1]  # 250 largest
    C = (G > m - 1e-7)
    deg = np.sort(C.sum(axis=1).astype(float))
    return m, top, deg

def _fp_dist(fp1, fp2):
    _, t1, d1 = fp1
    _, t2, d2 = fp2
    return float(np.sum(np.abs(d1 - d2)) + 2000.0 * float(np.max(np.abs(t1 - t2))))

def solve():
    t0 = time.time()
    HARD = 174.0
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # quick snap
    try:
        bx, bv = _lp_polish_masked(bestX, t0 + 3.0, exclude=None)
        if bv < bestV:
            bestX, bestV = bx, bv
    except Exception:
        pass
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)
    base_fp = _fingerprint(bestX)
    # cached dual info
    dual_sel = None; dual_w = None; dual_idx = None
    def refresh_duals(X):
        nonlocal dual_sel, dual_w, dual_idx
        try:
            _, sel, w = _lp_duals(X, radius=2e-3)
            dual_sel = sel
            if w is not None:
                # map to global pair indices
                gidx = np.where(sel)[0]
                order = np.argsort(-w)
                dual_idx = gidx[order]
                dual_w = w[order]
            else:
                dual_idx = None; dual_w = None
        except Exception:
            dual_idx = None; dual_w = None
    refresh_duals(bestX)
    div_pool = []  # list of (X, v)
    it = 0
    while time.time() - t0 < HARD - 14:
        it += 1
        tl = HARD - 14 - (time.time() - t0)
        if tl < 4.0:
            break
        # choose start: mostly best, sometimes diverse
        if div_pool and (it % 4 == 0):
            si = int(rng.integers(len(div_pool)))
            startX = div_pool[si][0].copy()
        else:
            startX = bestX.copy()
        # contact info
        try:
            G0 = startX @ startX.T
            np.fill_diagonal(G0, -2.0)
            mcur = float(G0.max())
            c_all = G0[iu0, iu1]
            Cgr = (G0 > mcur - 1e-7)
            deg = Cgr.sum(axis=1)
            hi_nodes = np.where(deg >= np.median(deg))[0]
        except Exception:
            mcur = bestV
            c_all = (startX @ startX.T)[iu0, iu1]
            deg = np.zeros(n)
            hi_nodes = np.arange(n)
        # pick exempt set strategy
        strat = it % 3  # 0 largest-dual, 1 common-neighbour, 2 random core
        K = int(rng.integers(30, 81))
        delta = float(rng.uniform(0.002, 0.005))
        # candidate active pool: pairs within 0.01 of max or dual sel
        try:
            pool = np.where(c_all > mcur - 0.01)[0]
        except Exception:
            pool = np.arange(P)
        # restrict to core-concentrated: at least one endpoint high-degree
        # build core pool
        is_core_edge = np.isin(iu0, hi_nodes) | np.isin(iu1, hi_nodes)
        core_pool = pool[is_core_edge[pool]] if len(pool) else np.where(is_core_edge)[0]
        if len(core_pool) < K:
            core_pool = np.where(is_core_edge)[0]
        exempt = np.zeros(P, dtype=bool)
        chosen = None
        if strat == 0 and dual_idx is not None and len(dual_idx) > 0:
            # largest-dual intersect core if possible
            filt = [g for g in dual_idx if is_core_edge[g]]
            if len(filt) < K:
                filt = list(dual_idx)
            # add jitter: randomly drop 20%
            filt = np.asarray(filt[:max(K * 2, K + 10)])
            if len(filt) > K:
                # take top K with small random subsample
                if rng.random() < 0.5:
                    chosen = filt[:K]
                else:
                    # random subset of top 2K
                    chosen = rng.choice(filt[:min(len(filt), 2 * K)], size=K, replace=False)
            else:
                chosen = filt
        elif strat == 1:
            # highest common neighbour among core pool top by cosine
            cand = core_pool[np.argsort(-c_all[core_pool])[:min(len(core_pool), 300)]]
            # common neighbours
            try:
                scores = np.empty(len(cand), dtype=float)
                for k2, g in enumerate(cand):
                    i = int(iu0[g]); j = int(iu1[g])
                    scores[k2] = float(np.dot(Cgr[i], Cgr[j])) + 0.1 * float(c_all[g])
                order = np.argsort(-scores)[:K]
                chosen = cand[order]
            except Exception:
                chosen = rng.choice(core_pool, size=min(K, len(core_pool)), replace=False)
        else:
            chosen = rng.choice(core_pool, size=min(K, len(core_pool)), replace=False)
        if chosen is None or len(chosen) == 0:
            continue
        exempt[np.asarray(chosen, dtype=int)] = True
        # perturb points involved in exempt set
        involved = np.unique(np.concatenate([iu0[exempt], iu1[exempt]]))
        Xtry = startX.copy()
        try:
            sig = float(rng.uniform(0.015, 0.045))
            Xtry[involved] += sig * rng.standard_normal((len(involved), d))
            # tiny global jitter
            Xtry += rng.standard_normal((n, d)) * 0.002
            Xtry /= np.maximum(np.linalg.norm(Xtry, axis=1, keepdims=True), 1e-12)
        except Exception:
            pass
        beta = float(rng.choice([150.0, 300.0, 500.0]))
        lam = float(rng.uniform(8.0, 40.0))
        # exempt LSE
        try:
            Xsm = _lse_exempt_refine(Xtry, exempt, beta=beta, delta=delta, m_ref=mcur, lam=lam, maxiter=90)
        except Exception:
            continue
        # masked LP then full LP
        t_use = time.time()
        rem = HARD - 14 - (time.time() - t0)
        if rem < 3.0:
            break
        try:
            Xp1, _ = _lp_polish_masked(Xsm, time.time() + min(0.9, rem * 0.35), exclude=exempt)
        except Exception:
            Xp1 = Xsm
        try:
            rem2 = HARD - 14 - (time.time() - t0)
            Xp2, vp2 = _lp_polish_masked(Xp1, time.time() + min(1.1, max(0.5, rem2 * 0.4)), exclude=None)
        except Exception:
            continue
        try:
            vp2 = _max_cos(Xp2)
        except Exception:
            continue
        fp = _fingerprint(Xp2)
        dist = _fp_dist(fp, base_fp)
        is_new = dist > 4.0  # different contact graph
        if vp2 < bestV - 1e-12:
            bestX, bestV = Xp2.copy(), vp2
            base_fp = fp
            refresh_duals(bestX)
            # keep pool fresh: clear far worse entries
            if len(div_pool) > 8:
                div_pool = div_pool[-8:]
        elif is_new and vp2 < bestV + 0.004:
            # keep diverse candidate even if worse (different basin)
            div_pool.append((Xp2.copy(), vp2))
            if len(div_pool) > 10:
                # drop worst
                div_pool.sort(key=lambda z: z[1])
                div_pool = div_pool[:10]
            # also occasionally refresh duals from diverse point to get new exempt sets
            if it % 6 == 0:
                refresh_duals(Xp2)
        # periodic dual refresh from best
        if it % 7 == 0:
            refresh_duals(bestX)
    # final intensive polish of champion
    try:
        rem = HARD - (time.time() - t0) - 0.8
        if rem > 1.2:
            bx, bv = _lp_polish_masked(bestX, time.time() + min(rem, 8.0), exclude=None)
            if bv < bestV:
                bestX, bestV = bx, bv
    except Exception:
        pass
    bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    bestX = bestX / np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
