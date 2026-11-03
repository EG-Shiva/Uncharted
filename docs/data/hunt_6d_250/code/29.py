import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt_full(X0, beta, maxiter, maxcor=12):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _block_opt(R0, F, beta, maxiter):
    # jointly optimise block R (b,6) against frozen F (Nf,6)
    b, d = R0.shape
    Nf = F.shape[0]
    n = Nf + b
    Y0 = R0.copy()
    # precompute frozen-frozen irrelevant; build masks
    # relevant upper mask: pairs with at least one endpoint in R
    # i.e. all pairs except F-F
    def fg(Yflat):
        Y = Yflat.reshape(b, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        R = Y / nrm
        X = np.vstack([F, R])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = -2.0
        # max over relevant only: F-R and R-R
        # F-R block
        Gfr = G[:Nf, Nf:]
        m1 = float(np.max(Gfr)) if Gfr.size else -2.0
        if b > 1:
            Gr = G[Nf:, Nf:]
            np.fill_diagonal(Gr, -2.0)
            m2 = float(np.max(Gr))
        else:
            m2 = -2.0
        M = m1 if m1 > m2 else m2
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        # mask out F-F and diag
        E[:Nf, :Nf] = 0.0
        np.fill_diagonal(E, 0.0)
        # keep only upper then symmetrise via W_sym construction
        # S = sum upper relevant
        # use triu sum
        triu = np.triu(E, 1)
        S = float(np.sum(triu))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        # symmetric weight (each unordered weight in both triangles)
        # W_sym = (triu + triu.T)/S
        Wsym = (triu + triu.T) / S
        GX = Wsym @ X  # (n,6): for R rows this equals sum w*x_neigh
        # gradient only for R rows
        GR = GX[Nf:, :]
        dot = np.sum(GR * R, axis=1, keepdims=True)
        GY = (GR - dot * R) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": 10})
        Y = res.x.reshape(b, d)
    except Exception:
        Y = Y0
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _refine_single(x0, Xexist, beta=6000.0, iters=60, lr0=0.25):
    x = x0 / max(np.linalg.norm(x0), 1e-12)
    for it in range(iters):
        dots = Xexist @ x
        M = float(np.max(dots))
        D = beta * (dots - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.exp(D)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        g = W @ Xexist
        gt = g - float(g @ x) * x
        lr = lr0 / (1.0 + it * 0.06)
        x = x - lr * gt
        nn = np.linalg.norm(x)
        if nn < 1e-12:
            break
        x /= nn
    return x

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(15.0 * radius, 1e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            thr = np.sort(iu_c[sel])[-4000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _excision(X, k, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    flat = int(np.argmax(G))
    a = flat // n; b = flat % n
    thr = m - 2e-5
    Adj = G >= thr
    np.fill_diagonal(Adj, False)
    # BFS component containing a
    seen = np.zeros(n, dtype=bool)
    stack = [a]
    seen[a] = True
    while stack:
        u = stack.pop()
        nbrs = np.where(Adj[u])[0]
        for w in nbrs:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    comp = np.where(seen)[0]
    # degrees within subgraph
    if len(comp) >= k:
        # rank by degree + rowmax
        rowmax = G.max(axis=1)
        deg = Adj[comp][:, comp].sum(axis=1) if len(comp) else np.zeros(0)
        # score: degree primary, rowmax secondary
        order = np.lexsort((-rowmax[comp], -deg))
        chosen = comp[order[:k]]
        chosen = set(chosen.tolist())
        chosen.add(a); chosen.add(b)
        # if over k due to add, drop lowest
        cl = list(chosen)
        if len(cl) > k:
            # drop non-bottleneck with lowest rowmax
            rest = [c for c in cl if c not in (a, b)]
            rm = np.array([G[c].max() for c in rest])
            drop = np.argsort(rm)  # smallest first? we want keep largest, drop smallest
            # we added at most 2 extra, drop smallest
            ndrop = len(cl) - k
            todrop = set([rest[i] for i in drop[:ndrop]])
            cl = [c for c in cl if c not in todrop]
        return sorted(cl), (a, b)
    else:
        # expand to k using rowmax order
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        chosen = set(comp.tolist())
        chosen.add(a); chosen.add(b)
        for c in order:
            if len(chosen) >= k:
                break
            chosen.add(int(c))
        return sorted(chosen), (a, b)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t0)
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.load(fn)
            z = np.asarray(z, dtype=np.float64)
            if z.shape == (250, 6):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                if np.all(np.isfinite(z)):
                    cands.append(z)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        z = rng0.standard_normal((250, 6))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    n, d = 250, 6

    # time-boxed excision / regrowth: ~140s
    regrow_deadline = t0 + 140.0
    sizes = [25, 15, 40, 30, 20, 25, 40, 15]
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 123, 7, 5]
    trial = 0
    NS = len(sizes)
    while time.time() < regrow_deadline and rem() > 28:
        k = sizes[trial % NS]
        seed = seeds[trial % len(seeds)] + trial * 1009
        trial += 1
        rng = np.random.default_rng(seed)
        try:
            rem_idx, _ = _excision(best, k, rng)
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[rem_idx] = False
        F = best[keep_mask].copy()  # (Nf,6)
        Nf = F.shape[0]
        # joint regrowth
        Rlist = []
        ok = True
        for ii in range(k):
            if time.time() > regrow_deadline - 8:
                ok = False
                break
            # propose 8192 candidates
            try:
                C = rng.standard_normal((8192, d))
                C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            except Exception:
                ok = False
                break
            if Rlist:
                Rso = np.vstack(Rlist)
                Exist = np.vstack([F, Rso])
            else:
                Exist = F
            # scores = max cosine to Exist
            try:
                S = C @ Exist.T
                smax = S.max(axis=1)
                bi = int(np.argmin(smax))
                xbest = C[bi].copy()
            except Exception:
                xbest = C[0].copy()
            # refine 60 iters single-point
            try:
                xbest = _refine_single(xbest, Exist, beta=6000.0, iters=60, lr0=0.25)
            except Exception:
                pass
            Rlist.append(xbest)
            # every 5 insertions jointly re-optimise block
            if len(Rlist) % 5 == 0 or len(Rlist) == k:
                if len(Rlist) >= 3 and rem() > 20:
                    try:
                        Rb = np.vstack(Rlist)
                        # two quick escalating betas
                        Rb = _block_opt(Rb, F, 4000.0, 60)
                        Rb = _block_opt(Rb, F, 9000.0, 40)
                        Rlist = [Rb[i].copy() for i in range(Rb.shape[0])]
                    except Exception:
                        pass
        if not ok:
            continue
        try:
            Rfin = np.vstack(Rlist)
            Xnew = np.zeros((n, d))
            Xnew[keep_mask] = F
            # place regrown in removed slots
            Xnew[np.array(rem_idx)] = Rfin
            Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            # short collective polish: block then full
            # joint block once more
            try:
                Rb2 = _block_opt(Rfin, F, 9000.0, 50)
                Xnew[np.array(rem_idx)] = Rb2
                Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            except Exception:
                pass
            # collective smooth polish (cheap)
            if rem() > 25:
                try:
                    Xnew = _smooth_opt_full(Xnew, 6000.0, 80)
                except Exception:
                    pass
            v = _max_cos(Xnew)
            if v < best_val and np.all(np.isfinite(Xnew)):
                best = Xnew.copy()
                best_val = v
        except Exception:
            continue
        # early exit if time low
        if rem() < 28:
            break

    # final short collective polish with remaining time
    try:
        if rem() > 12:
            cand = _smooth_opt_full(best, 12000.0, 80)
            v = _max_cos(cand)
            if v < best_val:
                best = cand; best_val = v
    except Exception:
        pass
    try:
        if rem() > 8:
            deadline = time.time() + min(rem() - 2.0, 12.0)
            cand2 = _lp_polish(best, deadline)
            v2 = _max_cos(cand2)
            if v2 < best_val:
                best = cand2; best_val = v2
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
