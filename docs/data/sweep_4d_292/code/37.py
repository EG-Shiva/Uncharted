import time
import numpy as np
from scipy.optimize import least_squares, lsq_linear, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X0):
    n = X0.shape[0]
    B = np.empty((n, 4, 3), dtype=np.float64)
    e1 = np.array([1.0, 0, 0, 0])
    I4 = np.eye(4)
    for i in range(n):
        x = X0[i]
        s = 1.0 if x[0] >= 0 else -1.0
        v = x + s * e1
        nv = float(v @ v)
        if nv < 1e-14:
            # x ~ -s*e1, pick another axis
            v = x.copy()
            v[1] += 1.0
            nv = float(v @ v)
        H = I4 - (2.0 / nv) * np.outer(v, v)
        B[i] = H[:, 1:4]
    return B

def _top_pairs(X, k=1200):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    order = np.argsort(-c)  # descending
    top = order[:k]
    I = iu[0][top].astype(np.int64)
    J = iu[1][top].astype(np.int64)
    return I, J, c, iu

def _equalize_once(X0, B, I, J, t_init, max_nfev=40):
    n = X0.shape[0]
    p = len(I)
    Nv = n * 3
    # closures
    def fun(z):
        y = z[:-1]
        t = z[-1]
        Ym = y.reshape(n, 3)
        # D = B @ ym
        # einsum
        D = np.einsum('nij,nj->ni', B, Ym)
        U = X0 + D
        nr = np.linalg.norm(U, axis=1)
        nr[nr < 1e-12] = 1e-12
        Xn = U / nr[:, None]
        c = np.einsum('pi,pi->p', Xn[I], Xn[J])
        return c - t
    def jac(z):
        y = z[:-1]
        Ym = y.reshape(n, 3)
        D = np.einsum('nij,nj->ni', B, Ym)
        U = X0 + D
        nr = np.linalg.norm(U, axis=1)
        nr_s = nr.copy()
        nr_s[nr_s < 1e-12] = 1e-12
        Xn = U / nr_s[:, None]
        c = np.einsum('pi,pi->p', Xn[I], Xn[J])
        XI = Xn[I]
        XJ = Xn[J]
        nI = nr_s[I]
        nJ = nr_s[J]
        VI = (XJ - c[:, None] * XI) / nI[:, None]
        VJ = (XI - c[:, None] * XJ) / nJ[:, None]
        BI = B[I]  # p,4,3
        BJ = B[J]
        GI = np.einsum('pij,pi->pj', BI, VI)
        GJ = np.einsum('pij,pi->pj', BJ, VJ)
        Jm = np.zeros((p, Nv + 1), dtype=np.float64)
        # scatter via loop (p~1200)
        for k in range(p):
            ii = int(I[k]); jj = int(J[k])
            Jm[k, ii*3:(ii+1)*3] = GI[k]
            Jm[k, jj*3:(jj+1)*3] += GJ[k]
        Jm[:, -1] = -1.0
        return Jm
    z0 = np.zeros(Nv + 1, dtype=np.float64)
    z0[-1] = t_init
    try:
        res = least_squares(fun, z0, jac=jac, method='lm',
                            xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=max_nfev, verbose=0)
    except Exception:
        return None
    return res

def _line_search(X0, B, y_star, cur_best):
    n = X0.shape[0]
    Ym = y_star.reshape(n, 3)
    D = np.einsum('nij,nj->ni', B, Ym)
    if not np.all(np.isfinite(D)):
        return X0, cur_best, 0.0
    dn = float(np.linalg.norm(D))
    if dn < 1e-16:
        return X0, cur_best, 0.0
    def val_at(a):
        U = X0 + a * D
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        Xn = U / nr
        return _max_cos(Xn)
    alphas = np.linspace(0.0, 2.0, 41)
    best_a = 0.0
    best_v = cur_best
    # evaluate
    for a in alphas[1:]:
        try:
            v = val_at(float(a))
        except Exception:
            continue
        if v < best_v:
            best_v = v
            best_a = float(a)
    if best_a == 0.0:
        return X0, cur_best, 0.0
    # refine around best_a 3 zooms
    lo = max(0.0, best_a - (2.0/40))
    hi = best_a + (2.0/40)
    for _ in range(4):
        grid = np.linspace(lo, hi, 21)
        bb_a = best_a; bb_v = best_v
        for a in grid:
            v = val_at(float(a))
            if v < bb_v:
                bb_v = v; bb_a = float(a)
        best_a, best_v = bb_a, bb_v
        w = (hi - lo) / 2.0 / 10.0
        lo = max(0.0, best_a - w*10/2 if False else best_a - (hi-lo)/2/2)
        # simpler: shrink interval
        span = (hi - lo) / 4.0
        lo = max(0.0, best_a - span)
        hi = best_a + span
    U = X0 + best_a * D
    nr = np.linalg.norm(U, axis=1, keepdims=True)
    np.maximum(nr, 1e-12, out=nr)
    Xn = U / nr
    return Xn, best_v, best_a

def _estimate_lambda(Xc, Bc, I, J):
    n = Xc.shape[0]
    p = len(I)
    Nv = n * 3
    try:
        XI = Xc[I]; XJ = Xc[J]
        BI = Bc[I]; BJ = Bc[J]
        GI = np.einsum('pij,pi->pj', BI, XJ)  # B_I^T x_J
        GJ = np.einsum('pij,pi->pj', BJ, XI)
        Grad = np.zeros((Nv, p), dtype=np.float64)
        for k in range(p):
            ii = int(I[k]); jj = int(J[k])
            Grad[ii*3:(ii+1)*3, k] = GI[k]
            Grad[jj*3:(jj+1)*3, k] += GJ[k]
        w = 10.0
        A = np.vstack([Grad, w*np.ones((1, p))])
        b = np.zeros(Nv+1); b[-1] = w
        res = lsq_linear(A, b, bounds=(0, np.inf), verbose=0)
        if not np.all(np.isfinite(res.x)):
            return None
        return res.x
    except Exception:
        return None

def _lp_polish_tight(x, seconds, r0=1e-5):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0.5:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0*radius, 1e-12)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        pp = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(pp), 2*d+1)
        ci = I[:,None]*d + np.arange(d)
        cj = J[:,None]*d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((pp,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pp,1))], axis=1).ravel()
        try:
            from scipy.sparse import coo_matrix
            A_ub = coo_matrix((vals,(rows,cols)), shape=(pp,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    LIMIT = 172.0
    try:
        Xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(np.float64)
        except Exception:
            rng0 = np.random.default_rng(0)
            Xb = rng0.standard_normal((292,4))
    if Xb.shape != (292,4) or not np.all(np.isfinite(Xb)):
        rng0 = np.random.default_rng(0)
        Xb = rng0.standard_normal((292,4))
    Xb = _norm(Xb)
    best_val = _max_cos(Xb)
    Xbest = Xb.copy()
    Xcur = Xb.copy()
    cur_val = best_val
    # active set: top 1200
    Iact, Jact, _, _ = _top_pairs(Xcur, k=1200)
    # equalization phase
    eq_deadline = t0 + 108.0
    outer = 0
    while time.time() < eq_deadline and outer < 6:
        outer += 1
        # expand to threshold 5e-7 if needed (already top1200 covers it)
        # check time for least_squares
        if time.time() + 8.0 > eq_deadline and outer > 1:
            break
        B = _tangent_bases(Xcur)
        # t init = current max among active (=global max)
        Gc = Xcur @ Xcur.T
        tact = float(np.max(Gc[Iact, Jact]))
        res = _equalize_once(Xcur, B, Iact, Jact, tact, max_nfev=40)
        if res is None or not np.all(np.isfinite(res.x)):
            break
        y_star = res.x[:-1]
        # line search on true max
        Xn, vn, abest = _line_search(Xcur, B, y_star, cur_val)
        if vn < cur_val - 1e-13 and np.all(np.isfinite(Xn)):
            Xcur = Xn
            cur_val = vn
            if vn < best_val:
                best_val = vn
                Xbest = Xn.copy()
            # check equalization spread
            Ga = Xcur @ Xcur.T
            ca = Ga[Iact, Jact]
            spread = float(ca.max() - ca.min())
            if spread < 1e-12:
                # converged active set; try to refresh with violators
                pass
            # KKT multipliers at new point (need new bases)
            try:
                Bn = _tangent_bases(Xcur)
                lam = _estimate_lambda(Xcur, Bn, Iact, Jact)
            except Exception:
                lam = None
            # build next active set: drop ~zero multipliers, add violators
            try:
                n = 292
                Gn = Xcur @ Xcur.T
                iu = np.triu_indices(n, 1)
                cn = Gn[iu]
                order = np.argsort(-cn)
                topI = iu[0][order[:1200]]
                topJ = iu[1][order[:1200]]
                if lam is not None:
                    mx = float(lam.max()) if lam.size else 0.0
                    thr = max(1e-9, 1e-7*mx) if mx>0 else 1e-9
                    keep_mask = lam > thr
                    kept_I = Iact[keep_mask]; kept_J = Jact[keep_mask]
                    # dropped set for exclusion
                    drop_set = set()
                    for a,b in zip(Iact[~keep_mask].tolist(), Jact[~keep_mask].tolist()):
                        if a>b: a,b=b,a
                        drop_set.add((a,b))
                    # union kept + global top excluding dropped
                    cand = {}
                    for a,b in zip(kept_I.tolist(), kept_J.tolist()):
                        if a>b: a,b=b,a
                        cand[(a,b)] = float(Gn[a,b])
                    for a,b in zip(topI.tolist(), topJ.tolist()):
                        aa,bb = (b,a) if a>b else (a,b)
                        if (aa,bb) in drop_set:
                            continue
                        if (aa,bb) not in cand:
                            cand[(aa,bb)] = float(Gn[aa,bb])
                    # if too few, pad with global top regardless
                    if len(cand) < 1200:
                        for idx in order[:1600]:
                            a=int(iu[0][idx]); b=int(iu[1][idx])
                            aa,bb=(b,a) if a>b else (a,b)
                            if (aa,bb) not in cand:
                                cand[(aa,bb)] = float(cn[idx])
                            if len(cand)>=1200:
                                break
                    # trim to 1200 tightest
                    items = sorted(cand.items(), key=lambda kv: -kv[1])[:1200]
                    Iact = np.array([k[0][0] for k in items], dtype=np.int64)
                    Jact = np.array([k[0][1] for k in items], dtype=np.int64)
                    # stationarity check: grad*lam small?
                    # if certified and spread tiny, stop
                    if spread < 5e-12 and np.sum(keep_mask) > 800:
                        # check residual norm roughly
                        pass
                else:
                    Iact = topI.astype(np.int64); Jact = topJ.astype(np.int64)
            except Exception:
                pass
            # continue to next equalization
            if time.time() > eq_deadline:
                break
        else:
            # no improvement: stop equalization, keep incumbent
            break
    # final tight LP with remaining time
    rem = LIMIT - (time.time() - t0) - 1.0
    rem = max(5.0, min(50.0, rem))
    try:
        Xbest = _lp_polish_tight(Xbest, rem, r0=1e-5)
    except Exception:
        pass
    Xbest = _norm(Xbest)
    if not np.all(np.isfinite(Xbest)):
        Xbest = _norm(Xb)
    return Xbest
