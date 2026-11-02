import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

_IDX13 = (np.arange(13)[None, :] - np.arange(13)[:, None]) % 13  # [j,i]=(i-j)%13
_MFLAT = _IDX13.ravel()
_IU0, _IU1 = np.triu_indices(78, 1)

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _gen_sym(Sflat):
    S = np.asarray(Sflat, dtype=np.float64).reshape(6, 15)
    S13 = S[:, :13]
    St = S[:, 13:]
    nn = np.linalg.norm(S, axis=1)
    nn = np.maximum(nn, 1e-12)
    X13 = S13[:, _IDX13] / nn[:, None, None]  # (6,13,13)
    X13 = X13.reshape(78, 13)
    Xt = np.repeat(St, 13, axis=0) / np.repeat(nn, 13)[:, None]
    X = np.empty((78, 15), dtype=np.float64)
    X[:, :13] = X13
    X[:, 13:] = Xt
    return X

def _make_sym_fun(beta):
    iu0, iu1 = _IU0, _IU1
    idx = _IDX13
    mflat = _MFLAT
    def fun(sf):
        S = np.asarray(sf, dtype=np.float64).reshape(6, 15)
        nn = np.linalg.norm(S, axis=1)
        nn = np.maximum(nn, 1e-12)
        S13 = S[:, :13]
        St = S[:, 13:]
        X13 = S13[:, idx] / nn[:, None, None]
        Xt = np.repeat(St, 13, axis=0) / np.repeat(nn, 13)[:, None]
        X = np.empty((78, 15), dtype=np.float64)
        X[:, :13] = X13.reshape(78, 13)
        X[:, 13:] = Xt
        G = X @ X.T
        gv = G[iu0, iu1]
        m = float(gv.max())
        e = np.exp(beta * (gv - m))
        Ssum = float(e.sum())
        L = m + np.log(Ssum) / beta
        w = e / Ssum
        W = np.zeros((78, 78), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1)
        rep = np.repeat(nn, 13)
        Da = (Gr - dot[:, None] * X) / rep[:, None]  # dL/dV
        Da13 = Da[:, :13].reshape(6, 13 * 13)
        Dat = Da[:, 13:].reshape(6, 13, 2)
        DS = np.zeros((6, 15), dtype=np.float64)
        for k in range(6):
            DS[k, :13] = np.bincount(mflat, weights=Da13[k], minlength=13)
            DS[k, 13:] = Dat[k].sum(axis=0)
        return float(L), DS.ravel()
    return fun

def _lp_polish(X, deadline, r_init=0.008, r_max=0.03, cap=2500, max_it=40):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv)
    obj[-1] = 1.0
    it = 0
    while time.time() + 0.5 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        if int(np.count_nonzero(sel)) == 0:
            radius *= 0.5
            continue
        if int(np.count_nonzero(sel)) > cap:
            idxw = np.where(sel)[0]
            order = np.argsort(-c_all[idxw])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idxw[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def _full_refine(X0, betas, maxiter_each=200):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Xcur = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    xf = Xcur.ravel().copy()
    for beta in betas:
        def fun(g):
            Y = g.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            gv = G[iu0, iu1]
            m = float(gv.max())
            e = np.exp(beta * (gv - m))
            Ssum = float(e.sum())
            F = m + np.log(Ssum) / beta
            w = e / Ssum
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            Gr = W @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            GX = (Gr - dot * X) / nr
            return float(F), GX.ravel()
        try:
            res = minimize(fun, xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter_each, "ftol": 1e-14, "gtol": 1e-7})
            xf = res.x
        except Exception:
            break
    Xo = xf.reshape(n, d)
    Xo = Xo / np.linalg.norm(Xo, axis=1, keepdims=True)
    return Xo

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    rng = np.random.default_rng(0)
    # incumbent
    bestX = None; bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < bestV:
                bestV = v; bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((78, 15))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    # ---- 6x13 cyclic orbit search (strict cap ~55s) ----
    orb_deadline = min(t0 + 55.0, HARD - 90.0)
    if orb_deadline < t0 + 10:
        orb_deadline = t0 + 10
    best_orb_S = None; best_orb_v = 1e9; best_orb_X = None
    nseeds = 4
    betas_sched = [(100.0, 100), (300.0, 100), (800.0, 120)]
    for s in range(nseeds):
        if time.time() > orb_deadline:
            break
        if s == 1:
            # sparse signed weight-3 start
            S0 = rng.standard_normal((6, 15)) * 0.15
            for k in range(6):
                pos = rng.choice(13, size=3, replace=False)
                S0[k, pos] = rng.choice(np.array([-1.0, 1.0]), size=3)
                S0[k, 13 + rng.integers(0, 2)] = rng.choice(np.array([-1.0, 1.0]))
        elif s == 2:
            S0 = rng.standard_normal((6, 15))
            S0[:, 13:] *= 0.5
        else:
            S0 = rng.standard_normal((6, 15))
        # reject near-constant first-13 (would duplicate orbit points)
        for k in range(6):
            if float(np.std(S0[k, :13])) < 1e-3:
                S0[k, :13] += rng.standard_normal(13) * 0.5
        Sflat = S0.ravel().copy()
        # normalize rows to unit to fix scale
        S = Sflat.reshape(6, 15)
        S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        Sflat = S.ravel().copy()
        ok = True
        for (beta, mi) in betas_sched:
            if time.time() + 1.0 > orb_deadline:
                ok = False
                break
            fun = _make_sym_fun(beta)
            try:
                res = minimize(fun, Sflat, jac=True, method="L-BFGS-B",
                               options={"maxiter": mi, "ftol": 1e-12, "gtol": 1e-6})
                Sflat = res.x
                S = Sflat.reshape(6, 15)
                nr = np.linalg.norm(S, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                # avoid collapse
                if np.any(~np.isfinite(Sflat)):
                    ok = False
                    break
                S = S / nr
                Sflat = S.ravel().copy()
            except Exception:
                ok = False
                break
        if not ok:
            continue
        try:
            Xorb = _gen_sym(Sflat)
            # check duplicates (periodic seed)
            G = Xorb @ Xorb.T
            np.fill_diagonal(G, -2.0)
            dup = float(G.max())
            if dup > 0.999:
                continue
            v = dup
        except Exception:
            continue
        if v < best_orb_v:
            best_orb_v = v; best_orb_S = Sflat.copy(); best_orb_X = Xorb.copy()

    # if orbit basin not promising, return incumbent quickly
    if best_orb_X is None or best_orb_v > 0.19:
        # brief safety polish of incumbent if time allows, then return
        try:
            if time.time() + 4.0 < HARD:
                Xp, vp = _lp_polish(bestX, min(HARD - 1.0, time.time() + 6.0), r_init=0.002, max_it=15)
                if vp < bestV:
                    bestX = Xp; bestV = vp
        except Exception:
            pass
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        return bestX

    # ---- symmetry-breaking release ----
    try:
        Xr = best_orb_X + rng.standard_normal((78, 15)) * 1e-3
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        # full smooth refine with remaining budget
        if time.time() + 10 < HARD:
            Xr2 = _full_refine(Xr, betas=[150.0, 500.0], maxiter_each=180)
        else:
            Xr2 = Xr
        vr2 = _max_cos(Xr2)
        # trust-region LP under strict cap
        remain = HARD - time.time()
        if remain > 5:
            Xp, vp = _lp_polish(Xr2, HARD - 0.8, r_init=0.006, r_max=0.03, cap=2500, max_it=40)
        else:
            Xp, vp = Xr2, vr2
        if vp < bestV:
            bestX = Xp; bestV = vp
        else:
            # keep released even if slightly worse? No - keep incumbent.
            pass
    except Exception:
        pass

    # final: ensure we return the best seen; small final snap if champion improved
    try:
        if time.time() + 3.0 < HARD:
            # only polish champion briefly to avoid timeout
            Xp2, vp2 = _lp_polish(bestX, min(HARD - 0.5, time.time() + 5.0), r_init=0.001, max_it=10)
            if vp2 < bestV:
                bestX = Xp2
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isnan(bestX), 0.0, bestX)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
