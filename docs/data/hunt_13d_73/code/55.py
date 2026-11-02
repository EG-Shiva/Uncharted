import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

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
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-7)
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    n, d = 73, 13
    rng = np.random.default_rng(0)
    # load incumbent
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
    Xbest = Xrec.copy()
    best = float(best_file)
    THRESH = 0.2073372218913

    # --- Z73 cyclic family setup ---
    N = 73
    NM = 6
    NLAG = 36  # lags 1..36 (73 odd, lag l and 73-l same)
    # cos table C[l,m]
    L = np.arange(1, NLAG + 1)[:, None]  # 36x1
    M = np.arange(1, NM + 1)[None, :]    # 1x6
    CTAB = np.cos(2.0 * np.pi * L * M / float(N))  # 36x6

    def lag_max_from_w7(w):
        w = np.asarray(w, dtype=float)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 1e-15:
            return 1.0
        wn = w / s
        r2 = wn[:6]
        h2 = wn[6]
        lags = h2 + CTAB @ r2  # 36
        return float(lags.max())

    # differential evolution on simplex via box [0,1]^7 normalized
    x_de = None
    v_de = 1e9
    try:
        bounds = [(0.0, 1.0)] * 7
        # population ~30: popsize=4 gives 4*7=28
        res = differential_evolution(
            lag_max_from_w7, bounds, maxiter=400, popsize=4,
            seed=0, polish=False, workers=1, updating='deferred',
            tol=1e-9, mutation=(0.3, 0.9), recombination=0.85,
        )
        x_de = np.asarray(res.x, dtype=float)
        v_de = float(res.fun)
    except Exception:
        x_de = None
    # fallback random
    if x_de is None:
        x_de = rng.random(7)
        v_de = lag_max_from_w7(x_de)
    # Nelder-Mead minimax polishing
    try:
        r2 = minimize(lag_max_from_w7, x_de, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-13, "disp": False})
        if float(r2.fun) < v_de and np.all(np.isfinite(r2.x)):
            x_de = np.asarray(r2.x, dtype=float)
            v_de = float(r2.fun)
    except Exception:
        pass
    # normalize weights
    wsum = float(x_de.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        x_de = np.ones(7) / 7.0
        wsum = 1.0
    wn = x_de / wsum
    r2w = wn[:6].copy()
    h2w = float(wn[6])
    r_m = np.sqrt(np.maximum(r2w, 0.0))
    h = float(np.sqrt(max(h2w, 0.0)))
    if h < 1e-6:
        h = 1e-6
        # renormalize slightly
        tot = float((r_m ** 2).sum() + h * h)
        r_m /= np.sqrt(tot)
        h /= np.sqrt(tot)

    # expand orbit with fixed phases (phases cancel in score)
    rng_ph = np.random.default_rng(1)
    phi = rng_ph.uniform(0.0, 2.0 * np.pi, size=NM)
    Xcyc = np.zeros((N, d), dtype=float)
    for k in range(N):
        for mm in range(NM):
            ang = 2.0 * np.pi * (mm + 1) * k / float(N) + float(phi[mm])
            Xcyc[k, 2 * mm] = r_m[mm] * np.cos(ang)
            Xcyc[k, 2 * mm + 1] = r_m[mm] * np.sin(ang)
        Xcyc[k, 12] = h
    # exact renormalization
    Xcyc /= np.linalg.norm(Xcyc, axis=1, keepdims=True)
    v_cyc = _max_cos(Xcyc)

    iu0, iu1 = np.triu_indices(n, 1)

    def lse_lbfgs(X_init, alphas, iters_list, time_end):
        X0 = X_init.copy()
        Y = X0.copy()  # we optimize Y unconstrained, project each eval
        yf = Y.ravel().copy()
        for alpha, itn in zip(alphas, iters_list):
            if time.time() > time_end:
                break
            def fun_grad(yflat):
                Ym = yflat.reshape(n, d)
                nr = np.linalg.norm(Ym, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                T = Ym / nr
                G = T @ T.T
                dots = G[iu0, iu1]
                Mx = float(dots.max())
                e = np.exp(alpha * (dots - Mx))
                S = float(e.sum())
                loss = Mx + np.log(max(S, 1e-300)) / alpha
                p = e / max(S, 1e-300)
                # grad w.r.t T
                W = np.zeros((n, n), dtype=float)
                W[iu0, iu1] = p
                W[iu1, iu0] = p
                gT = W @ T
                dot = np.sum(gT * T, axis=1, keepdims=True)
                gY = (gT - dot * T) / nr
                return float(loss), gY.ravel().astype(float)
            try:
                res2 = minimize(fun_grad, yf, jac=True, method="L-BFGS-B",
                                options={"maxiter": int(itn), "ftol": 1e-12, "gtol": 1e-7, "maxcor": 10})
                yf = res2.x
            except Exception:
                break
        Yf = yf.reshape(n, d)
        Yf /= np.linalg.norm(Yf, axis=1, keepdims=True)
        return Yf

    # symmetry-breaking perturbations
    eps_list = [1e-4, 1e-3, 1e-2]
    for ei, eps in enumerate(eps_list):
        if time.time() > T_END - 40:
            break
        rng_e = np.random.default_rng(100 + ei)
        Xp = Xcyc + rng_e.standard_normal(size=(n, d)) * float(eps)
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # short full-space LSE-LBFGS
        try:
            tlim = min(T_END - 25, time.time() + 22)
            # smaller iters for speed
            Xp2 = lse_lbfgs(Xp, alphas=[80.0, 220.0, 550.0], iters_list=[40, 40, 40], time_end=tlim - 16)
        except Exception:
            Xp2 = Xp
        v2 = _max_cos(Xp2)
        # 15s LP polish budget (shortened if near end)
        lp_sec = 15.0
        if time.time() + lp_sec > T_END - 5:
            lp_sec = max(1.0, T_END - 5 - time.time())
        try:
            Xp3, v3 = _lp_polish(Xp2, seconds=lp_sec, t_end=T_END - 2)
        except Exception:
            Xp3, v3 = Xp2, v2
        # retain only if beats incumbent threshold and global best
        if np.all(np.isfinite(Xp3)) and v3 < best and v3 < THRESH + 1e-12:
            # require strictly better than best to avoid fallback issues
            if v3 < best:
                Xbest = Xp3.copy()
                best = float(v3)
        # also consider unpolished if polish failed
        if v2 < best and v2 < THRESH:
            if np.all(np.isfinite(Xp2)):
                Xbest = Xp2.copy()
                best = float(v2)

    # if cyclic family never beat threshold, keep incumbent (already in Xbest)
    # safety: ensure Xbest is best known
    if best > best_file:
        Xbest = Xrec.copy()
        best = float(best_file)
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
