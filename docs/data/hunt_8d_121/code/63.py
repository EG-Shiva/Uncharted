import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    rng = np.random.default_rng(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def make_full_fun(beta):
        b = float(beta)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def anneal_full(Xinit, betas, iters):
        Xf = Xinit.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 22.0:
                break
            fun = make_full_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*5)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    # ---- antipodal symmetric subspace: 1 pole + 60 + 60 antipodes ----
    def anneal_antipodal(Rinit, betas, iters):
        # Rinit (61,d)
        Rf = Rinit.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 22.0:
                break
            b = float(beta)
            def fun(flat):
                R = flat.reshape(61, d)
                nr = np.linalg.norm(R, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Q = R / nr
                Xf = np.empty((n, d))
                Xf[0] = Q[0]
                Xf[1:61] = Q[1:]
                Xf[61:] = -Q[1:]
                G = Xf @ Xf.T
                c = G[iu]
                m = float(c.max())
                e = np.exp(b * (c - m))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(m), np.zeros_like(flat)
                f = m + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Xf
                # combine
                GQ = np.empty((61, d))
                GQ[0] = GY[0]
                GQ[1:] = GY[1:61] - GY[61:]
                # to R
                dot = (GQ * Q).sum(axis=1, keepdims=True)
                GR = (GQ - dot * Q) / nr
                return float(f), GR.ravel()
            try:
                res = minimize(fun, Rf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*5)})
                Rf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        R = Rf.reshape(61, d)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Xf = np.empty((n, d))
        Xf[0] = R[0]; Xf[1:61] = R[1:]; Xf[61:] = -R[1:]
        return Xf

    # ---- 11x11 cyclic subspace ----
    def build_P(freqs):
        # freqs length 4
        P1 = np.eye(8)
        R = np.zeros((8, 8))
        for k, f in enumerate(freqs):
            th = 2.0 * np.pi * float(f) / 11.0
            c, s_ = np.cos(th), np.sin(th)
            R[2*k:2*k+2, 2*k:2*k+2] = np.array([[c, -s_], [s_, c]])
        Ps = [np.eye(8)]
        for j in range(1, 11):
            Ps.append(Ps[-1] @ R)
        return Ps

    def anneal_cyclic(Ginit, Ps, betas, iters):
        Gf = Ginit.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 22.0:
                break
            b = float(beta)
            def fun(flat):
                Gr = flat.reshape(11, d)
                nr = np.linalg.norm(Gr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Q = Gr / nr
                Xf = np.empty((121, d))
                for a in range(11):
                    qa = Q[a]
                    for j in range(11):
                        Xf[a*11+j] = qa @ Ps[j].T
                G = Xf @ Xf.T
                c = G[iu]
                m = float(c.max())
                e = np.exp(b * (c - m))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(m), np.zeros_like(flat)
                f = m + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Xf  # (121,8)
                GQ = np.zeros((11, d))
                for a in range(11):
                    acc = np.zeros(d)
                    for j in range(11):
                        acc += GY[a*11+j] @ Ps[j]
                    GQ[a] = acc
                dot = (GQ * Q).sum(axis=1, keepdims=True)
                GR = (GQ - dot * Q) / nr
                return float(f), GR.ravel()
            try:
                res = minimize(fun, Gf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi*6)})
                Gf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Gr = Gf.reshape(11, d)
        Gr /= np.linalg.norm(Gr, axis=1, keepdims=True)
        Xf = np.empty((121, d))
        for a in range(11):
            for j in range(11):
                Xf[a*11+j] = Gr[a] @ Ps[j].T
        # renormalize (already unit)
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        return Xf

    def greedy_hole_fill(Yrest, k, rng_, pool=18000):
        m = Yrest.shape[0]
        C = rng_.standard_normal((pool, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        Smax = (Yrest @ C.T).max(axis=0)  # (pool,)
        chosen = []
        cur = Smax.copy()
        avail = np.ones(pool, dtype=bool)
        for _ in range(k):
            # pick min among avail
            curm = np.where(avail, cur, np.inf)
            j = int(np.argmin(curm))
            avail[j] = False
            chosen.append(C[j])
            # update cur with new point
            dj = C @ C[j]
            cur = np.maximum(cur, dj) if False else cur  # not needed for rest-only? need inter-new
            # for next picks, penalize closeness to chosen: update cur = max(cur, C@chosen)
            # we maintain inter-new penalty:
            cur = np.maximum(cur, (C @ C[j]))
            # Note: first pick uses rest-only; later picks include inter-new automatically after update?
            # fix: recompute rest part already in cur init; keep as max
        # Actually above mixes rest and inter; do proper greedy: first pick rest-only, then update
        # Re-do correctly with sequential recompute for clarity (k small):
        chosen2 = []
        for _ in range(k):
            if chosen2:
                Yc = np.vstack([Yrest] + [p[None, :] for p in chosen2])
            else:
                Yc = Yrest
            bv = (Yc @ C.T).max(axis=0)
            bv[~avail] = np.inf
            # need avail relative to already chosen? C pool still; chosen points are from C so avail excludes them
            j = int(np.argmin(bv))
            avail[j] = False
            chosen2.append(C[j].copy())
        return np.array(chosen2)

    def lp_polish(x, seconds, radius_init=2e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
            it += 1
            if it > 40 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 3e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2000:
                kth = np.sort(cf)[-2000]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 2200:
                    idxp = np.argpartition(-cf, 2200)[:2200]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 2200
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
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
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.01)
            else:
                radius *= 0.5
        return x, best

    # ---- Stage A: symmetric antipodal tries ----
    try:
        if time.time() < DEADLINE - 60:
            R0 = rng.standard_normal((61, d)); R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
            Xs = anneal_antipodal(R0, [2500.0, 12000.0, 40000.0], [70, 70, 70])
            vs = _max_cos(Xs)
            # release symmetry
            Xr = anneal_full(Xs + 0.002 * rng.standard_normal((n, d)), [12000.0, 45000.0], [60, 70])
            # renormalize inside
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            vr = _max_cos(Xr)
            if vr < Vbest - 1e-12:
                Xbest, Vbest = Xr, vr
            elif vs < Vbest - 1e-12:
                Xbest, Vbest = Xs, vs
    except Exception:
        pass

    # ---- Stage B: cyclic 11x11 tries ----
    try:
        if time.time() < DEADLINE - 55:
            for freqs in ([1, 2, 3, 4], [1, 2, 3, 5]):
                if time.time() > DEADLINE - 45:
                    break
                Ps = build_P(freqs)
                G0 = rng.standard_normal((11, d)); G0 /= np.linalg.norm(G0, axis=1, keepdims=True)
                Xc = anneal_cyclic(G0, Ps, [3000.0, 15000.0], [80, 80])
                vc = _max_cos(Xc)
                # release
                Xcr = anneal_full(Xc + 0.003 * rng.standard_normal((n, d)), [8000.0, 35000.0], [60, 60])
                Xcr /= np.linalg.norm(Xcr, axis=1, keepdims=True)
                vcr = _max_cos(Xcr)
                if vcr < Vbest - 1e-12:
                    Xbest, Vbest = Xcr, vcr
                if vc < Vbest - 1e-12:
                    Xbest, Vbest = Xc, vc
                # only one freq set if time tight
                break
    except Exception:
        pass

    # ---- Stage C: aggressive knockout regrowth ----
    trial = 0
    while time.time() < DEADLINE - 24.0 and trial < 24:
        trial += 1
        rem = DEADLINE - 24.0 - time.time()
        if rem < 2.5:
            break
        rgt = np.random.default_rng(1000 + trial * 6137)
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mode = trial % 3
        try:
            if mode == 0:
                K = 30 if (trial % 2 == 0) else 24
                u = rgt.standard_normal(d); u /= np.linalg.norm(u)
                proj = Xbest @ u
                idx = np.argsort(-proj)[:K]
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Yrest = Xbest[mask]
                Rnew = rgt.standard_normal((K, d)); Rnew /= np.linalg.norm(Rnew, axis=1, keepdims=True)
                Xtry = np.empty((n, d)); Xtry[mask] = Yrest; Xtry[idx] = Rnew
                Xtry += 0.002 * rgt.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = anneal_full(Xtry, [2500.0, 25000.0], [70, 90])
            elif mode == 1:
                K = 22
                worst = order[:K].copy()
                mask = np.ones(n, dtype=bool); mask[worst] = False
                Yrest = Xbest[mask]
                NP = greedy_hole_fill(Yrest, K, rgt, pool=15000)
                Xtry = Xbest.copy(); Xtry[worst] = NP
                Xtry += 0.0015 * rgt.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = anneal_full(Xtry, [3000.0, 35000.0], [70, 90])
            else:
                K = 34
                idx = rgt.choice(n, size=K, replace=False)
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Yrest = Xbest[mask]
                Rnew = rgt.standard_normal((K, d)); Rnew /= np.linalg.norm(Rnew, axis=1, keepdims=True)
                Xtry = np.empty((n, d)); Xtry[mask] = Yrest; Xtry[idx] = Rnew
                # shake rest a bit to allow topology shift
                Xtry[mask] += 0.02 * rgt.standard_normal((n - K, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = anneal_full(Xtry, [2000.0, 12000.0, 45000.0], [50, 60, 60])
        except Exception:
            continue
        v = _max_cos(Xtry)
        if v < Vbest - 1e-12:
            Xbest, Vbest = Xtry, v

    # ---- final squeeze ----
    try:
        left = DEADLINE - 1.0 - time.time()
        if left > 3.0:
            Xp, Vp = lp_polish(Xbest, seconds=min(left - 0.5, 18.0), radius_init=2e-4)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
    except Exception:
        pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
