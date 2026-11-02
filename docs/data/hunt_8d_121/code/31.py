import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 171.0
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

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
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
            it += 1
            if it > 120 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
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
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def make_global_fun(beta):
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

    def lbfgs_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 8.0:
                break
            fun = make_global_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def subset_refine(Yrest, F0, beta, maxiter):
        k = F0.shape[0]
        if k > 1:
            iu_k = np.triu_indices(k, 1)
        else:
            iu_k = None
        b = float(beta)
        def fun(flat):
            F = flat.reshape(k, d)
            nr = np.linalg.norm(F, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Ys = F / nr
            S = Ys @ Yrest.T
            if k > 1:
                Gss = Ys @ Ys.T
                c_ss = Gss[iu_k]
                cm = max(float(S.max()), float(c_ss.max()))
                E1 = np.exp(b * (S - cm))
                E2 = np.exp(b * (c_ss - cm))
                s = float(E1.sum() + E2.sum())
                f = cm + np.log(s) / b
                W1 = E1 / s
                Wss = np.zeros((k, k))
                Wss[iu_k] = (E2 / s)
                Wss = Wss + Wss.T
                GY = W1 @ Yrest + Wss @ Ys
            else:
                cm = float(S.max())
                E1 = np.exp(b * (S - cm))
                s = float(E1.sum())
                f = cm + np.log(s) / b
                W1 = E1 / s
                GY = W1 @ Yrest
            dot = (GY * Ys).sum(axis=1, keepdims=True)
            GF = (GY - dot * Ys) / nr
            return float(f), GF.ravel()
        try:
            r = minimize(fun, F0.ravel().copy(), jac=True, method="L-BFGS-B",
                         options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-14})
            F = r.x.reshape(k, d)
            F = F / np.linalg.norm(F, axis=1, keepdims=True)
            return F
        except Exception:
            return F0 / np.linalg.norm(F0, axis=1, keepdims=True)

    def void_greedy_reinsert(Xrest, k, rng, n_try=20000):
        # sequentially find k voids greedy
        Yrest_list = [Xrest.copy()]
        new_pts = []
        for _ in range(k):
            Yc = np.vstack([Yrest_list[0]] + [p[None, :] for p in new_pts]) if new_pts else Yrest_list[0]
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = Yc @ C.T
            bv = S.max(axis=0)
            j = int(np.argmin(bv))
            c0 = C[j]
            F1 = subset_refine(Yc, c0[None, :], beta=8000.0, maxiter=50)
            new_pts.append(F1[0] / np.linalg.norm(F1[0]))
        return np.array(new_pts)

    # initial quick polish
    if time.time() < DEADLINE - 10:
        Xp, Vp = lp_polish(Xbest, seconds=2.0, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    trial = 0
    while time.time() < DEADLINE - 11.0 and trial < 90:
        trial += 1
        rem = DEADLINE - 11.0 - time.time()
        if rem < 2.0:
            break
        rng = np.random.default_rng(1000 + trial * 7919)
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        rowmax0 = G0.max(axis=1)
        order0 = np.argsort(-rowmax0)
        mode = trial % 7
        Xtry = None
        try:
            if mode == 0:
                sigma = 0.18 if (trial % 2 == 0) else 0.30
                Xtry = Xbest + sigma * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [2000.0, 20000.0], [80, 110])
            elif mode == 1:
                k = 6 if (trial % 2 == 0) else 8
                worst = order0[:k].copy()
                Xtry = Xbest.copy()
                R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                Xtry[worst] = R
                Xtry += 0.004 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [2500.0, 25000.0], [80, 110])
            elif mode == 2:
                # cap cluster knockout
                u = rng.standard_normal(d); u /= np.linalg.norm(u)
                proj = Xbest @ u
                idx = np.argsort(-proj)[:8]
                Xtry = Xbest.copy()
                R = rng.standard_normal((8, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                Xtry[idx] = R
                Xtry += 0.003 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [1500.0, 12000.0, 50000.0], [60, 80, 80])
            elif mode == 3:
                # multi-void greedy reinsert of worst k
                k = 5
                worst = order0[:k].copy()
                mask = np.ones(n, dtype=bool); mask[worst] = False
                Yrest = Xbest[mask]
                NP = void_greedy_reinsert(Yrest, k, rng, n_try=15000)
                # joint refine
                for beta, mi in [(8000.0, 60), (30000.0, 60)]:
                    if time.time() > DEADLINE - 8.0:
                        break
                    NP = subset_refine(Yrest, NP, beta=beta, maxiter=mi)
                Xtry = Xbest.copy(); Xtry[worst] = NP
                Xtry = lbfgs_anneal(Xtry, [8000.0, 40000.0], [60, 80])
            elif mode == 4:
                # pure random restart, different basin
                Xtry = rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [1000.0, 8000.0, 30000.0], [70, 80, 90])
            elif mode == 5:
                # overload-targeted shake: shake top-degree + neighbors
                thr = Vbest - 5e-4
                Gc = Xbest @ Xbest.T
                deg = ((Gc > thr).sum(axis=1) - 1).astype(int)
                od = np.argsort(-deg)[:10]
                Xtry = Xbest + 0.01 * rng.standard_normal((n, d))
                Xtry[od] += 0.12 * rng.standard_normal((10, d))
                # also shake low-degree points strongly
                ud = np.argsort(deg)[:6]
                R = rng.standard_normal((6, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
                # mix: half random, half shake
                Xtry[ud[:3]] = R[:3]
                Xtry[ud[3:]] += 0.10 * rng.standard_normal((3, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [3000.0, 25000.0], [80, 110])
            else:
                sigma = 0.07
                Xtry = Xbest + sigma * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [1200.0, 10000.0, 60000.0], [60, 80, 70])
        except Exception:
            continue
        if Xtry is None:
            continue
        v = _max_cos(Xtry)
        if v < Vbest - 1e-12:
            # polish immediately
            Xtry2, v2 = lp_polish(Xtry, seconds=1.2, radius_init=2e-4)
            if v2 < v:
                Xtry, v = Xtry2, v2
            Xbest, Vbest = Xtry, v
        elif v < Vbest + 0.002 and time.time() < DEADLINE - 20:
            # promising: quick LP to see if it drops below
            Xtry2, v2 = lp_polish(Xtry, seconds=0.7, radius_init=1.5e-4)
            if v2 < Vbest - 1e-12:
                Xbest, Vbest = Xtry2, v2

    # final polish with remaining budget
    rem = DEADLINE - time.time()
    if rem > 2.0:
        try:
            Xp, Vp = lp_polish(Xbest, seconds=min(rem - 1.0, 6.0), radius_init=2e-4)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        except Exception:
            pass
    # final high-beta touch if time remains
    rem = DEADLINE - time.time()
    if rem > 4.0:
        try:
            Xn = lbfgs_anneal(Xbest, [50000.0], [60])
            vn = _max_cos(Xn)
            if vn < Vbest:
                Xbest, Vbest = Xn, vn
                rem2 = DEADLINE - time.time()
                if rem2 > 2.0:
                    Xp2, Vp2 = lp_polish(Xbest, seconds=min(rem2 - 1.0, 3.0), radius_init=1e-4)
                    if Vp2 < Vbest:
                        Xbest, Vbest = Xp2, Vp2
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
