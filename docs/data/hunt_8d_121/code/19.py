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
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng_global = np.random.default_rng(0)
    n, d = 121, 8

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
        X0 = rng_global.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 200 or radius < 1e-14:
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
            A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
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
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(beta * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / beta
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
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 5)})
                Xf = res.x
            except Exception:
                break
        X = Xf.reshape(n, d)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # subset (k points vs fixed rest) LSE
    def subset_refine(Yrest, F0, beta, maxiter):
        k = F0.shape[0]
        m = Yrest.shape[0]
        # precompute triu indices for kxk
        if k > 1:
            iu_k = np.triu_indices(k, 1)
        else:
            iu_k = None
        def fun(flat):
            F = flat.reshape(k, d)
            nr = np.linalg.norm(F, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Ys = F / nr
            S = Ys @ Yrest.T  # k x m
            if k > 1:
                Gss = Ys @ Ys.T
                c_ss = Gss[iu_k]
                c_all_max = max(float(S.max()), float(c_ss.max()))
                # stable exp
                E1 = np.exp(beta * (S - c_all_max))
                E2 = np.exp(beta * (c_ss - c_all_max))
                s = float(E1.sum() + E2.sum())
                f = c_all_max + np.log(s) / beta
                W1 = E1 / s  # k x m
                W2 = E2 / s
                # build symmetric Wss
                Wss = np.zeros((k, k))
                Wss[iu_k] = W2
                Wss = Wss + Wss.T
                GY = W1 @ Yrest + Wss @ Ys
            else:
                c_all_max = float(S.max())
                E1 = np.exp(beta * (S - c_all_max))
                s = float(E1.sum())
                f = c_all_max + np.log(s) / beta
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

    def greedy_multi(X, rng, n_try=16000, n_points=3, n_seeds=3):
        X = X.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        cur = float(rowmax.max())
        order = np.argsort(-rowmax)
        targets = order[:n_points].copy()
        # sequential single reinsertion with wide search
        for idx in targets:
            if time.time() > DEADLINE - 6.0:
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Xrest = X[mask]
            Grest = float((Xrest @ Xrest.T).max()) if n > 2 else -2.0
            # need off-diag max; diagonal is 1 so fix:
            Gr = Xrest @ Xrest.T
            np.fill_diagonal(Gr, -2.0)
            Grest = float(Gr.max())
            if Grest >= cur - 1e-12 and cur < Vbest + 0.002:
                # if rest already as bad as cur, still try but skip if hopeless
                pass
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = Xrest @ C.T  # m x t
            bv = S.max(axis=0)
            k = min(n_seeds, n_try)
            part = np.argpartition(bv, k - 1)[:k]
            part = part[np.argsort(bv[part])]
            best_c = C[part[0]]
            best_v = float(bv[part[0]])
            for j in part:
                c0 = C[j]
                if float(bv[j]) > best_v + 0.02:
                    continue
                # refine with subset k=1, two betas
                F0 = c0[None, :]
                F1 = subset_refine(Xrest, F0, beta=5000.0, maxiter=60)
                v1 = float((Xrest @ F1[0]).max())
                if v1 < best_v:
                    best_v = v1; best_c = F1[0]
                    # second stage higher beta
                    F2 = subset_refine(Xrest, F1, beta=20000.0, maxiter=60)
                    v2 = float((Xrest @ F2[0]).max())
                    if v2 < best_v:
                        best_v = v2; best_c = F2[0]
            new_global = Grest if Grest > best_v else best_v
            if new_global < cur - 1e-12:
                X[idx] = best_c / np.linalg.norm(best_c)
                cur = float(new_global)
                g = X @ X[idx]
                G[idx, :] = g; G[:, idx] = g; G[idx, idx] = -2.0
                rowmax = G.max(axis=1)
        # joint polish of the targets together
        if len(targets) >= 2 and time.time() < DEADLINE - 6.0:
            try:
                mask2 = np.ones(n, dtype=bool); mask2[targets] = False
                Yrest = X[mask2]
                F0 = X[targets].copy()
                for beta, mi in [(8000.0, 80), (30000.0, 80)]:
                    if time.time() > DEADLINE - 6.0:
                        break
                    F0 = subset_refine(Yrest, F0, beta=beta, maxiter=mi)
                Xn = X.copy(); Xn[targets] = F0
                v = _max_cos(Xn)
                if v < cur:
                    X = Xn; cur = v
            except Exception:
                pass
        return X, cur

    if DEADLINE - time.time() > 6:
        Xp, Vp = lp_polish(Xbest, seconds=2.5, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    trial = 0
    sigmas = [0.006, 0.015, 0.03, 0.01, 0.05, 0.008]
    while time.time() < DEADLINE - 12.0 and trial < 100:
        rem = DEADLINE - 12.0 - time.time()
        if rem < 2.0:
            break
        trial += 1
        rng = np.random.default_rng(2000 + trial * 7919)
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        rowmax0 = G0.max(axis=1)
        order0 = np.argsort(-rowmax0)
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        if mode == 0 or mode == 4:
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            if mode == 4:
                worst = order0[:8]
                Xtry[worst] += 0.06 * rng.standard_normal((8, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            if trial % 3 == 0:
                betas = [1500.0, 12000.0, 80000.0]; mis = [100, 130, 130]
            elif trial % 3 == 1:
                betas = [3000.0, 25000.0]; mis = [130, 170]
            else:
                betas = [800.0, 6000.0, 40000.0]; mis = [90, 110, 130]
            Xtry = lbfgs_anneal(Xtry, betas, mis)
            v = _max_cos(Xtry)
        elif mode == 1:
            # knock out 2 worst, randomize, anneal lightly then greedy
            k = 2
            worst = order0[:k].copy()
            Xtry = Xbest.copy()
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xtry[worst] = R
            Xtry += 0.004 * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry = lbfgs_anneal(Xtry, [2000.0, 20000.0], [100, 130])
            v = _max_cos(Xtry)
        elif mode == 2:
            # knock out 3 worst + greedy multi without full anneal first
            k = 3
            worst = order0[:k].copy()
            Xtry = Xbest.copy()
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            alpha = 0.85
            Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            # light anneal then greedy
            Xtry = lbfgs_anneal(Xtry, [1500.0, 15000.0], [80, 100])
            v = _max_cos(Xtry)
        else:
            # mode 3: direct greedy attack on current best with fresh rng
            Xtry = Xbest.copy()
            v = Vbest
        # greedy multi if within window
        if v < Vbest + 0.010 and time.time() < DEADLINE - 8.0:
            nt = 16000 if (trial % 2 == 0) else 10000
            np_pts = 3 if (trial % 2 == 0) else 2
            Xg, vg = greedy_multi(Xtry, rng, n_try=nt, n_points=np_pts, n_seeds=3)
            if vg < v:
                Xtry, v = Xg, _max_cos(Xg)
        promising = (v < Vbest + 0.003) or (trial % 5 == 0 and v < Vbest + 0.008)
        if promising:
            secs = min(4.0, DEADLINE - 12.0 - time.time() - 0.5)
            if secs > 0.7:
                Xp, Vp = lp_polish(Xtry, seconds=secs, radius_init=3e-4)
                Xtry, v = Xp, Vp
        if v < Vbest - 1e-12:
            Vbest = v
            Xbest = Xtry.copy()

    rem = DEADLINE - time.time() - 1.0
    if rem > 1.5:
        Xbest, Vbest = lp_polish(Xbest, seconds=min(rem, 9.0), radius_init=2e-4)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
