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
    Iall = iu[0]
    Jall = iu[1]

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 200:
                break
            if radius < 1e-14:
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
            I = iu[0][sel]
            J = iu[1][sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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

    def refine_single(Xrest, c_init, beta=3000.0, maxiter=120):
        # minimize LSE max dot to rest, 8-dim problem
        Xr = Xrest  # (m,d)
        def fun(f):
            nn = float(np.linalg.norm(f))
            if nn < 1e-12:
                return 1e3, np.zeros_like(f)
            y = f / nn
            s = Xr @ y  # (m,)
            m_ = float(s.max())
            e = np.exp(beta * (s - m_))
            ss = float(e.sum())
            fval = m_ + np.log(ss) / beta
            w = e / ss
            gy = Xr.T @ w  # (d,)
            # dy/df = (I - yy')/nn
            g = (gy - (gy @ y) * y) / nn
            return float(fval), g
        try:
            r = minimize(fun, c_init.copy(), jac=True, method="L-BFGS-B",
                         options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-14})
            c = r.x
            c = c / np.linalg.norm(c)
            # exact max
            v = float((Xr @ c).max())
            return c, v
        except Exception:
            v = float((Xr @ c_init).max())
            return c_init / np.linalg.norm(c_init), v

    def greedy_reinsert(X, rng, n_try=20000, n_points=8, n_seeds=4):
        X = X.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        cur = float(rowmax.max())
        order = np.argsort(-rowmax)
        targets = order[:n_points].copy()
        for idx in targets:
            if time.time() > DEADLINE - 6.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xrest = X[mask]
            # rest-rest max
            Grest = float(G[np.ix_(mask, mask)].max()) if n > 2 else -2.0
            if Grest >= cur - 1e-12 and True:
                # still worth trying only if we can beat cur
                pass
            C = rng.standard_normal((n_try, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            S = Xrest @ C.T
            bv = S.max(axis=0)
            # take best seeds
            k = min(n_seeds, n_try)
            part = np.argpartition(bv, k - 1)[:k]
            # order them
            part = part[np.argsort(bv[part])]
            best_c = None
            best_v = float(bv[part[0]])
            best_global = max(Grest, best_v)
            # refine each seed, keep best
            refined_best_c = C[part[0]]
            refined_best_v = best_v
            for j in part:
                c0 = C[j]
                # quick skip if raw value much worse than current best refined
                if float(bv[j]) > refined_best_v + 0.02:
                    continue
                cc, vv = refine_single(Xrest, c0, beta=4000.0, maxiter=80)
                if vv < refined_best_v:
                    refined_best_v = vv
                    refined_best_c = cc
            new_global = Grest if Grest > refined_best_v else refined_best_v
            if new_global < cur - 1e-12:
                X[idx] = refined_best_c
                cur = float(new_global)
                g = X @ X[idx]
                G[idx, :] = g
                G[:, idx] = g
                G[idx, idx] = -2.0
        return X, cur

    # quick initial squeeze
    if DEADLINE - time.time() > 6:
        Xp, Vp = lp_polish(Xbest, seconds=2.5, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    trial = 0
    sigmas_cycle = [0.006, 0.015, 0.030, 0.010, 0.050, 0.008]
    while time.time() < DEADLINE - 12.0 and trial < 80:
        rem = DEADLINE - 12.0 - time.time()
        if rem < 2.0:
            break
        trial += 1
        rng = np.random.default_rng(1000 + trial * 7919)
        mode = trial % 6
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        rowmax0 = G0.max(axis=1)
        order0 = np.argsort(-rowmax0)
        sigma = sigmas_cycle[trial % len(sigmas_cycle)]
        if mode == 0 or mode == 5:
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
        elif mode == 1:
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            # extra kick worst 10
            worst = order0[:10]
            Xtry[worst] += rng.standard_normal((10, d)) * sigma * 1.5
        elif mode == 2:
            k = 8
            worst = order0[:k]
            Xtry = Xbest.copy()
            Xtry[worst] += 0.15 * rng.standard_normal((k, d))
            Xtry += 0.005 * rng.standard_normal((n, d))
        elif mode == 3:
            k = 6
            worst = order0[:k]
            Xtry = Xbest.copy()
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            # interpolate toward random to keep some memory
            alpha = 0.7 + 0.3 * rng.random()
            Xtry[worst] = (1 - alpha) * Xtry[worst] + alpha * R
            Xtry += 0.004 * rng.standard_normal((n, d))
        else:
            Xtry = Xbest + 0.02 * rng.standard_normal((n, d))
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # choose annealing schedule
        if trial % 3 == 0:
            betas = [1500.0, 12000.0, 80000.0]
            mis = [120, 150, 150]
        elif trial % 3 == 1:
            betas = [3000.0, 25000.0]
            mis = [150, 200]
        else:
            betas = [800.0, 6000.0, 40000.0]
            mis = [100, 120, 150]
        Xtry = lbfgs_anneal(Xtry, betas, mis)
        v = _max_cos(Xtry)
        # greedy if within window
        if v < Vbest + 0.012:
            nt = 20000 if (trial % 2 == 0) else 12000
            Xtry2, v2 = greedy_reinsert(Xtry, rng, n_try=nt, n_points=8, n_seeds=4)
            # keep whichever better (greedy is monotonic)
            Xtry, v = Xtry2, _max_cos(Xtry2)
        promising = (v < Vbest + 0.004) or (trial % 4 == 0 and v < Vbest + 0.010)
        if promising:
            secs = min(5.0, DEADLINE - 12.0 - time.time() - 1.0)
            if secs > 0.8:
                Xp, Vp = lp_polish(Xtry, seconds=secs, radius_init=3e-4)
                Xtry, v = Xp, Vp
        if v < Vbest - 1e-12:
            Vbest = v
            Xbest = Xtry.copy()

    # final squeeze
    rem = DEADLINE - time.time() - 1.0
    if rem > 1.5:
        Xbest, Vbest = lp_polish(Xbest, seconds=min(rem, 9.0), radius_init=2e-4)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
