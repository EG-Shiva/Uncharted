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
    n, d = 121, 8
    rng0 = np.random.default_rng(0)
    np.random.seed(0)
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

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
        X0 = rng0.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
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

    def make_fun(beta):
        b = float(beta)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
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
            fun = make_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

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
            F /= np.linalg.norm(F, axis=1, keepdims=True)
            return F
        except Exception:
            return F0 / np.linalg.norm(F0, axis=1, keepdims=True)

    def contact_degrees(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        B = G >= m - tol
        deg = B.sum(axis=1)
        return deg, m

    def small_rotation(rng, theta):
        a = rng.standard_normal(d); b = rng.standard_normal(d)
        # orthonormalize
        a /= np.linalg.norm(a)
        b = b - (a @ b) * a
        nb = np.linalg.norm(b)
        if nb < 1e-12:
            b = rng.standard_normal(d); b = b - (a @ b) * a
            nb = np.linalg.norm(b) + 1e-18
        b /= nb
        c, s_ = np.cos(theta), np.sin(theta)
        R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s_ * (np.outer(a, b) - np.outer(b, a))
        return R

    def void_reinsert(Xrest, k, rng, n_pool=20000):
        m = Xrest.shape[0]
        C = rng.standard_normal((n_pool, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = Xrest @ C.T
        bv = S.max(axis=0)
        F = []
        Fmat = np.zeros((0, d))
        # greedy sequential maximin
        order_pool = np.argsort(bv)
        chosen = []
        # we will pick k points one by one scanning pool + refine
        for t in range(k):
            best_v = 1e9
            best_c = None
            # scan a subset of pool: best 2000 by rest-score plus random
            cand_idx = order_pool[:4000]
            # subsample 2000 for speed with diversity
            if len(cand_idx) > 2000:
                sel = rng.choice(len(cand_idx), 2000, replace=False)
                cand_idx = cand_idx[sel]
            for j in cand_idx:
                cj = C[j]
                v = float(Xrest @ cj .max()) if False else float((Xrest @ cj).max())
                if len(F) > 0:
                    v2 = float((Fmat @ cj).max())
                    if v2 > v:
                        v = v2
                # quick skip
                if v >= best_v:
                    continue
                best_v = v; best_c = cj
            if best_c is None:
                best_c = C[order_pool[0]]
                best_v = float((Xrest @ best_c).max())
            # local refine single point vs rest+F
            if len(F) > 0:
                YR = np.vstack([Xrest, Fmat])
            else:
                YR = Xrest
            F0 = best_c[None, :]
            F1 = subset_refine(YR, F0, beta=8000.0, maxiter=50)
            v1 = float((YR @ F1[0]).max())
            if v1 < best_v:
                best_c = F1[0]; best_v = v1
            F.append(best_c / np.linalg.norm(best_c))
            Fmat = np.vstack(F)
            # mask out pool candidates too close to chosen to diversify
            if t < k - 1:
                dots = C @ best_c
                order_pool = order_pool[dots[order_pool] < 0.97]
                if len(order_pool) < 500:
                    order_pool = np.argsort((C @ Fmat.T).max(axis=1))
        return np.array(F)

    # initial light polish
    if DEADLINE - time.time() > 8:
        Xp, Vp = lp_polish(Xbest, seconds=2.0, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    trial = 0
    while time.time() < DEADLINE - 14.0 and trial < 80:
        trial += 1
        rng = np.random.default_rng(1000 + trial * 6137)
        mode = trial % 4
        Xtry = None
        try:
            if mode == 0:
                # coherent cap rotation: preserves internal contacts, changes cross
                cdir = rng.standard_normal(d); cdir /= np.linalg.norm(cdir)
                proj = Xbest @ cdir
                s = int(rng.integers(12, 26))
                idx = np.argsort(-proj)[:s]
                theta = float(rng.uniform(0.015, 0.06)) * (1 if rng.random() < 0.5 else -1)
                R = small_rotation(rng, theta)
                Xtry = Xbest.copy()
                Xtry[idx] = (Xtry[idx] @ R.T)
                Xtry += 0.003 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [2000.0, 20000.0, 80000.0], [90, 110, 110])
            elif mode == 1:
                # hub knockout -> void reinsert (load transfer)
                deg, m = contact_degrees(Xbest, tol=3e-9)
                order = np.argsort(-deg)
                k = int(rng.choice([4, 5, 6]))
                hubs = order[:k]
                mask = np.ones(n, dtype=bool); mask[hubs] = False
                Xrest = Xbest[mask]
                F = void_reinsert(Xrest, k, rng, n_pool=16000)
                Xtry = Xbest.copy()
                Xtry[hubs] = F
                # joint refine of reinserted
                YR = Xtry[mask]
                F0 = Xtry[hubs].copy()
                for beta, mi in [(8000.0, 70), (30000.0, 70)]:
                    if time.time() > DEADLINE - 8.0:
                        break
                    F0 = subset_refine(YR, F0, beta=beta, maxiter=mi)
                Xtry[hubs] = F0
                Xtry += 0.002 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [2500.0, 25000.0], [100, 130])
            elif mode == 2:
                # kNN cluster coherent shift (common vector + noise)
                seed = int(rng.integers(0, n))
                dots = Xbest @ Xbest[seed]
                k = int(rng.integers(9, 18))
                cl = np.argsort(-dots)[:k]
                common = rng.standard_normal(d) * float(rng.uniform(0.02, 0.06))
                Xtry = Xbest.copy()
                Xtry[cl] += common[None, :]
                Xtry += 0.004 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                # rotate cluster slightly too
                theta = float(rng.uniform(-0.03, 0.03))
                R = small_rotation(rng, theta)
                Xtry[cl] = (Xtry[cl] @ R.T)
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [1500.0, 12000.0, 60000.0], [80, 110, 110])
            else:
                # large shake of under-locked + hubs with anneal
                deg, m = contact_degrees(Xbest, tol=3e-9)
                lo = np.argsort(deg)[:10]
                hi = np.argsort(-deg)[:10]
                Xtry = Xbest.copy()
                sig = float(rng.choice([0.02, 0.04, 0.07]))
                Xtry += sig * 0.5 * rng.standard_normal((n, d))
                Xtry[lo] += 0.05 * rng.standard_normal((10, d))
                Xtry[hi] += 0.03 * rng.standard_normal((10, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
                Xtry = lbfgs_anneal(Xtry, [1200.0, 10000.0, 50000.0], [80, 100, 120])
            v = _max_cos(Xtry)
            if v < Vbest + 0.002 and time.time() < DEADLINE - 6.0:
                # polish promising only
                if v < Vbest + 0.0005:
                    Xt2, v2 = lp_polish(Xtry, seconds=1.2, radius_init=4e-4)
                else:
                    Xt2, v2 = Xtry, v
                if v2 < Vbest - 1e-12:
                    Xbest, Vbest = Xt2, v2
            elif v < Vbest:
                Xbest, Vbest = Xtry, v
        except Exception:
            continue

    # final intensive polish of best
    try:
        if time.time() < DEADLINE - 6.0:
            Xb2 = lbfgs_anneal(Xbest, [50000.0, 150000.0], [120, 120])
            v = _max_cos(Xb2)
            if v < Vbest:
                Xbest, Vbest = Xb2, v
        if time.time() < DEADLINE - 2.0:
            rem = max(0.5, DEADLINE - 2.0 - time.time())
            Xp, Vp = lp_polish(Xbest, seconds=min(rem, 8.0), radius_init=2e-4)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    # tiny shrink guard (no-op, ensures finite)
    Xbest = np.where(np.isfinite(Xbest), Xbest, 0.0)
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
