import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]
    P = len(Iall)
    t_iso = float(n) / float(d)

    # load incumbents
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
    Xbal = Xbest.copy()

    def point_stats(X, tol=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cf = G[iu]
        m = float(cf.max())
        # per point max and argmax
        Gm = X @ X.T
        np.fill_diagonal(Gm, -2.0)
        p = Gm.max(axis=1)
        js = Gm.argmax(axis=1)
        deg = (Gm > m - tol).sum(axis=1).astype(float)
        avg = float(deg.mean()) + 1e-12
        # tangential residual with uniform weights on tol-neighbours
        res = np.zeros(n)
        for i in range(n):
            nb = np.where(Gm[i] > m - tol)[0]
            nb = nb[nb != i]
            k = len(nb)
            if k == 0:
                res[i] = 0.0
                continue
            cij = Gm[i, nb]
            # P_i x_j
            Vij = X[nb] - cij[:, None] * X[i][None, :]
            s = Vij.sum(axis=0)
            # normalize by k * typical norm sqrt(1-c^2)~0.89
            res[i] = float(np.linalg.norm(s) / (k * 0.89 + 1e-12))
        stdp = float(p.std())
        stddeg = float(deg.std() / (avg + 1e-12))
        meanres = float(res.mean())
        # isotropy
        S = X.T @ X
        iso = float(np.linalg.norm(S - t_iso * np.eye(d)) / t_iso)
        return m, p, js, deg, avg, res, meanres, stdp, stddeg, iso

    _, _, _, _, _, _, meanres0, stdp0, _, _ = point_stats(Xbest)
    Rbal = float(meanres0 + stdp0 * 5.0)

    def build_weights(X, tol=1e-4, gamma=0.7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G[iu].max())
        deg = (G > m - tol).sum(axis=1).astype(float)
        avg = float(deg.mean()) + 1e-12
        di = deg[Iall]; dj = deg[Jall]
        s = (di + dj) / (2.0 * avg)
        A = 1.0 / (1.0 + gamma * (s - 1.0))
        A = np.clip(A, 0.25, 2.0)
        # far pairs keep 1 to avoid distortion: blend to 1 when both endpoints far from max?
        # keep as is; far pairs have negligible exp weight anyway
        return A, deg, avg, m

    def make_fun(beta, Avec, lam_iso, lam_bal):
        b = float(beta)
        li = float(lam_iso)
        lb = float(lam_bal)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            c = G[iu]
            m = float(c.max())
            e = Avec * np.exp(b * (c - m))
            ssum = float(e.sum())
            if not np.isfinite(ssum) or ssum <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(ssum) / b
            w = e / ssum
            # grad lse wrt Y
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y  # (n,d)
            # isotropy penalty
            f_iso = 0.0
            GI = np.zeros_like(GY)
            if li != 0:
                S = Y.T @ Y
                E = S - t_iso * np.eye(d)
                f_iso = li * float((E ** 2).sum()) / (t_iso ** 2 * d)
                f += f_iso
                GI = (4.0 * li / (t_iso ** 2 * d)) * (Y @ E)
            # balance penalty on variance of per-point max
            GB = np.zeros_like(GY)
            if lb != 0:
                # need full G with diag -2 for argmax
                p = G.max(axis=1)
                js = G.argmax(axis=1)
                mu = float(p.mean())
                diff = p - mu
                var = float((diff ** 2).mean())
                f += lb * var
                coeff = (2.0 * lb / n) * diff  # (n,)
                # grad: for each i, coeff[i] * y_{js[i]} to row i, and coeff[i]*y_i to row js[i]
                # vectorized scatter
                GB = np.zeros((n, d))
                # term1
                GB += coeff[:, None] * Y[js]
                # term2: accumulate
                np.add.at(GB, js, coeff[:, None] * Y)
            GYt = GY + GI + GB
            dot = (GYt * Y).sum(axis=1, keepdims=True)
            GX = (GYt - dot * Y) / nr
            GX = np.nan_to_num(GX, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(f):
                f = float(m)
            return float(f), GX.ravel()
        return fun

    def lbfgs_opt(Xinit, beta, Avec, lam_iso, lam_bal, maxiter):
        fun = make_fun(beta, Avec, lam_iso, lam_bal)
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-12, "maxfun": int(maxiter * 5)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = Xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn
        return X

    def find_voids(base, k, nsample=22000):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ base.T
        cm = D.max(axis=1)
        outs = []
        B = base.copy()
        # greedy farthest
        for _ in range(k):
            C2 = rng.standard_normal((nsample // 2, d))
            C2 /= np.linalg.norm(C2, axis=1, keepdims=True)
            D2 = C2 @ B.T
            cm2 = D2.max(axis=1)
            idx = int(np.argmin(cm2))
            v = C2[idx].copy()
            # local refine
            cur = float(cm2[idx])
            for st in range(120):
                sig = 0.04 * (1.0 - st / 120.0) + 0.002
                prop = v + sig * rng.standard_normal(d)
                prop /= np.linalg.norm(prop)
                md = float((prop @ B.T).max())
                if md < cur:
                    v = prop; cur = md
            outs.append(v)
            B = np.vstack([B, v[None, :]])
        return outs

    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 150 or radius < 1e-14:
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
                    mk = np.zeros_like(sel); mk[idxp] = True
                    sel = mk; cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p_ = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p_), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p_, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p_, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p_, nv)).tocsr()
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

    # ---- stress-rebalanced annealing ----
    Xcur = Xbest.copy()
    # initial weights at 1e-4 tolerance
    Avec, _, _, _ = build_weights(Xcur, tol=1e-4, gamma=0.7)

    # Stage schedule: (beta, lam_iso, lam_bal, maxiter)
    # deliberately relaxed 0.465-0.475 equivalent: low beta + penalties, then tighten
    stages = [
        (110, 0.25, 8.0, 180),
        (160, 0.18, 6.0, 160),
        (220, 0.10, 4.0, 160),
        (320, 0.05, 2.0, 150),
        (500, 0.02, 1.0, 140),
        (800, 0.005, 0.3, 120),
    ]
    si = 0
    # interleave knockouts
    while time.time() < DEADLINE - 32.0:
        beta, li, lb, mi = stages[min(si, len(stages) - 1)]
        # refresh weights every round from current (stress-dual inverse)
        # alternate tolerance to probe different contact sets
        tol = 1e-4 if (si % 2 == 0) else 5e-4
        Avec, deg, avg, mcur = build_weights(Xcur, tol=tol, gamma=0.7)
        # re-optimize
        Xn = lbfgs_opt(Xcur, beta, Avec, li, lb, mi)
        Vn = _max_cos(Xn)
        m, p, js, degn, avgn, res, meanres, stdp, stddeg, iso = point_stats(Xn, tol=1e-4)
        Rnew = float(meanres + stdp * 5.0)
        improved_best = Vn < Vbest - 1e-12
        improved_bal = (Rnew < Rbal - 1e-6) and (Vn < Vbest + 0.004)
        if improved_best:
            Xbest, Vbest = Xn.copy(), float(Vn)
        if improved_bal or (Vn < _max_cos(Xbal) - 1e-12 and Rnew < Rbal + 0.002):
            Xbal = Xn.copy()
            Rbal = min(Rbal, Rnew)
        # choose next current: alternate between best-balanced and best-cosine
        # retain balanced lineage even if slightly worse
        if improved_bal and not improved_best:
            Xcur = Xn.copy()
        elif si % 3 == 2:
            # hop from balanced lineage with perturbation to escape basin
            sig = 0.015 if si < 4 else 0.008
            Xcur = Xbal + sig * rng.standard_normal((n, d))
            Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
            # quick relax toward Xbal basin happens next round
        else:
            if Vn < _max_cos(Xcur):
                Xcur = Xn.copy()
            else:
                # small global hop
                Xcur = Xn + 0.006 * rng.standard_normal((n, d))
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)

        # topology change: knockout worst residual / extreme-degree points
        if time.time() > DEADLINE - 32.0:
            break
        # identify worst K
        try:
            m2, p2, js2, deg2, avg2, res2, _, _, _, _ = point_stats(Xcur, tol=1e-4)
            worst_score = (p2 - p2.min()) + 0.03 * res2 + 0.004 * np.abs(deg2 - avg2) / (avg2 + 1e-9)
            K = 3 if si < 4 else 2
            worst_idx = np.argsort(-worst_score)[:K]
            voids = find_voids(np.delete(Xcur, worst_idx, axis=0), K)
            Xk = Xcur.copy()
            for ii, v in zip(worst_idx, voids):
                Xk[ii] = v
            # joint re-optimization at relaxed beta with same weights (recomputed)
            Avec2, _, _, _ = build_weights(Xk, tol=tol, gamma=0.5)
            Xk2 = lbfgs_opt(Xk, max(120, beta // 2), Avec2, li, lb, 90)
            Vk2 = _max_cos(Xk2)
            _, _, _, _, _, resk, meanresk, stdpk, _, _ = point_stats(Xk2, tol=1e-4)[0:9] if False else (None,)*9
            # recompute properly
            _, _, _, _, _, resk2, meanresk2, stdpk2, _, _ = point_stats(Xk2, tol=1e-4)
            Rk2 = float(meanresk2 + stdpk2 * 5.0)
            if Vk2 < Vbest - 1e-12:
                Xbest, Vbest = Xk2.copy(), float(Vk2)
            if Rk2 < Rbal and Vk2 < Vbest + 0.004:
                Xbal = Xk2.copy(); Rbal = Rk2
            # adopt knockout if promising (better balance or better cosine)
            if Vk2 < _max_cos(Xcur) - 2e-6 or (Rk2 < float(res2.mean() + p2.std() * 5.0) and Vk2 < _max_cos(Xcur) + 0.002):
                Xcur = Xk2.copy()
        except Exception:
            pass
        si += 1
        if si > 30:
            break

    # final selection: polish top candidates
    pool = [(Vbest, Xbest.copy())]
    try:
        vbal = _max_cos(Xbal)
        pool.append((vbal, Xbal.copy()))
    except Exception:
        pass
    try:
        vcur = _max_cos(Xcur)
        pool.append((vcur, Xcur.copy()))
    except Exception:
        pass
    pool.sort(key=lambda t: t[0])
    # deduplicate roughly
    uniq = []
    for v, x in pool:
        ok = True
        for _, y in uniq:
            if np.linalg.norm(x - y) < 1e-6 or np.linalg.norm(x + y) < 1e-6:
                ok = False; break
        if ok:
            uniq.append((v, x))
    remain = DEADLINE - time.time() - 1.0
    secs_each = max(4.0, remain / max(1, len(uniq[:2])))
    for v, x in uniq[:2]:
        if time.time() > DEADLINE - 2.0:
            break
        xp, vp = lp_polish(x, seconds=min(secs_each, DEADLINE - time.time() - 1.0))
        if vp < Vbest - 1e-12:
            Xbest, Vbest = xp.copy(), float(vp)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
