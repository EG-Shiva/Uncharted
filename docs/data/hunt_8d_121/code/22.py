import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())


def _lp_polish(x, deadline, radius_init=5e-4, seconds=1e9):
    t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    best = _max_cos(x)
    best_x = x.copy()
    radius = radius_init
    while time.time() < deadline and (time.time() - t0) < seconds:
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
        try:
            A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x / np.linalg.norm(best_x, axis=1, keepdims=True)


def solve():
    t0 = time.time()
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]
    Jall = iu[1]

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

    def lbfgs_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        cur_best_X = X.copy()
        cur_best_v = _max_cos(X)
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 10.0:
                break
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

            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
            Xt = Xf.reshape(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            v = _max_cos(Xt)
            if v < cur_best_v:
                cur_best_v = v
                cur_best_X = Xt.copy()
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        vn = _max_cos(Xn)
        if vn < cur_best_v:
            return Xn, vn
        return cur_best_X, cur_best_v

    def deepest_void(Xref, n_rand=2500, beta=350.0, n_refine=3):
        # Xref: (m,d) unit rows
        C = rng.standard_normal((n_rand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xref.T
        cm = D.max(axis=1)
        idx = np.argpartition(cm, n_refine)[:n_refine]
        best_c = C[idx[0]]
        best_v = float(cm[idx[0]])
        for k in idx:
            c0 = C[k]

            def fun(u):
                nu = float(np.linalg.norm(u)) + 1e-12
                y = u / nu
                g = y @ Xref.T
                m = float(g.max())
                e = np.exp(beta * (g - m))
                s = float(e.sum())
                f = m + np.log(s) / beta
                w = e / s
                gy = w @ Xref
                gu = (gy - float(gy @ y) * y) / nu
                return float(f), gu

            try:
                res = minimize(fun, c0, jac=True, method="L-BFGS-B",
                               options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-9})
                cf = np.asarray(res.x, dtype=float)
                cf /= np.linalg.norm(cf)
            except Exception:
                cf = c0
            v = float(cf @ Xref.T.max() if False else (cf @ Xref.T).max())
            if v < best_v:
                best_v = v
                best_c = cf
        return best_c, best_v

    # quick single-point optimal replacement scan on worst offenders
    try:
        rm, gv = _rowmax(Xbest)
        order = np.argsort(-rm)[:20]
        for idx in order:
            if time.time() > DEADLINE - 60:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xref = Xbest[mask]
            v_center, v_val = deepest_void(Xref, n_rand=1500, beta=300.0, n_refine=2)
            Xtry = Xbest.copy()
            Xtry[idx] = v_center
            nv = _max_cos(Xtry)
            if nv < Vbest - 1e-12:
                Xbest, Vbest = Xtry, nv
                break
    except Exception:
        pass

    # main basin-hopping loop
    sigmas = [0.012, 0.03, 0.06, 0.008, 0.02, 0.045]
    schedules = [
        ([80.0, 350.0, 1200.0], [120, 120, 160]),
        ([150.0, 600.0, 2000.0], [100, 120, 180]),
        ([50.0, 250.0, 900.0], [120, 120, 150]),
    ]
    it = 0
    while time.time() < DEADLINE - 28.0:
        it += 1
        # periodic multi-point void surgery
        if it % 6 == 0:
            try:
                rm, _ = _rowmax(Xbest)
                # contacts count for cover
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                thr = Vbest - 2e-4
                cnt = (G > thr).sum(axis=1)
                score = rm + 0.002 * (cnt / max(1, cnt.max()))
                worst = np.argsort(-score)[:3]
                Xtmp = Xbest.copy()
                # sequentially insert deepest voids of reduced set
                remaining = [i for i in range(n) if i not in set(worst.tolist())]
                Xref = Xtmp[remaining]
                new_pts = []
                for _k in range(len(worst)):
                    vc, _ = deepest_void(Xref, n_rand=2000, beta=350.0, n_refine=3)
                    new_pts.append(vc)
                    Xref = np.vstack([Xref, vc[None, :]])
                for w, vc in zip(worst, new_pts):
                    Xtmp[w] = vc
                # short anneal to settle
                betas, mis = schedules[(it // 6) % len(schedules)]
                Xa, va = lbfgs_anneal(Xtmp, betas, mis)
                if va < Vbest:
                    Xbest, Vbest = Xa, va
                    # short polish to lock
                    Xp = _lp_polish(Xbest, DEADLINE - 20.0, radius_init=3e-4, seconds=4.0)
                    vp = _max_cos(Xp)
                    if vp < Vbest:
                        Xbest, Vbest = Xp, vp
                    continue
            except Exception:
                pass

        sig = sigmas[it % len(sigmas)]
        Xp = Xbest + sig * rng.standard_normal((n, d))
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        betas, mis = schedules[it % len(schedules)]
        # shorten if low on time
        if time.time() > DEADLINE - 50:
            mis = [60, 60, 80]
        Xa, va = lbfgs_anneal(Xp, betas, mis)
        if va < Vbest:
            Xbest, Vbest = Xa, va
        # opportunistic 2-point local void tweak on result if promising
        if va < Vbest + 5e-4 and time.time() < DEADLINE - 35:
            try:
                rm2, _ = _rowmax(Xa)
                top2 = np.argsort(-rm2)[:2]
                # try replacing worst of top2 by void of rest
                for idx in top2:
                    mask = np.ones(n, dtype=bool)
                    mask[idx] = False
                    vc, _ = deepest_void(Xa[mask], n_rand=1200, beta=300.0, n_refine=2)
                    Xt = Xa.copy()
                    Xt[idx] = vc
                    vt = _max_cos(Xt)
                    if vt < Vbest:
                        Xbest, Vbest = Xt, vt
                        break
            except Exception:
                pass

    # final polish chain
    if time.time() < DEADLINE - 2.0:
        try:
            Xp = _lp_polish(Xbest, DEADLINE - 1.0, radius_init=5e-4, seconds=1e9)
            vp = _max_cos(Xp)
            if vp <= Vbest:
                Xbest, Vbest = Xp, vp
        except Exception:
            pass
    # extra squeeze with tiny radius if time remains
    if time.time() < DEADLINE - 2.0:
        try:
            Xp2 = _lp_polish(Xbest, DEADLINE - 0.5, radius_init=1e-4, seconds=1e9)
            vp2 = _max_cos(Xp2)
            if vp2 < Vbest:
                Xbest, Vbest = Xp2, vp2
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
