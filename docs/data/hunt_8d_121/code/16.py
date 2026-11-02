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
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall = iu[0]
    Jall = iu[1]
    P = len(Iall)

    # load starts
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
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def make_fun(beta):
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
            W = np.zeros((n, n), dtype=np.float64)
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def lbfgs_anneal(Xinit, betas, maxiters):
        Xf = (Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)).ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 28.0:
                break
            fun = make_fun(beta)
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 8)})
                Xf = res.x
                Xt = Xf.reshape(n, d)
                nn = np.linalg.norm(Xt, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                Xt = Xt / nn
                Xf = Xt.ravel()
            except Exception:
                break
        X = Xf.reshape(n, d)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 250 or radius < 1e-14:
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
            c = cf[sel]
            I = iu[0][sel]
            J = iu[1][sel]
            p = len(I)
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
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    Xcur = Xbest.copy()
    Vcur = Vbest
    stagnant = 0

    # basin hopping
    for it in range(300):
        if time.time() > DEADLINE - 30.0:
            break
        # propose initial
        if it % 5 == 4 or stagnant >= 6:
            # large kick from best (new basin)
            sig = 0.28 if (it % 10 == 4) else (0.18 if (it % 10 == 9) else 0.10)
            # vary sigma a bit deterministically with rng
            Yinit = Xbest + sig * rng.standard_normal((n, d))
            Yinit /= np.linalg.norm(Yinit, axis=1, keepdims=True)
        else:
            # contact-guided subgraph shake from current
            try:
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                # contacts count near max
                thr_c = G.max() - 1e-9
                cntc = (G > thr_c).sum(axis=1)
                score = rowmax + 0.002 * (cntc / max(1, cntc.max())) + 0.0005 * rng.standard_normal(n)
                center = int(np.argmax(score))
                dots = Xcur @ Xcur[center]
                k = 12 if (it % 2 == 0) else 8
                if it % 6 == 2:
                    k = 16
                neigh = np.argsort(-dots)[:k]
                s = 0.28 if (it % 2 == 0) else 0.13
                if it % 6 == 2:
                    s = 0.40
                Yinit = Xcur.copy()
                Yinit[neigh] += s * rng.standard_normal((k, d))
                Yinit += 0.004 * rng.standard_normal((n, d))
                Yinit /= np.linalg.norm(Yinit, axis=1, keepdims=True)
            except Exception:
                Yinit = Xcur + 0.05 * rng.standard_normal((n, d))
                Yinit /= np.linalg.norm(Yinit, axis=1, keepdims=True)

        if it % 7 == 6:
            betas = [50.0, 150.0, 500.0, 1500.0]
            mis = [70, 70, 90, 110]
        elif it % 3 == 2:
            betas = [80.0, 250.0, 900.0]
            mis = [60, 70, 90]
        else:
            betas = [80.0, 300.0, 1000.0]
            mis = [50, 60, 80]

        Xnew = lbfgs_anneal(Yinit, betas, mis)
        Vnew = _max_cos(Xnew)

        if Vnew < Vcur - 1e-12:
            Xcur, Vcur = Xnew, Vnew
            stagnant = 0
            if Vnew < Vbest - 1e-12:
                Xbest, Vbest = Xnew.copy(), Vnew
        else:
            stagnant += 1
            if Vnew < Vcur + 0.002 and rng.random() < 0.12:
                Xcur, Vcur = Xnew, Vnew
            if stagnant >= 8:
                Xcur = Xbest.copy()
                Vcur = Vbest
                stagnant = 0

    # final intensive refinement of best
    if time.time() < DEADLINE - 12.0:
        Xf = lbfgs_anneal(Xbest, [200.0, 800.0, 2500.0, 6000.0], [120, 150, 180, 220])
        Vf = _max_cos(Xf)
        if Vf < Vbest:
            Xbest, Vbest = Xf, Vf
    if time.time() < DEADLINE - 4.0:
        remain = max(2.0, (DEADLINE - 1.0) - time.time())
        Xp, Vp = lp_polish(Xbest, seconds=min(remain, 22.0), radius_init=3e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
        # one more short L-BFGS + LP alternating if time
        if time.time() < DEADLINE - 6.0:
            Xf2 = lbfgs_anneal(Xbest, [1000.0, 4000.0], [120, 150])
            Vf2 = _max_cos(Xf2)
            if Vf2 < Vbest:
                Xbest, Vbest = Xf2, Vf2
            if time.time() < DEADLINE - 3.0:
                rem2 = max(1.0, (DEADLINE - 0.5) - time.time())
                Xp2, Vp2 = lp_polish(Xbest, seconds=min(rem2, 10.0), radius_init=2e-4)
                if Vp2 < Vbest:
                    Xbest, Vbest = Xp2, Vp2

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
