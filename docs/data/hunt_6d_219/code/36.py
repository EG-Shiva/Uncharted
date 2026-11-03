import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _fast_relax(X, iters, lr0=0.02, margin0=0.004):
    X = _norm(X)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        Xn = cur - (lr / gn) * Gt
        Xn = _norm(Xn)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m + 1e-12:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 0 and it > 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 112.0
    rng = np.random.RandomState(12345)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum()) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.2e-3):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3800:
                k2 = 3800
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    def refine_void_single(v0, others, iters=160, lr0=0.06):
        v = v0 / (np.linalg.norm(v0) + 1e-12)
        lr = lr0
        best = float(np.max(others @ v))
        bestv = v.copy()
        for it in range(iters):
            dots = others @ v
            m = float(np.max(dots))
            tau = 0.002
            w = np.exp((dots - m) / tau)
            w[dots < m - 0.012] = 0.0
            g = w @ others
            gt = g - np.dot(g, v) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-14 or not np.isfinite(nrm):
                break
            vn = v - (lr / nrm) * gt
            vn /= (np.linalg.norm(vn) + 1e-12)
            mn = float(np.max(others @ vn))
            if mn < m:
                v = vn
                if mn < best:
                    best = mn; bestv = vn.copy()
            else:
                lr *= 0.7
                if lr < 1e-5:
                    break
        return bestv, best

    def kill_fill(Y, victims, n_cand=4000):
        Y = Y.copy()
        for w in victims:
            others = np.delete(Y, w, axis=0)
            cand = rng.randn(n_cand, d)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
            dots = cand @ others.T
            worst = dots.max(axis=1)
            order = np.argsort(worst)
            bv = None; bs = 1e9
            for bi in order[:3]:
                rv, rs = refine_void_single(cand[bi], others, iters=160, lr0=0.06)
                if rs < bs:
                    bs = rs; bv = rv
            Y[w] = bv
        return _norm(Y)

    # initial tighten
    try:
        Xi, vi = _fast_relax(x_best, iters=500, lr0=0.02, margin0=0.003)
        if vi < best_val:
            x_best, best_val = Xi, vi
        Y1 = run_lbfgs(x_best, beta=500, maxiter=150)
        v1 = _max_cos(Y1)
        if v1 < best_val:
            x_best, best_val = Y1, v1
        x_best, best_val = lp_polish(x_best, deadline=t0 + 12.0, r_init=1.0e-3)[0], lp_polish.__wrapped__ if False else best_val
    except Exception:
        pass
    # recompute best after polish attempt safely
    try:
        _bx, _bv = lp_polish(x_best, deadline=t0 + 14.0, r_init=1.0e-3)
        if _bv < best_val:
            x_best, best_val = _bx, _bv
        else:
            best_val = _max_cos(x_best)
    except Exception:
        best_val = _max_cos(x_best)

    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 50:
        trial += 1
        try:
            ppm, Gfull = _row_max(x_best)
        except Exception:
            ppm = np.zeros(n); Gfull = x_best @ x_best.T
        thr_c = best_val - 1e-7
        try:
            cc = (Gfull > thr_c).sum(axis=1)
        except Exception:
            cc = np.zeros(n)
        ccmax = max(1, int(np.max(cc)))
        order = np.argsort(-(ppm + 0.0015 * (cc / ccmax) + 1e-6 * rng.rand(n)))
        x_init = x_best.copy()
        mode = trial % 7
        if mode == 0:
            k = 10
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.13 * (0.7 + 0.6 * rng.rand(k, 1))
            x_init = x_init + sig * pert
            x_init = _norm(x_init)
        elif mode == 1:
            nw = 2 if rng.rand() < 0.5 else (3 if rng.rand() < 0.7 else 4)
            victims = list(order[:nw])
            if rng.rand() < 0.5:
                srt = np.sort(cc)
                thrh = srt[-30]
                hi = np.where(cc >= thrh)[0]
                if len(hi) > 0:
                    extra = int(hi[rng.randint(len(hi))])
                    if extra not in victims:
                        victims[-1] = extra
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = _norm(x_init + 0.006 * pert)
            try:
                x_init = kill_fill(x_init, victims, n_cand=4500)
            except Exception:
                pass
        elif mode == 2:
            w0 = int(order[0])
            Gj = Gfull[w0].copy(); Gj[w0] = -2
            nn = int(rng.randint(8, 16))
            neigh = np.argsort(-Gj)[:nn]
            clust = np.concatenate([[w0], neigh])
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[clust] = 0.055 + 0.03 * rng.rand()
            common = rng.randn(d)
            common = common - np.dot(common, x_init[w0]) * x_init[w0]
            common /= (np.linalg.norm(common) + 1e-12)
            x_init[clust] += 0.03 * common[None, :]
            x_init = _norm(x_init + sig * pert)
        elif mode == 3:
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = _norm(x_init + 0.011 * pert)
        elif mode == 4:
            k = int(rng.randint(20, 40))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.standard_cauchy((n, d)) * 0.15
            pert = np.clip(pert, -3, 3)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
            pert = pert / pn * np.minimum(pn, 1.5)
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.05
            x_init = _norm(x_init + sig * pert)
        elif mode == 5:
            k = int(rng.randint(3, 7))
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + 0.018 * pert
            x_init = _norm(x_init)
            G = x_init @ x_init.T
            np.fill_diagonal(G, -2)
            for w in idx:
                j = int(np.argmax(G[w]))
                v = x_init[w] - np.dot(x_init[w], x_init[j]) * x_init[j]
                nv = float(np.linalg.norm(v))
                if nv > 1e-9:
                    v /= nv
                    alpha = 0.3 + 0.4 * rng.rand()
                    x_init[w] = (1 - alpha) * x_init[w] + alpha * v
            x_init = _norm(x_init)
        else:
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = _norm(x_init + 0.03 * pert)
            w = int(order[0])
            try:
                x_init = kill_fill(x_init, [w], n_cand=3500)
            except Exception:
                pass

        try:
            Xr, vr = _fast_relax(x_init, iters=500, lr0=0.025, margin0=0.004)
        except Exception:
            continue
        if vr > best_val + 0.008:
            continue
        try:
            Y = run_lbfgs(Xr, beta=400, maxiter=120)
            v = _max_cos(Y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            # sharpen
            try:
                Y2 = run_lbfgs(Y, beta=1200, maxiter=120)
                v2 = _max_cos(Y2)
                if v2 < v:
                    Y, v = Y2, v2
            except Exception:
                pass
            x_best, best_val = Y, v
            # quick LP squeeze on new best (short deadline)
            try:
                dl = min(time.time() + 4.0, t0 + BUDGET - 8)
                Yp, vp = lp_polish(x_best, deadline=dl, r_init=8e-4)
                if vp < best_val:
                    x_best, best_val = Yp, vp
            except Exception:
                pass
        elif v < best_val + 0.002 and (trial % 6 == 0):
            try:
                Y2 = run_lbfgs(Y, beta=1000, maxiter=80)
                v2 = _max_cos(Y2)
                if v2 < best_val:
                    x_best, best_val = Y2, v2
            except Exception:
                pass

    # final intensive squeeze
    try:
        for beta, mi in [(250, 200), (600, 200), (1200, 250), (2500, 250)]:
            if time.time() - t0 > BUDGET - 12:
                break
            Y = run_lbfgs(x_best, beta=beta, maxiter=mi)
            v = _max_cos(Y)
            if v < best_val:
                x_best, best_val = Y, v
    except Exception:
        pass
    try:
        dl = t0 + BUDGET - 1.0
        Yp, vp = lp_polish(x_best, deadline=dl, r_init=1.0e-3)
        if vp < best_val:
            x_best, best_val = Yp, vp
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
