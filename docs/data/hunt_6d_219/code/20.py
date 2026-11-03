import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _contact_deg(X, m, tol=2e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1), G

def _relax(X, iters, lr0=0.025, margin0=0.004, noise0=0.0, rng=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    cur = X.copy()
    best_val = _max_cos(cur)
    bestX = cur.copy()
    lr = lr0
    margin = margin0
    n = cur.shape[0]
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
            margin *= 0.6
            if margin < 5e-7:
                break
            continue
        step = (lr / gn) * Gt
        if noise0 > 0 and rng is not None:
            nl = noise0 * (1.0 - it / max(1, iters)) 
            if nl > 1e-12:
                R = rng.randn(*cur.shape)
                R -= (R * cur).sum(1, keepdims=True) * cur
                rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
                step = step + nl * lr * R / rn
        Xn = cur - step
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m + 1e-12:
            cur = Xn
            if mn < best_val:
                best_val = mn
                bestX = Xn.copy()
            if it % 200 == 199:
                margin *= 0.85
                # keep lr
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best_val

def _refine_void(v, others, iters=200, lr0=0.06):
    v = v / np.linalg.norm(v)
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
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
        vn /= np.linalg.norm(vn)
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 118.0
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
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.randn(n, d)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
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
            m = g.max()
            e = np.exp(beta * (g - m))
            S = e.sum()
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def lp_polish(x, deadline):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = 1.2e-3
        while time.time() < deadline and time.time() - t0 < BUDGET - 2.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k = 3500
                part = np.argpartition(-csel, k - 1)[:k]
                I, J, csel = I[part], J[part], csel[part]
                p = k
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
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
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
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

    # light initial tighten
    try:
        Xi, vi = _relax(x_best, iters=600, lr0=0.02, margin0=0.003, noise0=0.0, rng=rng)
        if vi < best_val:
            x_best, best_val = Xi, vi
    except Exception:
        pass

    cur = x_best.copy()
    cur_val = best_val
    Temp = 0.004

    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 50:
        trial += 1
        mode = trial % 6
        ppm = _per_point_max(cur)
        order = np.argsort(-ppm)
        # contact degrees for cluster mode
        try:
            degs, Gall = _contact_deg(cur, float(ppm.max()))
        except Exception:
            degs, Gall = None, None
        x_init = cur.copy()
        if mode == 0:
            k = int(rng.randint(6, 16))
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.10 + 0.08 * rng.rand(k, 1)
            x_init = x_init + sig * pert
        elif mode == 1:
            nw = 1 if rng.rand() < 0.6 else 2
            victims = order[:nw]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + 0.005 * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            for w in victims:
                others = np.delete(x_init, w, axis=0)
                cand = rng.randn(15000, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ others.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                bv, _ = _refine_void(cand[bi], others, iters=180, lr0=0.06)
                x_init[w] = bv
        elif mode == 2:
            # cluster shake around most jammed point
            if degs is not None:
                seed = int(np.argmax(degs))
            else:
                seed = int(order[0])
            G = x_init @ x_init.T
            np.fill_diagonal(G, -2.0)
            nbrs = np.argsort(-G[seed])[:int(rng.randint(8, 16))]
            idx = np.concatenate([[seed], nbrs])
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.05 + 0.05 * rng.rand()
            x_init = x_init + sig * pert
        elif mode == 3:
            k = int(rng.randint(25, 50))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.06 + 0.04 * rng.rand()
            x_init = x_init + sig * pert
        elif mode == 4:
            # coherent anisotropic warp + shake
            A = rng.randn(d, d) * 0.03
            A = (A + A.T) * 0.5
            M = np.eye(d) + A
            x_init = x_init @ M
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + 0.008 * pert
        else:
            sigma = 0.014
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + sigma * pert
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)

        noisy = (trial % 3 == 0)
        try:
            if noisy:
                Xr, vr = _relax(x_init, iters=550, lr0=0.025, margin0=0.005, noise0=0.35, rng=rng)
                Xr2, vr2 = _relax(Xr, iters=350, lr0=0.02, margin0=0.0025, noise0=0.0, rng=rng)
                if vr2 < vr:
                    Xr, vr = Xr2, vr2
            else:
                Xr, vr = _relax(x_init, iters=650, lr0=0.022, margin0=0.004, noise0=0.0, rng=rng)
        except Exception:
            continue

        improved_global = vr < best_val
        # wandering acceptance
        if vr < cur_val + Temp or improved_global:
            cur, cur_val = Xr, vr
        else:
            Temp *= 0.999
            continue

        if vr > best_val + 0.008 and rng.rand() < 0.85:
            Temp *= 0.998
            continue

        try:
            b1 = 160.0 if (trial % 2 == 0) else 260.0
            Xo = run_lbfgs(Xr, b1, maxiter=70)
            v = _max_cos(Xo)
            if v < best_val + 0.002 and (trial % 2 == 0):
                Xo2 = run_lbfgs(Xo, 480.0, maxiter=60)
                v2 = _max_cos(Xo2)
                if v2 < v:
                    Xo, v = Xo2, v2
        except Exception:
            Temp *= 0.998
            continue

        if v < best_val - 1e-12:
            x_best, best_val = Xo, v
            cur, cur_val = Xo, v
            try:
                Xp, vp = lp_polish(x_best, deadline=time.time() + 2.0)
                if vp < best_val:
                    x_best, best_val = Xp, vp
                    cur, cur_val = Xp, vp
            except Exception:
                pass
        elif v < cur_val:
            cur, cur_val = Xo, v
        Temp *= 0.997
        if Temp < 3e-4:
            Temp = 3e-4 + 1e-3 * rng.rand()
        if time.time() - t0 > BUDGET - 50:
            break

    for beta, mi in [(180.0, 110), (380.0, 110), (700.0, 110)]:
        if time.time() - t0 > BUDGET - 36:
            break
        try:
            Xo = run_lbfgs(x_best, beta, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val:
                x_best, best_val = Xo, v
        except Exception:
            pass

    try:
        if time.time() < t0 + BUDGET - 5:
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 1.5)
            if vp < best_val:
                x_best, best_val = Xp, vp
    except Exception:
        pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
