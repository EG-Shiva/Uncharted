import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos_fn(X, iu0, iu1):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def solve():
    t0 = time.time()
    TIME_BUDGET = 165.0
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load start
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.randn(n, d)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = max_cos(x_best)

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
            S = float(e.sum())
            loss = m + np.log(S + 1e-300) / beta
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()

        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def lp_polish(x, seconds, r_init=8e-4):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = max_cos(x)
        radius = r_init
        t_end = time.time() + seconds
        while time.time() < t_end and radius > 1e-13:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            thr = m - max(15.0 * radius, 5e-10)
            sel = c > thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 6000:
                k = 6000
                idx = np.argpartition(-c, k)[:k]
                ns = np.zeros_like(sel)
                ns[idx] = True
                sel = ns
            I = iu0[sel]
            J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
            v = max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.4, 5e-3)
            else:
                radius *= 0.5
        return x

    # quick refine of start with moderate beta
    if time.time() - t0 < TIME_BUDGET - 60:
        try:
            xc = run_lbfgs(x_best, 250.0, maxiter=120)
            vc = max_cos(xc)
            if vc < best_val:
                x_best, best_val = xc, vc
        except Exception:
            pass

    # basin hopping with focused shakes + hole reseeding
    trial = 0
    beta1_pool = [120.0, 180.0, 250.0]
    beta2_pool = [400.0, 600.0, 900.0]
    while time.time() - t0 < TIME_BUDGET - 38:
        # time guard for final stages
        remain = TIME_BUDGET - 38 - (time.time() - t0)
        if remain < 0:
            break
        try:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            pw = G0.max(axis=1)
            order = np.argsort(-pw)
        except Exception:
            order = np.arange(n)
        mode = trial % 7
        if mode == 6:
            # hole reseed worst point
            w = int(order[0])
            others = np.delete(x_best, w, axis=0)
            cand = rng.randn(500, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ others.T
            md = dots.max(axis=1)
            bc = cand[int(np.argmin(md))]
            x_init = x_best.copy()
            x_init[w] = bc
            # tiny jitter rest
            pert = rng.randn(n, d)
            proj = (pert * x_init).sum(axis=1, keepdims=True) * x_init
            pert -= proj
            x_init = x_init + 0.003 * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            b1 = float(beta1_pool[trial % 3])
            b2 = float(beta2_pool[(trial + 1) % 3])
            try:
                xo = run_lbfgs(x_init, b1, maxiter=70)
                xo = run_lbfgs(xo, b2, maxiter=70)
            except Exception:
                trial += 1
                continue
        elif mode in (1, 3, 5):
            # worst-subset large shake
            Q = [12, 20, 30][trial % 3]
            sig_w = [0.05, 0.08, 0.035][trial % 3]
            sig_b = 0.004
            idx_w = order[:Q]
            pert = rng.randn(n, d)
            proj = (pert * x_best).sum(axis=1, keepdims=True) * x_best
            pert -= proj
            step = np.full((n, 1), sig_b)
            step[idx_w] = sig_w
            # random sign scaling per point for diversity
            x_init = x_best + step * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            b1 = float(beta1_pool[trial % 3])
            b2 = float(beta2_pool[trial % 3])
            try:
                xo = run_lbfgs(x_init, b1, maxiter=70)
                xo = run_lbfgs(xo, b2, maxiter=70)
            except Exception:
                trial += 1
                continue
        else:
            # global shake
            sig = [0.012, 0.022, 0.008, 0.03][trial % 4]
            pert = rng.randn(n, d)
            proj = (pert * x_best).sum(axis=1, keepdims=True) * x_best
            pert -= proj
            x_init = x_best + sig * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            b1 = float(beta1_pool[(trial + 2) % 3])
            b2 = float(beta2_pool[(trial + 2) % 3])
            try:
                xo = run_lbfgs(x_init, b1, maxiter=70)
                xo = run_lbfgs(xo, b2, maxiter=70)
            except Exception:
                trial += 1
                continue
        v = max_cos(xo)
        if v < best_val:
            x_best, best_val = xo, v
        trial += 1
        if trial > 80:
            break

    # high-beta sharpening
    for hb, it in [(800.0, 100), (1500.0, 100), (2500.0, 80)]:
        if time.time() - t0 > TIME_BUDGET - 30:
            break
        try:
            xo = run_lbfgs(x_best, hb, maxiter=it)
            v = max_cos(xo)
            if v < best_val:
                x_best, best_val = xo, v
        except Exception:
            continue

    # final micro LP polish with remaining time
    remain = TIME_BUDGET - (time.time() - t0)
    if remain > 6:
        try:
            xp = lp_polish(x_best, min(remain - 1.0, 28.0), r_init=5e-4)
            v = max_cos(xp)
            if v < best_val:
                x_best, best_val = xp, v
            else:
                # second pass even smaller radius to tie plateau
                if remain > 12:
                    xp2 = lp_polish(xp if v < best_val + 1e-7 else x_best,
                                    6.0, r_init=1e-4)
                    v2 = max_cos(xp2)
                    if v2 < best_val:
                        x_best, best_val = xp2, v2
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
