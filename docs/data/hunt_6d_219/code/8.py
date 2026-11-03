import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 170.0
    np.random.seed(0)
    rng = np.random.RandomState(7)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load
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
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def best_hole(others, ncand=3000):
        cand = rng.randn(ncand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ others.T
        md = D.max(axis=1)
        return cand[int(np.argmin(md))]

    def tangent_jitter(X, idx, sigma):
        P = rng.randn(len(idx), d)
        Xi = X[idx]
        P -= (P * Xi).sum(axis=1, keepdims=True) * Xi
        nn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        P /= nn
        amp = rng.randn(len(idx), 1) * sigma
        X2 = X.copy()
        X2[idx] += amp * P
        X2[idx] /= np.linalg.norm(X2[idx], axis=1, keepdims=True)
        return X2

    # quick refine start
    if time.time() - t0 < BUDGET - 80:
        try:
            xc = run_lbfgs(x_best, 300.0, maxiter=80)
            vc = max_cos(xc)
            if vc < best_val:
                x_best, best_val = xc, vc
        except Exception:
            pass

    trial = 0
    stall = 0
    beta1_pool = [150.0, 220.0, 320.0]
    beta2_pool = [550.0, 800.0, 1100.0]
    HOP_END = t0 + 118.0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 50:
        trial += 1
        try:
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            pw = G0.max(axis=1)
            order = np.argsort(-pw)
        except Exception:
            order = np.arange(n)
        mode = trial % 10
        x_init = None
        try:
            if mode in (0, 5, 7):  # hole reseed worst point
                w = int(order[rng.randint(0, 5)])
                others = np.delete(x_best, w, axis=0)
                nc = 3000 if (trial % 2 == 0) else 1200
                bh = best_hole(others, ncand=nc)
                x_init = x_best.copy()
                x_init[w] = bh
                # jitter neighbourhood + tiny global
                pert = rng.randn(n, d)
                pert -= (pert * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.003)
                # extra jitter on previous neighbours of w
                step[w] = 0.008
                x_init = x_init + step * pert
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (1, 4, 8):  # worst-subset kick
                Q = int(rng.choice([8, 12, 20, 30]))
                sig_w = float(rng.choice([0.04, 0.07, 0.11, 0.16]))
                idx_w = order[:Q]
                pert = rng.randn(n, d)
                pert -= (pert * x_best).sum(axis=1, keepdims=True) * x_best
                step = np.full((n, 1), 0.003)
                step[idx_w] = sig_w
                x_init = x_best + step * pert
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (2, 9):  # opposite pair-break
                # find worst pair
                Gc = x_best @ x_best.T
                np.fill_diagonal(Gc, -2.0)
                a, b = np.unravel_index(int(np.argmax(Gc)), Gc.shape)
                x_init = x_best.copy()
                # random tangent direction orthogonal-ish to both
                u = rng.randn(d)
                # make tangent at a and b approximately
                ua = u - (u @ x_init[a]) * x_init[a]
                ub = u - (u @ x_init[b]) * x_init[b]
                ua /= (np.linalg.norm(ua) + 1e-12)
                ub /= (np.linalg.norm(ub) + 1e-12)
                s = float(rng.choice([0.05, 0.09, 0.14]))
                x_init[a] += s * ua
                x_init[b] -= s * ub
                # jitter their top neighbours
                neigh = np.argsort(-Gc[a])[:6].tolist() + np.argsort(-Gc[b])[:6].tolist()
                for v in set(neigh):
                    if v == a or v == b:
                        continue
                    p = rng.randn(d)
                    p -= (p @ x_init[v]) * x_init[v]
                    p /= (np.linalg.norm(p) + 1e-12)
                    x_init[v] += 0.02 * rng.randn() * p
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            else:  # cluster expansion
                Q = int(rng.choice([15, 25]))
                idx_w = order[:Q]
                C = x_best[idx_w].mean(axis=0)
                C /= (np.linalg.norm(C) + 1e-12)
                x_init = x_best.copy()
                alpha = float(rng.choice([0.04, 0.08]))
                for v in idx_w:
                    away = x_init[v] - (x_init[v] @ C) * C
                    # tangent part at v of (x_v - C)
                    t = (x_init[v] - C)
                    t -= (t @ x_init[v]) * x_init[v]
                    nrm = np.linalg.norm(t) + 1e-12
                    t /= nrm
                    x_init[v] += alpha * t
                pert = rng.randn(n, d)
                pert -= (pert * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.004 * pert
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        except Exception:
            trial += 1
            continue
        b1 = float(beta1_pool[rng.randint(0, 3)])
        b2 = float(beta2_pool[rng.randint(0, 3)])
        if stall > 12:  # escalate disruption
            b1, b2 = 150.0, 550.0
        try:
            xo = run_lbfgs(x_init, b1, maxiter=50)
            xo = run_lbfgs(xo, b2, maxiter=50)
        except Exception:
            stall += 1
            continue
        v = max_cos(xo)
        if v < best_val:
            x_best, best_val = xo, v
            stall = 0
        else:
            stall += 1
        if trial > 200:
            break

    # high beta sharpening
    for hb, it in [(1200.0, 80), (2200.0, 60)]:
        if time.time() - t0 > BUDGET - 32:
            break
        try:
            xo = run_lbfgs(x_best, hb, maxiter=it)
            v = max_cos(xo)
            if v < best_val:
                x_best, best_val = xo, v
        except Exception:
            continue

    # greedy true-max micro hill climbing on worst points
    try:
        deadline = t0 + BUDGET - 12.0
        Xc = x_best.copy()
        cur = best_val
        step = 2e-4
        it = 0
        while time.time() < deadline and it < 6000:
            it += 1
            Gc = Xc @ Xc.T
            np.fill_diagonal(Gc, -2.0)
            pw = Gc.max(axis=1)
            order = np.argsort(-pw)
            # pick a victim among top 10
            vi = int(order[rng.randint(0, 10)])
            # worst partner
            pj = int(np.argmax(Gc[vi]))
            # move vi away from pj
            grad = Xc[pj] - (Xc[pj] @ Xc[vi]) * Xc[vi]
            gn = np.linalg.norm(grad) + 1e-12
            grad /= gn
            # random orthogonal component for diversity
            rr = rng.randn(d)
            rr -= (rr @ Xc[vi]) * Xc[vi] + (rr @ grad) * grad
            rr /= (np.linalg.norm(rr) + 1e-12)
            prop = Xc.copy()
            mv = -step * grad + (step * 0.5) * rr * rng.randn()
            prop[vi] += mv
            prop[vi] /= np.linalg.norm(prop[vi])
            nv = max_cos(prop)
            if nv < cur:
                Xc, cur = prop, nv
                if cur < best_val:
                    x_best, best_val = Xc.copy(), cur
                    step = min(step * 1.05, 5e-4)
            else:
                if it % 300 == 0:
                    step *= 0.7
                    if step < 3e-6:
                        step = 2e-4
    except Exception:
        pass

    # final tiny trust-region LP if time remains
    try:
        remain = (t0 + BUDGET - 3.0) - time.time()
        if remain > 7:
            x = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
            best = max_cos(x)
            radius = 4e-4
            t_end = time.time() + min(remain - 1.0, 10.0)
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
                if cnt > 4000:
                    k = 4000
                    idx = np.argpartition(-c, k)[:k]
                    ns = np.zeros_like(sel)
                    ns[idx] = True
                    sel = ns
                I = iu0[sel]; J = iu1[sel]; csel = c[sel]
                p = len(I)
                nv_ = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv_)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv_)).tocsr()
                obj = np.zeros(nv_); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
                if res.status != 0:
                    radius *= 0.5
                    continue
                y = x + res.x[:-1].reshape(n, d)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                v = max_cos(y)
                if v < best:
                    x, best = y, v
                    radius = min(radius * 1.4, 3e-3)
                else:
                    radius *= 0.5
            if best < best_val:
                x_best, best_val = x, best
    except Exception:
        pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
