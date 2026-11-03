import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(20261103)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

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

    def tangent_noise(X, sigma, idx=None):
        P = rng.randn(n, d)
        P -= (P * X).sum(axis=1, keepdims=True) * X
        if idx is not None:
            M = np.zeros((n, 1))
            M[idx] = 1.0
            # scale outside idx to small done by caller
            return P
        return P

    # initial light refine of start
    try:
        if time.time() - t0 < BUDGET - 100:
            xc = run_lbfgs(x_best, 300.0, maxiter=60)
            vc = max_cos(xc)
            if vc < best_val:
                x_best, best_val = xc, vc
    except Exception:
        pass

    x_cur = x_best.copy()
    cur_val = best_val
    stall = 0
    trial = 0
    b1_pool = [140.0, 220.0, 320.0]
    b2_pool = [550.0, 850.0, 1200.0]
    HOP_END = t0 + 115.0

    while time.time() < HOP_END and time.time() - t0 < BUDGET - 52:
        trial += 1
        if stall >= 18:
            x_cur = x_best.copy()
            cur_val = best_val
            stall = 0
        try:
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            pw = G0.max(axis=1)
            order = np.argsort(-pw)
        except Exception:
            order = np.arange(n)
        mode = rng.randint(0, 10)
        x_init = None
        try:
            if mode in (0, 5):  # farthest hole reseed worst point
                w = int(order[rng.randint(0, 4)])
                others = np.delete(x_cur, w, axis=0)
                nc = 3500 if (trial % 2 == 0) else 1500
                bh = best_hole(others, ncand=nc)
                x_init = x_cur.copy()
                x_init[w] = bh
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.003)
                step[w] = 0.006
                # extra jitter neighbours of w
                Gw = x_cur[w] @ x_cur.T
                nb = np.argsort(-Gw)[:8]
                step[nb] = np.maximum(step[nb], 0.006)
                x_init = x_init + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (1, 6):  # worst subset kick
                Q = int(rng.choice([8, 14, 22, 32]))
                sig_w = float(rng.choice([0.045, 0.08, 0.12, 0.17]))
                idx_w = order[:Q]
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                step = np.full((n, 1), 0.003)
                step[idx_w] = sig_w
                # random sign per point
                sgn = np.sign(rng.randn(n, 1))
                sgn[sgn == 0] = 1.0
                step = step * sgn
                x_init = x_cur + step * np.abs(P) * 0.0 + step * P * 0.0 + step * 0.0  # placeholder
                # simpler: direct scaled
                x_init = x_cur + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (2, 7):  # opposite pair-break of worst pair
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                flat = int(np.argmax(Gc))
                a, b = flat // n, flat % n
                x_init = x_cur.copy()
                dot = float(np.dot(x_init[a], x_init[b]))
                # move a away from b, b away from a tangentially
                s = float(rng.choice([0.02, 0.035, 0.06]))
                da = -(x_init[b] - dot * x_init[a])
                db = -(x_init[a] - dot * x_init[b])
                da -= float(np.dot(da, x_init[a])) * x_init[a]
                db -= float(np.dot(db, x_init[b])) * x_init[b]
                na = np.linalg.norm(da) + 1e-12
                nb = np.linalg.norm(db) + 1e-12
                da /= na; db /= nb
                x_init[a] = x_init[a] + s * da + 0.004 * rng.randn(d)
                x_init[b] = x_init[b] + s * db + 0.004 * rng.randn(d)
                # global tiny jitter
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.003 * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode == 3:  # cluster expansion
                # densest point by sum of top-k dots
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                k = 12
                part = np.partition(-Gc, k, axis=1)[:, :k]
                dens = (-part).sum(axis=1)
                c = int(np.argmax(dens))
                nn = np.argsort(-Gc[c])[:int(rng.choice([10, 14, 18]))]
                x_init = x_cur.copy()
                s = float(rng.choice([0.03, 0.05, 0.08]))
                for j in nn:
                    dot = float(np.dot(x_init[j], x_init[c]))
                    dj = -(x_init[c] - dot * x_init[j])
                    dj -= float(np.dot(dj, x_init[j])) * x_init[j]
                    nj = np.linalg.norm(dj) + 1e-12
                    dj /= nj
                    x_init[j] = x_init[j] + s * (0.5 + 0.5 * rng.rand()) * dj
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.003 * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode == 4:  # global shake
                sig = float(rng.choice([0.010, 0.020, 0.030, 0.008]))
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                x_init = x_cur + sig * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            else:  # rattler-to-core swap + jitter: move 2 rattlers into holes, jitter core
                # rattlers = lowest pw
                ratt = order[-20:]
                pick = rng.choice(ratt, size=2, replace=False)
                x_init = x_cur.copy()
                tmp_others = x_init.copy()
                for w in pick:
                    others = np.delete(tmp_others, w, axis=0)
                    bh = best_hole(others, ncand=2000)
                    tmp_others[w] = bh
                x_init = tmp_others
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.002)
                step[order[:10]] = 0.015
                x_init = x_init + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        except Exception:
            continue
        b1 = float(b1_pool[rng.randint(0, 3)])
        b2 = float(b2_pool[rng.randint(0, 3)])
        try:
            xo = run_lbfgs(x_init, b1, maxiter=55)
            xo = run_lbfgs(xo, b2, maxiter=55)
        except Exception:
            continue
        v = max_cos(xo)
        if v < best_val - 1e-12:
            x_best, best_val = xo, v
            x_cur, cur_val = xo, v
            stall = 0
        else:
            # sideways acceptance to wander
            tol = 0.0006 if stall < 8 else 0.0012
            if v < cur_val + tol and rng.rand() < 0.6:
                x_cur, cur_val = xo, v
                stall += 1
            else:
                # occasionally accept best-near miss to cur
                if v < cur_val:
                    x_cur, cur_val = xo, v
                    stall += 1
                else:
                    stall += 1
        if trial > 120:
            break

    # high beta sharpening
    for hb, it in [(800.0, 80), (1500.0, 80), (2800.0, 60)]:
        if time.time() - t0 > BUDGET - 30:
            break
        try:
            xo = run_lbfgs(x_best, hb, maxiter=it)
            v = max_cos(xo)
            if v < best_val:
                x_best, best_val = xo, v
        except Exception:
            continue

    # greedy true-max micro moves
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 8:
            X = x_best.copy()
            bv = best_val
            # budget ~ up to 4000 evals but time-guarded
            for it in range(3000):
                if it % 200 == 0 and time.time() - t0 > BUDGET - 22:
                    break
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                a, b = flat // n, flat % n
                curm = float(G[a, b])
                # choose endpoint
                tgt = a if rng.rand() < 0.5 else b
                oth = b if tgt == a else a
                dot = float(np.dot(X[tgt], X[oth]))
                grad = -(X[oth] - dot * X[tgt])
                grad -= float(np.dot(grad, X[tgt])) * X[tgt]
                ng = np.linalg.norm(grad) + 1e-12
                grad /= ng
                # random tangent orthogonal component for diversity
                rr = rng.randn(d)
                rr -= float(np.dot(rr, X[tgt])) * X[tgt] + float(np.dot(rr, grad)) * grad
                rr /= (np.linalg.norm(rr) + 1e-12)
                step = float(rng.choice([0.0004, 0.0008, 0.0002, 0.0015]))
                trial_dir = grad * 0.85 + rr * 0.35
                trial_dir /= (np.linalg.norm(trial_dir) + 1e-12)
                Xt = X.copy()
                Xt[tgt] = Xt[tgt] + step * trial_dir
                Xt[tgt] /= (np.linalg.norm(Xt[tgt]) + 1e-12)
                nv = max_cos(Xt)
                if nv < bv:
                    X, bv = Xt, nv
            if bv < best_val:
                x_best, best_val = X, bv
    except Exception:
        pass

    # micro trust-region LP polish
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 7:
            x = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
            best = max_cos(x)
            radius = 4e-4
            t_end = time.time() + min(remain - 1.0, 20.0)
            while time.time() < t_end and radius > 1e-13:
                G = x @ x.T
                c = G[iu0, iu1]
                m = float(c.max())
                thr = m - max(18.0 * radius, 1e-9)
                sel = c > thr
                cnt = int(sel.sum())
                if cnt == 0:
                    radius *= 0.5
                    continue
                if cnt > 4500:
                    k = 4500
                    idx = np.argpartition(-c, k)[:k]
                    ns = np.zeros_like(sel)
                    ns[idx] = True
                    sel = ns
                I = iu0[sel]; J = iu1[sel]
                csel = c[sel]
                p = len(I)
                nv_ = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv_)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv_)).tocsr()
                obj = np.zeros(nv_)
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
                    radius = min(radius * 1.4, 4e-3)
                else:
                    radius *= 0.5
            if best < best_val:
                x_best, best_val = x, best
    except Exception:
        pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
