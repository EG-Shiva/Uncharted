import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 112.0
    rng = np.random.RandomState(20261104)
    n, d = 219, 6
    dm = d - 1
    iu0, iu1 = np.triu_indices(n, 1)

    def _norm(X):
        q = np.linalg.norm(X, axis=1, keepdims=True)
        q = np.maximum(q, 1e-12)
        return X / q

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
                v = _norm(v)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
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
        Y = _norm(Y)
        return Y

    def best_hole(others, ncand=3000):
        cand = rng.randn(ncand, d)
        cand = _norm(cand)
        D = cand @ others.T
        md = D.max(axis=1)
        return cand[int(np.argmin(md))]

    # light initial refine
    try:
        if time.time() - t0 < BUDGET - 100:
            xc = run_lbfgs(x_best, 300.0, maxiter=50)
            vc = max_cos(xc)
            if vc < best_val:
                x_best, best_val = xc, vc
    except Exception:
        pass

    x_cur = x_best.copy()
    cur_val = best_val
    stall = 0
    trial = 0
    b1_pool = [150.0, 250.0, 350.0]
    b2_pool = [650.0, 1000.0, 1600.0]

    while time.time() < HOP_END and time.time() - t0 < BUDGET - 55:
        trial += 1
        if stall >= 16:
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
        mode = int(rng.randint(0, 10))
        x_init = None
        try:
            if mode in (0, 5):
                w = int(order[int(rng.randint(0, 4))])
                others = np.delete(x_cur, w, axis=0)
                nc = 3500 if (trial % 2 == 0) else 1500
                bh = best_hole(others, ncand=nc)
                x_init = x_cur.copy()
                x_init[w] = bh
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.003)
                step[w] = 0.006
                Gw = x_cur[w] @ x_cur.T
                nb = np.argsort(-Gw)[:8]
                step[nb] = np.maximum(step[nb], 0.006)
                x_init = x_init + step * P
                x_init = _norm(x_init)
            elif mode in (1, 6):
                Q = int(rng.choice(np.array([8, 14, 22, 32])))
                sig_w = float(rng.choice(np.array([0.045, 0.08, 0.12, 0.18])))
                idx_w = order[:Q]
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                step = np.full((n, 1), 0.003)
                step[idx_w] = sig_w
                sgn = np.sign(rng.randn(n, 1))
                sgn[sgn == 0] = 1.0
                step = step * sgn
                x_init = x_cur + step * P
                x_init = _norm(x_init)
            elif mode in (2, 7):
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                flat = int(np.argmax(Gc))
                a, b = flat // n, flat % n
                x_init = x_cur.copy()
                dot = float(np.dot(x_init[a], x_init[b]))
                s = float(rng.choice(np.array([0.02, 0.035, 0.06, 0.09])))
                da = -(x_init[b] - dot * x_init[a])
                db = -(x_init[a] - dot * x_init[b])
                da -= float(np.dot(da, x_init[a])) * x_init[a]
                db -= float(np.dot(db, x_init[b])) * x_init[b]
                da /= (np.linalg.norm(da) + 1e-12)
                db /= (np.linalg.norm(db) + 1e-12)
                x_init[a] = x_init[a] + s * da + 0.004 * rng.randn(d)
                x_init[b] = x_init[b] + s * db + 0.004 * rng.randn(d)
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.003 * P
                x_init = _norm(x_init)
            elif mode == 3:
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                k = 12
                part = np.partition(-Gc, k, axis=1)[:, :k]
                dens = (-part).sum(axis=1)
                c0 = int(np.argmax(dens))
                nn = np.argsort(-Gc[c0])[:int(rng.choice(np.array([10, 14, 18])))]
                x_init = x_cur.copy()
                s = float(rng.choice(np.array([0.03, 0.05, 0.08, 0.11])))
                for j in nn:
                    dot = float(np.dot(x_init[j], x_init[c0]))
                    dj = -(x_init[c0] - dot * x_init[j])
                    dj -= float(np.dot(dj, x_init[j])) * x_init[j]
                    dj /= (np.linalg.norm(dj) + 1e-12)
                    x_init[j] = x_init[j] + s * (0.5 + 0.5 * rng.rand()) * dj
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.003 * P
                x_init = _norm(x_init)
            elif mode == 4:
                sig = float(rng.choice(np.array([0.008, 0.012, 0.022, 0.032])))
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                x_init = x_cur + sig * P
                x_init = _norm(x_init)
            else:
                ratt = order[-20:]
                pick = rng.choice(ratt, size=2, replace=False)
                tmp = x_cur.copy()
                for w in pick:
                    others = np.delete(tmp, int(w), axis=0)
                    bh = best_hole(others, ncand=2000)
                    tmp[int(w)] = bh
                x_init = tmp
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.002)
                step[order[:10]] = 0.015
                x_init = x_init + step * P
                x_init = _norm(x_init)
        except Exception:
            continue
        b1 = float(b1_pool[int(rng.randint(0, 3))])
        b2 = float(b2_pool[int(rng.randint(0, 3))])
        try:
            xo = run_lbfgs(x_init, b1, maxiter=45)
            xo = run_lbfgs(xo, b2, maxiter=45)
        except Exception:
            continue
        v = max_cos(xo)
        if v < best_val - 1e-12:
            x_best, best_val = xo, v
            x_cur, cur_val = xo, v
            stall = 0
        else:
            tol = 0.0006 if stall < 8 else 0.0012
            if v < cur_val + tol and rng.rand() < 0.6:
                x_cur, cur_val = xo, v
                stall += 1
            else:
                if v < cur_val:
                    x_cur, cur_val = xo, v
                    stall += 1
                else:
                    stall += 1
        if trial > 140:
            break

    # high beta sharpening
    for hb, it in [(800.0, 70), (1600.0, 70), (3000.0, 60), (6000.0, 50)]:
        if time.time() - t0 > BUDGET - 32:
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
        if remain > 10:
            X = x_best.copy()
            bv = best_val
            for it in range(2500):
                if it % 200 == 0 and time.time() - t0 > BUDGET - 24:
                    break
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                a, b = flat // n, flat % n
                tgt = a if rng.rand() < 0.5 else b
                oth = b if tgt == a else a
                dot = float(np.dot(X[tgt], X[oth]))
                grad = -(X[oth] - dot * X[tgt])
                grad -= float(np.dot(grad, X[tgt])) * X[tgt]
                grad /= (np.linalg.norm(grad) + 1e-12)
                rr = rng.randn(d)
                rr -= float(np.dot(rr, X[tgt])) * X[tgt] + float(np.dot(rr, grad)) * grad
                rr /= (np.linalg.norm(rr) + 1e-12)
                step = float(rng.choice(np.array([0.0004, 0.0008, 0.0002, 0.0015])))
                tdir = grad * 0.85 + rr * 0.35
                tdir /= (np.linalg.norm(tdir) + 1e-12)
                Xt = X.copy()
                Xt[tgt] = Xt[tgt] + step * tdir
                Xt[tgt] /= (np.linalg.norm(Xt[tgt]) + 1e-12)
                G2 = Xt @ Xt.T
                np.fill_diagonal(G2, -2.0)
                nv = float(np.max(G2))
                if nv < bv:
                    X, bv = Xt, nv
            if bv < best_val:
                x_best, best_val = X, bv
    except Exception:
        pass

    # tangent trust-region LP polish
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 7:
            x = _norm(x_best)
            best = max_cos(x)
            # precompute eye
            E = np.eye(d)
            radius = 4e-4
            t_end = time.time() + min(remain - 1.0, 18.0)
            while time.time() < t_end and radius > 2e-12:
                G = x @ x.T
                cvec = G[iu0, iu1]
                m = float(cvec.max())
                # active set
                delta = max(0.003, 4.0 * (m - np.median(cvec) + 1e-12))
                # keep top pairs
                orderp = np.argsort(-cvec)
                K = min(len(cvec), 2200)
                sel = orderp[:K]
                # filter by delta
                keep = sel[cvec[sel] > m - delta]
                if len(keep) < 200:
                    keep = sel[:800]
                Ia = iu0[keep]
                Ja = iu1[keep]
                Cab = cvec[keep]
                # tangent bases
                T = np.zeros((n, d, dm))
                for i in range(n):
                    xi = x[i]
                    Ws = []
                    for kk in range(d):
                        w = E[:, kk] - float(E[:, kk] @ xi) * xi
                        Ws.append(w)
                    Ws = np.stack(Ws, axis=1)  # d x d
                    # gram-schmidt, drop weakest
                    Q = np.zeros((d, d))
                    cnt = 0
                    for kk in range(d):
                        w = Ws[:, kk].copy()
                        for qq in range(cnt):
                            w -= float(Q[:, qq] @ w) * Q[:, qq]
                        nw = float(np.linalg.norm(w))
                        if nw > 1e-8 and cnt < d:
                            # check not parallel: if this is the parallel dir, nw small
                            Q[:, cnt] = w / nw
                            cnt += 1
                    # Q should contain xi as one col (with nw~0 skipped); ensure orthogonality
                    # Take columns orthogonal to xi
                    dots = np.abs(Q.T @ xi)
                    idxs = np.argsort(dots)[:dm]
                    T[i] = Q[:, idxs]
                Nv = n * dm
                rows = len(keep)
                # build sparse A_ub
                rI = []
                cI = []
                dV = []
                for r in range(rows):
                    a = int(Ia[r]); b = int(Ja[r])
                    ca = T[a].T @ x[b]  # dm
                    cb = T[b].T @ x[a]
                    base_a = a * dm
                    for s_ in range(dm):
                        rI.append(r); cI.append(base_a + s_); dV.append(float(ca[s_]))
                    base_b = b * dm
                    for s_ in range(dm):
                        rI.append(r); cI.append(base_b + s_); dV.append(float(cb[s_]))
                    rI.append(r); cI.append(Nv); dV.append(-1.0)
                A_ub = coo_matrix((dV, (rI, cI)), shape=(rows, Nv + 1)).tocsr()
                b_ub = -Cab
                c_obj = np.zeros(Nv + 1)
                c_obj[Nv] = 1.0
                bounds = [(-radius, radius)] * Nv + [(m - 0.01, m)]
                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                                  options={"presolve": True})
                except Exception:
                    break
                if res.status != 0:
                    radius *= 0.5
                    continue
                y = np.asarray(res.x[:-1]).reshape(n, dm)
                dx = np.einsum("ijk,ik->ij", T, y)
                xn = _norm(x + dx)
                vn = max_cos(xn)
                if vn < best - 1e-12:
                    x = xn
                    best = vn
                    x_best, best_val = x, best
                    # keep radius
                else:
                    radius *= 0.5
                # slight radius decay to converge
                if radius < 1e-9:
                    break
    except Exception:
        pass

    x_best = _norm(x_best)
    return x_best
