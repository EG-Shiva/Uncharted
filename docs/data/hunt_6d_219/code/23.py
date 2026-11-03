import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(777)
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
                vn = np.linalg.norm(v, axis=1, keepdims=True)
                vn = np.maximum(vn, 1e-12)
                v = v / vn
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

    def fast_relax(X, iters=160, lr0=0.04, margin0=0.004):
        cur = X / np.linalg.norm(X, axis=1, keepdims=True)
        cur = cur.copy()
        bv = max_cos(cur)
        bX = cur.copy()
        lr = lr0
        margin = margin0
        for it in range(iters):
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            tau = margin * 0.25 + 1e-12
            W = np.exp((G - m) / tau)
            W[G < m - margin * 5.0] = 0.0
            np.fill_diagonal(W, 0.0)
            Grad = W @ cur
            proj = (Grad * cur).sum(axis=1, keepdims=True)
            Gt = Grad - proj * cur
            gn = float(np.max(np.linalg.norm(Gt, axis=1)))
            if not np.isfinite(gn) or gn < 1e-14:
                margin *= 0.6
                if margin < 5e-7:
                    break
                continue
            Xn = cur - (lr / gn) * Gt
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            mn = max_cos(Xn)
            if mn < m + 1e-12:
                cur = Xn
                if mn < bv:
                    bv = mn
                    bX = Xn.copy()
                if it % 60 == 0 and it > 0:
                    margin *= 0.9
            else:
                lr *= 0.55
                if lr < 2e-6:
                    lr = lr0 * 0.3
                    margin *= 0.7
                    if margin < 5e-7:
                        break
            if time.time() - t0 > BUDGET - 12:
                break
        return bX, bv

    def best_hole(others, ncand=3000):
        cand = rng.randn(ncand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ others.T
        md = D.max(axis=1)
        return cand[int(np.argmin(md))]

    def refine_hole(v, others, iters=120, lr0=0.05):
        v = v / (np.linalg.norm(v) + 1e-12)
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
            gt = g - float(np.dot(g, v)) * v
            nrm = float(np.linalg.norm(gt))
            if nrm < 1e-14 or not np.isfinite(nrm):
                break
            vn = v - (lr / nrm) * gt
            vn /= (np.linalg.norm(vn) + 1e-12)
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

    def lp_slide(x, radius=0.015, eps=2e-4, topk=2000):
        # sideways move along contact manifold: maximize random tangent direction
        try:
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(np.max(c))
            sel = c > m - 0.002
            I0 = iu0[sel]; J0 = iu1[sel]; cs = c[sel]
            if len(cs) > topk:
                part = np.argpartition(-cs, topk - 1)[:topk]
                I0, J0, cs = I0[part], J0[part], cs[part]
            p = len(I0)
            if p == 0:
                return x, False
            R = rng.randn(n, d)
            R -= (R * x).sum(axis=1, keepdims=True) * x
            rn = np.linalg.norm(R) + 1e-12
            R /= rn
            nv = n * d
            rows = np.repeat(np.arange(p), 2 * d)
            cols = np.concatenate([I0[:, None] * d + np.arange(d), J0[:, None] * d + np.arange(d)], 1).ravel()
            vals = np.concatenate([x[J0], x[I0]], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            b_ub = (m + eps) - cs
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = (-R.ravel())
            bounds = [(-radius, radius)] * (n * d)
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
            if res.status != 0:
                return x, False
            y = x + res.x.reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            return y, True
        except Exception:
            return x, False

    # light initial refine
    try:
        if time.time() - t0 < BUDGET - 110:
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
    HOP_END = t0 + 118.0
    b1_pool = [120.0, 200.0, 320.0]
    b2_pool = [600.0, 900.0, 1400.0]

    while time.time() < HOP_END and time.time() - t0 < BUDGET - 50:
        trial += 1
        if stall >= 20:
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
        # occasional LP slide to wander on manifold
        if trial % 9 == 0:
            try:
                ysl, ok = lp_slide(x_cur, radius=0.02, eps=3e-4)
                if ok:
                    ysl, _ = fast_relax(ysl, iters=80, lr0=0.03, margin0=0.003)
                    vsl = max_cos(ysl)
                    if vsl < cur_val + 0.0008:
                        x_cur, cur_val = ysl, vsl
                        if vsl < best_val - 1e-12:
                            # sharpen it
                            yo = run_lbfgs(ysl, 800.0, maxiter=50)
                            vo = max_cos(yo)
                            if vo < best_val:
                                x_best, best_val = yo, vo
                                x_cur, cur_val = yo, vo
                                stall = 0
                                continue
                    # fall through to normal move from new cur
            except Exception:
                pass
        mode = rng.randint(0, 10)
        x_init = None
        try:
            if mode in (0, 5):
                w = int(order[rng.randint(0, 4)])
                others = np.delete(x_cur, w, axis=0)
                nc = 3500 if (trial % 2 == 0) else 1500
                bh = best_hole(others, ncand=nc)
                bh, _ = refine_hole(bh, others, iters=80)
                x_init = x_cur.copy()
                x_init[w] = bh
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.003)
                step[w] = 0.005
                Gw = x_cur[w] @ x_cur.T
                nb = np.argsort(-Gw)[:8]
                step[nb] = np.maximum(step[nb], 0.006)
                x_init = x_init + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (1, 6):
                Q = int(rng.choice([8, 14, 22, 32]))
                sig_w = float(rng.choice([0.05, 0.09, 0.13, 0.18]))
                idx_w = order[:Q]
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                step = np.full((n, 1), 0.0025)
                step[idx_w] = sig_w
                sgn = np.sign(rng.randn(n, 1))
                sgn[sgn == 0] = 1.0
                step = step * sgn
                x_init = x_cur + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode in (2, 7):
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                flat = int(np.argmax(Gc))
                a, b = flat // n, flat % n
                x_init = x_cur.copy()
                dot = float(np.dot(x_init[a], x_init[b]))
                s = float(rng.choice([0.02, 0.04, 0.065]))
                da = -(x_init[b] - dot * x_init[a])
                db = -(x_init[a] - dot * x_init[b])
                da -= float(np.dot(da, x_init[a])) * x_init[a]
                db -= float(np.dot(db, x_init[b])) * x_init[b]
                da /= (np.linalg.norm(da) + 1e-12)
                db /= (np.linalg.norm(db) + 1e-12)
                x_init[a] = x_init[a] + s * da + 0.003 * rng.randn(d)
                x_init[b] = x_init[b] + s * db + 0.003 * rng.randn(d)
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.0025 * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode == 3:
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                k = 12
                part = np.partition(-Gc, k, axis=1)[:, :k]
                dens = (-part).sum(axis=1)
                c0 = int(np.argmax(dens))
                nn = np.argsort(-Gc[c0])[:int(rng.choice([10, 14, 18]))]
                x_init = x_cur.copy()
                s = float(rng.choice([0.03, 0.055, 0.085]))
                for j in nn:
                    dot = float(np.dot(x_init[j], x_init[c0]))
                    dj = -(x_init[c0] - dot * x_init[j])
                    dj -= float(np.dot(dj, x_init[j])) * x_init[j]
                    dj /= (np.linalg.norm(dj) + 1e-12)
                    x_init[j] = x_init[j] + s * (0.5 + 0.5 * rng.rand()) * dj
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = x_init + 0.0025 * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            elif mode == 4:
                sig = float(rng.choice([0.008, 0.015, 0.025, 0.035]))
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                x_init = x_cur + sig * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            else:
                ratt = order[-20:]
                pick = rng.choice(ratt, size=2, replace=False)
                tmp = x_cur.copy()
                for w in pick:
                    others = np.delete(tmp, int(w), axis=0)
                    bh = best_hole(others, ncand=2000)
                    bh, _ = refine_hole(bh, others, iters=60)
                    tmp[int(w)] = bh
                x_init = tmp
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.002)
                step[order[:10]] = 0.015
                x_init = x_init + step * P
                x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        except Exception:
            continue
        # fast screen
        try:
            xs, vs = fast_relax(x_init, iters=120, lr0=0.035, margin0=0.004)
        except Exception:
            continue
        # only LBFGS promising or lucky
        do_lbfgs = (vs < cur_val + 0.0015) or (rng.rand() < 0.12)
        if not do_lbfgs:
            stall += 1
            continue
        b1 = float(b1_pool[rng.randint(0, 3)])
        b2 = float(b2_pool[rng.randint(0, 3)])
        try:
            xo = run_lbfgs(xs, b1, maxiter=45)
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
        if trial > 200:
            break

    for hb, it in [(800.0, 70), (1600.0, 70), (3000.0, 50)]:
        if time.time() - t0 > BUDGET - 28:
            break
        try:
            xo = run_lbfgs(x_best, hb, maxiter=it)
            v = max_cos(xo)
            if v < best_val:
                x_best, best_val = xo, v
        except Exception:
            continue

    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 8:
            X = x_best.copy()
            bv = best_val
            for it in range(2500):
                if it % 200 == 0 and time.time() - t0 > BUDGET - 20:
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
                step = float(rng.choice([0.0004, 0.0008, 0.0002, 0.0015]))
                td = grad * 0.85 + rr * 0.35
                td /= (np.linalg.norm(td) + 1e-12)
                Xt = X.copy()
                Xt[tgt] = Xt[tgt] + step * td
                Xt[tgt] /= (np.linalg.norm(Xt[tgt]) + 1e-12)
                nv = max_cos(Xt)
                if nv < bv:
                    X, bv = Xt, nv
            if bv < best_val:
                x_best, best_val = X, bv
    except Exception:
        pass

    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 7:
            x = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
            best = max_cos(x)
            radius = 8e-4
            t_end = time.time() + min(remain - 1.0, 16.0)
            while time.time() < t_end and radius > 1e-13:
                Gfull = x @ x.T
                c = Gfull[iu0, iu1]
                m = float(np.max(c))
                thresh = m - max(20.0 * radius, 1e-9)
                sel = c > thresh
                I = iu0[sel]; J = iu1[sel]; csel = c[sel]
                p = len(I)
                if p == 0:
                    break
                if p > 3000:
                    k = 3000
                    part = np.argpartition(-csel, k - 1)[:k]
                    I, J, csel = I[part], J[part], csel[part]
                    p = k
                nv = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
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
                v = max_cos(y)
                if v < best:
                    x, best = y, v
                    x_best, best_val = y, v
                    radius = min(radius * 1.5, 0.02)
                else:
                    radius *= 0.5
    except Exception:
        pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
