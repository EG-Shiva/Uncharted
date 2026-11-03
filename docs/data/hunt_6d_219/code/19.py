import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _fast_relax(X, iters, lr0=0.025, margin0=0.004, noise0=0.0, rng=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
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
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        step = (lr / gn) * Gt
        if noise0 > 0 and rng is not None:
            decay = 1.0 - it / float(iters)
            nz = rng.randn(*cur.shape)
            nz -= (nz * cur).sum(1, keepdims=True) * cur
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            step = step + (noise0 * decay / np.sqrt(6)) * (nz / nn) * gn * 0.2
            # scale: keep noise proportional to lr
        Xn = cur - step
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m + (1e-12 if noise0 == 0 else 5e-4 * (1.0 - it / iters)):
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
            W = np.zeros((n, n), dtype=float)
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
        radius = 1.1e-3
        while time.time() < deadline and time.time() - t0 < BUDGET - 2:
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

    def refine_void_single(v0, others, iters=200, lr0=0.05):
        v = v0 / np.linalg.norm(v0)
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
            vn /= np.linalg.norm(vn)
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
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ others.T
            worst = dots.max(axis=1)
            order = np.argsort(worst)
            # take best of few and refine top-3, keep best
            bv = None; bs = 1e9
            for bi in order[:3]:
                rv, rs = refine_void_single(cand[bi], others, iters=180, lr0=0.06)
                if rs < bs:
                    bs = rs; bv = rv
            Y[w] = bv
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    # light initial tighten
    try:
        Xi, vi = _fast_relax(x_best, iters=600, lr0=0.02, margin0=0.003)
        if vi < best_val:
            x_best, best_val = Xi, vi
    except Exception:
        pass

    trial = 0
    stall = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 50:
        trial += 1
        mode = trial % 6
        ppm, Gfull = _row_max(x_best)
        # contact counts at 1e-7 slack for targeting
        thr_c = best_val - 1e-7
        cc = (Gfull > thr_c).sum(axis=1)
        order = np.argsort(-(ppm + 0.15 * (cc / max(1, cc.max())) * 0.01 + 1e-6 * rng.rand(n)))
        x_init = x_best.copy()
        if mode == 0:
            k = 10
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.13 * (0.7 + 0.6 * rng.rand(k, 1))
            x_init = x_init + sig * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        elif mode == 1:
            nw = 2 if (rng.rand() < 0.5) else (3 if rng.rand() < 0.7 else 4)
            # mix worst + high-contact random
            victims = list(order[:nw])
            if rng.rand() < 0.5:
                hi = np.where(cc >= np.sort(cc)[-30])[0]
                extra = hi[rng.randint(len(hi))]
                if extra not in victims:
                    victims[-1] = int(extra)
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + 0.006 * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            try:
                x_init = kill_fill(x_init, victims, n_cand=5000)
            except Exception:
                pass
        elif mode == 2:
            # shatter contact cluster around worst point
            w0 = int(order[0])
            Gj = Gfull[w0].copy()
            Gj[w0] = -2
            neigh = np.argsort(-Gj)[:int(rng.randint(8, 16))]
            clust = np.concatenate([[w0], neigh])
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            sig = np.full((n, 1), 0.004)
            sig[clust] = 0.055 + 0.03 * rng.rand()
            # coherent push: add common random tangent direction to cluster
            common = rng.randn(d)
            common -= (common * x_init[w0]) * x_init[w0]
            common /= (np.linalg.norm(common) + 1e-12)
            x_init[clust] += 0.03 * common[None, :]
            x_init = x_init + sig * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        elif mode == 3:
            sigma = 0.011
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + sigma * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        elif mode == 4:
            # heavy-tail Cauchy kick on random subset
            k = int(rng.randint(20, 40))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.standard_cauchy((n, d)) * 0.15
            pert = np.clip(pert, -3, 3)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
            pert = pert / pn * np.minimum(pn, 1.5)
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.05
            x_init = x_init + sig * pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        else:
            # orthogonalize worst few vs their worst neighbor
            k = int(rng.randint(3, 7))
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert * x_init).sum(1, keepdims=True) * x_init
            x_init = x_init + 0.018 * pert
            G = x_init @ x_init.T
            np.fill_diagonal(G, -2)
            for w in idx:
                j = int(np.argmax(G[w]))
                v = x_init[w] - (x_init[w] @ x_init[j]) * x_init[j]
                nv = float(np.linalg.norm(v))
                if nv > 1e-9:
                    v /= nv
                    alpha = 0.3 + 0.4 * rng.rand()
                    x_init[w] = (1 - alpha) * x_init[w] + alpha * v
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)

        use_noise = (mode in (2, 3, 4)) and (rng.rand() < 0.6)
        try:
            if use_noise:
                Xr, vr = _fast_relax(x_init, iters=650, lr0=0.025, margin0=0.0045,
                                     noise0=0.004, rng=rng)
            else:
                Xr, vr = _fast_relax(x_init, iters=650, lr0=0.022, margin0=0.004)
        except Exception:
            continue
        if vr > best_val + 0.009 and rng.rand() < 0.85:
            stall += 1
            continue
        # sharpen
        try:
            beta1 = 160.0 if (trial % 2 == 0) else 240.0
            Xo = run_lbfgs(Xr, beta1, maxiter=70)
            v = _max_cos(Xo)
            if vr < best_val + 0.003 or (trial % 3 == 0):
                beta2 = 450.0 if (trial % 3 == 0) else 340.0
                Xo2 = run_lbfgs(Xo, beta2, maxiter=70)
                v2 = _max_cos(Xo2)
                if v2 < v:
                    Xo, v = Xo2, v2
        except Exception:
            continue
        if v < best_val - 1e-12:
            x_best = Xo
            best_val = v
            stall = 0
            try:
                Xp, vp = lp_polish(x_best, deadline=time.time() + 2.0)
                if vp < best_val:
                    x_best, best_val = Xp, vp
            except Exception:
                pass
        else:
            stall += 1
            # simulated-annealing style sideways move if close
            if v < best_val + 0.0015 and rng.rand() < 0.12:
                x_best = Xo
                # keep best_val unchanged (don't raise bar)
        if stall > 25:
            # diversification burst: big shake of current (not just best basin)
            try:
                pert = rng.randn(n, d)
                pert -= (pert * x_best).sum(1, keepdims=True) * x_best
                x_best = x_best + 0.03 * pert
                x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
                stall = 0
            except Exception:
                pass
        if time.time() - t0 > BUDGET - 50:
            break

    for beta, mi in [(150.0, 110), (330.0, 110), (650.0, 110)]:
        if time.time() - t0 > BUDGET - 36:
            break
        try:
            Xo = run_lbfgs(x_best, beta, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val:
                x_best, best_val = Xo, v
        except Exception:
            pass

    if time.time() < t0 + BUDGET - 4:
        try:
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 1.5)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
