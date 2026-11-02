import time
import numpy as np
from scipy.optimize import minimize, linprog

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((74, 13))))
    vals = [_mc(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    def smooth_polish(X0, tau, maxiter):
        if time.time() - t0 > LIM - 8:
            return X0
        nloc, dloc = X0.shape
        def fun(y_flat):
            Y = y_flat.reshape(nloc, dloc)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = g.max()
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = np.sum(Wx * X, axis=1, keepdims=True)
            grad_Y = (Wx - X * dot) / r
            return F, grad_Y.ravel()
        y0 = X0.ravel().copy()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-9, "maxfun": 20000})
            Y = res.x.reshape(nloc, dloc)
        except Exception:
            return X0
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        if not np.all(np.isfinite(X)):
            return X0
        return X

    def minimax(X, steps, lr0):
        cur = _mc(X)
        lr = lr0
        for it in range(steps):
            if (it & 15) == 0 and time.time() - t0 > LIM - 5:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 8e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _norm(X - step)
            nv = _mc(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 5e-12:
                    break
        return X, cur

    def micro(X, seconds):
        bestl = _mc(X)
        start = time.time()
        step = 2.5e-6
        while time.time() - start < seconds and time.time() - t0 < LIM - 3:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni; gj /= nj
            improved = False
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                v = _mc(Y)
                if v < bestl:
                    X[:] = Y; bestl = v; improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-11:
                    break
        return X, bestl

    def lp_polish(X, seconds, radius_init=1e-3):
        X = _norm(X)
        bestl = _mc(X)
        radius = radius_init
        start = time.time()
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() - t0 < LIM - 3 and radius > 1e-14:
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0 * radius, 2e-9)
            if np.count_nonzero(sel) > 2200:
                thr = np.sort(Cfull)[-2200]
                sel = Cfull >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            try:
                from scipy.sparse import coo_matrix
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _norm(Y)
            v = _mc(Y)
            if v < bestl:
                X, bestl = Y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, bestl

    def farthest(X, exclude, n_cand=5000):
        cand = _norm(rng.standard_normal((n_cand, d)))
        S = cand @ X.T
        if exclude is not None:
            if isinstance(exclude, (list, np.ndarray)):
                S[:, list(exclude)] = -10.0
            else:
                S[:, int(exclude)] = -10.0
        sc = S.max(axis=1)
        return cand[int(np.argmin(sc))]

    def worst_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.argsort(-G.max(axis=1)), float(G.max())

    def anneal(X):
        # fast annealed softmax + minimax, time-aware
        remain = LIM - (time.time() - t0)
        if remain > 60:
            X = smooth_polish(X, 0.006, 180)
            X = smooth_polish(X, 0.002, 220)
            X, _ = minimax(X, 120, 3e-4)
            X = smooth_polish(X, 0.0006, 260)
        elif remain > 25:
            X = smooth_polish(X, 0.003, 150)
            X = smooth_polish(X, 0.0008, 200)
            X, _ = minimax(X, 80, 2e-4)
        else:
            X = smooth_polish(X, 0.001, 150)
            X, _ = minimax(X, 60, 2e-4)
        return X

    # initial squeeze of incumbent
    Xb = best.copy()
    Xb = anneal(Xb)
    Xb, vb = micro(Xb, 0.6)
    Xb, vb = lp_polish(Xb, 1.2, 8e-4)
    vb = _mc(Xb)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)

    basin = 0
    while time.time() - t0 < LIM - 10:
        if basin > 400:
            break
        Xc = best.copy()
        mode = basin % 8
        if mode == 0:
            Xc = _norm(Xc + 0.008 * rng.standard_normal(Xc.shape))
        elif mode == 1:
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            cnt = (G > gm - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            k = 5 + (basin % 5)
            idx = order[:k]
            Ns = 0.003 * rng.standard_normal(Xc.shape)
            Ns[idx] += 0.025 * rng.standard_normal((len(idx), d))
            Xc = _norm(Xc + Ns)
        elif mode == 2:
            wo, _ = worst_order(Xc)
            a = int(wo[0])
            Xc[a] = farthest(Xc, a, 6000)
            Xc = _norm(Xc + 0.002 * rng.standard_normal(Xc.shape))
        elif mode == 3:
            wo, _ = worst_order(Xc)
            for t in range(2):
                a = int(wo[t])
                Xc[a] = farthest(Xc, a, 5000)
            Xc = _norm(Xc + 0.002 * rng.standard_normal(Xc.shape))
        elif mode == 4:
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G)); i = k // n; j = k % n
            Xc[i] -= 0.015 * Xc[j]
            Xc[j] -= 0.015 * Xc[i]
            Xc = _norm(Xc + 0.004 * rng.standard_normal(Xc.shape))
        elif mode == 5:
            # cluster move: pick random direction, shift nearby points together
            v = _norm(rng.standard_normal((1, d)))[0]
            proj = Xc @ v
            thr = np.quantile(proj, rng.uniform(0.3, 0.7))
            sel = proj > thr if rng.random() < 0.5 else proj < thr
            sh = rng.standard_normal(d); sh -= float(sh @ v) * v
            nn = float(np.linalg.norm(sh))
            if nn > 1e-9:
                sh /= nn
                Xc[sel] = _norm(Xc[sel] + 0.010 * sh)
            Xc = _norm(Xc + 0.003 * rng.standard_normal(Xc.shape))
        elif mode == 6:
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            Xc[idx] = _norm(rng.standard_normal((k, d)))
        else:
            Xc = _norm(Xc + 0.014 * rng.standard_normal(Xc.shape))

        Xc = anneal(Xc)
        # opportunistic hole fix on worst point if still bad
        cur = _mc(Xc)
        if cur > best_val - 1e-6:
            wo, _ = worst_order(Xc)
            # try single hole relocation + quick rebalance
            for a in [int(wo[0])]:
                y = farthest(Xc, a, 4000)
                Xt = Xc.copy(); Xt[a] = y
                Xt = smooth_polish(Xt, 0.002, 120)
                vt = _mc(Xt)
                if vt < cur:
                    Xc = Xt; cur = vt
        Xc, cur = micro(Xc, 0.35)
        # occasional LP for promising basins
        if cur < best_val + 2e-6 and time.time() - t0 < LIM - 20:
            Xc, cur = lp_polish(Xc, 0.8, 6e-4)
        else:
            cur = _mc(Xc)
        if cur < best_val - 1e-12:
            best, best_val = Xc.copy(), float(cur)
        basin += 1

    # final polish on best
    if time.time() - t0 < LIM - 4:
        best, _ = micro(best, 0.5)
    if time.time() - t0 < LIM - 4:
        best, _ = lp_polish(best, max(0.5, LIM - (time.time() - t0) - 2.0), 5e-4)
    best = _norm(best)
    return best
