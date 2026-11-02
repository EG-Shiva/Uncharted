import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]
    J0 = iu0[1]
    mP = len(I0)

    def lse_polish(X0, tau, maxiter):
        if time.time() > DEADLINE - 7:
            return X0
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mx = float(g.max())
            e = np.exp((g - mx) / tau)
            S = float(e.sum()) + 1e-300
            F = mx + tau * np.log(S)
            w = e / S
            Wx = np.zeros_like(X)
            # scatter-add
            # vectorized via bincount-like: use np.add.at
            np.add.at(Wx, I0, (w[:, None] * X[J0]))
            np.add.at(Wx, J0, (w[:, None] * X[I0]))
            dot = np.sum(Wx * X, axis=1, keepdims=True)
            gradY = (Wx - X * dot) / r
            return F, gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-7, "maxfun": 10000})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        if not np.all(np.isfinite(Y)):
            return X0
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y

    def hinge_polish(X0, T, maxiter):
        if time.time() > DEADLINE - 7:
            return X0
        K = 1e4
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            diff = g - T
            pos = diff > 0
            if np.any(pos):
                dp = diff[pos] * K
                F = float(np.sum(dp * dp))
                w = np.zeros_like(g)
                w[pos] = 2.0 * K * K * diff[pos]
            else:
                F = 0.0
                w = np.zeros_like(g)
            Wx = np.zeros_like(X)
            if np.any(pos):
                np.add.at(Wx, I0, (w[:, None] * X[J0]))
                np.add.at(Wx, J0, (w[:, None] * X[I0]))
                dot = np.sum(Wx * X, axis=1, keepdims=True)
                gradY = (Wx - X * dot) / r
            else:
                gradY = np.zeros_like(X)
            return F, gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-8, "maxfun": 8000})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        if not np.all(np.isfinite(Y)):
            return X0
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() > DEADLINE - 4:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
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
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def worst_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.argsort(-G.max(axis=1)), G

    def farthest_reinsert(X, idx):
        cand = _normalize(rng.standard_normal((2500, 13)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    # initial deep squeeze
    Xb = best.copy()
    Xb = lse_polish(Xb, 0.004, 80)
    Xb = lse_polish(Xb, 0.0012, 120)
    Xb, vb = greedy(Xb, 350, 0.002)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)

    basin = 0
    while time.time() < DEADLINE - 9:
        if basin > 300:
            break
        remain = DEADLINE - time.time()
        # choose perturbation
        Xcur = best.copy()
        mode = basin % 7
        if mode == 0:
            Xcur = _normalize(Xcur + 0.016 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            order, G = worst_order(Xcur)
            k = 6 + (basin % 5)
            idx = order[:k]
            noise = 0.003 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.028 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + noise)
            # split worst pair
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] -= 0.012 * Xcur[jj]
            Xcur[jj] -= 0.012 * Xcur[ii]
            Xcur = _normalize(Xcur)
        elif mode == 2:
            order, _ = worst_order(Xcur)
            Xcur = farthest_reinsert(Xcur, int(order[0]))
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            Xcur = _normalize(Xcur + 0.007 * rng.standard_normal(Xcur.shape))
            order, G = worst_order(Xcur)
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] -= 0.010 * Xcur[jj]
            Xcur[jj] -= 0.010 * Xcur[ii]
            Xcur = _normalize(Xcur)
        elif mode == 4:
            order, _ = worst_order(Xcur)
            Xcur = farthest_reinsert(Xcur, int(order[0]))
            Xcur = farthest_reinsert(Xcur, int(order[1]))
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        elif mode == 5:
            # block shake of random subset
            k = 10 + (basin % 7)
            idx = rng.choice(n, size=k, replace=False)
            noise = 0.002 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.020 * rng.standard_normal((k, d))
            Xcur = _normalize(Xcur + noise)
        else:
            # hinge target mode: stronger shake + hinge chase
            Xcur = _normalize(Xcur + 0.010 * rng.standard_normal(Xcur.shape))
            # will run hinge below instead of plain LSE
            pass

        if remain < 25:
            # fast track
            Xcur = lse_polish(Xcur, 0.002, 50)
            Xcur = lse_polish(Xcur, 0.0008, 60)
            Xcur, vcur = greedy(Xcur, 200, 0.0015)
        else:
            if mode == 6:
                T = best_val - 4e-07
                Xcur = hinge_polish(Xcur, T, 100)
                Xcur = lse_polish(Xcur, 0.001, 60)
                Xcur, vcur = greedy(Xcur, 250, 0.002)
            else:
                Xcur = lse_polish(Xcur, 0.004, 50)
                Xcur = lse_polish(Xcur, 0.001, 70)
                Xcur, vcur = greedy(Xcur, 250, 0.002)

        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
            # extra squeeze immediately
            if time.time() < DEADLINE - 12:
                Xt = lse_polish(best.copy(), 0.0006, 70)
                Xt, vt = greedy(Xt, 250, 0.0008)
                if vt < best_val:
                    best, best_val = Xt.copy(), float(vt)
        basin += 1

    if time.time() < DEADLINE - 3:
        Xf = best.copy()
        Xf, vf = greedy(Xf, 250, 0.0006)
        if vf < best_val:
            best, best_val = Xf, vf

    best = _normalize(best)
    return best
