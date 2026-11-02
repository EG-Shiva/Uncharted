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
    LIM = 172.0
    rng = np.random.default_rng(20261102)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 74, 13

    # ---------- helpers ----------
    def adam_lse(X, tau, steps, lr0, m, v, cnt):
        for s in range(steps):
            if time.time() - t0 > LIM - 6:
                break
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gmax = float(G.max())
            E = np.exp((G - gmax) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.isfinite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            m[:] = 0.9 * m + 0.1 * Gt
            v[:] = 0.999 * v + 0.001 * Gt * Gt
            cnt[0] += 1
            c1 = 1.0 - 0.9 ** cnt[0]
            c2 = 1.0 - 0.999 ** cnt[0]
            mh = m / c1
            vh = v / c2
            lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
            X -= lr * mh / (np.sqrt(vh) + 1e-8)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def greedy_active(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() - t0 > LIM - 5:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 1e-9)
            mask = G > (gmax - delta)
            Gr = mask.astype(np.float64) @ X
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            nr = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = nr[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / nr[act] * lr
            Xn = _normalize(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        return X, cur

    def lbfgs_lse(X, tau, maxiter):
        iu0 = np.triu_indices(n, 1)
        I0, J0 = iu0
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            Xn = Y / r
            G = Xn @ Xn.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum()) + 1e-300
            F = mmax + tau * np.log(S)
            w = e / S
            Wx = np.zeros_like(Xn)
            np.add.at(Wx, I0, w[:, None] * Xn[J0])
            np.add.at(Wx, J0, w[:, None] * Xn[I0])
            dot = (Wx * Xn).sum(axis=1, keepdims=True)
            gY = (Wx - Xn * dot) / r
            return F, gY.ravel()
        try:
            res = minimize(fun, X.ravel().copy(), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-9, "maxfun": 10000})
            Y = res.x.reshape(n, d)
            Xn = _normalize(Y)
            if np.all(np.isfinite(Xn)):
                return Xn
        except Exception:
            pass
        return X

    def worst_indices(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(mx)[-k:]

    def farthest_reinsert(X, idx):
        cand = _normalize(rng.standard_normal((3000, 13)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    def maxpair_split(X, eps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        f = int(np.argmax(G))
        i, j = divmod(f, n)
        Xi = X[i] - eps * X[j]
        Xj = X[j] - eps * X[i]
        Xi /= np.linalg.norm(Xi)
        Xj /= np.linalg.norm(Xj)
        X[i] = Xi
        X[j] = Xj
        return X

    def micro_split_polish(X, cur):
        eps_list = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8]
        for _ in range(400):
            if time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            f = int(np.argmax(G))
            i, j = divmod(f, n)
            best_nv = cur
            best_e = 0.0
            # try two-sided move
            for e in eps_list:
                Xt = X.copy()
                a = X[i] - e * X[j]
                b = X[j] - e * X[i]
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                Xt[i] = a
                Xt[j] = b
                nv = _max_cos(Xt)
                if nv < best_nv:
                    best_nv = nv
                    best_e = e
            if best_e > 0:
                a = X[i] - best_e * X[j]
                b = X[j] - best_e * X[i]
                X[i] = a / np.linalg.norm(a)
                X[j] = b / np.linalg.norm(b)
                cur = best_nv
            else:
                break
        return X, cur

    # ---------- phase 1: squeeze start ----------
    m = np.zeros_like(best); v = np.zeros_like(best); cnt = [0]
    X0 = best.copy()
    for tau, steps, lr0 in [(0.015, 350, 0.010), (0.007, 500, 0.008), (0.003, 700, 0.005), (0.0012, 600, 0.003)]:
        if time.time() - t0 > LIM - 30:
            break
        X0 = adam_lse(X0, tau, steps, lr0, m, v, cnt)
        X0, val0 = greedy_active(X0, 250, 0.002)
        if val0 < best_val:
            best, best_val = X0.copy(), val0
    # one L-BFGS at small tau
    if time.time() - t0 < LIM - 40:
        try:
            Xl = lbfgs_lse(best.copy(), 0.002, 120)
            Xl, vall = greedy_active(Xl, 250, 0.0015)
            if vall < best_val:
                best, best_val = Xl.copy(), vall
                X0 = Xl.copy()
            else:
                X0 = best.copy()
        except Exception:
            X0 = best.copy()
    else:
        X0 = best.copy()

    # ---------- phase 2: basin hopping ----------
    it = 0
    while time.time() - t0 < LIM - 12:
        it += 1
        remain = LIM - (time.time() - t0)
        # choose budget
        if remain < 20:
            sched = [(0.003, 250, 0.004), (0.001, 250, 0.002)]
            gsteps = 150
        else:
            sched = [(0.006, 300, 0.007), (0.002, 400, 0.004), (0.0008, 300, 0.0025)]
            gsteps = 220
        Xc = best.copy()
        mode = it % 6
        if mode == 0:
            Xc = _normalize(Xc + 0.012 * rng.standard_normal(Xc.shape))
        elif mode == 1:
            wi = worst_indices(Xc, 8)
            noise = rng.standard_normal(Xc.shape)
            Xc[wi] += 0.06 * noise[wi]
            Xc = _normalize(Xc + 0.004 * rng.standard_normal(Xc.shape) * 0.2)
            Xc = _normalize(Xc)
        elif mode == 2:
            wi = worst_indices(Xc, 1)[0]
            # 50% farthest reinsert, 50% random hole
            if it % 2 == 0:
                farthest_reinsert(Xc, int(wi))
            else:
                Xc[int(wi)] = _normalize(rng.standard_normal((1, 13)))[0]
            Xc = _normalize(Xc + 0.003 * rng.standard_normal(Xc.shape))
        elif mode == 3:
            maxpair_split(Xc, float(rng.choice([0.02, 0.01, 0.005])))
            Xc = _normalize(Xc + 0.008 * rng.standard_normal(Xc.shape))
        elif mode == 4:
            # kick random subset
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            Xc[idx] = _normalize(rng.standard_normal((k, 13)) * 0.5 + Xc[idx])
            Xc = _normalize(Xc)
        else:
            Xc = _normalize(Xc + 0.030 * rng.standard_normal(Xc.shape))
        m2 = np.zeros_like(Xc); v2 = np.zeros_like(Xc); c2 = [0]
        for tau, steps, lr0 in sched:
            Xc = adam_lse(Xc, tau, steps, lr0, m2, v2, c2)
            if time.time() - t0 > LIM - 8:
                break
        Xc, valc = greedy_active(Xc, gsteps, 0.0015)
        # micro split try
        Xc, valc = micro_split_polish(Xc, valc)
        if valc < best_val:
            best, best_val = Xc.copy(), valc

    # ---------- phase 3: final intensive polish ----------
    for _ in range(3):
        if time.time() - t0 > LIM - 3:
            break
        m3 = np.zeros_like(best); v3 = np.zeros_like(best); c3 = [0]
        best = adam_lse(best, 0.0007, 400, 0.002, m3, v3, c3)
        best, best_val_new = greedy_active(best, 400, 0.001)
        best, best_val_new = micro_split_polish(best, best_val_new)
        best_val = min(best_val, best_val_new)
        if time.time() - t0 > LIM - 3:
            break
    best = _normalize(best)
    return best
