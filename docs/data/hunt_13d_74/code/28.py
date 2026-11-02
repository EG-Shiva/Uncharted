import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

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
    rng = np.random.default_rng(777)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU = np.triu_indices(n, 1)
    I0, J0 = IU[0].astype(np.int64), IU[1].astype(np.int64)

    def lbfgs_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.sqrt((Y * Y).sum(axis=1, keepdims=True))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.finite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=np.float64)
            W[I0, J0] = w
            W[J0, I0] = w
            Wx = W @ X
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 2:
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

    def micro_squeeze(X, cur, budget=0.25):
        start = time.time()
        step = 2.5e-6
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            ii, jj = divmod(k, n)
            c = float(G[ii, jj])
            xi = X[ii].copy(); xj = X[jj].copy()
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
                    Y[ii] = xi - step * gi
                    Y[jj] = xj - step * gj
                elif mode == 1:
                    Y[ii] = xi - step * 2.0 * gi
                else:
                    Y[jj] = xj - step * 2.0 * gj
                Y[ii] /= np.linalg.norm(Y[ii]); Y[jj] /= np.linalg.norm(Y[jj])
                v = _max_cos(Y)
                if v < cur:
                    X[:] = Y; cur = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-11:
                    break
        return X, cur

    def anneal(X, taus_iters):
        for tau, it in taus_iters:
            if time.time() - t0 > LIM - 6:
                break
            X[:] = lbfgs_polish(X, tau, it)
        return X

    def lp_polish(X, budget, radius_init=1e-3):
        start = time.time()
        cur = _max_cos(X)
        radius = float(radius_init)
        iu = IU
        nv = n * d + 1
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
            if radius < 1e-13:
                break
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0 * radius, 1e-9)
            if int(np.count_nonzero(sel)) > 2200:
                thr = float(np.sort(Cfull)[-2200])
                sel = Cfull >= thr
            I = iu[0][sel].astype(np.int64); J = iu[1][sel].astype(np.int64)
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
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
            Y = _normalize(Y)
            if not np.all(np.isfinite(Y)):
                radius *= 0.5
                continue
            v = _max_cos(Y)
            if v < cur:
                X[:] = Y
                cur = v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    def contacts_info(X, eps=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        C = G >= (gmax - eps)
        cnt = C.sum(axis=1)
        mx = G.max(axis=1)
        worst_order = np.argsort(-mx)
        contact_order = np.argsort(-cnt)
        rattlers = np.where(cnt == 0)[0]
        return gmax, cnt, worst_order, contact_order, rattlers

    def hole_replace(X, idx, M=6000):
        cand = _normalize(rng.standard_normal((M, d)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    def split_worst(X, alpha=0.012):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        ii, jj = divmod(k, n)
        c = float(G[ii, jj])
        gi = X[jj] - c * X[ii]
        gj = X[ii] - c * X[jj]
        gi /= (np.linalg.norm(gi) + 1e-12); gj /= (np.linalg.norm(gj) + 1e-12)
        X[ii] = X[ii] - alpha * gi
        X[jj] = X[jj] - alpha * gj
        X[ii] /= np.linalg.norm(X[ii]); X[jj] /= np.linalg.norm(X[jj])
        return X

    def isotropize(X, strength=0.4):
        try:
            M = X.T @ X
            w, V = np.linalg.eigh(M)
            w = np.maximum(w, 1e-8)
            target = n / d
            f = np.sqrt((1.0 - strength) + strength * target / w)
            T = (V * f) @ V.T
            Y = X @ T
            return _normalize(Y)
        except Exception:
            return X

    # initial polish
    Xb = best.copy()
    Xb = anneal(Xb, [(0.006, 60), (0.002, 80), (0.0008, 80)])
    Xb, vb = greedy(Xb, 250, 0.002)
    Xb, vb = micro_squeeze(Xb, vb, budget=0.4)
    Xb, vb = lp_polish(Xb, budget=2.0, radius_init=8e-4)
    Xb, vb = greedy(Xb, 150, 0.0006)
    if vb < best_val:
        best, best_val = Xb, vb

    basin = 0
    # reserve ~28s for final intensive polish
    while time.time() - t0 < LIM - 28:
        remain = LIM - (time.time() - t0)
        if remain < 40:
            taus = [(0.002, 60)]
            gsteps = 120
        else:
            if basin % 3 == 0:
                taus = [(0.006, 60), (0.0015, 70)]
            else:
                taus = [(0.005, 60), (0.0012, 60)]
            gsteps = 160
        Xcur = best.copy()
        gmax, cnt, worst_order, contact_order, rattlers = contacts_info(Xcur)
        mode = basin % 9
        if mode == 0:
            Xcur = _normalize(Xcur + 0.005 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            Xcur = _normalize(Xcur + 0.012 * rng.standard_normal(Xcur.shape))
        elif mode == 2:
            k = 6 + (basin % 5)
            idx = worst_order[:k]
            nz = 0.004 * rng.standard_normal(Xcur.shape)
            nz[idx] += 0.020 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + nz)
        elif mode == 3:
            hole_replace(Xcur, int(worst_order[0]), M=6000)
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        elif mode == 4:
            if len(rattlers) > 0:
                ri = int(rattlers[basin % len(rattlers)])
            else:
                ri = int(np.argmin(cnt))
            hole_replace(Xcur, ri, M=6000)
            wi = int(worst_order[0])
            if wi != ri:
                Xcur[wi] = _normalize((Xcur[wi] + 0.008 * rng.standard_normal(d)).reshape(1, -1))[0]
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 5:
            Xcur = split_worst(Xcur, alpha=0.012)
            Xcur = _normalize(Xcur + 0.005 * rng.standard_normal(Xcur.shape))
        elif mode == 6:
            k = 4 + (basin % 3)
            idx = contact_order[:k]
            nz = 0.003 * rng.standard_normal(Xcur.shape)
            nz[idx] += 0.018 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + nz)
        elif mode == 7:
            hole_replace(Xcur, int(worst_order[0]), M=4000)
            g2, c2, w2, co2, r2 = contacts_info(Xcur)
            idx2 = int(r2[0]) if len(r2) > 0 else int(co2[0])
            hole_replace(Xcur, idx2, M=4000)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        else:
            s = 0.35 + 0.1 * (basin % 3)
            Xcur = isotropize(Xcur, strength=float(s))
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        Xcur = anneal(Xcur, taus)
        Xcur, vcur = greedy(Xcur, gsteps, 0.002)
        Xcur, vcur = micro_squeeze(Xcur, vcur, budget=0.18)
        if vcur < best_val - 1e-13:
            # quick LP confirm to lock gain
            Xcur, vcur = lp_polish(Xcur, budget=0.8, radius_init=5e-4)
            Xcur, vcur = greedy(Xcur, 80, 0.0005)
            if vcur < best_val:
                best = Xcur.copy()
                best_val = float(vcur)
        basin += 1
        if basin > 400:
            break

    if time.time() - t0 < LIM - 1:
        Xf = best.copy()
        Xf = anneal(Xf, [(0.0015, 90), (0.0006, 90)])
        Xf, vf = greedy(Xf, 300, 0.0007)
        Xf, vf = lp_polish(Xf, budget=8.0, radius_init=8e-4)
        Xf, vf = greedy(Xf, 250, 0.0005)
        Xf, vf = micro_squeeze(Xf, vf, budget=0.8)
        Xf, vf = lp_polish(Xf, budget=5.0, radius_init=3e-4)
        Xf, vf = greedy(Xf, 200, 0.0003)
        if vf < best_val:
            best, best_val = Xf, vf
    best = _normalize(best)
    return best
