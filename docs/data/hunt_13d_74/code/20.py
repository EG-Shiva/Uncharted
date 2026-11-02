import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

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
    rng = np.random.default_rng(1)
    n, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((74, 13))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU0 = np.triu_indices(n, 1)
    I0, J0 = IU0[0].copy(), IU0[1].copy()

    def lbfgs_lse(X0, tau, maxiter):
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
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-22,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 10)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def greedy(X, steps, lr0, cur=None):
        if cur is None:
            cur = _max_cos(X)
        lr = float(lr0)
        for it in range(int(steps)):
            if (it & 15) == 0 and time.time() - t0 > LIM - 3:
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

    def micro(X, cur, budget=0.4):
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
                G2 = Y @ Y.T
                np.fill_diagonal(G2, -2.0)
                v = float(G2.max())
                if v < cur:
                    X[:] = Y
                    cur = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-12:
                    break
        return X, cur

    def hole_replace(X, idx_list, M=2500):
        for iw in idx_list:
            cand = _normalize(rng.standard_normal((M, d)))
            S = cand @ X.T
            S[:, iw] = -10.0
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X[iw] = cand[j]
        return X

    def split_pair(X, alpha=0.012):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        ij = int(np.argmax(G))
        ii, jj = divmod(ij, n)
        vi = X[ii] - alpha * X[jj]
        vj = X[jj] - alpha * X[ii]
        ni = np.linalg.norm(vi); nj = np.linalg.norm(vj)
        if ni > 1e-12 and np.isfinite(ni):
            X[ii] = vi / ni
        if nj > 1e-12 and np.isfinite(nj):
            X[jj] = vj / nj
        return X

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        tight = G > (gmax - 2e-7)
        loose = G > (gmax - 0.02)
        ct = tight.sum(axis=1)
        cl = loose.sum(axis=1)
        mx = G.max(axis=1)
        return gmax, ct, cl, mx, G

    def fast_polish(X):
        # annealed L-BFGS, cheap
        X = lbfgs_lse(X, 0.004, 220)
        if time.time() - t0 > LIM - 3:
            return X
        X = lbfgs_lse(X, 0.0015, 260)
        if time.time() - t0 > LIM - 3:
            return X
        X = lbfgs_lse(X, 0.0006, 280)
        return X

    def full_polish(X):
        X = lbfgs_lse(X, 0.01, 150)
        if time.time() - t0 > LIM - 3:
            return X
        X = lbfgs_lse(X, 0.004, 220)
        if time.time() - t0 > LIM - 3:
            return X
        X = lbfgs_lse(X, 0.0015, 280)
        if time.time() - t0 > LIM - 3:
            return X
        X = lbfgs_lse(X, 0.0005, 320)
        return X

    # initial deep polish of incumbent
    X0 = best.copy()
    X0 = full_polish(X0)
    X0, v0 = greedy(X0, 350, 0.002)
    X0, v0 = micro(X0, v0, budget=0.8)
    if v0 < best_val:
        best, best_val = X0, v0

    basin = 0
    while time.time() - t0 < LIM - 10:
        remain = LIM - (time.time() - t0)
        M = 1500 if remain < 25 else 2500
        Xcur = best.copy()
        gmax, ct, cl, mx, G = contact_info(Xcur)
        worst = np.argsort(-mx)
        rattlers = np.where(ct == 0)[0]
        hubs = np.argsort(-cl)
        mode = basin % 7
        if mode == 0:
            # single worst hole + tiny jitter
            hole_replace(Xcur, [int(worst[0])], M=M)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            # double worst hole
            hole_replace(Xcur, [int(worst[0]), int(worst[1])], M=M)
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 2:
            # rattler relocation: move up to 2 rattlers to holes
            if len(rattlers) > 0:
                k = min(len(rattlers), 2)
                sel = rng.choice(rattlers, size=k, replace=False)
                hole_replace(Xcur, [int(s) for s in sel], M=M)
                Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
            else:
                hole_replace(Xcur, [int(worst[0])], M=M)
                Xcur = split_pair(Xcur, 0.015)
        elif mode == 3:
            # hub neighbourhood kick
            h = int(hubs[0])
            # neighbours of hub with high cosine
            row = G[h].copy()
            row[h] = -2
            nbr = np.argsort(-row)[:5]
            idx = np.concatenate([[h], nbr[:3]])
            noise = 0.004 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.024 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + noise)
            Xcur = split_pair(Xcur, 0.008)
        elif mode == 4:
            Xcur = _normalize(Xcur + 0.012 * rng.standard_normal(Xcur.shape))
            Xcur = split_pair(Xcur, 0.012)
        elif mode == 5:
            # random subset large kick
            k = 8
            idx = rng.choice(n, size=k, replace=False)
            noise = 0.005 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.03 * rng.standard_normal((k, d))
            Xcur = _normalize(Xcur + noise)
        else:
            hole_replace(Xcur, [int(worst[0]), int(worst[1])], M=M)
            Xcur = _normalize(Xcur + 0.008 * rng.standard_normal(Xcur.shape))
            Xcur = split_pair(Xcur, 0.015)

        # polish
        if basin % 4 == 3 and remain > 40:
            Xcur = full_polish(Xcur)
            Xcur, vcur = greedy(Xcur, 350, 0.002)
        else:
            Xcur = fast_polish(Xcur)
            if time.time() - t0 > LIM - 3:
                break
            Xcur, vcur = greedy(Xcur, 280, 0.002)
        if vcur < best_val + 5e-8:
            Xcur, vcur = micro(Xcur, vcur, budget=0.35)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        basin += 1
        if basin > 400:
            break

    # final LP trust-region tightening + greedy
    def lp_polish(X, cur, seconds=5.0):
        start = time.time()
        radius = 1e-3
        iu = np.triu_indices(n, 1)
        while time.time() - start < seconds and time.time() - t0 < LIM - 2:
            if radius < 1e-14:
                break
            Cf = (X @ X.T)[iu]
            m = float(Cf.max())
            sel = Cf > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) > 2000:
                thr = np.sort(Cf)[-2000]
                sel = Cf >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cf[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            c1 = (I[:, None] * d + np.arange(d)).ravel()
            c2 = (J[:, None] * d + np.arange(d)).ravel()
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::2 * d + 1] = 0  # placeholder
            # build cols vectorized
            base = np.empty((p, 2 * d + 1), dtype=np.int64)
            base[:, :d] = I[:, None] * d + np.arange(d)
            base[:, d:2 * d] = J[:, None] * d + np.arange(d)
            base[:, 2 * d] = n * d
            cols = base.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=np.float64)
            vals[:, :d] = X[J]
            vals[:, d:2 * d] = X[I]
            vals[:, 2 * d] = -1.0
            try:
                A_ub = coo_matrix((vals.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                              options={"time_limit": 2.0})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y = _normalize(Y)
            v = _max_cos(Y)
            if v < cur:
                X[:] = Y
                cur = v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    if time.time() - t0 < LIM - 4:
        try:
            Xf = best.copy()
            Xf, vf = greedy(Xf, 250, 0.0008, cur=best_val)
            Xf, vf = lp_polish(Xf, vf, seconds=4.0)
            Xf, vf = greedy(Xf, 200, 0.0005, cur=vf)
            Xf, vf = micro(Xf, vf, budget=1.0)
            if vf < best_val:
                best, best_val = Xf, vf
        except Exception:
            pass

    best = _normalize(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1)
    for i in range(n):
        if nr[i] < 1e-12 or not np.isfinite(nr[i]):
            best[i] = _normalize(rng.standard_normal((1, d)))[0]
    best = _normalize(best)
    return best
