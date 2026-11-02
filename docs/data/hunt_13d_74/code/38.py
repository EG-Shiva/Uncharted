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
    rng = np.random.default_rng(12345)
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

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def lbfgs_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            # shift for stability
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            # correct accumulation (add.at, not fancy +=)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def minimax(X, steps, lr0, cur_init=None):
        cur = cur_init if cur_init is not None else _max_cos(X)
        lr = float(lr0)
        for it in range(steps):
            if (it & 15) == 0 and time.time() - t0 > LIM - 4:
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

    def micro_squeeze(X, cur, budget=0.5):
        start = time.time()
        step = 3e-6
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
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

    def worst_order(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx)[:k], float(G.max())

    def contact_counts(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = (G > gmax - thr).sum(axis=1)
        return cnt, gmax, G

    def refine_hole(c, X, forbid, steps=60, lr0=0.08):
        c = c / (np.linalg.norm(c) + 1e-12)
        lr = lr0
        for _ in range(steps):
            dots = X @ c
            dots[forbid] = -10.0
            j = int(np.argmax(dots))
            # move away from X[j]
            g = X[j] - float(dots[j]) * c
            ng = float(np.linalg.norm(g))
            if ng < 1e-12:
                break
            g /= ng
            c = c - lr * g
            c /= (np.linalg.norm(c) + 1e-12)
            lr *= 0.97
        return c

    def hole_replace(X, idx_list, n_cand=3000):
        for iw in idx_list:
            cand = _normalize(rng.standard_normal((n_cand, d)))
            S = cand @ X.T
            S[:, iw] = -10.0
            sc = S.max(axis=1)
            order = np.argsort(sc)[:4]
            # refine top few and pick best
            best_c = cand[order[0]]
            best_s = float(sc[order[0]])
            for o in order:
                cc = refine_hole(cand[o], X, iw)
                s = float((X @ cc).max() if iw < 0 else np.delete(X @ cc, iw).max())
                # compute correctly excluding iw
                if s < best_s:
                    best_s = s
                    best_c = cc
            X[iw] = best_c
        return X

    def split_worst_pair(X, alpha=0.012):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        ij = int(np.argmax(G))
        ii, jj = divmod(ij, n)
        a = X[ii].copy(); b = X[jj].copy()
        X[ii] = a - alpha * b
        X[jj] = b - alpha * a
        for k in (ii, jj):
            nr = float(np.linalg.norm(X[k]))
            if nr == 0 or not np.isfinite(nr):
                X[k] = _normalize(rng.standard_normal((1, d)))[0]
            else:
                X[k] /= nr
        return X

    # initial squeeze
    try:
        X0 = best.copy()
        if time.time() - t0 < LIM - 30:
            X0 = lbfgs_polish(X0, 0.004, 180)
        if time.time() - t0 < LIM - 25:
            X0 = lbfgs_polish(X0, 0.0012, 220)
        X0, v0 = minimax(X0, 250, 0.002)
        X0, v0 = micro_squeeze(X0, v0, budget=0.8)
        if v0 < best_val:
            best, best_val = X0, v0
    except Exception:
        pass

    basin = 0
    while time.time() - t0 < LIM - 9:
        remain = LIM - (time.time() - t0)
        if remain < 15:
            n_cand = 1200
            it1, it2 = 60, 80
            mm_steps = 150
        elif remain < 40:
            n_cand = 2000
            it1, it2 = 100, 140
            mm_steps = 200
        else:
            n_cand = 3000
            it1, it2 = 130, 180
            mm_steps = 250

        Xcur = best.copy()
        mode = basin % 7
        try:
            cnt, gmax, _ = contact_counts(Xcur, thr=1e-7)
        except Exception:
            cnt = np.zeros(n)
        rattlers = np.where(cnt == 0)[0]
        hubs = np.argsort(-cnt)[:8]

        if mode == 0:
            wo, _ = worst_order(Xcur, 1)
            Xcur = hole_replace(Xcur, [int(wo[0])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.0025 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            wo, _ = worst_order(Xcur, 2)
            Xcur = hole_replace(Xcur, [int(wo[0]), int(wo[1])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 2:
            wo, _ = worst_order(Xcur, 1)
            # worst + one rattler (recruit rattler)
            if len(rattlers) > 0:
                ri = int(rattlers[rng.integers(len(rattlers))])
                if ri == int(wo[0]) and len(rattlers) > 1:
                    ri = int(rattlers[(int(rng.integers(len(rattlers))) + 1) % len(rattlers)])
                Xcur = hole_replace(Xcur, [int(wo[0]), ri], n_cand=n_cand)
            else:
                wo2, _ = worst_order(Xcur, 2)
                Xcur = hole_replace(Xcur, [int(wo2[0]), int(wo2[1])], n_cand=n_cand)
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            # hub offload: kick hubs harder + split worst
            noise = 0.004 * rng.standard_normal(Xcur.shape)
            noise[hubs] += 0.022 * rng.standard_normal((len(hubs), d))
            Xcur = _normalize(Xcur + noise)
            Xcur = split_worst_pair(Xcur, 0.010)
        elif mode == 4:
            Xcur = _normalize(Xcur + 0.016 * rng.standard_normal(Xcur.shape))
            Xcur = split_worst_pair(Xcur, 0.012)
        elif mode == 5:
            wo, _ = worst_order(Xcur, 1)
            Xcur = hole_replace(Xcur, [int(wo[0])], n_cand=n_cand)
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] = _normalize(Xcur[ii:ii+1] + 0.006 * rng.standard_normal((1, d)))[0]
            Xcur[jj] = _normalize(Xcur[jj:jj+1] + 0.006 * rng.standard_normal((1, d)))[0]
        else:
            wo, _ = worst_order(Xcur, 2)
            Xcur = hole_replace(Xcur, [int(wo[0]), int(wo[1])], n_cand=max(1500, n_cand // 2))
            Xcur = _normalize(Xcur + 0.009 * rng.standard_normal(Xcur.shape))

        try:
            Xcur = lbfgs_polish(Xcur, 0.006, it1)
            if time.time() - t0 > LIM - 6:
                vcur = _max_cos(Xcur)
            else:
                Xcur = lbfgs_polish(Xcur, 0.0013, it2)
                Xcur, vcur = minimax(Xcur, mm_steps, 0.002)
                if vcur < best_val + 5e-08:
                    Xcur, vcur = micro_squeeze(Xcur, vcur, budget=0.35)
            if vcur < best_val - 1e-12:
                best = Xcur.copy()
                best_val = float(vcur)
        except Exception:
            pass
        basin += 1
        if basin > 200:
            break

    if time.time() - t0 < LIM - 2:
        try:
            Xf = best.copy()
            Xf, vf = minimax(Xf, 250, 0.0008, cur_init=best_val)
            Xf, vf = micro_squeeze(Xf, vf, budget=1.0)
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
