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

def _lp_polish(X, best, deadline):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 1.5e-3
    while time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(25.0 * radius, 2e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt > 2500:
            thr = np.sort(cfull)[-2500]
            sel = cfull >= thr
            cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        I = iu[0][sel]; J = iu[1][sel]
        cc = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _normalize(Y)
        v = _max_cos(Y)
        if v < best:
            X = Y
            best = v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    LIM = 170.0
    DEAD = t0 + LIM
    rng = np.random.default_rng(7)
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
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    def smooth_lbfgs(X0, tau, maxiter):
        # limit time
        if time.time() > DEAD - 15:
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
            S = e.sum()
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            # scatter-add (vectorized with bincount-like)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = np.sum(Wx * X, axis=1, keepdims=True)
            grad_Y = (Wx - X * dot) / r
            return F, grad_Y.ravel()
        y0 = X0.ravel().copy()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-9, "maxfun": 10000})
            Y = res.x.reshape(nloc, dloc)
        except Exception:
            return X0
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        if not np.all(np.isfinite(X)):
            return X0
        return X

    def refine_lse(X, sched, m, v, cnt):
        for tau, steps, lr0 in sched:
            for s in range(steps):
                if time.time() > DEAD - 18:
                    return X
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
                v[:] = 0.999 * v + 0.001 * (Gt * Gt)
                cnt[0] += 1
                c1 = 1.0 - 0.9 ** cnt[0]
                c2 = 1.0 - 0.999 ** cnt[0]
                mhat = m / c1
                vhat = v / c2
                lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
                X -= lr * mhat / (np.sqrt(vhat) + 1e-8)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X /= nr
        return X

    def greedy(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        for _ in range(steps):
            if time.time() > DEAD - 18:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            Gr = mask.astype(np.float64) @ X
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
        return np.argsort(-G.max(axis=1))

    def farthest_reinsert(X, idx, nc=2000):
        cand = _normalize(rng.standard_normal((nc, d)))
        S = cand @ X.T
        S[:, idx] = -10.0
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        X[idx] = cand[j]
        return X

    def directed_squeeze(X, cur, deadline, budget=6.0):
        start = time.time()
        step = 2.5e-6
        while time.time() - start < budget and time.time() < deadline:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            xi = X[i].copy(); xj = X[j].copy()
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
                v = _max_cos(Y)
                if v < cur:
                    X[:] = Y
                    cur = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-11:
                    break
        return X, cur

    full_sched = [(0.02, 250, 0.011), (0.007, 400, 0.007), (0.0022, 450, 0.004), (0.0009, 300, 0.0025)]
    fast_sched = [(0.008, 250, 0.007), (0.0025, 350, 0.0045), (0.0009, 250, 0.0025)]
    tiny_sched = [(0.0025, 200, 0.004), (0.0009, 200, 0.002)]

    # initial quick squeeze of incumbent with L-BFGS + greedy
    try:
        X0 = best.copy()
        m = np.zeros_like(X0); vv = np.zeros_like(X0); cc = [0]
        X0 = refine_lse(X0, [(0.007, 250, 0.007), (0.0022, 300, 0.004)], m, vv, cc)
        X0 = smooth_lbfgs(X0, 0.0015, 250)
        X0, v0 = greedy(X0, 250, 0.0015)
        X0 = smooth_lbfgs(X0, 0.0004, 250)
        X0, v0b = greedy(X0, 250, 0.0007)
        if v0b < best_val:
            best, best_val = X0.copy(), float(v0b)
    except Exception:
        pass

    basin = 0
    while time.time() < DEAD - 20:
        remain = DEAD - 20 - time.time()
        if remain < 10:
            sched = tiny_sched
        elif remain < 30:
            sched = fast_sched
        else:
            sched = fast_sched if (basin % 3 == 0) else full_sched
        Xcur = best.copy()
        mode = basin % 8
        if mode == 0:
            Xcur = _normalize(Xcur + 0.016 * rng.standard_normal(Xcur.shape))
        elif mode == 1:
            wo = worst_order(Xcur)
            k = 6 + (basin % 5)
            idx = wo[:k]
            noise = 0.0035 * rng.standard_normal(Xcur.shape)
            noise[idx] += 0.030 * rng.standard_normal((len(idx), d))
            Xcur = _normalize(Xcur + noise)
        elif mode == 2:
            iw = int(worst_order(Xcur)[0])
            Xcur = farthest_reinsert(Xcur, iw, nc=2500)
            Xcur = _normalize(Xcur + 0.004 * rng.standard_normal(Xcur.shape))
        elif mode == 3:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            Xcur[ii] -= 0.012 * Xcur[jj]
            Xcur[jj] -= 0.012 * Xcur[ii]
            Xcur = _normalize(Xcur + 0.007 * rng.standard_normal(Xcur.shape))
        elif mode == 4:
            iw = int(worst_order(Xcur)[0])
            Xcur = farthest_reinsert(Xcur, iw, nc=2500)
            iw2 = int(worst_order(Xcur)[0])
            if iw2 != iw:
                Xcur = farthest_reinsert(Xcur, iw2, nc=1500)
            Xcur = _normalize(Xcur + 0.003 * rng.standard_normal(Xcur.shape))
        elif mode == 5:
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            Xcur[idx] = _normalize(rng.standard_normal((k, d)))
            Xcur = _normalize(Xcur + 0.002 * rng.standard_normal(Xcur.shape))
        elif mode == 6:
            # L-BFGS basin: smooth then perturb jammed slightly
            try:
                Xcur = smooth_lbfgs(Xcur + 0.008 * rng.standard_normal(Xcur.shape), 0.003, 150)
            except Exception:
                Xcur = _normalize(Xcur + 0.008 * rng.standard_normal(Xcur.shape))
        else:
            Xcur = _normalize(Xcur + 0.010 * rng.standard_normal(Xcur.shape))

        m = np.zeros_like(Xcur); vv = np.zeros_like(Xcur); cc = [0]
        Xcur = refine_lse(Xcur, sched, m, vv, cc)
        # occasional L-BFGS mid-polish for diversity
        if basin % 4 == 3 and time.time() < DEAD - 30:
            try:
                Xcur = smooth_lbfgs(Xcur, 0.0012, 150)
            except Exception:
                pass
        Xcur, vcur = greedy(Xcur, 220, 0.0018)
        if vcur < best_val - 1e-12:
            best = Xcur.copy()
            best_val = float(vcur)
        basin += 1
        if basin > 300:
            break

    # final intensive squeeze
    if time.time() < DEAD - 2:
        Xf = best.copy()
        try:
            Xf = smooth_lbfgs(Xf, 0.0010, 300)
            v = _max_cos(Xf)
            if v < best_val:
                best, best_val = Xf.copy(), float(v)
            Xf, v = greedy(Xf, 200, 0.0007)
            if v < best_val:
                best, best_val = Xf.copy(), float(v)
            Xf2 = smooth_lbfgs(best.copy(), 0.00035, 350)
            v2 = _max_cos(Xf2)
            if v2 < best_val:
                best, best_val = Xf2.copy(), float(v2)
            Xf3, v3 = greedy(best.copy(), 250, 0.0005)
            if v3 < best_val:
                best, best_val = Xf3.copy(), float(v3)
        except Exception:
            pass
    if time.time() < DEAD - 1:
        try:
            Xp, vp = _lp_polish(best.copy(), best_val, DEAD - 7.0)
            if vp < best_val:
                best, best_val = Xp, vp
        except Exception:
            pass
    if time.time() < DEAD - 1:
        try:
            Xd, vd = directed_squeeze(best.copy(), best_val, DEAD - 0.5, budget=5.0)
            if vd < best_val:
                best, best_val = Xd, vd
        except Exception:
            pass
    # last greedy touch
    if time.time() < DEAD - 0.8:
        try:
            Xg, vg = greedy(best.copy(), 120, 0.0003)
            if vg < best_val:
                best, best_val = Xg.copy(), float(vg)
        except Exception:
            pass

    best = _normalize(best)
    return best
