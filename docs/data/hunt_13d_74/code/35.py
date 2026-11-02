import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(12345)
    D = 13
    N74 = 74

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (N74, D):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((N74, D))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    tri_cache = {}
    def _tri(n):
        if n not in tri_cache:
            iu = np.triu_indices(n, 1)
            tri_cache[n] = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        return tri_cache[n]

    def lbfgs_polish(X0, tau, maxiter):
        X0 = np.asarray(X0, dtype=np.float64)
        n, d = X0.shape
        if time.time() - t0 > LIM - 4:
            return X0
        I0, J0 = _tri(n)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.sqrt((Y * Y).sum(axis=1, keepdims=True))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / float(tau))
            S = float(e.sum())
            if not np.finite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + float(tau) * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter * 6)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def insert_one(X, M=8000):
        n, d = X.shape
        cand = _normalize(rng.standard_normal((int(M), d)))
        S = cand @ X.T
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        return np.vstack([X, cand[j:j+1]])

    def prune_to(X, target):
        X = X.copy()
        while X.shape[0] > int(target):
            n = X.shape[0]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            # per-point worst, degree at 1e-7, sum top-5
            mi = G.max(axis=1)
            deg = (G >= (gmax - 1e-7)).sum(axis=1).astype(float)
            # sum top-5
            part = np.sort(G, axis=1)[:, -5:]
            s5 = part.sum(axis=1)
            score = mi + 2e-4 * deg + 1e-4 * s5
            idx = int(np.argmax(score))
            X = np.delete(X, idx, axis=0)
        return _normalize(X)

    def anneal_seq(X, seq):
        for tau, it in seq:
            if time.time() - t0 > LIM - 6:
                break
            X = lbfgs_polish(X, tau, it)
        return X

    def greedy(X, steps, lr0):
        X = X.copy()
        cur = _max_cos(X)
        lr = float(lr0)
        n = X.shape[0]
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 3:
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
                X = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def lp_polish(X, budget, radius_init=8e-4):
        X = _normalize(X.copy())
        n, d = X.shape
        cur = _max_cos(X)
        radius = float(radius_init)
        start = time.time()
        iu = np.triu_indices(n, 1)
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
                X = Y
                cur = v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    # light initial touch on incumbent to have tight baseline
    if time.time() - t0 < LIM - 60:
        Xb = anneal_seq(best.copy(), [(0.002, 50), (0.0008, 50)])
        Xb, vb = greedy(Xb, 120, 0.001)
        if vb < best_val:
            best, best_val = Xb, vb

    cycle = 0
    # alternate overshoot/prune and undershoot/grow, keep best final 74
    while time.time() - t0 < LIM - 30:
        if cycle > 12:
            break
        if cycle % 2 == 0:
            # OVERSHOOT: 74 -> 74+extra -> anneal -> prune -> repolish
            if cycle % 4 == 0:
                extra = 3
            elif cycle % 4 == 2:
                extra = 4
            else:
                extra = 2
            # need enough time for at least overshoot+prune+repolish
            if time.time() - t0 > LIM - 55:
                # only try small overshoot
                extra = 2
            Xbig = best.copy()
            # small jitter to diversify hole search start
            Xbig = _normalize(Xbig + 0.002 * rng.standard_normal(Xbig.shape))
            for _ in range(extra):
                Xbig = insert_one(Xbig, M=8000)
            # pressurize: anneal harder problem 30-50s equivalent
            Xbig = anneal_seq(Xbig, [(0.007, 45), (0.0025, 55), (0.001, 55)])
            if time.time() - t0 > LIM - 28:
                # no time to repolish, just prune and eval
                Xpr = prune_to(Xbig, N74)
                vpr = _max_cos(Xpr)
                if vpr < best_val:
                    best, best_val = Xpr, vpr
                break
            Xpr = prune_to(Xbig, N74)
            Xpr = anneal_seq(Xpr, [(0.002, 50), (0.0007, 60)])
            Xpr, vpr = greedy(Xpr, 100, 0.001)
            if vpr < best_val - 1e-13:
                # confirm with quick LP-ish squeeze via greedy only (LP reserved for final)
                if vpr < best_val:
                    best, best_val = Xpr, vpr
        else:
            # UNDERSHOOT: 74 -> 70 -> anneal -> staged reinsert 4
            if time.time() - t0 > LIM - 55:
                break
            Xsmall = prune_to(best.copy(), 70)
            Xsmall = _normalize(Xsmall + 0.002 * rng.standard_normal(Xsmall.shape))
            Xsmall = anneal_seq(Xsmall, [(0.005, 45), (0.0015, 55)])
            Xgrow = Xsmall
            for k in range(4):
                if time.time() - t0 > LIM - 32:
                    break
                Xgrow = insert_one(Xgrow, M=8000)
                # polish between insertions
                Xgrow = lbfgs_polish(Xgrow, 0.002, 40)
                Xgrow = lbfgs_polish(Xgrow, 0.0008, 40)
            # ensure exactly 74
            if Xgrow.shape[0] != N74:
                if Xgrow.shape[0] > N74:
                    Xgrow = prune_to(Xgrow, N74)
                else:
                    while Xgrow.shape[0] < N74:
                        Xgrow = insert_one(Xgrow, M=6000)
            Xgrow = anneal_seq(Xgrow, [(0.0015, 50), (0.0006, 50)])
            Xgrow, vgr = greedy(Xgrow, 100, 0.001)
            if vgr < best_val - 1e-13:
                if vgr < best_val:
                    best, best_val = Xgrow, vgr
        cycle += 1

    # final intensive repolish of best within remaining budget
    remain = LIM - (time.time() - t0)
    if remain > 8:
        best = anneal_seq(best, [(0.0015, 60), (0.0006, 80)])
        best, best_val = greedy(best, 200, 0.0008)
    if LIM - (time.time() - t0) > 6:
        bg = min(6.0, LIM - (time.time() - t0) - 1.0)
        if bg > 1.0:
            best, best_val = lp_polish(best, budget=bg, radius_init=6e-4)
            best, bv2 = greedy(best, 120, 0.0005)
            if bv2 < best_val:
                best_val = bv2
                best = best if isinstance(best, np.ndarray) else best
            else:
                # greedy returns new X; keep best tracking consistent
                # recompute to be safe
                vv = _max_cos(best)
                best_val = float(vv)
    best = _normalize(best)
    # tiny shrink guard (strict feasibility not needed but keep finite)
    assert best.shape == (74, 13)
    assert np.all(np.isfinite(best))
    return best
