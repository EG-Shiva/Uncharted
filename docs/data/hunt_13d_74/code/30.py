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
    rng = np.random.default_rng(12345)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n0, d):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n0, d))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    _iu_cache = {}
    def _iu(n):
        if n not in _iu_cache:
            iu = np.triu_indices(n, 1)
            _iu_cache[n] = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        return _iu_cache[n]

    def lse_polish(X0, tau, maxiter):
        n = X0.shape[0]
        I0, J0 = _iu(n)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.sqrt(np.maximum((Y*Y).sum(axis=1, keepdims=True), 1e-24))
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
            Wx = np.zeros_like(X)
            Wx[I0] += w[:, None] * X[J0]
            Wx[J0] += w[:, None] * X[I0]
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-8, "maxfun": int(maxiter*6)})
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
        nn = X.shape[0]
        for _ in range(int(steps)):
            if time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0*lr, 5e-7)
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

    def anneal(X, taus):
        for tau, it in taus:
            if time.time() - t0 > LIM - 4:
                break
            X[:] = lse_polish(X, tau, it)
        return X

    def point_stats(X, eps=1e-7):
        nn = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        cnt = (G >= (G.max() - eps)).sum(axis=1)
        # sum top-5
        part = np.sort(G, axis=1)[:, -5:].sum(axis=1)
        return mx, cnt, part, float(G.max())

    def insert_caps(X, k_add, M=4000):
        for _ in range(int(k_add)):
            cand = _normalize(rng.standard_normal((M, d)))
            S = cand @ X.T
            sc = S.max(axis=1)
            j = int(np.argmin(sc))
            X = np.vstack([X, cand[j:j+1]])
        return _normalize(X)

    def prune_to_74(Xover):
        # delete worst points by combined score until 74
        while Xover.shape[0] > 74:
            mx, cnt, s5, gmax = point_stats(Xover, eps=1e-7)
            # badness: primary mx, secondary s5, tertiary cnt
            # lexsort: sort ascending (good first), drop last
            order = np.lexsort((s5, mx))  # ascending
            # drop worst = last
            drop = int(order[-1])
            Xover = np.delete(Xover, drop, axis=0)
        return _normalize(Xover)

    def drop_to(X, target):
        # drop worst to target
        Xc = X.copy()
        while Xc.shape[0] > target:
            mx, cnt, s5, gmax = point_stats(Xc, eps=1e-7)
            order = np.lexsort((s5, mx))
            drop = int(order[-1])
            Xc = np.delete(Xc, drop, axis=0)
        return _normalize(Xc)

    def lp_polish(X, budget, radius_init=8e-4):
        start = time.time()
        cur = _max_cos(X)
        radius = float(radius_init)
        nn = X.shape[0]
        iu0, iu1 = _iu(nn)
        nv = nn * d + 1
        while time.time() - start < budget and time.time() - t0 < LIM - 2:
            if radius < 1e-13:
                break
            Cfull = (X @ X.T)[(iu0, iu1)]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0*radius, 1e-9)
            if int(np.count_nonzero(sel)) > 2200:
                thr = float(np.sort(Cfull)[-2200])
                sel = Cfull >= thr
            I = iu0[sel].astype(np.int64); J = iu1[sel].astype(np.int64)
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), nn*d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p,1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(nn), d), np.arange(nn*d))), shape=(nn, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)]*(nn*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = _normalize(X + res.x[:-1].reshape(nn, d))
            if not np.all(np.isfinite(Y)):
                radius *= 0.5
                continue
            v = _max_cos(Y)
            if v < cur:
                X[:] = Y
                cur = v
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur

    # ---- initial quick polish of incumbent ----
    Xb = best.copy()
    Xb = anneal(Xb, [(0.004, 50), (0.0012, 50)])
    Xb, vb = greedy(Xb, 120, 0.002)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)
    else:
        # keep annealed version only if better; else keep original best
        pass

    cycle = 0
    # alternate overshoot / undershoot, reserve ~22s for final polish
    while time.time() - t0 < LIM - 24:
        if time.time() - t0 > LIM - 26:
            break
        if cycle % 2 == 0:
            # OVERSHOOT: 74 -> Nover -> 74
            Nover = 76 + (cycle % 3)  # 76,77,78 cycle
            # time guard: need at least ~10s left for this path
            if LIM - (time.time()-t0) < 14:
                break
            Xover = insert_caps(best.copy(), Nover - 74, M=3500)
            # anneal harder problem briefly (pressure reorganisation)
            # slightly noisy start helps change topology
            Xover = _normalize(Xover + 0.002*rng.standard_normal(Xover.shape))
            Xover = anneal(Xover, [(0.005, 45), (0.0018, 45)])
            Xover, _ = greedy(Xover, 90, 0.002)
            # prune back to 74 by worst combined score
            Xpr = prune_to_74(Xover)
            Xpr = anneal(Xpr, [(0.0015, 45), (0.0006, 40)])
            Xpr, vpr = greedy(Xpr, 100, 0.0012)
            if vpr < best_val - 1e-13:
                # confirm with tiny extra polish
                Xpr2 = anneal(Xpr.copy(), [(0.0004, 30)])
                Xpr2, vpr2 = greedy(Xpr2, 60, 0.0006)
                if vpr2 < vpr:
                    Xpr, vpr = Xpr2, vpr2
                if vpr < best_val:
                    best, best_val = Xpr.copy(), float(vpr)
        else:
            # UNDERSHOOT: 74 -> 70 -> 74 staged grow
            if LIM - (time.time()-t0) < 14:
                break
            Xunder = drop_to(best.copy(), 70)
            Xunder = _normalize(Xunder + 0.002*rng.standard_normal(Xunder.shape))
            Xunder = anneal(Xunder, [(0.004, 40), (0.0012, 40)])
            Xunder, _ = greedy(Xunder, 80, 0.002)
            # reinsert 4 one at a time with polishing between
            for _ in range(4):
                if time.time() - t0 > LIM - 26:
                    break
                Xunder = insert_caps(Xunder, 1, M=3500)
                Xunder = anneal(Xunder, [(0.002, 35)])
                Xunder, _ = greedy(Xunder, 60, 0.0015)
            if Xunder.shape[0] != 74:
                # fix size if interrupted
                if Xunder.shape[0] > 74:
                    Xunder = prune_to_74(Xunder)
                else:
                    Xunder = insert_caps(Xunder, 74 - Xunder.shape[0], M=2000)
            Xunder = anneal(Xunder, [(0.0008, 35)])
            Xunder, vunder = greedy(Xunder, 90, 0.001)
            if vunder < best_val:
                best, best_val = Xunder.copy(), float(vunder)
        cycle += 1
        if cycle > 12:
            break

    # ---- final intensive polish of best 74 ----
    Xf = best.copy()
    # annealed LSE + greedy
    remain = LIM - (time.time() - t0)
    if remain > 8:
        Xf = anneal(Xf, [(0.002, 55), (0.0007, 55)])
        Xf, vf = greedy(Xf, 180, 0.0012)
        if vf < best_val:
            best, best_val = Xf.copy(), float(vf)
        else:
            Xf = best.copy()
            vf = best_val
        # LP to squeeze jammed core
        lbudget = min(8.0, max(1.0, LIM - (time.time()-t0) - 3.0))
        if lbudget > 0.5:
            Xf, vf = lp_polish(Xf, budget=lbudget, radius_init=6e-4)
            Xf, vf = greedy(Xf, 100, 0.0005)
            if vf < best_val:
                best, best_val = Xf.copy(), float(vf)
    best = _normalize(best)
    return best
