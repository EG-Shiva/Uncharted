import time
import numpy as np
from scipy.optimize import minimize

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
    LIM = 168.0
    rng = np.random.default_rng(0)
    D = 13
    N74 = 74

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

    def time_left():
        return LIM - (time.time() - t0)

    def smooth_polish(X0, tau, maxiter):
        if time_left() < 4:
            return X0
        nloc, dloc = X0.shape
        iu = np.triu_indices(nloc, 1)
        I0 = iu[0]; J0 = iu[1]
        def fun(yf):
            Y = yf.reshape(nloc, dloc)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = np.sum(Wx * X, axis=1, keepdims=True)
            grad = (Wx - X * dot) / r
            return F, grad.ravel()
        y0 = X0.ravel().copy()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-8, "maxfun": 20000})
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
        lr = float(lr0)
        nloc = X.shape[0]
        for it in range(int(steps)):
            if (it & 15) == 0 and time_left() < 4:
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
        nloc = X.shape[0]
        bestl = _mc(X)
        start = time.time()
        step = 2.5e-6
        while time.time() - start < seconds and time_left() > 3:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // nloc; j = k % nloc
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
                Y[i] /= max(np.linalg.norm(Y[i]), 1e-12)
                Y[j] /= max(np.linalg.norm(Y[j]), 1e-12)
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

    def lp_polish(X, seconds, radius_init=8e-4):
        if time_left() < 5:
            return X, _mc(X)
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X, _mc(X)
        X = _norm(X)
        nloc, dloc = X.shape
        bestl = _mc(X)
        radius = float(radius_init)
        start = time.time()
        iu = np.triu_indices(nloc, 1)
        while time.time() - start < seconds and time_left() > 3 and radius > 1e-14:
            Cfull = (X @ X.T)[iu]
            m = float(Cfull.max())
            sel = Cfull > m - max(20.0 * radius, 2e-9)
            if np.count_nonzero(sel) > 2000:
                thr = np.sort(Cfull)[-2000]
                sel = Cfull >= thr
            I = iu[0][sel]; J = iu[1][sel]
            c = Cfull[sel]
            p = len(I)
            if p == 0:
                break
            nv = nloc * dloc + 1
            try:
                rows = np.repeat(np.arange(p), 2 * dloc + 1)
                cols = np.concatenate([I[:, None] * dloc + np.arange(dloc), J[:, None] * dloc + np.arange(dloc), np.full((p, 1), nloc * dloc)], 1).ravel()
                vls = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
                A_ub = coo_matrix((vls, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(nloc), dloc), np.arange(nloc * dloc))), shape=(nloc, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(nloc),
                              bounds=[(-radius, radius)] * (nloc * dloc) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if getattr(res, "status", 1) != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(nloc, dloc)
            Y = _norm(Y)
            v = _mc(Y)
            if v < bestl:
                X, bestl = Y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, bestl

    def farthest_pt(X, n_cand=4000):
        cand = _norm(rng.standard_normal((n_cand, D)))
        S = cand @ X.T
        sc = S.max(axis=1)
        return cand[int(np.argmin(sc))]

    def worst_rank(X):
        nloc = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        mx = G.max(axis=1)
        deg = (G > gmax - 1e-7).sum(axis=1)
        part = np.sort(G, axis=1)[:, -5:]
        s5 = part.sum(axis=1)
        order = np.lexsort((-s5, -deg, -mx))
        # lexsort ascending, we want worst first => largest first, so reverse
        order = order[::-1]
        return order, gmax

    def anneal_generic(X, budget):
        # budget: 'large','med','small'
        if budget == 'large':
            X = smooth_polish(X, 0.005, 150)
            X = smooth_polish(X, 0.0015, 200)
            X, _ = minimax(X, 80, 2.5e-4)
            X = smooth_polish(X, 0.0005, 200)
        elif budget == 'med':
            X = smooth_polish(X, 0.0025, 130)
            X = smooth_polish(X, 0.0007, 170)
            X, _ = minimax(X, 60, 2e-4)
        else:
            X = smooth_polish(X, 0.0012, 120)
            X, _ = minimax(X, 50, 2e-4)
        return X

    # initial squeeze
    Xb = best.copy()
    Xb = anneal_generic(Xb, 'med')
    Xb, _ = micro(Xb, 0.4)
    Xb, _ = lp_polish(Xb, 0.8, 6e-4)
    vb = _mc(Xb)
    if vb < best_val:
        best, best_val = Xb.copy(), float(vb)

    cyc = 0
    # alternate overshoot / undershoot
    while time_left() > 14 and cyc < 60:
        rem = time_left()
        if cyc % 2 == 0:
            # OVERSHOOT: 74 -> 74+k -> anneal -> prune -> 74
            k = [3, 4, 2, 4][(cyc // 2) % 4]
            if rem < 30:
                k = 2
            Xe = best.copy()
            for _ in range(k):
                y = farthest_pt(Xe, 5000)
                # tiny jitter to break symmetry
                y = _norm((y + 1e-3 * rng.standard_normal(D)).reshape(1, D))[0]
                Xe = np.vstack([Xe, y.reshape(1, D)])
            Xe = _norm(Xe)
            bgt = 'large' if rem > 55 else ('med' if rem > 28 else 'small')
            Xe = anneal_generic(Xe, bgt)
            Xe, _ = micro(Xe, 0.25)
            # prune k worst one-by-one with quick re-eval
            for _ in range(k):
                order, _ = worst_rank(Xe)
                worst = int(order[0])
                mask = np.ones(Xe.shape[0], dtype=bool)
                mask[worst] = False
                Xe = Xe[mask]
            Xe = _norm(Xe)
            # repolish 74
            Xe = anneal_generic(Xe, 'small' if rem < 40 else 'med')
            Xe, cur = micro(Xe, 0.3)
            if cur < best_val + 3e-6 and time_left() > 12:
                Xe, cur = lp_polish(Xe, 0.7, 5e-4)
            else:
                cur = _mc(Xe)
            if cur < best_val - 1e-12:
                best, best_val = Xe.copy(), float(cur)
        else:
            # UNDERSHOOT: 74 -> 70 -> anneal -> staged grow -> 74
            drop = 4
            order, _ = worst_rank(best)
            # hub removal: drop worst `drop` (mix of max and degree already)
            keep = np.ones(N74, dtype=bool)
            keep[order[:drop]] = False
            Xs = best[keep].copy()
            bgt = 'med' if rem > 35 else 'small'
            Xs = anneal_generic(Xs, bgt)
            # staged reinsertion one at a time with polishing
            for _ in range(drop):
                if time_left() < 10:
                    break
                y = farthest_pt(Xs, 5000)
                Xs = np.vstack([Xs, y.reshape(1, D)])
                Xs = _norm(Xs)
                Xs = smooth_polish(Xs, 0.0015, 110)
                Xs, _ = minimax(Xs, 40, 2e-4)
            if Xs.shape[0] != 74:
                cyc += 1
                continue
            Xs, cur = micro(Xs, 0.3)
            if cur < best_val + 3e-6 and time_left() > 12:
                Xs, cur = lp_polish(Xs, 0.7, 5e-4)
            else:
                cur = _mc(Xs)
            if cur < best_val - 1e-12:
                best, best_val = Xs.copy(), float(cur)
            else:
                # opportunistic small kick on best to try new basin cheaply
                if time_left() > 20 and (cyc % 4 == 3):
                    Xc = _norm(best + 0.006 * rng.standard_normal(best.shape))
                    Xc = anneal_generic(Xc, 'small')
                    Xc, cv = micro(Xc, 0.25)
                    if cv < best_val - 1e-12:
                        best, best_val = Xc.copy(), float(cv)
        cyc += 1

    if time_left() > 5:
        best, _ = micro(best, min(0.5, max(0.1, time_left() - 4)))
    if time_left() > 5:
        best, _ = lp_polish(best, max(0.4, time_left() - 2.5), 4e-4)
    best = _norm(best)
    return best
