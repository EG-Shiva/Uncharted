import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    t_end = t0 + BUDGET
    explore_end = t0 + 133.0
    np.random.seed(20261103)
    rng = np.random.default_rng(20261103)
    n, d = 249, 6

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append((_max_cos(an), an.copy()))
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bestX = X0
        bestVal = _max_cos(bestX)
    else:
        cands.sort(key=lambda t: t[0])
        bestVal, bestX = cands[0]
        bestX = bestX.copy()

    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0

    def make_fg(tau, creg=0.05):
        tau = float(tau)
        def fg(xf):
            X = xf.reshape(n, d)
            r = np.sqrt(np.sum(X * X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0, J0]
            gmax = np.max(g)
            e = np.exp((g - gmax) / tau)
            S = np.sum(e)
            Lsoft = gmax + tau * np.log(S)
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[I0, J0] = w
            W = W + W.T
            A = W @ Xn
            proj = np.sum(A * Xn, axis=1)
            Grad0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg * np.sum(diff * diff)
            Grad = Grad0 + (2.0 * creg * diff)[:, None] * Xn
            return float(L), Grad.ravel()
        return fg

    def soft_polish(Xinit, taus, iters):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > t_end - 12:
                break
            fg = make_fg(float(tau))
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xm = Xm / rn
            Xf = Xm.ravel()
        Xout = Xf.reshape(n, d)
        Xout /= np.maximum(np.linalg.norm(Xout, axis=1, keepdims=True), 1e-12)
        return Xout

    def hole_optimize(base, Yinit, tau=0.002, iters=60):
        m = base.shape[0]
        k = Yinit.shape[0]
        if k == 0:
            return Yinit.copy()
        tau = float(tau)
        y0 = Yinit.ravel().copy()
        def fg_h(yf):
            Y = yf.reshape(k, d)
            r = np.sqrt(np.sum(Y * Y, axis=1))
            r = np.maximum(r, 1e-12)
            Yn = Y / r[:, None]
            D1 = Yn @ base.T  # k x m
            # D2
            G2 = Yn @ Yn.T
            iu_k = np.triu_indices(k, 1)
            g2 = G2[iu_k] if k > 1 else np.empty(0)
            g1 = D1.ravel()
            if g2.size > 0:
                g = np.concatenate([g1, g2])
            else:
                g = g1
            gmax = np.max(g)
            e = np.exp((g - gmax) / tau)
            S = np.sum(e)
            Lsoft = gmax + tau * np.log(S)
            w = e / S
            w1 = w[:g1.size].reshape(k, m)
            grad = w1 @ base  # k x d
            if g2.size > 0:
                W2 = np.zeros((k, k), dtype=float)
                W2[iu_k] = w[g1.size:]
                W2 = W2 + W2.T
                grad = grad + W2 @ Yn
            proj = np.sum(grad * Yn, axis=1)
            Grad0 = (grad - proj[:, None] * Yn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + 0.05 * np.sum(diff * diff)
            Grad = Grad0 + (2.0 * 0.05 * diff)[:, None] * Yn
            return float(L), Grad.ravel()
        try:
            res = minimize(fg_h, y0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(iters), 'maxfun': int(iters * 4),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
            Y = res.x.reshape(k, d)
        except Exception:
            Y = Yinit.copy()
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        return Y

    def greedy_picks(base, k, n_samp=30000):
        cand = rng.standard_normal((n_samp, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ base.T
        cur_max = D.max(axis=1)
        picks = []
        for _ in range(k):
            j = int(np.argmin(cur_max))
            p = cand[j].copy()
            picks.append(p)
            nd = cand @ p
            cur_max = np.maximum(cur_max, nd)
            cur_max[j] = 2.0
        return np.array(picks)

    def contact_info(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        per_max = G.max(axis=1)
        deg = np.sum(G > (m - 5e-4), axis=1)
        deg2 = np.sum(G > (m - 1e-3), axis=1)
        return G, m, per_max, deg, deg2

    # current walker for Metropolis basin hopping
    curX = bestX.copy()
    curVal = bestVal
    elites = [(bestVal, bestX.copy())]
    TEMP = 2.0e-05

    taus_fast = [0.005, 0.0018, 0.0007]
    iters_fast = [50, 50, 80]
    taus_light = [0.006, 0.002]
    iters_light = [40, 40]

    it = 0
    while time.time() < explore_end and it < 400:
        it += 1
        # choose source: mostly current walker, sometimes best
        if rng.random() < 0.15:
            srcX = bestX
            srcVal = bestVal
        else:
            srcX = curX
            srcVal = curVal
            # restart walker if drifted too far
            if srcVal > bestVal + 0.0012:
                srcX = bestX
                srcVal = bestVal
                curX = bestX.copy()
                curVal = bestVal
        try:
            _, msrc, per_max, deg, deg2 = contact_info(srcX)
            rtype = rng.random()
            trial = None
            touched = None
            if rtype < 0.22:
                sig = float(rng.choice(np.array([0.004, 0.008, 0.015, 0.028])))
                trial = srcX + sig * rng.standard_normal((n, d))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                # jitter taus a bit for diversity
                j = float(rng.uniform(0.8, 1.25))
                taus = [taus_fast[0]*j, taus_fast[1]*j, taus_fast[2]]
                trial = soft_polish(trial, taus, iters_fast)
            elif rtype < 0.40:
                mcap = int(rng.integers(10, 30))
                u = rng.standard_normal(d); u /= np.linalg.norm(u)
                dots = srcX @ u
                idx = np.sort(np.argsort(dots)[-mcap:])
                trial = srcX.copy()
                trial[idx] = rng.standard_normal((mcap, d))
                trial[idx] /= np.linalg.norm(trial[idx], axis=1, keepdims=True)
                trial = soft_polish(trial, taus_fast, iters_fast)
                touched = idx
            elif rtype < 0.72:
                k = int(rng.integers(2, 12))
                score = per_max + 0.002 * deg
                order = np.argsort(score)
                pool = order[-min(30, n):]
                # bias to worst: sample without replacement
                idx = np.sort(rng.choice(pool, size=k, replace=False))
                keep = np.ones(n, dtype=bool); keep[idx] = False
                base = srcX[keep]
                gp = greedy_picks(base, k, n_samp=30000)
                gp = hole_optimize(base, gp, tau=float(rng.uniform(0.0015, 0.003)), iters=60)
                trial = srcX.copy()
                trial[idx] = gp
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = soft_polish(trial, taus_fast, iters_fast)
                touched = idx
            else:
                # hub excision: highest degree (overloaded)
                k = int(rng.integers(1, 4))
                # hubs by deg2 then per_max
                hub_order = np.lexsort((per_max, deg2))
                hubs = hub_order[-min(12, n):]
                # pick k among top hubs
                idx_h = np.sort(rng.choice(hubs, size=min(k, len(hubs)), replace=False))
                # optionally also kill one worst neighbor of first hub
                if rng.random() < 0.5:
                    Gtmp = srcX @ srcX.T
                    np.fill_diagonal(Gtmp, -2.0)
                    h0 = int(idx_h[0])
                    nbr = int(np.argmax(Gtmp[h0]))
                    extra = np.array([nbr], dtype=int)
                    idx_h = np.sort(np.unique(np.concatenate([idx_h, extra])))
                k2 = len(idx_h)
                keep = np.ones(n, dtype=bool); keep[idx_h] = False
                base = srcX[keep]
                gp = greedy_picks(base, k2, n_samp=35000)
                gp = hole_optimize(base, gp, tau=0.0018, iters=80)
                trial = srcX.copy()
                trial[idx_h] = gp
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                # light then full? just full fast
                trial = soft_polish(trial, taus_fast, iters_fast)
                touched = idx_h

            v = _max_cos(trial)
            # global best update
            if v < bestVal:
                bestVal = v
                bestX = trial.copy()
                elites.append((v, trial.copy()))
                elites.sort(key=lambda t: t[0])
                elites = elites[:5]
                curX = trial.copy()
                curVal = v
            else:
                # Metropolis acceptance for walker
                if v < curVal or rng.random() < np.exp((curVal - v) / TEMP):
                    curX = trial.copy()
                    curVal = v
                # keep near-best diverse elites
                if v < bestVal + 2e-4 and len(elites) < 5:
                    # check distance
                    dup = False
                    for _, Xe in elites:
                        if np.max(np.abs(Xe - trial)) < 1e-3:
                            dup = True
                            break
                    if not dup:
                        elites.append((v, trial.copy()))
                        elites.sort(key=lambda t: t[0])
        except Exception:
            continue

    # final intensive smoothing on best (and check elites)
    for _, Xe in list(elites[:3]):
        if time.time() > t_end - 25:
            break
        try:
            t2 = soft_polish(Xe, [0.004, 0.002, 0.001, 0.0005, 0.0003],
                             [100, 100, 130, 180, 220])
            v2 = _max_cos(t2)
            if v2 < bestVal:
                bestVal = v2
                bestX = t2.copy()
        except Exception:
            pass
    if time.time() < t_end - 10:
        try:
            t2 = soft_polish(bestX, [0.003, 0.0015, 0.0007, 0.00035, 0.0002],
                             [120, 130, 160, 200, 250])
            v2 = _max_cos(t2)
            if v2 < bestVal:
                bestVal = v2
                bestX = t2.copy()
        except Exception:
            pass

    def lp_polish(X, deadline):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        best = _max_cos(X)
        radius = 2e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > t_end - 1.0:
                break
            c = (X @ X.T)[iu]
            m = float(c.max())
            sel = c > m - max(20 * radius, 2e-7)
            if np.count_nonzero(sel) > 6000:
                idxs = np.argsort(c)[-6000:]
                ms = np.zeros_like(sel)
                ms[idxs] = True
                sel = ms
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs",
                              options={"time_limit": max(1.0, deadline - time.time() - 0.3)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Y = X + res.x[:-1].reshape(n, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            v = _max_cos(Y)
            if v < best:
                X, best = Y, v
                radius = min(radius * 1.5, 0.01)
            else:
                radius *= 0.5
        return X, best

    if time.time() < t_end - 3:
        try:
            Y, v = lp_polish(bestX, t_end - 0.8)
            if v < bestVal:
                bestVal = v
                bestX = Y
        except Exception:
            pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
