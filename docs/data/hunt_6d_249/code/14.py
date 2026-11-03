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
    explore_end = t0 + 132.0
    np.random.seed(20261103)
    rng = np.random.default_rng(20261103)
    n, d = 249, 6

    # load incumbent
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

    def subset_polish(Xinit, free_idx, tau, mi):
        free_idx = np.asarray(free_idx, dtype=int)
        k = len(free_idx)
        if k == 0:
            return Xinit.copy()
        fixed = Xinit.copy()
        x0 = Xinit[free_idx].ravel().copy()
        tau = float(tau)
        def fg_free(xf):
            Y = xf.reshape(k, d)
            r = np.sqrt(np.sum(Y * Y, axis=1))
            r = np.maximum(r, 1e-12)
            Yn = Y / r[:, None]
            X = fixed.copy()
            X[free_idx] = Yn
            # full normalized (fixed rows already unit)
            G = X @ X.T
            g = G[I0, J0]
            gmax = np.max(g)
            e = np.exp((g - gmax) / tau)
            S = np.sum(e)
            Lsoft = gmax + tau * np.log(S)
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[I0, J0] = w
            W = W + W.T
            A = W @ X
            proj = np.sum(A * X, axis=1)
            GradFull = (A - proj[:, None] * X)
            # chain for free rows: divide by r + reg
            Gf = GradFull[free_idx] / r[:, None]
            diff = r - 1.0
            L = Lsoft + 0.05 * np.sum(diff * diff)
            Gf = Gf + (2.0 * 0.05 * diff)[:, None] * Yn
            return float(L), Gf.ravel()
        try:
            res = minimize(fg_free, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 10})
            Y = res.x.reshape(k, d)
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            Xout = fixed.copy()
            Xout[free_idx] = Y
        except Exception:
            Xout = Xinit.copy()
        Xout /= np.maximum(np.linalg.norm(Xout, axis=1, keepdims=True), 1e-12)
        return Xout

    def greedy_fill(base, k, n_samp=25000):
        # base: (n-k, d) unit rows, return k new points
        cand = rng.standard_normal((n_samp, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # max dot to base
        D = cand @ base.T
        cur_max = D.max(axis=1)
        picks = []
        for _ in range(k):
            j = int(np.argmin(cur_max))
            p = cand[j].copy()
            picks.append(p)
            # update cur_max with new point
            nd = cand @ p
            cur_max = np.maximum(cur_max, nd)
            # avoid repicking same: set to large
            cur_max[j] = 2.0
        return np.array(picks)

    def pick_victims(X, kval, cur_best, randomize=True):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        per_max = G.max(axis=1)
        # contact degree near top
        deg = np.sum(G > (cur_best - 5e-4), axis=1)
        score = per_max + 0.0015 * deg
        order = np.argsort(score)  # ascending, worst at end
        if randomize and kval < 30:
            # sample kval among worst 30 to diversify
            pool = order[-30:]
            # weighted: worst more likely
            sel = rng.choice(pool, size=kval, replace=False)
            return np.sort(sel)
        else:
            return np.sort(order[-kval:])

    # basin hopping
    it = 0
    taus_fast = [0.006, 0.0022, 0.0009]
    iters_fast = [60, 60, 100]
    taus_mid = [0.005, 0.002, 0.0008]
    iters_mid = [80, 80, 120]
    while time.time() < explore_end and it < 300:
        mode = it % 5
        try:
            if mode == 0:
                sig = [0.012, 0.006, 0.02][(it // 5) % 3]
                trial = bestX + sig * rng.standard_normal((n, d))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = soft_polish(trial, taus_fast, iters_fast)
                v = _max_cos(trial)
                if v < bestVal:
                    bestVal = v
                    bestX = trial.copy()
            elif mode == 1:
                # regional cap reshake: random cap resample
                m = int(rng.integers(12, 32))
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                dots = bestX @ u
                idx = np.sort(np.argsort(dots)[-m:])
                trial = bestX.copy()
                # resample cap randomly
                trial[idx] = rng.standard_normal((m, d))
                trial[idx] /= np.linalg.norm(trial[idx], axis=1, keepdims=True)
                trial = soft_polish(trial, taus_mid, iters_mid)
                v = _max_cos(trial)
                if v < bestVal:
                    bestVal = v
                    bestX = trial.copy()
            elif mode == 2 or mode == 3 or mode == 4:
                if mode == 2:
                    k = int(rng.integers(4, 9))
                elif mode == 3:
                    k = int(rng.integers(8, 15))
                else:
                    k = int(rng.integers(3, 7))
                    # random cap kill instead of worst kill
                if mode == 4:
                    # kill a random cap
                    u = rng.standard_normal(d)
                    u /= np.linalg.norm(u)
                    dots = bestX @ u
                    idx = np.sort(np.argsort(dots)[-k:])
                else:
                    idx = pick_victims(bestX, k, bestVal, randomize=True)
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[idx] = False
                base = bestX[keep_mask]
                new_pts = greedy_fill(base, k)
                trial = bestX.copy()
                trial[idx] = new_pts
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                # subset refine inserted points
                trial = subset_polish(trial, idx, 0.002, 80)
                # full refine
                trial = soft_polish(trial, taus_fast, iters_fast)
                v = _max_cos(trial)
                if v < bestVal:
                    bestVal = v
                    bestX = trial.copy()
        except Exception:
            pass
        it += 1
        # early exit if very little time
        if time.time() > explore_end:
            break

    # final intensive smoothing
    if time.time() < t_end - 8:
        try:
            trial = soft_polish(bestX, [0.004, 0.002, 0.001, 0.0005, 0.0003],
                                [120, 120, 150, 200, 250])
            v = _max_cos(trial)
            if v < bestVal:
                bestVal = v
                bestX = trial.copy()
        except Exception:
            pass

    # final LP jam (trust region)
    def lp_polish(X, deadline):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        best = _max_cos(X)
        radius = 2e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > t_end - 1.5:
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

    if time.time() < t_end - 4:
        try:
            Y, v = lp_polish(bestX, t_end - 1.0)
            if v < bestVal:
                bestVal = v
                bestX = Y
        except Exception:
            pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
