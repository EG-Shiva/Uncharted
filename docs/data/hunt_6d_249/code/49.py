import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    RESERVE = 24.0
    rng = np.random.default_rng(0)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    cur = bestX.copy()
    curVal = bestVal
    archive = [(bestVal, bestX.copy())]

    def anneal(Xinit, taus, iters, creg=0.05):
        Xf = (Xinit / np.maximum(np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r2 = np.sum(X * X, axis=1)
                r = np.sqrt(np.maximum(r2, 1e-24))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = np.max(g)
                e = np.exp((g - gm) / tau)
                S = np.sum(e)
                Lsoft = gm + tau * np.log(S)
                w = e / S
                A = np.zeros((n, d), dtype=float)
                # A[I0] += w * Xn[J0]; A[J0] += w * Xn[I0]
                WxJ = w[:, None] * Xn[J0]
                WxI = w[:, None] * Xn[I0]
                np.add.at(A, I0, WxJ)
                np.add.at(A, J0, WxI)
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * np.sum(diff * diff)
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            try:
                Xm = Xf.reshape(n, d)
                rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                Xf = (Xm / rn).ravel()
            except Exception:
                break
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(keepX, k, pool):
        # keepX (nk,6), pool (C,6) normalized
        S = pool @ keepX.T
        curmax = np.max(S, axis=1)
        C = pool.shape[0]
        used = np.zeros(C, dtype=bool)
        # set used entries to inf lazily
        chosen = []
        for _ in range(k):
            bi = int(np.argmin(curmax))
            # if already used (inf), argmin still returns unused unless all used
            used[bi] = True
            pt = pool[bi]
            chosen.append(pt)
            # update
            dots = pool @ pt
            # maximum, keeping inf for used
            np.maximum(curmax, dots, out=curmax)
            curmax[bi] = np.inf
            # ensure previously used stay inf
            # (maximum of inf stays inf, so ok)
        return np.stack(chosen, axis=0)

    cycle_k = [10, 16, 22, 30, 12, 20, 28, 8, 18, 26]
    it = 0
    # quick short polish to have cur jammed-ish? skip, go to avalanches
    while time.time() < T_END - RESERVE:
        it += 1
        if it > 220:
            break
        k = int(cycle_k[(it - 1) % len(cycle_k)])
        # vary pool size deterministically
        Cpool = 10000 + (it % 3) * 4000
        mode = it % 3
        try:
            if mode == 0:
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                dots = cur @ u
                del_idx = np.argpartition(dots, -k)[-k:]
            elif mode == 1:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                # tiny random tie break deterministic via rng
                tb = rng.standard_normal(n) * 1e-6
                hub = int(np.argmax(rowmax + tb))
                r = G[hub]
                # k-1 neighbours + hub
                nn = np.argpartition(r, -(k - 1))[-(k - 1):]
                del_idx = np.concatenate(([hub], nn)).astype(int)
            else:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                top10 = np.argpartition(rowmax, -10)[-10:]
                hub = int(top10[rng.integers(0, 10)])
                r = G[hub]
                nn = np.argpartition(r, -(k - 1))[-(k - 1):]
                del_idx = np.concatenate(([hub], nn)).astype(int)
                # replace 1/3 of deletions by random low-contact points to move rattlers
                if k >= 12:
                    G2 = G
                    m = float(np.max(G2))
                    cnt = np.sum(G2 > m - 5e-4, axis=1)
                    ratt = np.where(cnt == 0)[0]
                    if len(ratt) > 2:
                        nr = min(3, k // 4)
                        pick = rng.choice(ratt, size=nr, replace=False)
                        del_idx[-nr:] = pick
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            keepX = cur[keep_mask]
            pool = rng.standard_normal(size=(Cpool, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            new_pts = greedy_refill(keepX, k, pool)
            Xt = np.empty_like(cur)
            Xt[keep_mask] = keepX
            Xt[~keep_mask] = new_pts
            # short re-jam
            if k >= 22:
                taus = [0.008, 0.0025, 0.0008]
                iters = [40, 50, 65]
            else:
                taus = [0.006, 0.002, 0.0007]
                iters = [35, 45, 55]
            Xp = anneal(Xt, taus, iters)
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp
                bestX = Xp.copy()
                archive.append((bestVal, bestX.copy()))
                cur = Xp.copy()
                curVal = vp
            elif vp < curVal + 0.0025 and vp < 0.735:
                cur = Xp.copy()
                curVal = vp
            else:
                if it % 14 == 0:
                    # diversify from best with moderate shake
                    shake = 0.012 * rng.standard_normal(size=(n, d))
                    cur = bestX + shake
                    cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                    curVal = _max_cos(cur)
        except Exception:
            continue

    # final intensive polish of top archive entries
    try:
        archive_sorted = sorted(archive, key=lambda t: t[0])[:3]
        # also include current if good
        cand_list = [a[1] for a in archive_sorted]
        for ci, Xc in enumerate(cand_list):
            if time.time() > T_END - 6.0:
                break
            if ci == 0:
                taus = [0.004, 0.0015, 0.0007, 0.00032]
                iters = [110, 140, 170, 160]
            else:
                taus = [0.003, 0.001, 0.0004]
                iters = [80, 100, 120]
            # temporarily allow anneal to use remaining time (override RESERVE)
            # run manual loop with tighter deadline
            Xf = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).ravel().copy()
            for tau, mi in zip(taus, iters):
                if time.time() > T_END - 3.0:
                    break
                tau = float(tau)
                def fg2(xf, tau=tau):
                    X = xf.reshape(n, d)
                    r = np.sqrt(np.maximum(np.sum(X * X, axis=1), 1e-24))
                    r = np.maximum(r, 1e-12)
                    Xn = X / r[:, None]
                    G = Xn @ Xn.T
                    g = G[I0, J0]
                    gm = np.max(g)
                    e = np.exp((g - gm) / tau)
                    S = np.sum(e)
                    Lsoft = gm + tau * np.log(S)
                    w = e / S
                    A = np.zeros((n, d), dtype=float)
                    np.add.at(A, I0, w[:, None] * Xn[J0])
                    np.add.at(A, J0, w[:, None] * Xn[I0])
                    proj = np.sum(A * Xn, axis=1)
                    G0 = (A - proj[:, None] * Xn) / r[:, None]
                    diff = r - 1.0
                    L = Lsoft + 0.05 * np.sum(diff * diff)
                    Gr = G0 + (0.1 * diff)[:, None] * Xn
                    return float(L), Gr.ravel()
                try:
                    res = minimize(fg2, Xf, jac=True, method='L-BFGS-B',
                                   options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                            'ftol': 1e-14, 'gtol': 1e-8, 'maxcor': 8})
                    Xf = res.x
                    Xm = Xf.reshape(n, d)
                    rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
                    Xf = (Xm / rn).ravel()
                except Exception:
                    break
            Xo = Xf.reshape(n, d)
            Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Xo)
            if v < bestVal:
                bestVal = v
                bestX = Xo.copy()
    except Exception:
        pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard against boundary ties (no-op, keep strictly feasible)
    return bestX
