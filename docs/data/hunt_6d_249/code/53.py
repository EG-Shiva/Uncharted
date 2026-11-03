import time as _time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = _time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    FINAL_RESERVE = 22.0
    rng = np.random.default_rng(7)
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
    archive = [(bestVal, bestX.copy())]

    from scipy.optimize import minimize, linprog
    from scipy.sparse import coo_matrix

    def anneal(Xinit, taus, iters, creg=0.05):
        from scipy.optimize import minimize as _min
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if _time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = float(np.max(g))
                e = np.exp((g - gm) / tau)
                S = float(np.sum(e))
                Lsoft = gm + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                W = np.zeros((n, n), dtype=float)
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = _min(fg, Xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'maxfun': int(mi * 5),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm / rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def greedy_refill(Xbase, del_idx, pool_size=30000):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.asarray(del_idx, dtype=int)] = False
        Xkeep = Xbase[keep_mask]
        Cp = rng.standard_normal(size=(pool_size, d))
        Cp /= np.maximum(np.linalg.norm(Cp, axis=1, keepdims=True), 1e-12)
        D = Cp @ Xkeep.T
        curmax = D.max(axis=1)
        used = np.zeros(pool_size, dtype=bool)
        new_pts = []
        for _ in range(len(del_idx)):
            curmax[used] = np.inf
            bi = int(np.argmin(curmax))
            used[bi] = True
            pt = Cp[bi].copy()
            new_pts.append(pt)
            dd = Cp @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        # local hole polish: each new pt try small random moves
        for s in range(len(new_pts)):
            others = np.vstack([Xkeep, np.delete(new_pts, s, axis=0)])
            best_pt = new_pts[s].copy()
            best_v = float(np.max(others @ best_pt))
            for _ in range(60):
                if _time.time() > T_END - FINAL_RESERVE:
                    break
                prop = best_pt + 0.02 * rng.standard_normal(size=(d,))
                prop /= (np.linalg.norm(prop) + 1e-12)
                v = float(np.max(others @ prop))
                if v < best_v:
                    best_v = v
                    best_pt = prop
            new_pts[s] = best_pt
        Xt = Xbase.copy()
        Xt[np.asarray(del_idx, dtype=int)] = new_pts
        return Xt

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy()
            v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = (v / nv).reshape(6, 1)
                H = E - 2.0 * (w @ w.T)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, deadline, radius_init=2e-3, slack=8e-4, max_it=60):
        nonlocal bestX, bestVal, archive
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal - 1e-15:
            bestVal = mcur
            bestX = X.copy()
            archive.append((bestVal, bestX.copy()))
        radius = float(radius_init)
        Nv = n * 5
        cvec = np.zeros(Nv + 1); cvec[Nv] = 1.0
        it = 0
        fails = 0
        while it < max_it and _time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            if _time.time() > T_END - 3.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - slack)[0]
            if len(sel) == 0:
                sel = np.argsort(gfull)[-800:]
            if len(sel) > 3500:
                part = np.argpartition(gfull[sel], -3500)[-3500:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]; TJ = T[Jc]
            XJ = X[Jc]; XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            r1 = np.repeat(np.arange(K), 5)
            rows = np.concatenate([r1, r1, np.arange(K)])
            cols = np.concatenate([(Ic[:, None] * 5 + np.arange(5)).ravel(),
                                   (Jc[:, None] * 5 + np.arange(5)).ravel(),
                                   np.full(K, Nv)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            bub = -cc
            bnds = [(-radius, radius)] * Nv + [(None, None)]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5
                fails += 1
                if fails > 6:
                    break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                fails += 1
                if fails > 8:
                    break
                continue
            fails = 0
            y = res.x[:Nv]
            Ym = y.reshape(n, 5)
            DX = np.einsum('nkm,nm->nk', T, Ym)
            Y = X + DX
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Y)
            if v < mcur - 1e-15:
                X = Y
                mcur = v
                if v < bestVal - 1e-15:
                    bestVal = v
                    bestX = Y.copy()
                    archive.append((bestVal, bestX.copy()))
                radius = min(radius * 1.4, 0.01)
            else:
                radius *= 0.5
        return X, mcur

    # quick polish of incumbent with short anneal to have clean base
    if _time.time() < T_END - FINAL_RESERVE - 10:
        try:
            Xp = anneal(bestX, [0.004, 0.0012], [90, 120])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp; bestX = Xp.copy()
                archive.append((bestVal, bestX.copy()))
        except Exception:
            pass

    trial = 0
    while _time.time() < T_END - FINAL_RESERVE and trial < 80:
        trial += 1
        mode = trial % 5
        try:
            if mode == 0:
                # pure random restart, different basin
                Xr = rng.standard_normal(size=(n, d))
                Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
                Xc = anneal(Xr, [0.02, 0.006, 0.0018, 0.0007], [70, 90, 120, 80])
            elif mode == 1:
                # big global shake
                sig = 0.35 if (trial % 2 == 0) else 0.12
                Xt = bestX + sig * rng.standard_normal(size=(n, d))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                Xc = anneal(Xt, [0.008, 0.0022, 0.0008], [80, 110, 90])
            elif mode == 2:
                # spherical cap deletion
                u = rng.standard_normal(size=(d,)); u /= (np.linalg.norm(u) + 1e-12)
                dots = bestX @ u
                k = int(12 + rng.integers(0, 19))
                del_idx = np.argsort(-dots)[:k]
                Xt = greedy_refill(bestX, del_idx, pool_size=26000)
                Xt += 1e-3 * rng.standard_normal(size=(n, d))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                Xc = anneal(Xt, [0.006, 0.0018, 0.0007], [80, 110, 80])
            elif mode == 3:
                # nearest-neighbour cluster deletion
                ci = int(rng.integers(0, n))
                dots = bestX @ bestX[ci]
                k = int(15 + rng.integers(0, 11))
                del_idx = np.argsort(-dots)[:k]
                # include centre guaranteed
                Xt = greedy_refill(bestX, del_idx, pool_size=26000)
                Xt += 1e-3 * rng.standard_normal(size=(n, d))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                Xc = anneal(Xt, [0.006, 0.0018, 0.0007], [80, 110, 80])
            else:
                # shake second-best archive for diversity
                if len(archive) > 1:
                    sv = sorted(archive, key=lambda z: z[0])
                    base = sv[1 % len(sv)][1] if len(sv) > 1 else bestX
                else:
                    base = bestX
                sig = 0.22
                Xt = base + sig * rng.standard_normal(size=(n, d))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                Xc = anneal(Xt, [0.01, 0.0025, 0.0008], [70, 100, 80])
            vc = _max_cos(Xc)
            if vc < bestVal - 2e-12:
                bestVal = vc; bestX = Xc.copy()
                archive.append((bestVal, bestX.copy()))
            elif vc < bestVal + 2e-4 and len(archive) < 12:
                archive.append((vc, Xc.copy()))
        except Exception:
            continue

    # final intensive jam on top candidates
    archive_sorted = sorted(archive, key=lambda z: z[0])[:3]
    for _, Xc in archive_sorted:
        if _time.time() > T_END - 4.0:
            break
        try:
            Xj, vj = lp_jam(Xc, deadline=min(T_END - 3.0, _time.time() + 12.0),
                            radius_init=1.5e-3, slack=8e-4, max_it=50)
        except Exception:
            continue
    # final short anneal polish at low tau on global best if time
    if _time.time() < T_END - 6.0:
        try:
            Xp = anneal(bestX, [0.0006, 0.00025], [90, 90])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp; bestX = Xp.copy()
        except Exception:
            pass
        try:
            Xj, vj = lp_jam(bestX, deadline=T_END - 2.0, radius_init=8e-4, slack=5e-4, max_it=60)
        except Exception:
            pass

    out = bestX.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    out[~np.isfinite(out)] = 1.0 / np.sqrt(6.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
