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
    FINAL_RESERVE = 24.0
    rng = np.random.default_rng(12345)
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

    from scipy.optimize import minimize as _min
    from scipy.optimize import linprog as _lp
    from scipy.sparse import coo_matrix as _coo

    def full_anneal(Xinit, taus, iters, creg=0.05):
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
                           options={'maxiter': int(mi), 'maxfun': int(mi * 4),
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

    def subset_anneal(Xinit, free_idx, taus, iters, creg=0.05):
        free_idx = np.asarray(free_idx, dtype=int)
        k = len(free_idx)
        if k == 0:
            return Xinit
        fix_mask = np.ones(n, dtype=bool)
        fix_mask[free_idx] = False
        Xfix = Xinit[fix_mask]
        x0 = Xinit[free_idx].ravel().copy()
        # map free position -> global for gradient assembly
        for tau, mi in zip(taus, iters):
            if _time.time() > T_END - FINAL_RESERVE:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                Xf = xf.reshape(k, d)
                rf = np.sqrt(np.sum(Xf * Xf, axis=1))
                rf = np.maximum(rf, 1e-12)
                Xfn = Xf / rf[:, None]
                # full normalized set
                Xn = np.empty((n, d), dtype=float)
                Xn[fix_mask] = Xfix
                Xn[free_idx] = Xfn
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
                A = W @ Xn  # (n,d)
                # gradient w.r.t. unnormalized free vars
                Af = A[free_idx]
                proj = np.sum(Af * Xfn, axis=1)
                G0 = (Af - proj[:, None] * Xfn) / rf[:, None]
                diff = rf - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xfn
                return float(L), Gr.ravel()
            try:
                res = _min(fg, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                    'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 6})
                x0 = res.x
            except Exception:
                break
            # renormalize free part for next tau
            Xm = x0.reshape(k, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            x0 = (Xm / rn).ravel()
        Xo = Xinit.copy()
        Xm = x0.reshape(k, d)
        Xm /= np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
        Xo[free_idx] = Xm
        return Xo

    def greedy_refill(Xbase, del_idx, pool_size=30000):
        del_idx = np.asarray(del_idx, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
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
        for s in range(len(new_pts)):
            others = np.vstack([Xkeep, np.delete(new_pts, s, axis=0)])
            bp = new_pts[s].copy()
            bv = float(np.max(others @ bp))
            for _ in range(40):
                prop = bp + 0.03 * rng.standard_normal(size=(d,))
                prop /= (np.linalg.norm(prop) + 1e-12)
                v = float(np.max(others @ prop))
                if v < bv:
                    bv = v
                    bp = prop
            new_pts[s] = bp
        Xt = Xbase.copy()
        Xt[del_idx] = new_pts
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

    def lp_jam(Xinit, deadline, radius_init=1.2e-3, slack=1.0e-3, max_it=18):
        nonlocal bestX, bestVal, archive
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal - 1e-15:
            bestVal = mcur; bestX = X.copy(); archive.append((bestVal, bestX.copy()))
        radius = float(radius_init)
        Nv = n * 5
        cvec = np.zeros(Nv + 1); cvec[Nv] = 1.0
        it = 0; fails = 0
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
            if len(sel) > 3200:
                part = np.argpartition(gfull[sel], -3200)[-3200:]
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
                Aub = _coo((data, (rows, cols)), shape=(K, Nv + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            bub = -cc
            bnds = [(-radius, radius)] * Nv + [(None, None)]
            try:
                res = _lp(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                          options={'presolve': True})
            except Exception:
                radius *= 0.5; fails += 1
                if fails > 5:
                    break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5; fails += 1
                if fails > 6:
                    break
                continue
            fails = 0
            y = res.x[:Nv].reshape(n, 5)
            DX = np.einsum('nkm,nm->nk', T, y)
            Y = X + DX
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Y)
            if v < mcur - 1e-15:
                X = Y; mcur = v
                if v < bestVal - 1e-15:
                    bestVal = v; bestX = Y.copy(); archive.append((bestVal, bestX.copy()))
                radius = min(radius * 1.4, 0.008)
            else:
                radius *= 0.5
        return X, mcur

    def contact_degrees(X, thresh):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > m - thresh
        # count upper only -> sum rows
        deg = np.sum(mask, axis=1)
        return deg, m

    # light polish incumbent first
    try:
        if _time.time() < T_END - FINAL_RESERVE - 15:
            Xp = full_anneal(bestX, [0.002, 0.0008], [60, 80])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp; bestX = Xp.copy(); archive.append((bestVal, bestX.copy()))
    except Exception:
        pass

    trial = 0
    # precompute pool reuse? generate per refill
    while _time.time() < T_END - FINAL_RESERVE and trial < 200:
        trial += 1
        mode = trial % 6
        try:
            # choose delete set
            if mode == 0 or mode == 1:
                # hub neighborhood surgery: pick hub among top degrees
                deg, mcur = contact_degrees(bestX, 4e-4)
                order = np.argsort(-deg)
                # pick random among top 15
                top = order[:15]
                hub = int(top[rng.integers(0, len(top))])
                dots = bestX @ bestX[hub]
                # neighbors = points with dot > mcur - 4e-4
                G row threshold
                nbrs = np.where(dots > mcur - 4e-4)[0]
                k = int(14 + rng.integers(0, 12))
                if len(nbrs) >= k:
                    # take k-1 closest neighbors + hub
                    idx_sorted = np.argsort(-dots)
                    del_idx = idx_sorted[:k]
                else:
                    # fill with closest
                    idx_sorted = np.argsort(-dots)
                    del_idx = idx_sorted[:k]
                # occasionally add second hub neighborhood
                if rng.random() < 0.35:
                    hub2 = int(top[rng.integers(0, len(top))])
                    d2 = bestX @ bestX[hub2]
                    idx2 = np.argsort(-d2)[:6]
                    del_idx = np.unique(np.concatenate([del_idx, idx2]))[:28]
                Xt = greedy_refill(bestX, del_idx, pool_size=28000)
                # fast subset hop
                Xs = subset_anneal(Xt, del_idx, [0.006, 0.0018], [50, 60])
                # spread + tighten full only if promising
                vs = _max_cos(Xs)
                if vs < bestVal + 8e-4:
                    Xc = full_anneal(Xs, [0.0012, 0.0005], [70, 60])
                else:
                    # small random kick on subset and continue
                    Xc = Xs
                vc = _max_cos(Xc)
            elif mode == 2:
                # spherical cap
                u = rng.standard_normal(size=(d,)); u /= (np.linalg.norm(u) + 1e-12)
                dots = bestX @ u
                k = int(12 + rng.integers(0, 16))
                del_idx = np.argsort(-dots)[:k]
                Xt = greedy_refill(bestX, del_idx, pool_size=28000)
                Xs = subset_anneal(Xt, del_idx, [0.006, 0.0015], [50, 60])
                vs = _max_cos(Xs)
                if vs < bestVal + 8e-4:
                    Xc = full_anneal(Xs, [0.0012, 0.0005], [70, 60])
                else:
                    Xc = Xs
                vc = _max_cos(Xc)
            elif mode == 3:
                # random cluster around random point (not necessarily hub)
                ci = int(rng.integers(0, n))
                dots = bestX @ bestX[ci]
                k = int(10 + rng.integers(0, 14))
                del_idx = np.argsort(-dots)[:k]
                Xt = greedy_refill(bestX, del_idx, pool_size=28000)
                Xs = subset_anneal(Xt, del_idx, [0.005, 0.0012], [50, 70])
                vs = _max_cos(Xs)
                if vs < bestVal + 5e-4:
                    Xc = full_anneal(Xs, [0.001, 0.00045], [60, 60])
                else:
                    Xc = Xs
                vc = _max_cos(Xc)
            elif mode == 4:
                # global shake + subset? full anneal from shaken best
                sig = 0.28 if (trial % 2 == 0) else 0.10
                Xt = bestX + sig * rng.standard_normal(size=(n, d))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                Xc = full_anneal(Xt, [0.006, 0.0018, 0.0006], [50, 60, 50])
                vc = _max_cos(Xc)
            else:
                # pure random restart with fast schedule (new basin)
                Xr = rng.standard_normal(size=(n, d))
                Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
                Xc = full_anneal(Xr, [0.015, 0.005, 0.0015, 0.0006], [40, 50, 60, 50])
                vc = _max_cos(Xc)

            if vc < bestVal - 1e-12:
                bestVal = vc; bestX = Xc.copy()
                archive.append((bestVal, bestX.copy()))
                # quick LP jam to squeeze new basin immediately
                try:
                    Xj, vj = lp_jam(bestX, deadline=min(T_END - FINAL_RESERVE, _time.time() + 6.0),
                                    radius_init=1.0e-3, slack=8e-4, max_it=12)
                except Exception:
                    pass
            elif vc < bestVal + 3e-4 and len(archive) < 14:
                # keep diverse near-best
                archive.append((vc, Xc.copy()))
            # occasional jam of current even if not best to explore jammed state then re-surgery
            if trial % 12 == 0 and _time.time() < T_END - FINAL_RESERVE - 8:
                try:
                    # jam the current candidate lightly to reveal hubs
                    _, _ = lp_jam(Xc, deadline=min(T_END - FINAL_RESERVE, _time.time() + 4.0),
                                  radius_init=8e-4, slack=8e-4, max_it=8)
                except Exception:
                    pass
        except Exception:
            continue

    # final intensive jam on top candidates
    archive_sorted = sorted(archive, key=lambda z: z[0])[:3]
    for _, Xc in archive_sorted:
        if _time.time() > T_END - 4.0:
            break
        try:
            Xj, vj = lp_jam(Xc, deadline=min(T_END - 2.0, _time.time() + 10.0),
                            radius_init=1.2e-3, slack=6e-4, max_it=30)
        except Exception:
            continue
    if _time.time() < T_END - 5.0:
        try:
            Xp = full_anneal(bestX, [0.0005, 0.00022], [70, 70])
            vp = _max_cos(Xp)
            if vp < bestVal:
                bestVal = vp; bestX = Xp
        except Exception:
            pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
