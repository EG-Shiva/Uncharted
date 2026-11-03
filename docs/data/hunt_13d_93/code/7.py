import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time as _tm
    t0 = _tm.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(0)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)

    n, d = 93, 13

    # ---------- annealed soft-max repulsion ----------
    def anneal(X, steps, tau0, tau1, s0, s1, mu=0.9):
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
        ls = np.log(s1 / s0) if s0 != s1 and s0 > 0 and s1 > 0 else 0.0
        for k in range(steps):
            if _tm.time() - t0 > DEADLINE:
                break
            frac = k / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            step = s0 * np.exp(ls * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            thresh = cmax - max(5.0 * tau, 0.004)
            # exp weights
            E = np.exp((G - cmax) / tau)
            E[G <= thresh] = 0.0
            np.fill_diagonal(E, 0.0)
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Ft = Gr - dot * X
            V = mu * V + (1.0 - mu) * Ft
            X = X - step * V
            # renorm
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X = X / nr
        return X

    # ---------- monotonic tangential greedy on true max ----------
    def greedy(Xinit, vinit, step_init, max_iters, tau=0.001):
        curX = Xinit.copy()
        cur_val = vinit
        bestX = Xinit.copy()
        bv = vinit
        step = step_init
        for _ in range(max_iters):
            if _tm.time() - t0 > DEADLINE:
                break
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            thresh = cmax - 0.006
            E = np.exp((G - cmax) / tau)
            E[G <= thresh] = 0.0
            np.fill_diagonal(E, 0.0)
            if E.sum() == 0:
                break
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ curX) / S
            dot = (Gr * curX).sum(axis=1, keepdims=True)
            Ft = Gr - dot * curX
            # normalize force scale to avoid blowup
            fn = np.linalg.norm(Ft)
            if not np.isfinite(fn) or fn == 0:
                break
            Xp = curX - step * Ft
            nr = np.linalg.norm(Xp, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Xp = Xp / nr
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            vp = float(np.max(Gp))
            if vp < cur_val:
                curX = Xp
                cur_val = vp
                if vp < bv:
                    bv = vp
                    bestX = Xp.copy()
                step = min(step * 1.2, 0.004)
            else:
                step *= 0.5
                if step < 1e-13:
                    break
        return bestX, bv, curX, cur_val

    # ---------- trust-region LP polish ----------
    def lp_polish(X, seconds):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return X, _maxcos(X)
        X = _norm(X.copy())
        cur_best = _maxcos(X)
        radius = 2e-3
        t_start = _tm.time()
        iu0 = np.triu_indices(n, 1)
        while _tm.time() - t_start < seconds and _tm.time() - t0 < DEADLINE - 1.0:
            if radius < 1e-14:
                break
            Gfull = X @ X.T
            c_all = Gfull[iu0]
            m = float(np.max(c_all))
            tol = max(20.0 * radius, 1e-9)
            sel_mask = c_all > m - tol
            # cap constraints for speed
            idx = np.where(sel_mask)[0]
            if len(idx) > 1500:
                # keep largest 1500
                order = np.argsort(c_all[idx])[-1500:]
                idx = idx[order]
                sel_mask = np.zeros_like(c_all, dtype=bool)
                sel_mask[idx] = True
            I = iu0[0][sel_mask]
            J = iu0[1][sel_mask]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            c_sel = c_all[sel_mask]
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            col_I = (I[:, None] * d + np.arange(d)).ravel()
            col_J = (J[:, None] * d + np.arange(d)).ravel()
            col_T = np.full(p, n * d)
            cols = np.concatenate([col_I, col_J, col_T]) if False else None
            # build cols per row correctly: interleave
            cols2 = np.empty(p * (2 * d + 1), dtype=np.int64)
            vals = np.empty(p * (2 * d + 1), dtype=np.float64)
            # X[J] block
            XJ = X[J]
            XI = X[I]
            for k in range(p):
                base = k * (2 * d + 1)
                cols2[base:base + d] = I[k] * d + np.arange(d)
                vals[base:base + d] = XJ[k]
                cols2[base + d:base + 2 * d] = J[k] * d + np.arange(d)
                vals[base + d:base + 2 * d] = XI[k]
                cols2[base + 2 * d] = n * d
                vals[base + 2 * d] = -1.0
            try:
                A_ub = coo_matrix((vals, (rows, cols2)), shape=(p, nv)).tocsr()
                # equality: X[i] . dy_i = 0
                erows = np.repeat(np.arange(n), d)
                ecols = np.arange(n * d)
                evals = X.ravel()
                A_eq = coo_matrix((evals, (erows, ecols)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                bnds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bnds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if getattr(res, "status", 1) != 0:
                radius *= 0.5
                continue
            try:
                dy = res.x[:-1].reshape(n, d)
            except Exception:
                radius *= 0.5
                continue
            Y = X + dy
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y = Y / nr
            v = _maxcos(Y)
            if v < cur_best:
                X = Y
                cur_best = v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return X, cur_best

    def hole_reinsert(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        k = int(np.argmax(rowmax))
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xr = X[mask]
        M = 12000
        C = _norm(rng.standard_normal((M, d)))
        D = C @ Xr.T
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        Xn = X.copy()
        Xn[k] = C[idx]
        # local refine of inserted point: few tangential steps vs others
        xk = Xn[k].copy()
        for _ in range(60):
            dots = Xr @ xk
            m = float(np.max(dots))
            # softmax weights to worst neighbors
            w = np.exp((dots - m) / 0.005)
            w = w / (w.sum() + 1e-12)
            g = w @ Xr
            gt = g - (g @ xk) * xk
            xk = xk - 0.05 * gt
            xk = xk / np.linalg.norm(xk)
        Xn[k] = xk
        Xn += 0.0008 * rng.standard_normal(Xn.shape)
        return _norm(Xn)

    # initial squeeze
    try:
        bX, bv, cX, cV = greedy(best, best_val, 8e-4, 600, tau=0.0012)
        if bv < best_val:
            best, best_val = bX, bv
    except Exception:
        pass
    # quick LP squeeze to exploit isolated pair (few seconds)
    try:
        if _tm.time() - t0 < DEADLINE - 10:
            LX, Lv = lp_polish(best, 6.0)
            if Lv < best_val:
                best, best_val = LX, Lv
    except Exception:
        pass

    sigmas = [0.02, 0.008, 0.03, 0.012, 0.005, 0.015, 0.025]
    trial = 0
    # basin hopping until ~125s, reserve rest for polish
    while _tm.time() - t0 < 118.0 and trial < 60:
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        if mode == 0:
            Xt = _norm(best + sigma * rng.standard_normal(best.shape))
        elif mode == 1:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            worst = np.argsort(rm)[-12:]
            Xt = best.copy() + 0.002 * rng.standard_normal(best.shape)
            Xt[worst] += sigma * rng.standard_normal((len(worst), d))
            Xt = _norm(Xt)
        elif mode == 2:
            try:
                Xt = hole_reinsert(best)
            except Exception:
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
        elif mode == 3:
            # split current max pair
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            ii, jj = divmod(ij, n)
            Xt = best.copy() + 0.003 * rng.standard_normal(best.shape)
            # push apart along tangent
            vi = best[jj] - (best[jj] @ best[ii]) * best[ii]
            vj = best[ii] - (best[ii] @ best[jj]) * best[jj]
            ni = np.linalg.norm(vi) + 1e-12
            nj = np.linalg.norm(vj) + 1e-12
            Xt[ii] = Xt[ii] - sigma * 0.8 * vi / ni
            Xt[jj] = Xt[jj] - sigma * 0.8 * vj / nj
            Xt = _norm(Xt)
        else:
            # large partial resample: resample 15 rattlers randomly
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(rm)  # smallest first = rattlers
            nr_rs = 15
            sel = order[:nr_rs]
            Xt = best.copy()
            Xt[sel] = rng.standard_normal((nr_rs, d))
            Xt = _norm(Xt + 0.005 * rng.standard_normal(Xt.shape))

        Xt = anneal(Xt, steps=900, tau0=0.012, tau1=0.001, s0=0.012, s1=0.0007, mu=0.9)
        if _tm.time() - t0 > DEADLINE:
            v = _maxcos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            break
        try:
            bX, bv, _, _ = greedy(Xt, _maxcos(Xt), 7e-4, 500, tau=0.001)
        except Exception:
            trial += 1
            continue
        # opportunistic LP if promising
        if bv < best_val + 0.002 and _tm.time() - t0 < 115.0:
            try:
                LX, Lv = lp_polish(bX, 2.0)
                if Lv < bv:
                    bX, bv = LX, Lv
            except Exception:
                pass
        if bv < best_val - 1e-12:
            best, best_val = bX, bv
        trial += 1

    # final intensive polish with remaining time
    try:
        for sinit, taut, iters in [(4e-4, 0.001, 1200), (2e-4, 0.0007, 1200), (1e-4, 0.0005, 1500)]:
            if _tm.time() - t0 > DEADLINE - 8:
                break
            bX, bv, _, _ = greedy(best, best_val, sinit, iters, tau=taut)
            if bv < best_val:
                best, best_val = bX, bv
    except Exception:
        pass
    try:
        remain = DEADLINE - (_tm.time() - t0) - 1.0
        if remain > 3.0:
            LX, Lv = lp_polish(best, min(remain, 25.0))
            if Lv < best_val:
                best, best_val = LX, Lv
    except Exception:
        pass
    # final micro greedy after LP
    try:
        if _tm.time() - t0 < DEADLINE - 1.0:
            bX, bv, _, _ = greedy(best, best_val, 1e-4, 800, tau=0.0005)
            if bv < best_val:
                best, best_val = bX, bv
    except Exception:
        pass

    best = _norm(best)
    return best
