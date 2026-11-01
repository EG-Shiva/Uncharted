import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LP_START = T_END - 12.0
    rng = np.random.default_rng(987654321)
    n, d = 68, 12
    IU0 = np.triu_indices(n, 1)
    I0_all, J0_all = IU0[0], IU0[1]

    # load incumbent
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                s = _max_cos(an)
                if s < best_score:
                    best_score = s
                    bestX = an.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)
    curX = bestX.copy()
    cur_score = best_score

    # ---------- helpers ----------
    def per_max_and_contacts(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        pm = G.max(axis=1)
        cnt = np.sum(G > m - tol, axis=1)
        return pm, cnt, m, G

    def single_fg_factory(Xr, beta):
        def fg(z):
            r = max(float(np.linalg.norm(z)), 1e-12)
            y = z / r
            s = Xr @ y
            m = float(s.max())
            e = np.exp(beta * (s - m))
            S = float(e.sum()) + 1e-300
            F = m + np.log(S) / beta
            w = e / S
            gy = Xr.T @ w
            gdot = float(gy @ y)
            gz = (gy - gdot * y) / r
            return F, gz
        return fg

    def opt_single(Xr, y_start, beta=500.0, maxiter=60):
        fg = single_fg_factory(Xr, beta)
        try:
            r = minimize(fg, y_start.copy(), jac=True, method='L-BFGS-B',
                         options={'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-7,
                                  'maxfun': int(maxiter*2)})
            z = r.x
            zn = float(np.linalg.norm(z))
            if zn < 1e-12 or not np.isfinite(zn):
                return y_start / (np.linalg.norm(y_start)+1e-300)
            return z / zn
        except Exception:
            return y_start / (np.linalg.norm(y_start)+1e-300)

    def make_global_fg(beta):
        def fg(yf):
            Y = yf.reshape(n, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            G = X @ X.T
            g = G[IU0]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum()) + 1e-300
            F = m + np.log(S) / beta
            w = e / S
            P = np.zeros((n, n))
            P[IU0] = w
            P = P + P.T
            GX = P @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return F, GY.ravel()
        return fg

    def run_global_lse(X0, betas, iters):
        y = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > LP_START - 1.0:
                break
            fg = make_global_fg(float(beta))
            try:
                r = minimize(fg, y, jac=True, method='L-BFGS-B',
                             options={'maxiter': int(mi), 'ftol': 1e-15, 'gtol': 1e-7,
                                      'maxfun': int(mi*2), 'maxcor': 25})
                y = r.x
            except Exception:
                break
        Y = y.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def hinge_insert_opt(F, Y0, tau, maxiter=200):
        m = F.shape[0]; k = Y0.shape[0]
        def fg(yf):
            Y = yf.reshape(k, d)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = Y / nr
            Gfy = F @ X.T  # (m,k)
            diff = Gfy - tau
            pos = diff > 0
            Fval = float(np.sum(np.where(pos, diff, 0.0)**2))
            W = np.where(pos, 2.0*diff, 0.0)  # (m,k)
            GX = F.T @ W  # (d,k) -> transpose
            GX = GX.T  # (k,d)
            if k > 1:
                Gxx = X @ X.T
                np.fill_diagonal(Gxx, -2.0)
                # upper only to avoid double; use full then half
                diff2 = Gxx - tau
                pos2 = diff2 > 0
                # keep symmetric, count each pair once: use triu
                # Fval add
                iu = np.triu_indices(k, 1)
                dp = diff2[iu]
                pp = dp > 0
                if np.any(pp):
                    Fval += float(np.sum(np.where(pp, dp[pp], 0.0)**2))
                    W2 = np.zeros((k, k))
                    W2[iu] = np.where(pp, 2.0*dp, 0.0)
                    W2 = W2 + W2.T
                    GX = GX + W2 @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GY = (GX - dot * X) / nr
            return Fval, GY.ravel()
        y0 = (Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)).ravel().copy()
        try:
            r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(maxiter), 'ftol': 1e-15, 'gtol': 1e-7,
                                  'maxfun': int(maxiter*2), 'maxcor': 25})
            Y = r.x.reshape(k, d)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            return Y, float(r.fun)
        except Exception:
            return Y0, 1e9

    def adjacency_set(X, thr=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G > m - thr
        # upper triangle indices as set of encoded ints
        ii, jj = np.where(np.triu(sel, 1))
        return set((int(a)*1000+int(b)) for a, b in zip(ii, jj)), m

    cur_adj, _ = adjacency_set(curX)
    # ---------- surgery loop ----------
    NCAND = 200000
    CHUNK = 20000
    trial = 0
    betas_full = [100.0, 400.0, 1500.0, 5000.0, 10000.0]
    iters_full = [60, 60, 60, 60, 80]

    while time.time() < LP_START - 2.0:
        trial += 1
        if time.time() > LP_START - 2.0:
            break
        # choose k
        k = int(rng.integers(2, 9))  # 2..8
        # contact analysis on curX
        _, cnt, mcur, Gcur = per_max_and_contacts(curX, tol=1e-7)
        mode = rng.random()
        idx_rem = None
        if mode < 0.40:
            # biased toward 24-28 contacts (high contacts)
            w = np.clip(cnt.astype(float) - 13.0, 0.5, None) ** 2.0
            # extra boost for 24-28
            w = w * np.where((cnt >= 24), 3.0, 1.0)
            w = w / w.sum()
            idx_rem = rng.choice(n, size=k, replace=False, p=w)
        elif mode < 0.60:
            # tightest cliques: top 120 pairs
            flat = Gcur[I0_all, J0_all]
            od = np.argsort(-flat)[:120]
            ti = np.concatenate([I0_all[od], J0_all[od]])
            # count frequency
            uniq, counts = np.unique(ti, return_counts=True)
            w = np.ones(n) * 0.3
            w[uniq] = counts.astype(float) + 0.5
            # boost overlap with high cnt
            w = w * (1.0 + 0.2*cnt)
            w = w / w.sum()
            idx_rem = rng.choice(n, size=k, replace=False, p=w)
        elif mode < 0.80:
            idx_rem = rng.choice(n, size=k, replace=False)
        else:
            # spatially clustered subset
            seed = int(rng.integers(0, n))
            row = Gcur[seed].copy()
            row[seed] = -2
            nn = np.argsort(-row)[:k-1]
            idx_rem = np.concatenate([[seed], nn]).astype(int)
            if len(np.unique(idx_rem)) < k:
                extra = rng.choice(n, size=k, replace=False)
                idx_rem = extra
        mask = np.ones(n, dtype=bool)
        mask[idx_rem] = False
        F = curX[mask]  # (68-k, d)
        mf = F.shape[0]

        # generate hole pool
        try:
            C = rng.standard_normal((NCAND, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-300
        except Exception:
            continue
        # score vs F chunked
        scores = np.empty(NCAND)
        for s0 in range(0, NCAND, CHUNK):
            e0 = min(s0+CHUNK, NCAND)
            S = C[s0:e0] @ F.T
            scores[s0:e0] = S.max(axis=1)
            if time.time() > LP_START - 1.0:
                break
        cur_set = F.copy()
        Yins = []
        # greedy insert one at a time reusing pool
        ok = True
        for t in range(k):
            if time.time() > LP_START - 1.0:
                ok = False
                break
            # best 30 by current scores
            if len(Yins) > 0:
                # scores already updated incrementally; ensure correct vs cur_set?
                pass
            part = np.argpartition(scores, 30)[:30]
            # order them by scores
            o = np.argsort(scores[part])
            top30 = part[o]
            best_y = None
            best_m = 1e9
            # refine each of 30
            for jj in top30:
                y0 = C[jj].copy()
                # quick refine vs cur_set
                yref = opt_single(cur_set, y0, beta=500.0, maxiter=50)
                mm = float((cur_set @ yref).max())
                if mm < best_m:
                    best_m = mm
                    best_y = yref
                if time.time() > LP_START - 1.0:
                    break
            if best_y is None:
                ok = False
                break
            Yins.append(best_y)
            # update scores incrementally: max(scores, C@best_y)
            for s0 in range(0, NCAND, CHUNK):
                e0 = min(s0+CHUNK, NCAND)
                upd = C[s0:e0] @ best_y
                # in-place max
                np.maximum(scores[s0:e0], upd, out=scores[s0:e0])
            cur_set = np.vstack([cur_set, best_y[None, :]])
        if not ok:
            continue
        del C  # free
        Yins = np.stack(Yins, axis=0)  # (k,d)
        Xnew = np.vstack([F, Yins])
        # hinge-only optimization on inserted points targeting below incumbent
        # margins 5e-5 and 2e-4 below best_score
        tau1 = best_score - 5e-5
        tau2 = best_score - 2e-4
        # only if tau sensible (>0)
        Yh, _ = hinge_insert_opt(F, Yins, tau1, maxiter=150)
        Yh, _ = hinge_insert_opt(F, Yh, tau2, maxiter=150)
        Xh = np.vstack([F, Yh])
        # full LSE release beta 100..10000
        # use slightly fewer iters if time short
        Xrel = run_global_lse(Xh, betas_full, iters_full)
        vrel = _max_cos(Xrel)
        # adjacency difference
        new_adj, _ = adjacency_set(Xrel)
        inter = len(cur_adj.intersection(new_adj))
        union = len(cur_adj.union(new_adj)) + 1e-300
        jacc = inter / union
        # update best
        if vrel < best_score - 1e-12:
            best_score = vrel
            bestX = Xrel.copy()
            curX = Xrel.copy()
            cur_score = vrel
            cur_adj = new_adj
        else:
            # accept diverse basin even if slightly worse to escape over-jammed vertex
            # accept if adjacency differs substantially (jacc < 0.75) and within tolerance
            if jacc < 0.75 and vrel < cur_score + 0.005:
                # probabilistic accept biased to better
                if vrel < cur_score or rng.random() < 0.35:
                    curX = Xrel.copy()
                    cur_score = vrel
                    cur_adj = new_adj
            elif vrel < cur_score - 1e-12:
                curX = Xrel.copy()
                cur_score = vrel
                cur_adj = new_adj
        # small time check: if best improved a lot, keep going
        if time.time() > LP_START - 2.0:
            break

    # ---------- final LP polishing on best ----------
    def lp_polish(x, deadline, cap_p=2200, r0=8e-4):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        nn, dd = x.shape
        best = _max_cos(x)
        radius = r0
        iu = np.triu_indices(nn, 1)
        while time.time() < deadline and radius > 1e-14:
            if time.time() > deadline - 0.4:
                break
            c = (x @ x.T)[iu]
            m = float(np.max(c))
            sel = c > m - max(20.0 * radius, 1e-9)
            I = iu[0][sel]; J = iu[1][sel]
            p = len(I)
            if p == 0:
                radius *= 0.5
                continue
            csel = c[sel]
            if p > cap_p:
                od = np.argsort(-csel)[:cap_p]
                I, J = I[od], J[od]
                csel = csel[od]
                p = cap_p
            nvv = nn * dd + 1
            rows = np.repeat(np.arange(p), 2 * dd + 1)
            cols = np.concatenate([I[:, None] * dd + np.arange(dd), J[:, None] * dd + np.arange(dd), np.full((p, 1), nn * dd)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nvv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(nn), dd), np.arange(nn * dd))), shape=(nn, nvv)).tocsr()
                obj = np.zeros(nvv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(nn),
                              bounds=[(-radius, radius)] * (nn * dd) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            step = res.x[:-1].reshape(nn, dd)
            y = x + step
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best - 1e-15:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                improved = False
                for shrink in (0.3, 0.1):
                    yy = x + step * shrink
                    yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                    vv = _max_cos(yy)
                    if vv < best:
                        x, best = yy, vv
                        improved = True
                        break
                if not improved:
                    radius *= 0.5
        return x

    try:
        if time.time() < T_END - 1.0:
            bestX = lp_polish(bestX, T_END - 0.5)
            best_score = _max_cos(bestX)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # tiny shrink guard not needed; ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
