import time
import numpy as np
from scipy.optimize import minimize

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
    LIM = 170.0
    rng = np.random.default_rng(0)
    n, d = 74, 13
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def global_lse_polish(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def bal_lse_polish(X0, tau, tau2, avec, maxiter):
        a = np.asarray(avec, dtype=np.float64).ravel()
        a = np.maximum(a, 1e-300)
        a = a / a.sum()
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r2 = (Y * Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            # per-row max excluding diag
            Gtmp = G.copy()
            np.fill_diagonal(Gtmp, -2.0)
            prow_max = Gtmp.max(axis=1)  # (n,)
            # E matrix
            # (G - prow_max)/tau, zero diag
            D = (G - prow_max[:, None]) / tau
            # mask diag to very negative so exp->0
            # set diag to -1e30/tau -> 0
            D[np.diag_indices(n)] = -1e12
            # clip lower to avoid underflow warning (exp handles it)
            E = np.exp(np.maximum(D, -700.0))
            E[np.diag_indices(n)] = 0.0
            S = E.sum(axis=1)  # (n,)
            S = np.maximum(S, 1e-300)
            m = prow_max + tau * np.log(S + 1e-300)  # (n,)
            mmax = float(m.max())
            e2 = a * np.exp(np.maximum((m - mmax) / tau2, -700.0))
            S2 = float(e2.sum())
            if not np.isfinite(S2) or S2 <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau2 * np.log(S2 + 1e-300)
            q = e2 / (S2 + 1e-300)  # (n,)
            P = E / S[:, None]  # p_ij
            # W = q_row * P + P.T * q_col
            W = q[:, None] * P + (P.T * q[None, :])
            Gr = W @ X
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            gradY = Gt / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def per_point_stats(X, contact_tol=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        spread = float(prow.max() - prow.min())
        # zero-contact with tight threshold 1e-9 like picture, and loose with 1e-4
        cnt_tight = (G > (gmax - 1e-9)).sum(axis=1)
        n_rattler_tight = int((cnt_tight == 0).sum())
        n_loose = int((prow < gmax - contact_tol).sum())
        # smoothed m_i at small tau for reporting (exact prow is fine)
        return gmax, prow, spread, n_rattler_tight, n_loose, cnt_tight

    def micro_split(X, cur, steps=250):
        step = 4e-6
        for _ in range(steps):
            if time.time() - t0 > LIM - 2:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            c = float(G[i, j])
            if c < cur - 1e-12:
                cur = c
            xi = X[i]; xj = X[j]
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
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                v = _max_cos(Y)
                if v < cur:
                    X[:] = Y; cur = v; improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 5e-13:
                    break
        return X, cur

    # initial quick global squeeze to have baseline
    if time.time() - t0 < LIM - 30:
        Xb = global_lse_polish(best, 2e-4, 60)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val, best = float(vb), Xb.copy()

    gmax0, prow0, spread0, nr0, nl0, _ = per_point_stats(best)
    best_bal = best.copy()
    best_bal_val = float(best_val)
    best_spread = float(spread0)
    best_ratt = int(nr0)

    # anneal schedule for (tau, tau2): 5 levels
    sched = [(2e-3, 1e-3), (8e-4, 4e-4), (2e-4, 1e-4), (8e-5, 3e-5), (5e-5, 2e-5)]

    n_cycles = 4
    for cyc in range(n_cycles):
        if time.time() - t0 > LIM - 24:
            break
        # start from most balanced state with small kick to change basin
        Xc = best_bal.copy()
        kick = 0.004 if cyc < 2 else 0.007
        Xc = _normalize(Xc + kick * rng.standard_normal(Xc.shape))
        # init weights uniform
        avec = np.ones(n) / n
        # balanced phase: iterate over schedule
        for (tau, tau2) in sched:
            if time.time() - t0 > LIM - 22:
                break
            # 1-2 weight updates per temperature
            for wrep in range(2):
                if time.time() - t0 > LIM - 22:
                    break
                Xc = bal_lse_polish(Xc, tau, tau2, avec, 60)
                # update weights: a <- a*exp(20(m_i - max)), renormalised
                G = Xc @ Xc.T
                np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                mx = float(prow.max())
                diff = prow - mx
                # clip exponent to [-3,0] for stability
                avec = avec * np.exp(np.maximum(20.0 * diff, -3.0))
                avec = np.maximum(avec, 1e-12)
                avec = avec / avec.sum()
                if time.time() - t0 > LIM - 22:
                    break
            # diagnostics
            gmax_c, prow_c, spread_c, nr_c, nl_c, _ = per_point_stats(Xc)
            # report (for log; no effect on verifier)
            # print(f"cyc{cyc} tau{tau:.0e} tau2{tau2:.0e} max{gmax_c:.10f} spread{spread_c:.3e} ratt{nr_c} loose{nl_c}")
        # ordinary global-max polish after balanced phase
        if time.time() - t0 > LIM - 20:
            break
        Xc = global_lse_polish(Xc, 4e-4, 60)
        Xc = global_lse_polish(Xc, 1e-4, 70)
        vc = _max_cos(Xc)
        Xc, vc = micro_split(Xc, vc, steps=180)
        gmax_c, prow_c, spread_c, nr_c, nl_c, _ = per_point_stats(Xc)
        # accept global best
        if vc < best_val - 1e-13:
            best_val, best = float(vc), Xc.copy()
        # accept balanced state: reduce rattlers or equalise m_i even at equal global max
        # allow tolerance 5e-4 above best for topology benefit
        tol = 5e-4
        improved_bal = False
        if vc < best_bal_val + tol:
            if nr_c < best_ratt:
                improved_bal = True
            elif nr_c == best_ratt and spread_c < best_spread * 0.995:
                improved_bal = True
            elif nl_c < int(((_normalize(best_bal) @ _normalize(best_bal).T).max(axis=1) < (_max_cos(best_bal) - 1e-4)).sum()) and nr_c <= best_ratt:
                improved_bal = True
            # also accept strictly better max with comparable balance
            if vc < best_bal_val - 1e-12 and spread_c < best_spread * 1.05:
                improved_bal = True
        if improved_bal:
            best_bal = Xc.copy()
            best_bal_val = float(vc)
            best_spread = float(spread_c)
            best_ratt = int(nr_c)
        # also if this cycle gave new global best, feed it to balanced pool if reasonably balanced
        if vc <= best_val + 1e-12 and (nr_c < best_ratt or spread_c < best_spread):
            best_bal = Xc.copy()
            best_bal_val = float(vc)
            best_spread = float(spread_c)
            best_ratt = int(nr_c)

    # finish from the most balanced state with standard squeeze
    # choose start: best if clearly better, else best_bal if close
    if best_bal_val < best_val + 2e-4:
        Xf = best_bal.copy()
    else:
        Xf = best.copy()
    # also consider best itself
    # polish both lightly and pick verified best
    cands_final = [best.copy(), best_bal.copy()]
    # final standard LP squeeze (global LSE at tiny tau) + micro
    best_out = best.copy()
    best_out_val = float(best_val)
    for X0 in cands_final:
        if time.time() - t0 > LIM - 8:
            break
        X1 = global_lse_polish(X0, 6e-5, 100)
        v1 = _max_cos(X1)
        X1, v1 = micro_split(X1, v1, steps=300)
        if v1 < best_out_val:
            best_out_val, best_out = float(v1), X1.copy()
    # one more ultra-fine polish on winner
    if time.time() - t0 < LIM - 3:
        X2 = global_lse_polish(best_out, 4e-5, 80)
        v2 = _max_cos(X2)
        if v2 < best_out_val:
            best_out_val, best_out = float(v2), X2.copy()
        X2, v2 = micro_split(best_out, best_out_val, steps=250)
        if v2 < best_out_val:
            best_out_val, best_out = float(v2), X2.copy()

    best_out = _normalize(best_out)
    return best_out
