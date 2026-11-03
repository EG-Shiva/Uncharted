import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, t0, deadline):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - t0 >= deadline:
            break
        if now - t0 + 0.4 >= deadline:
            break
        # budget for this polish
        if budget is not None and budget <= 0:
            break
        t_start = time.time()
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1600:
            csel_all = c[sel]
            idx = np.argpartition(-csel_all, 1600)[:1600]
            I = I[idx]; J = J[idx]
            csel = csel_all[idx]
        else:
            csel = c[sel]
            p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        spent = time.time() - t_start
        if budget is not None:
            budget -= spent
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if budget is not None and budget < 0.5:
            break
    return x, best

def solve():
    t0 = time.time()
    DEADLINE = 150.0
    HARD = 168.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    incumbent = xb.copy()
    inc_val = _max_cos(incumbent)
    x = incumbent.copy()
    cur_best = inc_val

    def out_of_time():
        return (time.time() - t0) >= DEADLINE

    taus = [cur_best - 2e-4, cur_best - 5e-4, cur_best - 1e-3]

    iu0 = np.triu_indices(n, 1)
    # squeeze iterations
    # allocate roughly: 3 taus x 350 iters
    for ti, tau in enumerate(taus):
        if out_of_time():
            break
        # trust radius schedule for this tau
        NIT = 350
        # rebasing tau if we already improved a lot
        # keep tau as min(tau, cur_best - 1e-4)
        if cur_best < tau:
            # already feasible for this tau, tighten a bit
            tau = cur_best - 5e-5
        for it in range(NIT):
            if out_of_time():
                break
            if (time.time() - t0) >= HARD - 45.0 and ti >= 1 and it > 100:
                break
            frac = it / max(NIT - 1, 1)
            R = 0.08 - frac * 0.05  # 0.08 -> 0.03
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            if m < inc_val:
                inc_val = m
                incumbent = x.copy()
                cur_best = m
                # tighten tau slightly if feasible
                if m < tau:
                    # achieved target: move to next tau early if very good
                    pass
            # --- violation gradient step (min sum max(0, G-tau)) ---
            # symmetric mask
            Mask = G > tau
            # number of violations
            # gradient: Mask @ x
            # for speed use float
            if np.any(Mask):
                Gr = Mask.astype(float) @ x
                dot = np.sum(Gr * x, axis=1, keepdims=True)
                Gt = Gr - dot * x
                mxn = np.linalg.norm(Gt, axis=1).max()
                if np.isfinite(mxn) and mxn > 1e-14:
                    dx = -(R * 0.7) * Gt / mxn
                    x = x + dx
                    x /= np.linalg.norm(x, axis=1, keepdims=True)
            else:
                # feasible: take small random tangent kick to seek lower max
                Z = rng.standard_normal((n, d))
                Z -= np.sum(Z * x, axis=1, keepdims=True) * x
                zn = np.linalg.norm(Z, axis=1).max()
                if zn > 1e-12:
                    x = x + (R * 0.3) * Z / zn
                    x /= np.linalg.norm(x, axis=1, keepdims=True)
                if m < tau:
                    # update tau within same block to push further
                    tau -= 1e-4
            # --- null-space slide every 12 steps ---
            if it % 12 == 5:
                if out_of_time():
                    break
                # active contacts: top ~1100 pairs by G
                G2 = x @ x.T
                np.fill_diagonal(G2, -2.0)
                c = G2[iu0]
                K = 1100
                if len(c) < K:
                    K = len(c)
                part = np.argpartition(-c, K - 1)[:K]
                Ia = iu0[0][part]
                Ja = iu0[1][part]
                # random tangent noise
                Z = rng.standard_normal((n, d))
                Z -= np.sum(Z * x, axis=1, keepdims=True) * x
                # Kaczmarz projection: 2 sweeps
                order = np.arange(K)
                rng.shuffle(order)
                for sweep in range(2):
                    for kk in order:
                        i = int(Ia[kk]); j = int(Ja[kk])
                        # residual
                        ci = float(Z[i] @ x[j] + x[i] @ Z[j])
                        if ci == 0.0:
                            continue
                        s = float(x[i] @ x[j])
                        denom = 2.0 * (1.0 - s * s)
                        if denom < 1e-6:
                            continue
                        a = -ci / denom
                        # e vectors
                        eij = x[j] - s * x[i]
                        eji = x[i] - s * x[j]
                        Z[i] += a * eij
                        Z[j] += a * eji
                    # re-tangentize
                    Z -= np.sum(Z * x, axis=1, keepdims=True) * x
                zn = np.linalg.norm(Z, axis=1).max()
                if not np.finite(zn) or zn < 1e-12:
                    continue
                Rn = 0.05 if ti < 2 else 0.04
                Z = Z / zn * Rn
                # occasional larger second-order step: two consecutive null steps
                xt = x + Z
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                G3 = xt @ xt.T
                np.fill_diagonal(G3, -2.0)
                mt = float(G3.max())
                # accept if not too bad (allow uphill to change basin)
                if mt < m + 0.008:
                    x = xt
                    if mt < inc_val:
                        inc_val = mt
                        incumbent = x.copy()
                        cur_best = mt
                # else discard
        # end it loop
        # re-anchor slightly to incumbent if drifted too far
        cur_m = _max_cos(x)
        if cur_m > inc_val + 0.004:
            # restart from incumbent with shake to try new basin for next tau
            x = incumbent.copy()
            Z = rng.standard_normal((n, d))
            Z -= np.sum(Z * x, axis=1, keepdims=True) * x
            zn = np.linalg.norm(Z, axis=1).max()
            if zn > 1e-12:
                x = x + 0.04 * Z / zn
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        if out_of_time():
            break
    # short smoothing toward incumbent region + final LP polish
    if (time.time() - t0) < DEADLINE:
        # start polish from incumbent
        remain = DEADLINE - (time.time() - t0)
        # use best of x and incumbent
        vm = _max_cos(x)
        start_pt = incumbent.copy() if inc_val <= vm else x.copy()
        # if x is better, update incumbent
        if vm < inc_val:
            inc_val = vm
            incumbent = x.copy()
            start_pt = x.copy()
        budget = min(remain - 0.5, 40.0)
        if budget > 2.0:
            try:
                xp, vp = _lp_polish(start_pt, budget, t0, DEADLINE)
                if vp < inc_val:
                    incumbent, inc_val = xp, vp
                else:
                    # keep vertex even if equal
                    if vp <= inc_val + 1e-12:
                        incumbent = xp
                        inc_val = vp
            except Exception:
                pass
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
