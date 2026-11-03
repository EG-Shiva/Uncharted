import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def feasibility_target(x_init, t_target, best_inc, t_end, rng, P0=300):
    n, d = x_init.shape
    x = x_init.copy()
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c_all = G[iu]
    sum_viol = float(np.sum(np.maximum(0.0, c_all - t_target)))
    cur_max = float(c_all.max())
    r = 0.005
    r_max = 0.02
    r_min = 1e-9
    stall = 0
    # precompute arange for eq
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)
    it = 0
    max_it = 80
    P = P0
    while time.time() < t_end and it < max_it and r > r_min * 0.5:
        it += 1
        if cur_max < t_target:
            break
        # select currently most violated 200-500 pairs (largest cosine)
        # recompute Gram already available; get top P
        if len(c_all) > P:
            # argpartition for speed
            idx = np.argpartition(-c_all, P - 1)[:P]
            # sort those for stability (optional)
            o = np.argsort(-c_all[idx])
            sel_idx = idx[o]
        else:
            sel_idx = np.argsort(-c_all)
            Pcur = len(sel_idx)
        I_all, J_all = iu[0][sel_idx], iu[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = len(sel_idx)
        nv = n * d + p
        # Build A_ub: p rows, each: u_I * x_J + u_J * x_I - s_k <= t - c_k
        rows = np.repeat(np.arange(p), 2 * d + 1)
        col_uI = (I_all[:, None] * d + np.arange(d)).ravel()
        col_uJ = (J_all[:, None] * d + np.arange(d)).ravel()
        col_s = (n * d + np.arange(p)).repeat(2 * d + 1)
        # need cols aligned per row: [uI(d), uJ(d), s(1)]
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::(2 * d + 1)] = 0  # placeholder, will overwrite blockwise
        # build blockwise faster with concatenate
        cols = np.concatenate([ (I_all[:, None] * d + np.arange(d)),
                                (J_all[:, None] * d + np.arange(d)),
                                (np.full((p, 1), 0)) ], axis=1)  # temp
        # Actually construct cols array correctly:
        cols_blocks = np.concatenate([I_all[:, None] * d + np.arange(d),
                                      J_all[:, None] * d + np.arange(d),
                                      np.full((p, 1), -1)], axis=1)
        # last column is slack index
        cols_arr = cols_blocks.copy()
        cols_arr[:, -1] = n * d + np.arange(p)
        cols_flat = cols_arr.ravel()
        vals_blocks = np.concatenate([x[J_all], x[I_all], -np.ones((p, 1))], axis=1)
        vals_flat = vals_blocks.ravel()
        rows_flat = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals_flat, (rows_flat, cols_flat)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
        except Exception:
            r *= 0.5
            continue
        obj = np.zeros(nv)
        obj[n * d:] = 1.0
        b_ub = t_target - c_sel
        bounds = [(-r, r)] * (n * d) + [(0.0, None)] * p
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            r *= 0.5
            stall += 1
            continue
        if res.status != 0:
            r *= 0.5
            stall += 1
            if stall >= 6:
                # tiny tangent kick to change active set
                pert = rng.standard_normal((n, d)) * (r * 0.5)
                pert -= np.sum(pert * x, axis=1, keepdims=True) * x
                x = x + pert
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                c_all = G[iu]
                cur_max = float(c_all.max())
                sum_viol = float(np.sum(np.maximum(0.0, c_all - t_target)))
                stall = 0
            continue
        u = res.x[:n * d].reshape(n, d)
        # predicted slack sum
        # pred = float(res.fun)
        y = x + u
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        cy = Gy[iu]
        cur_max_y = float(cy.max())
        sum_viol_y = float(np.sum(np.maximum(0.0, cy - t_target)))
        # progress if total slack decreased
        if sum_viol_y < sum_viol * (1.0 - 1e-9) or (sum_viol == 0 and cur_max_y < cur_max):
            x = y
            G = y @ y.T  # will set diag next iter via c_all? keep diag -2
            # recompute c_all
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            # use G2 for next selection; store iu values
            c_all = G2[iu]
            cur_max = cur_max_y
            sum_viol = sum_viol_y
            r = min(r * 1.25, r_max)
            stall = 0
            # adapt P: if many violations, use larger set
            nviol = int(np.sum(c_all > t_target))
            if nviol > 450:
                P = 500
            elif nviol > 300:
                P = 400
            elif nviol < 150:
                P = 200
            else:
                P = 300
            # early feasible w.r.t incumbent?
            if cur_max < best_inc - 1e-12:
                break
        else:
            r *= 0.5
            stall += 1
            if stall >= 8:
                # diversify slightly without losing too much
                pert = rng.standard_normal((n, d)) * (max(r, 1e-4) * 0.7)
                pert -= np.sum(pert * x, axis=1, keepdims=True) * x
                xt = x + pert
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
                Gt = xt @ xt.T
                np.fill_diagonal(Gt, -2.0)
                ct = Gt[iu]
                svt = float(np.sum(np.maximum(0.0, ct - t_target)))
                # accept kick if not much worse (within factor 2)
                if svt <= max(sum_viol * 2.0, sum_viol + 1e-6):
                    x = xt
                    c_all = ct
                    cur_max = float(ct.max())
                    sum_viol = svt
                stall = 0
                r = max(r, 2e-4)
    return x, cur_max


def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    rng = np.random.default_rng(0)
    # load best
    xb = None
    best_val = 1e9
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best_val = max_cos(a)
    n, d = x_best.shape
    incumbent = best_val
    base = 0.26718814
    # if incumbent already below base, use incumbent as base
    ref = min(base, incumbent)
    deltas = [1e-6, 3e-6, 1e-5, 3e-5]
    targets = [ref - dd for dd in deltas]
    # time budget: split feasibility, reserve 30s for final polish
    # each target gets at most ~25s
    for ti, t_target in enumerate(targets):
        if time.time() > T_END - 32.0:
            break
        # skip if target already achieved
        if best_val < t_target:
            continue
        per_target = min(25.0, max(5.0, (T_END - 32.0 - time.time()) / (len(targets) - ti)))
        t_lim = min(time.time() + per_target, T_END - 30.0)
        if t_lim <= time.time() + 1.0:
            break
        # start from current best (feasible for higher t, infeasible for this t)
        x_start = x_best.copy()
        x_new, m_new = feasibility_target(x_start, float(t_target), float(best_val), t_lim, rng, P0=300)
        v_new = max_cos(x_new / np.linalg.norm(x_new, axis=1, keepdims=True))
        x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
        if v_new < best_val - 1e-12 and np.all(np.isfinite(x_new)):
            x_best = x_new
            best_val = v_new
            # if we beat target, continue to harder target from new point
            # quick intermediate polish to snap to local min (short)
            if time.time() < T_END - 30.0:
                remain = min(4.0, T_END - 30.0 - time.time())
                if remain > 1.0:
                    x_best = lp_polish(x_best, remain, T_END - 28.0)
                    best_val = max_cos(x_best)
        # also if we improved but not below target, keep best if better than incumbent
        # (feasibility may have reduced max even without reaching t)
        # check: x_new may be better than incumbent even if > t
        # already handled above
    # final ordinary LP polish under strict deadline
    time_left = T_END - time.time() - 1.0
    if time_left > 2.0:
        # use most of remaining but leave 1s margin
        x_best = lp_polish(x_best, time_left, T_END)
        best_val = max_cos(x_best)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
