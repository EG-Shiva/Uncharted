import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _ordinary_polish(x, seconds, r0=1e-3, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    best_x = x.copy()
    radius = float(r0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-14:
            break
        try:
            c = (x @ x.T)[iu0]
        except Exception:
            break
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            k = 2200
            idx = np.argpartition(c, len(c) - k)[len(c) - k:]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)):
            radius *= 0.5
            continue
        nr[nr < 1e-12] = 1e-12
        y = y / nr
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def solve():
    np.random.seed(0)
    t0 = time.time()
    hard_end = t0 + 165.0
    rng = np.random.RandomState(12345)
    n, d = 102, 12
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = rng.randn(n, d)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        if xb.size == n * d:
            xb = xb.reshape(n, d)
        else:
            xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass

    x_best = xb.copy()
    best_val = _max_cos(x_best)

    iu = np.triu_indices(n, 1)
    Npairs = len(iu[0])
    Nd = n * d

    deltas = [1e-6, 3e-6, 1e-5, 3e-5]

    # sequential target feasibility
    for di, delta in enumerate(deltas):
        if time.time() >= hard_end - 16.0:
            break
        t_target = float(best_val - delta)
        # time slice for this target: share remaining (reserve 14s for final polish)
        remain = (hard_end - 14.0) - time.time()
        if remain <= 2.0:
            break
        n_rem = len(deltas) - di
        slice_sec = min(32.0, remain / n_rem)
        slice_end = time.time() + slice_sec

        # start point for this target: current best (deterministic)
        x = x_best.copy()
        r = 0.003
        if delta >= 1e-5:
            r = 0.005
        K = 350
        # adapt K within 200..500: larger target -> larger K
        if delta >= 1e-5:
            K = 450
        # initial violation
        try:
            c_all0 = (x @ x.T)[iu]
            V_prev = float(np.sum(np.maximum(0.0, c_all0 - t_target)))
        except Exception:
            V_prev = float('inf')
        stall = 0

        # inner trust-region slack loop
        for it in range(250):
            if time.time() >= slice_end or time.time() >= hard_end - 14.0:
                break
            try:
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                # c over upper tri via iu view of G? recompute for selection
                c_all = G[iu]
                maxv = float(np.max(c_all))
            except Exception:
                break
            # track global incumbent (strictly below)
            if maxv < best_val - 1e-12:
                best_val = maxv
                x_best = x.copy()
                # tighten target to stay below new best? keep original t_target
                # if we already beat target, success
                if maxv <= t_target:
                    break
            if maxv <= t_target:
                # feasible for this target
                best_val = min(best_val, maxv)
                x_best = x.copy()
                break
            V_cur = float(np.sum(np.maximum(0.0, c_all - t_target)))
            # choose most violated K pairs
            if K >= Npairs:
                sel_idx = np.arange(Npairs)
            else:
                # argpartition top-K largest
                part = np.argpartition(-c_all, K)[:K]
                # sort them descending for determinism
                order = np.argsort(-c_all[part], kind='stable')
                sel_idx = part[order]
            ck = c_all[sel_idx]
            Ik = iu[0][sel_idx]
            Jk = iu[1][sel_idx]
            p = K if K < Npairs else Npairs
            nv = Nd + p
            # build LP: min sum s s.t. <u_i,x_j>+<u_j,x_i> - s_k <= t - c_k
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cI = (Ik[:, None] * d + np.arange(d)).ravel()
            cJ = (Jk[:, None] * d + np.arange(d)).ravel()
            # interleave per row: need cols in row-major order
            cols = np.empty(p * (2 * d + 1), dtype=np.int64)
            cols[0::(2 * d + 1)] = np.zeros(p, dtype=np.int64)  # placeholder
            # build row by row via reshape
            cols_m = np.concatenate([Ik[:, None] * d + np.arange(d),
                                     Jk[:, None] * d + np.arange(d),
                                     np.full((p, 1), Nd)], 1)
            # need s col = Nd + k
            cols_m[:, 2 * d] = Nd + np.arange(p)
            cols = cols_m.ravel()
            try:
                vals_m = np.concatenate([x[Jk], x[Ik], -np.ones((p, 1))], 1)
            except Exception:
                r *= 0.5
                continue
            vals = vals_m.ravel()
            b_ub = (t_target - ck)
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                r *= 0.5
                stall += 1
                continue
            obj = np.zeros(nv)
            obj[Nd:] = 1.0
            bounds = [(-r, r)] * Nd + [(0.0, None)] * p
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                r *= 0.5
                stall += 1
                continue
            if res.status != 0:
                r *= 0.5
                stall += 1
                if r < 1e-9:
                    # tiny deterministic kick to change topology
                    pert = 1e-4 * rng.randn(n, d)
                    # make tangent
                    pert -= np.sum(pert * x, axis=1, keepdims=True) * x
                    x = x + pert
                    x = x / np.linalg.norm(x, axis=1, keepdims=True)
                    r = 0.001
                    stall = 0
                continue
            try:
                u = res.x[:Nd].reshape(n, d)
            except Exception:
                r *= 0.5
                continue
            unorm = float(np.linalg.norm(u))
            if not np.isfinite(unorm):
                r *= 0.5
                continue
            if unorm < 1e-13:
                # linearised optimum with no move: shrink or kick
                r *= 0.5
                stall += 1
                if stall >= 6:
                    pert = (r + 1e-4) * rng.randn(n, d)
                    pert -= np.sum(pert * x, axis=1, keepdims=True) * x
                    x = x + pert
                    x = x / np.linalg.norm(x, axis=1, keepdims=True)
                    r = max(r, 8e-4)
                    stall = 0
                continue
            y = x + u
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[~np.isfinite(nr)] = 1.0
            nr[nr < 1e-12] = 1e-12
            y = y / nr
            try:
                Gy = y @ y.T
                np.fill_diagonal(Gy, -2.0)
                cy = Gy[iu]
                maxy = float(np.max(cy))
                Vy = float(np.sum(np.maximum(0.0, cy - t_target)))
            except Exception:
                r *= 0.5
                continue
            # progress if total slack decreased
            if Vy < V_cur - 1e-13 * max(1.0, abs(V_cur)):
                x = y
                V_prev = Vy
                r = min(r * 1.3, 0.02)
                stall = 0
                if maxy < best_val - 1e-12:
                    best_val = maxy
                    x_best = y.copy()
                    if maxy <= t_target:
                        break
            else:
                # stall: shrink trust region, keep x
                # but if y improves max even without slack progress, accept as new best start
                if maxy < best_val - 1e-12:
                    best_val = maxy
                    x_best = y.copy()
                    x = y
                    V_prev = Vy
                    stall = 0
                    r = min(r * 1.1, 0.02)
                    if maxy <= t_target:
                        break
                else:
                    r *= 0.6
                    stall += 1
                    if r < 5e-7:
                        # re-seed around current best with small kick
                        pert = 0.002 * rng.randn(n, d)
                        pert -= np.sum(pert * x, axis=1, keepdims=True) * x
                        x = x + pert
                        x = x / np.linalg.norm(x, axis=1, keepdims=True)
                        r = 0.0015
                        stall = 0
            # early exit if feasible found
            if _max_cos(x) <= t_target:
                if _max_cos(x) < best_val:
                    best_val = _max_cos(x)
                    x_best = x.copy()
                break

    # one final ordinary LP polish under strict deadline
    final_end = hard_end - 1.0
    secs = final_end - time.time()
    if secs > 0.5:
        secs = min(secs, 16.0)
        try:
            x_pol = _ordinary_polish(x_best, seconds=secs, r0=1e-3, t_end=final_end)
            if _max_cos(x_pol) < best_val:
                x_best = x_pol
                best_val = _max_cos(x_pol)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
