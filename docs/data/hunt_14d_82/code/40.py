import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _eval_all(X, iu0):
    G = X @ X.T
    c = G[iu0]
    m = float(np.max(c))
    # sort descending for tail stats
    s = np.sort(c)[::-1]
    mean100 = float(np.mean(s[:100]))
    top500 = s[:500]
    p = 2000.0
    # stable log-sum-exp
    d = p * (top500 - m)
    np.clip(d, -50.0, 0.0, out=d)
    E = np.exp(d)
    se = float(np.sum(E))
    if se <= 0 or not np.finite(se):
        lse = m
    else:
        lse = m + np.log(se) / p
    return m, mean100, lse, c, s

def solve():
    t0 = time.time()
    T_OPT_END = t0 + 132.0  # stop optimisation by ~132s
    n, d = 82, 14
    nd = n * d
    iu0 = np.triu_indices(n, 1)
    ar_d = np.arange(d)

    # ---- load start ----
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng = np.random.default_rng(0)
        cands = [_norm(rng.standard_normal((n, d)))]
    # pick lowest max as start
    def _m(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))
    cands.sort(key=_m)
    X_best = cands[0].copy()
    X_best = _norm(X_best)

    best_m, best_mean, best_lse, _, _ = _eval_all(X_best, iu0)
    X_cur = X_best.copy()
    cur_m, cur_mean, cur_lse = best_m, best_mean, best_lse

    radius = 5e-4
    RMIN = 1e-7
    RMAX = 1e-3
    it = 0
    fails = 0

    # preallocate eq row/col for speed (same every iter except values)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(nd)

    while time.time() < T_OPT_END:
        it += 1
        if radius < RMIN:
            # restart with medium radius to keep moving
            radius = 5e-4
            fails = 0
            # if stuck for long, tiny deterministic nudge to change active set
            # (kept tiny so max does not blow up)
            if it % 20 == 0:
                pass
        # evaluate current
        G = X_cur @ X_cur.T
        c_all = G[iu0]
        m_cur = float(np.max(c_all))

        # ---- build active set ----
        # alternate: even -> minimax wide, odd -> sum step (but both need wide constr)
        is_minimax = (it % 2 == 1)
        # wide set: top 1500 or all within 2e-3
        thr_wide = m_cur - 2e-3
        sel_wide = c_all > thr_wide
        cnt_wide = int(np.count_nonzero(sel_wide))
        K_WANT = 1500
        if cnt_wide < 1200:
            K_WANT = 1200
        elif cnt_wide > 2000:
            K_WANT = 2000
        else:
            K_WANT = max(cnt_wide, 1200)
            # cap to 2000 handled below
            K_WANT = min(K_WANT, 2000)
        if cnt_wide > K_WANT:
            idx = np.where(sel_wide)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, K_WANT - 1)[:K_WANT]
            keep = np.zeros_like(sel_wide, dtype=bool)
            keep[idx[part]] = True
            sel_wide = keep
        elif cnt_wide < K_WANT:
            # take top K_WANT
            K_WANT = min(K_WANT, len(c_all))
            order = np.argpartition(-c_all, K_WANT - 1)[:K_WANT]
            keep = np.zeros_like(sel_wide, dtype=bool)
            keep[order] = True
            sel_wide = keep

        Iw = iu0[0][sel_wide]
        Jw = iu0[1][sel_wide]
        cw = c_all[sel_wide]
        p_w = len(Iw)

        # top-500 for sum objective / metrics
        K_SUM = 500
        if len(c_all) >= K_SUM:
            o500 = np.argpartition(-c_all, K_SUM - 1)[:K_SUM]
        else:
            o500 = np.arange(len(c_all))
        I500 = iu0[0][o500]
        J500 = iu0[1][o500]
        c500 = c_all[o500]

        delta = None
        use_t = is_minimax

        # time check before heavy LP
        if time.time() >= T_OPT_END:
            break

        try:
            if is_minimax:
                # minimax LP: min t s.t. cw + Xj.di + Xi.dj - t <= 0
                nv = nd + 1
                rows = np.repeat(np.arange(p_w), 2 * d + 1)
                cols = np.concatenate(
                    [Iw[:, None] * d + ar_d, Jw[:, None] * d + ar_d,
                     np.full((p_w, 1), nd)], axis=1
                ).ravel()
                vals = np.concatenate([X_cur[Jw], X_cur[Iw], -np.ones((p_w, 1))], axis=1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p_w, nv)).tocsr()
                b_ub = -cw
                A_eq = coo_matrix((X_cur.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                obj[-1] = 1.0
                bounds = [(-radius, radius)] * nd + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs",
                              options={"time_limit": 1.0, "presolve": True})
                if res.status != 0 or res.x is None:
                    raise RuntimeError("lp fail")
                delta = res.x[:-1].reshape(n, d)
            else:
                # tail-sum LP: min sum_{top500} linearised dot
                # s.t. wide linearised <= best_m (no pair exceeding current best)
                # uniform weights
                nv = nd
                # constraints from wide set
                rows = np.repeat(np.arange(p_w), 2 * d)
                cols = np.concatenate(
                    [Iw[:, None] * d + ar_d, Jw[:, None] * d + ar_d], axis=1
                ).ravel()
                vals = np.concatenate([X_cur[Jw], X_cur[Iw]], axis=1).ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p_w, nv)).tocsr()
                # allow tiny slack for linearization error
                b_ub = (cur_m - cw) + 1e-12
                # b_ub >=0 since cw<=cur_m; clip negatives from roundoff
                # (keep as is, feasible at 0)
                A_eq = coo_matrix((X_cur.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv)
                # uniform weight 1.0
                # accumulate: obj[i*d:(i+1)*d] += X[j]
                # vectorized via bincount-like loop with np.add.at
                # build with iteration over 500 (cheap)
                for k in range(len(I500)):
                    ii = int(I500[k]); jj = int(J500[k])
                    obj[ii * d:(ii + 1) * d] += X_cur[jj]
                    obj[jj * d:(jj + 1) * d] += X_cur[ii]
                bounds = [(-radius, radius)] * nd
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs",
                              options={"time_limit": 1.0, "presolve": True})
                if res.status != 0 or res.x is None:
                    raise RuntimeError("lp fail")
                delta = res.x.reshape(n, d)
        except Exception:
            radius *= 0.5
            fails += 1
            continue

        # propose
        Y = X_cur + delta
        # project (tangent already, just normalize)
        Y = _norm(Y)
        if not np.all(np.isfinite(Y)):
            radius *= 0.5
            fails += 1
            continue
        new_m, new_mean, new_lse, _, _ = _eval_all(Y, iu0)

        # lexicographic acceptance
        accept = False
        if new_m < cur_m - 1e-12:
            accept = True
        elif abs(new_m - cur_m) <= 1e-12:
            if new_mean < cur_mean - 1e-12:
                accept = True
            elif abs(new_mean - cur_mean) <= 1e-12 and new_lse < cur_lse - 1e-10:
                accept = True
        # else reject (max increased)

        if accept:
            X_cur = Y
            cur_m, cur_mean, cur_lse = new_m, new_mean, new_lse
            # track global best (same as cur since monotonic, but keep copy)
            # update best tuple
            best_m, best_mean, best_lse = cur_m, cur_mean, cur_lse
            X_best = X_cur.copy()
            radius = min(radius * 1.5, RMAX)
            fails = 0
        else:
            radius *= 0.5
            fails += 1

        # occasional time check already in loop; limit iterations for safety
        if it > 500:
            break
        # if many fails, keep halving (handled); avoid infinite tight loop when time low
        if time.time() >= T_OPT_END:
            break

    X_best = _norm(np.asarray(X_best, dtype=float))
    # tiny safety: ensure no NaN/zero
    norms = np.linalg.norm(X_best, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_best = X_best / norms
    return X_best
