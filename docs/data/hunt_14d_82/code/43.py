import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _all_cos(X, iu):
    G = X @ X.T
    c = G[iu]
    return c, float(c.max())

def _metrics_from_c(c):
    m = float(c.max())
    # top sorts: n=3321 small
    s = np.sort(c)[::-1]
    mean100 = float(s[:100].mean())
    top500 = s[:500]
    beta = 1000.0
    # stable log-mean-exp
    # top500 max == m
    e = np.exp(np.clip(beta * (top500 - m), -30.0, 0.0))
    lse = m + float(np.log(np.mean(e) + 1e-300)) / beta
    return m, mean100, lse

def _is_better(new_t, cur_t):
    nm, nmean, nlse = new_t
    cm, cmean, clse = cur_t
    if nm < cm - 1e-12:
        return True
    if abs(nm - cm) <= 1e-12:
        if nmean < cmean - 1e-12:
            return True
        if abs(nmean - cmean) <= 1e-12:
            if nlse < clse - 5e-13:
                return True
    return False

def solve():
    t0 = time.time()
    DEADLINE = t0 + 135.0
    HARD = t0 + 138.0
    n, d = 82, 14
    nd = n * d
    iu = np.triu_indices(n, 1)
    I0 = iu[0]; J0 = iu[1]
    ar_d = np.arange(d)

    # load start
    X0 = None
    best_load_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                v = float(G.max())
                if v < best_load_val:
                    best_load_val = v
                    X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        rng = np.random.default_rng(0)
        X0 = _norm(rng.standard_normal((n, d)))

    X_cur = X0.copy()
    c_cur_all, _ = _all_cos(X_cur, iu)
    cur_trip = _metrics_from_c(c_cur_all)
    best_max = cur_trip[0]
    X_best = X_cur.copy()
    best_trip = cur_trip

    radius = 5e-4
    RMIN = 1e-7
    RMAX = 1e-3

    it = 0
    # preallocate eq row/col template (first nd cols)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(nd)

    while time.time() < DEADLINE - 2.0:
        it += 1
        if time.time() > HARD - 3.0:
            break
        # current wide set based on X_cur
        G = X_cur @ X_cur.T
        c_all = G[I0, J0]
        m_cur = float(c_all.max())
        # determine wide set: top 1500 plus all within 2e-3, capped 2000
        K0 = 1500
        CAPW = 2000
        # argpartition for top CAPW to evaluate threshold quickly
        if len(c_all) > CAPW:
            part = np.argpartition(-c_all, CAPW - 1)[:CAPW]
            # threshold = min among those CAPW
            thr_cap = float(c_all[part].min())
        else:
            part = np.arange(len(c_all))
            thr_cap = float(c_all.min())
        thr_shell = m_cur - 2e-3
        thr = thr_shell if thr_shell < thr_cap else thr_cap
        # select all > thr, but cap to 2000 tightest
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt > CAPW:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            pp = np.argpartition(-vals, CAPW - 1)[:CAPW]
            sel2 = np.zeros_like(sel)
            sel2[idx[pp]] = True
            sel = sel2
            cnt = CAPW
        if cnt < 200:
            # ensure at least 500
            idx2 = np.argpartition(-c_all, 499)[:500]
            sel[idx2] = True
            cnt = int(sel.sum())
        Iw = I0[sel]; Jw = J0[sel]
        cw = c_all[sel]
        p = len(Iw)

        # top-500 indices for metrics/objective (within full set)
        # get top 500 of full
        top500_idx = np.argpartition(-c_all, 499)[:500]
        c500 = c_all[top500_idx]
        I500 = I0[top500_idx]; J500 = J0[top500_idx]
        beta_w = 1000.0
        w500 = np.exp(np.clip(beta_w * (c500 - m_cur), -30.0, 0.0))
        wsum = float(w500.sum()) + 1e-300
        w500 = w500 / wsum

        # ---- LP1: minimise t ----
        # build A_ub sparse p x (nd+1)
        # rows repeat
        did_lp1 = False
        if time.time() < DEADLINE - 2.0 and radius >= RMIN:
            try:
                rows = np.repeat(np.arange(p), 2 * d + 1)
                cols_I = (Iw[:, None] * d + ar_d).ravel()
                cols_J = (Jw[:, None] * d + ar_d).ravel()
                cols_t = np.full(p, nd)
                cols = np.concatenate([cols_I, cols_J, cols_t]) if False else None
                # need per-row interleaving: rows = repeat, cols must align row-wise
                # build cols row-wise
                cI = Iw[:, None] * d + ar_d  # p x d
                cJ = Jw[:, None] * d + ar_d
                cols_mat = np.concatenate([cI, cJ, np.full((p, 1), nd)], axis=1)  # p x (2d+1)
                cols = cols_mat.ravel()
                vI = X_cur[Jw]  # p x d
                vJ = X_cur[Iw]
                vals_mat = np.concatenate([vI, vJ, -np.ones((p, 1))], axis=1)
                vals = vals_mat.ravel()
                nv = nd + 1
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X_cur.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                b_ub = -cw
                b_eq = np.zeros(n)
                bounds = [(-radius, radius)] * nd + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=bounds, method="highs",
                              options={"time_limit": 1.0})
                did_lp1 = (res.status == 0)
                if did_lp1:
                    Delta = res.x[:-1].reshape(n, d)
                    Y = _norm(X_cur + Delta)
                    cy, _ = _all_cos(Y, iu)
                    tripY = _metrics_from_c(cy)
                    if _is_better(tripY, cur_trip):
                        X_cur = Y
                        cur_trip = tripY
                        c_all = cy
                        m_cur = tripY[0]
                        if tripY[0] < best_max - 1e-13:
                            best_max = tripY[0]
                            X_best = Y.copy()
                            best_trip = tripY
                        radius = min(radius * 1.3, RMAX)
                    else:
                        radius *= 0.5
                else:
                    radius *= 0.5
            except Exception:
                radius *= 0.5

        if time.time() > HARD - 3.0 or time.time() > DEADLINE - 2.0:
            break
        if radius < RMIN:
            radius = 2e-4

        # ---- LP2: minimise weighted sum of top-500 subject to <= m_cur ----
        try:
            # rebuild wide set RHS w.r.t updated m_cur if LP1 accepted (reuse if not)
            # for simplicity reuse Iw,Jw,cw if max unchanged much, else recompute quickly
            # recompute RHS cap = m_cur (current)
            # A_ub2: p x nd
            rows2 = np.repeat(np.arange(p), 2 * d)
            cI2 = Iw[:, None] * d + ar_d
            cJ2 = Jw[:, None] * d + ar_d
            cols2 = np.concatenate([cI2, cJ2], axis=1).ravel()
            vI2 = X_cur[Jw]
            vJ2 = X_cur[Iw]
            vals2 = np.concatenate([vI2, vJ2], axis=1).ravel()
            A_ub2 = coo_matrix((vals2, (rows2, cols2)), shape=(p, nd)).tocsr()
            A_eq2 = coo_matrix((X_cur.ravel(), (eq_rows, eq_cols)), shape=(n, nd)).tocsr()
            # objective from top500 (recompute weights w.r.t X_cur if LP1 moved)
            if did_lp1:
                # refresh top500 weights for current X_cur
                G2 = X_cur @ X_cur.T
                c_all2 = G2[I0, J0]
                m2 = float(c_all2.max())
                top500_idx2 = np.argpartition(-c_all2, 499)[:500]
                c500_2 = c_all2[top500_idx2]
                I500_2 = I0[top500_idx2]; J500_2 = J0[top500_idx2]
                w2 = np.exp(np.clip(beta_w * (c500_2 - m2), -30.0, 0.0))
                w2 = w2 / (float(w2.sum()) + 1e-300)
            else:
                I500_2, J500_2, w2 = I500, J500, w500
                c_all2 = c_all
                m2 = m_cur
            obj2 = np.zeros(nd)
            # accumulate: loop 500 (fast)
            Xc = X_cur
            for k in range(len(I500_2)):
                i = int(I500_2[k]); j = int(J500_2[k])
                w = float(w2[k])
                obj2[i*d:(i+1)*d] += w * Xc[j]
                obj2[j*d:(j+1)*d] += w * Xc[i]
            # scale obj to avoid tiny numbers
            on = float(np.linalg.norm(obj2)) + 1e-300
            # keep scale ~1
            obj2 = obj2 / on * 10.0
            b_ub2 = (m2 - cw) if did_lp1 is False else None
            if b_ub2 is None:
                # need cw consistent with X_cur: recompute cw for Iw,Jw
                # cw was for old X_cur; recompute
                Gc = X_cur @ X_cur.T
                cw_new = Gc[Iw, Jw]
                b_ub2 = m2 - cw_new
                # also allow tiny slack 1e-12
                b_ub2 = np.maximum(b_ub2, 0.0) + 1e-12
            else:
                b_ub2 = np.maximum(b_ub2, 0.0) + 1e-12
            # clip large RHS to avoid loose constraints dominating? keep as is
            b_eq2 = np.zeros(n)
            bounds2 = [(-radius, radius)] * nd
            res2 = linprog(obj2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                           bounds=bounds2, method="highs",
                           options={"time_limit": 1.0})
            if res2.status == 0:
                Delta2 = res2.x.reshape(n, d)
                # if step tiny, skip eval? still eval
                Y2 = _norm(X_cur + Delta2)
                cy2, _ = _all_cos(Y2, iu)
                tripY2 = _metrics_from_c(cy2)
                if _is_better(tripY2, cur_trip):
                    X_cur = Y2
                    cur_trip = tripY2
                    if tripY2[0] < best_max - 1e-13:
                        best_max = tripY2[0]
                        X_best = Y2.copy()
                        best_trip = tripY2
                    radius = min(radius * 1.25, RMAX)
                else:
                    radius *= 0.5
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5

        if radius < RMIN:
            radius = 2e-4
        # safety: update best if cur drifted better due to numerics
        # periodic time check
        if it % 10 == 0:
            if time.time() > DEADLINE - 2.0:
                break

    X_best = _norm(np.asarray(X_best, dtype=float))
    # final safety: replace non-finite
    if not np.all(np.isfinite(X_best)):
        X_best = _norm(X0.copy())
    return X_best
