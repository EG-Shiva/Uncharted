import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 133.0
    n, d = 82, 14
    nd = n * d
    base = np.arange(d)
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]

    def metrics_of(X):
        G = X @ X.T
        c = G[I0, J0]
        m = float(np.max(c))
        s = np.sort(c)[::-1]
        mean100 = float(np.mean(s[:100]))
        top500 = s[:500]
        p = 2000.0
        # stable log-sum-exp / p
        e = np.exp(np.clip(p * (top500 - m), -50.0, 0.0))
        lse = m + float(np.log(np.sum(e) + 1e-300) / p)
        return m, mean100, lse, c, s

    def lex_better(new_t, old_t):
        nm, nmean, nlse = new_t
        om, omean, olse = old_t
        if nm < om - 1e-12:
            return True
        if nm > om + 1e-12:
            return False
        if nmean < omean - 5e-10:
            return True
        if nmean > omean + 5e-10:
            # tie on mean, check lse
            if nlse < olse - 5e-10:
                return True
            return False
        # mean essentially tied, lse decides but still better if strictly smaller
        if nlse < olse - 1e-11:
            return True
        return False

    # load start: pick best of best.npy / record.npy
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
        cands.append(_norm(rng.standard_normal((n, d))))
    # choose lowest max
    vals = [_max_cos(a) for a in cands]
    Xcur = cands[int(np.argmin(vals))].copy()
    Xcur = _norm(Xcur)
    m_c, mean_c, lse_c, _, _ = metrics_of(Xcur)
    cur_t = (m_c, mean_c, lse_c)
    Xbest = Xcur.copy()
    best_t = cur_t
    best_m = m_c

    radius = 5e-4
    it = 0
    # pre-build bound lists template sizes (rebuild when radius changes is cheap)
    while time.time() < DEADLINE and it < 300:
        it += 1
        if time.time() >= DEADLINE:
            break
        # current metrics (recompute cheap: 3321 pairs)
        Gc = Xcur @ Xcur.T
        c_all = Gc[I0, J0]
        m = float(np.max(c_all))
        # order descending
        order = np.argsort(-c_all)
        # wide active set: top 1500 + extension within 2e-3 up to 2000
        K0 = 1500
        Kmax = 2000
        K = min(K0, len(order))
        # extend if shell dense
        if len(order) > K0:
            thr_wide = m - 2e-3
            # count how many within thr up to Kmax
            # s sorted already via order
            cnt_within = int(np.count_nonzero(c_all > thr_wide))
            cnt_within = min(cnt_within, Kmax)
            K = max(K0, cnt_within)
            K = min(K, Kmax, len(order))
        sel_idx = order[:K]
        I = I0[sel_idx]; J = J0[sel_idx]
        c_sel = c_all[sel_idx]
        p_ = len(I)

        use_lp1 = (it % 2 == 1)  # alternate: odd -> minimax t, even -> tail sum
        step_ok = False
        Ynew = None
        new_t = None
        try:
            if use_lp1:
                # LP1: min t s.t. Xj.di+Xi.dj - t <= -c, Xi.di=0, |d|<=r
                nv = nd + 1
                rows = np.repeat(np.arange(p_), 2 * d + 1)
                cols_I = (I[:, None] * d + base).ravel()
                cols_J = (J[:, None] * d + base).ravel()
                cols_T = np.full(p_, nd)
                cols = np.concatenate([cols_I, cols_J, cols_T]) if False else None
                # build cols in row-major order: for each row [I-block, J-block, t]
                cols = np.empty(p_ * (2 * d + 1), dtype=int)
                cols[0::(2*d+1)] = I * d + 0  # placeholder, do direct below is faster vectorized
                # vectorized construction:
                cI = (I[:, None] * d + base)
                cJ = (J[:, None] * d + base)
                colsM = np.concatenate([cI, cJ, np.full((p_, 1), nd)], axis=1)
                cols = colsM.ravel()
                vI = Xcur[J]
                vJ = Xcur[I]
                valsM = np.concatenate([vI, vJ, -np.ones((p_, 1))], axis=1)
                vals = valsM.ravel()
                A_ub = coo_matrix((vals, (np.repeat(np.arange(p_), 2*d+1), cols)), shape=(p_, nv)).tocsr()
                A_eq = coo_matrix((Xcur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                b_ub = -c_sel
                bounds = [(-radius, radius)] * nd + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs", options={"time_limit": 0.9})
                if res.status != 0:
                    raise RuntimeError("lp1 fail")
                delta = res.x[:-1].reshape(n, d)
                Y = _norm(Xcur + delta)
            else:
                # LP2: min w'(Xj.di+Xi.dj) s.t. Xj.di+Xi.dj <= m - c, tangent, |d|<=r
                # weights on top-500
                K500 = min(500, len(order))
                top_idx = order[:K500]
                c_top = c_all[top_idx]
                eta = 5e-4
                w = np.exp(np.clip((c_top - m) / eta, -30.0, 0.0))
                # objective
                obj = np.zeros(nd)
                It = I0[top_idx]; Jt = J0[top_idx]
                # accumulate: obj[I*d:(I+1)*d] += w*X[J]
                # loop 500: fine
                for k in range(K500):
                    wk = float(w[k])
                    i = int(It[k]); j = int(Jt[k])
                    obj[i*d:(i+1)*d] += wk * Xcur[j]
                    obj[j*d:(j+1)*d] += wk * Xcur[i]
                # constraints from wide set with ceiling m
                nv = nd
                rows = np.repeat(np.arange(p_), 2 * d)
                cI = (I[:, None] * d + base)
                cJ = (J[:, None] * d + base)
                colsM = np.concatenate([cI, cJ], axis=1)
                cols = colsM.ravel()
                valsM = np.concatenate([Xcur[J], Xcur[I]], axis=1)
                vals = valsM.ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
                A_eq = coo_matrix((Xcur.ravel(), (np.repeat(np.arange(n), d), np.arange(nd))), shape=(n, nv)).tocsr()
                b_ub = m - c_sel  # >=0, zero for max pairs
                # tiny relaxation to keep feasibility under roundoff
                b_ub = b_ub + 1e-12
                bounds = [(-radius, radius)] * nd
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs", options={"time_limit": 0.9})
                if res.status != 0:
                    raise RuntimeError("lp2 fail")
                delta = res.x.reshape(n, d)
                Y = _norm(Xcur + delta)
            # evaluate true nonlinear metrics
            m2, mean2, lse2, _, _ = metrics_of(Y)
            Ynew = Y
            new_t = (m2, mean2, lse2)
        except Exception:
            # solver failure -> shrink
            radius *= 0.5
            if radius < 8e-5:
                radius = 5e-4
            continue
        # time check after solve
        if time.time() >= DEADLINE:
            break
        if Ynew is None or new_t is None or not np.all(np.isfinite(Ynew)):
            radius *= 0.5
            if radius < 8e-5:
                radius = 5e-4
            continue
        # accept iff lexicographically better than current
        if lex_better(new_t, cur_t):
            Xcur = Ynew
            cur_t = new_t
            radius = min(radius * 1.3, 1e-3)
            # update global best (max primary, then mean)
            bm, bmean, blse = best_t
            nm, nmean, nlse = new_t
            if (nm < bm - 1e-12) or (abs(nm - bm) <= 1e-12 and (nmean < bmean - 1e-12 or (abs(nmean-bmean) <= 1e-12 and nlse < blse - 1e-12))):
                Xbest = Xcur.copy()
                best_t = cur_t
                best_m = bm if False else new_t[0]
        else:
            radius *= 0.5
            if radius < 5e-5:
                # reset to avoid stall, keep current
                radius = 4e-4
        # early stop if radius collapsed repeatedly? we reset above so continue
        if radius < 1e-12:
            break

    Xbest = _norm(np.asarray(Xbest, dtype=float))
    # ensure finite
    if not np.all(np.isfinite(Xbest)):
        Xbest = _norm(Xcur)
    return Xbest
