import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _stats(X, I0, J0):
    G = X @ X.T
    c = G[I0, J0]
    m = float(np.max(c))
    slack = m - c
    cnt9 = int(np.count_nonzero(slack < 1e-9))
    cnt7 = int(np.count_nonzero(slack < 1e-7))
    cnt6 = int(np.count_nonzero(slack < 1e-6))
    cnt8 = int(np.count_nonzero(slack < 1e-8))
    # min contact degree at 1e-7
    sel = slack < 1e-7
    deg = np.zeros(X.shape[0], dtype=int)
    if np.any(sel):
        Is = I0[sel]; Js = J0[sel]
        np.add.at(deg, Is, 1)
        np.add.at(deg, Js, 1)
    mindeg = int(np.min(deg))
    return m, c, slack, cnt9, cnt7, cnt6, cnt8, mindeg, deg

def solve():
    t0 = time.time()
    OPT_END = t0 + 120.0
    n, d = 82, 14
    nd = n * d
    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        Xcur = _norm(rng0.standard_normal((n, d)))
    else:
        def mx(c):
            G = c @ c.T
            np.fill_diagonal(G, -2.0)
            return float(np.max(G))
        cands.sort(key=mx)
        Xcur = cands[0].copy()
    Xcur = _norm(Xcur)
    I0, J0 = np.triu_indices(n, 1)
    Npairs = len(I0)

    m_cur, _, _, cnt9_cur, cnt7_cur, cnt6_cur, cnt8_cur, mindeg_cur, _ = _stats(Xcur, I0, J0)
    Xbest = Xcur.copy()
    best_max = m_cur
    best_cnt7 = cnt7_cur
    best_mindeg = mindeg_cur

    eps_schedule = [1e-6, 3e-7, 1e-7, 3e-8, 1e-8]
    K_schedule = [200, 400, 400, 800, 800]
    radius = 1e-3
    radius_max = 1e-3
    radius_min = 5e-8

    # equality tangent matrix sparsity pattern (fixed rows/cols, values change)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(nd)

    stage_iters = [60, 60, 60, 60, 60]

    for stage, (eps, K) in enumerate(zip(eps_schedule, K_schedule)):
        if time.time() >= OPT_END - 1.0:
            break
        iters = stage_iters[stage]
        for it in range(iters):
            if time.time() >= OPT_END - 1.0:
                break
            if radius < radius_min:
                radius = max(radius_min, min(2e-4, eps * 10.0 if eps > 0 else 2e-4))
                # if still tiny, move to next stage
                if stage < len(eps_schedule) - 1 and it > 10:
                    break
            # current stats
            G = Xcur @ Xcur.T
            c_all = G[I0, J0]
            m = float(np.max(c_all))
            slack = m - c_all
            # active: slack<eps union top-K, capped at 1000
            mask_eps = slack < eps
            # top K
            if K >= Npairs:
                topK = np.arange(Npairs)
            else:
                part = np.argpartition(-c_all, K - 1)[:K]
                topK = part
            mask_active = np.zeros(Npairs, dtype=bool)
            mask_active[topK] = True
            mask_active = mask_active | mask_eps
            # cap at 1000 tightest
            cnt_act = int(np.count_nonzero(mask_active))
            if cnt_act > 1000:
                # keep 1000 largest c among active
                idx = np.where(mask_active)[0]
                vals = c_all[idx]
                keep = np.argpartition(-vals, 1000 - 1)[:1000]
                nm = np.zeros(Npairs, dtype=bool)
                nm[idx[keep]] = True
                mask_active = nm
                cnt_act = 1000
            act_idx = np.where(mask_active)[0]
            # sort active by c descending for determinism
            o = np.argsort(-c_all[act_idx])
            act_idx = act_idx[o]
            p_a = len(act_idx)
            Ia = I0[act_idx]; Ja = J0[act_idx]; ca = c_all[act_idx]
            # buffer: next tightest pairs not active, up to 600, for inactive-violation cap
            # take top 1600 overall
            Kbuf_total = min(Npairs, max(p_a + 600, 1500))
            partT = np.argpartition(-c_all, Kbuf_total - 1)[:Kbuf_total]
            # buffer = those in top set but not active
            is_act = np.zeros(Npairs, dtype=bool)
            is_act[act_idx] = True
            buf_cand = partT[~is_act[partT]]
            # keep tightest 600 of buffer
            if len(buf_cand) > 600:
                bv = c_all[buf_cand]
                kp = np.argpartition(-bv, 600 - 1)[:600]
                buf_cand = buf_cand[kp]
            # sort buffer descending
            if len(buf_cand) > 0:
                ob = np.argsort(-c_all[buf_cand])
                buf_cand = buf_cand[ob]
            p_b = len(buf_cand)
            Ib = I0[buf_cand] if p_b > 0 else np.zeros(0, dtype=int)
            Jb = J0[buf_cand] if p_b > 0 else np.zeros(0, dtype=int)
            cb = c_all[buf_cand] if p_b > 0 else np.zeros(0)

            # build LP: vars dx(nd)+t(1), min t
            # active rows: Xj*dxi + Xi*dxj - t <= -ca
            # buffer rows: Xj*dxi + Xi*dxj + 0*t <= cap - cb
            cap = float(min(best_max, m)) + 1e-12
            p_tot = p_a + p_b
            try:
                # active block
                rows_a = np.repeat(np.arange(p_a), 2 * d + 1) if p_a > 0 else np.zeros(0, dtype=int)
                Ca = np.empty((p_a, 2 * d + 1), dtype=int) if p_a > 0 else np.empty((0, 2 * d + 1), dtype=int)
                Ma = np.empty((p_a, 2 * d + 1)) if p_a > 0 else np.empty((0, 2 * d + 1))
                if p_a > 0:
                    cd = np.arange(d)
                    Ca[:, :d] = Ia[:, None] * d + cd
                    Ca[:, d:2 * d] = Ja[:, None] * d + cd
                    Ca[:, 2 * d] = nd
                    Ma[:, :d] = Xcur[Ja]
                    Ma[:, d:2 * d] = Xcur[Ia]
                    Ma[:, 2 * d] = -1.0
                if p_b > 0:
                    rows_b = p_a + np.repeat(np.arange(p_b), 2 * d)
                    Cb = np.empty((p_b, 2 * d), dtype=int)
                    Mb = np.empty((p_b, 2 * d))
                    cd2 = np.arange(d)
                    Cb[:, :d] = Ib[:, None] * d + cd2
                    Cb[:, d:] = Jb[:, None] * d + cd2
                    Mb[:, :d] = Xcur[Jb]
                    Mb[:, d:] = Xcur[Ib]
                    rows_all = np.concatenate([rows_a, rows_b])
                    cols_all = np.concatenate([Ca.ravel(), Cb.ravel()])
                    vals_all = np.concatenate([Ma.ravel(), Mb.ravel()])
                    b_ub = np.concatenate([-ca, cap - cb])
                else:
                    rows_all = rows_a
                    cols_all = Ca.ravel()
                    vals_all = Ma.ravel()
                    b_ub = -ca
                A_ub = coo_matrix((vals_all, (rows_all, cols_all)), shape=(p_tot, nd + 1)).tocsr()
                A_eq = coo_matrix((np.tile(Xcur.ravel(), 1), (eq_rows, eq_cols)), shape=(n, nd + 1)).tocsr()
                obj = np.zeros(nd + 1)
                obj[-1] = 1.0
                bounds = [(-radius, radius)] * nd + [(None, None)]
                tl = 1.5
                # leave time guard
                if OPT_END - time.time() < 2.0:
                    break
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs", options={"time_limit": float(tl)})
            except Exception:
                radius *= 0.5
                continue
            if res is None or res.status != 0:
                radius *= 0.5
                continue
            delta = res.x[:-1].reshape(n, d)
            # quick trust check: if step tiny, shrink
            stp = float(np.max(np.abs(delta)))
            if stp < radius * 1e-3:
                radius *= 0.5
                continue
            Y = _norm(Xcur + delta)
            m_n, _, _, cnt9_n, cnt7_n, cnt6_n, cnt8_n, mindeg_n, _ = _stats(Y, I0, J0)
            # acceptance: exact max decreases, or cnt7/mindeg improve without increasing max
            improved_max = m_n < m - 1e-12
            improved_best = m_n < best_max - 1e-12
            # equalization acceptance: no increase vs current, and contact metrics improve
            no_worse = m_n <= m + 1e-12
            eq_better = False
            if no_worse and not improved_max:
                if (cnt7_n > cnt7_cur or mindeg_n > mindeg_cur) and cnt7_n >= cnt7_cur and mindeg_n >= mindeg_cur:
                    eq_better = True
                # also allow if cnt within 1e-7 grows even if mindeg ties, provided max not worse
                # require both non-decreasing (above) so ok
            if improved_max or eq_better:
                Xcur = Y
                m_cur = m_n
                cnt7_cur = cnt7_n
                cnt9_cur = cnt9_n
                mindeg_cur = mindeg_n
                if m_n < best_max - 1e-12 or (abs(m_n - best_max) <= 1e-12 and (cnt7_n > best_cnt7 or (cnt7_n == best_cnt7 and mindeg_n > best_mindeg))):
                    # update best: strictly better max, or equal max with better balance
                    if m_n < best_max + 1e-12:
                        Xbest = Y.copy()
                        if m_n < best_max - 1e-12:
                            best_max = m_n
                            best_cnt7 = cnt7_n
                            best_mindeg = mindeg_n
                        else:
                            best_cnt7 = max(best_cnt7, cnt7_n)
                            best_mindeg = max(best_mindeg, mindeg_n)
                # adapt trust: grow slightly
                radius = min(radius * 1.2, radius_max)
            else:
                # check if it improves global best even if worse than current? (should not happen since current may be worse than best)
                if m_n < best_max - 1e-12:
                    Xbest = Y.copy()
                    best_max = m_n
                    best_cnt7 = cnt7_n
                    best_mindeg = mindeg_n
                    Xcur = Y
                    m_cur = m_n
                    cnt7_cur = cnt7_n
                    mindeg_cur = mindeg_n
                    radius = min(radius * 1.2, radius_max)
                else:
                    radius *= 0.5
            # early exit if radius collapsed -> next epsilon stage
            if radius < radius_min + 1e-12:
                break
    Xbest = _norm(Xbest)
    return Xbest
