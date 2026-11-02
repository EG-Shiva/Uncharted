import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_and_all(X, I0, J0):
    G = X @ X.T
    c = G[I0, J0]
    m = float(c.max())
    return m, c, G

def _contact_stats(c, m, I0, J0, n, thr=1e-7):
    sel = c > (m - thr)
    cnt = int(np.count_nonzero(sel))
    if cnt == 0:
        return 0, 0
    deg = np.zeros(n, dtype=int)
    Is = I0[sel]; Js = J0[sel]
    np.add.at(deg, Is, 1)
    np.add.at(deg, Js, 1)
    return cnt, int(deg.min())

def solve():
    t0 = time.time()
    OPT_END = t0 + 120.0
    HARD_END = t0 + 135.0
    n, d = 82, 14
    nd = n * d
    iu = np.triu_indices(n, 1)
    I0 = iu[0]; J0 = iu[1]
    ar_d = np.arange(d)

    # load start
    X0 = None
    best_load = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                v = float(G.max())
                if v < best_load:
                    best_load = v
                    X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        rng = np.random.default_rng(0)
        X0 = _norm(rng.standard_normal((n, d)))

    Xcur = X0.copy()
    m_cur, c_cur, _ = _max_and_all(Xcur, I0, J0)
    Xbest = Xcur.copy()
    best_max = m_cur
    cnt7_cur, mindeg_cur = _contact_stats(c_cur, m_cur, I0, J0, n, 1e-7)

    # stages: (epsilon, K forced)
    stages = [
        (1e-6, 200),
        (3e-7, 400),
        (1e-7, 400),
        (3e-8, 800),
        (1e-8, 800),
    ]

    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(nd)

    # trust region
    radius = 1e-3
    RMAX = 1e-3
    RMIN = 1e-7

    PEN_W = 0.05
    POOL_EXTRA = 500

    for si, (eps, K) in enumerate(stages):
        if time.time() > OPT_END - 2.0:
            break
        # per-stage iteration budget
        # give later stages a bit more
        max_it_stage = 40
        stall = 0
        for it_in in range(max_it_stage):
            if time.time() > OPT_END - 1.5 or time.time() > HARD_END - 3.0:
                break
            if radius < RMIN:
                radius = 2e-4
                # if still stalling, break stage
                if stall > 5:
                    break
            m_cur, c_cur, _ = _max_and_all(Xcur, I0, J0)
            # refresh cur stats (cheap)
            cnt7_cur, mindeg_cur = _contact_stats(c_cur, m_cur, I0, J0, n, 1e-7)

            # top-K forced
            Kcap = int(min(K, len(c_cur)))
            # argpartition for top (K + POOL_EXTRA)
            Ppool = int(min(len(c_cur), K + POOL_EXTRA))
            # get pool indices (top Ppool)
            part_pool = np.argpartition(-c_cur, Ppool - 1)[:Ppool]
            # top K among pool
            pool_vals = c_cur[part_pool]
            # find threshold for top K
            if Kcap < Ppool:
                kth = np.argpartition(-pool_vals, Kcap - 1)[:Kcap]
                topK_idx = part_pool[kth]
            else:
                topK_idx = part_pool
            # epsilon-active
            sel_eps = np.where(c_cur > (m_cur - eps))[0]
            # union, cap to 2000 tightest if too large
            # use boolean mask for pool+eps? full size 3321 so fine
            mask = np.zeros(len(c_cur), dtype=bool)
            mask[topK_idx] = True
            # add eps set (may be large for eps=1e-6? ~500 so fine)
            # cap eps set if huge
            if len(sel_eps) > 2000:
                # keep tightest 2000
                v = c_cur[sel_eps]
                pp = np.argpartition(-v, 1999)[:2000]
                sel_eps = sel_eps[pp]
            mask[sel_eps] = True
            act_idx = np.where(mask)[0]
            # cap active to 2000 tightest if still huge
            if len(act_idx) > 2000:
                v = c_cur[act_idx]
                pp = np.argpartition(-v, 1999)[:2000]
                act_idx = act_idx[pp]
                # rebuild mask? not needed
                # need to keep mask consistent for penalty; recompute mask
                mask[:] = False
                mask[act_idx] = True
            Ia = I0[act_idx]; Ja = J0[act_idx]
            ca = c_cur[act_idx]
            p = len(act_idx)

            # penalty set = pool \ active (inactive near shell)
            # pool mask minus active
            pool_mask = np.zeros(len(c_cur), dtype=bool)
            pool_mask[part_pool] = True
            pen_mask = pool_mask & (~mask)
            pen_idx = np.where(pen_mask)[0]
            # cap penalty to 500 (already <=500)
            Ip = I0[pen_idx]; Jp = J0[pen_idx]

            # build objective penalty gradient
            obj = np.zeros(nd + 1)
            obj[-1] = 1.0
            if len(pen_idx) > 0:
                # gradient: sum over penalty pairs of (Xj at i, Xi at j) / N * PEN_W
                g = np.zeros(nd)
                Xin = Xcur
                scale = PEN_W / float(len(pen_idx))
                # vectorized accumulation via bincount-like loop (500*14 small, loop ok)
                for k in range(len(pen_idx)):
                    i = int(Ip[k]); j = int(Jp[k])
                    g[i*d:(i+1)*d] += scale * Xin[j]
                    g[j*d:(j+1)*d] += scale * Xin[i]
                # normalize scale to avoid dominating t: keep norm ~ O(1)
                gn = float(np.linalg.norm(g))
                if gn > 1.0:
                    g = g / gn
                obj[:nd] = g

            # build sparse LP: minimize obj s.t. active linear - t <= -ca, tangent eq
            try:
                # A_ub p x (nd+1)
                rows = np.repeat(np.arange(p), 2*d+1)
                cI = Ia[:, None]*d + ar_d
                cJ = Ja[:, None]*d + ar_d
                Cmat = np.concatenate([cI, cJ, np.full((p,1), nd)], axis=1)
                cols = Cmat.ravel()
                Vmat = np.concatenate([Xcur[Ja], Xcur[Ia], -np.ones((p,1))], axis=1)
                vals = Vmat.ravel()
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nd+1)).tocsr()
                A_eq = coo_matrix((Xcur.ravel(), (eq_rows, eq_cols)), shape=(n, nd+1)).tocsr()
                b_ub = -ca
                bounds = [(-radius, radius)]*nd + [(None, None)]
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs",
                              options={"time_limit": 1.0})
            except Exception:
                radius *= 0.5
                stall += 1
                continue
            if res.status != 0:
                radius *= 0.5
                stall += 1
                continue
            Delta = res.x[:-1].reshape(n, d)
            # trust-region sanity: if step tiny, shrink?
            Y = _norm(Xcur + Delta)
            m_new, c_new, _ = _max_and_all(Y, I0, J0)
            cnt7_new, mindeg_new = _contact_stats(c_new, m_new, I0, J0, n, 1e-7)

            accepted = False
            if m_new < m_cur - 1e-12:
                accepted = True
            elif m_new <= m_cur + 1e-12:
                # sideways equalization: need more pairs within 1e-7 and no min-degree drop
                if cnt7_new > cnt7_cur and mindeg_new >= mindeg_cur:
                    accepted = True
                elif mindeg_new > mindeg_cur and cnt7_new >= cnt7_cur:
                    accepted = True
            if accepted:
                Xcur = Y
                m_cur = m_new
                c_cur = c_new
                cnt7_cur, mindeg_cur = cnt7_new, mindeg_new
                radius = min(radius*1.25, RMAX)
                stall = 0
                if m_new < best_max - 1e-13:
                    best_max = m_new
                    Xbest = Y.copy()
            else:
                radius *= 0.5
                stall += 1
            # early exit stage if time low
            if stall >= 12:
                # try reset radius once, else next stage
                if radius < 3e-4:
                    radius = 5e-4
                    stall = 0
                    break
        # end inner
        # decay radius for next tighter stage
        radius = min(max(radius, 3e-4), RMAX)
        # if best improved, keep; else continue tightening
        if time.time() > OPT_END - 1.5:
            break

    Xbest = _norm(np.asarray(Xbest, dtype=float))
    if not np.all(np.isfinite(Xbest)):
        Xbest = _norm(X0.copy())
    return Xbest
