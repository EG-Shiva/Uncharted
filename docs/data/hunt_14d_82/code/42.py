import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lex_stats(X, tau_lse=1e-3):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    # sort descending
    # full sort, n=3321 cheap
    s = np.sort(c)[::-1]
    m = float(s[0])
    mean100 = float(np.mean(s[:100]))
    top500 = s[:500]
    # stable LSE: m + tau*log(sum exp((c-m)/tau))
    # top500 only
    e = np.exp((top500 - m) / tau_lse)
    lse = float(m + tau_lse * np.log(np.sum(e) + 1e-300))
    return m, mean100, lse, c

def _lex_less(a, b, eps=1e-12):
    # a,b tuples (max, mean100, lse)
    if a[0] < b[0] - eps:
        return True
    if a[0] > b[0] + eps:
        return False
    if a[1] < b[1] - eps:
        return True
    if a[1] > b[1] + eps:
        return False
    if a[2] < b[2] - 1e-11:
        return True
    return False

def _active_set(c_all, m_cur, lo=1200, hi=2000, window=2e-3):
    slack = m_cur - c_all
    # all within window
    mask_w = slack < window
    cnt_w = int(np.count_nonzero(mask_w))
    if cnt_w >= lo and cnt_w <= hi:
        idx = np.where(mask_w)[0]
        # sort by slack (tightest first) for determinism
        o = np.argsort(slack[idx])
        return idx[o]
    # need top-k
    if cnt_w < lo:
        k = lo
    else:
        k = hi
    # top k smallest slack = largest c
    part = np.argpartition(-c_all, k - 1)[:k]
    # sort them descending c for determinism
    o = np.argsort(-c_all[part])
    return part[o]

def _lp_minimax(X, act_idx, c_all, radius):
    n, d = X.shape
    p = len(act_idx)
    I = None
    J = None
    # need I,J mapping from flat upper index -> need iu arrays
    # caller passes I,J directly; this helper expects I,J
    raise NotImplementedError

def solve():
    t0 = time.time()
    OPT_END = t0 + 132.0
    HARD_END = t0 + 138.0
    rng = np.random.default_rng(0)
    n, d = 82, 14
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
        X0 = _norm(rng.standard_normal((n, d)))
    else:
        # pick best by lex
        best_t = None
        X0 = cands[0]
        for c in cands:
            m, m100, lse, _ = _lex_stats(c)
            t = (m, m100, lse)
            if best_t is None or _lex_less(t, best_t):
                best_t = t
                X0 = c
    X0 = _norm(X0.copy())
    m0, m100_0, lse0, _ = _lex_stats(X0)
    best_lex = (m0, m100_0, lse0)
    best_max = m0
    X_best = X0.copy()
    X_cur = X0.copy()
    cur_lex = best_lex
    cur_max, cur_m100, cur_lse = best_lex

    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]
    J0 = iu0[1]
    nv = n * d + 1
    nv2 = n * d

    radius = 5e-4
    radius_min = 2e-7
    radius_max = 1e-3

    # preallocate row templates
    it = 0
    while time.time() < OPT_END:
        it += 1
        if time.time() > HARD_END - 2.0:
            break
        if radius < radius_min:
            # reset with tiny tangent jitter to escape stall
            radius = 5e-4
            # small deterministic jitter along tangent
            jr = 1e-5
            pert = rng.standard_normal((n, d)) * jr
            pert = pert - (np.sum(pert * X_cur, axis=1, keepdims=True)) * X_cur
            X_cur = _norm(X_cur + pert)
            m, m100, lse, _ = _lex_stats(X_cur)
            cur_lex = (m, m100, lse)
            # do not update best unless improved
            if m < best_max - 1e-12:
                best_max = m
                best_lex = cur_lex
                X_best = X_cur.copy()
            continue

        # current Gram and active set for minimax
        G = X_cur @ X_cur.T
        c_all = G[I0, J0]
        m_cur = float(np.max(c_all))

        act = _active_set(c_all, m_cur, lo=1200, hi=2000, window=2e-3)
        p = len(act)
        Ia = I0[act]
        Ja = J0[act]
        ca = c_all[act]

        # ---- Step A: minimax LP min t ----
        # build sparse matrices
        # rows: repeat p
        try:
            if time.time() > OPT_END:
                break
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cd = np.arange(d)
            cols_I = (Ia[:, None] * d + cd)
            cols_J = (Ja[:, None] * d + cd)
            cols = np.concatenate([cols_I, cols_J, np.full((p, 1), nv - 1)], axis=1).ravel()
            vals = np.concatenate([X_cur[Ja], X_cur[Ia], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X_cur.ravel(), (np.repeat(np.arange(n), d), np.arange(nv2))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -ca
            bounds = [(-radius, radius)] * nv2 + [(None, None)]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=bounds, method="highs", options={"time_limit": 1.0})
            except Exception:
                res = None
            if res is not None and res.status == 0:
                delta = res.x[:-1].reshape(n, d)
                # tangent-project numerically (remove normal component)
                # keep as is, renorm will handle
                Y = _norm(X_cur + delta)
                m_n, m100_n, lse_n, _ = _lex_stats(Y)
                new_lex = (m_n, m100_n, lse_n)
                if _lex_less(new_lex, cur_lex):
                    X_cur = Y
                    cur_lex = new_lex
                    m_cur = m_n
                    if _lex_less(new_lex, best_lex) and m_n <= best_max + 1e-12:
                        # update best if max improved or tail improved at same max
                        if m_n < best_max - 1e-12 or _lex_less(new_lex, best_lex):
                            # only replace X_best if max not worse; for scoring max matters
                            if m_n < best_max + 1e-12:
                                # if max strictly better, definitely update
                                if m_n < best_max - 1e-12:
                                    X_best = Y.copy()
                                    best_max = m_n
                                    best_lex = new_lex
                                else:
                                    # equal max but better tail: keep for continuation but also store?
                                    # keep X_best as tail-better for future? scoring only max,
                                    # but tail-better is better basin. Update X_best too if max equal.
                                    X_best = Y.copy()
                                    best_lex = new_lex
                    radius = min(radius * 1.25, radius_max)
                else:
                    radius *= 0.5
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5

        if time.time() > OPT_END or time.time() > HARD_END - 2.0:
            break

        # ---- Step B: tail-sum LP min weighted sum s.t. <= best_max ----
        # recompute current stats (may have changed after step A)
        try:
            G2 = X_cur @ X_cur.T
            c_all2 = G2[I0, J0]
            m_cur2 = float(np.max(c_all2))
            act2 = _active_set(c_all2, m_cur2, lo=1200, hi=2000, window=2e-3)
            p2 = len(act2)
            Ia2 = I0[act2]
            Ja2 = J0[act2]
            ca2 = c_all2[act2]
            # top-500 for objective
            k500 = 500
            if len(c_all2) > k500:
                part500 = np.argpartition(-c_all2, k500 - 1)[:k500]
            else:
                part500 = np.arange(len(c_all2))
            c500 = c_all2[part500]
            # softmax weights tau=5e-4
            tau_w = 5e-4
            w = np.exp(np.clip((c500 - m_cur2) / tau_w, -30.0, 0.0))
            # normalize to sum 1 for scaling stability
            sw = np.sum(w)
            if sw > 0:
                w = w / sw
            I500 = I0[part500]
            J500 = J0[part500]
            # objective vector
            c_obj = np.zeros(nv2)
            # accumulate: for each k, c_obj[I*d:(I+1)*d] += w*X[J]
            # vectorized via bincount-like loop (500*14 cheap)
            for kk in range(len(part500)):
                wk = float(w[kk])
                if wk < 1e-14:
                    continue
                i = int(I500[kk]); j = int(J500[kk])
                c_obj[i * d:(i + 1) * d] += wk * X_cur[j]
                c_obj[j * d:(j + 1) * d] += wk * X_cur[i]
            # scale to avoid tiny numbers: multiply by 1000?
            # keep as is; linprog handles
            rows2 = np.repeat(np.arange(p2), 2 * d)
            cols_I2 = (Ia2[:, None] * d + np.arange(d))
            cols_J2 = (Ja2[:, None] * d + np.arange(d))
            cols2 = np.concatenate([cols_I2, cols_J2], axis=1).ravel()
            vals2 = np.concatenate([X_cur[Ja2], X_cur[Ia2]], axis=1).ravel()
            A_ub2 = coo_matrix((vals2, (rows2, cols2)), shape=(p2, nv2)).tocsr()
            A_eq2 = coo_matrix((X_cur.ravel(), (np.repeat(np.arange(n), d), np.arange(nv2))), shape=(n, nv2)).tocsr()
            # cap: no pair exceeding current best (use best_max, but allow cur if cur worse?)
            cap_val = float(min(best_max, m_cur2))
            # small tolerance to keep feasible
            b_ub2 = cap_val - ca2 + 1e-12
            bounds2 = [(-radius, radius)] * nv2
            try:
                res2 = linprog(c_obj, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=np.zeros(n),
                               bounds=bounds2, method="highs", options={"time_limit": 1.0})
            except Exception:
                res2 = None
            if res2 is not None and res2.status == 0:
                delta2 = res2.x.reshape(n, d)
                Y2 = _norm(X_cur + delta2)
                m_n2, m100_n2, lse_n2, _ = _lex_stats(Y2)
                new_lex2 = (m_n2, m100_n2, lse_n2)
                if _lex_less(new_lex2, cur_lex):
                    X_cur = Y2
                    cur_lex = new_lex2
                    if m_n2 < best_max - 1e-12:
                        X_best = Y2.copy()
                        best_max = m_n2
                        best_lex = new_lex2
                    elif abs(m_n2 - best_max) <= 1e-12 and _lex_less(new_lex2, best_lex):
                        X_best = Y2.copy()
                        best_lex = new_lex2
                    radius = min(radius * 1.15, radius_max)
                else:
                    radius *= 0.5
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5

    X_best = _norm(X_best)
    return X_best
