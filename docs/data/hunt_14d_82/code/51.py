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

def _metrics_full(X):
    n = X.shape[0]
    iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    m = float(np.max(c))
    if len(c) >= 500:
        p500 = np.argpartition(-c, 500 - 1)[:500]
        t500 = c[p500]
        p100 = np.argpartition(-t500, 100 - 1)[:100]
        mean100 = float(np.mean(t500[p100]))
        mean500 = float(np.mean(t500))
    elif len(c) >= 100:
        p100 = np.argpartition(-c, 100 - 1)[:100]
        mean100 = float(np.mean(c[p100]))
        mean500 = float(np.mean(c))
    else:
        mean100 = float(np.mean(c))
        mean500 = float(np.mean(c))
    return m, mean100, mean500, c

def _lex_less(a, b, eps=1e-12):
    if a[0] < b[0] - eps:
        return True
    if abs(a[0] - b[0]) <= eps:
        if a[1] < b[1] - eps:
            return True
        if abs(a[1] - b[1]) <= eps:
            if a[2] < b[2] - 1e-12:
                return True
    return False

def _fingerprint(X, tol=1e-3):
    n = X.shape[0]
    iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    m = float(np.max(c))
    sel = np.where(c > m - tol)[0]
    I = iu0[0][sel]; J = iu0[1][sel]
    s = set((int(i) * n + int(j)) for i, j in zip(I, J))
    return frozenset(s)

def _weighted_degrees(X, m, c, iu0, tol=1e-3, tau=3e-4):
    n = X.shape[0]
    sel = c > m - tol
    I = iu0[0][sel]; J = iu0[1][sel]
    cv = c[sel]
    w = np.exp(np.clip((cv - m) / tau, -30.0, 0.0))
    deg = np.zeros(n)
    cnt = np.zeros(n, dtype=int)
    for ii, jj, ww in zip(I, J, w):
        deg[int(ii)] += float(ww)
        deg[int(jj)] += float(ww)
        cnt[int(ii)] += 1
        cnt[int(jj)] += 1
    return deg, cnt

def _lp_polish_short(X, deadline, radius_init=3e-4, cap=2000, max_it=6):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        cur = _metrics_full(X)
    except Exception:
        return X
    best_tup = (cur[0], cur[1], cur[2])
    best_max = cur[0]
    Xbest = X.copy()
    radius = float(radius_init)
    iu0 = np.triu_indices(n, 1)
    it = 0
    while it < max_it and time.time() < deadline - 0.4 and radius > 1e-12:
        it += 1
        G = X @ X.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - 1e-4
        # ensure at least some pairs
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c_all[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            C = np.empty((p, 2 * d + 1), dtype=int)
            cd = np.arange(d)
            C[:, :d] = I[:, None] * d + cd
            C[:, d:2 * d] = J[:, None] * d + cd
            C[:, 2 * d] = n * d
            cols = C.ravel()
            M = np.empty((p, 2 * d + 1))
            M[:, :d] = X[J]
            M[:, d:2 * d] = X[I]
            M[:, 2 * d] = -1.0
            vals = M.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            Y = X + res.x[:-1].reshape(n, d)
            Y = _norm(Y)
            m2, me100, me500, _ = _metrics_full(Y)
        except Exception:
            radius *= 0.5
            continue
        if m2 < best_max - 1e-12 or (abs(m2 - best_max) <= 1e-12 and (me100 < best_tup[1] - 1e-12)):
            X = Y
            Xbest = Y.copy()
            best_max = m2
            best_tup = (m2, me100, me500)
            radius = min(radius * 1.5, 2e-3)
        else:
            radius *= 0.5
    return Xbest

def _adam_refine(Xm, Xf, deadline, steps=450, a0=50.0, a1=500.0, lr0=0.008, lr1=0.0006):
    k, d = Xm.shape
    nf = Xf.shape[0]
    m1 = np.zeros_like(Xm)
    m2 = np.zeros_like(Xm)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # precompute nothing
    for t in range(1, steps + 1):
        if t % 50 == 0 and time.time() >= deadline:
            break
        frac = (t - 1) / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        # cosines
        Cmf = Xm @ Xf.T  # k x nf
        Cmm = Xm @ Xm.T  # k x k
        # flatten for softmax
        # max for stability
        mx1 = float(np.max(Cmf)) if Cmf.size else -2.0
        # triu max
        iu = np.triu_indices(k, 1)
        if len(iu[0]) > 0:
            mx2 = float(np.max(Cmm[iu])) if k > 1 else -2.0
        else:
            mx2 = -2.0
        mV = mx1 if mx1 >= mx2 else mx2
        Wmf = np.exp(np.clip(alpha * (Cmf - mV), -30.0, 0.0))
        if len(iu[0]) > 0:
            Wmm_vals = np.exp(np.clip(alpha * (Cmm[iu] - mV), -30.0, 0.0))
        else:
            Wmm_vals = np.zeros(0)
        S = float(np.sum(Wmf) + np.sum(Wmm_vals))
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Wmf /= S
        Wmm_vals /= S
        G = Wmf @ Xf  # k x d
        if len(iu[0]) > 0:
            # accumulate symmetric
            # need full matrix contribution: for each pair weight w, G[i]+=w*Xm[j]
            Wfull = np.zeros((k, k))
            Wfull[iu] = Wmm_vals
            Wfull = Wfull + Wfull.T
            G += Wfull @ Xm
        # tangent projection
        dot = np.sum(G * Xm, axis=1, keepdims=True)
        Gt = G - dot * Xm
        # adam
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** t)
        m2h = m2 / (1 - b2 ** t)
        step = m1h / (np.sqrt(m2h) + eps)
        Xm = Xm - lr * step
        # renormalize
        Xm = _norm(Xm)
    return Xm

def solve():
    t0 = time.time()
    HARD_END = t0 + 132.0
    n, d = 82, 14
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
        rng0 = np.random.default_rng(0)
        cands.append(_norm(rng0.standard_normal((n, d))))
    cands.sort(key=lambda c: _max_cos(c))
    Xcur = cands[0].copy()
    Xcur = _norm(Xcur)
    m0, me100, me500, c0 = _metrics_full(Xcur)
    cur_tup = (m0, me100, me500)
    Xbest = Xcur.copy()
    best_max = m0
    best_tup = cur_tup

    iu0 = np.triu_indices(n, 1)
    # pool of random sphere samples 50k
    rng_pool = np.random.default_rng(0)
    P = 50000
    pool = _norm(rng_pool.standard_normal((P, d)))
    rng_trial = np.random.default_rng(1)

    archive = set()
    try:
        archive.add(_fingerprint(Xcur, tol=1e-3))
    except Exception:
        pass
    lp_calls = 0

    # trial size pattern: total deleted 8,14,22 cycling with variants
    pattern = [8, 14, 8, 22, 14, 8, 22, 14, 8, 22]
    trial = 0
    max_trials = 200
    while trial < max_trials and time.time() < HARD_END - 6.0:
        trial += 1
        # recompute metrics/degrees for Xcur
        try:
            m_cur, mean100_cur, mean500_cur, c_cur = _metrics_full(Xcur)
        except Exception:
            break
        cur_tup = (m_cur, mean100_cur, mean500_cur)
        deg_w, deg_c = _weighted_degrees(Xcur, m_cur, c_cur, iu0, tol=1e-3)
        # contact pair(s) at 1e-9
        sel_c = c_cur >= m_cur - 1e-9
        Ic = iu0[0][sel_c]; Jc = iu0[1][sel_c]
        contact_pts = []
        for a, b in zip(Ic, Jc):
            if int(a) not in contact_pts:
                contact_pts.append(int(a))
            if int(b) not in contact_pts:
                contact_pts.append(int(b))
            if len(contact_pts) >= 4:
                break
        if len(contact_pts) < 2:
            # fallback: top pair
            idx = int(np.argmax(c_cur))
            contact_pts = [int(iu0[0][idx]), int(iu0[1][idx])]
        # desired total
        k_total = pattern[(trial - 1) % len(pattern)]
        # order by weighted degree descending
        order = np.argsort(-deg_w)
        del_set = []
        for cp in contact_pts[:2]:
            if cp not in del_set:
                del_set.append(cp)
        for o in order:
            if len(del_set) >= k_total:
                break
            o = int(o)
            if o not in del_set:
                del_set.append(o)
        # sometimes add rattlers
        if trial % 3 == 0:
            zero_idx = np.where(deg_c == 0)[0]
            if len(zero_idx) > 0:
                # add up to 2 rattlers replacing? just add 1-2 if room else swap
                # keep total fixed: replace lowest-degree among del_set beyond contacts?
                # simpler: add 2 extra (increase k by 2) capped at 24
                extra = []
                # deterministic pick via rng
                perm = rng_trial.permutation(zero_idx)
                for z in perm:
                    z = int(z)
                    if z not in del_set:
                        extra.append(z)
                    if len(extra) >= 2:
                        break
                for e in extra:
                    if len(del_set) < 24 and e not in del_set:
                        del_set.append(e)
        k_total = len(del_set)
        del_arr = np.array(del_set, dtype=int)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_arr] = False
        keep_idx = np.where(keep_mask)[0]
        Xf = Xcur[keep_mask]  # nf x d
        # far-hole candidates
        # scores to frozen
        try:
            S = pool @ Xf.T  # P x nf
            smax = np.max(S, axis=1)
        except Exception:
            continue
        C_CAND = 3000
        idx_sorted = np.argpartition(smax, C_CAND - 1)[:C_CAND]
        # sort those C by smax
        o2 = np.argsort(smax[idx_sorted])
        idx_sorted = idx_sorted[o2]
        cand_X = pool[idx_sorted]
        cand_s = smax[idx_sorted]
        # randomized greedy start
        start_choice = int(rng_trial.integers(0, min(20, C_CAND)))
        picked = [start_choice]
        # max to selected array
        # compute dots incrementally
        max_to_sel = cand_X @ cand_X[start_choice]  # C,
        max_to_sel[start_choice] = -2.0  # avoid repick? will mask
        used = np.zeros(C_CAND, dtype=bool)
        used[start_choice] = True
        for _ in range(1, k_total):
            if time.time() >= HARD_END - 5.0:
                break
            total = np.maximum(cand_s, max_to_sel)
            total[used] = 2.0
            nxt = int(np.argmin(total))
            picked.append(nxt)
            used[nxt] = True
            # update
            try:
                dvec = cand_X @ cand_X[nxt]
                max_to_sel = np.maximum(max_to_sel, dvec)
                max_to_sel[used] = -2.0
            except Exception:
                break
        try:
            Xm_init = cand_X[np.array(picked)].copy()
        except Exception:
            continue
        # annealed adam
        steps = 400 if k_total <= 10 else (500 if k_total <= 16 else 550)
        # vary alpha start slightly per trial for diversity
        a0 = 50.0
        a1 = 500.0
        # target descending concept: bias lr slightly by trial
        Xm_opt = _adam_refine(Xm_init, Xf, HARD_END - 4.0, steps=steps, a0=a0, a1=a1)
        # assemble
        Xnew = Xcur.copy()
        Xnew[del_arr] = Xm_opt
        Xnew = _norm(Xnew)
        try:
            m_new, mean100_new, mean500_new, _ = _metrics_full(Xnew)
        except Exception:
            continue
        new_tup = (m_new, mean100_new, mean500_new)
        # fingerprint
        try:
            fp = _fingerprint(Xnew, tol=1e-3)
            is_new_fp = fp not in archive
        except Exception:
            is_new_fp = False
            fp = None
        improved_lex = _lex_less(new_tup, cur_tup)
        improved_max = m_new < best_max - 1e-12
        # update best max regardless of lex? only if finite
        if m_new < best_max and np.all(np.isfinite(Xnew)):
            Xbest = Xnew.copy()
            best_max = m_new
            best_tup = new_tup
        if improved_lex and np.all(np.isfinite(Xnew)):
            Xcur = Xnew.copy()
            cur_tup = new_tup
            if fp is not None:
                archive.add(fp)
            # short wide LP polish only after fingerprint change and promising
            if is_new_fp and lp_calls < 8 and time.time() < HARD_END - 8.0:
                # only polish if within 3e-4 of best or lex improved a lot
                if m_new < best_max + 3e-4:
                    lp_calls += 1
                    try:
                        Xpol = _lp_polish_short(Xcur, HARD_END - 3.0, radius_init=3e-4, cap=2000, max_it=5)
                        mp, m100p, m500p, _ = _metrics_full(Xpol)
                        ptup = (mp, m100p, m500p)
                        if mp < best_max and np.all(np.isfinite(Xpol)):
                            Xbest = Xpol.copy()
                            best_max = mp
                            best_tup = ptup
                        if _lex_less(ptup, cur_tup):
                            Xcur = Xpol.copy()
                            cur_tup = ptup
                            try:
                                archive.add(_fingerprint(Xcur, tol=1e-3))
                            except Exception:
                                pass
                    except Exception:
                        pass
        else:
            # even without lex improvement, if fingerprint new and max close, consider archive but not move?
            # keep archive to avoid repeats: add only if it beats best max? no.
            # To encourage diversity, if is_new_fp and mean improved notably with tiny max regress, allow move with tolerance
            if is_new_fp and m_new < m_cur + 5e-06 and (mean100_new < mean100_cur - 2e-06 or mean500_new < mean500_cur - 2e-06):
                if time.time() < HARD_END - 6.0 and np.all(np.isfinite(Xnew)):
                    # accept as lateral move (lex with relaxed max)
                    Xcur = Xnew.copy()
                    cur_tup = new_tup
                    if fp is not None:
                        archive.add(fp)
        # occasional time check
        if time.time() >= HARD_END - 4.0:
            break
    # final tiny polish on best if time remains
    if time.time() < HARD_END - 2.0:
        try:
            Xpol = _lp_polish_short(Xbest, HARD_END - 0.5, radius_init=8e-5, cap=2000, max_it=4)
            mp, _, _, _ = _metrics_full(Xpol)
            if mp < best_max:
                Xbest = Xpol.copy()
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
