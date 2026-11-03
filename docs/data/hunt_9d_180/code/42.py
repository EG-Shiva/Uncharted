import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def full_max_and_counts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.count_nonzero(G > m - tol, axis=1)
    return m, cnt, G

def best_void_for_index(x, fidx, rng, n_sample=200000, batch=10000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[fidx] = False
    Xf = x[mask]  # (n-1,d)
    best_v = None
    best_m = 1e9
    nb = n_sample // batch
    for _ in range(nb):
        cand = rng.standard_normal((batch, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = cand @ Xf.T
        mx = sc.max(axis=1)
        j = int(np.argmin(mx))
        if float(mx[j]) < best_m:
            best_m = float(mx[j])
            best_v = cand[j].copy()
    return best_v, best_m

def hill_climb_void(v0, Xf, steps=200, k=30.0, lr=0.03):
    v = v0 / (np.linalg.norm(v0) + 1e-12)
    for _ in range(steps):
        s = Xf @ v
        smax = float(s.max())
        w = np.exp(k * (s - smax))
        # w sum not needed, direction = weighted sum
        g = (w[:, None] * Xf).sum(axis=0)
        g = g - float(g @ v) * v
        gn = float(np.linalg.norm(g)) + 1e-12
        g /= gn
        v = v - lr * g
        nv = float(np.linalg.norm(v)) + 1e-12
        v /= nv
    return v

def subset_lp(x_init, S, time_budget, t_deadline):
    # trust-region LP on subset S, core frozen
    x_cur = x_init.copy()
    n, d = x_cur.shape
    S = np.array(sorted(set(int(v) for v in S)), dtype=int)
    s = len(S)
    if s == 0:
        return x_cur, max_cos(x_cur)
    is_S = np.zeros(n, dtype=bool)
    is_S[S] = True
    pos = np.full(n, -1, dtype=int)
    pos[S] = np.arange(s)
    # triu indices for fast selection
    iu0, iu1 = np.triu_indices(n, 1)
    is_involve = is_S[iu0] | is_S[iu1]
    best_cur = max_cos(x_cur)
    radius = 1e-3
    start = time.time()
    ar = np.arange(d)
    while True:
        if time.time() >= t_deadline:
            break
        if time.time() - start >= time_budget:
            break
        if radius < 1e-14:
            break
        G = x_cur @ x_cur.T
        # global max
        # use iu for max
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        # select involving S
        sel = (c_all > thr) & is_involve
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        I_all = iu0[sel]
        J_all = iu1[sel]
        c_sel = c_all[sel]
        if p > 1500:
            idx = np.argpartition(-c_sel, 1500)[:1500]
            I_all = I_all[idx]
            J_all = J_all[idx]
            c_sel = c_sel[idx]
            p = 1500
        nv = s * d + 1
        # build dense A_ub (p, nv)
        A_ub = np.zeros((p, nv), dtype=float)
        # vectorize where possible: need x rows
        Xi = x_cur[I_all]  # (p,d) actually x_I
        Xj = x_cur[J_all]
        inI = is_S[I_all]
        inJ = is_S[J_all]
        # fill row by row for mixed cases (loop is fine for p<=1500)
        # Use vectorized for both-in-S via advanced indexing
        # For speed, loop in python but with numpy assignment per row
        # Split indices
        both = inI & inJ
        onlyI = inI & (~inJ)
        onlyJ = (~inI) & inJ
        # both
        if np.any(both):
            bidx = np.where(both)[0]
            pi = pos[I_all[bidx]]  # (nb,)
            pj = pos[J_all[bidx]]
            # A_ub[bidx, pi*d:(pi+1)*d] = Xj[bidx]
            # do loop over bidx but vectorized via fancy? use loop (usually few hundred)
            for t_, r_ in enumerate(bidx):
                a = pi[t_] * d
                b = pj[t_] * d
                A_ub[r_, a:a+d] = Xj[r_]
                A_ub[r_, b:b+d] = Xi[r_]
        if np.any(onlyI):
            bidx = np.where(onlyI)[0]
            pi = pos[I_all[bidx]]
            for t_, r_ in enumerate(bidx):
                a = pi[t_] * d
                A_ub[r_, a:a+d] = Xj[r_]
        if np.any(onlyJ):
            bidx = np.where(onlyJ)[0]
            pj = pos[J_all[bidx]]
            for t_, r_ in enumerate(bidx):
                a = pj[t_] * d
                A_ub[r_, a:a+d] = Xi[r_]
        A_ub[:, -1] = -1.0
        b_ub = -c_sel
        # A_eq s x nv
        A_eq = np.zeros((s, nv), dtype=float)
        for k_, g_ in enumerate(S):
            A_eq[k_, k_*d:(k_+1)*d] = x_cur[g_]
        b_eq = np.zeros(s)
        obj = np.zeros(nv)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * (s*d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dv = res.x[:-1].reshape(s, d)
        y = x_cur.copy()
        y[S] = x_cur[S] + dv
        y[S] /= np.linalg.norm(y[S], axis=1, keepdims=True)
        v = max_cos(y)
        if v < best_cur:
            x_cur = y
            best_cur = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x_cur, best_cur

def full_lp_polish(x_init, time_budget, t_deadline):
    x = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() >= t_deadline:
            break
        if time.time() - start >= time_budget:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        p = int(np.count_nonzero(sel))
        if p == 0:
            break
        I_all = iu0[sel]
        J_all = iu1[sel]
        c_sel = c_all[sel]
        if p > 2500:
            idx = np.argpartition(-c_sel, 2500)[:2500]
            I_all = I_all[idx]
            J_all = J_all[idx]
            c_sel = c_sel[idx]
            p = 2500
        nv = n * d + 1
        # vectorized sparse construction row-major
        cols2 = np.empty((p, 2*d+1), dtype=int)
        cols2[:, 0:d] = I_all[:, None] * d + ar
        cols2[:, d:2*d] = J_all[:, None] * d + ar
        cols2[:, 2*d] = n * d
        vals2 = np.empty((p, 2*d+1), dtype=float)
        vals2[:, 0:d] = x[J_all]
        vals2[:, d:2*d] = x[I_all]
        vals2[:, 2*d] = -1.0
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = cols2.ravel()
        vals = vals2.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 160.0
    t_deadline = t0 + HARD
    np.random.seed(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    x0 = None
    if xb is not None and xb.shape == (180, 9):
        x0 = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    else:
        try:
            xr = np.load("record.npy")
            x0 = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        except Exception:
            rng0 = np.random.default_rng(0)
            c = rng0.standard_normal((180, 9))
            x0 = c / np.linalg.norm(c, axis=1, keepdims=True)
    x_best = x0.copy()
    best_val = max_cos(x_best)
    # identify free points at 1e-7
    try:
        m0, cnt, G0 = full_max_and_counts(x_best, tol=1e-7)
    except Exception:
        cnt = np.zeros(180, dtype=int)
    free = np.where(cnt <= 3)[0].tolist()
    # ensure at least 2, at most 10
    if len(free) < 2:
        order = np.argsort(cnt)
        free = order[:5].tolist()
    if len(free) > 10:
        order = np.argsort(cnt)
        # keep lowest 10
        # cnt sorted ascending
        free = order[:10].tolist()
    free = [int(v) for v in free]
    # Cycle free points a few times
    n_cycles = 3
    for cyc in range(n_cycles):
        for fi, f in enumerate(list(free)):
            if time.time() >= t_deadline - 30.0:
                break
            # relocate f one at a time using 200k samples + hill climb
            rng = np.random.default_rng(9000 + cyc*1000 + fi*137 + f*13)
            # current max involving f
            dots_old = float(np.max(np.delete(x_best @ x_best[f], f)))
            try:
                v0, smax0 = best_void_for_index(x_best, f, rng, n_sample=200000, batch=10000)
            except Exception:
                continue
            if time.time() >= t_deadline - 28.0:
                break
            try:
                mask = np.ones(180, dtype=bool); mask[f] = False
                Xf = x_best[mask]
                v1 = hill_climb_void(v0, Xf, steps=180, k=30.0, lr=0.03)
                s1 = float(np.max(Xf @ v1))
            except Exception:
                continue
            # accept relocation only if not worsening global score
            # global best currently best_val; new global would be max(m_core, s1)
            # m_core = max over pairs not involving f
            # compute m_core quickly
            try:
                Gtmp = x_best @ x_best.T
                np.fill_diagonal(Gtmp, -2.0)
                # mask out row/col f
                Gtmp[f, :] = -2.0
                Gtmp[:, f] = -2.0
                m_core = float(Gtmp.max())
            except Exception:
                m_core = best_val
            new_global = m_core if m_core > s1 else s1
            # accept if new_global <= best_val + 1e-12 (no worsening) or s1 < dots_old
            accept = False
            if new_global <= best_val + 1e-12:
                accept = True
            elif s1 < dots_old and new_global <= best_val + 1e-4:
                accept = True
            if accept:
                x_best[f] = v1 / (np.linalg.norm(v1) + 1e-12)
                # update best_val if improved (unlikely here)
                bv = max_cos(x_best)
                if bv < best_val:
                    best_val = bv
            else:
                # keep old, still try subset LP around old position? continue to subset step anyway
                pass
            if time.time() >= t_deadline - 26.0:
                break
            # subset: f + 30-40 nearest neighbours with cosine above ~0.30
            try:
                dots = x_best @ x_best[f]
                dots[f] = -2.0
                order_n = np.argsort(-dots)
                # take top 35
                neigh = order_n[:35].tolist()
                # ensure threshold 0.30 roughly: if many below, keep at least 30
                # count above 0.30
                nabove = int(np.count_nonzero(dots > 0.30))
                if nabove >= 40:
                    neigh = order_n[:40].tolist()
                elif nabove < 30:
                    neigh = order_n[:30].tolist()
                else:
                    neigh = order_n[:max(35, nabove)].tolist()
                    neigh = neigh[:40]
                S = [f] + [int(v) for v in neigh]
            except Exception:
                continue
            # run subset LP ~3s
            rem = t_deadline - time.time() - 22.0
            if rem < 2.0:
                break
            sec = min(3.5, max(1.0, rem / 4.0))
            x_before = x_best.copy()
            bv_before = best_val
            try:
                x_sub, v_sub = subset_lp(x_best, S, sec, t_deadline - 20.0)
            except Exception:
                continue
            # accept only true full-score improvement
            try:
                v_full = max_cos(x_sub)
            except Exception:
                continue
            if v_full < best_val - 1e-12:
                x_best = x_sub
                best_val = v_full
            else:
                # revert (keep relocation but discard LP if no improvement)
                # relocation already in x_before? x_before includes relocation; subset started from relocated
                # if subset didn't improve, keep relocated version
                # x_best already holds relocated; ensure we keep it
                # recompute best_val for relocated
                try:
                    best_val = max_cos(x_best)
                    # if relocated worsened (shouldn't due to accept), revert fully
                    if best_val > bv_before + 1e-12:
                        x_best = x_before
                        best_val = bv_before
                except Exception:
                    pass
        if time.time() >= t_deadline - 30.0:
            break
    # one full LP polish with remaining time
    rem = t_deadline - time.time() - 1.0
    if rem > 2.0:
        try:
            x_best = full_lp_polish(x_best, min(rem, 18.0), t_deadline)
            best_val = max_cos(x_best)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
