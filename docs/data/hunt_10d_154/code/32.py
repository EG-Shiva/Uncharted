import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, radius0=1e-3):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() + 0.5 > start + seconds:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 4000:
            # keep worst 4000
            idx = np.argpartition(-c[sel], 4000)[:4000]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 4000
        else:
            csv = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _single_refine(y0, Xf, steps=350, lr=0.05, T0=0.02, T1=0.002):
    d = Xf.shape[1]
    y = y0 / max(np.linalg.norm(y0), 1e-12)
    m_ad = np.zeros(d)
    v_ad = np.zeros(d)
    b1p = 1.0
    b2p = 1.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    if steps > 1:
        logr = np.log(T1 / T0) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        s = Xf @ y
        mv = s.max()
        E = np.exp((s - mv) / max(T, 1e-12))
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (Xf.T @ E) / S
        Gt = Grad - float(Grad @ y) * y
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        y = y - lr * step
        ny = np.linalg.norm(y)
        if ny == 0 or not np.isfinite(ny):
            y = y0 / max(np.linalg.norm(y0), 1e-12)
            break
        y = y / ny
    return y

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(42)
    n, d = 154, 10
    best0 = None
    best0_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best0_val:
                    best0_val = v
                    best0 = a.copy()
        except Exception:
            pass
    if best0 is None:
        best0 = _norm(rng.normal(size=(n, d)))
        best0_val = _max_cos(best0)

    # contact analysis on incumbent
    G = best0 @ best0.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    cnt_tight = (G > m0 - 1e-9).sum(axis=1)
    rattlers = np.where(cnt_tight == 0)[0]
    # order by contacts (tight + loose for stability)
    cnt_loose = (G > m0 - 0.01).sum(axis=1)
    order = np.argsort(-(cnt_tight.astype(float) * 10 + cnt_loose.astype(float)))
    # decide deletion: all rattlers + top hubs to total 30
    NDEL = 30
    victims = set(rattlers.tolist())
    for idx in order:
        if len(victims) >= NDEL:
            break
        victims.add(int(idx))
    victims = sorted(victims)
    # if rattlers > NDEL, keep NDEL of them
    if len(victims) > 34:
        victims = victims[:34]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    X = best0[keep_mask].copy()
    m_core_target_n = X.shape[0]

    # LP-polish reduced core until substantially lower
    # budget 8s or time-aware
    remain0 = DEAD - time.time()
    core_budget = min(10.0, max(4.0, remain0 - 30 * 2.5 - 25.0))
    if core_budget > 1.0:
        X = _lp_polish(X, core_budget)
    m_prev = _max_cos(X)

    # regrowth to exactly 154
    n_need = n - X.shape[0]
    FINAL_RESERVE = 22.0
    for k in range(n_need):
        time_left = DEAD - time.time()
        points_left = n - X.shape[0]
        if points_left <= 0:
            break
        if time_left < FINAL_RESERVE + 1.0:
            # emergency greedy fill without LP
            while X.shape[0] < n:
                Cfast = 8000
                cand = _norm(rng.normal(size=(Cfast, d)))
                md = (cand @ X.T).max(axis=1)
                j = int(np.argmin(md))
                y0 = cand[j]
                y0 = _single_refine(y0, X, steps=200, lr=0.05, T0=0.015, T1=0.002)
                X = np.vstack([X, y0[None, :]])
            break
        # per-slot LP budget
        per_slot = (time_left - FINAL_RESERVE) / points_left
        per_slot = float(np.clip(per_slot, 0.7, 3.0))
        # sample at least 20000 holes
        C = 20000
        # batch to avoid huge memory if needed
        cand = _norm(rng.normal(size=(C, d)))
        Dots = cand @ X.T
        md = Dots.max(axis=1)
        # top 5 holes
        K = 5
        part = np.argpartition(md, K)[:K]
        orderk = part[np.argsort(md[part])]
        # try best hole first
        tried_vals = []
        tried_X = []
        # first attempt
        y0 = cand[orderk[0]].copy()
        y_ref = _single_refine(y0, X, steps=350, lr=0.05, T0=0.02, T1=0.002)
        X_trial = np.vstack([X, y_ref[None, :]])
        # short LP polish over all points
        X_pol = _lp_polish(X_trial, per_slot)
        v_pol = _max_cos(X_pol)
        # jump detection
        jump = v_pol - m_prev
        if jump > 0.015 and len(orderk) > 1 and points_left > 1:
            # retry next-best holes (up to 3 total)
            best_v = v_pol
            best_X = X_pol
            nretry = min(3, len(orderk))
            for rr in range(1, nretry):
                if DEAD - time.time() < FINAL_RESERVE + 0.5:
                    break
                y0b = cand[orderk[rr]].copy()
                y_refb = _single_refine(y0b, X, steps=300, lr=0.05, T0=0.02, T1=0.002)
                X_trialb = np.vstack([X, y_refb[None, :]])
                # slightly shorter polish for retries
                X_polb = _lp_polish(X_trialb, max(0.6, per_slot * 0.8))
                v_polb = _max_cos(X_polb)
                if v_polb < best_v - 1e-12:
                    best_v = v_polb
                    best_X = X_polb
            # swapped-order repair: if still sharp jump, remove an extra high-contact neighbour and reinsert both
            if best_v - m_prev > 0.02:
                try:
                    Gt = best_X @ best_X.T
                    np.fill_diagonal(Gt, -2.0)
                    # neighbour most aligned to last inserted point (last row)
                    last = best_X.shape[0] - 1
                    nbr = int(np.argmax(Gt[last, :-1]))
                    # remove nbr and reinsert both greedily swapped
                    tmp = np.delete(best_X, [nbr], axis=0)
                    # reinsert two holes sequentially (greedy, refined, no LP)
                    for _ in range(2):
                        # wait tmp has one fewer; need +1 to reach desired size
                        # first re-add one, then second = net +1 (since we removed 1)
                        Cc2 = 12000
                        c2 = _norm(rng.normal(size=(Cc2, d)))
                        md2 = (c2 @ tmp.T).max(axis=1)
                        jb = int(np.argmin(md2))
                        yb = _single_refine(c2[jb], tmp, steps=250, lr=0.05, T0=0.015, T1=0.002)
                        tmp = np.vstack([tmp, yb[None, :]])
                    # quick polish
                    tmp = _lp_polish(tmp, max(0.8, per_slot))
                    v_tmp = _max_cos(tmp)
                    if v_tmp < best_v:
                        best_v = v_tmp
                        best_X = tmp
                except Exception:
                    pass
            X = best_X
            m_prev = best_v
        else:
            X = X_pol
            m_prev = v_pol

    # ensure exactly 154
    if X.shape[0] > n:
        X = X[:n]
    while X.shape[0] < n:
        C = 10000
        cand = _norm(rng.normal(size=(C, d)))
        md = (cand @ X.T).max(axis=1)
        j = int(np.argmin(md))
        X = np.vstack([X, cand[j][None, :]])
    X = _norm(X)

    # finish with remaining time in LP polish
    time_left = DEAD - time.time()
    if time_left > 4.0:
        lp_time = min(max(5.0, time_left - 2.0), 30.0)
        try:
            X = _lp_polish(X, lp_time)
        except Exception:
            pass
    v_final = _max_cos(X)
    # never regress vs incumbent
    if v_final > best0_val and best0.shape == (n, d):
        # try quick polish of incumbent with leftover time instead
        left2 = DEAD - time.time()
        if left2 > 3.0:
            try:
                bpol = _lp_polish(best0, min(6.0, left2 - 1.0))
                v_bpol = _max_cos(bpol)
                if v_bpol < v_final:
                    return _norm(bpol)
            except Exception:
                pass
            return _norm(best0)
        return _norm(best0)
    return _norm(X)
