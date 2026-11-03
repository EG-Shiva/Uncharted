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

def _lp_polish(X, seconds, t_end=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    an_d = np.arange(n * d)
    while time.time() - start < seconds and time.time() < t_end - 0.5 and radius > 1e-14:
        C = (X @ X.T)[iu]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (rep_n, an_d)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_single(y0, Xfix, steps=500, lr0=0.03, T0=0.02, T1=0.003):
    y = y0.copy()
    yn = np.linalg.norm(y)
    if yn == 0:
        y = Xfix[0].copy()
    else:
        y = y / yn
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0); l1 = np.log(T1)
    for t in range(1, steps + 1):
        frac = t / max(steps, 1)
        T = float(np.exp(l0 + (l1 - l0) * frac))
        lr = lr0 * (1.0 - 0.6 * frac)
        dots = Xfix @ y
        mv = float(np.max(dots))
        E = np.exp((dots - mv) / max(T, 1e-12))
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        w = E / S
        g = Xfix.T @ w
        gt = g - float(g @ y) * y
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        step = (m_ad / c1) / (np.sqrt(v_ad / c2) + eps)
        y = y - lr * step
        nrm = np.linalg.norm(y)
        if nrm == 0 or not np.isfinite(nrm):
            y = y0 / (np.linalg.norm(y0) + 1e-12)
            continue
        y = y / nrm
    return y

def _best_holes(Xcur, rng, C=20000, K=6):
    d = Xcur.shape[1]
    cand = rng.standard_normal((C, d))
    cand = _norm(cand)
    D = cand @ Xcur.T
    md = D.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    order = idx[np.argsort(md[idx])]
    return cand, md, order

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    t_end_final = t0 + 174.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best_inc = None
    best_inc_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_inc_val:
                    best_inc_val = v
                    best_inc = a.copy()
        except Exception:
            pass
    if best_inc is None:
        best_inc = _norm(rng.standard_normal((n, d)))
        best_inc_val = _max_cos(best_inc)
    X0 = best_inc.copy()

    # --- identify deletions: 11 rattlers (lowest rowmax) + high-contact core ---
    G = X0 @ X0.T
    np.fill_diagonal(G, -2.0)
    m0 = float(np.max(G))
    rowmax = np.max(G, axis=1)
    # contact degree with moderate band to find overloaded core
    deg = np.sum(G > m0 - 0.01, axis=1)
    order_low = np.argsort(rowmax)  # rattlers first
    order_high = np.argsort(-deg - rowmax * 1.0)
    n_ratt = 11
    ratt_idx = list(order_low[:n_ratt])
    # 19-29 highest-contact/dual points, exclude rattlers
    n_core = 22  # total delete 33 -> 121 remain (in 114-124)
    core_idx = []
    for ii in order_high:
        if ii in ratt_idx:
            continue
        core_idx.append(int(ii))
        if len(core_idx) >= n_core:
            break
    del_idx = sorted(ratt_idx + core_idx)
    keep = np.ones(n, dtype=bool)
    keep[del_idx] = False
    Xred = X0[keep].copy()
    # --- LP-polish reduced set until substantially lower ---
    # use ~8s
    time_left = DEAD - time.time()
    red_polish_sec = min(9.0, max(4.0, time_left - 33 * 2.3 - 28.0))
    if red_polish_sec < 2.0:
        red_polish_sec = 2.0
    try:
        Xred = _lp_polish(Xred, red_polish_sec, t_end=DEAD - 33 * 2.0 - 20.0)
    except Exception:
        pass
    red_val = _max_cos(Xred)

    # --- regrow one at a time to exactly 154 ---
    Xcur = Xred.copy()
    # reserve final polish
    FINAL_RESERVE = 26.0
    n_need = n - Xcur.shape[0]
    jump_tol = 0.008
    for s in range(n_need):
        if time.time() > DEAD - FINAL_RESERVE - 2.0 * (n_need - s) - 1.0:
            # not enough time for careful per-slot LP; shrink LP time
            pass
        if time.time() > DEAD - FINAL_RESERVE:
            break
        prev_val = _max_cos(Xcur)
        # sample holes
        C = 20000
        # reduce C if very late
        if time.time() > DEAD - FINAL_RESERVE - 15:
            C = 12000
        cand, md, order = _best_holes(Xcur, rng, C=C, K=8)
        # try up to 3 hole candidates
        tried_best_X = None
        tried_best_v = 1e9
        # short LP seconds per try
        per_try_sec = 2.2
        if s > n_need - 6:
            per_try_sec = 1.6
        ntries = 3 if time.time() < DEAD - FINAL_RESERVE - 8 else 1
        for ti in range(ntries):
            pick = int(order[ti])
            y0 = cand[pick].copy()
            # locally refine single point against fixed set
            try:
                yref = _refine_single(y0, Xcur, steps=450, lr0=0.03, T0=0.02, T1=0.0025)
            except Exception:
                yref = y0
            Xnew = np.vstack([Xcur, yref[None, :]])
            Xnew = _norm(Xnew)
            # short LP polish over all points before next insertion
            try:
                Xpol = _lp_polish(Xnew, per_try_sec, t_end=DEAD - FINAL_RESERVE + 2.0 if s == n_need - 1 else DEAD)
            except Exception:
                Xpol = Xnew
            vpol = _max_cos(Xpol)
            if vpol < tried_best_v:
                tried_best_v = vpol
                tried_best_X = Xpol
            # if no sharp jump, accept immediately
            if vpol <= prev_val + jump_tol:
                break
            # else continue to next-best hole
        if tried_best_X is None:
            # fallback: insert best hole without polish
            yref = _refine_single(cand[int(order[0])], Xcur, steps=300)
            tried_best_X = _norm(np.vstack([Xcur, yref[None, :]]))
            tried_best_v = _max_cos(tried_best_X)
        # swapped-order fallback: if jump still sharp, remove a neighbouring high-contact point and reinsert both
        if tried_best_v > prev_val + 0.015 and Xcur.shape[0] >= 10 and time.time() < DEAD - FINAL_RESERVE - 6:
            try:
                y_new = tried_best_X[-1].copy()
                Gc = Xcur @ y_new
                nbr = int(np.argmax(Gc))
                # remove nbr, keep rest + y_new, then re-find hole for nbr slot
                mask = np.ones(Xcur.shape[0], dtype=bool); mask[nbr] = False
                Xtmp = np.vstack([Xcur[mask], y_new[None, :]])
                Xtmp = _norm(Xtmp)
                cand2, md2, order2 = _best_holes(Xtmp, rng, C=8000, K=4)
                y2 = _refine_single(cand2[int(order2[0])], Xtmp, steps=350)
                Xalt = np.vstack([Xtmp, y2[None, :]])
                Xalt = _lp_polish(_norm(Xalt), 1.8, t_end=DEAD)
                valt = _max_cos(Xalt)
                if valt < tried_best_v:
                    tried_best_X = Xalt
                    tried_best_v = valt
            except Exception:
                pass
        Xcur = tried_best_X
        # safety: if Xcur size wrong, fix
        if Xcur.shape[0] != Xred.shape[0] + s + 1:
            # truncate/pad
            if Xcur.shape[0] > Xred.shape[0] + s + 1:
                Xcur = Xcur[:Xred.shape[0] + s + 1]
    # ensure exactly 154 (pad with random if time abort left us short, truncate if over)
    if Xcur.shape[0] < n:
        need = n - Xcur.shape[0]
        extra = rng.standard_normal((need, d))
        extra = _norm(extra)
        # greedy place extras
        for k in range(need):
            cand, md, order = _best_holes(Xcur, rng, C=8000, K=1)
            y0 = cand[int(order[0])]
            try:
                y0 = _refine_single(y0, Xcur, steps=250)
            except Exception:
                pass
            Xcur = np.vstack([Xcur, y0[None, :]])
        Xcur = _norm(Xcur)
    elif Xcur.shape[0] > n:
        Xcur = Xcur[:n].copy()
    # --- finish with remaining time in LP polish ---
    rem = (t0 + 172.0) - time.time() - 0.8
    if rem > 2.0:
        try:
            Xcur = _lp_polish(Xcur, rem, t_end=t_end_final - 0.5)
        except Exception:
            pass
    cur_val = _max_cos(Xcur)
    # return the better of regrown vs incumbent (polished if time permits)
    if cur_val < best_inc_val:
        return _norm(Xcur)
    else:
        # spend any leftover <2s micro-polish on incumbent? return incumbent
        # try one tiny polish if regrow failed and time remains
        tiny = (t0 + 174.0) - time.time() - 0.5
        if tiny > 2.0:
            try:
                Xb = _lp_polish(best_inc, min(tiny, 4.0), t_end=t0 + 175.0)
                if _max_cos(Xb) < best_inc_val:
                    return _norm(Xb)
            except Exception:
                pass
        return _norm(best_inc)
