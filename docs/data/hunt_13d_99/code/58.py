import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n

def _metrics(X, m=None):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    pt = np.max(G, axis=1)
    std_pt = float(np.std(pt))
    c = G[np.triu_indices(X.shape[0], 1)]
    cnt5 = int(np.sum(c > m - 1e-5))
    cnt6 = int(np.sum(c > m - 1e-6))
    return m, std_pt, cnt5, cnt6, c

def _anneal_gd(X, a0, a1, steps, lr0, mu, deadline=None, check_every=64):
    n, d = X.shape
    v = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s & (check_every - 1)) == 0 and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        # exponential anneal
        alpha = a0 * ((a1 / max(a0, 1e-12)) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        np.maximum(rs, 1e-12, out=rs)
        Grad = W @ X
        Gav = Grad / rs
        dn = np.sum(Gav * X, axis=1, keepdims=True)
        Gt = Gav - dn * X
        cur_lr = lr0 * (1.0 - 0.6 * frac)
        v = mu * v - cur_lr * Gt
        # cap velocity
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        cap = 0.08
        over = vn > cap
        if np.any(over):
            v = np.where(over, v * (cap / np.maximum(vn, 1e-12)), v)
        X = X + v
        X = _norm_rows(X)
    return X

def _high_alpha_polish(X, alpha, steps, lr0, mu, deadline=None):
    n, d = X.shape
    v = np.zeros_like(X)
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        np.maximum(rs, 1e-12, out=rs)
        Grad = W @ X
        Gav = Grad / rs
        dn = np.sum(Gav * X, axis=1, keepdims=True)
        Gt = Gav - dn * X
        cur_lr = lr0 * (1.0 - 0.5 * s / max(steps, 1))
        v = mu * v - cur_lr * Gt
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        cap = 0.05
        over = vn > cap
        if np.any(over):
            v = np.where(over, v * (cap / np.maximum(vn, 1e-12)), v)
        X = X + v
        X = _norm_rows(X)
    return X

def _lp_polish(X, seconds, deadline_hard, max_iters=25, p_cap=800):
    t_start = time.time()
    X = _norm_rows(X)
    best = _max_cos(X)
    n, d = X.shape
    radius = 6e-4
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - t_start < seconds and time.time() < deadline_hard:
        if radius < 1e-12 or it >= max_iters:
            break
        it += 1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0 * radius, 3e-6)
        sel = c_all > m - window
        cnt = int(np.sum(sel))
        if cnt < 300:
            need = min(600, c_all.size)
            order = np.argpartition(c_all, -need)[-need:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order] = True
            sel = sel | sel2
            cnt = int(np.sum(sel))
        if cnt > p_cap:
            idx = np.where(sel)[0]
            cv = c_all[idx]
            keep = np.argpartition(cv, -p_cap)[-p_cap:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[keep]] = True
            sel = sel2
            cnt = p_cap
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu0[sel]
        J = iu1[sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best - 1e-12:
            X, best = y, vv
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 80.0
    n, d = 99, 13
    rng = np.random.RandomState(20261103)

    # load incumbent for reference / fallback
    cands0 = []
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
        if xb.shape == (n, d):
            xb = _norm_rows(np.where(np.isfinite(xb), xb, 0.0))
            cands0.append(xb)
    except Exception:
        pass
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
        if xr.shape == (n, d):
            xr = _norm_rows(np.where(np.isfinite(xr), xr, 0.0))
            cands0.append(xr)
    except Exception:
        pass
    if len(cands0) == 0:
        cands0.append(_norm_rows(rng.standard_normal((n, d))))
    vals0 = [_max_cos(c) for c in cands0]
    bi = int(np.argmin(vals0))
    incumbent = cands0[bi].copy()
    best_overall = incumbent.copy()
    best_val = float(vals0[bi])
    # incumbent fingerprint (sorted top cosines) for dissimilarity
    Ginc = incumbent @ incumbent.T
    np.fill_diagonal(Ginc, -2.0)
    cinc = Ginc[np.triu_indices(n, 1)]
    cinc_sorted = np.sort(cinc)[::-1][:600].copy()
    inc_max = best_val

    # ---- exploration: 30-60 random inits ----
    N_INIT = 44
    STEPS = 300
    elite = []  # (combined, maxc, std, cnt5, X)
    for k in range(N_INIT):
        if time.time() > EXPLORE_END - 1.0:
            break
        if time.time() > HARD_END - 60.0:
            break
        X0 = _norm_rows(rng.standard_normal((n, d)))
        # diverse hyperparams
        lr0 = float(rng.uniform(0.015, 0.040))
        mu = float(rng.uniform(0.78, 0.90))
        a0 = float(rng.uniform(10.0, 30.0))
        a1 = float(rng.uniform(80.0, 180.0))
        X = _anneal_gd(X0, a0, a1, STEPS, lr0, mu, deadline=EXPLORE_END)
        m, std_pt, cnt5, cnt6, c = _metrics(X)
        # dissimilarity from incumbent via sorted fingerprint
        cs = np.sort(c)[::-1][:600]
        dissim = float(np.linalg.norm(cs - cinc_sorted) / max(np.linalg.norm(cinc_sorted), 1e-12))
        # structural combined score: prefer low max, low spread, many contacts
        combined = m + 0.30 * std_pt - 1e-7 * cnt5 - 5e-4 * min(dissim, 0.01)
        # keep elite of size 6
        elite.append((combined, m, std_pt, cnt5, cnt6, dissim, X.copy()))
        # track outright best exact
        if m < best_val:
            best_val = float(m)
            best_overall = X.copy()

    # sort elite by combined then by exact max
    elite.sort(key=lambda e: (e[0], e[1]))
    # keep diverse + good: take top 6 already, choose up to 3 to polish
    # prefer low max among top combined
    pool = elite[:8]
    # re-sort pool by exact max for polishing priority, but require some quality
    pool.sort(key=lambda e: e[1])
    # select up to 3 with best max (they are new basins; also ensure dissim>tiny or very good max)
    to_polish = []
    for e in pool:
        if len(to_polish) >= 3:
            break
        # skip extremely bad max
        if e[1] > inc_max + 0.01 and len(to_polish) >= 1:
            continue
        to_polish.append(e)
    if len(to_polish) == 0 and len(elite) > 0:
        to_polish = [sorted(elite, key=lambda e: e[1])[0]]

    # ---- polish only best 2-3 ----
    for idx, e in enumerate(to_polish):
        if time.time() > HARD_END - 8.0:
            break
        X = e[6].copy()
        # high-alpha LogSumExp polish, bounded
        remain = HARD_END - time.time() - 4.0
        if remain < 3.0:
            break
        # allocate time: ~ 1/ (remaining polish slots) each
        slots_left = len(to_polish) - idx
        budget = min(remain / max(slots_left, 1), 25.0)
        t_dead = min(time.time() + budget, HARD_END - 3.0)
        # stage 1: mid-high alpha
        X = _high_alpha_polish(X, 220.0, 400, 0.010, 0.86, deadline=t_dead)
        if time.time() > t_dead - 1.0:
            m = _max_cos(X)
            if m < best_val:
                best_val, best_overall = float(m), X.copy()
            continue
        X = _high_alpha_polish(X, 380.0, 300, 0.006, 0.88, deadline=t_dead)
        m = _max_cos(X)
        # stage 2: bounded LP (use leftover of budget)
        lp_sec = max(0.5, t_dead - time.time() - 0.5)
        lp_sec = min(lp_sec, 8.0)
        if lp_sec > 0.6 and time.time() < t_dead:
            X, m2 = _lp_polish(X, seconds=lp_sec, deadline_hard=t_dead)
            m = float(m2)
        if m < best_val:
            best_val = float(m)
            best_overall = X.copy()

    # final safety: if best_overall not better than incumbent, still try brief LP on incumbent if time
    if best_val >= inc_max - 1e-12 and (HARD_END - time.time()) > 6.0:
        X = incumbent.copy()
        X = _high_alpha_polish(X, 300.0, 250, 0.006, 0.88, deadline=HARD_END - 4.0)
        X, m2 = _lp_polish(X, seconds=min(4.0, HARD_END - time.time() - 2.0), deadline_hard=HARD_END - 1.0)
        if float(m2) < best_val:
            best_val, best_overall = float(m2), X.copy()

    best_overall = _norm_rows(np.where(np.isfinite(best_overall), best_overall, 0.0))
    return best_overall
