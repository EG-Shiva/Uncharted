import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_run(X0, t, steps, lr, T_lse, w_lse, seed_off, t_deadline, window=150):
    rng = np.random.default_rng(1000 + seed_off)
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_mc = _max_cos(X)
    best_X = X.copy()
    # initial violation count
    G0 = X @ X.T
    np.fill_diagonal(G0, -10.0)
    viol0 = int(np.count_nonzero(G0 > t))
    viol_window_start = viol0
    # excess sum for tracking best
    for s in range(1, steps + 1):
        if time.time() > t_deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mc = float(G.max())
        if mc < best_mc:
            best_mc = mc
            best_X = X.copy()
            if best_mc <= t:
                break
        excess = G - t
        np.maximum(excess, 0.0, out=excess)
        np.fill_diagonal(excess, 0.0)
        # quartic penalty derivative 4*excess^3
        S = 4.0 * excess * excess * excess
        # tiny LSE term: w*T*log sum exp(G/T)
        mmax = mc
        # avoid overflow: (G-mmax)/T <=0
        E = np.exp((G - mmax) / T_lse)
        np.fill_diagonal(E, 0.0)
        ssum = E.sum() * 0.5
        if ssum < 1e-300:
            ssum = 1e-300
        Slse = (w_lse / ssum) * E
        Stot = S + Slse
        Ge = Stot @ X
        dot = (Ge * X).sum(axis=1, keepdims=True)
        Gt = Ge - dot * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** s)
        vhat = v / (1.0 - b2 ** s)
        direction = mhat / (np.sqrt(vhat) + eps)
        dd = (direction * X).sum(axis=1, keepdims=True)
        direction = direction - dd * X
        # step decay slightly
        cur_lr = lr * (1.0 - 0.5 * (s - 1) / max(steps - 1, 1))
        X = X - cur_lr * direction
        X = _norm(X)
        if s % window == 0:
            Gc = X @ X.T
            np.fill_diagonal(Gc, -10.0)
            vc = int(np.count_nonzero(Gc > t))
            # abort if no decrease in window
            if vc >= viol_window_start:
                # allow one stall if excess magnitude improved a lot? check best_mc improvement
                # abort strictly per spec
                break
            viol_window_start = vc
    return best_X, best_mc

def _lp_polish(X, seconds):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    p_cap = 2500
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((X.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    HOMOTOPY_END = t0 + 145.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 123, 13
    Xcur = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                vv = _max_cos(a)
                if vv < best_val:
                    best_val = vv
                    Xcur = a.copy()
        except Exception:
            pass
    if Xcur is None:
        Xcur = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(Xcur)
    global_best = Xcur.copy()
    global_val = float(best_val)
    cur_base = Xcur.copy()

    targets = [0.2715, 0.2710, 0.2705, 0.2700]
    consec_fail = 0
    run_id = 0
    T_lse = 0.005
    w_lse = 2e-4
    for ti, t in enumerate(targets):
        if time.time() > HOMOTOPY_END:
            break
        # build starts: base, sigma 0.05, sigma 0.15, 2 random
        starts = []
        starts.append(("base", cur_base.copy(), 900, 0.005))
        # sigma 0.05 perturbation
        p1 = cur_base + 0.05 * rng.normal(size=(n, d))
        starts.append(("s05", _norm(p1), 800, 0.005))
        # sigma 0.15 perturbation
        p2 = cur_base + 0.15 * rng.normal(size=(n, d))
        starts.append(("s15", _norm(p2), 800, 0.007))
        # random starts (shorter)
        for k in range(2):
            rr = _norm(rng.normal(size=(n, d)))
            starts.append((f"rnd{k}", rr, 600, 0.008))
        target_best_X = None
        target_best_v = 1e9
        feasible = False
        for name, sX, steps, lr in starts:
            if time.time() > HOMOTOPY_END:
                break
            # per-run deadline: leave time for remaining starts
            t_dead = min(HOMOTOPY_END, HARD_END - 22.0)
            bx, bv = _adam_run(sX, t, steps, lr, T_lse, w_lse, run_id, t_dead)
            run_id += 1
            if bv < target_best_v:
                target_best_v = bv
                target_best_X = bx.copy()
            if bv < global_val:
                global_val = bv
                global_best = bx.copy()
            if bv <= t + 1e-12:
                feasible = True
                # don't break: still try other starts? break to save time and go lower
                break
            if time.time() > HOMOTOPY_END:
                break
        # update base for next target: use target_best (least-violating)
        if target_best_X is not None:
            # if target_best improved over cur_base, adopt it
            cur_base = target_best_X.copy()
        if feasible or (target_best_v < global_val + 1e-12 and target_best_v <= t + 5e-4):
            # consider success if feasible; otherwise check if we made progress toward t?
            # spec: stop after two consecutive failures (no feasible)
            if feasible:
                consec_fail = 0
            else:
                # if target_best_v significantly lower than before target, not a full failure?
                # define failure as not feasible
                consec_fail += 1
        else:
            consec_fail += 1
        if consec_fail >= 2:
            break
    # LP polish least-violating (global best / cur_base best) for <=20s
    remain = HARD_END - time.time()
    if remain > 2.0:
        lp_time = min(20.0, remain - 1.0)
        # polish the best violating config (cur_base if better max else global)
        cand = global_best.copy()
        # also consider cur_base
        try:
            v1 = _max_cos(cand)
            v2 = _max_cos(cur_base)
            if v2 < v1:
                cand = cur_base.copy()
        except Exception:
            pass
        try:
            polished = _lp_polish(cand, lp_time)
            vp = _max_cos(polished)
            if vp < global_val:
                global_val = vp
                global_best = polished.copy()
        except Exception:
            pass
    global_best = _norm(global_best)
    # final safety: replace any bad rows
    norms = np.linalg.norm(global_best, axis=1)
    bad = ~(np.isfinite(norms) & (norms > 1e-12))
    if np.any(bad):
        rr = rng.normal(size=(int(bad.sum()), d))
        global_best[bad] = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    return global_best
