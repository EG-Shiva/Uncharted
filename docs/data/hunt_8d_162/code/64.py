import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _smooth_descend(X, betas, iters_list, lr0=0.12, mu=0.85, deadline=None):
    n, d = X.shape
    X = _normalize(X.copy())
    delta = np.zeros_like(X)
    lr = float(lr0)
    for bi, beta in enumerate(betas):
        if deadline is not None and time.time() >= deadline - 1.0:
            break
        iters = iters_list[bi] if bi < len(iters_list) else iters_list[-1]
        for k in range(iters):
            if deadline is not None and (k % 40 == 0) and time.time() >= deadline - 1.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if S < 1e-300:
                S = 1e-300
            L = m + np.log(max(S, 1e-300)) / beta
            row = W.sum(axis=1, keepdims=True)
            row = np.maximum(row, 1e-30)
            Gavg = (W @ X) / row
            dot = (Gavg * X).sum(axis=1, keepdims=True)
            Gt = Gavg - dot * X
            delta = mu * delta - lr * Gt
            ddot = (delta * X).sum(axis=1, keepdims=True)
            dt = delta - ddot * X
            Xt = _normalize(X + dt)
            Gn = Xt @ Xt.T
            np.fill_diagonal(Gn, -2.0)
            mn = float(Gn.max())
            try:
                Sn = float(np.exp(beta * (Gn - mn)).sum() * 0.5)
            except Exception:
                Sn = 1e-300
            if Sn < 1e-300:
                Sn = 1e-300
            Ln = mn + np.log(max(Sn, 1e-300)) / beta
            if Ln < L + 1e-12:
                X = Xt
                lr *= 1.03
                if lr > 1.0:
                    lr = 1.0
            else:
                lr *= 0.5
                delta *= 0.2
                if lr < 1e-5:
                    lr = 1e-5
        delta *= 0.5
        if lr < 1e-3:
            lr = 1e-3
        elif lr > 0.5:
            lr = 0.5
    return X

def _lp_polish(X, seconds, deadline=None, r0=1e-3, rmax=0.05):
    X = _normalize(X.copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = float(r0)
    t_start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while True:
        now = time.time()
        if now - t_start >= seconds:
            break
        if deadline is not None and now >= deadline - 1.0:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.argpartition(-cf, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]
        c = cf[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([(I[:, None] * d + np.arange(d)).ravel(),
                                   (J[:, None] * d + np.arange(d)).ravel(),
                                   np.full(p, n * d)] if False else [np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()], axis=0)
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        Y = _normalize(X + dx)
        v = _max_cos(Y)
        if v < best - 1e-15:
            X = Y
            best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        if time.time() - t_start >= seconds:
            break
        if deadline is not None and time.time() >= deadline - 1.0:
            break
    return X

def _refine_hole(c0, Xk, iters=350):
    c = c0 / max(np.linalg.norm(c0), 1e-30)
    try:
        best = float((Xk @ c).max())
    except Exception:
        return c
    lr = 0.03
    for _ in range(iters):
        s = Xk @ c
        j = int(np.argmax(s))
        dot = float(s[j])
        g = Xk[j] - dot * c
        cn = c - lr * g
        nrm = float(np.linalg.norm(cn))
        if not np.isfinite(nrm) or nrm < 1e-30:
            break
        cn = cn / nrm
        sn = float((Xk @ cn).max())
        if sn < best:
            c = cn
            best = sn
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return c

def _find_hole(Xk, rng, n_cand=50000):
    n, d = Xk.shape
    try:
        C = rng.standard_normal((n_cand, d))
    except Exception:
        C = np.random.standard_normal((n_cand, d))
    C = _normalize(C)
    # batched scoring to bound memory
    best_scores = None
    Smax = None
    bs = 10000
    scores = np.empty(n_cand, dtype=float)
    for a in range(0, n_cand, bs):
        b = min(a + bs, n_cand)
        S = C[a:b] @ Xk.T
        scores[a:b] = S.max(axis=1)
    k0 = min(8, n_cand)
    top = np.argpartition(scores, k0 - 1)[:k0]
    # order top by score
    top = top[np.argsort(scores[top])]
    best_c = C[top[0]]
    best_v = float(scores[top[0]])
    for t in top:
        cr = _refine_hole(C[t], Xk, iters=350)
        v = float((Xk @ cr).max())
        if v < best_v:
            best_v = v
            best_c = cr
    return best_c / max(np.linalg.norm(best_c), 1e-30), best_v

def solve():
    np.random.seed(0)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (162, 8):
                a = _normalize(a)
                # filter bad
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        cands.append(_normalize(rng0.standard_normal((162, 8))))
    best = None
    best_val = 1e9
    for a in cands:
        try:
            v = _max_cos(a)
        except Exception:
            continue
        if v < best_val:
            best_val = v
            best = a.copy()
    if best is None:
        rng0 = np.random.default_rng(0)
        best = _normalize(rng0.standard_normal((162, 8)))
        best_val = _max_cos(best)
    global_best = best.copy()
    global_val = float(best_val)

    # quick initial LP touch if time allows (already polished basin, keep short)
    # skip long initial polish to save time for continuation

    ks_cycle = [12, 8, 20, 4, 12, 20, 8, 4, 12, 8, 20, 4]
    trial = 0
    max_trials = 12
    while time.time() < DEADLINE - 16.0 and trial < max_trials:
        k = int(ks_cycle[trial % len(ks_cycle)])
        try:
            G = global_best @ global_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            tight = G > m - 1e-7
            cnt = tight.sum(axis=1)
        except Exception:
            trial += 1
            continue
        order_hi = np.argsort(-cnt)
        order_lo = np.argsort(cnt)
        trial_rng = np.random.default_rng(2000 + trial)
        ins_rng = np.random.default_rng(5000 + trial * 131)
        victims = set()
        try:
            victims.add(int(order_lo[0]))
        except Exception:
            victims.add(int(trial_rng.integers(0, 162)))
        # always include top hubs to rewire overload
        try:
            victims.add(int(order_hi[0]))
            if k > 1:
                victims.add(int(order_hi[1]))
        except Exception:
            pass
        pool = [int(x) for x in order_hi[:25]]
        guard = 0
        while len(victims) < k and guard < 200:
            guard += 1
            r = float(trial_rng.random())
            if r < 0.8:
                cand = int(pool[int(trial_rng.integers(0, len(pool)))])
            else:
                cand = int(trial_rng.integers(0, 162))
            victims.add(cand)
        if len(victims) < k:
            while len(victims) < k:
                victims.add(int(trial_rng.integers(0, 162)))
        victims_sorted = sorted(list(victims))[:k]
        try:
            mask = np.ones(162, dtype=bool)
            mask[victims_sorted] = False
            X_keep = global_best[mask].copy()
        except Exception:
            trial += 1
            continue
        # optimize remaining 162-k points with staged repulsion + LP
        try:
            X_keep = _smooth_descend(X_keep, [60.0, 150.0, 300.0], [120, 120, 120], lr0=0.12, mu=0.85, deadline=DEADLINE)
            if time.time() < DEADLINE - 10.0:
                X_keep = _lp_polish(X_keep, seconds=1.2, deadline=DEADLINE)
        except Exception:
            trial += 1
            continue
        X_cur = X_keep.copy()
        # reinsert one at a time
        broken = False
        for ii in range(k):
            if time.time() >= DEADLINE - 13.0:
                broken = True
                break
            try:
                c_new, _ = _find_hole(X_cur, ins_rng, n_cand=50000)
            except Exception:
                try:
                    v = ins_rng.standard_normal(8); v /= np.linalg.norm(v)
                    c_new = v
                except Exception:
                    broken = True
                    break
            try:
                X_cur = np.vstack([X_cur, c_new.reshape(1, 8)])
                X_cur = _normalize(X_cur)
            except Exception:
                broken = True
                break
            # short smooth + short LP after each insertion (allow temporary worsening)
            try:
                X_cur = _smooth_descend(X_cur, [80.0, 200.0], [60, 60], lr0=0.08, mu=0.85, deadline=DEADLINE)
            except Exception:
                pass
            try:
                if time.time() < DEADLINE - 8.0:
                    X_cur = _lp_polish(X_cur, seconds=0.35, deadline=DEADLINE)
            except Exception:
                pass
            # retain best feasible 162-point config whenever seen
            if X_cur.shape[0] == 162:
                try:
                    v = _max_cos(X_cur)
                    if v < global_val - 1e-12 and np.all(np.isfinite(X_cur)):
                        global_best = X_cur.copy()
                        global_val = float(v)
                except Exception:
                    pass
        if broken:
            trial += 1
            continue
        # thorough polish of completed 162 code
        if X_cur.shape[0] == 162:
            try:
                if time.time() < DEADLINE - 8.0:
                    X_cur = _smooth_descend(X_cur, [150.0, 350.0], [100, 100], lr0=0.06, mu=0.85, deadline=DEADLINE)
                if time.time() < DEADLINE - 6.0:
                    X_cur = _lp_polish(X_cur, seconds=1.2, deadline=DEADLINE)
                v = _max_cos(X_cur)
                if v < global_val - 1e-12 and np.all(np.isfinite(X_cur)):
                    global_best = X_cur.copy()
                    global_val = float(v)
            except Exception:
                pass
        trial += 1
    # final long polish of incumbent best
    try:
        remain = DEADLINE - time.time()
        if remain > 2.5:
            sec = min(10.0, max(2.0, remain - 1.0))
            global_best = _lp_polish(global_best, seconds=sec, deadline=DEADLINE)
            global_val = _max_cos(global_best)
    except Exception:
        pass
    try:
        global_best = _normalize(global_best)
    except Exception:
        pass
    # crash-proof fallback
    try:
        if global_best.shape != (162, 8) or not np.all(np.isfinite(global_best)):
            raise ValueError("bad shape")
        # ensure no zero rows
        nr = np.linalg.norm(global_best, axis=1)
        if np.any(nr < 1e-12):
            raise ValueError("zero row")
    except Exception:
        try:
            fb = np.load("best.npy")
            fb = np.asarray(fb, dtype=float)
            if fb.shape == (162, 8):
                return _normalize(fb)
        except Exception:
            pass
        rngf = np.random.default_rng(0)
        return _normalize(rngf.standard_normal((162, 8)))
    return global_best
