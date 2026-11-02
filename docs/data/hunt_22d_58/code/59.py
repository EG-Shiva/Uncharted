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
    return float(G.max())

def _eval(X, beta=600.0):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    m = float(c.max())
    c066 = int((c > 0.066).sum())
    c064 = int((c > 0.064).sum())
    c060 = int((c > 0.060).sum())
    # stable softmax
    b = beta
    sm = float(np.log(np.exp(b * (c - m)).sum()) / b + m)
    return (c066, c064, c060, sm, m), m, c

def _lp_polish(X, seconds, deadline, max_pairs=600, max_iter=30, r0=6e-4):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > max_iter:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < max_pairs:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= max_pairs:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _get_stress(X):
    # one LP at current X, return per-point stress and pair info
    n, d = X.shape
    iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    m = float(c.max())
    radius = 6e-4
    window = max(20.0 * radius, 8e-5)
    order = np.argsort(-c)
    sel_list = []
    for idx in order:
        if c[idx] > m - window and len(sel_list) < 700:
            sel_list.append(int(idx))
        elif c[idx] <= m - window:
            break
        if len(sel_list) >= 700:
            break
    if len(sel_list) == 0:
        return np.zeros(n), None, None
    sel = np.array(sel_list)
    I = iu0[0][sel]; J = iu0[1][sel]
    p = len(sel); nv = n * d + 1
    try:
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs", options={"time_limit": 3.0})
    except Exception:
        res = None
    pair_w = None
    if res is not None and getattr(res, "status", 1) == 0:
        try:
            marg = None
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                try:
                    marg = np.asarray(res.ineqlin.marginals, dtype=float)
                except Exception:
                    try:
                        marg = np.asarray(res.ineqlin["marginals"], dtype=float)
                    except Exception:
                        marg = None
            if marg is not None and marg.shape[0] == p:
                pair_w = np.abs(marg)
                # zero out tiny
                pair_w[~np.isfinite(pair_w)] = 0.0
            else:
                pair_w = None
        except Exception:
            pair_w = None
    if pair_w is None or float(pair_w.sum()) <= 0:
        # fallback: exponential proximity to max
        pair_w = np.exp((c[sel] - m) / 2e-4)
    pt = np.zeros(n)
    np.add.at(pt, I, pair_w)
    np.add.at(pt, J, pair_w)
    return pt, (I, J, pair_w), m

def _refine_hole(c0, Y, steps=220, beta0=60.0, beta1=600.0, lr0=0.08, lr1=0.005):
    c = c0 / (np.linalg.norm(c0) + 1e-12)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ c
        m = float(dots.max())
        e = np.exp(beta * (dots - m))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            break
        w = e / se
        grad = w @ Y
        gt = grad - float(grad @ c) * c
        c = c - lr * gt
        n = float(np.linalg.norm(c))
        if n == 0 or not np.isfinite(n):
            c = c0 / (np.linalg.norm(c0) + 1e-12)
            continue
        c = c / n
    return c

def _greedy_reinsert(Xret, k, rng, n_cand=20000):
    # Xret: (n-k, d) retained; returns full (n,d) with k greedily inserted
    m0, d = Xret.shape
    Y = Xret.copy()
    for t in range(k):
        C = rng.standard_normal((n_cand, d))
        C = _norm(C)
        dots = C @ Y.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        # take top-5 candidates and refine each
        order = np.argpartition(worst, min(5, n_cand - 1))[:5]
        best_c = C[bi]
        best_w = float(worst[bi])
        for oi in order:
            rc = _refine_hole(C[int(oi)], Y, steps=220)
            wv = float((Y @ rc).max())
            if wv < best_w:
                best_w = wv
                best_c = rc
        Y = np.vstack([Y, best_c.reshape(1, d)])
    return Y

def _quench(X, steps, b0=60.0, b1=900.0, lr0=0.025, lr1=0.002, mu=0.80):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        V = mu * V + Gt
        step = lr * V
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > 0.05, step * (0.05 / sn), step)
        X = X - step
        X = _norm(X)
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None; best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = _norm(v)
            c = _max_cos(v)
            if c < best_val:
                best_val = c; best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d)); best = _norm(z); best_val = _max_cos(best)
    # light initial polish to reach local optimum quickly (capped)
    try:
        if time.time() < DEADLINE - 100:
            best = _lp_polish(best, seconds=4.0, deadline=DEADLINE - 100)
            best_val = _max_cos(best)
    except Exception:
        pass
    base = best.copy()
    base_tup, _, _ = _eval(base)
    best_max_X = base.copy()
    best_max = float(base_tup[4])
    # surgery loop
    k_cycle = [4, 5, 3, 6, 4, 5]
    att = 0
    while time.time() < DEADLINE - 12:
        if att >= 40:
            break
        k = k_cycle[att % len(k_cycle)]
        att += 1
        # stress from current base
        try:
            pt, _, _ = _get_stress(base)
        except Exception:
            pt = np.zeros(n)
        # rank hubs; add small randomization among top-10 to diversify
        order = np.argsort(-pt)
        # stochastic hub choice: always take top-2, sample rest from top-10
        top_pool = order[:10]
        # deterministic pseudo-random per attempt
        r = np.random.default_rng(9000 + att)
        if k <= 2:
            hubs = order[:k]
        else:
            keep_first = order[:2]
            rest_pool = [i for i in top_pool if i not in keep_first]
            need = k - 2
            # pick highest-stress among shuffled? bias to stress
            perm = r.permutation(len(rest_pool))
            # score = stress rank + noise: just take permuted then sort by stress with noise
            cand = np.array(rest_pool)[perm[:need * 2]]
            # rank cand by pt + noise
            noise = r.standard_normal(len(cand)) * (float(pt.max()) * 0.02 + 1e-9)
            sc = np.array([pt[i] for i in cand]) + noise
            chosen = cand[np.argsort(-sc)[:need]]
            hubs = np.concatenate([keep_first, chosen])
        hubs = np.unique(hubs)
        # if duplicates collapsed, fill
        if len(hubs) < k:
            for i in order:
                if i not in hubs:
                    hubs = np.append(hubs, i)
                if len(hubs) >= k:
                    break
        mask = np.ones(n, dtype=bool); mask[hubs] = False
        Xret = base[mask]
        # greedy reinsertion with >=20000 candidates
        try:
            Y = _greedy_reinsert(Xret, len(hubs), r, n_cand=20000)
        except Exception:
            continue
        if time.time() >= DEADLINE - 11:
            break
        # short annealed quench 200-400 steps (vary)
        steps = 260 if (att % 2 == 0) else 340
        try:
            Y = _quench(Y, steps=steps)
        except Exception:
            pass
        if time.time() >= DEADLINE - 10:
            # still consider Y without LP
            try:
                tup, mv, _ = _eval(_norm(Y))
                if mv < best_max:
                    best_max = mv; best_max_X = _norm(Y).copy()
            except Exception:
                pass
            break
        # capped LP polish 2.5s
        try:
            Yp = _lp_polish(Y, seconds=2.5, deadline=DEADLINE - 2, max_pairs=600, max_iter=14)
        except Exception:
            Yp = _norm(Y)
        try:
            tup, mv, _ = _eval(Yp)
        except Exception:
            continue
        # lexicographic acceptance vs base
        if tup < base_tup and mv < best_max + 0.0025:
            base = Yp.copy(); base_tup = tup
            if mv < best_max - 1e-12:
                best_max = mv; best_max_X = Yp.copy()
        else:
            # track pure max improvement regardless
            if mv < best_max - 1e-12:
                best_max = mv; best_max_X = Yp.copy()
                # adopt as base if close in lex (allow re-basining on max win)
                # only adopt if lex not catastrophically worse
                if tup[0] <= base_tup[0] + 40 and tup[1] <= base_tup[1] + 40:
                    base = Yp.copy(); base_tup = tup
        # early exit if big win
        if best_max < 0.064:
            break
    # final safety: return lowest-max found, normalized
    try:
        best_max_X = _norm(best_max_X)
        # tiny shrink not needed; just ensure finite
        if not np.all(np.isfinite(best_max_X)):
            best_max_X = _norm(base)
    except Exception:
        best_max_X = _norm(base)
    return best_max_X
