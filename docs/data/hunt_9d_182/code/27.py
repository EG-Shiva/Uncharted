import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-16] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _single_refine(y0, Others, beta=500.0, iters=300, lr=0.03):
    y = np.asarray(y0, dtype=float).copy()
    nn = float(np.linalg.norm(y))
    if not np.isfinite(nn) or nn < 1e-16:
        y = Others[0].copy()
        nn = float(np.linalg.norm(y))
    y = y / nn
    m1 = np.zeros_like(y)
    m2 = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_y = y.copy()
    try:
        best_s = float(np.max(Others @ y))
    except Exception:
        best_s = 1e9
    c1p = 1.0
    c2p = 1.0
    for t in range(1, iters + 1):
        dots = Others @ y
        m0 = float(np.max(dots))
        W = np.exp(beta * (dots - m0))
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        grad = (W @ Others) / S
        gt = grad - float(grad @ y) * y
        c1p *= b1
        c2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        mhat = m1 / (1.0 - c1p)
        vhat = m2 / (1.0 - c2p)
        st = mhat / (np.sqrt(vhat) + eps)
        st = st - float(st @ y) * y
        lrt = lr / (1.0 + t / 250.0)
        y = y - lrt * st
        nn = float(np.linalg.norm(y))
        if not np.finite(nn) or nn < 1e-16:
            break
        y = y / nn
        if (t % 25) == 0 or t == iters:
            s = float(np.max(Others @ y))
            if s < best_s:
                best_s = s
                best_y = y.copy()
    try:
        s2 = float(np.max(Others @ y))
        if s2 < best_s:
            return y, s2
    except Exception:
        pass
    return best_y, best_s

def _global_adam(X0, steps=450, beta0=200.0, beta1=1500.0, lr=0.004, deadline=1e18):
    X = _norm(X0.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    bestv = _max_cos(X)
    best = X.copy()
    t = 0
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        t += 1
        b1p *= b1
        b2p *= b2
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp(beta * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1p)
        vhat = m2 / (1.0 - b2p)
        upd = mhat / (np.sqrt(vhat) + eps)
        dn = np.sum(upd * X, axis=1, keepdims=True)
        upd = upd - dn * X
        X = X - lr * upd
        X = _norm(X)
        if (s % 100) == 99:
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                best = X.copy()
    v = _max_cos(X)
    if v < bestv:
        return X, v
    return best, bestv

def _lp_wide(X0, seconds, deadline):
    X = _norm(X0.copy())
    best = _max_cos(X)
    bestX = X.copy()
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 8e-4
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(30.0 * radius, 8e-7)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt < 800:
            kk = min(1200, c.size)
            idxp = np.argpartition(-c, kk - 1)[:kk]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idxp] = True
            sel = mask
            cnt = kk
        if cnt > 4500:
            idxp = np.argpartition(-c, 4500)[:4500]
            # keep only those above thr among top? just take top 4500 intersect
            mask = np.zeros_like(sel, dtype=bool)
            mask[idxp] = True
            sel = sel & mask
            # if too few, use top
            if int(np.count_nonzero(sel)) < 500:
                sel = mask
            cnt = int(np.count_nonzero(sel))
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, ci)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            Y = X + alpha * dx
            Y = _norm(Y)
            v = _max_cos(Y)
            if v < best - 1e-12:
                best = v
                bestX = Y.copy()
                X = Y
                radius = min(radius * 1.5, 0.015)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return bestX, best

def _cluster_trial(Xbest, bestv, rng, k_total, deadline):
    n, d = Xbest.shape
    G = Xbest @ Xbest.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(n, 1)
    c = G[iu]
    # top 100 pairs
    kk = min(100, c.size)
    idx = np.argpartition(-c, kk - 1)[:kk]
    I = iu[0][idx]
    J = iu[1][idx]
    pool = np.unique(np.concatenate([I, J]))
    seed = int(rng.choice(pool))
    row = Xbest @ Xbest[seed]
    row[seed] = -2.0
    neigh = np.where(row > 0.30)[0]
    # shuffle neighbours
    perm = rng.permutation(len(neigh))
    neigh = neigh[perm]
    n_rand = 2 if k_total >= 12 else 1
    need = k_total - 1
    take_nbr = min(len(neigh), need - n_rand)
    if take_nbr < 0:
        take_nbr = 0
        n_rand = need
    chosen = neigh[:take_nbr].astype(int) if take_nbr > 0 else np.zeros(0, dtype=int)
    remaining = need - take_nbr
    exclude = np.zeros(n, dtype=bool)
    exclude[seed] = True
    exclude[chosen] = True
    avail = np.where(~exclude)[0]
    if remaining > 0:
        if remaining >= len(avail):
            rand_pts = avail
        else:
            ri = rng.choice(len(avail), size=remaining, replace=False)
            rand_pts = avail[ri]
    else:
        rand_pts = np.zeros(0, dtype=int)
    del_idx = np.concatenate([np.array([seed], dtype=int), chosen, rand_pts]).astype(int)
    mask = np.ones(n, dtype=bool)
    mask[del_idx] = False
    Cur = Xbest[mask].copy()
    # sequential reinsertion
    n_cand = 30000
    for _ in range(k_total):
        if time.time() > deadline - 6.0:
            return None, 1e9
        R = rng.standard_normal(size=(n_cand, d))
        R = _norm(R)
        # score in chunks to bound memory
        best_smax = np.empty(n_cand, dtype=float)
        bs = 10000
        for a in range(0, n_cand, bs):
            b = min(n_cand, a + bs)
            S = R[a:b] @ Cur.T
            best_smax[a:b] = np.max(S, axis=1)
        order = np.argpartition(best_smax, 20)[:20]
        # order by smax
        o2 = order[np.argsort(best_smax[order])]
        best_y = None
        best_s = 1e9
        for tii in range(20):
            y0 = R[o2[tii]]
            y, _ = _single_refine(y0, Cur, beta=500.0, iters=300, lr=0.03)
            s = float(np.max(Cur @ y))
            if s < best_s:
                best_s = s
                best_y = y.copy()
        Cur = np.vstack([Cur, best_y[None, :]])
    # short global Adam relaxation beta 200 -> 1500
    try:
        Cur2, v2 = _global_adam(Cur, steps=400, beta0=200.0, beta1=1500.0, lr=0.004, deadline=deadline - 3.5)
    except Exception:
        Cur2, v2 = Cur, _max_cos(Cur)
    # at most 3 seconds of wide-window LP
    rem_lp = 3.0
    # ensure we leave time
    if time.time() + rem_lp + 0.5 > deadline:
        rem_lp = max(0.5, deadline - time.time() - 0.5)
    try:
        Cur3, v3 = _lp_wide(Cur2, seconds=min(3.0, rem_lp), deadline=min(deadline, time.time() + min(3.0, rem_lp) + 0.8))
    except Exception:
        Cur3, v3 = Cur2, v2
    return Cur3, min(v2, v3) if 'v3' in locals() else v2

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    end_surgery = t0 + 160.0
    rng = np.random.default_rng(20261103)
    Xbest = None
    bestv = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            Y = np.load(fname)
            Y = np.asarray(Y, dtype=float)
            if Y.shape != (182, 9):
                continue
            Y = _norm(Y)
            if np.any(~np.isfinite(Y)):
                continue
            v = _max_cos(Y)
            if v < bestv:
                bestv = v
                Xbest = Y.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = _norm(rng.standard_normal(size=(182, 9)))
        bestv = _max_cos(Xbest)
    sizes = [8, 12, 20]
    trial = 0
    while time.time() < end_surgery - 3.0:
        trial += 1
        rem = end_surgery - time.time()
        if rem < 5.0:
            break
        k = sizes[(trial - 1) % len(sizes)]
        # vary seed ordering deterministically via rng
        if rem < 10.0 and k > 8:
            k = 8
        try:
            Y, v = _cluster_trial(Xbest, bestv, rng, k, deadline=end_surgery)
            if Y is not None and np.isfinite(v) and v < bestv - 1e-12:
                # verify
                Y = _norm(Y)
                vv = _max_cos(Y)
                if vv < bestv:
                    Xbest = Y.copy()
                    bestv = vv
        except Exception:
            continue
    return _norm(Xbest)
