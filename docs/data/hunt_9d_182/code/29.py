import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _refine_batch(Y, C0, beta=500, steps=300, lr=0.015):
    # single-point tangent Adam on soft-max vs fixed set Y
    B, d = C0.shape
    X = _norm(C0.copy())
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    T = 1.0 / float(beta)
    t = 0
    for it in range(steps):
        S = X @ Y.T  # B x m
        mx = S.max(axis=1, keepdims=True)
        E = np.exp((S - mx) / T)
        s = E.sum(axis=1, keepdims=True)
        W = E / np.maximum(s, 1e-30)
        Gr = W @ Y  # B x d
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        t += 1
        m1 *= b1; m1 += (1 - b1) * Gt
        m2 *= b2; m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1.0 - b1 ** t)
        m2h = m2 / (1.0 - b2 ** t)
        step = m1h / (np.sqrt(m2h) + eps)
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        nm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 2.0 / nm)
        X = X - lr * step * sc
        X = _norm(X)
    return X

def _global_adam(X0, steps, beta0, beta1, lr0, lr1, deadline):
    X = _norm(X0.copy())
    n, d = X.shape
    try:
        bv = _max_cos(X)
    except Exception:
        bv = 1e9
    bx = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    glob = 0
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        T = 1.0 / beta
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if (it % 150) == 0 and m < bv:
            bv = m; bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob += 1
        m1 *= b1; m1 += (1 - b1) * Gt
        m2 *= b2; m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1.0 - b1 ** glob)
        m2h = m2 / (1.0 - b2 ** glob)
        step = m1h / (np.sqrt(m2h) + eps)
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, 2.0 / nm)
        X = X - lr * step * sc
        X = _norm(X)
    v = _max_cos(X)
    if v < bv:
        bv = v; bx = X.copy()
    return bx, bv

def _lp_wide(X0, seconds):
    t_end = time.time() + seconds
    x = _norm(X0.copy())
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    radius = 6e-4
    while time.time() < t_end and radius > 1e-13:
        Gf = x @ x.T
        cf = Gf[iu0, iu1]
        m = float(cf.max())
        thresh = m - max(20.0 * radius, 4e-7)
        sel = cf > thresh
        p = int(np.count_nonzero(sel))
        if p < 600:
            k = min(900, cf.size)
            idx = np.argpartition(-cf, k - 1)[:k]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = k
        if p > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-3000:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = 3000
        I = iu0[sel]; J = iu1[sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v; bx = y.copy(); x = y
                radius = min(radius * 1.4, 1.2e-3)
                improved = True
                break
        if not improved:
            radius *= 0.5
        if time.time() > t_end:
            break
    return bx, best

def solve():
    t_start = time.time()
    HARD = t_start + 172.0
    TRIAL_END = t_start + 160.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.RandomState(0).randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        rngf = np.random.RandomState(1)
        y = rngf.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182); mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    n, d = 182, 9
    iu0, iu1 = np.triu_indices(n, 1)
    sizes = [8, 12, 20]
    trial = 0
    NCAND = 30000
    NREF = 20
    while time.time() < TRIAL_END and time.time() < HARD - 8:
        k = sizes[trial % len(sizes)]
        rng = np.random.RandomState(5000 + trial * 977)
        # seed point among top 100 pairs
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -10.0)
            flat = G[iu0, iu1]
            top100 = np.argpartition(-flat, 100)[:100]
            pts = np.concatenate([iu0[top100], iu1[top100]])
            uniq = np.unique(pts)
            seed = int(rng.choice(uniq))
        except Exception:
            seed = int(rng.randint(0, n))
        # neighbours above 0.30
        try:
            row = (x_best @ x_best[seed])
            row[seed] = -10.0
            order = np.argsort(-row)
            near = [int(i) for i in order if float(row[i]) > 0.30]
        except Exception:
            near = []
        need = k - 1
        del_list = [seed]
        # take near neighbours first
        for idx in near:
            if len(del_list) >= k:
                break
            if idx not in del_list:
                del_list.append(idx)
        # fill remainder with random
        while len(del_list) < k:
            c = int(rng.randint(0, n))
            if c not in del_list:
                del_list.append(c)
        # ensure a few random: replace last 2 (if k large) with random when near filled all
        if len(near) >= need and k >= 8:
            for _ in range(2):
                c = int(rng.randint(0, n))
                if c not in del_list and c != seed:
                    del_list[-1 - _] = c
        del_arr = np.array(sorted(del_list), dtype=int)
        mask = np.ones(n, dtype=bool); mask[del_arr] = False
        cur = x_best[mask].copy()  # (n-k, d)
        # sequential reinsertion
        ok = True
        for ins in range(k):
            if time.time() > TRIAL_END or time.time() > HARD - 8:
                ok = False
                break
            R = rng.randn(NCAND, d)
            R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-30)
            S = R @ cur.T
            smax = S.max(axis=1)
            best20_idx = np.argpartition(smax, NREF - 1)[:NREF]
            C0 = R[best20_idx]
            try:
                Cref = _refine_batch(cur, C0, beta=500, steps=300, lr=0.015)
            except Exception:
                Cref = C0
            S2 = Cref @ cur.T
            s2max = S2.max(axis=1)
            wi = int(np.argmin(s2max))
            chosen = Cref[wi]
            # true max check vs argpartition best (guard refinement failure)
            if float(s2max[wi]) > float(np.min(smax)):
                # keep refined choice anyway (smallest true max among refined)
                pass
            cur = np.vstack([cur, chosen.reshape(1, d)])
            cur = _norm(cur)
        if not ok:
            trial += 1
            continue
        # cur is (182,d) full trial config
        trialX = _norm(cur.copy())
        # short global Adam beta 200->1500
        remain = min(TRIAL_END - time.time(), HARD - time.time() - 4.5)
        if remain > 1.0:
            dl = time.time() + min(remain - 3.2 if remain > 4.2 else remain * 0.5, 6.0)
            if dl > time.time() + 0.3:
                try:
                    trialX2, v2 = _global_adam(trialX, steps=1200, beta0=200, beta1=1500,
                                              lr0=0.002, lr1=0.0003, deadline=dl)
                    trialX = trialX2
                except Exception:
                    pass
        # at most 3s wide-window LP
        remain2 = min(TRIAL_END - time.time(), HARD - time.time() - 1.0)
        if remain2 > 0.6:
            secs = min(3.0, remain2)
            try:
                trialX3, v3 = _lp_wide(trialX, seconds=secs)
                trialX = trialX3
            except Exception:
                pass
        try:
            v = _max_cos(trialX)
        except Exception:
            trial += 1
            continue
        if v < best_val - 1e-12:
            best_val = v
            x_best = _norm(trialX.copy())
        trial += 1
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
