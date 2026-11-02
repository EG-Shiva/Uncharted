import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _fast_quench(X, target, steps, lr0, lr1, rng=None, p_pow=4.0, topk=250):
    # Adam on tangent space with power / top-K repulsion. Pure numpy, fast.
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    l0, l1 = np.log(lr0), np.log(lr1)
    iu0 = np.triu_indices(n, 1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        lr = float(np.exp(l0 + (l1 - l0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        # active set: violations above target plus top-k to keep gradient alive
        # get threshold for topk quickly via partition on a copy of upper triangle
        c = G[iu0]
        if topk < c.size:
            kth = float(np.partition(c, -topk)[-topk])
        else:
            kth = float(c.min())
        thr = target if target < kth else target
        # use max(thr, kth - something)? ensure at least topk active:
        eff = thr
        if kth > thr:
            eff = kth
        D = G - eff
        pos = D > 0
        if not np.any(pos):
            # nothing to do
            break
        # power weights
        W = np.zeros_like(G)
        # scale to avoid overflow: normalize by max D
        dm = float(D[pos].max())
        if dm <= 0:
            break
        # W = (D/dm)^p  -> focuses on worst but keeps many active
        W[pos] = (D[pos] / dm) ** p_pow
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1; b2p *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        mh = m1 / (1.0 - b1p); vh = m2 / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        # clip update norm to avoid jumps
        X = X - lr * upd
        X = _norm(X)
    return X

def _hole_for(fixed, rng, n_try=8000):
    d = fixed.shape[1]
    Z = rng.standard_normal((n_try, d))
    Z = _norm(Z)
    S = Z @ fixed.T
    worst = S.max(axis=1)
    i = int(np.argmin(worst))
    return Z[i].copy(), float(worst[i])

def _individ_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _lp_polish(X, deadline, radius_init=2e-3, cap=1500):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < 12:
        it += 1
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            m2 = np.zeros_like(sel, dtype=bool)
            m2[idx[part]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_final = t0 + BUDGET - 3.0
    rng = np.random.default_rng(0)
    n, d = 82, 14
    # load start
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    if xb is None or _max_cos(a) < _max_cos(xb):
                        xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm(rng.standard_normal((n, d)))
    best_val = _max_cos(xb)
    X_best = xb.copy()

    it = 0
    # Phase 1: aggressive basin hopping with far-hole refill + fast quench
    # reserve ~25s for final polish
    hop_deadline = t0 + BUDGET - 28.0
    while time.time() < hop_deadline:
        it += 1
        if time.time() > hop_deadline:
            break
        # every 15th: fresh random restart quenched hard
        if it % 15 == 0:
            Xr = _norm(rng.standard_normal((n, d)))
            tgt = best_val - 4e-4
            Xr = _fast_quench(Xr, tgt, steps=380, lr0=0.04, lr1=0.003, topk=300, p_pow=3.0)
            # second stage with tighter target
            tgt2 = _max_cos(Xr) - 2e-4
            Xr = _fast_quench(Xr, tgt2, steps=250, lr0=0.02, lr1=0.002, topk=250, p_pow=5.0)
            v = _max_cos(Xr)
            if v < best_val - 1e-12:
                best_val = v; X_best = Xr.copy()
            continue
        # otherwise mutate best
        X = X_best.copy()
        cur_m = best_val
        # choose victims: worst individual maxima + random rattlers
        imax, G = _individ_max(X)
        order = np.argsort(-imax)  # descending worst first
        k = int(rng.integers(2, 7))  # 2..6 points replaced -> topology change
        # mix: half worst, half random
        nworst = max(1, k // 2)
        victims = list(order[:nworst])
        while len(victims) < k:
            c = int(rng.integers(0, n))
            if c not in victims:
                victims.append(c)
        # sequential far-hole replacement
        for vv in victims:
            mask = np.ones(n, dtype=bool); mask[vv] = False
            fixed = X[mask]
            h, _ = _hole_for(fixed, rng, n_try=6000)
            # small random tilt to avoid exact ties
            h = h + 1e-4 * rng.standard_normal(d)
            h /= np.linalg.norm(h)
            X[vv] = h
        X = _norm(X)
        # quench toward below-best target
        tgt = best_val - float(rng.uniform(2e-4, 6e-4))
        steps = int(rng.integers(280, 420))
        p_pow = float(rng.choice([3.0, 4.0, 5.0, 6.0]))
        Xq = _fast_quench(X, tgt, steps=steps, lr0=0.03, lr1=0.002, topk=280, p_pow=p_pow)
        v = _max_cos(Xq)
        # optional second short quench if close
        if v < best_val + 2e-4 and v >= best_val - 1e-12:
            # try one more push with tighter target and smaller lr
            Xq2 = _fast_quench(Xq, v - 2e-4, steps=180, lr0=0.012, lr1=0.0015, topk=200, p_pow=6.0)
            v2 = _max_cos(Xq2)
            if v2 < v:
                Xq, v = Xq2, v2
        if v < best_val - 1e-12:
            best_val = v; X_best = Xq.copy()
        # early exit if we already beat by a good margin? keep going to improve
        if it > 800:
            break

    # Phase 2: hill-climb single points of best (cheap, no LP)
    if time.time() < deadline_final - 14.0:
        # a few rounds of greedy single-point moves on worst points
        for _ in range(3):
            if time.time() > deadline_final - 14.0:
                break
            imax, Gfull = _individ_max(X_best)
            order = np.argsort(-imax)[:12]
            improved = False
            for idx in order:
                if time.time() > deadline_final - 14.0:
                    break
                cur = _max_cos(X_best)
                # try random tangent steps
                base = X_best[idx].copy()
                others = np.delete(X_best, idx, axis=0)
                best_v = base.copy(); best_w = float(np.max(others @ base))
                # if this point not among global worst, skip heavy search
                step = 0.01
                for t in range(120):
                    prop = base + step * rng.standard_normal(d)
                    # project to tangent then renormalize (approx random walk on sphere)
                    prop /= np.linalg.norm(prop)
                    w = float(np.max(others @ prop))
                    # global max may be elsewhere; compute full max only if local improves
                    if w < best_w:
                        # compute global
                        Xt = X_best.copy(); Xt[idx] = prop
                        g = float(_max_cos(Xt))
                        if g < cur - 1e-12:
                            X_best = Xt; cur = g; best_val = g
                            base = prop.copy(); best_w = w
                            improved = True
                            step *= 1.05
                        else:
                            # accept local improvement direction anyway with prob
                            base = prop.copy(); best_w = w
                    else:
                        step *= 0.98
                        if step < 2e-5:
                            break
            if not improved:
                break

    # Phase 3: final LP squeeze (time-capped)
    if time.time() < deadline_final:
        try:
            X_best = _lp_polish(X_best, deadline=min(deadline_final, time.time() + 12.0), radius_init=2e-3, cap=1500)
            best_val = _max_cos(X_best)
        except Exception:
            pass
    # ensure finite normalized
    X_best = _norm(X_best)
    return X_best
