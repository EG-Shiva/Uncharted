import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    return _normalize(x)

def _soft_adam(x, steps, tau0, tau1, lr0, lr1, beta1=0.9, beta2=0.999, eps=1e-8):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    b1p = 1.0
    b2p = 1.0
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-12)
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        g = grad - dot * x
        b1p *= beta1
        b2p *= beta2
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta2 * m2 + (1.0 - beta2) * (g * g)
        m1h = m1 / (1.0 - b1p)
        m2h = m2 / (1.0 - b2p)
        u = m1h / (np.sqrt(m2h) + eps)
        # re-project Adam step to tangent
        ud = np.sum(u * x, axis=1, keepdims=True)
        u = u - ud * x
        # clip extreme
        un = np.linalg.norm(u, axis=1, keepdims=True)
        # Adam steps ~ O(1); clip at 3
        scale = np.minimum(1.0, 3.0 / np.maximum(un, 1e-12))
        u = u * np.where(un > 3.0, scale, 1.0)
        x = x - lr * u
        x = _normalize(x)
    return x

def _hard_descent(x, steps, lr0, lr1, marg0, marg1, mom=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for t in range(steps):
        f = t / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * f
        marg = marg0 + (marg1 - marg0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > (m - marg)).astype(np.float64)
        # symmetrize (already symmetric) zero diag
        deg = np.sum(Mask, axis=1, keepdims=True)
        grad = (Mask @ x) / np.maximum(deg, 1.0)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        g = grad - dot * x
        # only move points that have contacts; others get zero
        has = (deg[:, 0] > 0.5)
        g = g * has[:, None]
        vel = mom * vel + (1.0 - mom) * g
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _normalize(x)
    return x

def _refine_hole(y, X, steps=200, tau0=0.02, tau1=0.002, lr0=0.08, lr1=0.008):
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        d = X @ y
        m = float(np.max(d))
        w = np.exp((d - m) / tau)
        s = float(np.sum(w))
        if s <= 0:
            break
        grad = (w @ X) / s
        grad = grad - float(np.dot(grad, y)) * y
        y = y - lr * grad
        ny = float(np.linalg.norm(y))
        if ny == 0:
            break
        y = y / ny
    return y

def _find_hole(X, rng, n_sample=12000, n_best=4):
    d = X.shape[1]
    Y = _random_sphere(n_sample, d, rng)
    S = Y @ X.T
    vals = np.max(S, axis=1)
    idx = np.argpartition(vals, n_best - 1)[:n_best]
    best_y = None
    best_v = 1e9
    for k in idx:
        y = Y[k].copy()
        y = _refine_hole(y, X)
        v = float(np.max(X @ y))
        if v < best_v:
            best_v = v
            best_y = y.copy()
    return best_y, best_v

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
        if p0 > 2500:
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
            p0 = 2500
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x, best

def _degrees(x, thresh):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > thresh, axis=1), G

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    POlISH_RESERVE = 32.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = _normalize(a)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    # quick initial LP to lock baseline (2s)
    try:
        if _time.time() < HARD_END - POlISH_RESERVE - 10:
            bx, bv = _lp_polish(best_x, min(_time.time() + 2.0, HARD_END - POlISH_RESERVE))
            if bv < best_v:
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
    except Exception:
        pass

    it = 0
    tol0 = 0.004
    while True:
        if _time.time() > HARD_END - POlISH_RESERVE:
            break
        # time guard per iter: need at least ~0.5s
        if _time.time() > HARD_END - POlISH_RESERVE - 0.5:
            break
        mode = it % 10
        # choose base and perturbation
        if mode <= 2:
            # small jitter around cur/best
            base = cur_x if (it % 3 != 2) else best_x
            sigma = [0.008, 0.02, 0.045][mode]
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 = _normalize(x0)
            steps = 1400
            tau0, tau1 = 0.03, 0.0025
            lr0, lr1 = 0.06, 0.003
        elif mode == 3 or mode == 4:
            base = cur_x if (rng.random() < 0.7) else best_x
            sigma = 0.10 if mode == 3 else 0.20
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 = _normalize(x0)
            # contact-guided extra kick on worst points
            try:
                G = x0 @ x0.T
                np.fill_diagonal(G, -2.0)
                worst = np.max(G, axis=1)
                k = 8
                widx = np.argpartition(worst, -k)[-k:]
                x0[widx] += 0.25 * rng.standard_normal(size=(k, d))
                x0 = _normalize(x0)
            except Exception:
                pass
            steps = 1800
            tau0, tau1 = 0.04, 0.003
            lr0, lr1 = 0.08, 0.003
        elif mode == 5:
            # large kick
            base = best_x if (rng.random() < 0.5) else cur_x
            sigma = 0.45
            x0 = base + sigma * rng.standard_normal(size=(n, d))
            x0 = _normalize(x0)
            # reshuffle few points
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            rp = _random_sphere(k, d, rng)
            x0[idx] = rp
            steps = 2200
            tau0, tau1 = 0.055, 0.0035
            lr0, lr1 = 0.09, 0.003
        elif mode == 6:
            # fresh random with medium anneal
            x0 = _random_sphere(n, d, rng)
            steps = 2400
            tau0, tau1 = 0.06, 0.004
            lr0, lr1 = 0.10, 0.004
        else:
            # hole-insertion modes 7,8,9
            base = best_x if (rng.random() < 0.6) else cur_x
            x0 = base.copy()
            # decide victims
            try:
                G = x0 @ x0.T
                np.fill_diagonal(G, -2.0)
                m0 = float(np.max(G))
                worst = np.max(G, axis=1)
                deg, _ = _degrees(x0, m0 - 1e-3)
                ratt = np.where(deg == 0)[0]
                if mode == 7:
                    # replace worst point(s) by hole
                    k = 1 if rng.random() < 0.7 else 2
                    victims = np.argpartition(worst, -k)[-k:]
                elif mode == 8:
                    # replace rattler / low-degree by hole
                    if len(ratt) > 0:
                        k = min(len(ratt), 1 if rng.random() < 0.7 else 2)
                        victims = rng.choice(ratt, size=k, replace=False)
                    else:
                        # lowest degree
                        k = 1 if rng.random() < 0.7 else 2
                        victims = np.argpartition(deg, k - 1)[:k]
                        # with 50% instead pick random
                        if rng.random() < 0.4:
                            victims = rng.choice(n, size=k, replace=False)
                else:
                    # replace random 1-3 by holes
                    k = int(rng.integers(1, 4))
                    victims = rng.choice(n, size=k, replace=False)
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                Xkeep = x0[keep_mask]
                for v_i in victims:
                    # hole w.r.t. current keep set (update as we fill)
                    hy, _ = _find_hole(Xkeep, rng, n_sample=8000, n_best=3)
                    x0[v_i] = hy
                    # update keep for next victim (append)
                    Xkeep = np.concatenate([Xkeep, hy[None, :]], axis=0)
                # small jitter others to escape
                x0 = x0 + 0.015 * rng.standard_normal(size=(n, d))
                x0 = _normalize(x0)
            except Exception:
                x0 = base + 0.05 * rng.standard_normal(size=(n, d))
                x0 = _normalize(x0)
            steps = 1800
            tau0, tau1 = 0.035, 0.0025
            lr0, lr1 = 0.07, 0.003

        # smooth Adam
        x1 = _soft_adam(x0, steps, tau0, tau1, lr0, lr1)
        # hard max descent to snap to minimax
        x1 = _hard_descent(x1, 250, 0.008, 0.001, 0.015, 0.002)
        v1 = _max_cos(x1)

        # short LP if promising
        do_polish = (v1 < best_v + 0.0035)
        if do_polish and _time.time() < HARD_END - POlISH_RESERVE - 1.0:
            dl = min(_time.time() + 0.9, HARD_END - POlISH_RESERVE)
            if dl > _time.time() + 0.15:
                try:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
                except Exception:
                    pass

        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        # walker acceptance with decaying tolerance
        frac = min(1.0, it / 120.0)
        tol = tol0 * (1.0 - frac) + 0.0004
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.008:
                # reset wandering walker
                if rng.random() < 0.7:
                    cur_x = best_x.copy()
                    cur_v = best_v
        else:
            if rng.random() < 0.12:
                cur_x = best_x.copy()
                cur_v = best_v
        it += 1
        # safety: cap iterations to avoid infinite fast loop
        if it > 400:
            break

    # final thorough polish + small ensemble around best
    try:
        # polish best first
        best_x, best_v = _lp_polish(best_x, HARD_END - 6.0)
    except Exception:
        pass
    # try a few jittered re-polishes if time left
    try:
        for _ in range(6):
            if _time.time() > HARD_END - 1.5:
                break
            cand = best_x + 0.004 * rng.standard_normal(size=(n, d))
            cand = _normalize(cand)
            cand = _hard_descent(cand, 150, 0.004, 0.0005, 0.008, 0.002)
            dl = min(_time.time() + 1.0, HARD_END - 0.5)
            if dl <= _time.time() + 0.1:
                break
            cand, cv = _lp_polish(cand, dl)
            if cv < best_v:
                best_x, best_v = cand, cv
    except Exception:
        pass
    best_x = _normalize(best_x)
    return best_x
