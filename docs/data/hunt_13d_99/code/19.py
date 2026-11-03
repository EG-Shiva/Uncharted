import time
import numpy as np

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), float(g.max())

def _smooth_descent(x_init, rng, steps_list, lr0, deadline, noise0=0.0):
    # steps_list = [(k, steps), ...]
    X = x_init.copy()
    n, d = X.shape
    best = _max_cos(X)
    Xbest = X.copy()
    vel = np.zeros_like(X)
    mu = 0.9
    t_global = 0
    for (k, steps) in steps_list:
        for t in range(steps):
            if time.time() > deadline:
                break
            # Gram
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            # weights, avoid overflow
            E = np.exp(k * (G - m))
            np.fill_diagonal(E, 0.0)
            S = E.sum() * 0.5
            if S < 1e-300:
                S = 1e-300
            Gw = (E @ X) / S
            # tangent projection
            dot = (Gw * X).sum(axis=1, keepdims=True)
            Gt = Gw - dot * X
            # momentum
            vel = mu * vel + (1.0 - mu) * Gt
            # lr schedule
            lr = lr0 / (1.0 + t_global * 0.002)
            # annealed noise
            if noise0 > 0:
                ns = noise0 / (1.0 + t_global * 0.005)
                vel = vel + ns * rng.randn(n, d) * 0.1
            Xn = X - lr * vel
            # project noise tangent? just normalize
            Xn = _normalize(Xn)
            # quick eval every few steps
            if (t & 7) == 0:
                v = _max_cos(Xn)
                if v < best:
                    best = v
                    Xbest = Xn.copy()
                    # accelerate: reset vel partially
                    # keep going
                # if blow up, pull back
                if v > m + 0.05 and t_global > 50:
                    lr0 *= 0.9
                    vel *= 0.5
                    continue
            X = Xn
            t_global += 1
        if time.time() > deadline:
            break
    # return best encountered relaxed a bit more with tiny steps? just return Xbest
    return Xbest, best

def _refine_single(v0, others, rng, iters=200):
    v = v0 / max(np.linalg.norm(v0), 1e-15)
    best_v = v.copy()
    best_m = float((others @ v).max())
    for it in range(iters):
        dts = others @ v
        m = float(dts.max())
        if m < best_m:
            best_m = m
            best_v = v.copy()
        margin = 0.02
        mask = dts > m - margin
        if not np.any(mask):
            break
        grad = others[mask].sum(axis=0)
        # tangent
        grad = grad - (grad @ v) * v
        gn = np.linalg.norm(grad)
        if gn < 1e-12:
            break
        grad /= gn
        step = 0.02 / (1.0 + it * 0.03)
        v = v - step * grad
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        v /= nv
    return best_v, best_m

def _reinsert_indices(x, idx_list, rng, n_try=8000):
    n, d = x.shape
    x = x.copy()
    for ii in idx_list:
        others_mask = np.ones(n, dtype=bool)
        others_mask[ii] = False
        # also exclude other indices to be replaced that not yet done? keep current values for them
        # use all others as obstacles
        others = x[others_mask]
        cand = rng.randn(n_try, d)
        cand = _normalize(cand)
        dots = cand @ others.T
        s = dots.max(axis=1)
        # take few best and refine
        order = np.argsort(s)[:8]
        bv = None
        bm = 1e9
        for o in order:
            v0 = cand[int(o)]
            v_opt, m_opt = _refine_single(v0, others, rng, iters=150)
            if m_opt < bm:
                bm = m_opt
                bv = v_opt
        x[ii] = bv
    return x

def _lp_polish(x, time_budget, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x, _max_cos(x)
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < time_budget and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t_start = time.time()
    deadline = t_start + 165.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(99, 13)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        flat = xb.ravel()
        need = 99 * 13
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(99, 13)
    xb = _normalize(xb)
    best = _max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur = best
    Temp = 0.004

    trial = 0
    # precompute nothing
    while time.time() < deadline - 20.0 and trial < 1000:
        # choose move type
        r = rng.rand()
        per, _ = _per_point_max(x_cur)
        order_worst = np.argsort(-per)
        if r < 0.30:
            # global gaussian shake, sigma annealed wide
            sig = float(rng.choice([0.01, 0.03, 0.08, 0.15, 0.25, 0.4]))
            xp = x_cur + sig * rng.randn(*x_cur.shape)
            xp = _normalize(xp)
        elif r < 0.55:
            # block shake worst points
            b = int(rng.randint(3, 12))
            idx = order_worst[:b].copy()
            # sometimes random block
            if rng.rand() < 0.3:
                idx = rng.choice(99, b, replace=False)
            xp = x_cur.copy()
            sig = float(rng.uniform(0.2, 0.6))
            xp[idx] = xp[idx] + sig * rng.randn(b, 13)
            # occasionally randomize fully
            if rng.rand() < 0.25:
                kk = int(rng.randint(1, 3))
                rr = rng.choice(idx, kk, replace=False)
                rnd = rng.randn(kk, 13)
                xp[rr] = _normalize(rnd)
            xp = _normalize(xp)
        elif r < 0.80:
            # hub deletion + hole reinsert
            b = int(rng.randint(2, 5))
            idx = order_worst[:b]
            # add jitter to others slightly to break symmetry
            xp = x_cur.copy()
            # small global jitter
            xp = xp + 0.005 * rng.randn(*xp.shape)
            xp = _normalize(xp)
            try:
                xp = _reinsert_indices(xp, list(idx), rng, n_try=6000)
            except Exception:
                pass
        else:
            # large jump: mix best with random
            sig = 0.5
            xp = x_cur + sig * rng.randn(*x_cur.shape)
            # replace 1-2 points randomly
            kk = int(rng.randint(1, 3))
            ridx = rng.choice(99, kk, replace=False)
            xp[ridx] = rng.randn(kk, 13)
            xp = _normalize(xp)

        # relax with smooth descent; adapt budget by promise
        # two-stage schedule
        k1 = int(rng.choice([25, 40, 60]))
        k2 = int(rng.choice([120, 200, 350]))
        lr0 = float(rng.uniform(0.6, 1.6))
        noise0 = 0.0
        if rng.rand() < 0.3:
            noise0 = 0.02
        try:
            xp_opt, v_opt = _smooth_descent(xp, rng, [(k1, 350), (k2, 350)], lr0, deadline - 18.0, noise0=noise0)
        except Exception:
            trial += 1
            continue
        # extra short high-k sharpening if promising
        if v_opt < cur + 0.01 and time.time() < deadline - 20.0:
            try:
                xp_opt2, v_opt2 = _smooth_descent(xp_opt, rng, [(500, 200)], lr0 * 0.5, deadline - 18.0)
                if v_opt2 < v_opt:
                    xp_opt, v_opt = xp_opt2, v_opt2
            except Exception:
                pass
        # annealing acceptance on walker
        accept = False
        if v_opt < cur:
            accept = True
        else:
            if Temp > 1e-9:
                prob = np.exp(-(v_opt - cur) / Temp)
                if rng.rand() < prob:
                    accept = True
        if accept:
            x_cur = xp_opt.copy()
            cur = v_opt
        # best update
        if v_opt < best - 1e-12:
            best = v_opt
            x_best = xp_opt.copy()
            # pull walker to best occasionally
            if rng.rand() < 0.7:
                x_cur = xp_opt.copy()
                cur = v_opt
        # decay temp
        Temp *= 0.997
        if Temp < 0.0003:
            Temp = 0.0003
        # restart walker to best if too bad
        if cur > best + 0.02 and rng.rand() < 0.5:
            x_cur = x_best.copy()
            cur = best
            Temp = 0.002
        trial += 1

    # final polish best
    remain = deadline - time.time()
    if remain > 4.0:
        try:
            # one more hole attempt on best if time
            per_b, _ = _per_point_max(x_best)
            ow = np.argsort(-per_b)[:3]
            xc = _reinsert_indices(x_best, list(ow), rng, n_try=8000)
            _, vc0 = _smooth_descent(xc, rng, [(60, 300), (250, 300)], 1.0, deadline - 2.0)
            vb = _max_cos(x_best)
            if vc0 < vb:
                # need corresponding x; re-evaluate: smooth_descent returns best; use it
                # redo to get x (already have inside, approximate by xc polish via LP below)
                x_best, best = _smooth_descent(xc, rng, [(60, 300), (250, 300)], 1.0, deadline - 2.0)
                best = _max_cos(x_best)
        except Exception:
            pass
    remain = deadline - time.time()
    if remain > 3.0:
        try:
            x_best, best = _lp_polish(x_best, time_budget=min(remain - 0.5, 18.0), deadline=deadline)
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best
