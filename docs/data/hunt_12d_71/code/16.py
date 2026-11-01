import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end_limit=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 6e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end_limit is not None and time.time() > t_end_limit:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = c.max()
        win = max(25.0 * radius, 3e-6)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt < 200:
            # take top 200
            idx = np.argpartition(-c, 200)[:200]
            sel = np.zeros_like(c, dtype=bool)
            sel[idx] = True
            cnt = 200
        elif cnt > 1200:
            idx = np.argpartition(-c, 1200)[:1200]
            sel = np.zeros_like(c, dtype=bool)
            sel[idx] = True
            cnt = 1200
        I = iu0[0][sel]; J = iu0[1][sel]
        p = cnt
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _adam_anneal(X0, steps, beta0, beta1, lr0, lr1):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_m = _max_cos(X)
    best_X = X.copy()
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        beta = float(np.exp(lb0 * (1 - frac) + lb1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        gv = G[iu]
        gm = gv.max()
        e = np.exp(beta * (gv - gm))
        S = e.sum()
        w = e / S
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        grad = W @ X
        proj = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - proj * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mh = m1 / (1 - b1 ** s)
        vh = v1 / (1 - b2 ** s)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if s % 25 == 0 or s == steps:
            cm = _max_cos(X)
            if cm < best_m:
                best_m = cm
                best_X = X.copy()
    cm = _max_cos(X)
    if cm < best_m:
        best_X = X.copy()
    return best_X

def _smooth_lbfgs(X0, betas, iters_list):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    for beta, it in zip(betas, iters_list):
        def fun(Yflat, beta=beta):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = gv.max()
            e = np.exp(beta * (gv - gm))
            S = e.sum()
            F = gm + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            gradX = W @ X
            dot = (gradX * X).sum(axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gradY = gradY + (2.0 / n) * r1[:, None] * X
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-6, 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def _refine_one_point(R, x0, beta, rng, ncand=2000):
    from scipy.optimize import minimize
    d = R.shape[1]
    cand = rng.standard_normal(size=(ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ R.T).max(axis=1)
    order = np.argsort(sc)
    seeds = [x0 / np.linalg.norm(x0)]
    for k in range(min(2, len(order))):
        seeds.append(cand[order[k]])
    best_x = seeds[0]
    best_v = float((R @ best_x).max())
    for s0 in seeds:
        y0 = s0.copy()
        def fun(yflat):
            rn = max(float(np.linalg.norm(yflat)), 1e-12)
            x = yflat / rn
            sv = R @ x
            sm = sv.max()
            e = np.exp(beta * (sv - sm))
            S = e.sum()
            F = sm + np.log(S) / beta
            w = e / S
            gx = w @ R
            dot = float(gx @ x)
            gy = (gx - dot * x) / rn
            r1 = rn - 1.0
            Ftot = float(F) + r1 * r1
            gy = gy + 2.0 * r1 * x
            return Ftot, gy
        try:
            res = minimize(fun, y0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-7})
            y = res.x
            y /= np.linalg.norm(y)
        except Exception:
            y = y0 / np.linalg.norm(y0)
        v = float((R @ y).max())
        if v < best_v:
            best_v = v
            best_x = y
    return best_x, best_v

def _coord_sweep(X, rng, beta=2000.0, npts=6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mxr = G.max(axis=1)
    order = np.argsort(-mxr)[:npts]
    cur_max = float(G.max())
    for idx in order:
        R = np.delete(X, idx, axis=0)
        x0 = X[idx].copy()
        nx, nv = _refine_one_point(R, x0, beta, rng)
        Xt = X.copy()
        Xt[idx] = nx
        g = Xt @ Xt.T
        np.fill_diagonal(g, -2.0)
        m = float(g.max())
        if m < cur_max - 1e-12:
            X = Xt
            cur_max = m
    return X

def _farthest(R, rng, ncand=6000):
    d = R.shape[1]
    cand = rng.standard_normal(size=(ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ R.T).max(axis=1)
    return cand[np.argmin(scores)]

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    try:
        Xbest = np.load("best.npy")
    except Exception:
        try:
            Xbest = np.load("record.npy")
        except Exception:
            r0 = np.random.default_rng(0)
            Xbest = r0.standard_normal(size=(71, 12))
    Xbest = np.asarray(Xbest, dtype=float)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    best = _max_cos(Xbest)
    n, d = Xbest.shape
    rng = np.random.default_rng(987654321)
    iu_all = np.triu_indices(n, 1)
    EXPLORE_END = t0 + 136.0
    trial = 0
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 30 and trial < 500:
        trial += 1
        X0 = Xbest.copy()
        Gc = X0 @ X0.T
        np.fill_diagonal(Gc, -2.0)
        mxr = Gc.max(axis=1)
        cnt_loose = (Gc > best - 2e-3).sum(axis=1)
        u = rng.random()
        mode = trial % 7
        if mode == 0:
            sig = [0.04, 0.10, 0.20, 0.025, 0.07][trial % 5]
            w = 1.0 + 1.5 * (cnt_loose / max(cnt_loose.max(), 1))
            noise = rng.standard_normal(size=X0.shape)
            X0 = X0 + sig * noise * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode == 1:
            order = np.argsort(-mxr)
            k = int(rng.integers(0, 4))
            victim = int(order[k])
            R = np.delete(X0, victim, axis=0)
            nh = _farthest(R, rng, ncand=7000)
            # refine insertion
            try:
                nh2, _ = _refine_one_point(R, nh, 800.0, rng, ncand=1500)
                nh = nh2
            except Exception:
                pass
            X0[victim] = nh
            X0 = X0 + 0.008 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode == 2:
            k = 3 + int(rng.integers(0, 3))
            order = np.argsort(-mxr)
            pool = order[:14].copy()
            rng.shuffle(pool)
            victims = np.sort(pool[:k])
            keep = np.ones(n, dtype=bool); keep[victims] = False
            R = X0[keep]
            for vv in victims:
                nv = _farthest(R, rng, ncand=4000)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.006 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode == 3:
            # pair-split top pairs
            G2 = X0 @ X0.T
            np.fill_diagonal(G2, -2.0)
            for _ in range(4):
                ii, jj = np.unravel_index(np.argmax(G2), G2.shape)
                eps = float(rng.normal(0, 0.03))
                if abs(eps) < 0.005:
                    eps = 0.015 * np.sign(eps + 1e-9)
                # push apart
                ti = X0[jj] - (X0[jj] @ X0[ii]) * X0[ii]
                tj = X0[ii] - (X0[ii] @ X0[jj]) * X0[jj]
                ni = np.linalg.norm(ti); nj = np.linalg.norm(tj)
                if ni > 1e-9:
                    X0[ii] = X0[ii] - eps * ti / ni
                if nj > 1e-9:
                    X0[jj] = X0[jj] - eps * tj / nj
                G2[ii, jj] = G2[jj, ii] = -2.0
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            X0 = X0 + 0.01 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode == 4:
            du = rng.standard_normal(size=(d,)); du /= np.linalg.norm(du)
            dots = X0 @ du
            cap = 6 + int(rng.integers(0, 9))
            idx = np.argsort(-dots)[:cap]
            e1 = rng.standard_normal(size=(d,)); e1 /= np.linalg.norm(e1)
            tmp = rng.standard_normal(size=(d,))
            tmp = tmp - (tmp @ e1) * e1
            ne = np.linalg.norm(tmp)
            if ne < 1e-9:
                tmp = rng.standard_normal(size=(d,)); tmp -= (tmp @ e1) * e1
                ne = np.linalg.norm(tmp)
            e2 = tmp / ne
            theta = float(rng.normal(0, 0.20))
            theta = float(np.clip(theta, -0.5, 0.5))
            ct, st = np.cos(theta), np.sin(theta)
            Xc = X0[idx]
            a = Xc @ e1; b = Xc @ e2
            ap = ct * a - st * b; bp = st * a + ct * b
            Xc = Xc + (ap - a)[:, None] * e1[None, :] + (bp - b)[:, None] * e2[None, :]
            Xc = Xc + 0.02 * rng.standard_normal(size=Xc.shape)
            X0[idx] = Xc
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif mode == 5:
            # rattler large jump
            order = np.argsort(mxr)
            nr = 2 + int(rng.integers(0, 3))
            victims = order[:nr]
            for vv in victims:
                R = np.delete(X0, vv, axis=0)
                X0[vv] = _farthest(R, rng, ncand=5000)
            X0 = X0 + 0.015 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            sig = 0.06
            X0 = X0 + sig * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # annealed adam quench
        try:
            if u < 0.5:
                X0 = _adam_anneal(X0, 280, 40.0, 500.0, 0.025, 0.004)
            else:
                X0 = _adam_anneal(X0, 320, 25.0, 350.0, 0.018, 0.003)
        except Exception:
            pass
        try:
            Xopt = _smooth_lbfgs(X0, [150.0, 600.0, 2000.0], [50, 50, 50])
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best - 1e-12:
            try:
                Xp = _lp_polish(Xopt, seconds=2.0, t_end_limit=HARD_END - 14)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xopt, v
            # ultra high-beta squeeze + coord sweep
            if vp < best:
                try:
                    Xp2 = _smooth_lbfgs(Xp, [8000.0, 30000.0], [40, 30])
                    vp2 = _max_cos(Xp2)
                    if vp2 < vp:
                        Xp, vp = Xp2, vp2
                        Xp = _lp_polish(Xp, seconds=1.0, t_end_limit=HARD_END - 14)
                        vp = _max_cos(Xp)
                except Exception:
                    pass
                try:
                    if vp < best:
                        Xc2 = _coord_sweep(Xp, rng, beta=3000.0, npts=5)
                        vc = _max_cos(Xc2)
                        if vc < vp:
                            Xc2 = _lp_polish(Xc2, seconds=1.0, t_end_limit=HARD_END - 14)
                            vc2 = _max_cos(Xc2)
                            if vc2 < vp:
                                Xp, vp = Xc2, vc2
                except Exception:
                    pass
                Xbest, best = Xp, vp
            else:
                Xbest, best = Xopt, v
        elif v < best + 8e-4 and (trial % 4 == 0):
            try:
                Xc = _coord_sweep(Xopt, rng, beta=2000.0, npts=4)
                vc = _max_cos(Xc)
                if vc < best - 1e-12:
                    Xc = _lp_polish(Xc, seconds=1.2, t_end_limit=HARD_END - 14)
                    vc = _max_cos(Xc)
                    if vc < best:
                        # high beta squeeze
                        try:
                            Xc2 = _smooth_lbfgs(Xc, [10000.0], [30])
                            vc2 = _max_cos(Xc2)
                            if vc2 < vc:
                                Xc, vc = Xc2, vc2
                        except Exception:
                            pass
                        Xbest, best = Xc, vc
                elif vc < best + 2e-4 and trial % 12 == 0:
                    Xc = _lp_polish(Xc, seconds=0.8, t_end_limit=HARD_END - 14)
                    vc = _max_cos(Xc)
                    if vc < best - 1e-13:
                        Xbest, best = Xc, vc
            except Exception:
                pass
        if _time.time() > HARD_END - 28:
            break
    # exploitation on best: coord sweeps
    try:
        if _time.time() < HARD_END - 20:
            Xbest = _coord_sweep(Xbest, rng, beta=4000.0, npts=8)
    except Exception:
        pass
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        try:
            Xbest = _smooth_lbfgs(Xbest, [3000.0, 15000.0], [40, 30])
        except Exception:
            pass
        try:
            Xbest = _lp_polish(Xbest, seconds=min(remain - 1, 22.0))
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
