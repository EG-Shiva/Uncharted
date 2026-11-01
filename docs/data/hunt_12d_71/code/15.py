import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts_info(x, best):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mxr = G.max(axis=1)
    cnt = (G > best - 5e-4).sum(axis=1)
    iu = np.unravel_index(int(np.argmax(G)), G.shape)
    return G, mxr, cnt, int(iu[0]), int(iu[1])

def _farthest_hole(R, rng, ncand=6000):
    d = R.shape[1]
    cand = rng.standard_normal(size=(ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ R.T
    m = s.max(axis=1)
    return cand[int(np.argmin(m))]

def _adam_anneal(X0, beta0, beta1, steps, lr0, lr1):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        f = (t - 1) / max(steps - 1, 1)
        beta = float(np.exp(lb0 * (1 - f) + lb1 * f))
        lr = float(np.exp(ll0 * (1 - f) + ll1 * f))
        G = X @ X.T
        gv = G[iu]
        gm = gv.max()
        e = np.exp(beta * (gv - gm))
        s = e.sum()
        w = e / s
        W = np.zeros((n, n))
        W[iu] = w
        W = W + W.T
        grad = W @ X
        proj = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - proj * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mh = m1 / (1 - b1 ** t)
        vh = v1 / (1 - b2 ** t)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lbfgs_stages(X0, betas, iters):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    for beta, it in zip(betas, iters):
        def fun(Yf, beta=beta):
            Ym = Yf.reshape(n, d)
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
            gX = W @ X
            dot = (gX * X).sum(axis=1, keepdims=True)
            gY = (gX - dot * X) / rn
            r1 = rn.squeeze() - 1.0
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gY = gY + (2.0 / n) * r1[:, None] * X
            return Ftot, gY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 10})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 140.0
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = np.random.default_rng(0).standard_normal(size=(71, 12))
    Xb = np.asarray(Xb, dtype=float)
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    best = _max_cos(Xb)
    n, d = Xb.shape
    rng = np.random.default_rng(777)
    Xcur = Xb.copy()
    cur = best
    scheds = [
        ([80.0, 300.0], [50, 60]),
        ([40.0, 150.0, 450.0], [40, 50, 50]),
        ([120.0, 400.0], [50, 70]),
        ([30.0, 100.0, 250.0], [40, 40, 50]),
    ]
    trial = 0
    reset_every = 25
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 30 and trial < 800:
        trial += 1
        # occasionally reset wanderer to best
        if trial % reset_every == 0:
            Xcur, cur = Xb.copy(), best
        base = Xcur.copy()
        base_cur = cur
        try:
            _, mxr, cnt, wi, wj = _contacts_info(base, best)
        except Exception:
            mxr = np.zeros(n); cnt = np.zeros(n); wi, wj = 0, 1
        u = rng.random()
        X0 = base.copy()
        if u < 0.22:
            sigmas = [0.03, 0.07, 0.14, 0.25, 0.05]
            sigma = sigmas[trial % len(sigmas)]
            w = np.ones(n)
            order = np.argsort(-cnt)
            w[order[:8]] *= 2.2
            w[mxr > best - 2e-6] *= 1.5
            # rattlers get moderate weight to drift
            X0 = X0 + sigma * rng.standard_normal(size=X0.shape) * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.38:
            order = np.argsort(-cnt - 0.3 * (mxr - mxr.min()) / (mxr.max() - mxr.min() + 1e-12))
            k = 1 + int(rng.integers(0, 2))
            pool = order[:10].copy()
            rng.shuffle(pool)
            victims = pool[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            R = X0[keep]
            for vv in victims:
                nv = _farthest_hole(R, rng, ncand=7000)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.010 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.52:
            # pair-split: break current worst pair directly
            Gtmp = X0 @ X0.T
            np.fill_diagonal(Gtmp, -2.0)
            a = int(np.argmax(Gtmp.max(axis=1)))
            b = int(np.argmax(Gtmp[a]))
            # push apart
            eps = 0.15 + 0.15 * rng.random()
            sig = 0.02 + 0.03 * rng.random()
            Xa = X0[a] - eps * X0[b] + sig * rng.standard_normal(d)
            Xb_ = X0[b] - eps * X0[a] + sig * rng.standard_normal(d)
            X0[a] = Xa / np.linalg.norm(Xa)
            X0[b] = Xb_ / np.linalg.norm(Xb_)
            # jitter neighbours of a,b
            neigh = np.argsort(-Gtmp[a])[:6].tolist() + np.argsort(-Gtmp[b])[:6].tolist()
            for q in set(neigh):
                if q != a and q != b:
                    X0[q] = X0[q] + 0.015 * rng.standard_normal(d)
                    X0[q] /= np.linalg.norm(X0[q])
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.68:
            k = 2 + int(rng.integers(0, 3))
            order = np.argsort(-cnt)
            pool = order[:14].copy()
            rng.shuffle(pool)
            victims = np.sort(pool[:k])
            keep = np.ones(n, dtype=bool); keep[victims] = False
            R = X0[keep]
            for vv in victims:
                nv = _farthest_hole(R, rng, ncand=4500)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.008 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            du = rng.standard_normal(d); du /= np.linalg.norm(du)
            dots = X0 @ du
            cap = 6 + int(rng.integers(0, 10))
            idx = np.argsort(-dots)[:cap]
            e1 = rng.standard_normal(d); e1 /= np.linalg.norm(e1)
            tmp = rng.standard_normal(d)
            tmp = tmp - (tmp @ e1) * e1
            nn = np.linalg.norm(tmp)
            if nn < 1e-9:
                tmp = rng.standard_normal(d); tmp -= (tmp @ e1) * e1; nn = np.linalg.norm(tmp)
            e2 = tmp / max(nn, 1e-12)
            theta = float(rng.normal(0, 0.20))
            theta = float(np.clip(theta, -0.5, 0.5))
            ct, st = np.cos(theta), np.sin(theta)
            Xc = X0[idx]
            a1 = Xc @ e1; b1 = Xc @ e2
            ap = ct * a1 - st * b1; bp = st * a1 + ct * b1
            Xc = Xc + (ap - a1)[:, None] * e1[None, :] + (bp - b1)[:, None] * e2[None, :]
            Xc = Xc + 0.02 * rng.standard_normal(size=Xc.shape)
            X0[idx] = Xc
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        sc = scheds[trial % len(scheds)]
        betas, iters = sc
        try:
            if trial % 2 == 0:
                X0 = _adam_anneal(X0, 25.0, 320.0, 220, 0.02, 0.004)
            else:
                X0 = _adam_anneal(X0, 15.0, 200.0, 280, 0.015, 0.003)
        except Exception:
            continue
        try:
            vq = _max_cos(X0)
        except Exception:
            continue
        if vq > best + 0.025 and vq > base_cur + 0.025:
            continue
        try:
            Xopt = _lbfgs_stages(X0, betas, iters)
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best - 1e-12:
            try:
                Xp = _lp_polish(Xopt, seconds=1.8, deadline=HARD_END - 12)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xopt, v
            if vp < best:
                Xb, best = Xp, vp
                Xcur, cur = Xp.copy(), vp
            else:
                Xb, best = Xopt, v
                Xcur, cur = Xopt.copy(), v
        elif v < best + 0.0012:
            do_lp = (v < best + 0.00035) or (trial % 4 == 0)
            if do_lp:
                try:
                    Xp = _lp_polish(Xopt, seconds=1.0, deadline=HARD_END - 12)
                    vp = _max_cos(Xp)
                except Exception:
                    Xp, vp = Xopt, v
            else:
                Xp, vp = Xopt, v
            if vp < best - 1e-13:
                Xb, best = Xp, vp
                Xcur, cur = Xp.copy(), vp
            elif vp < cur + 0.0008:
                # sideways wander
                Xcur, cur = Xp.copy(), vp
        if _time.time() > HARD_END - 30:
            break
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        try:
            Xb = _lp_polish(Xb, seconds=min(remain, 26.0))
        except Exception:
            pass
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return Xb
