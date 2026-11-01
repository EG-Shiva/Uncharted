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
    radius = 2e-3
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
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam_quench(X0, beta, steps, lr):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
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
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        cur_lr = lr * (1.0 - 0.5 * t / steps)
        X = X - cur_lr * mh / (np.sqrt(vh) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _smooth_lbfgs(X0, betas, iters_list, best_ref=1e9):
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
                           options={'maxiter': int(it), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 12})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        if best_ref < 0.5:
            Gc = Y @ Y.T
            np.fill_diagonal(Gc, -2.0)
            if float(Gc.max()) > best_ref + 0.025:
                break
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def _farthest_hole(R, rng, ncand=5000):
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
    rng = np.random.default_rng(777123)
    # elite pool: list of (score, X)
    elite = [(best, Xbest.copy())]
    # try load record as diversity
    try:
        Xr = np.load("record.npy").astype(float)
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        vr = _max_cos(Xr)
        if vr > best + 1e-12:
            elite.append((vr, Xr))
    except Exception:
        pass
    scheds = [
        ([70.0, 260.0], [55, 65]),
        ([40.0, 130.0, 420.0], [45, 50, 55]),
        ([110.0, 380.0], [55, 75]),
        ([25.0, 80.0, 200.0, 500.0], [35, 40, 45, 40]),
        ([60.0, 180.0, 600.0], [45, 50, 50]),
    ]
    EXPLORE_END = t0 + 136.0
    trial = 0
    sigmas = [0.025, 0.06, 0.12, 0.20, 0.32, 0.05, 0.10]
    while _time.time() < EXPLORE_END and _time.time() < HARD_END - 30 and trial < 800:
        trial += 1
        # pick base
        u0 = rng.random()
        if len(elite) > 1 and u0 < 0.18:
            # second elite
            base = elite[1][1]
            base_score = elite[1][0]
        elif u0 < 0.05:
            # fresh random for diversity (rare)
            base = rng.standard_normal(size=(n, d))
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            base_score = 1.0
        else:
            base = elite[0][1]
            base_score = elite[0][0]
        X0 = base.copy()
        # contact stats of base
        try:
            Gc = X0 @ X0.T
            np.fill_diagonal(Gc, -2.0)
            cnt = (Gc > base_score - 5e-4).sum(axis=1)
            mxr = Gc.max(axis=1)
            # worst pair
            flat = np.argmax(Gc)
            wi, wj = flat // n, flat % n
        except Exception:
            cnt = np.zeros(n); mxr = np.zeros(n); wi, wj = 0, 1
        u = rng.random()
        if u < 0.32:
            sigma = sigmas[trial % len(sigmas)]
            w = np.ones(n)
            order = np.argsort(-cnt)
            w[order[:10]] *= 2.2
            w[mxr > base_score - 2e-6] *= 1.5
            # extra boost to worst pair
            w[wi] *= 1.5; w[wj] *= 1.5
            noise = rng.standard_normal(size=X0.shape)
            X0 = X0 + sigma * noise * w[:, None]
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.52:
            # kill 1-2 hubs
            order = np.argsort(-cnt - 0.3 * rng.random(n))
            k = 1 if rng.random() < 0.6 else 2
            victims = order[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            R = X0[keep]
            for vv in victims:
                nv = _farthest_hole(R, rng, ncand=6000)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.010 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.68:
            # multi kill 3-5
            k = 3 + int(rng.integers(0, 3))
            order = np.argsort(-cnt - 0.3 * rng.random(n))
            pool = order[:14].copy()
            rng.shuffle(pool)
            victims = np.sort(pool[:k])
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            R = X0[keep_mask]
            for vv in victims:
                nv = _farthest_hole(R, rng, ncand=4000)
                X0[vv] = nv
                R = np.vstack([R, nv[None, :]])
            X0 = X0 + 0.008 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.84:
            # cap rotation + worst-pair repulsion
            if rng.random() < 0.5:
                du = rng.standard_normal(size=(d,)); du /= np.linalg.norm(du)
                dots = X0 @ du
                cap = 6 + int(rng.integers(0, 10))
                idx = np.argsort(-dots)[:cap]
                e1 = rng.standard_normal(size=(d,)); e1 /= np.linalg.norm(e1)
                tmp = rng.standard_normal(size=(d,))
                tmp = tmp - (tmp @ e1) * e1
                ne = np.linalg.norm(tmp)
                if ne < 1e-9:
                    tmp = rng.standard_normal(size=(d,)); tmp -= (tmp @ e1) * e1
                    ne = np.linalg.norm(tmp)
                e2 = tmp / max(ne, 1e-12)
                theta = float(rng.normal(0, 0.20))
                if abs(theta) > 0.55:
                    theta = 0.55 * np.sign(theta)
                ct, st = np.cos(theta), np.sin(theta)
                Xc = X0[idx]
                a = Xc @ e1; b = Xc @ e2
                ap = ct * a - st * b; bp = st * a + ct * b
                Xc = Xc + (ap - a)[:, None] * e1[None, :] + (bp - b)[:, None] * e2[None, :]
                Xc = Xc + 0.015 * rng.standard_normal(size=Xc.shape)
                X0[idx] = Xc
            else:
                # push worst pair apart tangentially
                for (a, b) in [(wi, wj)]:
                    va = X0[a]; vb = X0[b]
                    # move a away from b
                    step = 0.05 + 0.15 * rng.random()
                    na = va - step * vb
                    nb = vb - step * va
                    na /= np.linalg.norm(na); nb /= np.linalg.norm(nb)
                    X0[a] = na; X0[b] = nb
                X0 = X0 + 0.02 * rng.standard_normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # global random rotation of subset + noise
            msub = 10 + int(rng.integers(0, 20))
            idx = rng.choice(n, size=msub, replace=False)
            sigma = 0.08 + 0.10 * rng.random()
            X0[idx] = X0[idx] + sigma * rng.standard_normal(size=(msub, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        sc = scheds[trial % len(scheds)]
        betas, iters = sc
        try:
            abeta = 90.0 if (trial % 2 == 0) else 55.0
            steps = 220 if u < 0.7 else 320
            lr = 0.022 if u < 0.4 else 0.016
            X0 = _adam_quench(X0, abeta, steps, lr)
        except Exception:
            pass
        try:
            Xopt = _smooth_lbfgs(X0, betas, iters, best_ref=best)
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best - 1e-12:
            try:
                Xopt2 = _lp_polish(Xopt, seconds=1.8, t_end_limit=HARD_END - 14)
                v2 = _max_cos(Xopt2)
            except Exception:
                Xopt2, v2 = Xopt, v
            if v2 < best:
                Xbest, best = Xopt2, v2
                elite[0] = (best, Xbest.copy())
                elite = sorted(elite, key=lambda z: z[0])[:4]
            elif v < best:
                Xbest, best = Xopt, v
                elite[0] = (best, Xbest.copy())
                elite = sorted(elite, key=lambda z: z[0])[:4]
        elif v < best + 0.002:
            # maybe elite diversity
            if len(elite) < 4 and v > best + 5e-5:
                # check distance to existing
                elite.append((v, Xopt.copy()))
                elite = sorted(elite, key=lambda z: z[0])[:4]
            if (trial % 3 == 0 and v < best + 0.001) or (v < best + 0.00035):
                try:
                    Xopt2 = _lp_polish(Xopt, seconds=1.0, t_end_limit=HARD_END - 14)
                    v2 = _max_cos(Xopt2)
                except Exception:
                    v2 = v + 1.0; Xopt2 = Xopt
                if v2 < best - 1e-13:
                    Xbest, best = Xopt2, v2
                    elite[0] = (best, Xbest.copy())
                    elite = sorted(elite, key=lambda z: z[0])[:4]
        if _time.time() > HARD_END - 30:
            break
    remain = HARD_END - _time.time() - 2.0
    if remain > 3:
        try:
            Xbest = _lp_polish(Xbest, seconds=min(remain, 26.0))
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
