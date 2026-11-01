import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_hard=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 3e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_hard is not None and time.time() > t_hard:
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_opt(X0, betas, iters, t_limit):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    for beta in betas:
        if time.time() > t_limit:
            break
        b = float(beta)
        def fun(Yflat, b=b):
            Ym = Yflat.reshape(n, d)
            rn = np.linalg.norm(Ym, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Ym / rn
            G = X @ X.T
            gv = G[iu]
            gm = gv.max()
            e = np.exp(b * (gv - gm))
            S = e.sum()
            F = float(gm) + float(np.log(S)) / b
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            gradX = W @ X
            dot = (gradX * X).sum(axis=1, keepdims=True)
            gradY = (gradX - dot * X) / rn
            r1 = (rn.squeeze() - 1.0)
            pen = float((r1 ** 2).mean())
            Ftot = float(F) + pen
            gradY = gradY + (2.0 / n) * r1[:, None] * X
            return Ftot, gradY.ravel()
        try:
            res = minimize(fun, Y.ravel(), method='L-BFGS-B', jac=True,
                           options={'maxiter': int(iters), 'ftol': 1e-12, 'gtol': 1e-7, 'maxcor': 15})
            Y = res.x.reshape(n, d)
        except Exception:
            pass
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        if time.time() > t_limit:
            break
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    EXPLORE_END = t0 + 132.0
    try:
        Xb = np.load("best.npy")
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy")
    except Exception:
        Xr = None
    cands = []
    if Xb is not None:
        cands.append(np.asarray(Xb, dtype=float))
    if Xr is not None:
        cands.append(np.asarray(Xr, dtype=float))
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        cands.append(rng0.standard_normal((71, 12)))
    best_val = 1e9
    Xbest = cands[0]
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            Xbest = c.copy()
    n, d = Xbest.shape
    rng = np.random.default_rng(0)
    iu_all = np.triu_indices(n, 1)
    trial = 0
    # beta schedules to cycle
    scheds = [
        [25, 70, 180, 400],
        [15, 50, 150, 350],
        [40, 120, 300, 600],
        [20, 60, 160, 500],
    ]
    while time.time() < EXPLORE_END and time.time() < HARD - 30 and trial < 100:
        trial += 1
        mode = trial % 5
        X0 = None
        if mode == 0:
            # full random restart
            X0 = rng.standard_normal((n, d))
            betas = scheds[trial % len(scheds)]
            iters = 100
        elif mode == 1:
            # kill overloaded hubs
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            # contacts within 2e-4 of best to identify jammed points
            cnt = (G > best_val - 2e-4).sum(axis=1)
            mxr = G.max(axis=1)
            k = 4 + (trial % 4)  # 4..7
            order = np.argsort(-(cnt * 10 + mxr))
            idx = order[:k]
            X0 = Xbest.copy()
            X0[idx] = rng.standard_normal((k, d))
            betas = scheds[(trial + 1) % len(scheds)]
            iters = 80
        elif mode == 2:
            # global large shake
            sigma = 0.45 + 0.1 * (trial % 3)
            X0 = Xbest + sigma * rng.standard_normal((n, d))
            betas = scheds[(trial + 2) % len(scheds)]
            iters = 80
        elif mode == 3:
            # subset shake + random replacement mix
            X0 = Xbest.copy()
            k = 10 + (trial % 6)
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.6
            X0[idx] = X0[idx] + sigma * rng.standard_normal((k, d))
            # also fully replace 2 of them
            idx2 = rng.choice(idx, size=2, replace=False)
            X0[idx2] = rng.standard_normal((2, d))
            betas = scheds[(trial + 3) % len(scheds)]
            iters = 80
        else:
            # greedy refill: remove worst point(s) and greedily insert
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            mxr = G.max(axis=1)
            k = 3
            order = np.argsort(-mxr)[:k]
            keep = np.ones(n, dtype=bool)
            keep[order] = False
            Xkeep = Xbest[keep]
            newpts = []
            cur = Xkeep
            for _ in range(k):
                C = rng.standard_normal((3000, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                # score = max dot to cur (+ among new)
                S = C @ cur.T
                m = S.max(axis=1)
                # pick best (smallest max) with slight randomization among top
                o = np.argsort(m)[:20]
                pick = o[rng.integers(len(o))]
                bp = C[pick]
                newpts.append(bp)
                cur = np.vstack([cur, bp[None, :]])
            X0 = Xbest.copy()
            X0[order] = np.array(newpts)
            # add small noise to neighbours to open cavity
            X0 = X0 + 0.08 * rng.standard_normal((n, d))
            betas = scheds[trial % len(scheds)]
            iters = 90
        # time budget for this smooth run
        t_lim = min(EXPLORE_END, HARD - 30)
        # leave at least a bit for LP
        if t_lim - time.time() < 1.0:
            break
        try:
            X1 = _smooth_opt(X0, betas, iters, t_lim)
        except Exception:
            continue
        v1 = max_cos(X1)
        if v1 < best_val:
            Xbest = X1
            best_val = v1
            # quick tighten
            rem = min(3.0, max(0.5, (HARD - 28 - time.time()) / 3))
            if time.time() + rem < HARD - 25:
                try:
                    Xp = _lp_polish(Xbest, seconds=rem, t_hard=HARD - 25)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        Xbest, best_val = Xp, vp
                except Exception:
                    pass
        elif v1 < best_val + 0.002:
            # promising: short LP to see true basin depth
            if time.time() < EXPLORE_END - 2 and time.time() < HARD - 32:
                try:
                    Xp = _lp_polish(X1, seconds=1.2, t_hard=HARD - 25)
                    vp = max_cos(Xp)
                    if vp < best_val:
                        Xbest, best_val = Xp, vp
                except Exception:
                    pass
    # final long polish
    trem = HARD - time.time() - 1.0
    if trem > 2:
        try:
            Xbest = _lp_polish(Xbest, seconds=min(trem, 32.0), t_hard=HARD - 0.5)
        except Exception:
            pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
