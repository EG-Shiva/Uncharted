import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() > t_end - 0.5:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        # cap number of constraints for speed
        if int(sel.sum()) > 1500:
            # keep tightest 1500
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # respect outer seconds
        if time.time() - (t_end - seconds) > seconds:
            break
    return x

def solve():
    import numpy as np
    import time
    t0 = time.time()
    deadline = t0 + 172.0
    # load
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    if xb is None and xr is None:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    def normed(a):
        a = a.astype(float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a
    cands = []
    if xb is not None and xb.shape == (76, 15):
        xb = normed(xb)
        cands.append(xb)
    if xr is not None and xr.shape == (76, 15):
        xr = normed(xr)
        cands.append(xr)
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    best = best.copy()
    n, d = best.shape  # 76,15

    sigmas = [0.03, 0.06, 0.10, 0.16, 0.045, 0.09]
    max_trials = 30
    trial = 0
    # reserve ~28s for final polish
    while trial < max_trials and time.time() < deadline - 30:
        rng = np.random.default_rng(1000 + trial)
        sigma = sigmas[trial % len(sigmas)]
        Y = best + sigma * rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # reinsert k points into largest holes
        k_num = 3 + (trial % 3)  # 3,4,5
        idxs = rng.choice(n, size=k_num, replace=False)
        for k in idxs:
            if time.time() > deadline - 30:
                break
            mask = np.ones(n, dtype=bool); mask[k] = False
            Xo = Y[mask]  # (n-1,d)
            C = rng.normal(size=(6000, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            s = C @ Xo.T
            smax = s.max(axis=1)
            ib = int(np.argmin(smax))
            h = C[ib].copy()
            beta_h = 200.0
            # refine hole 150 steps
            for it in range(150):
                g = Xo @ h
                gm = float(g.max())
                e = np.exp(beta_h * (g - gm))
                se = float(e.sum())
                if se == 0 or not np.isfinite(se):
                    break
                w = e / se
                gr = Xo.T @ w
                gh = float(gr @ h)
                grt = gr - gh * h
                # decaying lr 0.05 -> 0.005
                lr = 0.05 * (0.1 ** (it / 149.0))
                h = h - lr * grt
                nh = np.linalg.norm(h)
                if nh == 0 or not np.isfinite(nh):
                    h = C[ib]
                    break
                h /= nh
            Y[k] = h
        # annealed soft-max repulsion with Adam + tangent projection
        steps = 1100
        # shorten if low on time
        if time.time() > deadline - 60:
            steps = 700
        beta0, beta1 = 40.0, 350.0
        lr0, lr1 = 0.008, 0.0008
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = np.zeros_like(Y)
        vv = np.zeros_like(Y)
        b1t, b2t = 1.0, 1.0
        lb = np.log(beta1 / beta0)
        ll = np.log(lr1 / lr0)
        for s_idx in range(steps):
            frac = s_idx / (steps - 1) if steps > 1 else 0.0
            beta = beta0 * np.exp(lb * frac)
            lr = lr0 * np.exp(ll * frac)
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            # exp may underflow -> fine
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S == 0 or not np.isfinite(S):
                break
            E /= S
            Gr = 2.0 * (E @ Y)
            Gt = Gr - (np.sum(Gr * Y, axis=1, keepdims=True)) * Y
            b1t *= b1; b2t *= b2
            m *= b1; m += (1 - b1) * Gt
            vv *= b2; vv += (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1t)
            vhat = vv / (1 - b2t)
            Y -= lr * mhat / (np.sqrt(vhat) + eps)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        val = _max_cos(Y)
        if val < bv:
            # accept any strict improvement
            best, bv = Y, val
        trial += 1

    # final polish
    remain = deadline - time.time()
    if remain > 6:
        secs = min(remain - 2.0, 28.0)
        t_end = time.time() + secs
        try:
            best = _lp_polish(best, secs, t_end)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    # guard NaN
    if not np.all(np.isfinite(best)):
        best = cands[0]
    return best
