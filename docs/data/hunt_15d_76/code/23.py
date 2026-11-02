import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _pt_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _find_hole(Xo, rng, nCand=4000, iters=80):
    n1, d = Xo.shape
    C = rng.normal(size=(nCand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    s = C @ Xo.T
    smax = s.max(axis=1)
    ib = int(np.argmin(smax))
    h = C[ib].copy()
    beta = 150.0
    for it in range(iters):
        g = Xo @ h
        gm = float(g.max())
        e = np.exp(beta * (g - gm))
        se = float(e.sum())
        if se == 0 or not np.isfinite(se):
            break
        w = e / se
        gr = Xo.T @ w
        grt = gr - float(gr @ h) * h
        lr = 0.05 * (0.1 ** (it / max(1, iters - 1)))
        h = h - lr * grt
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            h = C[ib]
            break
        h /= nh
    return h

def _adam_run(Y, steps, beta0, beta1, lr0, lr1):
    n, d = Y.shape
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
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
    return Y

def _lp_polish(x, t_end):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.4 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1500:
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
    return x

def solve():
    import numpy as np
    import time
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    def normed(a):
        a = a.astype(float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a
    cands = []
    if xb is not None and xb.shape == (76, 15):
        cands.append(normed(xb))
    if xr is not None and xr.shape == (76, 15):
        cands.append(normed(xr))
    if not cands:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    best = best.copy()
    n, d = best.shape
    cur = best.copy()
    cur_val = bv

    sigmas = [0.02, 0.05, 0.09, 0.15, 0.25, 0.06, 0.12, 0.035]
    trial = 0
    max_trials = 200
    while trial < max_trials and time.time() < deadline - 32:
        frac_all = trial / max_trials
        temp = 0.0015 * (1 - frac_all) + 0.00005
        rng = np.random.default_rng(7000 + trial)
        sigma = sigmas[trial % len(sigmas)]
        # occasional shear move to change topology
        Y = cur + sigma * rng.normal(size=(n, d))
        if trial % 5 == 4:
            u = rng.normal(size=(d,)); u /= np.linalg.norm(u)
            proj = cur @ u
            med = np.median(proj)
            shift = (0.12 + 0.08 * rng.random()) * u[None, :]
            Y[proj > med] += shift
            Y[proj <= med] -= shift * 0.5
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # pick worst-contact points for hole reinsertion
        try:
            pm = _pt_max(Y)
            order = np.argsort(-pm)
        except Exception:
            order = rng.permutation(n)
        k_num = 3 + (trial % 3)
        # mix worst + random
        n_worst = min(k_num - 1, n)
        chosen = list(order[:n_worst])
        # add one random index to keep rattlers moving
        r1 = int(rng.integers(0, n))
        if r1 not in chosen:
            chosen.append(r1)
        else:
            chosen.append(int(rng.integers(0, n)))
        chosen = chosen[:k_num]
        for k in chosen:
            if time.time() > deadline - 32:
                break
            mask = np.ones(n, dtype=bool); mask[k] = False
            Xo = Y[mask]
            h = _find_hole(Xo, rng, nCand=3500, iters=60)
            # small jitter so Adam has gradient
            h = h + 1e-3 * rng.normal(size=(d,))
            h /= np.linalg.norm(h)
            Y[k] = h
        steps = 850
        if time.time() > deadline - 60:
            steps = 550
        Y = _adam_run(Y, steps, 30.0, 320.0, 0.009, 0.0009)
        val = _max_cos(Y)
        if val < bv - 1e-15:
            best, bv = Y.copy(), val
        if val < cur_val - 1e-15:
            cur, cur_val = Y.copy(), val
        else:
            # Metropolis walk to different basin
            if val < cur_val + temp:
                prob = np.exp(-(val - cur_val) / temp) if temp > 0 else 0.0
                if prob > rng.random():
                    cur, cur_val = Y.copy(), val
            # occasional restart from best if wandering too high
            if cur_val > bv + 0.008 and (trial % 7 == 6):
                cur, cur_val = best.copy(), bv
        trial += 1
        # quick opportunistic micro-polish if promising? skip to save time
    remain = deadline - time.time()
    if remain > 7:
        t_end = time.time() + min(remain - 2.0, 27.0)
        try:
            best = _lp_polish(best, t_end)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    if not np.all(np.isfinite(best)):
        best = cands[0]
    return best
