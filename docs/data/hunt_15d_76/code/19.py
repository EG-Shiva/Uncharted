import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _find_hole(Xo, rng, n_cand=4000, refine=100, beta_h=200.0):
    n1, d = Xo.shape
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    s = C @ Xo.T
    smax = s.max(axis=1)
    ib = int(np.argmin(smax))
    h = C[ib].copy()
    for it in range(refine):
        g = Xo @ h
        gm = float(g.max())
        e = np.exp(beta_h * (g - gm))
        se = float(e.sum())
        if se == 0 or not np.isfinite(se):
            break
        w = e / se
        gr = Xo.T @ w
        grt = gr - float(gr @ h) * h
        lr = 0.05 * (0.1 ** (it / max(1, refine - 1)))
        h = h - lr * grt
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            h = C[ib]
            break
        h /= nh
    return h

def _anneal(Y, steps, beta0, beta1, lr0, lr1):
    n, d = Y.shape
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
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
        v *= b2; v += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1t)
        vhat = v / (1 - b2t)
        Y -= lr * mhat / (np.sqrt(vhat) + eps)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, t_end):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.4 and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mval - thr
        if int(sel.sum()) > 1300:
            idx = np.argpartition(-c, 1300)[:1300]
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
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
        vv = _max_cos(c)
        if vv < bv:
            best, bv = c.copy(), vv
    best = best.copy()
    n, d = best.shape
    sigmas = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.04, 0.10]
    trial = 0
    max_trials = 80
    while trial < max_trials and time.time() < deadline - 30:
        rng = np.random.default_rng(2000 + trial)
        sigma = sigmas[trial % len(sigmas)]
        Y = best + sigma * rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # choose victims: worst points + rattlers/random
        pm = _point_max(Y)
        order = np.argsort(-pm)
        # contact count approx with 1e-9 threshold
        Gtmp = Y @ Y.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = float(Gtmp.max())
        contacts = np.sum(Gtmp > mtmp - 1e-9, axis=1)
        rattlers = np.where(contacts == 0)[0]
        k_num = 2 + (trial % 4)  # 2..5
        victims = list(order[:max(1, k_num - 1)])
        if len(rattlers) > 0:
            # add one rattler not already chosen
            for r in rng.permutation(rattlers):
                if int(r) not in victims:
                    victims.append(int(r))
                    break
        while len(victims) < k_num:
            r = int(rng.integers(0, n))
            if r not in victims:
                victims.append(r)
        victims = victims[:k_num]
        # reinsert each victim into largest hole of remainder
        for kk in victims:
            if time.time() > deadline - 30:
                break
            mask = np.ones(n, dtype=bool); mask[kk] = False
            Xo = Y[mask]
            h = _find_hole(Xo, rng, n_cand=4000, refine=100)
            Y[kk] = h
        steps = 650
        if time.time() > deadline - 60:
            steps = 400
        # vary annealing hardness
        if trial % 3 == 0:
            b0, b1_ = 30.0, 250.0
        elif trial % 3 == 1:
            b0, b1_ = 50.0, 400.0
        else:
            b0, b1_ = 20.0, 180.0
        Y = _anneal(Y, steps, b0, b1_, 0.010, 0.0008)
        val = _max_cos(Y)
        if val < bv - 1e-13:
            best, bv = Y, val
        elif val < bv:
            best, bv = Y, val
        trial += 1
    remain = deadline - time.time()
    if remain > 7:
        t_end = time.time() + min(remain - 2.0, 24.0)
        try:
            polished = _lp_polish(best, t_end)
            pv = _max_cos(polished)
            if pv < bv:
                best, bv = polished, pv
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    if not np.all(np.isfinite(best)):
        best = cands[0]
    return best
