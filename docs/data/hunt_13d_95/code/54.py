import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _anneal(X, steps, T0, T1, lr0, lr1, mu=0.9, rng=None, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        if noise > 0 and rng is not None:
            # tangential noise, decaying
            N = rng.standard_normal(size=X.shape)
            N -= np.sum(N * X, axis=1, keepdims=True) * X
            X = X + noise * (1.0 - f) * N
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_vec(others, rng, n_samples=3000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]

def _greedy_polish(X, iters=1200, step0=2e-3):
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    step = step0
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # active set within window
        W = m - 0.02
        mask = G > W
        # weight pairs close to max more
        # build gradient: for each i, sum_j w_ij * (x_j - (x_j.x_i) x_i)
        # use exponential weights
        T = 0.002
        Wgt = np.exp((G - m) / T) * mask
        np.fill_diagonal(Wgt, 0.0)
        s = Wgt.sum()
        if s <= 0 or not np.isfinite(s):
            step *= 0.5
            continue
        Wgt /= s
        Grad = Wgt @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize scale
        gn = np.linalg.norm(Gt)
        if gn < 1e-14:
            step *= 0.5
            continue
        Y = X - step * (Gt / (gn / n + 1e-12) * 0.5 + Gt)
        # simpler: step proportional
        Y = X - step * Gt * n
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            best = v
            X = Y
            Xb = Y.copy()
            step = min(step * 1.1, 5e-3)
        else:
            step *= 0.7
            if step < 1e-7:
                step = step0 * 0.2
                # restart from best with tiny jitter to escape plateau
                # but don't lose best
                X = Xb.copy()
        if step < 1e-12:
            break
    return Xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    EXPL_END = t0 + 150.0
    rng = np.random.default_rng(1)
    n, d = 95, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val

    # Phase 1: a few fresh random starts with strong anneal (fast, diverse basins)
    for rep in range(6):
        if time.time() > EXPL_END - 20:
            break
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z = _anneal(z, steps=1200, T0=0.022, T1=0.0007, lr0=0.35, lr1=0.015, mu=0.88)
        z = _anneal(z, steps=600, T0=0.004, T1=0.0003, lr0=0.08, lr1=0.008, mu=0.9)
        v = _max_cos(z)
        if v < best_val - 1e-12:
            x_best = z.copy(); best_val = v
            cur = z.copy(); cur_val = v
        elif v < cur_val and rng.random() < 0.5:
            cur = z.copy(); cur_val = v

    # also try symmetrized / doubled constructions: antipodal-ish init
    # Phase 2: basin hopping
    it = 0
    sigmas = [0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.20, 0.32, 0.45]
    while time.time() < EXPL_END:
        it += 1
        if time.time() > DEADLINE - 25:
            break
        mode = rng.random()
        Xt = cur.copy()
        if mode < 0.40:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            Xt = Xt + sigma * rng.standard_normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            if sigma < 0.05:
                Xt = _anneal(Xt, steps=450, T0=0.010, T1=0.0006, lr0=0.22, lr1=0.010, mu=0.88)
            elif sigma < 0.15:
                Xt = _anneal(Xt, steps=650, T0=0.015, T1=0.0007, lr0=0.28, lr1=0.012, mu=0.88)
            else:
                Xt = _anneal(Xt, steps=850, T0=0.020, T1=0.0008, lr0=0.32, lr1=0.012, mu=0.88)
        elif mode < 0.65:
            # hub cluster coherent shake
            rm, _ = _row_max(cur)
            order = np.argsort(-rm)
            hub = int(order[int(rng.integers(0, 12))]) if rng.random() < 0.85 else int(rng.integers(0, n))
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            ksz = int(rng.integers(5, 13))
            neigh = np.argsort(-G[hub])[:ksz]
            sigc = float(rng.choice(np.array([0.03, 0.07, 0.13, 0.22])))
            common = rng.standard_normal(size=(d,))
            common /= np.linalg.norm(common)
            for jj in neigh:
                Xt[int(jj)] = Xt[int(jj)] + sigc * common + (sigc * 0.6) * rng.standard_normal(size=(d,))
            Xt[hub] += sigc * rng.standard_normal(size=(d,))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _anneal(Xt, steps=600, T0=0.012, T1=0.0006, lr0=0.26, lr1=0.010, mu=0.88)
        elif mode < 0.88:
            # kill worst k and reinsert at holes (joint)
            rm, _ = _row_max(cur)
            oh = np.argsort(-rm)
            r = rng.random()
            if r < 0.4: k = 1
            elif r < 0.7: k = 2
            elif r < 0.9: k = 3
            else: k = int(rng.integers(4, 7))
            if rng.random() < 0.75:
                idx = oh[:k].copy()
                if k > 1 and rng.random() < 0.35:
                    idx[-1] = int(rng.integers(0, n))
            else:
                idx = rng.choice(n, size=k, replace=False)
            # remove sequentially, each hole w.r.t. updated set
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[int(j)] = False
                others = Xt[mask]
                ns = 4000 if k <= 2 else 2000
                if rng.random() < 0.85:
                    Xt[int(j)] = _hole_vec(others, rng, n_samples=ns)
                else:
                    vv = rng.standard_normal(size=(d,))
                    vv /= np.linalg.norm(vv)
                    Xt[int(j)] = vv
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _anneal(Xt, steps=500, T0=0.009, T1=0.0005, lr0=0.22, lr1=0.010, mu=0.88)
        else:
            # large subset tangential jump
            ksub = int(rng.integers(8, 26))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.choice(np.array([0.15, 0.30, 0.50])))
            for j in idx:
                vv = rng.standard_normal(size=(d,))
                vv = vv - np.dot(vv, cur[int(j)]) * cur[int(j)]
                nn = np.linalg.norm(vv)
                if nn > 1e-12:
                    vv /= nn
                Xt[int(j)] = cur[int(j)] + sigma * vv + 0.25 * sigma * rng.standard_normal(size=(d,))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _anneal(Xt, steps=750, T0=0.017, T1=0.0007, lr0=0.30, lr1=0.012, mu=0.88)
        v = _max_cos(Xt)
        if v < best_val - 1e-12:
            x_best = Xt.copy(); best_val = v
            cur = Xt.copy(); cur_val = v
        elif v < cur_val:
            cur = Xt.copy(); cur_val = v
        else:
            if rng.random() < 0.04 and v < cur_val + 0.010:
                cur = Xt.copy(); cur_val = v
            if rng.random() < 0.08:
                cur = x_best.copy(); cur_val = best_val

    # final squeeze: greedy active-set descent (cheap, no LP solver)
    if time.time() < DEADLINE - 3:
        try:
            x_best = _greedy_polish(x_best, iters=800, step0=1.5e-3)
        except Exception:
            pass
    # extra short anneal at very low T to settle
    if time.time() < DEADLINE - 2:
        try:
            x_best = _anneal(x_best, steps=300, T0=0.0012, T1=0.0002, lr0=0.03, lr1=0.004, mu=0.92)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
