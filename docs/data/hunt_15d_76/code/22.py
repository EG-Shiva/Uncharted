import time
import numpy as np

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _anneal_soft(X, steps, beta0, beta1, lr0, lr1, mu=0.9):
    n, d = X.shape
    vel = np.zeros_like(X)
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        beta = beta0 * np.exp(lb * f)
        lr = lr0 * np.exp(ll * f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S == 0 or not np.isfinite(S):
            break
        E /= S
        Gr = 2.0 * (E @ X)
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gt
        vv *= b2; vv += (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1t)
        vh = vv / (1 - b2t)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _polish_hard(X, steps, lr0):
    n, d = X.shape
    for s in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = 2e-4
        # active mask
        M = G > (m - thr)
        # degree
        deg = M.sum(axis=1)
        if deg.max() == 0:
            break
        # gradient: sum of neighbours
        Gr = M.astype(np.float64) @ X
        # remove radial part, normalise by degree
        Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
        sc = deg[:, None] + 1.0
        Gt /= sc
        lr = lr0 * (0.3 ** (s / max(1, steps - 1)))
        # clip
        nr = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True)) + 1e-12
        # limit step to lr
        step = Gt / np.maximum(nr / 1.0, 1.0)
        X = X - lr * step
        X = _norm(X)
    return X

def _hole_replace(Xt, idx, rng):
    n, d = Xt.shape
    mask = np.ones(n, dtype=bool); mask[idx] = False
    Y = Xt[mask]
    NS = 3000
    C = rng.normal(size=(NS, d))
    C = _norm(C)
    scores = (C @ Y.T).max(axis=1)
    bi = int(np.argmin(scores))
    h = C[bi]
    # short refinement 60 steps
    for it in range(60):
        g = Y @ h
        gm = float(g.max())
        # softmax weights
        beta = 150.0
        e = np.exp(beta * (g - gm))
        se = float(e.sum())
        if se == 0 or not np.isfinite(se):
            break
        w = e / se
        gr = Y.T @ w
        grt = gr - float(gr @ h) * h
        lr = 0.05 * (0.1 ** (it / 59.0))
        h = h - lr * grt
        nh = float(np.linalg.norm(h))
        if nh == 0 or not np.isfinite(nh):
            h = C[bi]; break
        h /= nh
    Xt[idx] = h

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
    cands = []
    for a in (xb, xr):
        if a is not None and a.shape == (76, 15):
            a = _norm(a.astype(float))
            if np.all(np.isfinite(a)):
                cands.append(a)
    if not cands:
        rng0 = np.random.default_rng(0)
        X = _norm(rng0.normal(size=(76, 15)))
        return X
    best = cands[0].copy()
    bv = _maxcos(best)
    for c in cands[1:]:
        v = _maxcos(c)
        if v < bv:
            best, bv = c.copy(), v
    n, d = 76, 15
    rng_master = np.random.default_rng(7)

    # quick contact info of incumbent for targeted choice
    def rowmax_and_cnt(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rm = G.max(axis=1)
        cnt = (G > m - 1e-4).sum(axis=1)
        return rm, cnt, m

    trial = 0
    sigmas = [0.05, 0.12, 0.22, 0.35, 0.08, 0.18, 0.03, 0.28]
    # budget: leave 12s for final squeeze
    while time.time() < deadline - 12:
        if trial > 60:
            break
        sigma = sigmas[trial % len(sigmas)]
        rng = np.random.default_rng(10000 + trial * 7919)
        rm, cnt, mcur = rowmax_and_cnt(best)
        Xt = best.copy()
        # full random restart every 8th trial for new basin
        if trial % 8 == 7:
            Xt = _norm(rng.normal(size=(n, d)))
        else:
            # global shake (moves core, not just rattlers)
            Xt = Xt + sigma * rng.normal(size=(n, d))
            Xt = _norm(Xt)
            # reinsert 2-4 points: prefer rattlers + random + overloaded
            order_low = np.argsort(rm)  # rattlers first
            order_high = np.argsort(-cnt)
            k_num = 2 + (trial % 3)  # 2,3,4
            chosen = set()
            chosen.add(int(order_low[trial % n]))
            chosen.add(int(order_high[(trial * 3) % n]))
            while len(chosen) < k_num:
                chosen.add(int(rng.integers(0, n)))
            for idx in chosen:
                if time.time() > deadline - 12:
                    break
                _hole_replace(Xt, idx, rng)
            # extra subset shake on a random cluster to change topology
            if trial % 2 == 0:
                sub = rng.choice(n, size=12, replace=False)
                Xt[sub] += 0.5 * sigma * rng.normal(size=(12, d))
                Xt = _norm(Xt)
        # anneal params depend on sigma
        if sigma > 0.2 or trial % 8 == 7:
            Xt = _anneal_soft(Xt, 900, 25.0, 300.0, 0.010, 0.0006)
        elif sigma > 0.08:
            Xt = _anneal_soft(Xt, 900, 35.0, 350.0, 0.008, 0.0005)
        else:
            Xt = _anneal_soft(Xt, 700, 40.0, 400.0, 0.006, 0.0004)
        if time.time() > deadline - 11:
            v = _maxcos(Xt)
            if v < bv:
                best, bv = Xt, v
            break
        Xt = _polish_hard(Xt, 150, 0.004)
        v = _maxcos(Xt)
        if v < bv - 1e-12:
            best, bv = Xt.copy(), v
            # opportunistic extra squeeze on improvement
            if time.time() < deadline - 20:
                X2 = _anneal_soft(best.copy(), 600, 200.0, 800.0, 0.0012, 0.00015)
                X2 = _polish_hard(X2, 200, 0.0015)
                v2 = _maxcos(X2)
                if v2 < bv:
                    best, bv = X2, v2
        trial += 1

    # final squeeze on incumbent
    if time.time() < deadline - 4:
        try:
            Xf = best.copy()
            # low-temp polish
            Xf = _anneal_soft(Xf, 800, 150.0, 900.0, 0.0015, 0.0001)
            Xf = _polish_hard(Xf, 300, 0.0012)
            vf = _maxcos(Xf)
            if vf < bv:
                best, bv = Xf, vf
        except Exception:
            pass
    best = _norm(best)
    if not np.all(np.isfinite(best)):
        best = _norm(cands[0])
    return best
