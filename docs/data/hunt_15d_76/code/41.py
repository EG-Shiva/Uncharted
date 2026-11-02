import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _refine_hole(v, Y, steps=80):
    # minimise max_j <v,Yj> on sphere via soft-max descent
    v = v / np.linalg.norm(v)
    T0, T1 = 0.01, 0.001
    lr0, lr1 = 0.15, 0.01
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        s = Y @ v
        m = s.max()
        # stable softmax
        w = np.exp((s - m) / T)
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        g = g - float(g @ v) * v
        v = v - lr * g
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv):
            break
        v /= nv
    return v

def _anneal(X, n_iter, T0, T1, lr0, lr1):
    n = X.shape[0]
    for k in range(n_iter):
        f = k / max(1, n_iter - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        # renorm
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X = X / nn
    return X

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(12345)
    n, d = 76, 15
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        Xb = None
    try:
        Xr = np.load("record.npy").astype(float)
    except Exception:
        Xr = None
    cands = []
    for c in (Xb, Xr):
        if c is not None and c.shape == (n, d):
            c = _norm(np.asarray(c, dtype=float))
            if np.all(np.isfinite(c)):
                cands.append(c)
    if not cands:
        X0 = _norm(rng.normal(size=(n, d)))
        cands.append(X0)
    best = cands[0].copy()
    bv = _maxcos(best)
    for c in cands[1:]:
        v = _maxcos(c)
        if v < bv:
            best, bv = c.copy(), v
    Xbest = best.copy()
    cur = best.copy()
    curv, _ = _per_point_max(cur)
    cur_max = float(curv.max())

    trial = 0
    sigmas = [0.03, 0.07, 0.13, 0.05, 0.10]
    while True:
        if time.time() - t0 > LIMIT - 14:
            break
        trial += 1
        # victim selection: break the jammed core
        pm, gm = _per_point_max(cur)
        order = np.argsort(-pm)  # worst first
        # number to reinsert
        if trial % 4 == 0:
            k_re = 3
        elif trial % 4 == 1:
            k_re = 2
        else:
            k_re = 2 if (trial % 2 == 0) else 4
        k_re = min(k_re, 4)
        victims = []
        victims.append(int(order[0]))
        # one from top-20 random
        top20 = order[:20]
        victims.append(int(rng.choice(top20)))
        while len(victims) < k_re:
            victims.append(int(rng.integers(0, n)))
        # dedup
        victims = list(dict.fromkeys(victims))
        Xt = cur.copy()
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # sequential hole reinsertion
        for idx in victims:
            if time.time() - t0 > LIMIT - 14:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            NS = 3000
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = C[bi]
            v0 = _refine_hole(v0, Y, steps=80)
            Xt[idx] = v0
        # shake rest slightly
        noise = rng.normal(size=(n, d))
        # don't shake reinserted as much: half strength
        shake = np.full(n, sigma)
        for idx in victims:
            shake[idx] = sigma * 0.15
        Xt = Xt + shake[:, None] * noise
        Xt = _norm(Xt)
        # anneal params depend on trial
        if sigma < 0.06:
            Xt = _anneal(Xt, n_iter=900, T0=0.012, T1=0.0008, lr0=3.0, lr1=0.12)
        elif sigma < 0.11:
            Xt = _anneal(Xt, n_iter=1100, T0=0.016, T1=0.0008, lr0=4.0, lr1=0.10)
        else:
            Xt = _anneal(Xt, n_iter=1300, T0=0.022, T1=0.0010, lr0=5.0, lr1=0.12)
        v = _maxcos(Xt)
        if v < bv - 1e-12:
            bv = v
            Xbest = Xt.copy()
            cur = Xt.copy()
            cur_max = v
            # quick squeeze on improvement if time allows
            if time.time() - t0 < LIMIT - 25:
                Xsq = _anneal(Xbest.copy(), n_iter=800, T0=0.0010, T1=0.0002, lr0=0.6, lr1=0.05)
                v2 = _maxcos(Xsq)
                if v2 < bv:
                    bv = v2
                    Xbest = Xsq.copy()
                    cur = Xsq.copy()
                    cur_max = v2
        else:
            # random-walk acceptance to change basin: accept if not much worse
            if v < cur_max + 0.008:
                # bias: accept worse with prob decaying
                if v < cur_max or rng.random() < 0.6:
                    cur = Xt.copy()
                    cur_max = v
            else:
                # occasional restart from best
                if trial % 25 == 0:
                    cur = Xbest.copy()
                    cur_max = bv
        # safety
        if trial > 300:
            break
    # final squeeze of best
    if time.time() - t0 < LIMIT - 8:
        try:
            Xsq = _anneal(Xbest.copy(), n_iter=1500, T0=0.0010, T1=0.0002, lr0=0.5, lr1=0.04)
            v2 = _maxcos(Xsq)
            if v2 < bv and np.all(np.isfinite(Xsq)):
                Xbest, bv = Xsq, v2
        except Exception:
            pass
    Xbest = _norm(Xbest)
    # tiny shrink guard not needed; just ensure finite/unit
    return Xbest
