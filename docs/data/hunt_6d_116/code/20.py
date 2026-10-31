import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _permax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_softmax(X, steps, b0, b1, lr0, lr1):
    X = _norm(X.copy())
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1_ = 0.9; b2_ = 0.999; eps = 1e-8
    p1 = 1.0; p2 = 1.0
    for s in range(1, steps + 1):
        f = s / max(steps, 1)
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(G.max())
        W = np.exp(beta * (G - cm))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        Gr = W @ X
        pr = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - pr * X
        p1 *= b1_; p2 *= b2_
        m1 *= b1_; m1 += (1 - b1_) * Gt
        vv *= b2_; vv += (1 - b2_) * Gt * Gt
        mh = m1 / (1 - p1); vh = vv / (1 - p2)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _adam_hinge(X, steps, margin, lr0, lr1):
    # push only pairs within margin of current max
    X = _norm(X.copy())
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1_ = 0.9; b2_ = 0.999; eps = 1e-8
    p1 = 1.0; p2 = 1.0
    for s in range(1, steps + 1):
        f = s / max(steps, 1)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        th = m - margin
        W = np.maximum(G - th, 0.0)
        # square hinge weight
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S <= 0:
            margin *= 2.0
            continue
        W /= S
        Gr = W @ X
        pr = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - pr * X
        p1 *= b1_; p2 *= b2_
        m1 *= b1_; m1 += (1 - b1_) * Gt
        vv *= b2_; vv += (1 - b2_) * Gt * Gt
        mh = m1 / (1 - p1); vh = vv / (1 - p2)
        X = X - lr * mh / (np.sqrt(vh) + eps)
        X = _norm(X)
    return X

def _lp(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    rad = 1.5e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and rad > 1e-14:
        if time.time() - t0 > seconds - 0.15:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * rad, 1e-9)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            break
        if len(idx) > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        I, J = iu[0][idx], iu[1][idx]
        cs = c[idx]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            rad *= 0.5
            continue
        if res.status != 0:
            rad *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return bx, best

def _hole_of(others, rng, nc=10000, beta=60.0, refine=150):
    d = others.shape[1]
    cand = _norm(rng.standard_normal((nc, d)))
    dots = cand @ others.T
    worst = dots.max(axis=1)
    y = cand[int(np.argmin(worst))].copy()
    st0 = 0.05
    for it in range(refine):
        dd = others @ y
        mx = float(dd.max())
        w = np.exp(beta * (dd - mx)); w /= w.sum()
        g = w @ others
        g = g - (g @ y) * y
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        g /= gn
        st = st0 / (1.0 + it * 0.02)
        y = y - st * g
        y /= (np.linalg.norm(y) + 1e-15)
    return y

def _multi_hole(Y, idxs, rng):
    # sequential greedy reinsertion: remove all idxs, reinsert one by one
    for r in idxs:
        others = np.delete(Y, int(r), axis=0)
        Y[int(r)] = _hole_of(others, rng, nc=9000, beta=55.0 + float(rng.uniform(0, 30)), refine=120)
    return Y

def solve():
    t0 = time.time()
    DEAD = 176.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (116, 6) and np.all(np.isfinite(xb)):
        cands.append(_norm(xb))
    if xr is not None and xr.shape == (116, 6) and np.all(np.isfinite(xr)):
        cands.append(_norm(xr))
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy(); best = float(vals[bi])
    cur = Xbest.copy(); curv = best
    # initial light squeeze of best if time allows
    if time.time() - t0 < 150:
        Xp, vp = _lp(Xbest, min(4.0, max(0.5, DEAD - (time.time() - t0) - 145)))
        if vp < best:
            Xbest, best = Xp, vp
            cur, curv = Xp.copy(), vp
    it = 0
    explore_end = t0 + 141.0
    n = 116
    while time.time() < explore_end and time.time() - t0 < DEAD - 34:
        it += 1
        mode = it % 7
        pm, Gfull = _permax(cur)
        order = np.argsort(-pm)
        # contact counts approx
        m0 = float(Gfull.max())
        cnt = (Gfull > m0 - 2e-7).sum(axis=1)
        Y = cur.copy()
        if mode == 0:
            # single worst hole
            r = int(order[0] if rng.random() < 0.7 else order[int(rng.integers(0, 6))])
            others = np.delete(Y, r, axis=0)
            Y[r] = _hole_of(others, rng, nc=12000, beta=70.0, refine=180)
            Y = Y + 0.004 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _adam_softmax(Y, steps=700, b0=40, b1=1100, lr0=0.02, lr1=0.003)
        elif mode == 1:
            # triple sequential hole on worst 3
            k = int(rng.integers(2, 4))
            idxs = order[:k].tolist()
            # occasionally include a random rattler-like low contact point
            if rng.random() < 0.3:
                low = np.argsort(cnt)[:4]
                idxs[ -1] = int(low[int(rng.integers(0, len(low)))])
            Y = _multi_hole(Y, idxs, rng)
            Y = Y + 0.003 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _adam_softmax(Y, steps=800, b0=60, b1=1400, lr0=0.022, lr1=0.0025)
        elif mode == 2:
            # hub + neighbours joint kick
            hub = int(order[int(rng.integers(0, 4))])
            row = Gfull[hub].copy(); row[hub] = -2
            nbrs = np.argsort(-row)[:int(rng.integers(4, 9))]
            Y[hub] = Y[hub] + 0.18 * rng.standard_normal(6)
            Y[nbrs] = Y[nbrs] + 0.09 * rng.standard_normal((len(nbrs), 6))
            Y = Y + 0.006 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _adam_hinge(Y, steps=250, margin=0.02, lr0=0.02, lr1=0.005)
            Y = _adam_softmax(Y, steps=550, b0=50, b1=1200, lr0=0.02, lr1=0.003)
        elif mode == 3:
            # global shake + hinge then softmax
            sig = float(rng.uniform(0.015, 0.05))
            Y = Y + sig * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _adam_hinge(Y, steps=200, margin=0.03, lr0=0.025, lr1=0.005)
            Y = _adam_softmax(Y, steps=600, b0=30, b1=900, lr0=0.02, lr1=0.003)
        elif mode == 4:
            # contact-biased shake
            amp = 0.004 + 0.016 * (cnt / max(1, cnt.max()))
            Y = Y + amp[:, None] * rng.standard_normal(Y.shape) * float(rng.uniform(0.8, 1.6))
            Y = _norm(Y)
            Y = _adam_softmax(Y, steps=750, b0=80, b1=1800, lr0=0.016, lr1=0.002)
        elif mode == 5:
            # two worst holes + polish
            for r in order[:2]:
                others = np.delete(Y, int(r), axis=0)
                Y[int(r)] = _hole_of(others, rng, nc=9000, beta=60.0, refine=140)
            Y = _norm(Y)
            Y = _adam_softmax(Y, steps=850, b0=70, b1=1600, lr0=0.02, lr1=0.002)
        else:
            # random restart-ish large shake from best occasionally
            if rng.random() < 0.25:
                Y = Xbest.copy() + 0.03 * rng.standard_normal(Y.shape)
            else:
                Y = Y + float(rng.uniform(0.008, 0.025)) * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _adam_softmax(Y, steps=650, b0=90, b1=2200, lr0=0.014, lr1=0.002)
        v = _maxcos(Y)
        if v < best - 2e-9:
            # squeeze new record immediately, short
            rem = min(2.5, max(0.4, explore_end - time.time()))
            if rem > 0.4:
                Y2, v2 = _lp(Y, rem)
                if v2 < v:
                    Y, v = Y2, v2
            if v < best:
                Xbest, best = Y.copy(), v
                cur, curv = Y.copy(), v
                continue
        # annealing acceptance for cur
        frac = (time.time() - t0) / max(1.0, explore_end - t0)
        temp = 0.0005 * max(0.0, 1.0 - frac)
        if v < curv or (temp > 0 and rng.random() < np.exp(-(v - curv) / max(temp, 1e-12))):
            cur, curv = Y.copy(), v
        if it % 15 == 0 and time.time() < explore_end - 3:
            # occasional diversification: restart cur from best with kick if stuck
            if curv > best + 0.002 and rng.random() < 0.7:
                cur = Xbest.copy() + 0.02 * rng.standard_normal(Xbest.shape)
                cur = _norm(cur)
                curv = _maxcos(cur)
    left = DEAD - (time.time() - t0)
    if left > 1.2:
        Xp, vp = _lp(Xbest, min(left - 0.6, 30.0))
        if vp < best:
            Xbest, best = Xp, vp
    left = DEAD - (time.time() - t0)
    if left > 5.0:
        Y = _adam_softmax(Xbest, steps=1100, b0=120, b1=2600, lr0=0.009, lr1=0.001)
        v = _maxcos(Y)
        Y2, v2 = _lp(Y if v < best + 2e-6 else Xbest, left - 0.6)
        if v2 < best:
            Xbest = Y2
    return _norm(Xbest)
