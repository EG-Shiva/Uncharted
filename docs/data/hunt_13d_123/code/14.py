import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(x, seconds, max_pairs=650):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        remain = seconds - (time.time() - start)
        if remain < 0.3:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        tol = max(12.0 * radius, 2e-9)
        sel = c > m - tol
        # cap pairs to keep HiGHS fast
        if int(sel.sum()) > max_pairs:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            cnt = 0
            # must include very tight ones
            tight = c > m - 2e-9
            keep[tight] = True
            cnt = int(keep.sum())
            for ii in idx:
                if cnt >= max_pairs:
                    break
                if not keep[ii]:
                    keep[ii] = True
                    cnt += 1
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _anneal(v0, steps, b0, b1, l0, l1, mu=0.85):
    v = v0.copy()
    n, d = v.shape
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        v = _norm(v)
    return v

def _hard(v0, steps=160, lr0=0.004, margin=0.003):
    v = v0.copy()
    best = _mc(v)
    bestv = v.copy()
    n = v.shape[0]
    for s in range(steps):
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        M = (G > m - margin).astype(float)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        grad = M @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / cnt
        lr = lr0 * (1.0 - s / steps) + 0.0003 * (s / steps)
        v = _norm(v - lr * gt)
        cur = _mc(v)
        if cur < best:
            best = cur
            bestv = v.copy()
    return bestv

def _hole(Y, rng, n_samples=20000, n_pick=1, sep=0.9):
    d = Y.shape[1]
    S = _norm(rng.randn(n_samples, d))
    dots = S @ Y.T
    wd = dots.max(axis=1)
    order = np.argsort(wd)
    picks = []
    pvecs = []
    for idx in order:
        cand = S[idx]
        ok = True
        for pv in pvecs:
            if float(cand @ pv) > sep:
                ok = False
                break
        if ok:
            picks.append(idx)
            pvecs.append(cand)
            if len(picks) >= n_pick:
                break
    if len(picks) < n_pick:
        for idx in order:
            if idx not in picks:
                picks.append(idx)
                if len(picks) >= n_pick:
                    break
    return S[picks]

def solve():
    t0 = time.time()
    rng = np.random.RandomState(7)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(n, d)
    xb = np.array(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = _norm(xb)
    best = _mc(xb)
    xbest = xb.copy()
    trial = 0
    # budget: explore until ~135s, reserve ~30s for final LP
    while time.time() - t0 < 132 and trial < 120:
        trial += 1
        if time.time() - t0 > 130:
            break
        G = xbest @ xbest.T
        np.fill_diagonal(G, -2.0)
        rmax = G.max(axis=1)
        # near-contact degree for hub detection
        m0 = float(G.max())
        deg = (G > m0 - 0.002).sum(axis=1)
        worst = int(np.argmax(rmax + 1e-9 * rng.randn(n)))
        mode = trial % 5
        xt = xbest.copy()
        if mode == 0:
            # single worst -> deepest hole
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Y = xbest[mask]
            hv = _hole(Y, rng, n_samples=22000, n_pick=1)
            xt[worst] = hv[0] + 1e-3 * rng.randn(d)
            xt = _norm(xt)
            xt = xt + 0.003 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 900, 40.0, 550.0, 0.007, 0.0007
        elif mode == 1:
            # hub cluster shake: worst + nearest neighbours
            nn = np.argsort(-G[worst])[:14]
            xt = xt + 0.004 * rng.randn(n, d)
            xt[nn] = xt[nn] + 0.055 * rng.randn(len(nn), d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 1100, 32.0, 450.0, 0.009, 0.0008
        elif mode == 2:
            # two-worst hole reinsert
            w2 = np.argsort(-rmax)[:2]
            mask = np.ones(n, dtype=bool)
            mask[w2] = False
            Y = xbest[mask]
            hv = _hole(Y, rng, n_samples=18000, n_pick=2, sep=0.85)
            for ii, vv in zip(w2, hv):
                xt[ii] = vv + 1e-3 * rng.randn(d)
            xt = _norm(xt)
            xt = xt + 0.003 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 1000, 40.0, 600.0, 0.007, 0.0007
        elif mode == 3:
            # high-degree hubs shake
            hubs = np.argsort(-deg)[:8]
            xt = xt + 0.004 * rng.randn(n, d)
            xt[hubs] = xt[hubs] + 0.04 * rng.randn(len(hubs), d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 1000, 35.0, 500.0, 0.008, 0.0007
        else:
            # global medium kick
            xt = xt + 0.016 * rng.randn(n, d)
            xt = _norm(xt)
            steps, b0, b1, l0, l1 = 900, 30.0, 450.0, 0.009, 0.0008
        v = _anneal(xt, steps, b0, b1, l0, l1, mu=0.85)
        sv = _mc(v)
        # cheap hard-max refinement if promising
        if sv < best + 0.004:
            v2 = _hard(v, steps=120, lr0=0.003, margin=0.003)
            sv2 = _mc(v2)
            if sv2 < sv:
                v, sv = v2, sv2
        if sv < best:
            best = sv
            xbest = v.copy()
    remain = 172 - (time.time() - t0)
    if remain > 6:
        xbest = _lp_polish(xbest, min(remain - 2, 30.0))
    xbest = _norm(xbest)
    return xbest
