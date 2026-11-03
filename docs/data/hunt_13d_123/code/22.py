import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish(x, seconds):
    x = _normalize(np.array(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        tol = max(10.0 * radius, 5e-7)
        sel = c > m - tol
        I = iu[0][sel]; J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            # keep tightest 2500 to keep LP cheap
            k = np.argsort(-cc)[:2500]
            I, J, cc = I[k], J[k], cc[k]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.concatenate([ci, cj, np.full(p, n * d)])
        # interleave per row: need row-major ordering matching rows
        # build vals in row order
        vals = np.empty(p * (2 * d + 1))
        vals[0::2*d+1] = 0  # placeholder
        # simpler: construct via loop-free concat per pair
        vmat = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        # rows repeats each row 2d+1 times, cols above is block-ordered, fix cols to row order
        cols_row = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
        A_ub = coo_matrix((vmat, (rows, cols_row)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _anneal(v0, steps, b0, b1, lr0, lr1, mu=0.85):
    v = v0.copy()
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / row
        vel = mu * vel + gt
        v = v - lr * vel
        # normalize fast
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        v /= n
    return v

def _contacts(x, thr=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G > m - thr).sum(axis=1)
    return G, m, cnt

def _reinsert(x, k, rng, ns=4000, div=0.9):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    worst = np.argsort(-rm)[:k]
    keep = np.ones(n, dtype=bool); keep[worst] = False
    Y = x[keep]
    S = rng.randn(ns, d)
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ Y.T
    wd = dots.max(axis=1)
    order = np.argsort(wd)
    chosen = []
    for idx in order:
        cand = S[idx]
        ok = True
        for cv in chosen:
            if float(cand @ cv) > div:
                ok = False; break
        if ok:
            chosen.append(cand)
            if len(chosen) >= k:
                break
    xn = x.copy()
    for ii, cv in zip(worst, chosen):
        xn[ii] = cv + 1e-3 * rng.randn(d)
    return _normalize(xn)

def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    n, d = 123, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None:
            a = np.array(a, dtype=float)
            if a.shape == (n, d):
                cands.append(a)
    if not cands:
        cands = [rng.randn(n, d)]
    # pick best of loaded
    x_best = None; best_val = 1e9
    for a in cands:
        a = _normalize(a)
        v = _max_cos(a)
        if v < best_val:
            best_val = v; x_best = a.copy()
    DEADLINE = 168.0  # leave margin under 180
    trial = 0
    lp_calls = 0
    # quick initial tighten
    if time.time() - t0 < DEADLINE - 30:
        try:
            xt = _lp_polish(x_best, min(6.0, max(1.0, DEADLINE - (time.time()-t0) - 100)))
            v = _max_cos(xt)
            if v < best_val:
                best_val = v; x_best = xt
        except Exception:
            pass
    while time.time() - t0 < 118 and trial < 80:
        trial += 1
        if time.time() - t0 > 115:
            break
        mode = trial % 5
        G, m, cnt = _contacts(x_best)
        if mode == 0:
            # rattler / low-contact reinsertion
            k = 1
            # force include lowest contact point
            xt = _reinsert(x_best, k, rng, ns=4000, div=0.9)
            xt = xt + 0.004 * rng.randn(n, d); xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 450, 40.0, 550.0, 0.008, 0.0008
        elif mode == 1:
            xt = _reinsert(x_best, 2, rng, ns=4000, div=0.88)
            xt = xt + 0.005 * rng.randn(n, d); xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 500, 35.0, 600.0, 0.009, 0.0008
        elif mode == 2:
            # hub kick: most contacts
            hubs = np.argsort(-cnt)[:6]
            xt = x_best + 0.006 * rng.randn(n, d)
            xt[hubs] += 0.05 * rng.randn(len(hubs), d)
            xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 450, 35.0, 550.0, 0.009, 0.0008
        elif mode == 3:
            xt = x_best + 0.018 * rng.randn(n, d); xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 500, 28.0, 450.0, 0.010, 0.0009
        else:
            xt = x_best + 0.035 * rng.randn(n, d); xt = _normalize(xt)
            steps, b0, b1, l0, l1 = 550, 22.0, 380.0, 0.012, 0.0010
        v = _anneal(xt, steps, b0, b1, l0, l1)
        sv = _max_cos(v)
        if sv < best_val:
            best_val = sv; x_best = v.copy()
            # polish immediate improvements
            remain = DEADLINE - (time.time() - t0)
            if remain > 12:
                try:
                    vp = _lp_polish(v, min(4.0, remain - 6))
                    pv = _max_cos(vp)
                    lp_calls += 1
                    if pv < best_val:
                        best_val = pv; x_best = vp.copy()
                except Exception:
                    pass
            continue
        # selective LP for near-best
        if sv < best_val + 0.0015 and lp_calls < 8 and (trial % 3 == 0):
            remain = 121 - (time.time() - t0)
            if remain < 3:
                continue
            try:
                vp = _lp_polish(v, min(2.5, remain))
                pv = _max_cos(vp)
                lp_calls += 1
                if pv < best_val:
                    best_val = pv; x_best = vp.copy()
            except Exception:
                pass
    remain = DEADLINE - (time.time() - t0)
    if remain > 4:
        try:
            xf = _lp_polish(x_best, min(remain - 1, 30.0))
            if _max_cos(xf) < best_val:
                x_best = xf
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best
