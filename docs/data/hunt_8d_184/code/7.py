import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, seconds, radius_init=4e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = radius_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 4000:
            idx = np.argpartition(-csel, 4000)[:4000]
            I, J, csel = I[idx], J[idx], csel[idx]
            p = 4000
        nv = n * d + 1
        rows_I = np.repeat(np.arange(p), d)
        rows = np.concatenate([rows_I, rows_I, np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7 and p > 1500:
                break
    return xb

def _smooth(x, steps, T0, T1, lr0, lr1):
    x = _norm(x.copy())
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        x = x - lr * Dt
        x = _norm(x)
    return x

def _void_centers(x, rng, nsamp, k):
    n, d = x.shape
    u = rng.randn(nsamp, d)
    u = _norm(u)
    prox = u @ x.T
    sc = prox.max(axis=1)
    take = min(nsamp, max(k * 10, k + 10))
    idx = np.argpartition(sc, take - 1)[:take]
    cand = u[idx]
    scc = sc[idx]
    o = np.argsort(scc)
    cand = cand[o]
    sel = []
    for c in cand:
        if len(sel) >= k:
            break
        ok = True
        for s in sel:
            if float(c @ s) > 0.92:
                ok = False
                break
        if ok:
            sel.append(c)
    while len(sel) < k:
        sel.append(cand[len(sel) % len(cand)])
    return np.array(sel)

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.RandomState(12345)
    try:
        xb = _norm(np.asarray(np.load("best.npy"), dtype=np.float64))
    except Exception:
        xb = None
    try:
        xr = _norm(np.asarray(np.load("record.npy"), dtype=np.float64))
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (184, 8):
            cands.append((_max_cos(c), c))
    if not cands:
        xb = _norm(rng.randn(184, 8))
        best_val = _max_cos(xb)
    else:
        cands.sort(key=lambda t: t[0])
        best_val, xb = cands[0][0], cands[0][1].copy()
        xb = _norm(xb)
    n, d = 184, 8

    trial = 0
    # main hopping loop, reserve ~14s for final polish
    while time.time() - t0 < LIMIT - 14.0:
        trial += 1
        # contact stats of current best (cheap)
        Go = xb @ xb.T
        np.fill_diagonal(Go, -2.0)
        mo = float(Go.max())
        cnt = (Go > mo - 1e-7).sum(axis=1)
        overloaded = np.argsort(-cnt)[:30]
        low = np.where(cnt <= 3)[0]
        pool = np.concatenate([overloaded, low]) if len(low) > 0 else overloaded
        pool = np.unique(pool)

        r = rng.rand()
        x = xb.copy()
        steps = 900
        T0, T1, lr0, lr1 = 0.03, 0.0025, 0.045, 0.004
        if r < 0.45:
            sigma = 10.0 ** rng.uniform(-2.2, -1.0)
            x = x + sigma * rng.randn(n, d)
            x = _norm(x)
            steps = 800 if sigma < 0.03 else 1200
        elif r < 0.85:
            k = int(rng.randint(1, 10))
            if rng.rand() < 0.75:
                vc = _void_centers(xb, rng, 3000, k)
                # choose indices to move: prefer pool
                idx_move = []
                # take some from pool, some random
                np.random.shuffle(pool)
                for ii in range(k):
                    if ii < len(pool) and rng.rand() < 0.8:
                        idx_move.append(int(pool[ii % len(pool)]))
                    else:
                        idx_move.append(int(rng.randint(0, n)))
                idx_move = np.array(idx_move, dtype=int)
                jit = 0.015 * rng.randn(k, d)
                x[idx_move] = _norm(vc + jit)
            else:
                idx_move = rng.choice(n, size=k, replace=False)
                x[idx_move] = _norm(rng.randn(k, d))
            steps = 1300
            T0, T1, lr0, lr1 = 0.025, 0.002, 0.05, 0.003
        else:
            # move single worst point to best void + global tiny shake
            vc = _void_centers(xb, rng, 4000, 1)
            # worst point = max contacts
            w = int(np.argmax(cnt))
            x[w] = _norm(vc[0] + 0.01 * rng.randn(d))
            x = x + 0.004 * rng.randn(n, d)
            x = _norm(x)
            steps = 1000

        x = _smooth(x, steps, T0, T1, lr0, lr1)
        v = _max_cos(x)
        # promising enough to spend LP time?
        if v < best_val + 0.006:
            budget = 2.2 if v > best_val else 3.0
            # don't exceed remaining time
            remain = LIMIT - 14.0 - (time.time() - t0)
            if remain < 1.0:
                break
            budget = min(budget, remain)
            x2 = _lp_polish(x, seconds=budget, radius_init=4e-3)
            v2 = _max_cos(x2)
            if v2 < best_val:
                xb = x2
                best_val = v2
        elif v < best_val:
            xb = _norm(x)
            best_val = v

    # final tightening of the best
    remain = LIMIT - (time.time() - t0) - 1.0
    if remain > 2.0:
        xb = _lp_polish(xb, seconds=min(remain, 12.0), radius_init=3e-3)
    xb = _norm(xb)
    return xb
