import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _quench(y, steps, lr0, T0, T1, hard_tail=80, hard_lr=0.02):
    n, d = y.shape
    y = y.copy()
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f if T1 > 0 else T0
        lr = lr0 * (1.0 - 0.6 * f)
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = (G - m) / max(T, 1e-12)
        np.clip(E, -20, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / rs
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # normalize per-row to keep step stable
        grad = grad / gn
        # scale by mean weight to avoid oscillation late
        y = y - lr * grad
        y = _proj(y)
    # hard max tail: direct subgradient on near-max pairs
    for s in range(hard_tail):
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        tol = 2e-3
        M = G > m - tol
        # weighted repulsion only on active pairs
        W2 = np.where(M, 1.0, 0.0)
        np.fill_diagonal(W2, 0.0)
        rs = W2.sum(axis=1, keepdims=True) + 1e-12
        # only move points that have active contacts
        has = (rs[:, 0] > 0.5)
        if not np.any(has):
            break
        grad = (W2 @ y) / rs
        grad -= (np.sum(grad * y, axis=1, keepdims=True)) * y
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad / gn
        lr = hard_lr * (1.0 - 0.8 * s / max(1, hard_tail - 1))
        yn = y - lr * grad
        y = _proj(yn)
    return y

def _hole_opt(Y, v, steps=200, lr0=0.08, T0=0.02, T1=0.002):
    v = v / (np.linalg.norm(v) + 1e-12)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (1.0 - 0.5 * f)
        g = Y @ v
        m = float(g.max())
        w = np.exp((g - m) / max(T, 1e-12))
        w /= (w.sum() + 1e-300)
        grad = w @ Y
        grad = grad - float(grad @ v) * v
        gn = float(np.linalg.norm(grad)) + 1e-12
        v = v - lr * grad / gn
        v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _lp_polish(x, seconds, r0=2e-3):
    x = _proj(x)
    n, d = x.shape
    best = _mc(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        remain = seconds - (time.time() - start)
        if remain < 0.6:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csv = c[sel]
        if p > 2200:
            k = 2200
            idx = np.argpartition(-csv, k)[:k]
            I, J, csv = I[idx], J[idx], csv[idx]
            p = k
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": min(2.5, max(0.5, remain - 0.15))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _proj(x + res.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best - 1e-13:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    try:
        b1 = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        b1 = None
    try:
        b2 = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        b2 = None
    cands = []
    if b1 is not None and b1.shape == (54, 21):
        cands.append(_proj(b1))
    if b2 is not None and b2.shape == (54, 21):
        cands.append(_proj(b2))
    if not cands:
        cands.append(_proj(rng.normal(size=(54, 21))))
    vals = [_mc(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    bv = float(min(vals))
    n, d = 54, 21

    def left():
        return LIM - (time.time() - T0)

    # fast initial quench of incumbent (cheap, may already improve)
    if left() > 30:
        q = _quench(best.copy(), steps=500, lr0=0.05, T0=0.02, T1=0.002, hard_tail=40, hard_lr=0.01)
        v = _mc(q)
        if v < bv:
            best, bv = q, v

    it = 0
    RESERVE = 13.0
    while left() > RESERVE:
        it += 1
        remain = left() - RESERVE
        if remain < 1.5:
            break
        mode = it % 6
        y = best.copy()
        if mode == 1:
            k = int(rng.integers(4, 9))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = _proj(rng.normal(size=(k, d)))
            y = _quench(y, steps=550, lr0=0.06, T0=0.03, T1=0.0025, hard_tail=50, hard_lr=0.015)
        elif mode == 2:
            s = float(rng.choice([0.15, 0.3, 0.5]))
            nz = rng.normal(size=(n, d)) * s
            nz -= (np.sum(nz * y, axis=1, keepdims=True)) * y
            y = _proj(y + nz)
            y = _quench(y, steps=350, lr0=0.04, T0=0.018, T1=0.002, hard_tail=40, hard_lr=0.012)
        elif mode == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            wi = int(order[int(rng.integers(0, 3))])
            Y = np.delete(best, wi, axis=0)
            C = _proj(rng.normal(size=(1200, d)))
            sc = (C @ Y.T).max(axis=1)
            bi = np.argsort(sc)[:4]
            v0 = C[int(bi[int(rng.integers(0, len(bi)))])]
            vo = _hole_opt(Y, v0, steps=180, lr0=0.07, T0=0.02, T1=0.002)
            y[wi] = vo
            y = _quench(y, steps=320, lr0=0.03, T0=0.014, T1=0.002, hard_tail=40, hard_lr=0.01)
        elif mode == 4:
            # fresh random restart, long quench
            y = _proj(rng.normal(size=(n, d)))
            y = _quench(y, steps=700, lr0=0.07, T0=0.035, T1=0.002, hard_tail=60, hard_lr=0.015)
        elif mode == 5:
            k = 12
            idx = rng.choice(n, size=k, replace=False)
            for i in idx:
                vv = rng.normal(size=d)
                vv -= float(vv @ y[i]) * y[i]
                nv = float(np.linalg.norm(vv))
                if nv < 1e-12:
                    continue
                vv /= nv
                ang = float(0.4 * (0.5 + rng.random()))
                y[i] = np.cos(ang) * y[i] + np.sin(ang) * vv
            y = _proj(y)
            y = _quench(y, steps=450, lr0=0.045, T0=0.022, T1=0.002, hard_tail=40, hard_lr=0.012)
        else:
            s = 0.06
            nz = rng.normal(size=(n, d)) * s
            nz -= (np.sum(nz * y, axis=1, keepdims=True)) * y
            # also kick worst pair apart
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            f = int(np.argmax(G))
            a, b_ = f // n, f % n
            for q in (a, b_):
                o = b_ if q == a else a
                away = -y[o] + float(y[q] @ y[o]) * y[q]
                away -= float(away @ y[q]) * y[q]
                an = float(np.linalg.norm(away))
                if an > 1e-12:
                    y[q] = y[q] - 0.05 * away / an
            y = _proj(y + nz)
            y = _quench(y, steps=380, lr0=0.035, T0=0.016, T1=0.002, hard_tail=50, hard_lr=0.012)
        v = _mc(y)
        if v < bv - 1e-12:
            best, bv = y, v
        # early stop if clearly record-beating and time is short? keep going
        if left() <= RESERVE + 0.5:
            break

    if left() > 2.0:
        sec = min(left() - 0.6, 11.0)
        if sec > 1.0:
            best = _lp_polish(best, seconds=sec)
    best = _proj(best)
    return best
