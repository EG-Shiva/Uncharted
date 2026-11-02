import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, lr0, lr1, tau0, tau1):
    n = X.shape[0]
    # log interpolation
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * (lr1 / lr0) ** f if lr0 != lr1 else lr0
        tau = tau0 * (tau1 / tau0) ** f if tau0 != tau1 else tau0
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        # softmax weights
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        WX = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        F = WX - X * WG
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        F /= rs
        X = X - lr * F
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_replace(X, idxs, rng, n_cand=2000):
    n, d = X.shape
    X2 = X.copy()
    # replace sequentially greedily
    for idx in idxs:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        # also exclude other victims to be replaced? keep current values for others except this
        Y = X2[mask]
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ Y.T
        worst = dots.max(axis=1)
        best_c = C[int(np.argmin(worst))]
        X2[idx] = best_c
    return X2

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_deadline:
        if radius < 1e-13:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        window = max(20 * radius, 1e-9)
        sel = c > m - window
        if int(sel.sum()) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # if we improved a lot, keep going; if stuck with tiny radius, break
    return best_x

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    HOP_END = t0 + 118.0
    rng = np.random.default_rng(0)
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape == (58, 22):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c = np.where(np.isfinite(c), c, 0.0)
                nrm = np.linalg.norm(c, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                c = c / nrm
                cands.append(c)
        except Exception:
            pass
    if not cands:
        c = rng.standard_normal((58, 22))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    # short clean of best to have baseline
    if time.time() < HOP_END:
        best = _quench(best.copy(), 1500, 0.008, 0.001, 0.005, 0.0012)
        best_val = _max_cos(best)

    trial = 0
    # precomputed quench presets (lr0,lr1,tau0,tau1,steps)
    while time.time() < HOP_END and trial < 200:
        trial += 1
        mode = trial % 5
        cur = best.copy()
        if mode == 0:
            sig = 0.008 + 0.004 * ((trial // 5) % 3)
            cur = cur + sig * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 1800, 0.02, 0.002, 0.008, 0.0015)
        elif mode == 1:
            idxs = rng.choice(58, size=6, replace=False)
            cur[idxs] += 0.05 * rng.standard_normal((6, 22))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 1800, 0.03, 0.002, 0.010, 0.002)
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            k = int(rng.integers(0, 4))
            victim = int(order[k])
            cur = _hole_replace(best, [victim], rng, n_cand=2500)
            cur += 0.004 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2000, 0.025, 0.0015, 0.009, 0.0015)
        elif mode == 3:
            # multi-hole: 3 worst-degree points
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            victims = [int(order[0]), int(order[1]), int(order[2])]
            # randomize one of them sometimes
            if rng.random() < 0.3:
                victims[2] = int(rng.integers(0, 58))
            cur = _hole_replace(best, victims, rng, n_cand=2500)
            cur += 0.003 * rng.standard_normal(cur.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 2000, 0.03, 0.002, 0.010, 0.0018)
        else:
            # max-pair victim + neighbor shake
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            kk = int(np.argmax(G))
            ii = kk // 58
            jj = kk % 58
            victim = ii if rng.random() < 0.5 else jj
            cur = _hole_replace(best, [victim], rng, n_cand=2500)
            other = jj if victim == ii else ii
            cur[other] += 0.01 * rng.standard_normal(22)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            q = _quench(cur, 1600, 0.02, 0.002, 0.008, 0.0012)
        v = _max_cos(q)
        if v < best_val - 1e-12:
            best, best_val = q, v
        # time guard
        if time.time() >= HOP_END:
            break

    # final fine quench on best (short, cheap)
    if time.time() < T_END - 40:
        # budget ~ up to 10s
        best = _quench(best.copy(), 5000, 0.010, 0.0003, 0.004, 0.0006)
        best_val = _max_cos(best)

    # single final LP polish with remaining time (cap 35s)
    remain = T_END - time.time()
    if remain > 6:
        lp_sec = min(remain - 2, 35.0)
        try:
            polished = _lp_polish(best.copy(), lp_sec, T_END - 1.0)
            pv = _max_cos(polished)
            if pv < best_val:
                best, best_val = polished, pv
        except Exception:
            pass

    # safety: tiny final quench if LP left slack and time remains
    if T_END - time.time() > 4:
        try:
            q2 = _quench(best.copy(), 1500, 0.003, 0.0002, 0.0012, 0.0004)
            v2 = _max_cos(q2)
            if v2 < best_val:
                best, best_val = q2, v2
        except Exception:
            pass

    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
