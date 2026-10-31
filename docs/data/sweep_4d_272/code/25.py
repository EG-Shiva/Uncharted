import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    x = x / n
    bad = ~np.isfinite(x)
    if np.any(bad):
        x[bad] = 0.0
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _personal_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), float(np.max(g))

def _anneal(X0, stages, mu=0.85, max_step=0.05):
    X = X0.copy()
    vel = np.zeros_like(X)
    for beta, steps, lr in stages:
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(np.max(G))
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            rs = np.sum(W, axis=1, keepdims=True) + 1e-12
            grad = W @ X
            d = np.sum(grad * X, axis=1, keepdims=True)
            gt = (grad - d * X) / rs
            vel = mu * vel - lr * gt
            vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
            scale = np.minimum(1.0, max_step / vn)
            vel = vel * scale
            X = X + vel
            # renormalize
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _find_holes(X, k, rng, n_cand=5000):
    C = rng.standard_normal((n_cand, 4))
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    s = C @ X.T
    scores = np.max(s, axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        c = C[idx]
        ok = True
        for p in picked:
            if float(c @ p) > 0.70:
                ok = False
                break
        if ok:
            picked.append(c.copy())
            if len(picked) >= k:
                break
    while len(picked) < k:
        v = rng.standard_normal(4)
        v = v / max(np.linalg.norm(v), 1e-12)
        picked.append(v)
    return np.array(picked)

def _splice(A, B, rng):
    u = rng.standard_normal(4)
    u = u / max(np.linalg.norm(u), 1e-12)
    da = A @ u
    db = B @ u
    # take positive hemisphere from A, negative from B
    ptsA = A[da > 0]
    ptsB = B[db <= 0]
    pts = np.concatenate([ptsA, ptsB], axis=0) if len(ptsA) and len(ptsB) else (A.copy())
    # adjust to 272
    n = A.shape[0]
    if len(pts) > n:
        # drop points with largest personal max greedily (a few at a time)
        # compute personal max within pts
        cur = pts.copy()
        while len(cur) > n:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            pm = np.max(G, axis=1)
            # drop worst one
            wi = int(np.argmax(pm))
            cur = np.delete(cur, wi, axis=0)
        pts = cur
    elif len(pts) < n:
        need = n - len(pts)
        holes = _find_holes(pts, need, rng, n_cand=4000)
        pts = np.concatenate([pts, holes], axis=0)
    # shuffle slightly to break symmetry
    pts = pts + 0.004 * rng.standard_normal(pts.shape)
    return _normalize(pts)

def _lp_polish(x, t_end):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 1.0 > t_end and radius < 5e-4:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_HOP = t0 + 150.0
    rng = np.random.default_rng(12345)
    try:
        best = np.load("best.npy")
    except Exception:
        best = np.load("record.npy")
    best = _normalize(np.asarray(best, dtype=np.float64))
    if best.shape != (272, 4):
        best = _normalize(rng.standard_normal((272, 4)))
    try:
        rec = _normalize(np.asarray(np.load("record.npy"), dtype=np.float64))
        if rec.shape == (272, 4):
            if _max_cos(rec) < _max_cos(best):
                best = rec
    except Exception:
        pass
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    pool = [best.copy()]
    if len(pool) < 3:
        # add slightly shaken copies for crossover diversity
        for s in (0.01, 0.03):
            pool.append(_normalize(best + s * rng.standard_normal(best.shape)))

    stages_fast = [(80.0, 180, 0.014), (220.0, 220, 0.007), (600.0, 220, 0.003)]
    stages_big = [(60.0, 250, 0.012), (150.0, 300, 0.008), (400.0, 350, 0.004), (800.0, 250, 0.002)]

    n = 272
    hop = 0
    Temp0 = 0.0025
    Temp1 = 0.00015
    while time.time() < T_HOP and hop < 500:
        frac = hop / 500.0
        Temp = Temp0 * (1.0 - frac) + Temp1 * frac
        # time guard for one anneal (~0.4s worst)
        if time.time() + 0.6 > T_HOP:
            break
        op = hop % 8
        Xp = None
        if op == 0:
            Xp = cur + 0.012 * rng.standard_normal(cur.shape)
        elif op == 1:
            Xp = cur + 0.045 * rng.standard_normal(cur.shape)
        elif op == 2:
            # targeted kick of high-contact points
            pm, m = _personal_max(cur)
            cnt_thr = m - 2e-3
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            cnt = np.sum(G > cnt_thr, axis=1)
            idx = np.argsort(-cnt)[:20]
            Xp = cur + 0.008 * rng.standard_normal(cur.shape)
            Xp[idx] += 0.14 * rng.standard_normal((len(idx), 4))
        elif op == 3:
            # worst-point void reinsertion 1-4 pts
            pm, m = _personal_max(cur)
            k = int(rng.integers(1, 5))
            worst = np.argsort(-pm)[:k+4]
            # randomize among top to diversify
            sel = rng.choice(worst, size=k, replace=False)
            holes = _find_holes(np.delete(cur, sel, axis=0), k, rng, n_cand=5000)
            Xp = cur.copy()
            Xp[sel] = holes + 0.003 * rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool); mask[sel] = False
            Xp[mask] += 0.007 * rng.standard_normal((n-k, 4))
        elif op == 4:
            # rattler teleport
            pm, m = _personal_max(cur)
            ratt = np.where(pm < m - 5e-4)[0]
            Xp = cur + 0.008 * rng.standard_normal(cur.shape)
            if len(ratt) > 0:
                k = min(len(ratt), int(rng.integers(1, 4)))
                sel = rng.choice(ratt, size=k, replace=False)
                others_mask = np.ones(n, dtype=bool); others_mask[sel] = False
                holes = _find_holes(cur[others_mask], k, rng, n_cand=5000)
                Xp[sel] = holes + 0.003 * rng.standard_normal((k, 4))
            else:
                # no rattlers: relocate 2 worst to holes
                k = 2
                worst = np.argsort(-pm)[:k]
                holes = _find_holes(np.delete(cur, worst, axis=0), k, rng, n_cand=5000)
                Xp[worst] = holes + 0.003 * rng.standard_normal((k, 4))
        elif op == 5:
            # hemisphere crossover with pool
            A = cur
            B = pool[int(rng.integers(0, len(pool)))]
            Xp = _splice(A, B, rng)
        elif op == 6:
            # random subset resample
            k = int(rng.integers(4, 9))
            idx = rng.choice(n, size=k, replace=False)
            Xp = cur + 0.009 * rng.standard_normal(cur.shape)
            Xp[idx] = rng.standard_normal((k, 4))
        else:
            # large shake for basin escape
            Xp = cur + 0.09 * rng.standard_normal(cur.shape)

        Xp = _normalize(Xp)
        # choose stages by op
        st = stages_big if (op in (5, 7)) else stages_fast
        Xnew = _anneal(Xp, st)
        v = _max_cos(Xnew)
        if v < best_val:
            best = Xnew.copy()
            best_val = v
            pool.append(Xnew.copy())
            if len(pool) > 6:
                pool.pop(1)
            cur = Xnew.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp((cur_val - v) / max(Temp, 1e-12)):
            cur = Xnew.copy()
            cur_val = v
            # keep pool diverse: occasionally add sideways
            if len(pool) < 6 and rng.random() < 0.05:
                pool.append(Xnew.copy())
        hop += 1

    # final intensify on best with a bit more anneal if time remains
    if time.time() + 4.0 < T_HARD:
        # one more sharp anneal from best with tiny shake
        try:
            Xf = _normalize(best + 0.002 * rng.standard_normal(best.shape))
            Xf = _anneal(Xf, [(200.0, 300, 0.006), (800.0, 300, 0.002)])
            vf = _max_cos(Xf)
            if vf < best_val:
                best = Xf
                best_val = vf
        except Exception:
            pass
    # LP polish with remaining time
    if time.time() + 3.0 < T_HARD:
        try:
            best = _lp_polish(best, T_HARD - 1.0)
        except Exception:
            pass
    best = _normalize(best)
    return best
