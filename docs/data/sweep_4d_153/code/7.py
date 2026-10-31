import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    deadline = time.time() + seconds
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns: I-block, J-block, t
        cI = (I[:, None] * d + np.arange(d)).ravel()
        cJ = (J[:, None] * d + np.arange(d)).ravel()
        cT = np.full(p, n * d)
        # interleave per row: need order matching rows repeat
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::2 * d + 1] = cI[0::d]
        # simpler: build explicitly
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        r_eq = np.repeat(np.arange(n), d)
        c_eq = np.arange(n * d)
        A_eq = coo_matrix((x.ravel(), (r_eq, c_eq)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel],
                A_eq=A_eq, b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs",
            )
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def softmax_anneal(v0, steps, T0, T1, lr0, lr1, mu=0.9):
    v = v0.copy()
    vel = np.zeros_like(v)
    best = v.copy()
    best_val = max_cos(v)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = v @ v.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        W = E / S
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        tg = grad - dot * v
        vel = mu * vel + tg
        v = v - lr * vel
        nr = np.linalg.norm(v, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        v /= nr
        if (s % 120 == 119) or (s == steps - 1):
            cv = max_cos(v)
            if cv < best_val:
                best_val = cv
                best = v.copy()
    return best


def solve():
    t_start = time.time()
    TIME_LIMIT = 172.0
    FINAL_POLISH = 22.0
    rng = np.random.default_rng(7)
    # load starts
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        z = rng.standard_normal((153, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = None
    best_val = 1e9
    for c in cands:
        c = np.asarray(c, dtype=float)
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    cur = best.copy()
    cur_val = best_val
    it = 0
    # main basin loop
    while time.time() - t_start < TIME_LIMIT - FINAL_POLISH - 2.0:
        it += 1
        # choose base: mostly best, sometimes cur for diversity
        if rng.random() < 0.85:
            base = best
        else:
            base = cur
        y = base.copy()
        mode = it % 5
        if mode == 0 or mode == 4:
            sigma = 0.015 + 0.01 * float(rng.random())
            y = y + sigma * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            k = int(6 + rng.integers(0, 7))
            idx = rng.choice(153, size=k, replace=False)
            sigma = 0.06
            y[idx] += sigma * rng.standard_normal((k, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            k = 8
            idx = np.argsort(cnt)[-k:]
            sigma = 0.08
            y[idx] += sigma * rng.standard_normal((k, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt_tight = (G > m - 1e-7).sum(axis=1)
            ratt = np.where(cnt_tight == 0)[0]
            if len(ratt) > 0 and float(rng.random()) < 0.7:
                nr = min(1 + int(rng.integers(0, 2)), len(ratt))
                idx = rng.choice(ratt, size=nr, replace=False)
            else:
                cnt2 = (G > m - 0.004).sum(axis=1)
                idx = np.argsort(cnt2)[-2:]
            for j in idx:
                cand = rng.standard_normal((400, 4))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                mask = np.ones(153, dtype=bool)
                mask[j] = False
                others = y[mask]
                scores = (cand @ others.T).max(axis=1)
                y[j] = cand[np.argmin(scores)]
            y = y + 0.005 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        T0 = 0.02
        T1 = 0.0025
        if it % 7 == 0:
            T0 = 0.04
        y2 = softmax_anneal(y, steps=1500, T0=T0, T1=T1, lr0=0.6, lr1=0.06, mu=0.9)
        y2 = lp_polish(y2, seconds=1.2)
        v2 = max_cos(y2)
        if v2 < best_val - 1e-12:
            best = y2
            best_val = v2
            cur = y2
            cur_val = v2
        else:
            Temp = 0.0005
            if v2 < cur_val or float(rng.random()) < float(np.exp((cur_val - v2) / Temp)):
                cur = y2
                cur_val = v2
        # safety: stop if time low
        if time.time() - t_start > TIME_LIMIT - FINAL_POLISH - 1.0:
            break
    remain = TIME_LIMIT - (time.time() - t_start) - 1.0
    if remain < 2.0:
        remain = 2.0
    # cap final polish to fit 180s hard limit
    remain = min(remain, 178.0 - (time.time() - t_start) - 1.0)
    if remain > 0.5:
        best = lp_polish(best, seconds=remain)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
