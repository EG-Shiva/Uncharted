import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        # cap size for speed
        if np.count_nonzero(sel) > 2500:
            # keep top 2500
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:2500]] = True
            sel = sel & keep
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, steps, T0, T1, lr):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = float(np.max(G))
        if steps > 1:
            T = T0 * (T1 / T0) ** ((t - 1) / (steps - 1))
        else:
            T = T0
        W = np.exp((G - mv) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        Gr = W @ x
        Gr -= (np.sum(Gr * x, axis=1, keepdims=True)) * x
        m1 = b1 * m1 + (1 - b1) * Gr
        v1 = b2 * v1 + (1 - b2) * (Gr * Gr)
        mh = m1 / (1 - b1 ** t)
        vh = v1 / (1 - b2 ** t)
        st = lr * mh / (np.sqrt(vh) + eps)
        rn = np.linalg.norm(st, axis=1, keepdims=True) + 1e-12
        st *= np.minimum(1.0, 0.03 / rn)
        x = x - st
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy(keep, k, rng, n_cand=15000, refine=250, sigma=0.02):
    d = keep.shape[1]
    pl = keep.copy()
    for _ in range(k):
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        sc = np.max(cand @ pl.T, axis=1)
        bi = int(np.argmin(sc))
        best = cand[bi].copy()
        bs = float(sc[bi])
        for __ in range(refine):
            pr = best + sigma * rng.standard_normal(size=d)
            nrm = float(np.linalg.norm(pr))
            if nrm == 0 or not np.isfinite(nrm):
                continue
            pr /= nrm
            s = float(np.max(pl @ pr))
            if s < bs:
                best, bs = pr, s
        pl = np.vstack([pl, best.reshape(1, -1)])
    return pl

def _hole(placed, rng, C=20000):
    d = placed.shape[1]
    cand = rng.standard_normal(size=(C, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = np.max(cand @ placed.T, axis=1)
    return cand[int(np.argmin(sc))]

def solve():
    t0 = time.time()
    dead = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (154, 10):
        n0, d0 = xb.shape
        tmp = rng.standard_normal(size=(154, 10))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n0, 154)
        dd = min(d0, 10)
        tmp[:m, :dd] = xb[:m, :dd]
        tmp[:m] /= np.linalg.norm(tmp[:m], axis=1, keepdims=True)
        xb = tmp
    n, d = 154, 10
    xb = xb[:n, :d].copy()
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    x_best = xb.copy()
    best = _mc(x_best)
    trial = 0
    while True:
        trial += 1
        if time.time() > dead - 12.0 or trial > 2000:
            break
        op = trial % 6
        try:
            if op == 0:
                y = x_best + 0.03 * rng.standard_normal(size=x_best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=500, T0=0.025, T1=0.004, lr=0.004)
            elif op == 1:
                y = x_best + 0.10 * rng.standard_normal(size=x_best.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                st = 400 if (dead - 12 - time.time()) < 30 else 600
                y = _smooth(y, steps=st, T0=0.03, T1=0.005, lr=0.005)
            elif op == 2:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                mv = float(np.max(G))
                cnt = np.sum(G > mv - 0.004, axis=1)
                order = np.argsort(-cnt)
                kk = 3 + ((trial // 6) % 4)  # 3..6, occasionally larger
                if trial % 24 == 2:
                    kk = 8 + int(rng.integers(0, 4))
                pool = order[:14]
                idx = np.sort(rng.choice(pool, size=min(kk, len(pool)), replace=False))
                mask = np.ones(n, dtype=bool); mask[idx] = False
                y = _greedy(x_best[mask], int(np.sum(~mask)), rng, n_cand=15000, refine=250, sigma=0.02)
                y = y + 0.01 * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=600, T0=0.02, T1=0.003, lr=0.004)
            elif op == 3:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                mv = float(np.max(G))
                cnt = np.sum(G > mv - 0.004, axis=1)
                c0 = int(np.argmax(cnt + rng.random(n) * 2.0))
                sims = G[c0].copy()
                sims[c0] = -2
                nn = np.argsort(-sims)[:5]
                kk_idx = np.sort(np.concatenate([[c0], nn[:3 + int(rng.integers(0, 3))]]))
                mask = np.ones(n, dtype=bool); mask[kk_idx] = False
                y = _greedy(x_best[mask], int(np.sum(~mask)), rng, n_cand=16000, refine=250, sigma=0.025)
                y = y + 0.012 * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=600, T0=0.022, T1=0.0035, lr=0.0045)
            elif op == 4:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                mv = float(np.max(G))
                cnt = np.sum(G > (mv - 5e-4), axis=1)
                ratt = np.where(cnt == 0)[0]
                if len(ratt) >= 2:
                    if len(ratt) > 12:
                        ratt = np.sort(rng.choice(ratt, size=12, replace=False))
                    mask = np.ones(n, dtype=bool); mask[ratt] = False
                    y = _greedy(x_best[mask], len(ratt), rng, n_cand=12000, refine=220, sigma=0.025)
                    y = y + 0.008 * rng.standard_normal(size=y.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    y = _smooth(y, steps=550, T0=0.02, T1=0.0035, lr=0.004)
                else:
                    G2 = x_best @ x_best.T
                    np.fill_diagonal(G2, -2.0)
                    per = np.max(G2, axis=1)
                    w = int(np.argmax(per))
                    mask = np.ones(n, dtype=bool); mask[w] = False
                    h = _hole(x_best[mask], rng, C=20000)
                    y = x_best.copy(); y[w] = h
                    y = y + 0.005 * rng.standard_normal(size=y.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    y = _smooth(y, steps=550, T0=0.018, T1=0.003, lr=0.004)
            else:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                per = np.max(G, axis=1)
                w = int(np.argmax(per))
                mask = np.ones(n, dtype=bool); mask[w] = False
                h = _hole(x_best[mask], rng, C=20000)
                y = x_best.copy(); y[w] = h
                # shake its old neighbours a bit
                nb = np.argsort(-G[w])[:4]
                y[nb] += 0.05 * rng.standard_normal(size=(len(nb), d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth(y, steps=550, T0=0.015, T1=0.0025, lr=0.005)
        except Exception:
            continue
        try:
            v = _mc(y)
        except Exception:
            continue
        if v < best + 0.015:
            bud = min(4.0, (dead - 12.0) - time.time())
            if bud > 1.1:
                try:
                    y2 = _lp(y, seconds=bud)
                    v2 = _mc(y2)
                except Exception:
                    y2, v2 = y, v
                if v2 < best - 1e-12:
                    x_best, best = y2, v2
                elif v < best - 1e-12:
                    x_best, best = y / np.linalg.norm(y, axis=1, keepdims=True), v
            else:
                if v < best - 1e-12:
                    x_best, best = y / np.linalg.norm(y, axis=1, keepdims=True), v
    rem = dead - time.time() - 0.5
    if rem > 2.0:
        try:
            x_best = _lp(x_best, seconds=min(rem, 10.0))
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    if not np.all(np.isfinite(x_best)):
        x_best = xb
    return x_best
