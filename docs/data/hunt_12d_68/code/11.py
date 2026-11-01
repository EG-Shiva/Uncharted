import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(x0, iters=850, lr0=0.006, s0=60.0, s1=1000.0):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, iters + 1):
        frac = t / iters
        s = s0 + (s1 - s0) * frac
        lr = lr0 * (1.0 - 0.9 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        W = np.exp(s * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S <= 0 or not np.isfinite(S):
            break
        grad = (2.0 / S) * (W @ x)
        proj = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - proj * x
        b1p *= b1; b2p *= b2
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        mm = c.max()
        sel = c > mm - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y = y / nrm
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _lp_walk(x, m_ref, slack, radius, rng, steps=6):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    y = x.copy()
    for _ in range(steps):
        Gfull = y @ y.T
        c = Gfull[iu]
        # pairs that could violate m_ref+slack after step
        sel = c > (m_ref + slack - 4.0 * radius - 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            # free random tangent step
            R = rng.standard_normal((n, d))
            R -= (R * y).sum(axis=1, keepdims=True) * y
            nn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            y = y + radius * R / nn.mean()
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            continue
        R = rng.standard_normal((n, d))
        R -= (R * y).sum(axis=1, keepdims=True) * y
        obj_v = -R.ravel()
        nv = n * d
        rows = np.repeat(np.arange(p), 2 * d)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d)], 1).ravel()
        vals = np.concatenate([y[J], y[I]], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((y.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.concatenate([obj_v, [0.0]])
            # need extra dummy? use nv vars only: build with nv
            res = linprog(obj_v, A_ub=A_ub, b_ub=(m_ref + slack - c[sel]), A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * nv, method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.7
            if radius < 1e-6:
                break
            continue
        u = res.x.reshape(n, d)
        y = y + u
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (68, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((68, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = 68, 12
    it = 0
    batch_best = None
    batch_val = np.inf
    sigmas = [0.02, 0.05, 0.10, 0.20]
    while time.time() < DEADLINE - 14.0:
        it += 1
        mode = it % 8
        xt = best.copy()
        if mode == 0:
            # overloaded re-insertion
            G = xt @ xt.T
            np.fill_diagonal(G, -2.0)
            cnt = (G > best_val - 0.002).sum(axis=1)
            idx = int(np.argmax(cnt))
            Nc = 2000
            cand = rng.standard_normal((Nc, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(xt, idx, axis=0)
            scores = (cand @ others.T).max(axis=1)
            xt[idx] = cand[int(np.argmin(scores))]
        elif mode == 1 or mode == 5:
            # slack manifold tunneling then quench
            slack = float(rng.choice([0.003, 0.006, 0.012]))
            rad = float(rng.choice([0.03, 0.05, 0.08]))
            try:
                xt = _lp_walk(xt, best_val, slack, rad, rng, steps=int(rng.integers(4, 9)))
            except Exception:
                s = sigmas[(it // 8) % len(sigmas)]
                xt = xt + s * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 2:
            k = int(rng.integers(3, 9))
            idxs = rng.choice(n, size=k, replace=False)
            xt[idxs] += 0.35 * rng.standard_normal((k, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif mode == 3:
            idx = int(rng.integers(0, n))
            Nc = 2000
            cand = rng.standard_normal((Nc, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(xt, idx, axis=0)
            scores = (cand @ others.T).max(axis=1)
            xt[idx] = cand[int(np.argmin(scores))]
        elif mode == 4:
            if it % 56 == 4:
                xt = rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            else:
                s = sigmas[(it // 8) % len(sigmas)]
                xt = xt + s * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        else:
            s = sigmas[(it // 8) % len(sigmas)]
            xt = xt + s * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        # quench with randomized sharpness
        s1 = float(rng.choice([800.0, 1000.0, 1400.0]))
        xq = _quench(xt, iters=800, lr0=0.006, s0=60.0, s1=s1)
        vq = _max_cos(xq)
        if vq < batch_val:
            batch_val = float(vq)
            batch_best = xq.copy()
        if vq < best_val - 1e-7:
            if time.time() < DEADLINE - 16.0:
                xp = _lp_polish(xq, 1.2, DEADLINE - 14.0)
                vp = _max_cos(xp)
                if vp < best_val:
                    best, best_val = xp, float(vp)
                    batch_val = np.inf; batch_best = None
                    continue
                elif vq < best_val:
                    best, best_val = xq, float(vq)
                    batch_val = np.inf; batch_best = None
                    continue
        if it % 12 == 0 and batch_best is not None:
            if batch_val < best_val + 0.012 and time.time() < DEADLINE - 15.0:
                xp = _lp_polish(batch_best, 2.0, DEADLINE - 14.0)
                vp = _max_cos(xp)
                if vp < best_val:
                    best, best_val = xp, float(vp)
            batch_val = np.inf; batch_best = None
    remain = DEADLINE - time.time()
    if remain > 2.0:
        try:
            xq = _quench(best, iters=500, lr0=0.002, s0=300.0, s1=2200.0)
            if _max_cos(xq) < best_val:
                best = xq; best_val = _max_cos(xq)
        except Exception:
            pass
        best = _lp_polish(best, max(2.0, remain - 1.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
