import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def adam_smooth(X, steps, a0, a1, lr0, lr1, deadline=1e30):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for s in range(steps):
        if (s % 300 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        W = np.exp(alpha * (G - mval))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def lp_polish_mono(x, seconds, radius_init=1.5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # stay at x (not bx) to allow radius shrink search, but keep bx as best
            # move x only if very close? keep x unchanged
            radius *= 0.5
            # if stuck, reset x to bx
            if radius < 1e-7:
                x = bx.copy()
                radius = 1e-6
                break
    return bx / np.linalg.norm(bx, axis=1, keepdims=True)

def solve():
    t0 = time.time()
    deadline_explore = t0 + 140.0
    deadline_total = t0 + 174.0
    rng = np.random.default_rng(12345)
    # load start
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    n, d = best.shape
    # initial monotonic LP squeeze of record (short) to get true local optimum value
    try:
        if deadline_explore - time.time() > 12:
            cand = lp_polish_mono(best, seconds=8.0, radius_init=1e-3)
            v = max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    T0 = 0.002
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.10]
    while time.time() < deadline_explore:
        trial += 1
        elapsed = time.time() - t0
        T = T0 * max(0.05, 1.0 - elapsed / 142.0)
        Xtry = cur.copy()
        r = rng.random()
        if r < 0.55:
            sigma = sigmas[rng.integers(0, len(sigmas))]
            if rng.random() < 0.45:
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                cnt = (G > mx - 5e-4).sum(axis=1)
                order_loose = np.argsort(cnt)
                order_jammed = np.argsort(-cnt)
                k = int(rng.integers(2, 6))
                victims = set()
                for _ in range(k):
                    if rng.random() < 0.5:
                        victims.add(int(order_loose[rng.integers(0, 10)]))
                    else:
                        victims.add(int(order_jammed[rng.integers(0, 10)]))
                victims = list(victims)
                noise = rng.standard_normal(size=Xtry.shape) * (sigma * 0.5)
                if len(victims) > 0:
                    noise[victims, :] *= 5.0
                Xtry = Xtry + noise
            else:
                Xtry = Xtry + sigma * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order = np.argsort(-worst)
            k = int(rng.integers(1, 4))
            victims = [int(order[0])]
            if k > 1:
                pool = [int(p) for p in order[:16] if int(p) not in victims]
                rng.shuffle(pool)
                for p in pool[:k-1]:
                    victims.append(int(p))
            for vv in victims:
                cand = rng.standard_normal(size=(500, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xtry.T
                dots[:, vv] = -10.0
                mv = dots.max(axis=1)
                bi = int(np.argmin(mv))
                Xtry[vv] = cand[bi]
            Xtry = Xtry + 0.002 * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # fast smooth screening
        Xtry = adam_smooth(Xtry, steps=700, a0=25.0, a1=500.0, lr0=0.005, lr1=0.0007, deadline=deadline_explore)
        if not np.all(np.isfinite(Xtry)):
            continue
        v0 = max_cos(Xtry)
        # only LP-polish if promising
        if v0 < best_val + 0.004 or rng.random() < 0.08:
            # slightly longer smooth to centre
            Xtry = adam_smooth(Xtry, steps=300, a0=500.0, a1=1500.0, lr0=0.001, lr1=0.0003, deadline=deadline_explore)
            v0 = max_cos(Xtry)
            try:
                Xp = lp_polish_mono(Xtry, seconds=0.7, radius_init=2e-3)
                vp = max_cos(Xp)
                if vp < v0:
                    Xtry, v0 = Xp, vp
            except Exception:
                pass
        v = v0
        if v < best_val:
            # squeeze new global best a bit more
            try:
                Xf = lp_polish_mono(Xtry, seconds=1.5, radius_init=1e-3)
                vf = max_cos(Xf)
                if vf < v:
                    Xtry, v = Xf, vf
            except Exception:
                pass
            best_val = v
            best = Xtry.copy()
            cur, cur_val = Xtry.copy(), v
        else:
            if v < cur_val or rng.random() < np.exp((cur_val - v) / max(T, 1e-9)):
                cur, cur_val = Xtry.copy(), v
            if rng.random() < 0.07:
                cur, cur_val = best.copy(), best_val
    # final monotonic polish of global best
    try:
        remain = deadline_total - time.time() - 1.0
        if remain > 4:
            cand = lp_polish_mono(best, seconds=min(remain, 28.0), radius_init=1e-3)
            v = max_cos(cand)
            if v < best_val:
                best = cand
                best_val = v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
