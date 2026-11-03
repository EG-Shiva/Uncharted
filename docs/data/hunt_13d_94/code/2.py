import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def adam_relax(X, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for s in range(steps):
        if deadline is not None and (s % 400 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        # weights focused on large dots
        W = np.exp(alpha * (G - mval))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / bc1
        vhat = v1 / bc2
        upd = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline_phase1 = t0 + 142.0
    deadline_total = t0 + 175.0
    rng = np.random.default_rng(0)
    # load starting points
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
    cur = best.copy()
    cur_val = best_val
    T0 = 0.0025
    trial = 0
    sigmas = [0.004, 0.008, 0.015, 0.03, 0.06, 0.12]
    while time.time() < deadline_phase1:
        trial += 1
        elapsed = time.time() - t0
        # annealing temperature
        T = T0 * max(0.05, 1.0 - elapsed / 145.0)
        r = rng.random()
        Xtry = cur.copy()
        if r < 0.62:
            sigma = sigmas[rng.integers(0, len(sigmas))]
            # occasionally shake only subset
            if rng.random() < 0.35:
                # targeted shake of loose/overloaded points based on contacts
                G = Xtry @ Xtry.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                cnt = (G > mx - 1e-4).sum(axis=1)
                # loose = few contacts, jammed = many contacts
                order_loose = np.argsort(cnt)
                order_jammed = np.argsort(-cnt)
                k = int(rng.integers(2, 6))
                victims = set()
                for _ in range(k):
                    if rng.random() < 0.5:
                        victims.add(int(order_loose[rng.integers(0, 8)]))
                    else:
                        victims.add(int(order_jammed[rng.integers(0, 8)]))
                victims = list(victims)
                noise = rng.standard_normal(size=Xtry.shape) * (sigma * 0.5)
                noise[victims, :] *= 4.0
                Xtry = Xtry + noise
            else:
                Xtry = Xtry + sigma * rng.standard_normal(size=Xtry.shape)
            nr = np.linalg.norm(Xtry, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Xtry /= nr
        else:
            # remove and re-insert worst points
            G = Xtry @ Xtry.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order = np.argsort(-worst)
            k = int(rng.integers(1, 4))
            victims = [int(order[0])]
            if k > 1:
                pool = order[:16].tolist()
                # remove already chosen
                pool = [p for p in pool if p not in victims]
                rng.shuffle(pool)
                for p in pool[:k-1]:
                    victims.append(int(p))
            for v in victims:
                # greedy far-point sampling
                cand = rng.standard_normal(size=(600, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xtry.T
                dots[:, v] = -10.0
                mv = dots.max(axis=1)
                bi = int(np.argmin(mv))
                Xtry[v] = cand[bi]
            # tiny shake of others
            Xtry = Xtry + 0.002 * rng.standard_normal(size=Xtry.shape)
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        # relax: coarse to fine
        # adapt steps to remaining time
        Xtry = adam_relax(Xtry, steps=2200, a0=22.0, a1=650.0, lr0=0.006, lr1=0.0006, deadline=deadline_phase1)
        if not np.all(np.isfinite(Xtry)):
            continue
        v = max_cos(Xtry)
        if v < best_val:
            # fine squeeze with very large alpha
            Xf = adam_relax(Xtry, steps=700, a0=700.0, a1=2200.0, lr0=0.0012, lr1=0.00025, deadline=deadline_phase1)
            if np.all(np.isfinite(Xf)):
                vf = max_cos(Xf)
                if vf < v:
                    Xtry, v = Xf, vf
            best_val = v
            best = Xtry.copy()
            cur, cur_val = Xtry.copy(), v
        else:
            # simulated-annealing acceptance for exploration
            if v < cur_val or rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                cur, cur_val = Xtry.copy(), v
            # occasionally restart cur from best
            if rng.random() < 0.08:
                cur, cur_val = best.copy(), best_val
    # final polish of global best with remaining time
    remain = deadline_total - time.time()
    if remain > 5:
        try:
            # quick fine adam before LP to centre
            best = adam_relax(best, steps=600, a0=500.0, a1=2000.0, lr0=0.001, lr1=0.0003, deadline=deadline_total - 20)
        except Exception:
            pass
        remain2 = deadline_total - time.time() - 1.0
        if remain2 > 4:
            try:
                best = lp_polish(best, min(remain2, 28.0))
            except Exception:
                pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
