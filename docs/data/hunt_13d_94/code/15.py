import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_refine(X, steps, k0, k1, lr0, lr1, mu):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme
        V = mu * V + Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _refine_single(c, others, steps=200, k=80.0):
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(k * (dots - m))
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.2 * (1.0 - 0.6 * it / steps)
        c = c - step * gt
        n = np.linalg.norm(c)
        if n == 0 or not np.isfinite(n):
            break
        c = c / n
    return c

def _best_random_reinsert(others, rng, n_cand=4000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    i = int(np.argmin(scores))
    return cand[i]

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 172.0
    HOP_END = T0 + 136.0
    rng = np.random.default_rng(0)
    # load start
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    T = 0.0025
    trial = 0
    sigmas = [0.008, 0.02, 0.05, 0.12, 0.25]
    while time.time() < HOP_END:
        trial += 1
        elapsed = time.time() - T0
        # anneal temperature
        Tcur = T * max(0.05, 1.0 - elapsed / 140.0)
        r = rng.random()
        y = cur.copy()
        if r < 0.55:
            # global / subset jitter
            sigma = sigmas[rng.integers(0, len(sigmas))]
            if rng.random() < 0.45:
                # targeted subset: worst + random low-contact
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                worst = G.max(axis=1)
                cnt = (G > mx - 1e-4).sum(axis=1)
                # victims: mix of worst and low/high contact
                k = int(rng.integers(2, 7))
                order_w = np.argsort(-worst)
                order_lo = np.argsort(cnt)
                victims = set()
                for _ in range(k):
                    u = rng.random()
                    if u < 0.5:
                        victims.add(int(order_w[rng.integers(0, 10)]))
                    elif u < 0.75:
                        victims.add(int(order_lo[rng.integers(0, 10)]))
                    else:
                        victims.add(int(rng.integers(0, n)))
                victims = list(victims)
                noise = rng.standard_normal(size=(len(victims), d)) * sigma * 2.0
                y[victims] = y[victims] + noise
                nr = np.linalg.norm(y[victims], axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                y[victims] = y[victims] / nr
            else:
                y = y + sigma * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = int(rng.integers(600, 1300))
            k0 = float(rng.choice([15.0, 25.0, 40.0]))
            k1 = float(rng.choice([150.0, 250.0, 350.0]))
            lr0 = float(rng.choice([0.05, 0.08, 0.12]))
            lr1 = 0.005
            mu = 0.85
            y = _smooth_refine(y, steps, k0, k1, lr0, lr1, mu)
        else:
            # reinsert 1-3 worst points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            nr_ins = 1 if rng.random() < 0.5 else (2 if rng.random() < 0.7 else 3)
            order = np.argsort(-worst)
            # pick mostly worst with some randomness
            victims = []
            for k_ in range(min(nr_ins + 2, n)):
                if rng.random() < 0.8 and len(victims) < nr_ins:
                    for o in order:
                        if int(o) not in victims:
                            victims.append(int(o))
                            break
                else:
                    victims.append(int(rng.integers(0, n)))
                if len(victims) >= nr_ins:
                    break
            victims = victims[:nr_ins]
            keep = [i for i in range(n) if i not in set(victims)]
            others = y[keep]
            for v_ in victims:
                c = _best_random_reinsert(others, rng, n_cand=4000)
                c = _refine_single(c, others, steps=250, k=80.0)
                y[v_] = c
                others = y[[i for i in range(n) if i != v_]] if len(victims) > 1 else others
                # update others for next victim
                if len(victims) > 1:
                    keep2 = [i for i in range(n) if i != v_]
                    # placeholder, recompute lazily next loop from y
                    pass
            steps = int(rng.integers(700, 1400))
            y = _smooth_refine(y, steps, 25.0, 280.0, 0.08, 0.004, 0.85)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        elif v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tcur, 1e-12)):
            cur = y.copy()
            cur_val = v
        # occasional restart to best
        if trial % 40 == 0 and rng.random() < 0.3:
            cur = best.copy()
            cur_val = best_val
        if time.time() > HOP_END:
            break
    # final polish best
    try:
        best = _lp_polish(best, T0 + BUDGET)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
