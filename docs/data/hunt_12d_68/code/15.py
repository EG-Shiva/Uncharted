import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def anneal_exp(x, steps, beta0, beta1, lr0, lr1, mu=0.0, row_norm=False):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        if row_norm:
            rn = np.sqrt(np.sum(Tg * Tg, axis=1, keepdims=True)) + 1e-12
            step = Tg / rn * lr
        else:
            rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
            step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def anneal_pow(x, steps, p0, p1, lr0, lr1, mu=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        p = p0 + (p1 - p0) * t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, 0.0)
        P = np.maximum(G, 0.0) ** (p - 1.0)
        np.fill_diagonal(P, 0.0)
        Grad = P @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.3:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            # keep tightest 1500
            order = np.argsort(-c[sel])[:1500]
            I, J = I[order], J[order]
            p = 1500
            csel = c[sel][order]
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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


def best_hole_vec(rest, rng, ncand=4000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]


def greedy_reinsert(x, rng, rounds=4, ncand=4000):
    n = x.shape[0]
    for _ in range(rounds):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        per = G.max(axis=1)
        wi = int(np.argmax(per))
        cur = float(per[wi])
        rest = np.delete(x, wi, axis=0)
        hole = best_hole_vec(rest, rng, ncand=ncand)
        s = float((hole @ rest.T).max())
        if s < cur - 1e-12:
            x[wi] = hole
        else:
            break
    return x


def solve():
    t0 = time.time()
    LIMIT = 172.0
    FINAL = 30.0
    rng = np.random.default_rng(20261031)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (68, 12) and np.all(np.isfinite(a)):
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((68, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = min(cands, key=lambda z: max_cos(z)).copy()
    best_val = max_cos(x_best)
    n, d = 68, 12
    explore_end = t0 + LIMIT - FINAL
    trial = 0
    while time.time() < explore_end:
        trial += 1
        remain = explore_end - time.time()
        if remain < 0.6:
            break
        r = rng.random()
        x = x_best.copy()
        if r < 0.45:
            sigma = float(rng.choice(np.array([0.004, 0.012, 0.03, 0.07, 0.15])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = int(rng.integers(500, 1200))
            mu = float(rng.choice(np.array([0.0, 0.0, 0.5, 0.85])))
            row_norm = bool(rng.random() < 0.3)
            if rng.random() < 0.72:
                beta0 = float(rng.uniform(40, 100))
                beta1 = float(rng.uniform(300, 650))
                lr0 = float(rng.uniform(0.02, 0.05))
                lr1 = lr0 * 0.12
                x = anneal_exp(x, steps, beta0, beta1, lr0, lr1, mu=mu, row_norm=row_norm)
            else:
                p0 = float(rng.uniform(6, 12))
                p1 = float(rng.uniform(28, 55))
                lr0 = float(rng.uniform(0.02, 0.05))
                lr1 = lr0 * 0.12
                x = anneal_pow(x, steps, p0, p1, lr0, lr1, mu=mu)
            if rng.random() < 0.5:
                x = greedy_reinsert(x, rng, rounds=3, ncand=3000)
        elif r < 0.78:
            # kick overloaded / worst points then reinsert into holes
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            k = int(rng.integers(1, 5))
            idx = []
            nw = min(k, 2)
            for j in range(nw):
                idx.append(int(order[j]))
            while len(idx) < k:
                q = int(rng.integers(0, n))
                if q not in idx:
                    idx.append(q)
            for ii in idx:
                rest = np.delete(x, ii, axis=0)
                if rng.random() < 0.7:
                    x[ii] = best_hole_vec(rest, rng, ncand=2500)
                else:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    x[ii] = v
            x = x + 0.004 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = int(rng.integers(400, 900))
            if rng.random() < 0.75:
                beta0 = float(rng.uniform(50, 110))
                beta1 = float(rng.uniform(300, 600))
                lr0 = float(rng.uniform(0.02, 0.045))
                x = anneal_exp(x, steps, beta0, beta1, lr0, lr0 * 0.12,
                               mu=float(rng.uniform(0, 0.85)))
            else:
                x = anneal_pow(x, steps, 8.0, 40.0, 0.03, 0.004, mu=0.0)
            x = greedy_reinsert(x, rng, rounds=2, ncand=3000)
        else:
            # far basin attempt
            if rng.random() < 0.45:
                x = rng.standard_normal((n, d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = int(rng.integers(1500, 2400))
            else:
                x = x + 0.35 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                steps = int(rng.integers(1200, 2000))
            beta0 = float(rng.uniform(20, 60))
            beta1 = float(rng.uniform(300, 550))
            lr0 = float(rng.uniform(0.03, 0.06))
            x = anneal_exp(x, steps, beta0, beta1, lr0, lr0 * 0.08,
                           mu=float(rng.uniform(0, 0.6)))
            x = greedy_reinsert(x, rng, rounds=4, ncand=4000)
        v = max_cos(x)
        if v < best_val:
            # squeeze with short LP if time allows
            if time.time() < explore_end - 4.5:
                x2 = lp_polish(x, seconds=min(4.0, explore_end - time.time() - 0.3))
                v2 = max_cos(x2)
                if v2 < v:
                    x, v = x2, v2
            if v < best_val:
                x_best = x.copy()
                best_val = v
    remain_final = LIMIT - (time.time() - t0) - 1.0
    if remain_final > 4.0:
        x_best = lp_polish(x_best, seconds=remain_final)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
