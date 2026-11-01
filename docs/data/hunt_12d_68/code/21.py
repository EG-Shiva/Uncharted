import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _anneal_lse(x, steps, b0, b1, lr0, lr1, mu, rng, noise=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    lb = np.log(b1 / b0) if b0 > 0 and b1 != b0 else 0.0
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = b0 * np.exp(lb * t) if lb != 0.0 else b0
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        step = Tg * (lr / rms)
        if mu > 0:
            vel = mu * vel + step
            step = vel
        x = x - step
        if noise > 0:
            ns = noise * (1.0 - t)
            nz = rng.standard_normal(size=x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            x = x + ns * (nz / nn)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_subgrad(x, iters, lr0=0.02, margin0=0.01, margin1=2e-4):
    n, d = x.shape
    for s in range(iters):
        t = s / max(iters - 1, 1)
        margin = margin0 * (margin1 / margin0) ** t if margin0 > 0 else margin1
        lr = lr0 * (1.0 - 0.85 * t)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = G > m - margin
        iu_r = np.triu_indices(n, 1)
        sel = Mask[iu_r]
        if not np.any(sel):
            continue
        I = iu_r[0][sel]; J = iu_r[1][sel]
        Grad = np.zeros_like(x)
        np.add.at(Grad, I, x[J])
        np.add.at(Grad, J, x[I])
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _best_hole(rest, rng, ncand=8000):
    d = rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.max(cand @ rest.T, axis=1)
    return cand[int(np.argmin(scores))]

def _lp_polish(x, deadline, cap_p=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            order = np.argsort(-csel)[:cap_p]
            I, J = I[order], J[order]
            csel = csel[order]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    T_END = t_start + 170.0
    rng = np.random.default_rng(20261031)
    n, d = 68, 12
    bestX = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < best_score and np.all(np.isfinite(a)):
                    best_score = s
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best_score = _max_cos(bestX)

    def cap_kick(X, m, alpha):
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        proj = X @ u
        idx = np.argsort(proj)[-m:]
        v = rng.standard_normal(d); v -= (v @ u) * u
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = rng.standard_normal(d); nv = np.linalg.norm(v)
        v /= nv
        X2 = X.copy()
        X2[idx] += alpha * v + 0.02 * rng.standard_normal((m, d))
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
        return X2

    explore_end = t_start + 122.0
    trial = 0
    while time.time() < explore_end and time.time() < T_END - 48.0:
        trial += 1
        kind = trial % 8
        Xinit = bestX.copy()
        b0, b1, lr0, lr1, steps, mu, noise = 15.0, 320.0, 0.03, 0.002, 1300, 0.0, 0.0
        sub_iters = 250
        if kind == 0:
            sig = 0.04 + 0.02 * ((trial // 8) % 4)
            Xinit = bestX + sig * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b0, b1, lr0, lr1, steps, mu, noise = 15.0, 350.0, 0.03, 0.002, 1300, 0.0, 0.001
        elif kind == 1:
            sig = 0.10 + 0.05 * ((trial // 8) % 3)
            Xinit = bestX + sig * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b0, b1, lr0, lr1, steps, mu = 10.0, 400.0, 0.035, 0.002, 1500, 0.5
        elif kind == 2:
            k = 5 + int(rng.integers(0, 4))
            idx = rng.choice(n, size=k, replace=False)
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xinit[idx] = R
            b0, b1, lr0, lr1, steps, mu = 8.0, 280.0, 0.04, 0.002, 1500, 0.0
        elif kind == 3:
            G = Xinit @ Xinit.T; np.fill_diagonal(G, -2.0)
            per = np.max(G, axis=1)
            order = np.argsort(-per)
            k = 2 + (trial // 8) % 3
            for kk in range(k):
                victim = int(order[kk])
                rest = np.delete(Xinit, victim, axis=0)
                h = _best_hole(rest, rng, ncand=8000)
                h = h + 0.005 * rng.standard_normal(d); h /= np.linalg.norm(h)
                Xinit[victim] = h
            b0, b1, lr0, lr1, steps = 12.0, 380.0, 0.035, 0.002, 1500
            mu = 0.0
        elif kind == 4:
            m = 4 + int(rng.integers(0, 7))
            alpha = 0.15 + 0.1 * ((trial // 8) % 4)
            Xinit = cap_kick(bestX, m, alpha)
            b0, b1, lr0, lr1, steps, mu = 14.0, 320.0, 0.03, 0.002, 1400, 0.3
        elif kind == 5:
            sig = 0.28
            Xinit = bestX + sig * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b0, b1, lr0, lr1, steps, mu, noise = 8.0, 260.0, 0.05, 0.003, 1500, 0.7, 0.002
        elif kind == 6:
            G = Xinit @ Xinit.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 1e-7, axis=1)
            hub = int(np.argmax(cnt))
            neigh = np.where(G[hub] > m - 0.01)[0]
            if len(neigh) > 0:
                push = np.sum(Xinit[neigh], axis=0)
                push -= np.dot(push, Xinit[hub]) * Xinit[hub]
                push /= (np.linalg.norm(push) + 1e-12)
                Xinit[hub] -= 0.25 * push
                Xinit[hub] /= np.linalg.norm(Xinit[hub])
            Xinit += 0.03 * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b0, b1, lr0, lr1, steps, mu = 15.0, 360.0, 0.03, 0.002, 1400, 0.0
        else:
            Xinit = rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b0, b1, lr0, lr1, steps, mu = 4.0, 60.0, 0.05, 0.004, 1800, 0.0
            sub_iters = 350
        Y = _anneal_lse(Xinit, steps, b0, b1, lr0, lr1, mu, rng, noise)
        if time.time() > T_END - 48.0:
            s = _max_cos(Y)
            if s < best_score and np.all(np.isfinite(Y)):
                best_score, bestX = s, Y.copy()
            break
        Y = _max_subgrad(Y, sub_iters, lr0=0.015, margin0=0.008, margin1=2e-4)
        s = _max_cos(Y)
        if np.all(np.isfinite(Y)) and s < best_score:
            # quick squeeze to lock in improvement
            if time.time() < T_END - 55.0:
                Yq = _lp_polish(Y, min(time.time() + 4.0, T_END - 40.0))
                sq = _max_cos(Yq)
                if sq < s and np.all(np.isfinite(Yq)):
                    Y, s = Yq, sq
            best_score = s
            bestX = Y.copy()
        if trial > 200:
            break

    if time.time() < T_END - 36.0:
        budget = T_END - time.time()
        steps = 3000 if budget > 60 else 1500
        Y = _anneal_lse(bestX.copy(), steps, 25.0, 800.0, 0.012, 0.0006, 0.0, rng, 0.0)
        Y = _max_subgrad(Y, 300, lr0=0.008, margin0=0.004, margin1=1e-4)
        s = _max_cos(Y)
        if s < best_score and np.all(np.isfinite(Y)):
            best_score, bestX = s, Y.copy()

    remain = T_END - time.time()
    if remain > 8.0:
        lp_time = min(32.0, remain - 2.0)
        bestX = _lp_polish(bestX, time.time() + lp_time)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
