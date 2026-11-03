import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_smooth(X, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    for s in range(steps):
        if deadline is not None and (s % 300 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        a = a0 * (a1 / a0) ** frac if a1 != a0 else a0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
        W = np.exp(a * (G - mv))
        np.fill_diagonal(W, 0.0)
        # normalize to keep scale stable
        S = W.sum()
        if S <= 0 or not np.isfinite(S):
            break
        # gradient of sum-exp
        grad = (W @ X) / S
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

def _reinsert_one(others, rng, n_cand=4000, steps=150):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(scores)].copy()
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        kk = 50.0
        w = np.exp(kk * (dots - m))
        sw = w.sum()
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.2 * (1.0 - 0.5 * it / steps)
        c = c - step * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
    BUDGET_END = T0 + 172.0
    HOP_END = T0 + 122.0
    rng = np.random.default_rng(0)
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
    sigmas = [0.004, 0.01, 0.025, 0.06, 0.13, 0.26]
    trial = 0
    Tstart = 0.004
    while time.time() < HOP_END and time.time() < BUDGET_END - 45:
        trial += 1
        elapsed = time.time() - T0
        Temp = Tstart * max(0.05, 1.0 - elapsed / 130.0)
        r = rng.random()
        y = cur.copy()
        if r < 0.40:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            if rng.random() < 0.30:
                # shake only a subset (loose or jammed)
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                mx = G.max()
                cnt = (G > mx - 1e-4).sum(axis=1)
                order_loose = np.argsort(cnt)
                order_jam = np.argsort(-cnt)
                k = int(rng.integers(2, 7))
                victims = set()
                for _ in range(k):
                    if rng.random() < 0.5:
                        victims.add(int(order_loose[int(rng.integers(0, 10))]))
                    else:
                        victims.add(int(order_jam[int(rng.integers(0, 10))]))
                victims = list(victims)
                y[victims] += sigma * 2.0 * rng.standard_normal(size=(len(victims), d))
                nr = np.linalg.norm(y, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                y /= nr
            else:
                y = y + sigma * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = int(rng.integers(500, 1400))
            a1 = float(rng.choice(np.array([150.0, 250.0, 400.0, 600.0])))
            lr0 = float(rng.choice(np.array([0.008, 0.015, 0.03])))
            lr1 = lr0 * 0.15
            y = _adam_smooth(y, steps, 20.0, a1, lr0, lr1, deadline=HOP_END)
        elif r < 0.70:
            # targeted shake of worst points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            order = np.argsort(-worst)
            k = int(rng.integers(2, 6))
            victims = order[:k].tolist()
            # occasionally add random
            if rng.random() < 0.4:
                victims.append(int(rng.integers(0, n)))
                victims = list(set(victims))
            sigma = float(rng.choice(np.array([0.05, 0.12, 0.25])))
            y[victims] += sigma * rng.standard_normal(size=(len(victims), d))
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            y /= nr
            steps = int(rng.integers(600, 1300))
            a1 = float(rng.choice(np.array([200.0, 350.0, 500.0])))
            y = _adam_smooth(y, steps, 25.0, a1, 0.015, 0.002, deadline=HOP_END)
        else:
            # removal / re-insertion of 1-2 worst points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            worst = G.max(axis=1)
            mx = G.max()
            cnt = (G > mx - 1e-4).sum(axis=1)
            # score: high worst + low count sometimes
            order_worst = np.argsort(-worst)
            order_loose = np.argsort(cnt)
            nv = 1 if rng.random() < 0.6 else 2
            victims = set()
            for _ in range(nv):
                rr = rng.random()
                if rr < 0.6:
                    # worst among top 8
                    victims.add(int(order_worst[int(rng.integers(0, 8))]))
                else:
                    victims.add(int(order_loose[int(rng.integers(0, 8))]))
                if rng.random() < 0.15:
                    victims.add(int(rng.integers(0, n)))
            victims = list(victims)
            keep = [i for i in range(n) if i not in victims]
            others = y[keep]
            newpts = []
            for _ in victims:
                c = _reinsert_one(others, rng, n_cand=4000, steps=120)
                newpts.append(c)
                others = np.vstack([others, c[None, :]])
            y = np.zeros_like(cur)
            y[keep] = cur[keep] if False else y[keep]  # placeholder
            # rebuild: keep original others + new
            # simpler: start from cur, replace victims
            y = cur.copy()
            for vi, c in zip(victims, newpts):
                y[vi] = c
            # small global jitter then smooth
            y = y + 0.005 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = int(rng.integers(700, 1400))
            a1 = float(rng.choice(np.array([200.0, 350.0, 550.0])))
            y = _adam_smooth(y, steps, 20.0, a1, 0.012, 0.002, deadline=HOP_END)

        v = _max_cos(y)
        # decide LP probe
        do_lp = False
        if v < best_val:
            do_lp = True
        elif v < best_val + 0.012 and (trial % 2 == 0 or v < cur_val + 0.002):
            # limit frequency to save time
            do_lp = (rng.random() < 0.55)
        elif v < cur_val and rng.random() < 0.12:
            do_lp = True
        v_lp = v
        y_lp = y
        if do_lp:
            if time.time() > HOP_END - 1:
                break
            probe_dead = min(time.time() + 4.0, HOP_END + 2.0, BUDGET_END - 35)
            if probe_dead - time.time() > 0.8:
                y_lp = _lp_polish(y, probe_dead)
                v_lp = _max_cos(y_lp)
        if v_lp < best_val - 1e-12:
            best = y_lp.copy()
            best_val = v_lp
            cur = y_lp.copy()
            cur_val = v_lp
        elif v_lp < cur_val or rng.random() < np.exp(-(v_lp - cur_val) / max(Temp, 1e-6)):
            # accept as new basin (allow uphill)
            cur = y_lp.copy()
            cur_val = v_lp
        # early exit if big win
        if best_val < 0.2435:
            break
    # final long polish of best
    if time.time() < BUDGET_END - 2:
        best = _lp_polish(best, BUDGET_END - 1.0)
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
