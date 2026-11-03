import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _relax(X, steps, k0, k1, lr0, lr1, mu, rng=None, jitter=0.0):
    n, d = X.shape
    if jitter > 0 and rng is not None:
        X = X + jitter * rng.standard_normal(size=X.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    V = np.zeros_like(X)
    eye = np.eye(n)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = X @ X.T
        m = float(np.max(G - eye * 2.0))
        # avoid overflow: exp(k*(G-m)), G-m <=0
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = (E @ X) / S
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

def _void_point(others, rng, n_cand=3000, steps=150):
    n_o, d = others.shape
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = cand @ others.T
    best_i = int(np.argmin(sc.max(axis=1)))
    c = cand[best_i].copy()
    for it in range(steps):
        dots = others @ c
        m = float(dots.max())
        kk = 60.0
        w = np.exp(kk * (dots - m))
        ws = w.sum()
        if ws == 0 or not np.isfinite(ws):
            break
        w /= ws
        grad = w @ others
        gt = grad - float(grad @ c) * c
        step = 0.20 * (1.0 - 0.6 * it / steps)
        c = c - step * gt
        nc = float(np.linalg.norm(c))
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
        m = float(cmat.max())
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
    HOP_END = T0 + 122.0
    LP_END = T0 + BUDGET
    rng = np.random.default_rng(0)

    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
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

    def scores(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        per = G.max(axis=1)
        cnt = (G > mx - 1e-4).sum(axis=1)
        return per, cnt, mx

    it = 0
    T_start = 0.004
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - T0
        # linear cooling
        Temp = T_start * max(0.03, 1.0 - elapsed / 125.0)
        mode = it % 7
        y = cur.copy()
        try:
            if mode == 0:
                # fresh random basin, long relax
                y = rng.standard_normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _relax(y, steps=2200, k0=25.0, k1=500.0, lr0=0.06, lr1=0.008, mu=0.85)
            elif mode == 1:
                # small global shake
                sig = float(rng.choice([0.002, 0.005, 0.01, 0.02]))
                y = y + sig * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _relax(y, steps=1200, k0=40.0, k1=450.0, lr0=0.04, lr1=0.006, mu=0.9)
            elif mode == 2:
                # larger shake to escape
                sig = float(rng.choice([0.04, 0.08, 0.15]))
                # shake subset sometimes
                if rng.random() < 0.5:
                    per, cnt, mx = scores(y)
                    k = int(rng.integers(4, 12))
                    # mix loose and jammed
                    order = np.argsort(cnt)
                    victims = []
                    for _ in range(k):
                        if rng.random() < 0.5:
                            victims.append(int(order[int(rng.integers(0, 12))]))
                        else:
                            victims.append(int(order[int(rng.integers(n - 12, n))]))
                    victims = list(set(victims))
                    y[victims] += sig * rng.standard_normal(size=(len(victims), d))
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                else:
                    y = y + sig * rng.standard_normal(size=y.shape)
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _relax(y, steps=1500, k0=30.0, k1=400.0, lr0=0.05, lr1=0.007, mu=0.88)
            elif mode == 3 or mode == 5:
                # reinsert worst point(s)
                per, cnt, mx = scores(y)
                nw = 1 if rng.random() < 0.7 else 2
                worst = np.argsort(-per)[:nw + 2]
                # pick among worst randomly
                picks = [int(worst[int(rng.integers(0, len(worst)))]) for _ in range(nw)]
                picks = list(set(picks))
                for pidx in picks:
                    mask = np.ones(n, dtype=bool); mask[pidx] = False
                    c = _void_point(y[mask], rng, n_cand=3000, steps=120)
                    y[pidx] = c
                y = _relax(y, steps=1000, k0=50.0, k1=500.0, lr0=0.03, lr1=0.005, mu=0.9)
            else:
                # reinsert loose / jammed + small jitter
                per, cnt, mx = scores(y)
                order_lo = np.argsort(cnt)
                # loose candidates
                k = int(rng.integers(1, 3))
                picks = []
                for _ in range(k):
                    if rng.random() < 0.6:
                        picks.append(int(order_lo[int(rng.integers(0, 10))]))
                    else:
                        picks.append(int(np.argsort(-cnt)[int(rng.integers(0, 10))]))
                picks = list(set(picks))
                for pidx in picks:
                    mask = np.ones(n, dtype=bool); mask[pidx] = False
                    c = _void_point(y[mask], rng, n_cand=2500, steps=100)
                    # mix with old to keep some memory
                    alpha = float(rng.random() * 0.3)
                    c = (1 - alpha) * c + alpha * y[pidx]
                    c /= np.linalg.norm(c)
                    y[pidx] = c
                sig = 0.005
                y = y + sig * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _relax(y, steps=1100, k0=35.0, k1=450.0, lr0=0.035, lr1=0.006, mu=0.9)

            v = _max_cos(y)
            if v < cur_val:
                cur, cur_val = y, v
                if v < best_val:
                    best, best_val = y.copy(), v
            else:
                # SA acceptance
                if rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-9)):
                    cur, cur_val = y, v
                # keep best anyway if close? no
            # opportunistic: if we beat record by margin, polish early? continue
        except Exception:
            continue
        # safety: if time nearly up break
        if time.time() > HOP_END:
            break

    # final polishing of best (and cur if different and good)
    # polish best
    try:
        best = _lp_polish(best, LP_END - 2.0)
        best_val = _max_cos(best)
    except Exception:
        pass
    # if cur is better than best after polish attempts, ensure we return min
    try:
        if cur_val < best_val - 1e-12 and time.time() < LP_END - 10:
            cur2 = _lp_polish(cur, LP_END - 1.0)
            if _max_cos(cur2) < best_val:
                best = cur2
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
