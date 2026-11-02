import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=8e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2 * d] = J[:, None] * d + ar
        Cb[:, 2 * d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contacts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _greedy_reinsert(y, to_replace, rng, n_cand=4000):
    n, d = y.shape
    s = set(int(v) for v in to_replace)
    for k in to_replace:
        k = int(k)
        mask = np.ones(n, dtype=bool)
        mask[k] = True
        # exclude other to-be-replaced not yet placed? use current y but with
        # not-yet-replaced still present; better use fixed + already placed
        # simple: use all others as they currently are
        others = y[mask]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    # tiny local hill climb on replaced
    for k in to_replace:
        k = int(k)
        mask = np.ones(n, dtype=bool)
        mask[k] = True
        others = y[mask]
        cur = y[k].copy()
        cur_w = float((others @ cur).max())
        trial = rng.standard_normal(size=(500, d)) * 0.04 + cur[None, :]
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        dw = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cur_w:
            y[k] = trial[bi]
    return y

def _anneal(x, steps, a0, a1, l0, l1, mu):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        t = k / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** t if a0 > 0 else a1
        lr = l0 * (l1 / l0) ** t if l0 > 0 else l1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # weights, stable (G-m<=0)
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        s = np.sum(grad * x, axis=1, keepdims=True)
        grad -= s * x
        nr = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= nr
        vel = mu * vel - lr * grad
        x = x + vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < b0:
                    b0 = v
                    x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape
    it = 0
    no_improve = 0
    HOP_END = 148.0
    while True:
        if time.time() - t0 > HOP_END:
            break
        remain = HOP_END - (time.time() - t0)
        if remain < 3.0:
            break
        s = it % 7
        x = x_best.copy()
        # propose move
        if s == 0:
            sig = 0.022 * (1.0 + 0.05 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 600, 30.0, 220.0, 0.008, 0.0006, 0.85
            budget, init_r = 2.5, 0.008
        elif s == 1:
            w = _worst(x)
            k = 35
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.075
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 700, 30.0, 250.0, 0.007, 0.0005, 0.85
            budget, init_r = 2.5, 0.008
        elif s == 2:
            cc = _contacts(x, best, tol=1e-9)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = [int(r) for r in ratt]
            for o in order:
                if len(torep) >= 7:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:7]
            x = _greedy_reinsert(x, torep, rng, n_cand=4000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 900, 25.0, 200.0, 0.006, 0.0004, 0.88
            budget, init_r = 3.0, 0.008
        elif s == 3:
            # cluster surgery around worst hub
            w = _worst(x)
            c0 = int(np.argmax(w))
            dots = x[c0] @ x.T
            nn = np.argsort(-dots)[:9].tolist()
            x = _greedy_reinsert(x, [int(v) for v in nn], rng, n_cand=4000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 900, 25.0, 220.0, 0.007, 0.0004, 0.86
            budget, init_r = 3.0, 0.008
        elif s == 4:
            k = 32
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.06
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 600, 35.0, 250.0, 0.008, 0.0006, 0.82
            budget, init_r = 2.5, 0.009
        elif s == 5:
            w = _worst(x)
            k = 14
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.12
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 700, 30.0, 260.0, 0.008, 0.0005, 0.85
            budget, init_r = 2.5, 0.009
        else:
            x = x + rng.standard_normal(size=x.shape) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1, mu = 500, 40.0, 250.0, 0.005, 0.0004, 0.85
            budget, init_r = 2.2, 0.005
        budget = min(budget, remain - 0.2)
        if budget < 0.8:
            break
        try:
            x = _anneal(x, steps, a0, a1, l0, l1, mu)
        except Exception:
            it += 1
            continue
        # time left for LP?
        remain2 = HOP_END - (time.time() - t0)
        if remain2 < 0.8:
            # evaluate annealed point without LP
            v = _max_cos(x)
            if v < best - 1e-12:
                best = v
                x_best = x
                no_improve = 0
            it += 1
            continue
        try:
            y = _lp_polish(x, seconds=min(budget, remain2), init_radius=init_r, cap=2000)
        except Exception:
            it += 1
            no_improve += 1
            continue
        v = _max_cos(y)
        if v < best - 1e-12:
            best = v
            x_best = y
            no_improve = 0
        else:
            no_improve += 1
        it += 1
    remain = 178.0 - (time.time() - t0)
    if remain > 2.0:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain, 22.0), init_radius=2e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
