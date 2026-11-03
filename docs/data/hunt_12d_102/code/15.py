import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr, noise=0.0, rng=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0)
    l1 = np.log(T1)
    for s in range(1, steps + 1):
        frac = s / steps
        T = float(np.exp(l0 + (l1 - l0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        P = W / rs
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        if noise > 0.0:
            amp = noise * (1.0 - frac)
            N = rng.standard_normal((n, d)) if rng is not None else np.random.standard_normal((n, d))
            nd = np.sum(N * x, axis=1, keepdims=True)
            N = N - nd * x
            Gt = Gt + amp * N
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, budget, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= budget or now >= deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def greedy_holes(X_rem, nk, rng, n_samp=12000):
    d = X_rem.shape[1]
    cand = rng.standard_normal((n_samp, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cur = X_rem.copy()
    new_pts = []
    for _ in range(nk):
        dots = cand @ cur.T
        scores = dots.max(axis=1)
        bi = int(np.argmin(scores))
        q = cand[bi].copy()
        # quick local refine of q against cur (farthest point)
        for _ in range(80):
            dv = cur @ q
            mm = dv.max()
            w = np.exp((dv - mm) / 0.004)
            s = w.sum()
            if s == 0 or not np.isfinite(s):
                break
            g = (w[:, None] * cur).sum(axis=0) / s
            g = g - np.dot(g, q) * q
            gn = np.linalg.norm(g)
            if gn == 0 or not np.isfinite(gn):
                break
            q = q - 0.05 * g / (gn + 1e-12)
            q /= np.linalg.norm(q)
        new_pts.append(q)
        cur = np.vstack([cur, q[None, :]])
    return np.array(new_pts)


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    xb = None
    xr = None
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        pass
    try:
        xr = np.array(np.load("record.npy"), dtype=float)
    except Exception:
        pass
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                if a.shape != (102, 12):
                    continue
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(a)):
                    continue
                cands.append((max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    it = 0
    while time.time() < T_END - 22.0 and it < 500:
        it += 1
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        mode = int(rng.integers(0, 6))
        x_try = x_best.copy()
        if mode == 0:
            sig = 0.012
            x_try = x_best + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(600, 1100))
            T0 = float(rng.uniform(0.015, 0.025))
            T1 = float(rng.uniform(0.0025, 0.0045))
            lr = float(rng.uniform(0.0015, 0.0028))
            noise = 0.0
        elif mode == 1:
            k = int(rng.integers(2, 5))
            idx = order[:k]
            x_try[idx] += 0.11 * rng.standard_normal((k, d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps = int(rng.integers(800, 1300))
            T0 = float(rng.uniform(0.018, 0.03))
            T1 = float(rng.uniform(0.002, 0.004))
            lr = float(rng.uniform(0.0018, 0.0026))
            noise = 0.001
        elif mode == 2:
            ci = int(rng.integers(n))
            Gr = (x_best @ x_best[ci])
            Gr[ci] = -2.0
            nbr = np.argsort(-Gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.07 * rng.standard_normal((len(cl), d))
            x_try += 0.002 * rng.standard_normal((n, d))
            steps = int(rng.integers(700, 1200))
            T0 = float(rng.uniform(0.015, 0.028))
            T1 = float(rng.uniform(0.0025, 0.004))
            lr = float(rng.uniform(0.0015, 0.0025))
            noise = 0.0005
        elif mode == 3:
            nk = int(rng.integers(1, 4))
            worst = order[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xrem = x_best[mask]
            new_pts = greedy_holes(Xrem, nk, rng, n_samp=12000)
            for ii, wi in enumerate(worst):
                x_try[wi] = new_pts[ii]
            x_try[mask] += 0.002 * rng.standard_normal(((n - nk), d))
            steps = int(rng.integers(800, 1300))
            T0 = float(rng.uniform(0.02, 0.03))
            T1 = float(rng.uniform(0.002, 0.0035))
            lr = float(rng.uniform(0.0018, 0.0026))
            noise = 0.0
        elif mode == 4:
            nk = int(rng.integers(4, 8))
            idx = rng.choice(n, size=nk, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xrem = x_best[mask]
            new_pts = greedy_holes(Xrem, nk, rng, n_samp=10000)
            for ii, wi in enumerate(idx):
                x_try[wi] = new_pts[ii]
            steps = int(rng.integers(900, 1400))
            T0 = float(rng.uniform(0.02, 0.032))
            T1 = float(rng.uniform(0.002, 0.004))
            lr = float(rng.uniform(0.0018, 0.0028))
            noise = 0.0
        else:
            sig = 0.04
            x_try = x_best + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(900, 1400))
            T0 = float(rng.uniform(0.02, 0.03))
            T1 = 0.003
            lr = float(rng.uniform(0.002, 0.003))
            noise = 0.0015
        nr = np.linalg.norm(x_try, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x_try /= nr
        x_q = adam_quench(x_try, steps, T0, T1, lr, noise=noise, rng=rng)
        cur = max_cos(x_q)
        if cur < best - 1e-12:
            # direct win without LP
            best = cur
            x_best = x_q.copy()
            # snap it immediately
            if time.time() < T_END - 22.0:
                x_p = lp_polish(x_best, 1.0, T_END - 20.0)
                vp = max_cos(x_p)
                if vp < best:
                    best, x_best = vp, x_p
            continue
        if cur < best + 0.01 and time.time() < T_END - 22.0:
            x_p = lp_polish(x_q, 0.9, T_END - 20.0)
            vp = max_cos(x_p)
            if vp < best - 1e-12:
                best, x_best = vp, x_p.copy()
    remain = T_END - time.time()
    if remain > 4.0:
        x_best = lp_polish(x_best, remain - 1.0, T_END)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
