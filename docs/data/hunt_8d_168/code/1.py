import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _e8_roots():
    pts = []
    # D8 part: (+-1,+-1,0*6)
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si; v[j] = sj
                    pts.append(v)
    # 128 half-integer with even number of minus
    # generate deterministically via counter
    for mask in range(256):
        signs = np.ones(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -1.0
                nm += 1
        if nm % 2 == 0:
            pts.append(0.5 * signs)
    a = np.array(pts, dtype=float)  # 240 x 8, norm sqrt(2)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(0)
    n, d = 168, 8
    # load starts
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        best = None
    try:
        rec = np.load("record.npy").astype(float)
    except Exception:
        rec = None
    cands = []
    if best is not None and best.shape == (n, d):
        cands.append(best)
    if rec is not None and rec.shape == (n, d):
        cands.append(rec)
    if len(cands) == 0:
        cands.append(rng.normal(size=(n, d)))
    # normalize
    for i in range(len(cands)):
        c = cands[i]
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        cands[i] = c
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)

    e8 = _e8_roots()

    def soft_run(x_init, steps, a0, a1, lr0, lr1, mu):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            # time guard inside long runs
            if s % 500 == 0 and time.time() - t0 > BUDGET - 35:
                break
            frac = s / max(steps - 1, 1)
            alpha = a0 + (a1 - a0) * frac
            lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # weights focused on large dots
            W = np.exp(alpha * (G - m))
            np.fill_diagonal(W, 0.0)
            S = W.sum(axis=1, keepdims=True) + 1e-12
            A = (W @ x) / S
            dot = np.sum(A * x, axis=1, keepdims=True)
            At = A - dot * x
            vel = mu * vel + At
            x = x - lr * vel
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    trial = 0
    # quick E8-subset trials first (structured different basin)
    for e8trial in range(3):
        if time.time() - t0 > BUDGET - 45:
            break
        idx = rng.choice(240, size=n, replace=False)
        x0 = e8[idx] + rng.normal(size=(n, d)) * 0.02
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        xopt = soft_run(x0, 2500, 40.0, 160.0, 0.06, 0.004, 0.85)
        v = _max_cos(xopt)
        if v < best_val:
            best_val = v
            best_x = xopt
        trial += 1

    # main basin-hopping loop
    while time.time() - t0 < BUDGET - 40:
        trial += 1
        r = rng.random()
        if r < 0.12:
            # fresh random start
            x0 = rng.normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(2500, 4500))
            a0 = float(rng.uniform(25, 50)); a1 = float(rng.uniform(120, 200))
            lr0 = float(rng.uniform(0.04, 0.09)); lr1 = float(rng.uniform(0.002, 0.008))
            mu = 0.85
        elif r < 0.52:
            # global jitter of current best
            scale = 10.0 ** float(rng.uniform(-2.6, -0.6))
            x0 = best_x + rng.normal(size=(n, d)) * scale
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1200, 3000))
            a0 = float(rng.uniform(30, 60)); a1 = float(rng.uniform(100, 200))
            lr0 = float(rng.uniform(0.03, 0.08)); lr1 = float(rng.uniform(0.002, 0.01))
            mu = float(rng.uniform(0.7, 0.9))
        elif r < 0.82:
            # subset re-insertion
            k = int(rng.integers(3, 16))
            x0 = best_x.copy()
            sel = rng.choice(n, size=k, replace=False)
            x0[sel] = rng.normal(size=(k, d))
            # small jitter elsewhere
            x0 += rng.normal(size=(n, d)) * 0.005
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3500))
            a0 = float(rng.uniform(30, 60)); a1 = float(rng.uniform(120, 220))
            lr0 = float(rng.uniform(0.04, 0.09)); lr1 = float(rng.uniform(0.002, 0.008))
            mu = float(rng.uniform(0.75, 0.9))
        else:
            # overloaded-point shake
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            thr = m - 0.004
            cnt = (G > thr).sum(axis=1)
            k = int(rng.integers(4, 13))
            order = np.argsort(-cnt)
            # mix top with random to keep determinism but diversity
            pool = order[:30]
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            x0 = best_x.copy()
            # heavy kick on selected
            s_big = float(rng.uniform(0.15, 0.5))
            x0[sel] += rng.normal(size=(len(sel), d)) * s_big
            x0 += rng.normal(size=(n, d)) * 0.008
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1500, 3000))
            a0 = float(rng.uniform(30, 70)); a1 = float(rng.uniform(120, 220))
            lr0 = float(rng.uniform(0.03, 0.07)); lr1 = float(rng.uniform(0.002, 0.007))
            mu = float(rng.uniform(0.75, 0.9))
        xopt = soft_run(x0, steps, a0, a1, lr0, lr1, mu)
        v = _max_cos(xopt)
        if v < best_val - 1e-12:
            best_val = v
            best_x = xopt
            # intensify around new best with short anneal
            if time.time() - t0 < BUDGET - 50:
                xopt2 = soft_run(best_x, 1200, 80.0, 220.0, 0.02, 0.002, 0.8)
                v2 = _max_cos(xopt2)
                if v2 < best_val:
                    best_val = v2
                    best_x = xopt2
        # safety: stop if time low
        if time.time() - t0 > BUDGET - 40:
            break

    remain = BUDGET - (time.time() - t0) - 2.0
    if remain > 5:
        # final LP polish with most of remaining time
        best_x = _lp_polish(best_x, min(remain, 38.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
