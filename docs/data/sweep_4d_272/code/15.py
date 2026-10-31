import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm < 1e-12] = 1.0
    x = x / nrm
    bad = ~np.isfinite(x).all(axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _anneal(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _find_void(x_fixed, rng, n_samples=8000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    idx = int(best.argmin())
    b = cand[idx].copy()
    bs = float(best[idx])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ x_fixed.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j].copy(), float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b

def _random_rot(rng):
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q

def _crossover(a, b, rng):
    R = _random_rot(rng)
    br = b @ R
    u = rng.standard_normal(4)
    u /= np.linalg.norm(u)
    da = a @ u
    db = br @ u
    # complementary hemispheres, random split
    k = int(rng.integers(100, 173))
    ia = np.argsort(-da)[:k]
    ib = np.argsort(db)[:272 - k]
    child = np.concatenate([a[ia], br[ib]], axis=0)
    # tiny jitter to break symmetry
    child = child + 1e-3 * rng.standard_normal(child.shape)
    return _normalize(child)

def _lp_polish(x, seconds, deadline):
    x = _normalize(x)
    best = _max_cos(x)
    n, d = x.shape
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
        y = _normalize(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _normalize(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        x0 = _normalize(tmp)
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    T = 0.0012
    trial = 0
    # exploration budget ~ up to deadline-30s
    while time.time() < t0 + 140.0 and time.time() < deadline - 32.0:
        trial += 1
        r = rng.random()
        if r < 0.32:
            sigma = float(rng.choice(np.array([0.008, 0.015, 0.03, 0.06])))
            y = cur + sigma * rng.standard_normal((272, 4))
            y = _normalize(y)
            tau0, tau1 = (0.015, 0.0012) if trial % 3 == 0 else ((0.03, 0.002) if trial % 3 == 1 else (0.008, 0.0007))
            lr0, lr1 = 0.01, 0.0008
            steps = 280
            y = _anneal(y, steps, tau0, tau1, lr0, lr1)
        elif r < 0.58:
            y = cur.copy()
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            order = np.argsort(-pw)
            nk = int(rng.integers(2, 5))
            # pick nk among top-8 randomly
            pool = order[:8]
            victims = rng.choice(pool, size=nk, replace=False)
            for wi in victims:
                others = np.delete(y, int(wi), axis=0)
                y[int(wi)] = _find_void(others, rng, n_samples=6000)
                # update y for next victim selection marginally
            y = _anneal(y, 320, 0.02, 0.001, 0.012, 0.0008)
        elif r < 0.80:
            a = best
            b = cur if rng.random() < 0.7 else best + 0.04 * rng.standard_normal((272, 4))
            b = _normalize(b)
            y = _crossover(a, b, rng)
            y = _anneal(y, 350, 0.025, 0.0012, 0.012, 0.0008)
        else:
            y = cur.copy()
            u = rng.standard_normal(4)
            u /= np.linalg.norm(u)
            dots = y @ u
            k = int(rng.integers(12, 36))
            idx = np.argsort(-dots)[:k]
            sig = float(rng.choice(np.array([0.15, 0.25, 0.35])))
            y[idx] = y[idx] + sig * rng.standard_normal((k, 4))
            y = _normalize(y)
            y = _anneal(y, 300, 0.02, 0.001, 0.01, 0.0008)
        new_val = _max_cos(y)
        if new_val < best_val:
            best, best_val = y.copy(), new_val
            cur, cur_val = y.copy(), new_val
        elif new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
            cur, cur_val = y.copy(), new_val
        # periodic restart of walker to best if drifted too far
        if trial % 25 == 0 and cur_val > best_val + 0.004:
            cur, cur_val = best.copy(), best_val
        # decay temperature slightly
        T *= 0.9995
        if T < 0.0003:
            T = 0.0003
    # final tightening of best
    remain = deadline - time.time()
    if remain > 5:
        best = _lp_polish(best, min(remain - 1.0, 28.0), deadline - 1.0)
    best = _normalize(best)
    return best
