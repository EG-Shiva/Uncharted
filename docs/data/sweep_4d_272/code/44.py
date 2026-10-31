import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (nrm[:, 0] < 1e-12)
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


def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _greedy_fill(fixed, k, rng, M=8000):
    cand = rng.standard_normal((M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ fixed.T).max(axis=1)
    s_cur = s.copy()
    picks = []
    for _ in range(k):
        idx = int(np.argmin(s_cur))
        picks.append(idx)
        d = cand @ cand[idx]
        np.maximum(s_cur, d, out=s_cur)
        s_cur[idx] = np.inf
    return cand[picks]


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((272, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (272, 4):
        tmp = rng.standard_normal((272, 4))
        xb = tmp
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # Stage 0: a couple of cold random starts for a different basin
    for ci in range(2):
        if time.time() > t0 + 25:
            break
        xc = rng.standard_normal((272, 4))
        xc = _norm(xc)
        xc = _anneal(xc, steps=700, tau0=0.02, tau1=0.002, lr0=0.01, lr1=0.001)
        v = _max_cos(xc)
        if v < best_val + 0.006 and time.time() < deadline - 60:
            xp = _lp_polish(xc, seconds=4.0, deadline=deadline - 50)
            vp = _max_cos(xp)
            if vp < best_val:
                x_best, best_val = xp, vp

    # Stage 1: collective surgery + anneal basin hopping
    trial = 0
    ks = [16, 12, 20, 10, 24, 14, 18]
    while time.time() < t0 + 108 and time.time() < deadline - 55:
        trial += 1
        k = int(ks[trial % len(ks)])
        x = x_best.copy()
        n = 272
        if trial % 2 == 0:
            # random cap deletion
            u = rng.standard_normal(4)
            u /= np.linalg.norm(u)
            dots = x @ u
            victims = np.argsort(-dots)[:k]
        else:
            # worst-offender deletion (highest personal max)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            # pick k among top 2k randomly for diversity
            order = np.argsort(-pw)
            pool = order[: 2 * k]
            perm = rng.permutation(len(pool))[:k]
            victims = pool[perm]
        mask = np.ones(n, dtype=bool)
        mask[victims] = False
        fixed = x[mask]
        try:
            fill = _greedy_fill(fixed, k, rng, M=7000)
        except Exception:
            continue
        x[victims] = fill
        # jitter slightly to break symmetry
        x[victims] += 0.002 * rng.standard_normal((k, 4))
        x = _norm(x)
        # anneal whole config to blend patch
        if trial % 3 == 0:
            x = _anneal(x, steps=380, tau0=0.014, tau1=0.0018, lr0=0.009, lr1=0.0009)
        else:
            x = _anneal(x, steps=450, tau0=0.010, tau1=0.0012, lr0=0.007, lr1=0.0007)
        v = _max_cos(x)
        # promising? jam it briefly to see true value
        if v < best_val + 0.004:
            xp = _lp_polish(x, seconds=2.5, deadline=deadline - 45)
            vp = _max_cos(xp)
            if vp < best_val:
                x_best, best_val = xp, vp
        # also accept near-best annealed states as hopping base occasionally
        # (don't lose time: only continue from best)

    # Stage 2: final jam of best basin
    remain = deadline - time.time() - 2.0
    if remain > 5:
        x_best = _lp_polish(x_best, seconds=min(remain, 55.0), deadline=deadline - 1.0)
    x_best = _norm(x_best)
    return x_best
