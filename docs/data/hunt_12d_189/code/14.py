import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _pt_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)


def _hole_point(x, exclude_idx, rng, M=20000):
    # random candidates, pick minimizing max dot to others (excluding exclude_idx)
    R = rng.standard_normal((M, 12))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ x.T
    if exclude_idx is not None:
        # ignore self column by setting very low so it doesn't affect max
        # but we will overwrite that row position anyway
        D[:, exclude_idx] = -2.0
    vals = np.max(D, axis=1)
    j = int(np.argmin(vals))
    return R[j]


def _soft_anneal(x, steps, a0, a1, l0, l1, rng=None):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights focused on near-max
        # clip exponent for stability
        E = alpha * (G - m)
        # E <= 0, exp in (0,1]
        W = np.exp(E, out=E)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        # limit tangent norm to 1 for stability
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        np.clip(nr, 1.0, None, out=nr)
        Gt = Gt / nr
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    nv = n * d + 1
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        # cap pairs to bound linprog time: keep tightest 1500
        if np.count_nonzero(sel) > 1500:
            # select top 1500
            idx = np.argpartition(-c, 1500)[:1500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # if very little time left, break
        if time.time() > deadline - 1.0:
            break
    return x, best


def solve():
    t0 = time.time()
    hard_end = t0 + 173.0
    explore_end = t0 + 140.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n = 189

    trial = 0
    # precompute order each trial from current best
    while time.time() < explore_end:
        trial += 1
        x = best_x.copy()
        pm = _pt_max(x)
        order = np.argsort(pm)  # loose first
        mode = trial % 6
        if mode == 0:
            k = 1 + (trial // 6) % 3  # 1..3
            loose = order[:k]
            for idx in loose:
                x[idx] = _hole_point(x, idx, rng, M=20000)
            # tiny shake of tight hubs
            tight = order[-3:]
            x[tight] += 0.006 * rng.standard_normal((len(tight), 12))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 250, 25.0, 250.0, 0.22, 0.02
            glob = 0.004
        elif mode == 1:
            k = 2
            loose = order[:k]
            for idx in loose:
                x[idx] = _hole_point(x, idx, rng, M=15000)
            # shake all a bit
            x += 0.006 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 300, 30.0, 300.0, 0.25, 0.02
            glob = 0.0
        elif mode == 2:
            # single rattler teleport + medium anneal
            idx = order[0]
            x[idx] = _hole_point(x, idx, rng, M=25000)
            x += 0.003 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 200, 20.0, 200.0, 0.20, 0.02
            glob = 0.0
        elif mode == 3:
            # shake overloaded hubs
            tight = order[-6:]
            # reinsert worst hub into hole (topology change)
            idx = order[-1]
            if rng.random() < 0.5:
                x[idx] = _hole_point(x, idx, rng, M=20000)
            x[tight] += 0.015 * rng.standard_normal((len(tight), 12))
            x[tight] /= np.linalg.norm(x[tight], axis=1, keepdims=True)
            x += 0.002 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 350, 40.0, 350.0, 0.25, 0.015
            glob = 0.0
        elif mode == 4:
            # global small shake, no teleport (basin explore)
            x += (0.008 + 0.004 * ((trial // 6) % 3)) * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 300, 30.0, 250.0, 0.22, 0.02
            glob = 0.0
        else:
            # two random points teleported to holes
            picks = rng.choice(n, size=2, replace=False)
            for idx in picks:
                x[idx] = _hole_point(x, idx, rng, M=15000)
            x += 0.005 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, a0, a1, l0, l1 = 280, 25.0, 300.0, 0.24, 0.02
            glob = 0.0

        if glob != 0.0:
            x += glob * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # cheap anneal
        x = _soft_anneal(x, steps, a0, a1, l0, l1)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            best_x = x
            # early exit if big win? continue to polish
        # time check for timeout safety
        if time.time() > explore_end:
            break
        # if many trials, continue; no LP inside loop to save time

    # final LP polish of best
    try:
        xp, vp = _lp_polish(best_x, hard_end - 1.0)
        if vp < best_val and np.all(np.isfinite(xp)):
            best_x, best_val = xp, vp
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
