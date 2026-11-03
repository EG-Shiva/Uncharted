import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 6e-4:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c_all, -2800)[-2800]
            sel = c_all >= kth
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_exp(x, steps, a0, a1, s0, rng=None):
    n, d = x.shape
    for k in range(steps):
        frac = k / max(1, steps - 1)
        alpha = a0 + (a1 - a0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(alpha * (G - gmax))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        step = s0 * (1.0 - 0.4 * frac) / nrm
        x = x - step * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def deepest_hole(x, exclude, rng, n_cand=8000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[np.atleast_1d(exclude)] = False
    Xo = x[mask]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xo.T
    vals = dots.max(axis=1)
    return cand[int(np.argmin(vals))]


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    x_best = cand[bi].copy()
    v_best = scores[bi]
    # current walk state
    xc = x_best.copy()
    bc = v_best
    # no initial LP: parent already locally optimal; save time
    it = 0
    # explore until reserve for final polish
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 34.0:
            break
        # stop exploring a bit earlier to guarantee final polish time
        if time.time() - t0 > 132.0:
            break
        it += 1
        # contact stats of walk state
        Gc = xc @ xc.T
        np.fill_diagonal(Gc, -2.0)
        vmax = float(Gc.max())
        # degrees for bias (pairs within 0.01 of max)
        deg = np.sum(Gc > vmax - 0.01, axis=1).astype(float)
        # per-point max (for rattler detection)
        prow = Gc.max(axis=1)
        order_lo = np.argsort(prow)  # smallest first = loosest
        order_hi = np.argsort(-deg)  # most overloaded first

        y = xc.copy()
        mode = it % 8
        if mode == 0:
            # small global jitter
            y = y + rng.standard_normal(y.shape) * 0.006
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=280, a0=40.0, a1=160.0, s0=0.006)
        elif mode == 1:
            # shake overloaded cluster
            k = int(rng.integers(4, 10))
            # pick one of top-15 overloaded as center
            c0 = int(order_hi[int(rng.integers(0, 15))])
            dists = Gc[c0]
            neigh = np.argsort(-dists)[:25]
            pool = np.concatenate([[c0], neigh])
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
            y[idx] += rng.standard_normal((len(idx), d)) * float(rng.uniform(0.03, 0.07))
            y += rng.standard_normal(y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=420, a0=60.0, a1=220.0, s0=0.005)
        elif mode == 2 or mode == 3:
            # teleport 1-2 loosest points to deepest holes
            nt = 1 if mode == 2 else 2
            moving = order_lo[:nt].copy()
            # add small random among 10 loosest for diversity
            if rng.random() < 0.5:
                moving[0] = int(order_lo[int(rng.integers(0, 10))])
            for j in moving:
                try:
                    h = deepest_hole(y, exclude=j, rng=rng, n_cand=8000)
                    y[j] = h
                except Exception:
                    y[j] = rng.standard_normal(d)
                    y[j] /= np.linalg.norm(y[j])
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=500, a0=50.0, a1=250.0, s0=0.005)
        elif mode == 4:
            # random subset medium shake
            k = int(rng.integers(6, 16))
            probs = (deg + 2.0)
            probs /= probs.sum()
            idx = rng.choice(n, size=k, replace=False, p=probs)
            y[idx] += rng.standard_normal((k, d)) * float(rng.uniform(0.02, 0.05))
            y += rng.standard_normal(y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=400, a0=50.0, a1=200.0, s0=0.006)
        elif mode == 5:
            # large kick to change basin
            y = y + rng.standard_normal(y.shape) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=350, a0=30.0, a1=150.0, s0=0.007)
        elif mode == 6:
            # hole + cluster shake combo
            j = int(order_lo[int(rng.integers(0, 8))])
            try:
                y[j] = deepest_hole(y, exclude=j, rng=rng, n_cand=8000)
            except Exception:
                pass
            c0 = int(order_hi[int(rng.integers(0, 10))])
            idx2 = np.argsort(-Gc[c0])[:6]
            y[idx2] += rng.standard_normal((len(idx2), d)) * 0.03
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=500, a0=60.0, a1=260.0, s0=0.005)
        else:
            # fine jitter + long smooth
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_exp(y, steps=300, a0=80.0, a1=300.0, s0=0.004)

        v = max_cos(y)
        # promising -> short LP to see true basin value (rare)
        if v < v_best - 2e-6:
            rem2 = BUDGET - (time.time() - t0)
            if rem2 > 38.0:
                try:
                    y2 = lp_polish(y, seconds=min(3.0, rem2 - 34.0), init_radius=0.004)
                    v2 = max_cos(y2)
                except Exception:
                    y2, v2 = y, v
                if v2 < v_best:
                    x_best, v_best = y2.copy(), v2
                    xc, bc = y2.copy(), v2
                    continue
                # even if not global best, maybe accept as walk
                if v2 < bc + 1e-4:
                    xc, bc = y2.copy(), v2
                    continue
        # Metropolis walk (small temperature to hop basins)
        T = 3e-04 * max(0.2, 1.0 - (time.time() - t0) / 140.0)
        if v < bc or rng.random() < np.exp(-(v - bc) / T):
            xc, bc = y, v
            # keep walk from drifting too bad
            if bc > v_best + 0.004:
                xc, bc = x_best.copy(), v_best

    # final polish of global best with remaining budget
    rem = BUDGET - (time.time() - t0)
    if rem > 2.0:
        try:
            x_best = lp_polish(x_best, seconds=min(rem - 1.0, 32.0), init_radius=2e-3)
            v_best = max_cos(x_best)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
