import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _smooth_opt(x, schedule):
    n = x.shape[0]
    x = _normalize(x)
    for K, steps, lr in schedule:
        for _ in range(steps):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # stable weights exp(K*(dot-m))
            W = np.exp(K * (G - m))
            np.fill_diagonal(W, 0.0)
            S = float(np.sum(W) * 0.5)
            if S <= 0:
                continue
            Grad = (W @ x) / S
            d = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - d * x
            x = x - lr * Gt
            x = _normalize(x)
    return x


def _hard_refine(x, iters=250, lr=0.02, tol0=0.01, tol1=0.001):
    n = x.shape[0]
    x = _normalize(x)
    for it in range(iters):
        tol = tol0 + (tol1 - tol0) * (it / max(1, iters - 1))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        A = (G > m - tol).astype(np.float64)
        # no self
        np.fill_diagonal(A, 0.0)
        deg = np.sum(A, axis=1, keepdims=True)
        if np.max(deg) <= 0:
            break
        H = A @ x
        hd = np.sum(H * x, axis=1, keepdims=True)
        Ht = H - hd * x
        Ht = Ht / np.maximum(deg, 1.0)
        # clip large steps
        x = x - lr * Ht
        x = _normalize(x)
    return x


def _best_void(Y, rng, n_sample=25000, iters=250):
    # Y: (m,4) existing points, find z minimizing max_j z.Y_j
    m = Y.shape[0]
    Z = rng.standard_normal(size=(n_sample, 4))
    Z = _normalize(Z)
    dots = Z @ Y.T  # (n_sample,m)
    worst = np.max(dots, axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx:idx+1, :].copy()  # (1,4)
    z = z / np.linalg.norm(z)
    lr = 0.05
    for it in range(iters):
        d = (Y @ z.ravel())  # (m,)
        k = 4 if m >= 4 else m
        # indices of largest dots
        part = np.argpartition(d, -k)[-k:]
        H = np.mean(Y[part, :], axis=0, keepdims=True)
        hd = float(H @ z.ravel())
        Ht = H - hd * z
        cur_lr = lr * (1.0 - 0.8 * it / max(1, iters))
        z = z - cur_lr * Ht
        z = z / np.linalg.norm(z)
    return z.ravel()


def _contacts(x, thresh=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - thresh)
    deg = np.sum(C, axis=1)
    return m, deg, G


def _lp_polish(x, deadline):
    x = _normalize(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(12345)
    np.random.seed(0)

    # load starting points
    candidates = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (163, 4):
                a = _normalize(a)
                # remove nan
                if np.all(np.isfinite(a)):
                    candidates.append(a)
        except Exception:
            pass
    if len(candidates) == 0:
        a = _normalize(rng.standard_normal(size=(163, 4)))
        candidates.append(a)
    best = min(candidates, key=_max_cos)
    best_val = _max_cos(best)

    sched_light = [(25.0, 250, 0.07), (90.0, 300, 0.035), (300.0, 350, 0.018), (800.0, 300, 0.010)]
    sched_medium = [(30.0, 300, 0.06), (120.0, 350, 0.03), (450.0, 450, 0.015)]

    it = 0
    # precompute contacts for targeted moves
    while time.time() < EXPLORE_END:
        it += 1
        mode = it % 7
        # refresh contact info every few iters from best
        try:
            _, deg, Gfull = _contacts(best, thresh=1e-7)
        except Exception:
            deg = np.ones(163)
        if mode == 0:
            sigma = 0.025
            y = best + sigma * rng.standard_normal(size=best.shape)
            y = _normalize(y)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=180, lr=0.02, tol0=0.008, tol1=0.0008)
        elif mode == 1:
            sigma = 0.075
            y = best + sigma * rng.standard_normal(size=best.shape)
            y = _normalize(y)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=180, lr=0.02, tol0=0.008, tol1=0.0008)
        elif mode == 2:
            # targeted: perturb most-contacted points strongly
            order = np.argsort(-deg)
            k = 10
            tgt = order[:k]
            noise = 0.008 * rng.standard_normal(size=best.shape)
            noise[tgt, :] += 0.16 * rng.standard_normal(size=(k, 4))
            # also nudge rattlers (deg~0)
            ratt = np.where(deg <= 0.5)[0]
            for r in ratt:
                noise[r, :] += 0.20 * rng.standard_normal(size=(4,))
            y = _normalize(best + noise)
            y = _smooth_opt(y, sched_medium)
            y = _hard_refine(y, iters=220, lr=0.018, tol0=0.006, tol1=0.0005)
        elif mode == 3:
            # remove worst point and reinsert in void
            order = np.argsort(-deg)
            # pick among top 5 worst randomly
            pick = order[int(rng.integers(0, min(5, len(order))))]
            mask = np.ones(163, dtype=bool)
            mask[pick] = False
            Y = best[mask]
            z = _best_void(Y, rng, n_sample=20000, iters=200)
            y = best.copy()
            y[pick] = z
            # small shake of neighbours that were in contact with pick
            try:
                row = Gfull[pick].copy()
                # neighbours with large dot
                nbr = np.argsort(-row)[:12]
                for nb in nbr:
                    if nb == pick:
                        continue
                    y[nb] = y[nb] + 0.02 * rng.standard_normal(size=(4,))
            except Exception:
                pass
            y = _normalize(y)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=200, lr=0.02, tol0=0.007, tol1=0.0007)
        elif mode == 4:
            # reinsert rattler(s) to best void, small global shake
            ratt = np.where(deg <= 0.5)[0]
            y = best + 0.012 * rng.standard_normal(size=best.shape)
            y = _normalize(y)
            if len(ratt) > 0:
                for r in ratt[:2]:
                    mask = np.ones(163, dtype=bool)
                    mask[r] = False
                    Y = y[mask]
                    z = _best_void(Y, rng, n_sample=20000, iters=200)
                    y[r] = z
                y = _normalize(y)
            else:
                # no rattler: pick random low-degree point to relocate
                order = np.argsort(deg)
                r = order[int(rng.integers(0, 8))]
                mask = np.ones(163, dtype=bool)
                mask[r] = False
                z = _best_void(y[mask], rng, n_sample=20000, iters=200)
                y[r] = z
                y = _normalize(y)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=180, lr=0.02, tol0=0.008, tol1=0.0008)
        elif mode == 5:
            sigma = 0.16
            y = best + sigma * rng.standard_normal(size=best.shape)
            y = _normalize(y)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=150, lr=0.022, tol0=0.01, tol1=0.001)
        else:
            # moderate subset shake
            k = int(rng.integers(4, 14))
            idx = rng.choice(163, size=k, replace=False)
            noise = 0.006 * rng.standard_normal(size=best.shape)
            noise[idx] += 0.10 * rng.standard_normal(size=(k, 4))
            y = _normalize(best + noise)
            y = _smooth_opt(y, sched_light)
            y = _hard_refine(y, iters=180, lr=0.02, tol0=0.008, tol1=0.0008)

        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-14:
            best = y
            best_val = v
            # quick opportunistic LP squeeze if promising and time allows (2s)
            if time.time() < EXPLORE_END - 8:
                try:
                    y2 = _lp_polish(best, min(time.time() + 2.0, EXPLORE_END))
                    v2 = _max_cos(y2)
                    if v2 < best_val:
                        best, best_val = y2, v2
                except Exception:
                    pass
        # safety: stop if time nearly up
        if time.time() >= EXPLORE_END:
            break

    # final polish with remaining time
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 3:
            best = _lp_polish(best, time.time() + remain)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = _normalize(best)
    return best
