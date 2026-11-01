import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, r0=0.01):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        # cap size to keep linprog fast
        if np.count_nonzero(sel) > 1500:
            thr = np.sort(c[sel])[-1500]
            sel = c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_anneal(x, steps, tau0, tau1, lr0, lr1, rng=None):
    # minimize soft-max of dots: fast projected gradient on sphere
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # avoid overflow
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (2.0 / S) * (E @ x)
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip step
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _worst_indices(x, k, tol=1e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    return np.argsort(-C)[:k], C


def solve():
    t0 = time.time()
    LIMIT = 172.0  # leave margin for return
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    # safety: fix shape
    if xb.shape != (71, 12):
        # try to reshape / truncate deterministically
        flat = xb.ravel()
        need = 71 * 12
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(71, 12)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    best_x = xb.copy()

    # quick tighten of starting point (5s)
    deadline = t0 + 5.0
    if deadline < t0 + LIMIT - 30:
        nx, nv = _lp_polish(best_x, deadline, r0=0.005)
        if nv < best_val and np.all(np.isfinite(nx)):
            best_x, best_val = nx, nv

    trial = 0
    # explore until ~ LIMIT-28s, reserve final polish
    while time.time() - t0 < LIMIT - 28.0:
        trial += 1
        if time.time() - t0 > LIMIT - 30:
            break
        x = best_x.copy()
        mode = trial % 6
        if mode == 0:
            sigma = 0.03
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.08
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            # shake random subset strongly
            k = 4
            idx = rng.choice(71, size=k, replace=False)
            x[idx] += 0.35 * rng.standard_normal((k, 12))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 3:
            # delete-and-reinsert worst points
            k = 2
            w, _ = _worst_indices(x, k, tol=2e-3)
            n, d = x.shape
            for j in w:
                cand = rng.standard_normal((3000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                other = np.delete(x, j, axis=0)
                scores = (cand @ other.T).max(axis=1)
                x[j] = cand[np.argmin(scores)]
        elif mode == 4:
            # reinsert random points
            k = 2
            idx = rng.choice(71, size=k, replace=False)
            n, d = x.shape
            for j in idx:
                cand = rng.standard_normal((3000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                other = np.delete(x, j, axis=0)
                scores = (cand @ other.T).max(axis=1)
                x[j] = cand[np.argmin(scores)]
        else:
            # small global shake + one-point reinsert of most loaded
            sigma = 0.015
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            w, _ = _worst_indices(x, 1, tol=2e-3)
            j = int(w[0])
            cand = rng.standard_normal((2000, 12))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            other = np.delete(x, j, axis=0)
            scores = (cand @ other.T).max(axis=1)
            # only replace if clearly better empty spot
            if scores.min() < (x[j] @ other.T).max() - 0.005:
                x[j] = cand[np.argmin(scores)]

        # coarse soft-max anneal (very fast)
        x = _smooth_anneal(x, steps=350, tau0=0.012, tau1=0.0025, lr0=0.35, lr1=0.05)
        v0 = _max_cos(x)
        # filter: only LP-polish promising ones
        # threshold adapts: allow up to best+0.02 early, tighter later
        if v0 > best_val + 0.02:
            continue
        # short LP to reach local optimum
        short_deadline = min(time.time() + 1.6, t0 + LIMIT - 28.0)
        nx, nv = _lp_polish(x, short_deadline, r0=0.012)
        if nv < best_val - 1e-12 and np.all(np.isfinite(nx)):
            best_x, best_val = nx, nv
        # also try extra anneal+lp second pass for very promising
        elif nv < best_val + 0.002 and (trial % 7 == 0):
            # second perturbation of rattlers / low-contact points
            G = nx @ nx.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-3).sum(axis=1)
            # shake low-contact points a bit and re-anneal
            low = np.where(cnt <= np.median(cnt))[0]
            if low.size > 0:
                k2 = min(5, low.size)
                sel = rng.choice(low, size=k2, replace=False)
                nx[sel] += 0.05 * rng.standard_normal((k2, 12))
                nx /= np.linalg.norm(nx, axis=1, keepdims=True)
                nx = _smooth_anneal(nx, steps=200, tau0=0.006, tau1=0.002, lr0=0.2, lr1=0.03)
                d2 = min(time.time() + 1.2, t0 + LIMIT - 28.0)
                nx2, nv2 = _lp_polish(nx, d2, r0=0.008)
                if nv2 < best_val - 1e-12 and np.all(np.isfinite(nx2)):
                    best_x, best_val = nx2, nv2

    # final deep polish of best
    final_deadline = t0 + LIMIT - 1.0
    best_x, best_val = _lp_polish(best_x, final_deadline, r0=0.004)
    # extra smooth + lp cycle if time remains
    if time.time() - t0 < LIMIT - 8.0:
        best_x = _smooth_anneal(best_x, steps=200, tau0=0.003, tau1=0.0015, lr0=0.05, lr1=0.01)
        best_x, best_val = _lp_polish(best_x, t0 + LIMIT - 0.5, r0=0.002)

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # ensure finite
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    # re-normalize any repaired rows
    nrm = np.linalg.norm(best_x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    best_x = best_x / nrm
    return best_x
