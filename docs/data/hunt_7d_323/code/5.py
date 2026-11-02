import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t_deadline=None):
    """Trust-region LP polish, time limited."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_refine(x, iters=900, beta0=30.0, beta1=310.0, lr0=0.025, lr1=0.002, mu=0.7):
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(iters):
        frac = it / (iters - 1) if iters > 1 else 0.0
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = beta * (G - m)
        np.clip(A, -30.0, 0.0, out=A)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ x
        s = np.sum(grad * x, axis=1, keepdims=True)
        gt = (grad - s * x) / rs
        v = mu * v + (1.0 - mu) * gt
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _best_hole(others, rng, nsample=12000, refine_steps=250):
    """Find unit vector minimizing max dot to others."""
    n, d = others.shape
    cand = rng.standard_normal(size=(nsample, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = cand @ others.T
    vals = scores.max(axis=1)
    bi = int(np.argmin(vals))
    xk = cand[bi].copy()
    # single-point subgradient refinement (softmax)
    for it in range(refine_steps):
        frac = it / max(1, refine_steps - 1)
        beta = 40.0 + 260.0 * frac
        lr = 0.05 * (1 - frac) + 0.002 * frac
        dots = others @ xk
        mm = float(dots.max())
        a = beta * (dots - mm)
        np.clip(a, -30, 0, out=a)
        w = np.exp(a)
        w /= (w.sum() + 1e-12)
        g = w @ others  # (d,)
        g = g - float(g @ xk) * xk
        xk = xk - lr * g
        xk /= np.linalg.norm(xk)
    return xk


def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    EXPL_END = t0 + 138.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    best = _max_cos(xb)
    x_best = xb.copy()

    def contacts_of(x, tol=1e-7):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = np.sum(G > m - tol, axis=1) - 1  # exclude self? self is -2 so no
        # correction: diagonal is -2, not counted
        return cnt, m, G

    trial = 0
    while time.time() < EXPL_END:
        trial += 1
        mode = trial % 5
        cnt, mval, _ = contacts_of(x_best)
        x_try = x_best.copy()

        if mode == 1:
            # hub-to-hole: pick most overloaded point, reinsert to largest hole
            k = int(np.argmax(cnt))
            others = np.delete(x_try, k, axis=0)
            xk = _best_hole(others, rng, nsample=10000, refine_steps=200)
            x_try[k] = xk
            # tiny global jitter to allow topology change
            x_try += rng.standard_normal(size=x_try.shape) * 0.002
        elif mode == 2:
            # kick top hubs strongly + global tiny
            order = np.argsort(-cnt)
            hubs = order[:4]
            x_try[hubs] += rng.standard_normal(size=(len(hubs), d)) * 0.22
            x_try += rng.standard_normal(size=x_try.shape) * 0.003
        elif mode == 3:
            # random subset medium kick
            k = 18
            idx = rng.choice(n, size=k, replace=False)
            x_try[idx] += rng.standard_normal(size=(k, d)) * 0.08
        elif mode == 4:
            # worst-pair kick
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            ii, jj = divmod(ij, n)
            x_try[ii] += rng.standard_normal(size=d) * 0.12
            x_try[jj] += rng.standard_normal(size=d) * 0.12
            # also randomize rattlers (0-contact points)
            ratt = np.where(cnt <= 0)[0]
            if len(ratt) > 0:
                x_try[ratt] += rng.standard_normal(size=(len(ratt), d)) * 0.3
        else:
            # global jitter small
            x_try += rng.standard_normal(size=x_try.shape) * 0.012

        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # smooth annealed repulsion
        iters = 850 if mode != 1 else 700
        x_try = _smooth_refine(x_try, iters=iters)
        cur = _max_cos(x_try)
        if cur < best - 1e-12:
            # direct win without LP
            x_best, best = x_try, cur
            continue
        # promising -> short LP
        if cur < best + 0.006 and time.time() < EXPL_END:
            x_pol = _lp_polish(x_try, seconds=1.3, t_deadline=EXPL_END + 5)
            cur2 = _max_cos(x_pol)
            if cur2 < best - 1e-12:
                x_best, best = x_pol, cur2

        if time.time() > EXPL_END:
            break

    # final long polish of the best
    remain = DEADLINE - time.time()
    if remain > 6:
        x_best = _lp_polish(x_best, seconds=min(remain - 2.0, 28.0), t_deadline=DEADLINE - 1.0)

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
