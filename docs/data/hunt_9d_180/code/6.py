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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_opt(X, T_init, steps, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    b1p = beta1
    b2p = beta2
    if steps > 1:
        decay = (lr1 / lr0) ** (1.0 / steps)
    else:
        decay = 1.0
    lr = lr0
    T = float(T_init)
    one_m_b1 = 1.0 - beta1
    one_m_b2 = 1.0 - beta2
    for t in range(1, steps + 1):
        G = X @ X.T
        W = G - T
        np.maximum(W, 0, out=W)
        np.fill_diagonal(W, 0.0)
        wmax = W.max()
        if wmax == 0.0:
            # feasible: lower target to keep pressure
            T -= 0.0008
            W = G - T
            np.maximum(W, 0, out=W)
            np.fill_diagonal(W, 0.0)
            # if still zero (far below), keep lowering next iter via update
            if W.max() == 0.0:
                # no gradient, just decay lr and continue
                lr *= decay
                b1p *= beta1
                b2p *= beta2
                # avoid stall: force small T drop handled next iter
                continue
        grad = W @ X
        s = np.sum(grad * X, axis=1, keepdims=True)
        g = grad - s * X
        # Adam
        m *= beta1
        m += one_m_b1 * g
        v *= beta2
        v += one_m_b2 * (g * g)
        denom1 = 1.0 - b1p
        denom2 = 1.0 - b2p
        mhat = m / denom1
        vhat = v / denom2
        upd = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * upd
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        lr *= decay
        b1p *= beta1
        b2p *= beta2
    return X


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    try:
        best = np.load("best.npy")
    except Exception:
        best = np.load("record.npy")
    best = np.asarray(best, dtype=np.float64)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    n, d = best.shape
    # safety: ensure shape (180,9)
    if best.shape != (180, 9):
        # try to reshape/interpolate deterministically
        tmp = rng.standard_normal(size=(180, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n, 180)
        tmp[:m, :min(d, 9)] = best[:m, :min(d, 9)]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        best = tmp
        n, d = best.shape
    best_val = _max_cos(best)

    it = 0
    # basin hopping phase, reserve time for final polish
    HOP_END = t0 + 143.0
    while time.time() < HOP_END:
        it += 1
        # time check for at least one more attempt (~0.4s)
        if time.time() > t0 + 142.0:
            break
        # choose kick size
        if it % 6 == 0:
            k = int(rng.integers(15, 31))
        elif it % 7 == 0:
            k = int(rng.integers(2, 5))
        else:
            k = int(rng.integers(4, 13))
        k = min(k, n - 1)
        sigma_small = float(10.0 ** rng.uniform(-2.2, -1.3))
        delta = float(rng.uniform(0.0015, 0.0045))
        steps = int(rng.integers(3000, 6001))
        lr0 = float(10.0 ** rng.uniform(-2.3, -1.8))
        lr1 = lr0 * float(10.0 ** rng.uniform(-1.5, -0.7))
        # per-point worst to find jammed points
        Gb = best @ best.T
        np.fill_diagonal(Gb, -2.0)
        worst = Gb.max(axis=1)
        idx = np.argpartition(worst, -k)[-k:]
        X0 = best.copy()
        X0 += sigma_small * rng.standard_normal(size=(n, d))
        X0[idx] = rng.standard_normal(size=(k, d))
        # occasional extra random rattler move
        if it % 3 == 0:
            r2 = rng.choice(n, size=2, replace=False)
            X0[r2] = rng.standard_normal(size=(2, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        T_init = best_val - delta
        # avoid T too low if best already very low
        Xopt = _smooth_opt(X0, T_init, steps, lr0, lr1)
        val = _max_cos(Xopt)
        if val < best_val - 5e-9:
            # short squeeze to maximize gain
            remain_short = 3.0
            # only polish if we have time
            if time.time() + remain_short < t0 + 150.0:
                Xopt = _lp_polish(Xopt, remain_short)
                val = _max_cos(Xopt)
            if val < best_val:
                best = Xopt
                best_val = val
        # opportunistic: if very close, tiny chance to polish into win
        # (skip to save time except rarely)
        if it % 25 == 0 and time.time() < HOP_END - 5:
            # try a pure small-shake local search without big kick
            Xs = best + 0.004 * rng.standard_normal(size=(n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = _smooth_opt(Xs, best_val - 0.001, 2500, 0.004, 0.0003)
            vs = _max_cos(Xs)
            if vs < best_val:
                if time.time() + 2.0 < t0 + 150.0:
                    Xs = _lp_polish(Xs, 2.0)
                    vs = _max_cos(Xs)
                if vs < best_val:
                    best = Xs
                    best_val = vs
    # final squeeze with remaining time
    now = time.time()
    # must finish within 180s, leave 2s margin
    remaining = 178.0 - (now - t0)
    if remaining > 3.0:
        # cap to avoid overrun, keep deterministic-ish
        remaining = min(remaining, 30.0)
        best = _lp_polish(best, remaining)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
