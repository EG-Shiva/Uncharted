import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1), np.max(g)


def random_sphere(rng, n, d):
    y = rng.standard_normal((n, d))
    return _norm(y)


def smooth_descent(x_init, steps, T0, T1, lr0, lr1, ms0, ms1, beta=0.8, deadline=None):
    x = _norm(x_init.copy())
    n, d = x.shape
    v = np.zeros_like(x)
    logT0, logT1 = np.log(T0), np.log(T1)
    loglr0, loglr1 = np.log(lr0), np.log(lr1)
    logms0, logms1 = np.log(ms0), np.log(ms1)
    for s in range(steps):
        if deadline is not None and s % 200 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(logT0 + frac * (logT1 - logT0)))
        lr = float(np.exp(loglr0 + frac * (loglr1 - loglr0)))
        ms = float(np.exp(logms0 + frac * (logms1 - logms0)))
        G = x @ x.T
        m = float(np.max(G - 2.0 * np.eye(n)))
        # weights exp((G-m)/T), diag excluded
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) / 2.0)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # momentum
        v = beta * v + Gt
        step = lr * v
        # per-row clip
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        scale = np.minimum(1.0, ms / sn)
        step = step * scale
        x = x - step
        x = _norm(x)
    return x


def lp_polish(x, seconds, deadline=None):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # precompute iu arrays
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if deadline is not None and now > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # cols: I*d+k, J*d+k, last
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        # interleave per row: need order matching rows repeat
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::2 * d + 1] = I * d + 0
        # easier: build via concatenation per block then reshape trick
        # rebuild correctly:
        col_block = np.concatenate([I[:, None] * d + np.arange(d),
                                    J[:, None] * d + np.arange(d),
                                    np.full((p, 1), n * d)], axis=1).ravel()
        cols = col_block
        vals_block = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals_block, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 272, 4
    # load start
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = random_sphere(rng, n, d)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (n, d):
        # try to reshape / truncate
        if x0.size == n * d:
            x0 = x0.reshape(n, d)
        else:
            x0 = random_sphere(rng, n, d)
    x0 = _norm(x0)
    best = x0
    best_val = max_cos(best)
    # quick initial LP touch (already at optimum, keep short)
    # main basin-hopping
    trial = 0
    # parameter cycles
    while True:
        if _time.time() > HARD_END - 12:
            break
        trial += 1
        mode = trial % 6
        y = best.copy()
        if mode == 1:
            sigma = 0.008
            y = _norm(y + sigma * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 800, 0.02, 0.004, 0.25, 0.03, 0.04, 0.006
            lp_sec = 2.0
        elif mode == 2:
            sigma = 0.03
            y = _norm(y + sigma * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 1000, 0.02, 0.003, 0.3, 0.02, 0.05, 0.005
            lp_sec = 2.5
        elif mode == 3:
            # worst-point surgery k=6..12
            k = int(6 + (trial % 3) * 3)
            pm, m = per_point_max(y)
            worst = np.argsort(-pm)[:k]
            y[worst] = random_sphere(rng, k, d)
            # small shake rest
            y = _norm(y + 0.01 * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 1200, 0.025, 0.003, 0.3, 0.02, 0.06, 0.005
            lp_sec = 3.0
        elif mode == 4:
            # larger surgery + rattler hunt
            k = 18
            pm, m = per_point_max(y)
            worst = np.argsort(-pm)[:k]
            y[worst] = random_sphere(rng, k, d)
            y = _norm(y + 0.015 * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 1400, 0.03, 0.0025, 0.35, 0.02, 0.07, 0.004
            lp_sec = 3.0
        elif mode == 5:
            # big global shake
            sigma = 0.09
            y = _norm(y + sigma * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 1200, 0.03, 0.003, 0.35, 0.025, 0.07, 0.005
            lp_sec = 2.5
        else:
            # occasional fresh restart (every 6th)
            if trial % 12 == 0:
                y = random_sphere(rng, n, d)
            else:
                # contact-guided: shake high-contact points more
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                cnt = np.sum(G > m - 1e-9, axis=1)
                amp = 0.005 + 0.04 * (cnt / max(1, np.max(cnt)))
                y = _norm(y + amp[:, None] * rng.standard_normal(y.shape))
            steps, T0, T1, lr0, lr1, ms0, ms1 = 1500, 0.03, 0.0025, 0.35, 0.02, 0.07, 0.004
            lp_sec = 3.0
        # smooth phase with inner deadline
        inner_dead = min(HARD_END - 11, _time.time() + 6)
        y = smooth_descent(y, steps, T0, T1, lr0, lr1, ms0, ms1, beta=0.8, deadline=inner_dead)
        if _time.time() > HARD_END - 11:
            v_sm = max_cos(_norm(y))
            if v_sm < best_val:
                best, best_val = _norm(y), v_sm
            break
        # LP phase
        lp_dead = min(HARD_END - 10, _time.time() + lp_sec + 0.5)
        y2 = lp_polish(y, seconds=lp_sec, deadline=lp_dead)
        v = max_cos(y2)
        if v < best_val - 1e-12:
            best, best_val = y2, v
        # also consider pre-LP if LP failed to improve but smooth did
        else:
            v1 = max_cos(_norm(y))
            if v1 < best_val - 1e-12:
                # polish it a bit more next round; keep it
                best, best_val = _norm(y), v1
        if best_val < 0.895:
            break
    # final polish with remaining time
    rem = HARD_END - _time.time() - 1.0
    if rem > 2:
        best = lp_polish(best, seconds=min(rem, 12.0), deadline=HARD_END - 0.5)
    best = _norm(best)
    return best
