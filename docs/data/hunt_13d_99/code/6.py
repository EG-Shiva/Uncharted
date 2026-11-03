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


def _smooth_anneal(y, steps, T0, T1, lr0, mu=0.9):
    n, d = y.shape
    v = np.zeros_like(y)
    for k in range(steps):
        frac = k / steps
        T = T0 * ((T1 / T0) ** frac) if T1 > 0 else T0
        lr = lr0 * (1.0 - 0.7 * frac)
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S == 0 or not np.isfinite(S):
            break
        grad = (E @ y) / S
        dot = np.sum(grad * y, axis=1, keepdims=True)
        grad -= dot * y
        # clip extreme gradients for stability
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.2 / (gn + 1e-12))
        # only clip if huge; average gn ~0.02 so clip~1
        grad = grad * np.minimum(1.0, np.max(clip) * 0 + 1.0)
        v *= mu
        v -= lr * grad
        y += v
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
    return y


def solve():
    import time
    import numpy as np
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(0)

    # load starting point
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (99, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = rng.normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        best_val = _max_cos(xb)

    n, d = xb.shape

    def contacts_idx(x, thresh):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        cnt = np.sum(G > m - thresh, axis=1)
        return cnt, m

    trial = 0
    # reserve ~35s for final LP
    explore_end = t0 + TIME_LIMIT - 38.0

    sigmas = [0.03, 0.07, 0.12, 0.20, 0.32, 0.05, 0.15, 0.25]
    params = [
        (1500, 0.020, 0.0020, 0.08),
        (1500, 0.010, 0.0010, 0.05),
        (1800, 0.015, 0.0015, 0.06),
        (1200, 0.025, 0.0025, 0.10),
    ]

    while time.time() < explore_end:
        trial += 1
        steps, T0, T1, lr0 = params[trial % len(params)]
        # shrink steps if little time left
        if time.time() > explore_end - 8 and steps > 900:
            steps = 900
        mode = trial % 5
        y = xb.copy()
        if mode in (0, 1, 3):
            sigma = sigmas[trial % len(sigmas)]
            y = y + sigma * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # kill-and-reinsert 1-3 worst points
            cnt, m = contacts_idx(xb, 0.003)
            # overloaded first
            order = np.argsort(-cnt)
            nk = 1 + (trial % 3)  # 1..3
            # mix deterministic worst + random
            victims = list(order[:nk])
            if trial % 2 == 0:
                victims[-1] = int(rng.integers(0, n))
            for idx in victims:
                v = rng.normal(size=d)
                v /= np.linalg.norm(v)
                y[idx] = v
            # small shake of rest to open hole
            y = y + 0.02 * rng.normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        _smooth_anneal(y, steps, T0, T1, lr0, mu=0.9)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            xb = y.copy()
            # quick local squeeze if promising: short LP (2s) when we have time
            if time.time() < explore_end - 6:
                y2 = _lp_polish(xb, 2.0)
                v2 = _max_cos(y2)
                if v2 < best_val:
                    best_val = v2
                    xb = y2.copy()
        # side-step: if very close, occasionally diversify base without losing best
        # (no action needed, base stays at best)

        if time.time() >= explore_end:
            break

    # final polish with remaining time
    remain = TIME_LIMIT - (time.time() - t0) - 2.0
    if remain > 4:
        # first a final short anneal at low T to smooth, then LP
        try:
            yf = xb.copy()
            _smooth_anneal(yf, 600, 0.004, 0.0008, 0.02, mu=0.9)
            vf = _max_cos(yf)
            if vf < best_val:
                best_val = vf
                xb = yf.copy()
        except Exception:
            pass
        remain = TIME_LIMIT - (time.time() - t0) - 1.0
        if remain > 3:
            xb = _lp_polish(xb, min(remain, 32.0))

    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # sanitize
    xb = np.where(np.isfinite(xb), xb, 0.0)
    nr = np.linalg.norm(xb, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    xb = xb / nr
    return xb
