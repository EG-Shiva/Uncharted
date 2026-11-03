import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_softmax(x0, steps, K0, K1, lr, seed_offset=0):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # project step to tangent
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(15.0 * radius, 8e-5)
        sel = c > (m - margin)
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 4000:
            part = np.argpartition(-csel, 4000)[:4000]
            I = I[part]; J = J[part]; csel = csel[part]
            p = 4000
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
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
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 128.0
    rng = _np.random.default_rng(0)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    n, d = 256, 6
    # quick LP warm start to exploit non-jammed plateau
    try:
        if _time.time() < ANNEAL_END:
            x_best = _lp_polish(x_best, min(_time.time() + 8.0, ANNEAL_END))
            best = _max_cos(x_best)
    except Exception:
        pass
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 3:
            break
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # per-point worst cosine to find loose vs jammed points
        prow = G.max(axis=1)
        # loose points (small prow) and overloaded (large prow / many near contacts)
        cnt = _np.sum(G > gmax - 2e-4, axis=1)
        order = _np.argsort(-cnt)
        loose = _np.argsort(prow)[:8]
        # schedule perturbation size
        if cycle % 7 == 0:
            sigma, steps, a0, a1, lr = 0.22, 700, 40.0, 900.0, 0.012
        elif cycle % 3 == 0:
            sigma, steps, a0, a1, lr = 0.05, 600, 60.0, 1500.0, 0.009
        elif cycle % 2 == 0:
            sigma, steps, a0, a1, lr = 0.02, 500, 100.0, 2000.0, 0.007
        else:
            sigma, steps, a0, a1, lr = 0.008, 450, 150.0, 2500.0, 0.006
        if rem < 25:
            steps = min(steps, 350)
        # re-insert loosest points into largest voids by sampling
        if cycle % 2 == 0:
            for r in loose[:4]:
                Z = rng.normal(size=(8000, 6))
                Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ x.T
                S[:, r] = -2.0
                mm = S.max(axis=1)
                idx = int(_np.argmin(mm))
                # only replace if void is better than current
                if mm[idx] < prow[r] - 1e-4:
                    x[r] = Z[idx]
        over = order[:10]
        x[over] += sigma * rng.normal(size=(10, 6))
        rnd = rng.choice(n, size=6, replace=False)
        x[rnd] += (sigma * 0.6) * rng.normal(size=(6, 6))
        x /= _np.linalg.norm(x, axis=1, keepdims=True)
        xb, vb = _adam_softmax(x, steps, a0, a1, lr)
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
        # occasional big kick diversification
        if cycle % 12 == 0 and _time.time() < ANNEAL_END - 6:
            x2 = x_best.copy()
            pick = rng.choice(n, size=24, replace=False)
            x2[pick] += 0.4 * rng.normal(size=(24, 6))
            x2 /= _np.linalg.norm(x2, axis=1, keepdims=True)
            xb2, vb2 = _adam_softmax(x2, 400, 50.0, 1200.0, 0.012)
            if vb2 < best - 1e-12:
                x_best = xb2
                best = vb2
    # final intensive polish: alternate Adam fine tune + LP
    try:
        if _time.time() < T_END - 3:
            # fine Adam with large K to equalize plateau
            xb, vb = _adam_softmax(x_best, 600, 400.0, 3000.0, 0.004)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 2:
            x_best = _lp_polish(x_best, T_END - 1.0, r0=1e-3)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
