import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr):
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

def _lp_polish(x, deadline, r0=3e-3, cap=5000):
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
        margin = max(25.0 * radius, 3e-4)
        sel = c > (m - margin)
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
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
    ANNEAL_END = t0 + 138.0
    rng = _np.random.default_rng(1)
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
    # light warm LP to squeeze current basin
    try:
        if _time.time() < ANNEAL_END:
            x_best = _lp_polish(x_best, min(_time.time() + 6.0, ANNEAL_END), r0=2e-3)
            best = _max_cos(x_best)
    except Exception:
        pass
    cycle = 0
    stall = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 2.5:
            break
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        flat_arg = int(_np.argmax(G))
        ai = flat_arg // n
        aj = flat_arg % n
        prow = G.max(axis=1)
        cnt = _np.sum(G > gmax - 1e-3, axis=1)
        order_cnt = _np.argsort(-cnt)
        order_row = _np.argsort(-prow)
        mode = cycle % 5
        if stall > 12:
            # big diversification
            mode = 4
        if mode == 0:
            # targeted worst-pair split + focused noise
            alpha = 0.06 + 0.04 * rng.random()
            # move ai, aj apart along tangent
            gia = x[aj] - float(x[ai] @ x[aj]) * x[ai]
            gjb = x[ai] - float(x[ai] @ x[aj]) * x[aj]
            gia /= (_np.linalg.norm(gia) + 1e-12)
            gjb /= (_np.linalg.norm(gjb) + 1e-12)
            x[ai] = x[ai] - alpha * gia + 0.015 * rng.normal(size=(6,))
            x[aj] = x[aj] - alpha * gjb + 0.015 * rng.normal(size=(6,))
            hub = order_cnt[:8]
            x[hub] += 0.025 * rng.normal(size=(len(hub), 6))
            steps, K0, K1, lr = 500, 80.0, 2200.0, 0.008
        elif mode == 1:
            # coherent cap rotation
            u = rng.normal(size=(6,))
            u /= _np.linalg.norm(u)
            dots = x @ u
            kcap = int(rng.integers(10, 28))
            idx = _np.argsort(-dots)[:kcap]
            A = rng.normal(size=(6, 6))
            A = A - A.T
            theta = float(rng.uniform(0.08, 0.30)) * (1 if rng.random() < 0.5 else -1)
            rot = theta * (A @ x[idx].T).T
            x[idx] = x[idx] + rot + 0.008 * rng.normal(size=(len(idx), 6))
            # also jitter worst points lightly
            w = order_row[:6]
            x[w] += 0.015 * rng.normal(size=(6, 6))
            steps, K0, K1, lr = 500, 60.0, 1800.0, 0.009
        elif mode == 2:
            # void reinsertion for worst points
            nw = 3
            worst = order_row[:nw].copy()
            Z = rng.normal(size=(12000, 6))
            Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
            S = Z @ x.T
            for r in worst:
                S[:, r] = -2.0
            mm = S.max(axis=1)
            # replace each worst with distinct voids greedily
            used = set()
            for r in worst:
                # recompute excluding r already replaced? simple: pick best unused
                idx = int(_np.argmin(mm))
                # ensure improvement over current prow
                if mm[idx] < prow[r] - 5e-5:
                    x[r] = Z[idx]
                    # penalize reuse
                    mm[idx] = 2.0
            # global light noise
            x += 0.008 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 550, 100.0, 2500.0, 0.007
        elif mode == 3:
            # hub shake: medium noise on overloaded + random subset
            over = order_cnt[:12]
            x[over] += 0.04 * rng.normal(size=(len(over), 6))
            rnd = rng.choice(n, size=8, replace=False)
            x[rnd] += 0.02 * rng.normal(size=(8, 6))
            steps, K0, K1, lr = 450, 150.0, 3000.0, 0.006
        else:
            # large kick diversification
            pick = rng.choice(n, size=28, replace=False)
            x[pick] += 0.35 * rng.normal(size=(28, 6))
            x += 0.01 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 600, 40.0, 1200.0, 0.011
        if rem < 25:
            steps = min(steps, 350)
        x /= _np.linalg.norm(x, axis=1, keepdims=True)
        xb, vb = _adam_sm(x, steps, K0, K1, lr)
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
            stall = 0
            # squeeze immediately with short LP
            try:
                if _time.time() < ANNEAL_END - 4:
                    xs = _lp_polish(x_best, min(_time.time() + 2.5, ANNEAL_END), r0=1.5e-3)
                    vs = _max_cos(xs)
                    if vs < best:
                        x_best, best = xs, vs
            except Exception:
                pass
        else:
            stall += 1
    # final intensive polish
    try:
        if _time.time() < T_END - 4:
            xb, vb = _adam_sm(x_best, 700, 400.0, 3500.0, 0.004)
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
