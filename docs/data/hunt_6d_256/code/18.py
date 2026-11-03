import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(y0, steps, T0, T1, lr0, lr1):
    y = y0.copy()
    n, d = y.shape
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best = _max_cos(y)
    best_x = y.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = y @ y.T
        np.fill_diagonal(G, -np.inf)
        mval = float(G.max())
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        W = E / S
        Grad = W @ y
        dot = np.sum(Grad * y, axis=1, keepdims=True)
        Gt = Grad - dot * y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (s + 1))
        vhat = v / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * y, axis=1, keepdims=True)
        step = step - sdot * y
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if (s + 1) % 150 == 0 or s == steps - 1:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    return best_x, best

def _topk_descent(y0, steps=250, K=200, lr0=0.006, lr1=0.0003):
    y = y0 / np.linalg.norm(y0, axis=1, keepdims=True)
    n, d = y.shape
    best = _max_cos(y)
    best_x = y.copy()
    iu0 = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        lr = lr0 * (lr1 / lr0) ** f
        G = y @ y.T
        flat = G[iu0]
        if K < flat.size:
            thr = np.partition(flat, -K)[-K]
        else:
            thr = flat.min() - 1
        sel = flat >= thr
        I = iu0[0][sel]; J = iu0[1][sel]
        if len(I) == 0:
            break
        w = flat[sel] - thr + 1e-9
        w = np.sqrt(w)
        w = w / w.sum() * len(I)
        grad = np.zeros_like(y)
        np.add.at(grad, I, w[:, None] * y[J])
        np.add.at(grad, J, w[:, None] * y[I])
        grad /= max(len(I), 1)
        dot = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - dot * y
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(gn, 0.5)
        y = y - lr * gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if (s + 1) % 50 == 0:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    val = _max_cos(y)
    if val < best:
        best, best_x = val, y.copy()
    return best_x, best

def _lp_polish(x, deadline, r0=5e-4, max_pairs=6000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    iuI, iuJ = iu[0], iu[1]
    while time.time() < deadline and radius > 1e-15:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        win = max(10.0 * radius, 2e-6)
        # shrink floor when very close to jammed
        if radius < 2e-7:
            win = max(10.0 * radius, 2e-7)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[::-1]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order[:max_pairs]]] = True
            sel = keep
        I, J = iuI[sel], iuJ[sel]
        csel = c[sel]
        p = len(I)
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
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.5, 5e-3)
        else:
            # accept tiny improvement too
            if vv < best:
                x, best = y, vv
            radius *= 0.5
        # early exit if many constraints (jammed) and radius tiny
        if cnt >= max_pairs and radius < 1e-7:
            break
    return x

def _void_point(x, exclude, rng, ntry=12000):
    # find point on S5 minimizing max inner prod to x (excluding index)
    Z = rng.normal(size=(ntry, 6))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ x.T
    if exclude is not None:
        S[:, exclude] = -2.0
    mm = S.max(axis=1)
    return Z[int(np.argmin(mm))], float(mm.min())

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 174.0
    ANNEAL_END = t0 + 132.0
    rng = _np.random.default_rng(20261103)
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
    # immediate pair-breaking: tight top-K descent + tight LP
    try:
        xb, vb = _topk_descent(x_best, steps=250, K=120, lr0=0.004, lr1=0.0002)
        if vb < best:
            x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < ANNEAL_END:
            x_best = _lp_polish(x_best, min(_time.time() + 10.0, ANNEAL_END), r0=5e-4)
            best = _max_cos(x_best)
    except Exception:
        pass
    try:
        xb, vb = _adam_lse(x_best, 400, 0.004, 0.0004, 0.008, 0.002)
        if vb < best:
            x_best, best = xb, vb
    except Exception:
        pass

    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 4:
            break
        G = x_best @ x_best.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        cnt = _np.sum(G > gmax - 3e-6, axis=1)
        # choose kick mode
        mode = cycle % 5
        x = x_best.copy()
        if mode == 0:
            # worst-pair break: move one endpoint to void
            flat_idx = int(_np.argmax(G))
            ii, jj = flat_idx // n, flat_idx % n
            victim = ii if prow[ii] >= prow[jj] else jj
            zv, _ = _void_point(x, victim, rng, ntry=12000)
            # partial move toward void to keep some memory
            alpha = 0.9
            x[victim] = alpha * zv + (1 - alpha) * x[victim]
            x[victim] /= _np.linalg.norm(x[victim])
            # shake its old neighbours a bit
            nbrs = _np.where(G[victim] > gmax - 2e-4)[0]
            for nb in nbrs[:6]:
                x[nb] += 0.02 * rng.normal(size=(6,))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 450, 0.005, 0.0003, 0.010, 0.002
        elif mode == 1:
            # hub shake: most crowded points
            order = _np.argsort(-cnt)
            hubs = order[:8]
            x[hubs] += 0.06 * rng.normal(size=(8, 6))
            # rattler to void
            loose = _np.argsort(prow)[:4]
            for r in loose[:2]:
                zv, mv = _void_point(x, int(r), rng, ntry=8000)
                if mv < prow[r] - 5e-5:
                    x[int(r)] = zv
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 500, 0.006, 0.0004, 0.009, 0.002
        elif mode == 2:
            # global small kick + loose reinsert
            x += 0.012 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            loose = _np.argsort(prow)[:6]
            for r in loose[:3]:
                zv, mv = _void_point(x, int(r), rng, ntry=6000)
                if mv < prow[r]:
                    mix = 0.7
                    x[int(r)] = mix * zv + (1 - mix) * x[int(r)]
                    x[int(r)] /= _np.linalg.norm(x[int(r)])
            steps, T0, T1, l0, l1 = 450, 0.004, 0.0002, 0.008, 0.002
        elif mode == 3:
            # medium subset reinit around worst region
            order = _np.argsort(-prow)
            worst = order[:12]
            x[worst] += 0.12 * rng.normal(size=(12, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 550, 0.008, 0.0005, 0.011, 0.002
        else:
            # fine shake
            x += 0.006 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 400, 0.003, 0.0002, 0.007, 0.0015
        if rem < 25:
            steps = min(steps, 320)
        try:
            xb, vb = _adam_lse(x, steps, T0, T1, l0, l1)
        except Exception:
            continue
        # tight top-K squeeze after Adam
        try:
            xb2, vb2 = _topk_descent(xb, steps=120, K=150, lr0=0.003, lr1=0.0002)
            if vb2 < vb:
                xb, vb = xb2, vb2
        except Exception:
            pass
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb
            best = vb
            # opportunistic LP squeeze on improvement
            try:
                if _time.time() < ANNEAL_END - 6:
                    x_best = _lp_polish(x_best, min(_time.time() + 4.0, ANNEAL_END), r0=3e-4)
                    best = _max_cos(x_best)
            except Exception:
                pass
    # final intensive polish
    try:
        if _time.time() < T_END - 8:
            xb, vb = _adam_lse(x_best, 500, 0.002, 0.00015, 0.005, 0.001)
            if vb < best:
                x_best, best = xb, vb
            xb2, vb2 = _topk_descent(x_best, steps=300, K=300, lr0=0.002, lr1=0.0001)
            if vb2 < best:
                x_best, best = xb2, vb2
    except Exception:
        pass
    try:
        if _time.time() < T_END - 2:
            x_best = _lp_polish(x_best, T_END - 1.0, r0=2e-4)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
