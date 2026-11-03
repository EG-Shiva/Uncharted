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

def _lp_top(x0, deadline, radius0=5e-5, topP=400):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = radius0
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]
    iuJ = iu0[1]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cfull = G[iuI, iuJ]
        # select topP
        if cfull.size > topP:
            thr = np.partition(cfull, -topP)[-topP]
            sel = cfull >= thr
            # trim to topP if ties exceed
            idx = np.where(sel)[0]
            if idx.size > topP:
                o = np.argsort(-cfull[idx])[:topP]
                idx = idx[o]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
        else:
            sel = np.ones_like(cfull, dtype=bool)
        I = iuI[sel]
        J = iuJ[sel]
        c = cfull[sel]
        p = len(I)
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = val
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 5e-4)
        else:
            radius *= 0.5
            if radius < 2e-7:
                # try smaller topP to isolate peak
                if topP > 60:
                    topP = max(60, topP // 2)
                    radius = radius0 * 0.2
                else:
                    break
    return best_x, best

def _topk_linesearch(x0, deadline, K=120, iters=200):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    iu = np.triu_indices(n, 1)
    # candidate step sizes (global scale)
    steps_grid = np.array([3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8])
    it = 0
    while it < iters and time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G[iu]
        if K < flat.size:
            thr = np.partition(flat, -K)[-K]
        else:
            thr = flat.min() - 1.0
        sel = flat >= thr
        I = iu[0][sel]
        J = iu[1][sel]
        if len(I) == 0:
            break
        grad = np.zeros_like(x)
        # uniform weight
        np.add.at(grad, I, x[J])
        np.add.at(grad, J, x[I])
        grad /= max(len(I), 1)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # normalize global scale to avoid blowup
        gn = float(np.sqrt((gt * gt).sum()))
        if gn < 1e-16:
            break
        gt = gt / gn * np.sqrt(n)
        improved = False
        # try steps
        for s in steps_grid:
            y = x - s * gt
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
                x = y
                improved = True
                break
        if not improved:
            # adapt K: if stuck, widen then narrow
            if K < 800:
                K = min(800, K * 2)
            else:
                # try per-point normalized direction
                gn2 = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
                gt2 = gt / np.maximum(gn2, 1.0)
                found = False
                for s in steps_grid * 0.5:
                    y = x - s * gt2
                    y /= np.linalg.norm(y, axis=1, keepdims=True)
                    val = _max_cos(y)
                    if val < best:
                        best = val
                        best_x = y.copy()
                        x = y
                        found = True
                        break
                if not found:
                    K = max(20, K // 2)
                    if K <= 25:
                        break
    return best_x, best

def _squeeze_single(x0, deadline):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G.argmax()
        ai, aj = divmod(int(flat), n)
        m = float(G[ai, aj])
        if m < best:
            best = m
            best_x = x.copy()
        di = -(x[aj] - m * x[ai])
        dj = -(x[ai] - m * x[aj])
        # project (already tangent)
        di = di - np.dot(di, x[ai]) * x[ai]
        dj = dj - np.dot(dj, x[aj]) * x[aj]
        ndi = float(np.linalg.norm(di))
        ndj = float(np.linalg.norm(dj))
        if ndi < 1e-16 and ndj < 1e-16:
            break
        if ndi > 1e-16:
            di /= ndi
        if ndj > 1e-16:
            dj /= ndj
        improved = False
        for s in (2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8):
            y = x.copy()
            y[ai] = x[ai] + s * di
            y[aj] = x[aj] + s * dj
            y[ai] /= np.linalg.norm(y[ai])
            y[aj] /= np.linalg.norm(y[aj])
            # quick global max
            Gy = y @ y.T
            np.fill_diagonal(Gy, -2.0)
            val = float(Gy.max())
            if val < best:
                best = val
                x = y
                best_x = y.copy()
                improved = True
                break
        if not improved:
            break
    return best_x, best

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 140.0
    rng = np.random.default_rng(0)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    n, d = 256, 6
    # immediate squeeze of isolated peak
    try:
        xb, vb = _squeeze_single(x_best, min(time.time() + 4.0, ANNEAL_END))
        if vb < best:
            x_best, best = xb, vb
    except Exception:
        pass
    try:
        xb, vb = _topk_linesearch(x_best, min(time.time() + 8.0, ANNEAL_END), K=80, iters=300)
        if vb < best:
            x_best, best = xb, vb
    except Exception:
        pass
    try:
        xb, vb = _lp_top(x_best, min(time.time() + 8.0, ANNEAL_END), radius0=3e-5, topP=300)
        if vb < best:
            x_best, best = xb, vb
    except Exception:
        pass
    cycle = 0
    while time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - time.time()
        if rem < 4:
            break
        # elitist tiny kick focused on current top cluster
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # top pairs
        iu = np.triu_indices(n, 1)
        flat = G[iu]
        Ktop = 120
        thr = np.partition(flat, -Ktop)[-Ktop]
        sel = flat >= thr
        I = iu[0][sel]; J = iu[1][sel]
        uniq = np.unique(np.concatenate([I, J]))
        x_try = x_best.copy()
        if cycle % 6 == 0:
            sigma = 3e-3
            k1 = min(len(uniq), 20)
            tgt = rng.choice(uniq, size=k1, replace=False) if len(uniq) > 0 else rng.choice(n, size=8, replace=False)
            x_try[tgt] += sigma * rng.normal(size=(len(tgt), d))
            rnd = rng.choice(n, size=4, replace=False)
            x_try[rnd] += (sigma * 0.5) * rng.normal(size=(4, d))
        elif cycle % 2 == 0:
            sigma = 6e-4
            k1 = min(len(uniq), 12)
            tgt = rng.choice(uniq, size=k1, replace=False) if len(uniq) > 0 else rng.choice(n, size=6, replace=False)
            x_try[tgt] += sigma * rng.normal(size=(len(tgt), d))
        else:
            sigma = 2.5e-4
            k1 = min(len(uniq), 8)
            tgt = rng.choice(uniq, size=k1, replace=False) if len(uniq) > 0 else rng.choice(n, size=4, replace=False)
            x_try[tgt] += sigma * rng.normal(size=(len(tgt), d))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # anneal with softmax adam (short)
        if cycle % 6 == 0:
            steps, a0, a1, lr = 500, 80.0, 1500.0, 0.008
        else:
            steps, a0, a1, lr = 350, 1500.0, 12000.0, 0.004
        if rem < 25:
            steps = min(steps, 250)
        try:
            xb, vb = _adam_sm(x_try, steps, a0, a1, lr)
        except Exception:
            continue
        # tight line search on result
        try:
            xb2, vb2 = _topk_linesearch(xb, time.time() + 1.5, K=100, iters=60)
            if vb2 < vb:
                xb, vb = xb2, vb2
        except Exception:
            pass
        # squeeze single peak
        try:
            xb3, vb3 = _squeeze_single(xb, time.time() + 0.5)
            if vb3 < vb:
                xb, vb = xb3, vb3
        except Exception:
            pass
        if vb < best - 1e-12:
            # LP tighten before accepting to squeeze further
            try:
                xb4, vb4 = _lp_top(xb, min(time.time() + 2.0, ANNEAL_END), radius0=2e-5, topP=250)
                if vb4 < vb:
                    xb, vb = xb4, vb4
            except Exception:
                pass
            x_best, best = xb, vb
    # final intensive polish
    try:
        if time.time() < T_END - 6:
            xb, vb = _adam_sm(x_best, 500, 800.0, 20000.0, 0.003)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if time.time() < T_END - 5:
            xb, vb = _topk_linesearch(x_best, T_END - 3.0, K=150, iters=400)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if time.time() < T_END - 2:
            xb, vb = _lp_top(x_best, T_END - 1.0, radius0=1.5e-5, topP=200)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if time.time() < T_END - 1:
            xb, vb = _squeeze_single(x_best, T_END - 0.5)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
