import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _adam_full(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        # cap per-row step
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step = step * sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X


def _refine_one(x_init, Others, steps, beta, lr):
    x = x_init.copy() / (np.linalg.norm(x_init) + 1e-16)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    for _ in range(steps):
        d = Others @ x
        Mx = float(np.max(d))
        E = np.exp(np.clip((d - Mx) * beta, -30.0, 0.0))
        S = float(np.sum(E)) + 1e-300
        W = E / S
        g = Others.T @ W
        g = g - float(np.dot(g, x)) * x
        t += 1
        m = b1m * m + (1.0 - b1m) * g
        v = b2m * v + (1.0 - b2m) * (g * g)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = float(np.linalg.norm(step)) + 1e-16
        cap = 5.0 * lr
        if sn > cap:
            step = step * (cap / sn)
        x = x - step
        n = float(np.linalg.norm(x))
        if n == 0 or not np.isfinite(n):
            x = x_init / (np.linalg.norm(x_init) + 1e-16)
            continue
        x /= n
        m = m - float(np.dot(m, x)) * x
    return x


def _void_reinsert(cur, worst, rng):
    n, d = cur.shape
    mask = np.ones(n, dtype=bool)
    mask[worst] = False
    Others = cur[mask]
    # sample voids
    S = 6000
    R = rng.standard_normal((S, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
    D = R @ Others.T
    md = D.max(axis=1)
    k = 4
    idx = np.argpartition(md, k)[:k]
    best_x = cur[worst].copy()
    # current value
    best_v = float(np.max(Others @ best_x))
    for ii in idx:
        x0 = R[ii]
        for beta, lr, st in [(2000.0, 0.01, 120), (10000.0, 0.003, 120)]:
            try:
                x0 = _refine_one(x0, Others, st, beta, lr)
            except Exception:
                pass
        vv = float(np.max(Others @ x0))
        if vv < best_v:
            best_v = vv
            best_x = x0.copy()
    Y = cur.copy()
    Y[worst] = best_x / (np.linalg.norm(best_x) + 1e-16)
    return Y


def _lp_polish(x, deadline, k_top=4000, r0=5e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-13:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        K = k_top if k_top < c_all.size else c_all.size
        if K < c_all.size:
            top = np.argpartition(-c_all, K)[:K]
            I = iu0[top]; J = iu1[top]; c_sel = c_all[top]
        else:
            I = iu0; J = iu1; c_sel = c_all
        p = I.shape[0]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cands.append(X)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)

    def time_left():
        return HARD - time.time()

    # Stage 0: annealed smoothing from best
    try:
        if time.time() < t0 + 35:
            Y = _adam_full(best, 700, 300.0, 2500.0, 0.004, 0.002)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    try:
        if time.time() < t0 + 45:
            Y = _adam_full(best, 700, 2500.0, 12000.0, 0.002, 0.0008)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass

    cur = best.copy()
    cur_val = best_val
    sigma = 0.012
    fail = 0

    while time.time() < HOP_END and time_left() > 30:
        r = float(rng.random())
        try:
            if r < 0.62:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                order = np.argsort(-rm)
                # pick 1-3 among top 12
                nt = int(rng.integers(1, 4))
                pool = order[:12]
                # always include worst
                sel = [int(order[0])]
                if nt > 1:
                    extra = rng.choice(pool[1:], size=min(nt - 1, 11), replace=False)
                    for e in extra:
                        sel.append(int(e))
                Y = cur.copy()
                nz = rng.standard_normal((len(sel), d))
                # tangentialize
                for k2, idx in enumerate(sel):
                    v0 = nz[k2]
                    v0 = v0 - float(np.dot(v0, Y[idx])) * Y[idx]
                    n0 = float(np.linalg.norm(v0))
                    if n0 < 1e-12:
                        v0 = rng.standard_normal(d)
                        v0 = v0 - float(np.dot(v0, Y[idx])) * Y[idx]
                        n0 = float(np.linalg.norm(v0)) + 1e-16
                    v0 = v0 / n0 * sigma * (0.5 + float(rng.random()))
                    Y[idx] = Y[idx] + v0
                Y[sel] /= np.linalg.norm(Y[sel], axis=1, keepdims=True) + 1e-16
                # occasional tiny global shake
                if float(rng.random()) < 0.3:
                    Y = Y + rng.standard_normal((n, d)) * (sigma * 0.15)
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
                Y2 = _adam_full(Y, 380, 1200.0, 9000.0, 0.0025, 0.001)
            elif r < 0.85:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                worst = int(np.argmax(rm))
                # sometimes second worst
                if float(rng.random()) < 0.3:
                    order = np.argsort(-rm)
                    worst = int(order[int(rng.integers(1, min(6, n)))])
                Y = _void_reinsert(cur, worst, rng)
                Y2 = _adam_full(Y, 380, 1200.0, 9000.0, 0.0025, 0.001)
            else:
                Y = cur + rng.standard_normal((n, d)) * sigma * 0.5
                Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
                Y2 = _adam_full(Y, 450, 800.0, 10000.0, 0.003, 0.001)
            v = _max_cos(Y2)
        except Exception:
            fail += 1
            continue
        if v < best_val - 1e-12:
            best, best_val = Y2, v
            cur, cur_val = Y2, v
            fail = 0
            sigma = max(0.004, sigma * 0.92)
        elif v < cur_val - 1e-12:
            cur, cur_val = Y2, v
            fail = 0
        else:
            fail += 1
            if fail >= 8:
                cur = best.copy()
                cur_val = best_val
                sigma = min(0.035, sigma * 1.35)
                fail = 0

    # Final squeeze: alternate Adam + LP
    try:
        if time_left() > 12:
            Y = _adam_full(best, 600, 3000.0, 18000.0, 0.0012, 0.0005)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    try:
        if time_left() > 8:
            Y = _lp_polish(best, HARD - 4.0, k_top=4000, r0=5e-4)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    try:
        if time_left() > 4:
            Y = _adam_full(best, 250, 8000.0, 25000.0, 0.0006, 0.0002)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
