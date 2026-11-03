import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _smooth_run(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        G -= M
        G *= beta
        np.clip(G, -20.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-300:
            continue
        H = (G @ X) * (2.0 / S)
        dot = np.sum(H * X, axis=1, keepdims=True)
        H -= dot * X
        V *= mu
        V += (1.0 - mu) * H
        vn = np.sqrt(np.sum(V * V, axis=1))
        mv = float(np.max(vn))
        if mv < 1e-15:
            continue
        X -= V * (lr / mv)
        nn = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nn[nn == 0] = 1.0
        X /= nn
    return X


def _adam_softmax(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    t = 0
    b1p = 1.0
    b2p = 1.0
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        t += 1
        b1p *= b1
        b2p *= b2
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        Mx = float(np.max(G))
        G -= Mx
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (G @ X) / S
        dd = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dd * X
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        dn = np.sum(upd * X, axis=1, keepdims=True)
        upd -= dn * X
        # normalize update scale to keep trust region
        un = np.sqrt(np.sum(upd * upd, axis=1, keepdims=True))
        # per-point clip
        # global scale: max norm -> lr
        mu_ = float(np.max(un)) + 1e-30
        X = X - upd * (lr / mu_ * 2.0)
        X = _norm(X)
    return X


def _greedy(X, iters, step0=4e-4, deadline=None):
    X = _norm(X.copy())
    best = _max_cos(X)
    step = step0
    n = X.shape[0]
    for it in range(iters):
        if deadline is not None and (it & 255) == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        # tangent separation directions
        vi = dot * X[i] - X[j]
        vj = dot * X[j] - X[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-13:
                step = step0 * 0.3
            continue
        vi /= ni
        vj /= nj
        y = X.copy()
        y[i] = X[i] + step * vi
        y[j] = X[j] + step * vj
        y[i] /= np.linalg.norm(y[i])
        y[j] /= np.linalg.norm(y[j])
        # fast incremental check: only need max involving i,j + old max?
        # full eval for safety (cheap)
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        vv = float(np.max(Gy))
        if vv < best:
            X = y
            best = vv
            step = min(step * 1.2, 4e-3)
        else:
            # try single sided on i
            y2i = X[i] + step * vi
            y2i /= np.linalg.norm(y2i)
            # quick bound: dots of y2i to others
            d2 = y @ y2i if False else None
            # full eval single-sided (only every other fail to save time)
            if (it & 1) == 0:
                Y2 = X.copy()
                Y2[i] = y2i
                G2 = Y2 @ Y2.T
                np.fill_diagonal(G2, -2.0)
                v2 = float(np.max(G2))
                Y3 = X.copy()
                y3j = X[j] + step * vj
                y3j /= np.linalg.norm(y3j)
                Y3[j] = y3j
                G3 = Y3 @ Y3.T
                np.fill_diagonal(G3, -2.0)
                v3 = float(np.max(G3))
                if v2 < best and v2 <= v3:
                    X, best = Y2, v2
                    step = min(step * 1.1, 4e-3)
                elif v3 < best:
                    X, best = Y3, v3
                    step = min(step * 1.1, 4e-3)
                else:
                    step *= 0.55
                    if step < 5e-14:
                        step = step0 * 0.3
            else:
                step *= 0.6
                if step < 5e-14:
                    step = step0 * 0.3
    return X, best


def _contact_jam(X, best, deadline, tol0=2e-3):
    X = _norm(X.copy())
    lr = 2e-3
    tol = tol0
    it = 0
    while time.time() < deadline:
        it += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        A = (G > m - tol).astype(np.float64)
        deg = A.sum(axis=1)
        # avoid empty
        AG = A * G
        F = A @ X - AG.sum(axis=1)[:, None] * X
        sc = np.maximum(deg, 1.0)[:, None]
        F /= sc
        y = _norm(X - lr * F)
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        vv = float(np.max(Gy))
        if vv < best:
            X, best = y, vv
            lr = min(lr * 1.07, 4e-3)
            tol = max(tol * 0.999, 2e-7)
        else:
            lr *= 0.6
            if lr < 1e-7:
                tol = max(tol * 0.5, 1e-9)
                lr = 5e-4
        if it % 400 == 0:
            tol = max(tol * 0.8, 5e-10)
    return X, best


def _lp_polish(X, best, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    radius = 1.2e-3
    while time.time() < deadline and radius > 1e-14:
        Gf = X @ X.T
        call = Gf[iu]
        m = float(np.max(call))
        thr = m - max(15.0 * radius, 3e-7)
        sel = call > thr
        p = int(np.sum(sel))
        if p < 10:
            kk = min(800, call.size)
            idx = np.argpartition(-call, kk - 1)[:kk]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = kk
        if p > 2800:
            c_sel = call[sel]
            order = np.argpartition(-c_sel, 2800 - 1)[:2800]
            full = np.where(sel)[0][order]
            ns = np.zeros_like(sel, dtype=bool)
            ns[full] = True
            sel = ns
            p = 2800
        if p == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        c = call[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    hard = t0 + 168.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((182, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (182, 9):
        tmp = np.zeros((182, 9))
        m0 = min(xb.shape[0], 182)
        d0 = min(xb.shape[1] if xb.ndim > 1 else 1, 9)
        tmp[:m0, :d0] = xb[:m0, :d0] if xb.ndim > 1 else xb[:m0, None]
        if m0 < 182:
            rr = rng.standard_normal((182 - m0, 9))
            tmp[m0:] = rr
        xb = tmp
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (182, 9):
            xr = _norm(xr)
            vr = _max_cos(xr)
            xb = _norm(xb)
            vb = _max_cos(xb)
            if vr < vb:
                xb = xr.copy()
                vb = vr
    except Exception:
        pass
    xb = _norm(xb)
    best_val = _max_cos(xb)
    best_X = xb.copy()

    n, d = 182, 9
    explore_dead = t0 + 112.0
    sigmas = [0.005, 0.012, 0.025, 0.045, 0.07, 0.008]
    cycle = 0
    while time.time() < explore_dead and time.time() < hard - 45:
        cycle += 1
        sigma = sigmas[cycle % len(sigmas)]
        mode = cycle % 5
        X = best_X.copy()
        if mode == 4:
            k = int(rng.integers(2, 5))
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            idx = np.argsort(-rowmax)[:k]
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            cur = X[mask]
            for ii in idx:
                cand = rng.standard_normal((4000, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ cur.T
                scores = np.max(dots, axis=1)
                bc = cand[int(np.argmin(scores))]
                X[ii] = bc
                cur = np.vstack([cur, bc[None, :]])
            om = np.where(mask)[0]
            X[om] += 0.004 * rng.standard_normal((len(om), d))
        elif mode == 3:
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            idx = np.argsort(-rowmax)[:10]
            X += sigma * 0.4 * rng.standard_normal((n, d))
            X[idx] += sigma * 1.6 * rng.standard_normal((len(idx), d))
        elif mode == 2:
            # block rotation of a random cluster
            X += sigma * rng.standard_normal((n, d))
            # random orthogonal jiggle in a random 2D subspace for subset
            sub = rng.choice(n, size=30, replace=False)
            a = rng.standard_normal(d)
            a /= np.linalg.norm(a)
            b = rng.standard_normal(d)
            b = b - (b @ a) * a
            b /= (np.linalg.norm(b) + 1e-30)
            ang = sigma * 2.0 * rng.standard_normal(len(sub))
            for tii, ii in enumerate(sub):
                x0 = X[ii]
                ca, sa = np.cos(ang[tii]), np.sin(ang[tii])
                # rotate components along a,b slightly
                pa, pb = float(x0 @ a), float(x0 @ b)
                X[ii] = x0 + sa * (pa * b - pb * a)
        else:
            X += sigma * rng.standard_normal((n, d))
        X = _norm(X)
        if sigma > 0.04:
            X = _smooth_run(X, steps=1600, beta0=30.0, beta1=180.0, lr0=0.025, lr1=0.002, mu=0.82)
        else:
            X = _smooth_run(X, steps=1600, beta0=45.0, beta1=350.0, lr0=0.018, lr1=0.0012, mu=0.86)
        if time.time() > hard - 45:
            v = _max_cos(X)
            if v < best_val:
                best_val, best_X = v, X.copy()
            break
        # short adam sharpen
        X = _adam_softmax(X, steps=300, beta0=350.0, beta1=700.0, lr0=0.004, lr1=0.0008)
        v = _max_cos(X)
        if v < best_val:
            best_val = v
            best_X = X.copy()
            # extra greedy squeeze on improvement
            if time.time() < hard - 45:
                Xg, vg = _greedy(X.copy(), iters=1500, step0=3e-4, deadline=hard - 40)
                if vg < best_val:
                    best_val, best_X = vg, Xg.copy()
        else:
            if cycle % 4 == 0 and time.time() < hard - 50:
                Xg, vg = _greedy(X.copy(), iters=800, step0=2e-4, deadline=hard - 45)
                if vg < best_val:
                    best_val, best_X = vg, Xg.copy()

    # final intensive squeeze
    try:
        if time.time() < hard - 30:
            best_X = _smooth_run(best_X.copy(), steps=900, beta0=400.0, beta1=800.0,
                                 lr0=0.004, lr1=0.0006, mu=0.9)
            best_val = _max_cos(best_X)
    except Exception:
        pass
    try:
        if time.time() < hard - 25:
            best_X = _adam_softmax(best_X.copy(), steps=700, beta0=500.0, beta1=1200.0,
                                   lr0=0.003, lr1=0.0005)
            best_val = _max_cos(best_X)
    except Exception:
        pass
    try:
        if time.time() < hard - 20:
            best_X, best_val = _greedy(best_X.copy(), iters=12000, step0=3e-4, deadline=hard - 14)
    except Exception:
        pass
    try:
        if time.time() < hard - 12:
            best_X, best_val = _contact_jam(best_X.copy(), best_val, deadline=hard - 8, tol0=1e-3)
    except Exception:
        pass
    try:
        if time.time() < hard - 6:
            best_X, best_val = _lp_polish(best_X.copy(), best_val, deadline=hard - 1.5)
    except Exception:
        pass

    best_X = _norm(best_X)
    best_X = np.nan_to_num(best_X, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(best_X, axis=1, keepdims=True)
    zr = (nn[:, 0] < 1e-12)
    if np.any(zr):
        rr = rng.standard_normal((int(np.sum(zr)), 9))
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best_X[zr] = rr
        best_X = _norm(best_X)
    return best_X
