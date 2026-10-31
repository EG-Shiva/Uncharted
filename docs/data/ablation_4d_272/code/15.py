import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 5000 - 1)[:5000]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def lbfgs_smooth(x_init, alpha, maxiter=80):
    n, d = x_init.shape
    y0 = (x_init / np.linalg.norm(x_init, axis=1, keepdims=True)).ravel().copy()
    triu = np.triu_indices(n, 1)

    def fun(yflat):
        Y = yflat.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m0 = float(G.max())
        E = np.exp(alpha * (G - m0))
        np.fill_diagonal(E, 0.0)
        S = float(E[triu].sum()) + 1e-300
        L = m0 + np.log(S) / alpha
        Gx = (E / S) @ X
        dot = np.sum(Gx * X, axis=1, keepdims=True)
        Gt = (Gx - dot * X) / norms
        return float(L), Gt.ravel().astype(np.float64)

    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12, "maxfun": maxiter * 5})
        Yopt = res.x.reshape(n, d)
    except Exception:
        return x_init.copy()
    Yn = np.linalg.norm(Yopt, axis=1, keepdims=True)
    Yn = np.maximum(Yn, 1e-12)
    return Yopt / Yn


def optimize_single(Y, c0, steps=200, s0=0.03, mu=0.5):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    s = s0
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].sum(axis=0)
        gt = g - np.dot(g, c) * c
        nrm = np.linalg.norm(gt)
        if nrm < 1e-14:
            break
        gt = gt / nrm
        vel = mu * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            nc = np.linalg.norm(c)
            if nc < 1e-14:
                break
            c /= nc
            s *= 0.997
    return c


def hole_reinsert(x, rng, n_samples=15000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    i, j = divmod(k, n)
    Gi = G[i].copy(); Gi[j] = -2
    Gj = G[j].copy(); Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.2:
        move = i if rng.random() < 0.5 else j
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None
    best_s = 2.0
    bs = 3000
    for a in range(0, n_samples, bs):
        Cb = C[a:a + bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        k2 = int(np.argmin(sm))
        if sm[k2] < best_s:
            best_s = float(sm[k2])
            best_c = Cb[k2].copy()
    best_c = optimize_single(Y, best_c, steps=250, s0=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(1)
    n, d = 272, 4

    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None

    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    best_x = None
    best_v = 2.0
    if bx is not None and np.asarray(bx).shape == (272, 4):
        bx = normed(bx)
        best_x, best_v = bx.copy(), max_cos(bx)
    if rx is not None and np.asarray(rx).shape == (272, 4):
        rx = normed(rx)
        rv = max_cos(rx)
        if rv < best_v:
            best_x, best_v = rx.copy(), rv
    if best_x is None:
        best_x = rng.standard_normal((n, d))
        best_x = normed(best_x)
        best_v = max_cos(best_x)

    sigmas = [0.004, 0.010, 0.022, 0.050, 0.100, 0.015, 0.035]
    schedules = [
        [70.0, 250.0],
        [150.0, 500.0],
        [40.0, 120.0, 400.0],
        [100.0, 350.0],
        [250.0, 800.0],
    ]
    it = 0
    # main basin loop, reserve ~13s for final polish
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        sched = schedules[(it - 1) % len(schedules)]
        mode = (it - 1) % 5

        x = best_x.copy()
        try:
            rm, mcur = row_max(x)
        except Exception:
            rm = np.full(n, mcur if 'mcur' in locals() else best_v)
            mcur = best_v

        if mode == 0:
            # global small kick
            s = sigma * (0.7 + 0.6 * rng.random())
            noise = rng.standard_normal(x.shape) * s
            x = x + noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            # rattler-targeted kick
            is_rat = rm < (mcur - 5e-4)
            if np.count_nonzero(is_rat) == 0:
                is_rat = rm < (mcur - 2e-4)
            noise = rng.standard_normal(x.shape) * sigma * 0.8
            if np.count_nonzero(is_rat) > 0:
                noise[is_rat] *= 4.0
                # also kick neighbours of worst pair a bit more
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G))
                i, j = divmod(k, n)
                noise[i] *= 2.0; noise[j] *= 2.0
            x = x + noise
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            # hole reinsertion (+ tiny noise elsewhere)
            try:
                x = hole_reinsert(x, rng, n_samples=15000)
                x = x + rng.standard_normal(x.shape) * (sigma * 0.3)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            except Exception:
                x = best_x + rng.standard_normal(x.shape) * sigma
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 3:
            # worst-pair antipodal push + noise
            try:
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                # top few worst pairs
                flat = G[np.triu_indices(n, 1)]
                K = min(8, flat.size)
                thr = np.partition(flat, -K)[-K]
                iu = np.triu_indices(n, 1)
                sel = flat >= thr
                Ii = iu[0][sel][:K]; Ji = iu[1][sel][:K]
                push = np.zeros_like(x)
                for a, b in zip(Ii, Ji):
                    diff = x[a] - x[b]
                    diff /= (np.linalg.norm(diff) + 1e-12)
                    push[a] += diff; push[b] -= diff
                x = x + push * (sigma * 1.5) + rng.standard_normal(x.shape) * (sigma * 0.5)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            except Exception:
                x = x + rng.standard_normal(x.shape) * sigma
                x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # double hole move for big topology change
            try:
                x = hole_reinsert(x, rng, n_samples=12000)
                x = hole_reinsert(x, rng, n_samples=12000)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            except Exception:
                x = best_x + rng.standard_normal(x.shape) * sigma
                x /= np.linalg.norm(x, axis=1, keepdims=True)

        # annealed L-BFGS smoothing
        for ai, alpha in enumerate(sched):
            if time.time() - t0 > BUDGET - 13.0:
                break
            mi = 70 if ai < len(sched) - 1 else 110
            # smaller budget early iterations
            x = lbfgs_smooth(x, alpha, maxiter=mi)
            # early exit if terrible
            vq = max_cos(x)
            if vq > best_v + 0.02:
                break

        if time.time() - t0 > BUDGET - 13.0:
            # keep if better without LP
            vq = max_cos(x)
            if vq < best_v:
                best_x, best_v = x, vq
            break

        # short LP lock
        try:
            remain = BUDGET - 13.0 - (time.time() - t0)
            lp_t = min(2.0, max(0.6, remain / 3.0))
            x = lp_polish(x, lp_t)
        except Exception:
            pass
        v = max_cos(x)
        if v < best_v:
            best_x, best_v = x, v

    # final long polish of champion
    try:
        remain = BUDGET - (time.time() - t0)
        ft = max(1.0, min(11.0, remain - 1.0))
        if ft > 0.5:
            best_x = lp_polish(best_x, ft)
            best_v = max_cos(best_x)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
