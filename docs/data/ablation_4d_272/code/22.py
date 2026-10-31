import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


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
        if np.count_nonzero(sel) > 4000:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], 4000 - 1)[:4000]
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


def lbfgs_quench(x_init, alphas=(50.0, 140.0, 300.0), maxiters=(70, 70, 60), pen_w=5.0):
    n, d = x_init.shape
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]
    J0 = iu0[1]
    y = x_init.reshape(-1).copy()
    # ensure unit rows
    Yn = y.reshape(n, d)
    Yn /= np.linalg.norm(Yn, axis=1, keepdims=True)
    y = Yn.reshape(-1).copy()
    for alpha, mi in zip(alphas, maxiters):
        def fun(yf, alpha=alpha):
            Y = yf.reshape(n, d)
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r = np.maximum(r, 1e-12)
            Xn = Y / r
            G = Xn @ Xn.T
            c = G[I0, J0]
            m = c.max()
            # shift for stability
            e = np.exp(alpha * (c - m))
            Z = e.sum()
            L = m + np.log(Z) / alpha
            w = e / Z
            # dense weight matrix
            Wm = np.zeros((n, n), dtype=np.float64)
            Wm[I0, J0] = w
            Wm[J0, I0] = w
            GX = Wm @ Xn
            row_wc = np.sum(Wm * G, axis=1, keepdims=True)
            grad_Xn = GX - row_wc * Xn
            rs = r.squeeze()
            grad_Y = grad_Xn / rs[:, None]
            diff = (rs - 1.0)
            pen = pen_w * np.sum(diff * diff)
            grad_Y = grad_Y + (2.0 * pen_w * diff[:, None] * Xn)
            return (L + pen), grad_Y.ravel()
        try:
            res = minimize(fun, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
            y = res.x
        except Exception:
            break
    X = y.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def refine_hole(Y, c0, steps=120, s0=0.05):
    c = c0 / np.linalg.norm(c0)
    for it in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].mean(axis=0)
        gt = g - np.dot(g, c) * c
        nrm = np.linalg.norm(gt)
        if nrm < 1e-14:
            break
        gt /= nrm
        s = s0 * (1.0 - it / steps) + 0.002 * (it / steps)
        c = c - s * gt
        c /= np.linalg.norm(c)
    return c


def hole_replace(x, rng, n_replace=1, n_samples=16000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    order = np.argsort(-rowmax)
    y = x.copy()
    for k in range(n_replace):
        # worst among remaining (recompute occasionally)
        if k == 1:
            G2 = y @ y.T
            np.fill_diagonal(G2, -2.0)
            rowmax2 = G2.max(axis=1)
            move = int(np.argmax(rowmax2))
        else:
            move = int(order[k % len(order)])
            if k >= 1 and rng.random() < 0.5:
                # randomize choice a bit
                move = int(order[rng.integers(0, min(n, 30))])
        mask = np.ones(n, dtype=bool)
        mask[move] = False
        Y = y[mask]
        C = rng.standard_normal((n_samples, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        best_c = None
        best_s = 2.0
        bs = 4000
        for a in range(0, n_samples, bs):
            Cb = C[a:a + bs]
            S = Cb @ Y.T
            sm = S.max(axis=1)
            kk = int(np.argmin(sm))
            if sm[kk] < best_s:
                best_s = float(sm[kk])
                best_c = Cb[kk].copy()
        best_c = refine_hole(Y, best_c)
        y[move] = best_c / np.linalg.norm(best_c)
    return y


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    n, d = 272, 4
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    cands0 = []
    if bx is not None:
        bx = np.asarray(bx, dtype=np.float64)
        if bx.shape == (n, d):
            cands0.append(bx / np.linalg.norm(bx, axis=1, keepdims=True))
    if rx is not None:
        rx = np.asarray(rx, dtype=np.float64)
        if rx.shape == (n, d):
            cands0.append(rx / np.linalg.norm(rx, axis=1, keepdims=True))
    if len(cands0) == 0:
        tmp = rng.standard_normal((n, d))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands0.append(tmp)
    best_x = None
    best_v = 2.0
    for c in cands0:
        v = max_cos(c)
        if v < best_v:
            best_v = v
            best_x = c.copy()
    it = 0
    # staged alpha jitter choices
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        remain = BUDGET - 13.0 - (time.time() - t0)
        if remain < 3:
            break
        u = rng.random()
        x0 = None
        if u < 0.30:
            sigma = 0.006 + 0.014 * rng.random()
            noise = rng.standard_normal(best_x.shape) * sigma
            # boost rattlers
            try:
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                mcur = G.max()
                ratt = rm < (mcur - 5e-4)
                if np.any(ratt):
                    noise[ratt] *= 3.0
            except Exception:
                pass
            x0 = best_x + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            al = (40.0 + 15.0 * rng.random(), 120.0 + 50.0 * rng.random(), 260.0 + 120.0 * rng.random())
            mi = (50, 60, 50)
            nsamp = None
        elif u < 0.50:
            sigma = 0.03 + 0.05 * rng.random()
            noise = rng.standard_normal(best_x.shape) * sigma
            x0 = best_x + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            al = (35.0 + 10.0 * rng.random(), 100.0 + 40.0 * rng.random(), 220.0 + 100.0 * rng.random())
            mi = (60, 60, 50)
            nsamp = None
        elif u < 0.70:
            nr = 1 if rng.random() < 0.6 else 2
            if rng.random() < 0.3:
                nr = 3
            ns = 12000 if remain < 40 else 16000
            x0 = hole_replace(best_x, rng, n_replace=nr, n_samples=ns)
            # small jitter after surgery
            x0 = x0 + rng.standard_normal(x0.shape) * 0.002
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            al = (45.0, 130.0, 280.0)
            mi = (60, 60, 60)
            nsamp = None
        else:
            # subset scramble of worst points
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            k = int(12 + rng.integers(0, 25))
            idx = order[:k]
            x0 = best_x.copy()
            # randomize those points substantially
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            blend = 0.5 + 0.5 * rng.random()
            for ii, jj in enumerate(idx):
                x0[jj] = (1 - blend) * x0[jj] + blend * R[ii]
            # add small noise to rest
            x0 += rng.standard_normal(x0.shape) * 0.004
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            al = (30.0, 100.0, 240.0)
            mi = (70, 70, 60)
            nsamp = None
        # quench
        try:
            xq = lbfgs_quench(x0, alphas=al, maxiters=mi)
        except Exception:
            continue
        vq = max_cos(xq)
        # decide LP polish
        improved = vq < best_v - 1e-12
        near = vq < best_v + 0.003
        do_lp = False
        lp_t = 2.5
        if improved:
            do_lp = True
            lp_t = min(5.0, max(2.5, remain / 5.0))
        elif near and rng.random() < (0.35 if vq < best_v + 0.001 else 0.12):
            # only if we have time
            if remain > 20:
                do_lp = True
                lp_t = 2.5
        if do_lp:
            remain2 = BUDGET - 13.0 - (time.time() - t0)
            if remain2 < 2:
                # no time for LP, just compare
                if vq < best_v:
                    best_x, best_v = xq, vq
                continue
            yp = lp_polish(xq, min(lp_t, remain2))
            vp = max_cos(yp)
            if vp < best_v - 1e-12:
                best_x, best_v = yp, vp
            elif vq < best_v - 5e-5:
                # keep quenched as base for next? no, keep best but allow basin shift if much better shape?
                # if LP failed to improve but quench alone is close, keep quench if better than best
                if vq < best_v:
                    best_x, best_v = xq, vq
        else:
            if vq < best_v - 1e-12:
                best_x, best_v = xq, vq
        # opportunistic: if quench beat best by tiny margin without LP, polish best a bit later in final
    remain = BUDGET - (time.time() - t0)
    if remain > 3.5:
        try:
            best_x = lp_polish(best_x, min(remain - 1.0, 11.0))
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
