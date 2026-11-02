import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best


def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju, movable=None):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    grad = np.empty_like(x)
    use_mask = movable is not None
    if use_mask:
        mov = np.asarray(movable, dtype=bool)
    else:
        mov = None
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        if Z == 0 or not np.isfinite(Z):
            continue
        w = e / Z
        grad.fill(0.0)
        Wx = w[:, None] * x[Ju]
        np.add.at(grad, Iu, Wx)
        Wx2 = w[:, None] * x[Iu]
        np.add.at(grad, Ju, Wx2)
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        if use_mask:
            grad[~mov] = 0.0
        m1 = b1 * m1 + (1.0 - b1) * grad
        v1 = b2 * v1 + (1.0 - b2) * (grad * grad)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        if use_mask:
            x[mov] = x[mov] - lr * step[mov]
            nr = np.linalg.norm(x[mov], axis=1, keepdims=True)
            np.maximum(nr, 1e-12, out=nr)
            x[mov] /= nr
        else:
            x = x - lr * step
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            np.maximum(nr, 1e-12, out=nr)
            x /= nr
    return x


def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        xb = xb.reshape(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    archive = [x_best.copy()]
    arch_vals = [best_val]
    Iu, Ju = np.triu_indices(n, 1)

    it = 0
    while True:
        now = time.time()
        if now - t0 > 135.0:
            break
        if HARD - (now - t0) < 34.0:
            break
        it += 1
        mode = it % 8

        y0 = None
        subset_idx = None
        # build starting point
        if mode == 0:  # small global jitter
            sigma = 0.02
            y0 = x_best + rng.normal(0, sigma, size=(n, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            s1_steps, b0, b1, l0, l1 = 450, 30.0, 180.0, 0.008, 0.0006
            tol1, tol2 = 0.003, 0.0015
            use_subset_first = False
        elif mode == 1:  # 2-point replacement
            y0 = x_best.copy()
            rep = rng.choice(n, size=2, replace=False)
            y0[rep] = rng.normal(size=(2, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            subset_idx = rep
            s1_steps, b0, b1, l0, l1 = 450, 30.0, 180.0, 0.008, 0.0006
            tol1, tol2 = 0.006, 0.003
            use_subset_first = True
        elif mode == 2:  # cap reshuffle 7-12 pts
            k = int(rng.integers(7, 13))
            pole = rng.normal(size=4)
            pole /= np.linalg.norm(pole)
            dots = x_best @ pole
            idx = np.argsort(-dots)[:k]
            y0 = x_best.copy()
            y0[idx] = rng.normal(size=(k, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            subset_idx = idx
            s1_steps, b0, b1, l0, l1 = 450, 25.0, 160.0, 0.008, 0.0006
            tol1, tol2 = 0.009, 0.004
            use_subset_first = True
        elif mode == 3:  # from scratch random
            y0 = rng.normal(size=(n, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            s1_steps, b0, b1, l0, l1 = 600, 15.0, 130.0, 0.010, 0.0008
            tol1, tol2 = 0.010, 0.004
            use_subset_first = False
        elif mode == 4:  # crossover
            if len(archive) >= 2:
                a = int(rng.integers(len(archive)))
                b = int(rng.integers(len(archive)))
                if b == a:
                    b = (b + 1) % len(archive)
                X = archive[a]
                Y = archive[b]
                M = Y.T @ X
                U, _, Vt = np.linalg.svd(M)
                Q = U @ Vt
                Ya = Y @ Q
                mask = rng.random(n) < 0.5
                y0 = np.where(mask[:, None], X, Ya)
                y0 += rng.normal(0, 0.015, size=(n, d))
                y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            else:
                y0 = x_best + rng.normal(0, 0.08, size=(n, d))
                y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            s1_steps, b0, b1, l0, l1 = 500, 25.0, 170.0, 0.008, 0.0006
            tol1, tol2 = 0.008, 0.004
            use_subset_first = False
        elif mode == 5:  # rattler targeted
            G = x_best @ x_best.T
            gall = G[Iu, Ju]
            mx = gall.max()
            tight = gall > mx - 1e-7
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, Iu[tight], 1)
            np.add.at(cnt, Ju[tight], 1)
            ratt = np.where(cnt <= 2)[0]
            hubs = np.where(cnt >= 7)[0]
            y0 = x_best + rng.normal(0, 0.025, size=(n, d))
            if len(ratt) > 0:
                for rr in ratt:
                    if rng.random() < 0.7:
                        y0[rr] = rng.normal(size=4)
                    else:
                        y0[rr] += rng.normal(0, 0.35, size=4)
            if len(hubs) > 0:
                y0[hubs] += rng.normal(0, 0.02, size=(len(hubs), d))
            # random extra replacement 1-2
            nr = int(rng.integers(1, 3))
            rep = rng.choice(n, size=nr, replace=False)
            y0[rep] = rng.normal(size=(nr, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            s1_steps, b0, b1, l0, l1 = 450, 35.0, 200.0, 0.007, 0.0005
            tol1, tol2 = 0.005, 0.0025
            use_subset_first = False
        elif mode == 6:  # large global shake
            y0 = x_best + rng.normal(0, 0.12, size=(n, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            s1_steps, b0, b1, l0, l1 = 550, 20.0, 150.0, 0.009, 0.0006
            tol1, tol2 = 0.009, 0.004
            use_subset_first = False
        else:  # aggressive 4-6 replacement
            k = int(rng.integers(4, 7))
            y0 = x_best.copy()
            rep = rng.choice(n, size=k, replace=False)
            y0[rep] = rng.normal(size=(k, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            subset_idx = rep
            s1_steps, b0, b1, l0, l1 = 450, 25.0, 170.0, 0.008, 0.0006
            tol1, tol2 = 0.009, 0.004
            use_subset_first = True

        # stage 1
        if use_subset_first and subset_idx is not None:
            mov = np.zeros(n, dtype=bool)
            mov[np.asarray(subset_idx).ravel()] = True
            y1 = _smooth_opt(y0, 350, b0, b1 * 0.6, l0, l1, Iu, Ju, movable=mov)
            # follow with full short relax as stage1
            y1 = _smooth_opt(y1, s1_steps, b0, b1, l0, l1, Iu, Ju, movable=None)
        else:
            y1 = _smooth_opt(y0, s1_steps, b0, b1, l0, l1, Iu, Ju, movable=None)
        v1 = _max_cos(y1)
        if v1 > best_val + tol1:
            continue
        # stage 2 full high-beta
        y2 = _smooth_opt(y1, 700, 80.0, 350.0, 0.004, 0.0003, Iu, Ju, movable=None)
        v2 = _max_cos(y2)
        if v2 > best_val + tol2:
            # keep archive diversity even without LP
            if v2 < best_val + 0.002 and len(archive) < 5:
                # avoid duplicates
                dup = False
                for av in arch_vals:
                    if abs(av - v2) < 1e-9:
                        dup = True
                        break
                if not dup:
                    archive.append(y2.copy())
                    arch_vals.append(v2)
            continue
        remain = HARD - (time.time() - t0) - 24.0
        if remain < 1.2:
            break
        b = min(2.6, max(1.2, remain / 7.0))
        yp, vp = _lp_polish(y2, b)
        if vp < best_val - 1e-12:
            x_best, best_val = yp, vp
            archive[0] = x_best.copy()
            arch_vals[0] = best_val
            # add to archive, keep size small
            if len(archive) < 6:
                archive.append(x_best.copy())
                arch_vals.append(best_val)
        elif vp < best_val + 0.002 and len(archive) < 6:
            dup = False
            for av in arch_vals:
                if abs(av - vp) < 1e-9:
                    dup = True
                    break
            if not dup:
                # keep best archive sorted-ish: replace worst if full
                if len(archive) >= 6:
                    wm = int(np.argmax(arch_vals))
                    archive[wm] = yp.copy()
                    arch_vals[wm] = vp
                else:
                    archive.append(yp.copy())
                    arch_vals.append(vp)
        elif vp < best_val + 1e-12 and rng.random() < 0.02:
            x_best = yp

    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 26.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
