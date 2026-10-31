import time
import numpy as np


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    return rm, float(np.max(G))


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _hard_refine(x, steps=150, lr=0.012, window=0.006):
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = (G > m - window).astype(np.float64)
        # symmetrize not needed, G symmetric
        deg = np.sum(W, axis=1, keepdims=True)
        # gradient of sum of active dots
        Gact = W * G
        s = np.sum(Gact, axis=1, keepdims=True)
        grad = (W @ x) - s * x
        # average by degree
        grad = grad / np.maximum(deg, 1.0)
        # clip
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 1.0
        if np.any(big):
            grad = grad * np.where(big, 1.0 / np.maximum(gn, 1e-12), 1.0)
        x = x - lr * grad
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _find_hole(x, rng, nsample=6000):
    d = x.shape[1]
    c = rng.standard_normal(size=(nsample, d))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    # max inner with set
    s = c @ x.T
    m = np.max(s, axis=1)
    idx = int(np.argmin(m))
    return c[idx]


def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    EXPLORE_END = t0 + 125.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    # candidate pool for final polish
    pool_x = [best_x.copy()]
    pool_v = [best_v]

    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        progress = (time.time() - t0) / max(1e-9, (EXPLORE_END - t0))
        Temp = 0.003 * (1.0 - progress) + 0.0002
        mode = it % 10
        # choose base
        if it % 7 == 0:
            base = best_x
        else:
            base = cur_x
        # reset cur if too bad
        if cur_v > best_v + 0.02:
            base = best_x
            cur_x = best_x.copy()
            cur_v = best_v
        y = base.copy()

        if mode == 0:
            sigma = 0.012
            nz = rng.standard_normal(size=(n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 250, 0.010, 0.0015, 0.05, 0.005)
            y = _hard_refine(y, 80, 0.010, 0.005)
        elif mode == 1:
            sigma = 0.05
            nz = rng.standard_normal(size=(n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 400, 0.020, 0.0010, 0.08, 0.005)
            y = _hard_refine(y, 120, 0.012, 0.006)
        elif mode == 2:
            sigma = 0.20
            nz = rng.standard_normal(size=(n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 600, 0.030, 0.0008, 0.10, 0.004)
            y = _hard_refine(y, 150, 0.012, 0.006)
        elif mode == 3:
            k = 8 + (it % 9)
            idx = rng.choice(n, size=k, replace=False)
            nz = rng.standard_normal(size=(k, d)) * 0.15
            # tangent
            dot = np.sum(nz * y[idx], axis=1, keepdims=True)
            nz = nz - dot * y[idx]
            y[idx] = y[idx] + nz
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _smooth_opt(y, 400, 0.018, 0.0010, 0.07, 0.005)
            y = _hard_refine(y, 100, 0.010, 0.005)
        elif mode == 4:
            k = 5 + (it % 6)
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = _random_sphere(k, d, rng)
            y = _smooth_opt(y, 500, 0.025, 0.0009, 0.09, 0.004)
            y = _hard_refine(y, 150, 0.012, 0.006)
        elif mode == 5:
            # hole insertion: replace worst with hole
            rm, gv = _row_max(y)
            worst = int(np.argmax(rm))
            # second worst sometimes
            if it % 2 == 0:
                # pick among top-3 worst
                order = np.argpartition(rm, -3)[-3:]
                worst = int(order[rng.integers(0, len(order))])
            hole = _find_hole(y, rng, nsample=6000)
            y[worst] = hole
            # jitter neighbours slightly
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 500, 0.020, 0.0008, 0.08, 0.004)
            y = _hard_refine(y, 150, 0.012, 0.006)
        elif mode == 6:
            rm, gv = _row_max(y)
            ratt = np.where(rm < gv - 0.003)[0]
            if ratt.size == 0:
                ratt = np.where(rm < gv - 0.001)[0]
            if ratt.size > 0:
                # shake rattlers hard
                k = min(ratt.size, 8 + (it % 8))
                sel = rng.choice(ratt, size=k, replace=False)
                nz = rng.standard_normal(size=(k, d)) * 0.5
                dot = np.sum(nz * y[sel], axis=1, keepdims=True)
                nz = nz - dot * y[sel]
                y[sel] = y[sel] + nz
                y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
            else:
                nz = rng.standard_normal(size=(n, d)) * 0.08
                nz -= np.sum(nz * y, axis=1, keepdims=True) * y
                y = y + nz
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 400, 0.015, 0.0010, 0.07, 0.005)
            y = _hard_refine(y, 100, 0.010, 0.005)
        elif mode == 7:
            sigma = 0.38
            nz = rng.standard_normal(size=(n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 700, 0.035, 0.0009, 0.10, 0.003)
            y = _hard_refine(y, 150, 0.012, 0.007)
        elif mode == 8:
            k = 18 + (it % 12)
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = _random_sphere(k, d, rng)
            y = _smooth_opt(y, 700, 0.030, 0.0008, 0.10, 0.003)
            y = _hard_refine(y, 150, 0.012, 0.006)
        else:
            # full random restart, short but aggressive
            y = _random_sphere(n, d, rng)
            y = _smooth_opt(y, 700, 0.040, 0.0010, 0.10, 0.004)
            y = _hard_refine(y, 150, 0.012, 0.007)

        v = _max_cos(y)
        if v < best_v:
            best_v = v
            best_x = y.copy()
            pool_x.append(y.copy())
            pool_v.append(v)
            # keep pool small
            if len(pool_x) > 5:
                # drop worst
                wi = int(np.argmax(pool_v))
                pool_x.pop(wi)
                pool_v.pop(wi)
            cur_x = y.copy()
            cur_v = v
        else:
            # SA acceptance for cur
            if v < cur_v or rng.random() < np.exp((cur_v - v) / max(Temp, 1e-9)):
                cur_x = y.copy()
                cur_v = v
        if time.time() > EXPLORE_END - 1.0:
            break

    # final LP polish: polish best and up to 2 runners
    # order pool by value
    order = np.argsort(np.array(pool_v))
    time_left = HARD_END - time.time() - 1.0
    if time_left < 5.0:
        time_left = max(1.0, HARD_END - time.time() - 0.5)
    # allocate most time to best
    best polished
    if len(order) > 0:
        # polish best first with most budget
        per = time_left / min(len(order), 3)
        polished_best_x = best_x
        polished_best_v = best_v
        for k, oi in enumerate(order[:3]):
            if time.time() > HARD_END - 2.0:
                break
            cand = pool_x[oi]
            # last candidate (best) gets remaining time
            if k == 0:
                dl = time.time() + per * 1.6
            else:
                dl = time.time() + per * 0.7
            dl = min(dl, HARD_END - 1.0)
            px, pv = _lp_polish(cand, dl)
            if pv < polished_best_v:
                polished_best_v = pv
                polished_best_x = px.copy()
        best_x = polished_best_x
        best_v = polished_best_v
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
