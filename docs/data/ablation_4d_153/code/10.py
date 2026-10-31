import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _row_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)


def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _power_opt(x, steps, p0, p1, lr0, lr1, mom=0.86):
    n, d = x.shape
    vel = np.zeros_like(x)
    lp0, lp1 = np.log(p0), np.log(p1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        P = float(np.exp(lp0 + (lp1 - lp0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Q = (1.0 + G) * 0.5
        np.clip(Q, 0.0, 1.0, out=Q)
        # power weights Q^(P-1)
        # avoid 0^0, P>=1 so fine
        with np.errstate(under='ignore', over='ignore'):
            W = np.power(Q, P - 1.0)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-18)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        # clip at 2
        bad = gn > 2.0
        grad[bad[:, 0]] *= (2.0 / gn[bad]).reshape(-1, 1)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / max(bc, 1e-12)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _best_hole(x, idx, S, rng):
    n, d = x.shape
    others = np.delete(x, idx, axis=0)
    cand = rng.standard_normal(size=(S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    vals = np.max(dots, axis=1)
    b = int(np.argmin(vals))
    return cand[b], float(vals[b])


def _hole_sweep(x, rng, S=5000, topk=10):
    # monotone coordinate descent on minimax: never increases global max
    x = x.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    cur_m = float(np.max(g))
    row = np.max(g, axis=1)
    order = np.argsort(-row)[:topk]
    for idx in order:
        idx = int(idx)
        # current row max (recompute cheap for this row)
        cur_row = float(np.max(x[idx] @ x.T[np.arange(x.shape[0]) != idx].T if False else 0)) if False else 0
        # fast current row max:
        cur_row = float(row[idx])
        # if point already well below max, still try? skip if far below to save time
        # we still try worst ones only; order already worst-first
        h, hv = _best_hole(x, idx, S, rng)
        if hv < cur_row - 1e-12:
            # check new global max will not increase
            # rest max <= cur_m, hv <= cur_row <= cur_m so safe
            x[idx] = h
            # update row/g lazily for remaining: recompute row for all (cheap n^2)
            g = x @ x.T
            np.fill_diagonal(g, -2.0)
            cur_m = float(np.max(g))
            row = np.max(g, axis=1)
            # re-sort remaining? keep simple
    return x, cur_m


def _rattler_shuffle(x, rng, margin=8e-4, S=6000):
    x = x.copy()
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    row = np.max(g, axis=1)
    ratt = np.where(row < m - margin)[0]
    if len(ratt) == 0:
        # try smaller margin to find free points
        ratt = np.where(row < m - 2e-4)[0]
        if len(ratt) == 0:
            return x, m
    for idx in ratt:
        idx = int(idx)
        others = np.delete(x, idx, axis=0)
        cand = rng.standard_normal(size=(S, x.shape[1]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        vals = np.max(cand @ others.T, axis=1)
        feas = np.where(vals < m - 1e-4)[0]
        if len(feas) == 0:
            continue
        # pick random feasible to diversify (not necessarily best)
        pick = int(rng.choice(feas))
        # occasionally pick best for quality
        if rng.random() < 0.3:
            pick = int(np.argmin(vals))
        x[idx] = cand[pick]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, _max_cos(x)


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
        sel = c > thresh
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
        if p0 > 2800:
            order = np.argpartition(c[idx], -2800)[-2800:]
            idx = idx[order]
            p0 = 2800
        I = iu[0][idx]
        J = iu[1][idx]
        csel = c[idx]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
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
        if time.time() >= deadline:
            break
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
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

    sigmas = [0.004, 0.010, 0.025, 0.06, 0.12, 0.25, 0.5]

    NITER = 80
    for it in range(NITER):
        now = time.time()
        if now > HARD_END - 58.0:
            break
        mode = it % 7
        sigma = sigmas[it % len(sigmas)]
        # choose base
        base = best_x if (it % 3 != 0) else cur_x
        if mode == 0:
            # rattler shuffle preserving score + tiny noise
            tmp, _ = _rattler_shuffle(base, rng, margin=8e-4, S=5000)
            noise = rng.standard_normal(size=(n, d))
            x0 = tmp + 0.006 * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, p0, p1, lr0, lr1 = 1100, 20.0, 220.0, 0.07, 0.003
        elif mode == 1:
            # worst-point hole jump (monotone) then perturb slightly
            tmp, _ = _hole_sweep(base, rng, S=6000, topk=3)
            noise = rng.standard_normal(size=(n, d))
            x0 = tmp + sigma * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # random respawn of 1-2 points occasionally
            if it % 2 == 0:
                k = int(rng.integers(1, 3))
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.standard_normal(size=(k, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x0[idx] = rp
            steps, p0, p1, lr0, lr1 = 1300, 25.0, 280.0, 0.08, 0.003
        elif mode == 2:
            noise = rng.standard_normal(size=(n, d))
            x0 = base + sigma * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, p0, p1, lr0, lr1 = 1200, 18.0, 200.0, 0.07, 0.003
        elif mode == 3:
            noise = rng.standard_normal(size=(n, d))
            x0 = base + sigma * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            k = int(rng.integers(2, 6))
            idx = rng.choice(n, size=k, replace=False)
            rp = rng.standard_normal(size=(k, d))
            rp /= np.linalg.norm(rp, axis=1, keepdims=True)
            x0[idx] = rp
            steps, p0, p1, lr0, lr1 = 1500, 22.0, 260.0, 0.09, 0.003
        elif mode == 4:
            # rattler shuffle of best + medium power opt
            tmp, _ = _rattler_shuffle(best_x, rng, margin=5e-4, S=7000)
            noise = rng.standard_normal(size=(n, d))
            x0 = tmp + 0.012 * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, p0, p1, lr0, lr1 = 1400, 30.0, 320.0, 0.08, 0.0025
        elif mode == 5:
            if it % 14 == 12:
                x0 = _random_sphere(n, d, rng)
                steps, p0, p1, lr0, lr1 = 2000, 15.0, 250.0, 0.10, 0.003
            else:
                noise = rng.standard_normal(size=(n, d))
                x0 = cur_x + 0.08 * noise
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                steps, p0, p1, lr0, lr1 = 1300, 25.0, 300.0, 0.08, 0.003
        else:
            noise = rng.standard_normal(size=(n, d))
            x0 = base + 0.02 * noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps, p0, p1, lr0, lr1 = 1100, 20.0, 240.0, 0.06, 0.0025

        x1 = _power_opt(x0, steps, p0, p1, lr0, lr1, mom=0.86)
        v1 = _max_cos(x1)
        # monotone hole refinement (cheap, never hurts)
        try:
            x1h, v1h = _hole_sweep(x1, rng, S=4000, topk=8)
            if v1h <= v1 + 1e-15:
                x1, v1 = x1h, v1h
        except Exception:
            pass
        # short LP if promising
        if v1 < best_v + 0.003 and time.time() < HARD_END - 58.0:
            dl = min(time.time() + 0.9, HARD_END - 56.0)
            if dl > time.time() + 0.15:
                try:
                    x1p, v1p = _lp_polish(x1, dl)
                    if v1p < v1:
                        x1, v1 = x1p, v1p
                except Exception:
                    pass
        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        tol = 0.003 * (1.0 - it / NITER) + 0.0004
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            if cur_v > best_v + 0.008:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            if rng.random() < 0.12:
                cur_x = best_x.copy()
                cur_v = best_v

    # final thorough polish: alternate LP + hole sweep + tiny power tune
    try:
        # first ensure best is LP-polished
        best_x, best_v = _lp_polish(best_x, HARD_END - 12.0)
    except Exception:
        pass
    # try a few hole + LP rounds to squeeze topology
    for _ in range(4):
        if time.time() > HARD_END - 10.0:
            break
        try:
            xh, vh = _hole_sweep(best_x, rng, S=12000, topk=20)
        except Exception:
            break
        if vh < best_v - 1e-13:
            best_x, best_v = xh, vh
            try:
                best_x, best_v = _lp_polish(best_x, min(time.time() + 4.0, HARD_END - 2.0))
            except Exception:
                pass
        else:
            break
    # tiny jitter + high-exponent tune attempts if time remains
    for _ in range(3):
        if time.time() > HARD_END - 6.0:
            break
        try:
            x0 = best_x + 1e-4 * rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            x1 = _power_opt(x0, 600, 80.0, 400.0, 0.02, 0.001, mom=0.85)
            v1 = _max_cos(x1)
            if v1 < best_v:
                best_x, best_v = x1, v1
                best_x, best_v = _lp_polish(best_x, min(time.time() + 3.0, HARD_END - 1.5))
        except Exception:
            pass
    try:
        if time.time() < HARD_END - 1.0:
            best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
