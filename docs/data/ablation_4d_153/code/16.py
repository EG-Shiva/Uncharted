import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _rand_sphere(n, d, rng):
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
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad * np.where(gn > 2.0, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _hard_descent(x, steps, lr=0.01, topk=800):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # threshold for topk
        flat = G[iu0]
        if topk < flat.size:
            thr = float(np.partition(flat, -topk)[-topk])
        else:
            thr = float(flat.min())
        # gradient from pairs above thr
        mask = G > thr
        # symmetric weight 1 for selected
        grad = (mask.astype(float) @ x)
        # tangent
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # normalize per-point to avoid blowup
        grad = grad / np.maximum(gn, 1.0)
        x = x - lr * grad
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


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
        if p0 > 3000:
            order = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[order]
            p0 = 3000
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
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    EXPL_END = t0 + 128.0
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
        best_x = _rand_sphere(n, d, rng)
        best_v = _max_cos(best_x)

    cur_x = best_x.copy()
    cur_v = best_v

    iu = np.triu_indices(n, 1)
    sigmas = [0.004, 0.01, 0.025, 0.06, 0.15, 0.35]
    it = 0
    no_improve = 0
    while _time.time() < EXPL_END and it < 600:
        sigma = sigmas[it % len(sigmas)]
        if it % 37 == 36:
            sigma = 0.6
        mode = it % 6
        y = cur_x.copy()
        if mode == 0:
            y = cur_x + sigma * rng.standard_normal(size=(n, d))
        elif mode == 1:
            k = int(rng.integers(2, 9))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += sigma * 3.0 * rng.standard_normal(size=(k, d))
            y += (sigma * 0.15) * rng.standard_normal(size=(n, d))
        elif mode == 2:
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            # worst pairs
            flat = G[iu]
            order = np.argpartition(flat, -30)[-30:]
            wi = set()
            for q in order:
                wi.add(int(iu[0][q])); wi.add(int(iu[1][q]))
            wi = list(wi)
            # push apart along tangent
            for i_ in wi:
                # sum of neighbours
                nbr = np.where(G[i_] > m - 0.005)[0]
                push = np.zeros(d)
                for j_ in nbr:
                    if j_ == i_:
                        continue
                    c_ = float(cur_x[i_] @ cur_x[j_])
                    push += (cur_x[j_] - c_ * cur_x[i_])
                y[i_] -= sigma * 2.0 * push + sigma * rng.standard_normal(size=(d,))
            y += (sigma * 0.1) * rng.standard_normal(size=(n, d))
        elif mode == 3:
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            invol = np.zeros(n, dtype=bool)
            # pairs within 0.008 are contacts
            sel = G > (m - 0.008)
            invol = np.any(sel, axis=1)
            ratt = np.where(~invol)[0]
            if len(ratt) > 0:
                # fully randomize rattlers
                y[ratt] = rng.standard_normal(size=(len(ratt), d))
                y += (sigma * 0.2) * rng.standard_normal(size=(n, d))
            else:
                # no rattlers: randomize points in most contacts + noise
                cnt = np.sum(sel, axis=1)
                k = int(min(n, 4))
                idx = np.argpartition(cnt, -k)[-k:]
                y[idx] = rng.standard_normal(size=(k, d))
                y += (sigma * 0.2) * rng.standard_normal(size=(n, d))
        elif mode == 4:
            y = cur_x + sigma * rng.standard_normal(size=(n, d))
            # plus one fully random point
            k = int(rng.integers(0, n))
            y[k] = rng.standard_normal(size=(d,))
        else:
            # occasional fresh start blended with best
            if it % 24 == 23:
                y = _rand_sphere(n, d, rng)
                # inject a few good points from best to keep some structure
                k = 30
                idx = rng.choice(n, size=k, replace=False)
                y[idx] = best_x[idx]
                y += 0.05 * rng.standard_normal(size=(n, d))
            else:
                y = cur_x + sigma * rng.standard_normal(size=(n, d))

        y /= np.linalg.norm(y, axis=1, keepdims=True)

        tau1 = 0.0012 + float(rng.random()) * 0.0025
        steps = 900 + int(rng.integers(0, 700))
        lr0 = 0.025 + float(rng.random()) * 0.015
        lr1 = 0.003 + float(rng.random()) * 0.003
        y = _smooth_opt(y, steps, 0.04, tau1, lr0, lr1, mom=0.9)
        # short hard-max refinement
        y = _hard_descent(y, 80, lr=0.005, topk=600)
        v = _max_cos(y)

        if v < best_v - 1e-12:
            # squeeze new record with short LP
            tl = min(2.5, max(0.5, EXPL_END - _time.time() - 1.0))
            if tl > 0.6 and _time.time() < EXPL_END - 1.0:
                try:
                    yp, vp = _lp_polish(y, _time.time() + tl)
                    if vp < v:
                        y, v = yp, vp
                except Exception:
                    pass
            if v < best_v:
                best_x, best_v = y.copy(), float(v)
                cur_x, cur_v = y.copy(), float(v)
                no_improve = 0
            else:
                # LP didn't help beyond best, still update cur
                cur_x, cur_v = y.copy(), float(v)
        else:
            if v < cur_v:
                cur_x, cur_v = y.copy(), float(v)
                no_improve = 0
                # if very close to best, check true polished value
                if v < best_v + 0.0004 and (it % 8 == 0) and _time.time() < EXPL_END - 3.0:
                    try:
                        yp, vp = _lp_polish(y, _time.time() + 1.0)
                        if vp < best_v - 1e-12:
                            best_x, best_v = yp.copy(), float(vp)
                            cur_x, cur_v = yp.copy(), float(vp)
                            no_improve = 0
                    except Exception:
                        pass
            else:
                prog = (_time.time() - t0) / max(1e-9, (EXPL_END - t0))
                Temp = 0.004 * (1.0 - prog) + 0.0004
                if rng.random() < np.exp((cur_v - v) / max(Temp, 1e-6)):
                    cur_x, cur_v = y.copy(), float(v)
                no_improve += 1
            if no_improve > 25:
                # jump back near best
                cur_x = best_x + 0.12 * rng.standard_normal(size=(n, d))
                cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
                cur_x = _smooth_opt(cur_x, 500, 0.03, 0.003, 0.02, 0.004)
                cur_v = _max_cos(cur_x)
                no_improve = 0
                if it % 3 == 2:
                    # fresh exploration
                    f = _rand_sphere(n, d, rng)
                    f = _smooth_opt(f, 800, 0.04, 0.003, 0.03, 0.004)
                    fv = _max_cos(f)
                    if fv < cur_v:
                        cur_x, cur_v = f, fv
        it += 1

    # final thorough polish
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 2.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
