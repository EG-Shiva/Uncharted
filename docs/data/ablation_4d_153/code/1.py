import time
import numpy as np


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


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
        # weights exp((G-m)/tau)
        # G has -inf on diagonal -> 0 weight
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        # tangent projection
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        # clip huge grads
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        # avoid extreme steps: clip at 2
        scale = np.minimum(1.0, 2.0 / np.maximum(gn, 1e-12))
        # only clip when gn>2
        grad = grad * np.where(gn > 2.0, scale, 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        # bias correction for early steps
        bc = 1.0 - mom ** (t + 1)
        step = (vel / bc)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x


def _lp_polish(x, deadline, verbose=False):
    # lazy imports
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    c_all = None
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        # cap number of constraints for speed
        p0 = int(np.count_nonzero(sel))
        if p0 == 0:
            break
        idx = np.where(sel)[0]
        if p0 > 2500:
            # keep largest 2500
            order = np.argpartition(c[idx], -2500)[-2500:]
            idx = idx[order]
            p0 = 2500
            sel2 = np.zeros_like(sel)
            sel2[idx] = True
            sel = sel2
            I = iu[0][idx]
            J = iu[1][idx]
            csel = c[idx]
        else:
            I = iu[0][sel]
            J = iu[1][sel]
            csel = c[sel]
        p = p0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
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
    # exploration budget, reserve final polish
    rng = np.random.default_rng(0)
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

    sigmas = [0.005, 0.012, 0.03, 0.07, 0.13, 0.25, 0.45, 0.8]

    # number of basin hops; fixed for determinism, with time guard
    NITER = 45
    for it in range(NITER):
        if _time.time() > HARD_END - 55.0:
            break
        sigma = sigmas[it % len(sigmas)]
        do_random = (it % 11 == 10)
        if do_random:
            x0 = _random_sphere(n, d, rng)
            steps = 3500
            tau0, tau1 = 0.06, 0.004
            lr0, lr1 = 0.10, 0.004
        else:
            # perturb current walker (basin hopping), occasionally from best
            base = cur_x if (it % 3 != 2) else best_x
            noise = rng.standard_normal(size=(n, d))
            x0 = base + sigma * noise
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
            if it % 4 == 3:
                k = int(rng.integers(1, 7))
                idx = rng.choice(n, size=k, replace=False)
                rp = rng.standard_normal(size=(k, d))
                rp /= np.linalg.norm(rp, axis=1, keepdims=True)
                x0[idx] = rp
            if sigma < 0.05:
                steps = 2200
                tau0, tau1 = 0.025, 0.0025
                lr0, lr1 = 0.06, 0.003
            elif sigma < 0.2:
                steps = 2600
                tau0, tau1 = 0.035, 0.003
                lr0, lr1 = 0.08, 0.003
            else:
                steps = 3000
                tau0, tau1 = 0.05, 0.0035
                lr0, lr1 = 0.09, 0.003
        x1 = _smooth_opt(x0, steps, tau0, tau1, lr0, lr1, mom=0.88)
        v1 = _max_cos(x1)
        # quick LP refine if promising (close to best)
        if v1 < best_v + 0.004 and _time.time() < HARD_END - 55.0:
            # short polish 1.2s
            dl = min(_time.time() + 1.2, HARD_END - 50.0)
            if dl > _time.time() + 0.2:
                x1p, v1p = _lp_polish(x1, dl)
                # keep polished version if better
                if v1p < v1:
                    x1, v1 = x1p, v1p
        if v1 < best_v:
            best_x = x1.copy()
            best_v = v1
        # walker move: accept if better or slightly worse
        # uphill tolerance decays
        tol = 0.004 * (1.0 - it / NITER) + 0.0005
        if v1 < cur_v + tol:
            cur_x = x1.copy()
            cur_v = v1
            # occasionally reset walker to best if wandering too high
            if cur_v > best_v + 0.01:
                cur_x = best_x.copy()
                cur_v = best_v
        else:
            # small chance to restart walker from best
            if rng.random() < 0.15:
                cur_x = best_x.copy()
                cur_v = best_v
        # hard time guard
        if _time.time() > HARD_END - 55.0:
            break

    # final thorough polish of best
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
