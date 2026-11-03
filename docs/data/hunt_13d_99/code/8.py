import time
import numpy as np

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _lbfgs_generic(x_init, kind, param, maxiter):
    from scipy.optimize import minimize
    n, d = x_init.shape
    Y0 = x_init.ravel().copy()
    iu = np.triu_indices(n, 1)
    if kind == 'lse':
        k = float(param)
        def fg(Yflat):
            Y = Yflat.reshape(n, d)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = Y / norms
            G = X @ X.T
            gv = G[iu]
            m = gv.max()
            e = np.exp(k * (gv - m))
            S = e.sum()
            loss = m + np.log(S) / k
            wv = e / S
            W = np.zeros((n, n))
            W[iu] = wv
            W = W + W.T
            grad_X = W @ X
            dot = (grad_X * X).sum(axis=1, keepdims=True)
            grad_Y = (grad_X - dot * X) / norms
            return float(loss), grad_Y.ravel()
    else:
        s = float(param)
        def fg(Yflat):
            Y = Yflat.reshape(n, d)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = Y / norms
            G = X @ X.T
            D2 = 2.0 - 2.0 * G
            np.fill_diagonal(D2, 1.0)
            D2 = np.maximum(D2, 1e-9)
            P = D2 ** (-s / 2.0)
            np.fill_diagonal(P, 0.0)
            loss = float(P[iu].sum())
            Wp = s * (D2 ** (-s / 2.0 - 1.0))
            np.fill_diagonal(Wp, 0.0)
            grad_X = Wp @ X
            dot = (grad_X * X).sum(axis=1, keepdims=True)
            grad_Y = (grad_X - dot * X) / norms
            # scale to avoid huge numbers
            sc = 1.0 / max(1.0, loss / n)
            return float(loss * sc), (grad_Y * sc).ravel()
    try:
        res = minimize(fg, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": maxiter * 6})
        Yopt = res.x.reshape(n, d)
        return _normalize(Yopt)
    except Exception:
        return x_init

def _lp_polish(x, time_budget, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < time_budget and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], ], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _void_point(others, rng, n_try=6000):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand = _normalize(cand)
    dots = cand @ others.T
    s = dots.max(axis=1)
    j = int(np.argmin(s))
    v = cand[j].copy()
    # local tangent refinement on smooth max
    for it in range(250):
        dts = others @ v
        m = dts.max()
        mask = dts > m - 0.03
        if not np.any(mask):
            break
        # softmax weighting
        w = np.exp(30.0 * (dts[mask] - m))
        w = w / w.sum()
        grad = (w[:, None] * others[mask]).sum(axis=0)
        grad_t = grad - (grad @ v) * v
        gn = np.linalg.norm(grad_t)
        if gn < 1e-12:
            break
        grad_t /= gn
        step = 0.02 / (1 + it * 0.03)
        v = v - step * grad_t
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        v /= nv
    return v

def _contacts_count(x, thresh=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - thresh).sum(axis=1)
    return cnt, m

def _anneal_single(x, rng, steps, step0, temp0):
    x = x.copy()
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    cur_max = float(G.max())
    per = G.max(axis=1)
    for st in range(steps):
        # pick worst-biased index
        if rng.rand() < 0.8:
            # weighted among worst 15
            order = np.argpartition(-per, 15)[:15]
            ii = int(order[rng.randint(len(order))])
        else:
            ii = int(rng.randint(n))
        step = step0 * (1.0 - 0.7 * st / max(1, steps))
        r = rng.randn(x.shape[1])
        r = r - (r @ x[ii]) * x[ii]
        rn = np.linalg.norm(r)
        if rn < 1e-12:
            continue
        r /= rn
        # random tangent magnitude
        alpha = step * (0.5 + rng.rand())
        v = x[ii] + alpha * r
        v /= max(1e-15, np.linalg.norm(v))
        dots = x @ v
        dots[ii] = -2.0
        new_per_ii = float(dots.max())
        # global max without ii
        mask = np.ones(n, dtype=bool)
        mask[ii] = False
        Go = G[np.ix_(mask, mask)].max() if n > 2 else -2.0
        new_global = new_per_ii if new_per_ii > Go else float(Go)
        # acceptance: greedy if improves point or global; else metropolis
        if new_global < cur_max - 1e-12 or new_per_ii < per[ii] - 1e-9:
            accept = True
        else:
            delta = new_global - cur_max
            T = temp0 * (1.0 - st / max(1, steps)) + 1e-6
            accept = rng.rand() < np.exp(-delta / max(T, 1e-9))
            # cap bad moves
            if delta > 0.02:
                accept = False
        if accept:
            x[ii] = v
            # update G row/col
            drow = x @ v
            G[ii, :] = drow
            G[:, ii] = drow
            G[ii, ii] = -2.0
            cur_max = float(G.max())
            per = G.max(axis=1)
            if cur_max < 0.20:
                pass
    return x, cur_max

def solve():
    t_start = time.time()
    deadline = t_start + 168.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(99, 13)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        flat = xb.ravel()
        need = 99 * 13
        if flat.size < need:
            extra = rng.randn(need - flat.size)
            flat = np.concatenate([flat, extra])
        xb = flat[:need].reshape(99, 13)
    xb = _normalize(xb)
    best = _max_cos(xb)
    x_best = xb.copy()

    trial = 0
    sigmas = [0.004, 0.01, 0.025, 0.06, 0.12]
    while time.time() < deadline - 22.0 and trial < 300:
        mode = trial % 6
        xp = x_best.copy()
        if mode == 0:
            sig = sigmas[(trial // 6) % len(sigmas)]
            xp = _normalize(xp + sig * rng.randn(*xp.shape))
            xp = _lbfgs_generic(xp, 'lse', 120, 90)
            if time.time() < deadline - 22.0:
                xp = _lbfgs_generic(xp, 'lse', 350, 70)
        elif mode == 1:
            # hub kill: remove 2-3 highest-contact points
            try:
                cnt, m = _contacts_count(xp, thresh=2e-6)
            except Exception:
                cnt = np.zeros(99)
            kkill = 2 + int(rng.randint(2))  # 2 or 3
            if trial % 12 == 1:
                kkill = 1
            hubs = np.argsort(-cnt)[:kkill + 2]
            # pick among top randomly
            pick = rng.choice(hubs, kkill, replace=False)
            for ii in pick:
                others = np.delete(xp, int(ii), axis=0)
                v = _void_point(others, rng, n_try=5000)
                xp[int(ii)] = v
            xp = _lbfgs_generic(xp, 'riesz', 6, 80)
            if time.time() < deadline - 22.0:
                xp = _lbfgs_generic(xp, 'lse', 250, 80)
        elif mode == 2:
            # patch shake: points near random direction
            u = rng.randn(13); u /= np.linalg.norm(u)
            proj = xp @ u
            thr = np.quantile(proj, 0.85)
            idx = np.where(proj >= thr)[0]
            if len(idx) < 3:
                idx = np.argsort(-proj)[:8]
            xp[idx] = _normalize(xp[idx] + 0.25 * rng.randn(len(idx), 13))
            # also small global noise
            xp = _normalize(xp + 0.005 * rng.randn(*xp.shape))
            xp = _lbfgs_generic(xp, 'lse', 80, 80)
            if time.time() < deadline - 22.0:
                xp = _lbfgs_generic(xp, 'lse', 300, 60)
        elif mode == 3:
            # single-point annealing then smooth
            sig = 0.01
            xp = _normalize(xp + sig * rng.randn(*xp.shape))
            xp, _ = _anneal_single(xp, rng, steps=800, step0=0.08, temp0=0.002)
            if time.time() < deadline - 22.0:
                xp = _lbfgs_generic(xp, 'lse', 200, 70)
        elif mode == 4:
            # Riesz heating with low s then cool
            xp = _normalize(xp + 0.03 * rng.randn(*xp.shape))
            xp = _lbfgs_generic(xp, 'riesz', 3, 60)
            if time.time() < deadline - 22.0:
                xp = _lbfgs_generic(xp, 'lse', 150, 70)
        else:
            # big random reinsert of worst point
            _, G = _per_point_max(xp)
            per = G.max(axis=1)
            ii = int(np.argmax(per))
            others = np.delete(xp, ii, axis=0)
            v = _void_point(others, rng, n_try=8000)
            xp[ii] = v
            # also kick second worst slightly
            per2 = per.copy(); per2[ii] = -1
            jj = int(np.argmax(per2))
            xp[jj] = _normalize(xp[jj] + 0.1 * rng.randn(13))
            xp = _lbfgs_generic(xp, 'lse', 180, 90)

        cur = _max_cos(xp)
        if cur < best + 0.008 and time.time() < deadline - 20.0:
            # short LP if promising
            try:
                xp2, v2 = _lp_polish(xp, time_budget=1.2, deadline=deadline - 18.0)
            except Exception:
                xp2, v2 = xp, cur
            cur2 = _max_cos(xp2)
            if cur2 < cur:
                xp, cur = xp2, cur2
        if cur < best - 1e-12:
            best = cur
            x_best = xp.copy()
        trial += 1

    # final polish
    remain = deadline - time.time()
    if remain > 4.0:
        try:
            # a few greedy void swaps on hubs
            for _ in range(3):
                cnt, _ = _contacts_count(x_best, thresh=2e-6)
                ii = int(np.argmax(cnt))
                others = np.delete(x_best, ii, axis=0)
                v = _void_point(others, rng, n_try=8000)
                xt = x_best.copy(); xt[ii] = v
                # quick smooth
                xt = _lbfgs_generic(xt, 'lse', 250, 50)
                c0 = _max_cos(xt)
                if c0 < _max_cos(x_best) - 1e-9:
                    x_best = xt
                else:
                    break
        except Exception:
            pass
    remain = deadline - time.time()
    if remain > 3.0:
        try:
            x_best, best = _lp_polish(x_best, time_budget=min(remain - 0.5, 22.0), deadline=deadline)
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best
