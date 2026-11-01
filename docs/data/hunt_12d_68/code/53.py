import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _make_lse(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _make_hinge(n, d, IU, T):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        diff = g - T
        pos = diff > 0
        dp = np.where(pos, diff, 0.0)
        F = float(dp @ dp)
        w = 2.0 * dp
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_lbfgs(x0, fg, maxiter):
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = r.x.reshape(x0.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0 / np.linalg.norm(x0, axis=1, keepdims=True)

def _best_hole(Xr, rng, ncand=20000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(s))]

def _single_fg(Xr, beta):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        m = float(s.max())
        e = np.exp(beta * (s - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    return fg

def _opt_single(Xr, z0, beta, maxiter=80):
    fg = _single_fg(Xr, beta)
    try:
        r = minimize(fg, z0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        z = r.x
        z = z / (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        return z

def _lp_polish(x, deadline, r0=6e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    n, d = 68, 12
    IU = np.triu_indices(n, 1)
    rng = np.random.default_rng(1234567)
    x_best = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = _max_cos(a)
                if s < bv:
                    bv = s
                    x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.standard_normal((n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        bv = _max_cos(x_best)
    # quick LP squeeze of incumbent to have tight baseline
    x_best = _lp_polish(x_best, t0 + 8.0, r0=5e-4, cap_p=2200)
    bv = _max_cos(x_best)
    # pool of diverse configs
    pool = [x_best.copy()]
    # seed pool with heavily perturbed versions polished lightly
    explore_end = t0 + TLIM - 20.0
    it = 0
    while time.time() < explore_end:
        it += 1
        mode = it % 6
        # pick base: mostly best, sometimes second
        if len(pool) > 1 and (it % 4 == 0):
            base = pool[rng.integers(len(pool))]
        else:
            base = x_best
        x = base.copy()
        if mode == 0:
            # global large Gaussian kick
            sig = float(rng.uniform(0.12, 0.35))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 1:
            # block replacement with random points
            k = int(rng.integers(6, 18))
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            # mix: partial interpolation to keep some memory
            alpha = float(rng.uniform(0.7, 1.0))
            x[idx] = alpha * rnd + (1 - alpha) * x[idx]
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 2:
            # multi-hole reinsertion of worst points
            pm = _per_max(x)
            k = int(rng.integers(3, 7))
            order = np.argsort(-pm)[:k]
            # remove all k at once, then refill sequentially with farthest sampling
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[order] = False
            Xr = x[keep_mask]
            newpts = []
            cur_set = Xr
            for _ in range(k):
                hole = _best_hole(cur_set, rng, ncand=20000)
                # local single-point refine
                hole = _opt_single(cur_set, hole, 250.0, maxiter=60)
                newpts.append(hole)
                cur_set = np.vstack([cur_set, hole[None, :]])
            for ii, idx in enumerate(order):
                x[idx] = newpts[ii]
            # small jitter on rest to break symmetry
            x = x + 0.01 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 3:
            # contact-break: push current contacts apart with few explicit steps
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            thr = m - 2e-4
            W = np.where(G > thr, G - thr, 0.0)
            np.fill_diagonal(W, 0.0)
            for _ in range(int(rng.integers(5, 15))):
                Grad = W @ x
                Tg = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
                step = float(rng.uniform(0.02, 0.08))
                x = x - step * Tg
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                thr = m - 2e-4
                W = np.where(G > thr, G - thr, 0.0)
                np.fill_diagonal(W, 0.0)
            # then add medium noise
            sig = float(rng.uniform(0.05, 0.15))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode == 4:
            # random orthogonal twist on subset: rotate by random small rotation
            k = int(rng.integers(10, 24))
            idx = rng.choice(n, size=k, replace=False)
            A = rng.standard_normal((d, d)) * 0.4
            A = A - A.T
            # exp approx via I + A
            R = np.eye(d) + A
            # orthonormalize via QR
            Q, _ = np.linalg.qr(R)
            if rng.random() < 0.5:
                x[idx] = x[idx] @ Q.T
            else:
                x[idx] = x[idx] @ Q
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # pure random restart-ish but biased to best hole structure
            sig = float(rng.uniform(0.25, 0.5))
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # hinge feasibility hop targeting just below best
        eps = float(rng.uniform(2e-4, 1.2e-3))
        T = bv - eps
        # occasionally target more ambitious
        if it % 11 == 0:
            T = bv - float(rng.uniform(1.2e-3, 2.5e-3))
        fg_h = _make_hinge(n, d, IU, T)
        # hinge iterations depend on time left
        xh = _run_lbfgs(x, fg_h, maxiter=180)
        vh = _max_cos(xh)
        # follow with smooth minimax LSE polish (two betas)
        # if hinge already feasible, polish harder; else still polish
        if vh < bv + 0.002 or vh < T + 1e-6:
            fg1 = _make_lse(n, d, IU, 130.0)
            xh = _run_lbfgs(xh, fg1, maxiter=120)
            fg2 = _make_lse(n, d, IU, 380.0)
            xh = _run_lbfgs(xh, fg2, maxiter=140)
            vh = _max_cos(xh)
            # opportunistic LP if promising
            if vh < bv + 5e-4 and time.time() < explore_end - 3.0:
                xh = _lp_polish(xh, min(time.time() + 1.2, explore_end), r0=4e-4, cap_p=1800)
                vh = _max_cos(xh)
        else:
            # hinge failed badly, try quick LSE anyway with fewer iters
            fg1 = _make_lse(n, d, IU, 150.0)
            xh = _run_lbfgs(xh, fg1, maxiter=80)
            vh = _max_cos(xh)
        if vh < bv - 1e-12:
            bv = vh
            x_best = xh.copy()
            pool.append(xh.copy())
            if len(pool) > 6:
                # drop worst in pool (by score)
                scores = [_max_cos(p) for p in pool]
                # keep best 6
                order = np.argsort(scores)[:6]
                pool = [pool[i] for i in order]
        elif vh < bv + 0.0015 and len(pool) < 6:
            # keep diverse near-best for crossover base
            # avoid duplicates: check distance
            pool.append(xh.copy())
            if len(pool) > 6:
                scores = [_max_cos(p) for p in pool]
                order = np.argsort(scores)[:6]
                pool = [pool[i] for i in order]
        # occasional single-point sweep on current best to squeeze
        if it % 8 == 0 and time.time() < explore_end - 5.0:
            pm = _per_max(x_best)
            order = np.argsort(-pm)[:3]
            xt = x_best.copy()
            improved = False
            for idx in order:
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Xr = xt[mask]
                cur = xt[idx].copy()
                cur_m = float((Xr @ cur).max())
                hole = _best_hole(Xr, rng, ncand=15000)
                best_y = cur; best_m = cur_m
                for beta in (150.0, 450.0):
                    for s in (cur, hole):
                        y = _opt_single(Xr, s, beta, maxiter=70)
                        m = float((Xr @ y).max())
                        if m < best_m:
                            best_m = m; best_y = y
                if best_m < cur_m - 1e-12:
                    xt[idx] = best_y
                    improved = True
            if improved:
                fg2 = _make_lse(n, d, IU, 350.0)
                xt = _run_lbfgs(xt, fg2, maxiter=100)
                vt = _max_cos(xt)
                if vt < bv:
                    bv = vt
                    x_best = xt.copy()
    # final intensive polish of best
    fg = _make_lse(n, d, IU, 250.0)
    x_best = _run_lbfgs(x_best, fg, maxiter=200)
    fg = _make_lse(n, d, IU, 600.0)
    x_best = _run_lbfgs(x_best, fg, maxiter=200)
    x_best = _lp_polish(x_best, t0 + TLIM - 0.5, r0=6e-4, cap_p=2500)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
