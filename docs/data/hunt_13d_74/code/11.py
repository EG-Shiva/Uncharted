import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _lbfgs_smooth(x_init, T, maxiter, iu):
    n, d = x_init.shape
    x0 = x_init.ravel().copy()
    I0 = iu[0]; J0 = iu[1]
    def fun(g):
        X = g.reshape(n, d)
        r = np.linalg.norm(X, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Y = X / r
        G = Y @ Y.T
        dots = G[iu]
        mx = dots.max()
        e = np.exp((dots - mx) / T)
        s = e.sum()
        loss = mx + T * np.log(s + 1e-300)
        w = e / (s + 1e-300)
        Wx = np.zeros_like(Y)
        # vectorized scatter
        np.add.at(Wx, I0, w[:, None] * Y[J0])
        np.add.at(Wx, J0, w[:, None] * Y[I0])
        dot = np.sum(Wx * Y, axis=1, keepdims=True)
        gradX = (Wx - Y * dot) / r
        return float(loss), gradX.ravel()
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9, "maxfun": maxiter * 8})
        X = res.x.reshape(n, d)
        X = _norm(X)
        if not np.all(np.isfinite(X)):
            return x_init
        return X
    except Exception:
        return x_init

def _fine_max(X, steps, lr0, deadline=None):
    cur = _max_cos(X)
    lr = lr0
    n = X.shape[0]
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        delta = max(4.0 * lr, 1e-7)
        mask = G > (gmax - delta)
        A = mask.astype(np.float64)
        Gr = A @ X
        d = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - d * X
        norms = np.linalg.norm(Gt, axis=1, keepdims=True)
        step = np.zeros_like(Gt)
        act = norms[:, 0] > 1e-14
        if not np.any(act):
            break
        step[act] = Gt[act] / (norms[act] + 1e-12) * lr
        Xn = _norm(X - step)
        nv = _max_cos(Xn)
        if nv < cur:
            X[:] = Xn
            cur = nv
            lr *= 1.07
        else:
            lr *= 0.5
            if lr < 1e-11:
                break
    return X, cur

def _lp_polish(x, deadline, radius_init=0.003):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(5.0 * radius, 2e-9)
        sel = c_all > thr
        # cap to tightest 1800 for speed / tightness
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1800:
            idx = np.argpartition(c_all, -1800)[-1800:]
            msel = np.zeros_like(sel)
            msel[idx] = True
            sel = msel
            cnt = 1800
        I = iu[0][sel]; J = iu[1][sel]; c = c_all[sel]
        p = cnt
        nv = n * d + 1
        Cb = np.empty((p, 2 * d + 1), dtype=np.int64)
        Cb[:, :d] = I[:, None] * d + np.arange(d)
        Cb[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Cb[:, 2 * d] = n * d
        cols = Cb.ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Vb = np.empty((p, 2 * d + 1))
        Vb[:, :d] = x[J]
        Vb[:, d:2 * d] = x[I]
        Vb[:, 2 * d] = -1.0
        vals = Vb.ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _optimize_single(others, y_inits, T=0.0025, maxiter=200):
    best_y = None; best_v = 1e9
    for y0 in y_inits:
        z0 = y0.copy()
        def fun(z):
            nz = float(np.linalg.norm(z))
            nz = max(nz, 1e-12)
            y = z / nz
            dots = others @ y
            mx = float(dots.max())
            e = np.exp((dots - mx) / T)
            s = float(e.sum())
            F = mx + T * np.log(s + 1e-300)
            w = e / (s + 1e-300)
            gy = w @ others
            gz = (gy - y * float(gy @ y)) / nz
            return float(F), gz
        try:
            res = minimize(fun, z0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-9})
            z = res.x
            y = z / max(float(np.linalg.norm(z)), 1e-12)
        except Exception:
            y = y0 / max(float(np.linalg.norm(y0)), 1e-12)
        v = float((others @ y).max())
        if v < best_v:
            best_v = v; best_y = y
    return best_y, best_v

def _hole_reinsert(X, idx, rng):
    n, d = X.shape
    others = np.delete(X, idx, axis=0)
    cand = _norm(rng.standard_normal((3000, d)))
    sc = cand @ X.T
    sc[:, idx] = -10.0
    smax = sc.max(axis=1)
    order = np.argsort(smax)[:4]
    y_inits = [cand[j] for j in order]
    # add opposite of worst neighbor as init
    G = X @ X.T
    G[idx, idx] = -2.0
    jw = int(np.argmax(G[idx]))
    y_inits.append(_norm(-X[jw][None, :])[0])
    yb, _ = _optimize_single(others, y_inits, T=0.0025, maxiter=150)
    X[idx] = yb / max(float(np.linalg.norm(yb)), 1e-12)
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline_all = t0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr); xb = _norm(xb)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb = _norm(xb)
    n, d = xb.shape
    iu = np.triu_indices(n, 1)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def worst_list(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        pm = G.max(axis=1)
        return np.argsort(-pm)[:k]

    # initial squeeze of incumbent
    try:
        X0 = x_best.copy()
        X0 = _lbfgs_smooth(X0, 0.004, 80, iu)
        X0 = _lbfgs_smooth(X0, 0.0012, 120, iu)
        X0, v0 = _fine_max(X0, 350, 0.002, deadline_all - 25.0)
        if v0 < best_val:
            x_best, best_val = X0, v0
        # tight LP if time
        if deadline_all - time.time() > 30:
            Xp = _lp_polish(x_best, time.time() + 3.0, radius_init=0.002)
            vp = _max_cos(Xp)
            if vp < best_val:
                x_best, best_val = Xp, vp
    except Exception:
        pass

    sigma = 0.012
    outer = 0
    stag = 0
    while time.time() < deadline_all - 22.0 and outer < 60:
        remain = deadline_all - time.time()
        if remain < 24:
            break
        mode = outer % 8
        xc = x_best.copy()
        if mode == 0:
            xc = _norm(xc + rng.standard_normal((n, d)) * 0.006)
        elif mode == 1:
            xc = _norm(xc + rng.standard_normal((n, d)) * sigma)
        elif mode == 2:
            wl = worst_list(xc, 4)
            xc[wl] += rng.standard_normal((len(wl), d)) * 0.08
            xc = _norm(xc + rng.standard_normal((n, d)) * 0.005)
        elif mode == 3:
            wi = worst_list(x_best, 1)[0]
            xc = x_best.copy()
            try:
                xc = _hole_reinsert(xc, int(wi), rng)
            except Exception:
                v = rng.standard_normal(d); v /= np.linalg.norm(v); xc[int(wi)] = v
            xc = _norm(xc + rng.standard_normal((n, d)) * 0.002)
        elif mode == 4:
            xc = _norm(xc + rng.standard_normal((n, d)) * 0.035)
        elif mode == 5:
            wl = worst_list(x_best, 2)
            xc = x_best.copy()
            for idx in wl:
                try:
                    xc = _hole_reinsert(xc, int(idx), rng)
                except Exception:
                    v = rng.standard_normal(d); v /= np.linalg.norm(v); xc[int(idx)] = v
        elif mode == 6:
            sub = rng.choice(n, size=10, replace=False)
            xc[sub] += rng.standard_normal((10, d)) * 0.15
            xc = _norm(xc)
        else:
            xc = _norm(xc + rng.standard_normal((n, d)) * 0.015)

        # annealed smooth
        xc = _lbfgs_smooth(xc, 0.008, 60, iu)
        if time.time() > deadline_all - 22.0:
            cur = _max_cos(xc)
            if cur < best_val:
                x_best, best_val = xc, cur
            break
        xc = _lbfgs_smooth(xc, 0.0025, 80, iu)
        xc = _lbfgs_smooth(xc, 0.001, 80, iu)
        xc, cur = _fine_max(xc, 250, 0.0015, deadline_all - 20.0)
        if cur < best_val - 1e-12:
            x_best = xc.copy(); best_val = cur; stag = 0
            sigma = max(0.004, sigma * 0.9)
            # squeeze further with LP
            if deadline_all - time.time() > 25:
                Xp = _lp_polish(x_best, min(time.time() + 3.0, deadline_all - 20.0), radius_init=0.002)
                vp = _max_cos(Xp)
                if vp < best_val:
                    x_best, best_val = Xp, vp
        else:
            stag += 1
            sigma = min(0.035, sigma * 1.07)
            # occasional LP on promising near-best to change topology
            if cur < best_val + 0.002 and (outer % 5 == 0) and deadline_all - time.time() > 28:
                Xp = _lp_polish(xc, time.time() + 1.5, radius_init=0.002)
                vp = _max_cos(Xp)
                if vp < best_val - 1e-12:
                    x_best, best_val = Xp, vp
                    stag = 0
        outer += 1

    # final intensive polish
    try:
        if deadline_all - time.time() > 4.0:
            x_best = _lbfgs_smooth(x_best, 0.0012, 120, iu)
            x_best, bv = _fine_max(x_best, 400, 0.0012, deadline_all - 2.0)
            best_val = bv
        if deadline_all - time.time() > 2.5:
            x_best = _lp_polish(x_best, deadline_all - 1.0, radius_init=0.0015)
        if deadline_all - time.time() > 1.2:
            x_best = _lp_polish(x_best, deadline_all - 0.4, radius_init=0.0004)
    except Exception:
        pass
    x_best = _norm(np.where(np.isnan(x_best), 0.0, x_best))
    return x_best
