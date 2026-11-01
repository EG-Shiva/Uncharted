import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _make_global_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_global(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = _make_global_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 2, 'maxcor': 12})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _best_hole(Xrest, rng, ncand=20000):
    d = Xrest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ Xrest.T).max(axis=1)
    return cand[int(np.argmin(scores))]

def _single_fg_factory(Xr, beta):
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

def _opt_single(Xr, y_start, beta, maxiter=100):
    fg = _single_fg_factory(Xr, beta)
    try:
        r = minimize(fg, y_start.copy(), jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        return y_start / (np.linalg.norm(y_start) + 1e-300)

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() > deadline - 0.6:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    T_LIMIT = 168.0
    def tleft():
        return T_LIMIT - (time.time() - t_start)
    rng = np.random.default_rng(0)
    n, d = 68, 12
    try:
        bestX = np.load("best.npy")
    except Exception:
        try:
            bestX = np.load("record.npy")
        except Exception:
            bestX = rng.standard_normal((n, d))
    bestX = np.asarray(bestX, dtype=np.float64)
    if bestX.shape != (n, d):
        bestX = rng.standard_normal((n, d))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    best_score = _max_cos(bestX)
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=np.float64)
        if rec.shape == (n, d):
            rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
            rs = _max_cos(rec)
            if rs < best_score:
                best_score = rs
                bestX = rec.copy()
    except Exception:
        pass

    # phase 1: quick squeeze of current basin at rising beta
    for beta, mi in [(80.0, 180), (200.0, 220), (500.0, 250)]:
        if tleft() < 30:
            break
        Y = _run_global(bestX, beta, mi)
        s = _max_cos(Y)
        if s < best_score and np.all(np.isfinite(Y)):
            best_score = s
            bestX = Y

    # phase 2: basin hopping with topology changes
    trial = 0
    explore_end = t_start + 118.0
    while time.time() < explore_end and tleft() > 32.0 and trial < 70:
        trial += 1
        mode = trial % 6
        Xinit = None
        if mode == 0:
            sig = 0.03
            Xinit = bestX + sig * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b1, b2, m1, m2 = 60.0, 350.0, 180, 220
        elif mode == 1:
            sig = 0.14 + 0.04 * ((trial // 6) % 3)
            Xinit = bestX + sig * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b1, b2, m1, m2 = 50.0, 400.0, 200, 220
        elif mode == 2:
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T
            np.fill_diagonal(G, -2.0)
            thr = best_score - 0.005
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            k = 1 + (trial // 6) % 2
            for kk in range(k):
                victim = int(order[kk])
                mask = np.ones(n, dtype=bool); mask[victim] = False
                h = _best_hole(Xinit[mask], rng, ncand=16000)
                h = h + 0.01 * rng.standard_normal(d)
                h /= np.linalg.norm(h)
                Xinit[victim] = h
            b1, b2, m1, m2 = 70.0, 450.0, 200, 220
        elif mode == 3:
            Xinit = bestX.copy()
            G = Xinit @ Xinit.T
            np.fill_diagonal(G, -2.0)
            thr = best_score - 0.005
            cnt = np.sum(G > thr, axis=1)
            order = np.argsort(-cnt)
            for kk in range(3):
                victim = int(order[kk])
                mask = np.ones(n, dtype=bool); mask[victim] = False
                h = _best_hole(Xinit[mask], rng, ncand=12000)
                Xinit[victim] = h
            Xinit += 0.02 * rng.standard_normal((n, d))
            Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
            b1, b2, m1, m2 = 60.0, 500.0, 200, 250
        elif mode == 4:
            Xinit = bestX.copy()
            idx = rng.choice(n, size=6, replace=False)
            R = rng.standard_normal((6, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Xinit[idx] = R
            b1, b2, m1, m2 = 50.0, 380.0, 220, 230
        else:
            if (trial // 6) % 4 == 0:
                Xinit = rng.standard_normal((n, d))
                Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
                b1, b2, m1, m2 = 40.0, 300.0, 250, 250
            else:
                sig = 0.35
                Xinit = bestX + sig * rng.standard_normal((n, d))
                Xinit /= np.linalg.norm(Xinit, axis=1, keepdims=True)
                b1, b2, m1, m2 = 40.0, 320.0, 220, 230
        if tleft() < 32:
            break
        Y = _run_global(Xinit, b1, m1)
        Y = _run_global(Y, b2, m2)
        s = _max_cos(Y)
        if s < best_score and np.all(np.isfinite(Y)):
            best_score = s
            bestX = Y.copy()

    # phase 3: intensify best with high beta
    if tleft() > 28:
        for beta, mi in [(250.0, 300), (700.0, 350), (1400.0, 300)]:
            if tleft() < 26:
                break
            Y = _run_global(bestX, beta, mi)
            s = _max_cos(Y)
            if s < best_score and np.all(np.isfinite(Y)):
                best_score = s
                bestX = Y.copy()

    # phase 4: single-point minimax sweeps (helps if floppy bottlenecks)
    if tleft() > 24:
        for _sweep in range(2):
            if tleft() < 22:
                break
            pm = _per_max(bestX)
            order = np.argsort(-pm)[:12]
            improved = False
            for idx in order:
                if tleft() < 21.5:
                    break
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Xr = bestX[mask]
                rest_max = _max_cos(Xr)
                if rest_max >= best_score - 1e-12:
                    continue
                cur = bestX[idx].copy() / np.linalg.norm(bestX[idx])
                cur_max = float((Xr @ cur).max())
                hole = _best_hole(Xr, rng, ncand=20000)
                pv = hole + 0.05 * rng.standard_normal(d)
                pv /= np.linalg.norm(pv)
                best_y, best_m = cur, cur_max
                for st in (cur, hole, pv):
                    for beta in (150.0, 500.0):
                        y = _opt_single(Xr, st, beta, maxiter=80)
                        m = float((Xr @ y).max())
                        if m < best_m:
                            best_m, best_y = m, y
                new_global = max(rest_max, best_m)
                if new_global < best_score - 1e-12:
                    bestX[idx] = best_y / np.linalg.norm(best_y)
                    best_score = _max_cos(bestX)
                    improved = True
            if not improved:
                break

    # phase 5: trust-region LP polish
    remain = tleft()
    if remain > 7.0:
        lp_time = min(36.0, remain - 2.0)
        deadline = time.time() + lp_time
        bestX = _lp_polish(bestX, deadline)
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
