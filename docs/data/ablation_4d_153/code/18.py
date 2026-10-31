import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    l0, l1 = np.log(tau0), np.log(tau1)
    a0, a1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(l0 + (l1 - l0) * f))
        lr = float(np.exp(a0 + (a1 - a0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1, keepdims=True)
        np.maximum(rs, 1e-12, out=rs)
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad -= dot * x
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 3000:
            part = np.argpartition(c[idx], -3000)[-3000:]
            idx = idx[part]
        I = iu0[0][idx]
        J = iu0[1][idx]
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
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
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 132.0
    rng = np.random.default_rng(20261031)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.30]
    it = 0
    T0 = 0.0025
    while time.time() < EXPLORE_END and it < 600:
        it += 1
        T = max(T0 * (0.985 ** it), 0.00015)
        r = float(rng.random())
        y = None
        if r < 0.32:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            # occasionally huge kick
            if it % 17 == 0:
                sigma = 0.5
            y = cur_x + sigma * rng.standard_normal(size=(n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.52:
            # jitter worst points
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            rm = np.max(G, axis=1)
            k = int(rng.integers(2, 6))
            idx = np.argpartition(rm, -k)[-k:]
            y = cur_x.copy()
            sigma = float(rng.choice(np.array([0.03, 0.08, 0.2])))
            y[idx] += sigma * rng.standard_normal(size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif r < 0.68:
            # subset reinit
            k = int(rng.integers(3, 12))
            idx = rng.choice(n, size=k, replace=False)
            y = cur_x.copy()
            rnd = rng.standard_normal(size=(k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            y[idx] = rnd
        elif r < 0.85:
            # split worst pair
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            ii = flat // n
            jj = flat % n
            y = cur_x.copy()
            alpha = 0.12
            vi = y[ii] - y[jj]
            y[ii] = y[ii] + alpha * vi + 0.02 * rng.standard_normal(size=(d,))
            y[jj] = y[jj] - alpha * vi + 0.02 * rng.standard_normal(size=(d,))
            # also jitter a random third point a bit
            kk = int(rng.integers(0, n))
            y[kk] = y[kk] + 0.03 * rng.standard_normal(size=(d,))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # rattler void fill
            G = cur_x @ cur_x.T
            np.fill_diagonal(G, -2.0)
            rm = np.max(G, axis=1)
            nl = 3
            idx_low = np.argpartition(rm, nl)[:nl]
            y = cur_x.copy()
            for kk in idx_low:
                cand = rng.standard_normal(size=(512, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ y.T
                dots[:, kk] = -2.0
                md = np.max(dots, axis=1)
                y[kk] = cand[int(np.argmin(md))]
        # annealed smooth repulsion
        # vary steps with time
        steps = 1000
        if it % 5 == 0:
            steps = 1400
        tau0 = 0.025
        tau1 = 0.002
        if it % 7 == 0:
            tau0 = 0.04
            tau1 = 0.0012
        y = _smooth_opt(y, steps=steps, tau0=tau0, tau1=tau1, lr0=0.04, lr1=0.004, mom=0.9)
        v = _max_cos(y)
        if v < cur_v or float(rng.random()) < float(np.exp((cur_v - v) / T)):
            cur_x = y
            cur_v = v
            if v < best_v:
                # squeeze this basin quickly
                dl = min(time.time() + 2.5, EXPLORE_END)
                try:
                    y2, v2 = _lp_polish(y, dl)
                except Exception:
                    y2, v2 = y, v
                if v2 < best_v:
                    best_x = y2.copy()
                    best_v = v2
                    cur_x = y2.copy()
                    cur_v = v2
                else:
                    # keep raw if polish failed to improve but raw is better than old best
                    if v < best_v:
                        best_x = y.copy()
                        best_v = v
        if time.time() >= EXPLORE_END:
            break
    # final polish of global best
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
