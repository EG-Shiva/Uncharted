import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu0]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = cfull[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)[None, :].repeat(p, axis=0)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _softmax_descent(x0, steps, T0, T1, lr0, mu=0.85):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.7 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300:
            S = 1e-300
        grad = (E @ x) / S
        dot = (grad * x).sum(axis=1, keepdims=True)
        grad -= dot * x
        v = mu * v + (1.0 - mu) * grad
        x -= lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _riesz_descent(x0, s0, s1, steps, lr0, mu=0.9):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        frac = k / max(1, steps - 1)
        s = s0 * (s1 / s0) ** frac if s0 > 0 and s1 > 0 else s0
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        D = 2.0 - 2.0 * G
        np.fill_diagonal(D, 1.0)
        D = np.maximum(D, 1e-12)
        dmin = float(np.min(D[np.eye(n, dtype=bool) == False].reshape(n, n - 1).min(axis=1) if False else D[~np.eye(n, dtype=bool)]))
        # D min over off-diagonal
        # compute directly for stability
        # dmin already min of all off-diag (diag=1, off-diag ~0.28 so min is off-diag)
        pw = s / 2.0 + 1.0
        # normalized weights (max=1)
        W = (dmin / D) ** pw
        np.fill_diagonal(W, 0.0)
        Gx = W @ x
        dot = (Gx * x).sum(axis=1, keepdims=True)
        grad = Gx - dot * x
        # normalize grad scale to avoid blowup with s
        gn = float(np.sqrt((grad * grad).sum() / n))
        if gn > 1e-12:
            grad /= max(1.0, gn * 2.0)
        vel = mu * vel + (1.0 - mu) * grad
        x -= lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _find_hole(x_fixed, rng, n_samples=20000):
    Y = rng.randn(n_samples, 4)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    D = Y @ x_fixed.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    return Y[idx].copy(), float(vals[idx])

def _refine_holes(x_fixed, y_init, steps, T0, T1, lr0, mu=0.85):
    # jointly optimize k free points against fixed set + each other
    y = y_init.copy()
    k = y.shape[0]
    vel = np.zeros_like(y)
    def eval_max(yy):
        g1 = yy @ yy.T
        if k > 1:
            np.fill_diagonal(g1, -2.0)
            m1 = float(g1.max())
        else:
            m1 = -2.0
        g2 = yy @ x_fixed.T
        m2 = float(g2.max())
        return m1 if m1 > m2 else m2
    best_val = eval_max(y)
    best = y.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.7 * frac)
        Gyy = y @ y.T
        Gyx = y @ x_fixed.T
        if k > 1:
            np.fill_diagonal(Gyy, -2.0)
            m = Gyy.max()
        else:
            m = -2.0
        m2 = float(Gyx.max())
        if m2 > m:
            m = m2
        Eyy = np.exp((Gyy - m) / T) if k > 1 else np.zeros((k, k))
        if k > 1:
            np.fill_diagonal(Eyy, 0.0)
            Syy = float(Eyy.sum() * 0.5)
        else:
            Syy = 0.0
        Eyx = np.exp((Gyx - m) / T)
        Syx = float(Eyx.sum())
        S = Syy + Syx
        if S < 1e-300:
            S = 1e-300
        grad = np.zeros_like(y)
        if k > 1:
            grad += (Eyy @ y) / S
        grad += (Eyx @ x_fixed) / S
        dot = (grad * y).sum(axis=1, keepdims=True)
        grad -= dot * y
        vel = mu * vel + (1.0 - mu) * grad
        y -= lr * vel
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        cm = eval_max(y)
        if cm < best_val:
            best_val = cm
            best = y.copy()
    return best, best_val

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(163, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (163, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(163, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    n = 163
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    # reserve final polish; quick confirm already near optimum so skip long initial LP
    trial = 0
    sigmas = [0.02, 0.05, 0.10, 0.18, 0.008, 0.035]
    s_pairs = [(15, 60), (40, 150), (80, 300), (10, 30)]
    SEARCH_END = t0 + 128.0
    while time.time() < SEARCH_END:
        remain = SEARCH_END - time.time()
        if remain < 1.0:
            break
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 6
        cur = best_x.copy()
        use_riesz = False
        riesz_s0, riesz_s1 = s_pairs[(trial // 2) % len(s_pairs)]
        if mode == 0:
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 1:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            idx = np.argsort(deg)[-10:]
            noise = rng.randn(*cur.shape) * (sigma * 0.25)
            noise[idx] += rng.randn(len(idx), 4) * (sigma * 2.0)
            cur = cur + noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 2:
            # worst-k removal + joint hole refinement
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            k = 2 + (trial % 4)  # 2..5
            worst = np.argsort(deg)[-k:]
            # also include a random high-degree neighbor to break cluster
            keep = np.delete(np.arange(n), worst)
            xf = cur[keep].copy()
            # greedy hole init
            yinit = np.zeros((k, 4))
            xt = xf.copy()
            for ii in range(k):
                h, _ = _find_hole(xt, rng, n_samples=20000)
                h = h + rng.randn(4) * 0.01
                h /= np.linalg.norm(h)
                yinit[ii] = h
                xt = np.vstack([xt, h[None, :]])
            yopt, _ = _refine_holes(xf, yinit, steps=2500, T0=0.015, T1=0.0012, lr0=0.04)
            for ii, w in enumerate(worst):
                cur[w] = yopt[ii]
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.12)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        elif mode == 3:
            k = 24
            idx = rng.choice(n, k, replace=False)
            cur[idx] = cur[idx] + rng.randn(k, 4) * (sigma * 1.5)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            use_riesz = True
        elif mode == 4:
            # single rattler hole reinsertion
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.001).sum(axis=1)
            w = int(np.argmin(deg))
            keep = np.delete(np.arange(n), w)
            xf = cur[keep]
            h, _ = _find_hole(xf, rng, n_samples=25000)
            h = h + rng.randn(4) * 0.005
            h /= np.linalg.norm(h)
            cur[w] = h
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.3)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        else:
            if (trial // 6) % 3 == 2:
                cur = rng.randn(n, 4)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                use_riesz = True
            else:
                cur = cur + rng.randn(*cur.shape) * sigma
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                use_riesz = ((trial // 6) % 2 == 0)
        # anneal
        if use_riesz:
            # Riesz first then short softmax
            cur_r, v_r = _riesz_descent(cur, riesz_s0, riesz_s1, steps=3000, lr0=0.03)
            cur_r2, v2 = _softmax_descent(cur_r, steps=2500, T0=0.008, T1=0.0012, lr0=0.030)
            cur_f, v = (cur_r2, v2) if v2 < v_r else (cur_r, v_r)
        else:
            if mode == 2:
                cur_f, v = _softmax_descent(cur, steps=4000, T0=0.012, T1=0.0010, lr0=0.032)
            else:
                cur_f, v = _softmax_descent(cur, steps=4500, T0=0.015, T1=0.0012, lr0=0.030)
        if v < best_val:
            # confirm with short LP if time allows
            if time.time() - t0 < 150.0:
                pol = _lp_polish(cur_f, seconds=2.0, t_end=t0 + BUDGET - 32.0)
                vp = _max_cos(pol)
                if vp < best_val:
                    best_x, best_val = pol, vp
                # else keep descent result if still better than best (LP may have failed)
                elif v < best_val:
                    best_x, best_val = cur_f, v
            else:
                best_x, best_val = cur_f, v
        elif v < best_val + 0.0004 and (trial % 7 == 3) and time.time() - t0 < 140.0:
            # opportunistic LP on near-miss (can still jump below)
            pol = _lp_polish(cur_f, seconds=1.2, t_end=t0 + BUDGET - 32.0)
            vp = _max_cos(pol)
            if vp < best_val:
                best_x, best_val = pol, vp
        trial += 1
        if time.time() - t0 > BUDGET - 32.0:
            break
    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4.0:
        remain = min(remain, 38.0)
        best_x = _lp_polish(best_x, seconds=remain)
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
