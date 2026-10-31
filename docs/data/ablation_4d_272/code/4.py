import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth_anneal(x_init, steps, T0, T1, lr0, lr1, cap0, cap1, mu):
    x = x_init.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    # initial best
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_x = x.copy()
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        # log interp for T, linear for lr/cap
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 + (lr1 - lr0) * frac
        cap = cap0 + (cap1 - cap0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_x = x.copy()
        # weights exp((G-m)/T) <=1
        # in-place to save alloc
        W = np.exp((G - m) / T)
        S = float(W.sum() * 0.5)
        if S < 1e-12:
            S = 1e-12
        Grad = (W @ x) / S
        dot = np.einsum('ij,ij->i', Grad, x)[:, None]
        Gt = Grad - dot * x
        v = mu * v + Gt
        step = lr * v
        # cap max row move
        # fast norm
        n2 = np.sqrt((step * step).sum(axis=1))
        mx = float(n2.max())
        if mx > cap and mx > 1e-14:
            step *= (cap / mx)
        x = x - step
        nr = np.sqrt((x * x).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        x /= nr
    # final check
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_m:
        best_m = m
        best_x = x.copy()
    return best_x, best_m

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 2e-9)
        sel = cfull > thr
        I = iu[0][sel]
        J = iu[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > 6000:
            # keep worst 6000
            idx = np.argpartition(csel, -6000)[-6000:]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 6000
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    import time
    import numpy as np
    t0 = time.time()
    LIMIT = 178.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (272, 4):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((272, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = None
    m_best = 1e9
    for c in cands:
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        # guard nan
        if not np.all(np.isfinite(c)):
            continue
        m = _max_cos(c)
        if m < m_best:
            m_best = m
            x_best = c.copy()
    n, d = x_best.shape
    sigmas = [0.035, 0.025, 0.018, 0.012, 0.007, 0.004]
    trial = 0
    # Phase A: basin hopping until ~110s
    while time.time() - t0 < 108.0:
        trial += 1
        if time.time() - t0 > LIMIT - 40:
            break
        cyc = trial % 6
        sigma = sigmas[cyc]
        if trial % 3 == 0:
            T0, T1 = 0.022, 0.004
        elif trial % 3 == 1:
            T0, T1 = 0.014, 0.0025
        else:
            T0, T1 = 0.009, 0.0015
        lr0, lr1 = 0.008, 0.002
        cap0, cap1 = 0.015, 0.002
        mu = 0.9
        # generate try
        if trial % 7 == 0:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            mask = per < (m_best - 0.015)
            if 1 <= int(mask.sum()) < 100:
                x_try = x_best.copy()
                r = rng.standard_normal((int(mask.sum()), 4))
                r /= np.linalg.norm(r, axis=1, keepdims=True)
                x_try[mask] = r
                x_try = x_try + 0.005 * rng.standard_normal(x_try.shape)
            else:
                x_try = x_best + sigma * rng.standard_normal(x_best.shape)
        else:
            x_try = x_best + sigma * rng.standard_normal(x_best.shape)
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        steps = 900
        # shrink steps if little time left in phase A
        remA = 108.0 - (time.time() - t0)
        if remA < 4 and steps > 300:
            steps = 300
        try:
            bx, bm = _smooth_anneal(x_try, steps, T0, T1, lr0, lr1, cap0, cap1, mu)
        except Exception:
            continue
        if bm < m_best - 1e-12 and np.all(np.isfinite(bx)):
            x_best = bx
            m_best = bm
    # Phase B: long fine anneal from best
    if time.time() - t0 < LIMIT - 35:
        try:
            stepsB = 3000
            # adapt to remaining time: aim to leave 32s for LP
            rem = (LIMIT - 32) - (time.time() - t0)
            if rem < 8:
                stepsB = 800
            elif rem < 20:
                stepsB = 1500
            bx, bm = _smooth_anneal(x_best, stepsB, 0.007, 0.0012, 0.005, 0.001, 0.008, 0.001, 0.9)
            if bm < m_best and np.all(np.isfinite(bx)):
                x_best = bx
                m_best = bm
        except Exception:
            pass
    # Phase C: LP polish with remaining time
    deadline = t0 + LIMIT - 2.0
    if time.time() < deadline - 3.0:
        try:
            x_best = _lp_polish(x_best, deadline)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # final safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = cands[0] / np.linalg.norm(cands[0], axis=1, keepdims=True)
    return x_best
