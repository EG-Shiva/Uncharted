import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.max(g, axis=1)

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    # precompute log ratios for speed
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        # weights for near-max pairs
        W = np.zeros_like(G)
        # mask excludes diagonal automatically (D~-2)
        # manual loop with boolean index
        sel = D > -w
        # exp in stable range: D in [-w,0], k*D in [-k*w,0]
        Ds = D[sel]
        W[sel] = np.exp(k * Ds)
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        # momentum on normalized direction
        v = mu * v + Ft / an
        # normalize velocity RMS to ~1 to make lr geometric
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        # limit huge momentum build-up
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _best_void_vector(x, keep_mask, rng, n_try=2000):
    # find unit vector minimizing max dot to kept points
    kept = x[keep_mask]
    # random candidates
    C = rng.randn(n_try, x.shape[1])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[np.argmin(m)]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(1)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # quick initial tight polish of loaded best (short)
    # keep best unchanged if not better
    try:
        xc = _anneal(x_best.copy(), steps=400, k0=200, k1=3000, lr0=0.004, lr1=0.0004, w0=0.02, w1=0.004)
        vc = _max_cos(xc)
        if vc < best:
            best, x_best = vc, xc.copy()
    except Exception:
        pass

    it = 0
    # main hopping loop, reserve ~8s for final polish
    while time.time() - t0 < BUDGET - 8.0:
        it += 1
        remaining = BUDGET - 8.0 - (time.time() - t0)
        if remaining < 2.0:
            break
        # adapt steps to remaining time: ~600 default
        steps = 550
        if remaining < 20:
            steps = 350
        mode = it % 8

        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        # ranking windows
        wide = G > (best - 0.015)
        cnt = wide.sum(axis=1)
        ppm = np.max(G, axis=1)
        order_lo = np.argsort(cnt)       # few near-contacts first
        order_worst = np.argsort(-ppm)   # largest per-point max first

        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.009
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=80, k1=2500, lr0=0.012, lr1=0.0008, w0=0.05, w1=0.005)
        elif mode == 1:
            x += rng.randn(n, d) * 0.022
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2000, lr0=0.015, lr1=0.001, w0=0.06, w1=0.006)
        elif mode == 2:
            ksel = 3 + (it % 3)  # 3-5
            idx = order_lo[:ksel]
            # void reinsertion: replace with best of random voids
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                # temporarily remove j for void search using current x
                nv = _best_void_vector(x, mask, rng, n_try=1500)
                # mix with small noise around void for diversity
                x[j] = nv + rng.randn(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=2500, lr0=0.014, lr1=0.0008, w0=0.055, w1=0.005)
        elif mode == 3:
            ksel = 2 + (it % 2)
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _best_void_vector(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=3000, lr0=0.013, lr1=0.0007, w0=0.05, w1=0.004)
        elif mode == 4:
            # jammed cluster shake: shake high-contact points collectively
            order_hi = np.argsort(-cnt)
            idx = order_hi[:6]
            x[idx] += rng.randn(len(idx), d) * 0.05
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2200, lr0=0.014, lr1=0.0009, w0=0.06, w1=0.006)
        elif mode == 5:
            x += rng.randn(n, d) * 0.045
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps+150, k0=50, k1=1800, lr0=0.016, lr1=0.001, w0=0.07, w1=0.007)
        elif mode == 6:
            # single-point hole dig + small global noise
            j = int(order_lo[0])
            mask = np.ones(n, dtype=bool); mask[j] = False
            nv = _best_void_vector(x, mask, rng, n_try=2500)
            x[j] = nv
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=90, k1=2800, lr0=0.011, lr1=0.0006, w0=0.045, w1=0.004)
        else:
            # random restart partially mixed with best
            alpha = 0.35
            R = rng.randn(n, d)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            x = (1 - alpha) * x_best + alpha * R
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps+150, k0=50, k1=2000, lr0=0.015, lr1=0.001, w0=0.065, w1=0.006)

        v = _max_cos(x)
        if v < best:
            # extra tight re-polish immediately
            try:
                x2 = _anneal(x.copy(), steps=400, k0=300, k1=5000, lr0=0.003, lr1=0.0002, w0=0.015, w1=0.003)
                v2 = _max_cos(x2)
                if v2 < v:
                    x, v = x2, v2
            except Exception:
                pass
            best, x_best = v, x.copy()
            # early exit if clearly record-beating by margin? keep going to improve further
        # timeout guard
        if time.time() - t0 > BUDGET - 8.0:
            break

    # final tight polish of champion with remaining time (max ~7s)
    try:
        t_rem = BUDGET - (time.time() - t0)
        if t_rem > 1.5:
            # two-stage final: medium then very tight
            x_best = _anneal(x_best.copy(), steps=600, k0=200, k1=4000, lr0=0.003, lr1=0.00025, w0=0.018, w1=0.0035)
            best = _max_cos(x_best)
            t_rem2 = BUDGET - (time.time() - t0)
            if t_rem2 > 1.0:
                x_best = _anneal(x_best.copy(), steps=400, k0=500, k1=8000, lr0=0.0012, lr1=0.0001, w0=0.010, w1=0.002)
                best = _max_cos(x_best)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
