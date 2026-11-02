import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.80, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        W = np.zeros_like(G)
        sel = D > -w
        # stable exp: k*D in [-k*w,0]
        W[sel] = np.exp(k * D[sel])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        x = x - (lr * v / vn)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_vec(x_keep, d, rng, n_try=2000):
    C = rng.randn(n_try, d)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ x_keep.T
    return C[np.argmin(dots.max(axis=1))]

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and float((xr @ xr.T - np.eye(n)).max()) < 10:
            import numpy as _np
            g1 = xr @ xr.T; _np.fill_diagonal(g1, -2.0)
            g2 = xb @ xb.T; _np.fill_diagonal(g2, -2.0)
            if float(g1.max()) < float(g2.max()):
                xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # quick squeeze of starting point
    try:
        xc = _anneal(x_best.copy(), steps=500, k0=300.0, k1=4000.0,
                     lr0=0.004, lr1=0.0004, w0=0.02, w1=0.004, mu=0.85, rng=rng)
        vc = _max_cos(xc)
        if vc < best:
            best, x_best = vc, xc
    except Exception:
        pass

    it = 0
    while time.time() - t0 < BUDGET - 10.0:
        it += 1
        rem = BUDGET - 10.0 - (time.time() - t0)
        if rem < 2.5:
            break
        steps = 500
        if rem < 25:
            steps = 320
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        cnt = (G > best - 0.015).sum(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)
        mode = it % 7
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps, 80, 2500, 0.012, 0.0008, 0.05, 0.005, 0.82, rng)
        elif mode == 1:
            x += rng.randn(n, d) * 0.025
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps, 60, 2000, 0.015, 0.001, 0.06, 0.006, 0.85, rng)
        elif mode == 2:
            ksel = 2 + (it % 3)  # 2-4 worst -> voids
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, bool); mask[j] = False
                nv = _void_vec(x[mask], d, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps, 70, 2500, 0.014, 0.0008, 0.055, 0.005, 0.84, rng)
        elif mode == 3:
            ksel = 3 + (it % 3)  # 3-5 loose -> voids
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, bool); mask[j] = False
                nv = _void_vec(x[mask], d, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.02
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps, 70, 2500, 0.014, 0.0008, 0.055, 0.005, 0.84, rng)
        elif mode == 4:
            # coherent cap shift: pick random axis, move cap tangentially
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = x_best @ u
            kcap = 6 + (it % 7)  # 6-12
            idx = np.argsort(-proj)[:kcap]
            # random orthogonal direction
            w = rng.randn(d); w -= (w @ u) * u
            w /= np.linalg.norm(w) + 1e-12
            amp = 0.05 + 0.03 * rng.rand()
            x[idx] += amp * w
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps, 60, 2200, 0.014, 0.0009, 0.06, 0.005, 0.84, rng)
        elif mode == 5:
            # orthogonalize a random pair-cluster: respin 6-9 random points
            ksel = 6 + (it % 4)
            idx = rng.choice(n, size=ksel, replace=False)
            for j in idx:
                mask = np.ones(n, bool); mask[j] = False
                nv = _void_vec(x[mask], d, rng, n_try=800)
                x[j] = nv
            x += rng.randn(n, d) * 0.006
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, steps + 150, 50, 1800, 0.016, 0.001, 0.07, 0.006, 0.85, rng)
        else:
            # micro squeeze around champion for 1e-7 gains
            x += rng.randn(n, d) * 0.0035
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            xa = _anneal(x, 350, 250, 5000, 0.005, 0.0003, 0.025, 0.004, 0.80, rng)
        try:
            va = _max_cos(xa)
        except Exception:
            continue
        if va < best - 1e-12:
            best = va
            x_best = xa.copy()
        # opportunistic micro-polish of new champion
        if va < best + 1e-9 and (time.time() - t0) < BUDGET - 12.0:
            try:
                xb2 = _anneal(x_best.copy(), 250, 400, 6000, 0.003, 0.0002, 0.015, 0.003, 0.75, rng)
                vb2 = _max_cos(xb2)
                if vb2 < best - 1e-12:
                    best, x_best = vb2, xb2
            except Exception:
                pass

    # final squeeze
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2.0:
            st = int(min(900, max(300, left * 60)))
            xf = _anneal(x_best.copy(), st, 400, 7000, 0.003, 0.0002, 0.015, 0.003, 0.75, rng)
            if _max_cos(xf) < best:
                best, x_best = _max_cos(xf), xf
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
