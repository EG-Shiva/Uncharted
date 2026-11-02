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

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.8):
    n, d = x.shape
    v = np.zeros_like(x)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        D = G - m
        sel = D > -w
        W = np.zeros_like(G)
        W[sel] = np.exp(k * D[sel])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        step = lr * v / max(vn, 1.0)
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_vec(x, keep_mask, rng, n_try=2000):
    kept = x[keep_mask]
    C = rng.randn(n_try, x.shape[1])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[np.argmin(m)]

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
        tmp = rng.randn(n, d)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and float((xr @ xr.T - np.eye(n)*2).max()) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # quick squeeze of champion without LP
    try:
        xc = _anneal(x_best.copy(), steps=500, k0=300, k1=4000, lr0=0.003, lr1=0.0003, w0=0.015, w1=0.003, mu=0.8)
        vc = _max_cos(xc)
        if vc < best:
            best, x_best = vc, xc.copy()
    except Exception:
        pass

    it = 0
    while time.time() - t0 < BUDGET - 10.0:
        it += 1
        remaining = BUDGET - 10.0 - (time.time() - t0)
        if remaining < 2.5:
            break
        steps = 600
        if remaining < 25:
            steps = 350

        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        cnt = (G > (best - 0.015)).sum(axis=1)
        ppm = G.max(axis=1)
        order_lo = np.argsort(cnt)
        order_worst = np.argsort(-ppm)

        x = x_best.copy()
        mode = it % 8
        if mode == 0:
            x += rng.randn(n, d) * 0.010
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=80, k1=2500, lr0=0.012, lr1=0.0007, w0=0.05, w1=0.005, mu=0.82)
        elif mode == 1:
            x += rng.randn(n, d) * 0.025
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2000, lr0=0.015, lr1=0.0009, w0=0.06, w1=0.006, mu=0.85)
        elif mode == 2:
            ksel = 2 + (it % 3)
            idx = order_lo[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _void_vec(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=2800, lr0=0.013, lr1=0.0007, w0=0.055, w1=0.005, mu=0.82)
        elif mode == 3:
            ksel = 2
            idx = order_worst[:ksel]
            for j in idx:
                mask = np.ones(n, dtype=bool); mask[j] = False
                nv = _void_vec(x, mask, rng, n_try=1500)
                x[j] = nv + rng.randn(d) * 0.015
                x[j] /= np.linalg.norm(x[j])
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=70, k1=2800, lr0=0.013, lr1=0.0007, w0=0.055, w1=0.005, mu=0.82)
        elif mode == 4:
            # block kick: cluster by current geometry into ~6 groups, kick one group
            try:
                # cheap clustering: pick 6 seeds far apart greedily
                seeds = [int(rng.randint(0, n))]
                for _ in range(6):
                    dg = np.min(x[seeds] @ x.T, axis=0) if len(seeds) else np.zeros(n)
                    # farthest = smallest dot
                    cand = np.argsort(dg)
                    seeds.append(int(cand[0]))
                seeds = seeds[:6]
                S = x[seeds]  # 6xd
                lab = np.argmax(S @ x.T, axis=0)  # n labels
                b = int(rng.randint(0, 6))
                idx = np.where(lab == b)[0]
                if len(idx) == 0:
                    idx = rng.choice(n, size=9, replace=False)
                # random rotation in ambient space applied to block + noise
                A = rng.randn(d, d) * 0.06
                A = A - A.T
                R = np.eye(d) + A
                # orthonormalize approx via QR
                R, _ = np.linalg.qr(R)
                x[idx] = x[idx] @ R.T + rng.randn(len(idx), d) * 0.02
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            except Exception:
                x += rng.randn(n, d) * 0.015
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=50, k1=2200, lr0=0.014, lr1=0.0008, w0=0.06, w1=0.006, mu=0.85)
        elif mode == 5:
            j = int(order_lo[rng.randint(0, 8)])
            mask = np.ones(n, dtype=bool); mask[j] = False
            nv = _void_vec(x, mask, rng, n_try=3000)
            x[j] = nv
            x += rng.randn(n, d) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=90, k1=3000, lr0=0.011, lr1=0.0006, w0=0.05, w1=0.004, mu=0.8)
        elif mode == 6:
            # pairwise swap-hole: move two jammed points together
            idx = order_worst[:4]
            x[idx] += rng.randn(len(idx), d) * 0.06
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=60, k1=2400, lr0=0.014, lr1=0.0008, w0=0.06, w1=0.005, mu=0.85)
        else:
            x += rng.randn(n, d) * 0.006
            # random single-axis stretch to break symmetry
            ax = int(rng.randint(0, d))
            x[:, ax] *= (1.0 + 0.02 * rng.randn())
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, steps=steps, k0=100, k1=3200, lr0=0.009, lr1=0.0005, w0=0.04, w1=0.004, mu=0.8)

        v = _max_cos(x)
        if v < best - 1e-12:
            best = v
            x_best = x.copy()

    # final precision polish of champion with high-k low-lr passes
    try:
        left = BUDGET - (time.time() - t0)
        if left > 2.5:
            # two-stage fine polish
            x_best = _anneal(x_best.copy(), steps=min(800, int(400 + left*30)), k0=400, k1=6000, lr0=0.002, lr1=0.00015, w0=0.010, w1=0.0025, mu=0.75)
            v = _max_cos(x_best)
            if v < best:
                best = v
            else:
                # keep previous best value consistent
                pass
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
