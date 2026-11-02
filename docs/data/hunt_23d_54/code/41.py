import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _anneal(x, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
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
        sel = D > -w
        W = np.zeros_like(G)
        # D<=0 so stable
        W[sel] = np.exp(k * D[sel])
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        if an == 0:
            continue
        v = mu * v + Ft / an
        vn = float(np.sqrt(np.sum(v * v) / n)) + 1e-12
        x = x - (lr / vn) * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _refine_void(Y, p0, steps=250, k0=50.0, k1=3000.0, lr0=0.05, lr1=0.002):
    p = p0.copy() / np.linalg.norm(p0)
    lk0 = np.log(k0); lk1 = np.log(k1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        g = Y @ p
        m = float(np.max(g))
        w = np.exp(k * (g - m))
        F = w @ Y
        dot = float(np.dot(F, p))
        Ft = F - dot * p
        nrm = float(np.linalg.norm(Ft)) + 1e-12
        p = p - (lr / nrm) * Ft
        nrm2 = float(np.linalg.norm(p))
        if nrm2 == 0:
            p = p0 / np.linalg.norm(p0)
        else:
            p /= nrm2
    return p

def _best_void(x, keep_mask, rng, n_try=1500):
    kept = x[keep_mask]
    C = rng.randn(n_try, x.shape[1])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ kept.T
    m = dots.max(axis=1)
    i0 = int(np.argmin(m))
    p0 = C[i0]
    try:
        p = _refine_void(kept, p0)
    except Exception:
        p = p0
    # check refined vs raw
    return p

def _partition_min_intra(x, rng, iters=2500):
    n = x.shape[0]
    nb = 6
    G = x @ x.T
    np.fill_diagonal(G, 0.0)
    lab = np.repeat(np.arange(nb), 9)
    rng.shuffle(lab)
    # block member lists via counts, compute cost
    # cost = sum intra G
    def block_sums():
        return None
    for _ in range(iters):
        a = int(rng.randint(n)); b = int(rng.randint(n))
        la = int(lab[a]); lb = int(lab[b])
        if la == lb:
            continue
        # sums to blocks
        # indices in each block
        ia = np.where(lab == la)[0]
        ib = np.where(lab == lb)[0]
        # sum excluding self/other (other not in same block anyway)
        sa_a = float(np.sum(G[a, ia]))  # G[a,a]=0 so fine
        sb_b = float(np.sum(G[b, ib]))
        sa_b = float(np.sum(G[a, ib])) - float(G[a, b])
        sb_a = float(np.sum(G[b, ia])) - float(G[b, a])
        delta = (sa_b + sb_a) - (sa_a + sb_b)
        if delta < 0:
            lab[a], lab[b] = lb, la
        elif rng.rand() < 0.002:
            lab[a], lab[b] = lb, la
    return lab

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(0)
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
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()

    # initial quick polish of loaded point (mild, fast)
    try:
        xc = _anneal(x_best.copy(), steps=300, k0=300, k1=4000, lr0=0.003, lr1=0.0004, w0=0.012, w1=0.003, mu=0.88)
        vc = _max_cos(xc)
        if vc < best:
            best, x_best = vc, xc
    except Exception:
        pass

    try:
        labels = _partition_min_intra(x_best, rng, iters=2500)
    except Exception:
        labels = np.repeat(np.arange(6), 9)

    it = 0
    # main hopping, reserve ~10s for final
    while time.time() - t0 < BUDGET - 10.0:
        it += 1
        rem = BUDGET - 10.0 - (time.time() - t0)
        if rem < 2.0:
            break
        # refresh partition occasionally or on improvement
        if it % 25 == 0:
            try:
                labels = _partition_min_intra(x_best, rng, iters=1200)
            except Exception:
                pass
        # contact stats of best
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            cnt = np.sum(G > (best - 0.015), axis=1)
            ppm = np.max(G, axis=1)
            order_lo = np.argsort(cnt)
            order_worst = np.argsort(-ppm)
        except Exception:
            order_lo = np.argsort(rng.rand(n))
            order_worst = order_lo[::-1]

        mode = it % 7
        x = x_best.copy()
        steps = 450
        if rem < 25:
            steps = 300
        # defaults
        k0, k1, lr0, lr1, w0, w1, mu = 70, 2500, 0.014, 0.0008, 0.055, 0.005, 0.85
        try:
            if mode == 0:
                x += rng.randn(n, d) * 0.006
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 120, 3500, 0.008, 0.0005, 0.03, 0.004
            elif mode == 1:
                x += rng.randn(n, d) * 0.016
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 60, 2200, 0.015, 0.0009, 0.06, 0.006
            elif mode == 2:
                # block-coherent kick of one block
                b = int(rng.randint(6))
                idx = np.where(labels == b)[0]
                if len(idx) == 0:
                    idx = np.where(order_lo[:9])[0]
                common = rng.randn(d) * 0.045
                x[idx] += common + rng.randn(len(idx), d) * 0.006
                x += rng.randn(n, d) * 0.003
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 70, 2800, 0.013, 0.0007, 0.05, 0.005
            elif mode == 3:
                ksel = 2
                idx = order_worst[:ksel]
                for j in idx:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    nv = _best_void(x, mask, rng, n_try=1200)
                    x[j] = nv + rng.randn(d) * 0.015
                    x[j] /= np.linalg.norm(x[j])
                x += rng.randn(n, d) * 0.003
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 80, 2600, 0.013, 0.0007, 0.05, 0.005
            elif mode == 4:
                ksel = 2
                idx = order_lo[:ksel]
                for j in idx:
                    mask = np.ones(n, dtype=bool); mask[j] = False
                    nv = _best_void(x, mask, rng, n_try=1200)
                    x[j] = nv + rng.randn(d) * 0.015
                    x[j] /= np.linalg.norm(x[j])
                # also jitter worst slightly
                x[order_worst[0]] += rng.randn(d) * 0.01
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 70, 2600, 0.013, 0.0007, 0.05, 0.005
            elif mode == 5:
                # swap two points between blocks + jitter
                b1 = int(rng.randint(6)); b2 = int(rng.randint(6))
                if b1 != b2:
                    i1 = np.where(labels == b1)[0]
                    i2 = np.where(labels == b2)[0]
                    if len(i1) > 0 and len(i2) > 0:
                        a = int(rng.choice(i1)); b_ = int(rng.choice(i2))
                        tmpv = x[a].copy(); x[a] = x[b_]; x[b_] = tmpv
                x += rng.randn(n, d) * 0.008
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 90, 3000, 0.011, 0.0006, 0.04, 0.0045
            else:
                x += rng.randn(n, d) * 0.032
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                k0, k1, lr0, lr1, w0, w1 = 50, 1800, 0.016, 0.001, 0.07, 0.007
            x = _anneal(x, steps=steps, k0=k0, k1=k1, lr0=lr0, lr1=lr1, w0=w0, w1=w1, mu=mu)
            v = _max_cos(x)
            if v < best:
                best = v
                x_best = x.copy()
                try:
                    labels = _partition_min_intra(x_best, rng, iters=1200)
                except Exception:
                    pass
        except Exception:
            continue

    # final polish of best with narrow high-k anneal if time left
    try:
        rem = BUDGET - (time.time() - t0)
        if rem > 3.0:
            steps = int(min(1200, max(300, rem * 35)))
            xp = _anneal(x_best.copy(), steps=steps, k0=400, k1=6000, lr0=0.0025, lr1=0.00025, w0=0.010, w1=0.0025, mu=0.90)
            vp = _max_cos(xp)
            if vp <= best:
                best, x_best = vp, xp
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
