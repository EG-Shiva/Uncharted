import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam(X0, steps, b0, b1, lr0, lr1, deadline):
    Y = X0.copy()
    n, d = Y.shape
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1d = 0.9; b2d = 0.999; eps = 1e-8
    b1t = 1.0; b2t = 1.0
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    bestv = float(np.max(G0))
    best = Y.copy()
    if steps < 2:
        return best, bestv
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        if (t & 255) == 0 and time.time() > deadline:
            break
        f = (t - 1) / (steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(np.max(G))
        if gmax < bestv:
            bestv = gmax
            best = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(np.sum(G))
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1d; b2t *= b2d
        M *= b1d; M += 0.1 * Gr
        V *= b2d; V += 0.001 * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(np.max(Gf))
    if vf < bestv:
        bestv = vf
        best = Y.copy()
    return best, bestv

def _hole_refine(y0, Xr, steps, beta, lr0, lr1):
    y = y0.copy() / (np.linalg.norm(y0) + 1e-30)
    best = y.copy()
    d0 = Xr @ y
    bestv = float(np.max(d0))
    for t in range(steps):
        f = t / max(steps - 1, 1)
        lr = lr0 * (1 - f) + lr1 * f
        dv = Xr @ y
        m = float(np.max(dv))
        w = np.exp(beta * (dv - m))
        sw = float(np.sum(w))
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ Xr
        g -= y * float(g @ y)
        y = y - lr * g
        ny = float(np.linalg.norm(y))
        if ny == 0 or not np.isfinite(ny):
            y = best.copy()
            continue
        y /= ny
        mv = float(np.max(Xr @ y))
        if mv < bestv:
            bestv = mv
            best = y.copy()
    return best

def solve():
    t0 = time.time()
    BUDGET = 165.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 59, 22
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n, d))))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    bestv = float(np.min(vals))

    # initial polish
    if time.time() < deadline - 5:
        best, bestv = _adam(best, steps=8000, b0=50.0, b1=700.0,
                            lr0=0.006, lr1=0.0004, deadline=deadline)

    it = 0
    while time.time() < deadline - 8:
        it += 1
        remain = deadline - time.time()
        if remain < 5:
            break
        if remain > 60:
            steps = 4000
        elif remain > 25:
            steps = 3000
        else:
            steps = 1800
        mode = it % 3
        Y = best.copy()
        if mode == 0:
            # kill-and-fill: remove worst k points, reinsert in holes
            if it % 20 == 19:
                k = 3
            elif it % 9 == 8:
                k = 2
            else:
                k = 1
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            contacts = np.sum(G > (bestv - 0.004), axis=1)
            order = np.argsort(-rowmax - 0.0005 * contacts + rng.standard_normal(n) * 1e-6)
            victims = order[:k]
            keep = np.ones(n, dtype=bool)
            keep[victims] = False
            Xr = Y[keep]
            # find holes sequentially
            new_pts = []
            cur = Xr
            for j in range(k):
                m_samp = 6000 if remain > 30 else 2500
                C = rng.standard_normal((m_samp, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                S = C @ cur.T
                smax = np.max(S, axis=1)
                bi = int(np.argmin(smax))
                y0 = C[bi]
                y0 = _hole_refine(y0, cur, steps=250, beta=200.0, lr0=0.3, lr1=0.02)
                # small random orthogonal jitter to separate multiple inserts
                if k > 1:
                    y0 = y0 + 0.02 * rng.standard_normal(d)
                    y0 /= np.linalg.norm(y0)
                    y0 = _hole_refine(y0, cur, steps=120, beta=200.0, lr0=0.15, lr1=0.02)
                new_pts.append(y0)
                cur = np.vstack([cur, y0.reshape(1, -1)])
            for idx, vv in enumerate(victims):
                Y[vv] = new_pts[idx]
            Y = _norm(Y)
            Yp, Yv = _adam(Y, steps=steps, b0=60.0, b1=800.0, lr0=0.005, lr1=0.0004, deadline=deadline)
        elif mode == 1:
            # subset shake of worst + random points
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            worst = np.argsort(-rowmax)[:6]
            s = 6 if (it % 6 != 5) else 10
            extra = rng.choice(n, size=max(0, s - 6), replace=False)
            idx = np.unique(np.concatenate([worst[:min(6, s)], extra]))
            sig = 0.18 if (it % 2 == 0) else 0.35
            Y[idx] += sig * rng.standard_normal((len(idx), d))
            # occasional single big jump
            if it % 11 == 0:
                kk = int(rng.integers(n))
                Y[kk] = rng.standard_normal(d)
            Y = _norm(Y)
            Yp, Yv = _adam(Y, steps=steps, b0=80.0, b1=900.0, lr0=0.004, lr1=0.0003, deadline=deadline)
        else:
            # global small shake biased to high contacts
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            cm = float(np.max(G))
            contacts = np.sum(G > (cm - 0.004), axis=1)
            w = (contacts - contacts.min() + 1.0)
            w = w / w.max()
            sig0 = 0.012 if (it % 2 == 0) else 0.03
            noise = rng.standard_normal((n, d))
            Y = Y + (sig0 * (0.5 + w.reshape(-1, 1))) * noise
            Y = _norm(Y)
            Yp, Yv = _adam(Y, steps=steps, b0=150.0, b1=1200.0, lr0=0.003, lr1=0.00025, deadline=deadline)
        if Yv < bestv - 1e-12:
            best = Yp
            bestv = Yv

    if time.time() < deadline - 2:
        best, bestv = _adam(best, steps=2500, b0=800.0, b1=2500.0,
                            lr0=0.0015, lr1=0.00015, deadline=deadline)
    best = _norm(best)
    return best
