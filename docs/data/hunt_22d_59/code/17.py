import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_refine(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        return best_Y, best_val
    lbb0 = np.log(beta0)
    llr0 = np.log(lr0)
    lb = np.log(beta1 / beta0) / (steps - 1)
    ll = np.log(lr1 / lr0) / (steps - 1)
    b1t = 1.0
    b2t = 1.0
    for t in range(1, steps + 1):
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1
        b2t *= b2
        M *= b1
        M += (1.0 - b1) * Gr
        V *= b2
        V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _contact_refine(Y0, steps, step0, tol):
    Y = Y0.copy()
    n = Y.shape[0]
    Eye2 = 2.0 * np.eye(n, dtype=Y.dtype)
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    step = float(step0)
    for _ in range(steps):
        Gfull = Y @ Y.T
        m = float(np.max(Gfull - Eye2))
        A = Gfull > (m - tol)
        np.fill_diagonal(A, False)
        Af = A.astype(np.float64)
        AY = Af @ Y
        row_w = np.sum(Af * Gfull, axis=1, keepdims=True)
        F = -AY + Y * row_w
        Yt = Y + step * F
        rn = np.linalg.norm(Yt, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Yt /= rn
        Gt = Yt @ Yt.T
        np.fill_diagonal(Gt, -2.0)
        mt = float(Gt.max())
        if mt < m - 1e-12:
            Y = Yt
            if mt < best_val:
                best_val = mt
                best_Y = Y.copy()
            step = step * 1.08
            if step > 0.02:
                step = 0.02
        else:
            step *= 0.5
            if step < 1e-13:
                step = 1e-13
    return best_Y, best_val

def _hole_vector(Z, rng, n_cand=3000, beta=120.0, steps=350, lr=0.2):
    d = Z.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ Z.T
    scores = S.max(axis=1)
    idx = int(np.argmin(scores))
    x = C[idx].copy()
    best = x.copy()
    s0 = Z @ x
    best_val = float(s0.max())
    for it in range(steps):
        s = Z @ x
        m = float(s.max())
        w = np.exp(beta * (s - m))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ Z
        g -= float(np.dot(g, x)) * x
        # decay lr slightly
        cur = lr / (1.0 + it * 0.01)
        x = x - cur * g
        nx = float(np.linalg.norm(x))
        if nx == 0 or not np.finite(nx):
            x = best.copy()
            continue
        x /= nx
        s2 = Z @ x
        m2 = float(s2.max())
        if m2 < best_val:
            best_val = m2
            best = x.copy()
    return best

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0
    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best_val0:
                best_val0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best_val0 = _max_cos(X0)
    x_best = X0.copy()
    best_val = best_val0

    # Stage 0: direct polish
    y, v = _adam_refine(x_best, steps=4500, beta0=40.0, beta1=400.0, lr0=0.008, lr1=0.0008)
    if v < best_val:
        best_val = v
        x_best = y.copy()
    else:
        y = x_best.copy()
    y2, v2 = _contact_refine(y, steps=1200, step0=0.005, tol=0.006)
    if v2 < best_val:
        best_val = v2
        x_best = y2.copy()

    sigmas = [0.02, 0.06, 0.13, 0.28, 0.035, 0.09, 0.19, 0.015, 0.05, 0.11]
    trial = 0
    max_trials = 60
    while trial < max_trials and time.time() < deadline - 22.0:
        mode = trial % 5
        Xt = x_best.copy()
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowmax = G.max(axis=1)
            # contact counts with tol 2e-4 for ranking overload
            counts = (G > (m - 0.002)).sum(axis=1)
            order = np.argsort(-rowmax - 0.0005 * counts)
        except Exception:
            order = np.arange(59)
            rowmax = np.zeros(59)
        if mode == 3 or mode == 4:
            # kill and fill 1-2 worst points with optimized holes
            k = 1 if mode == 3 else 2
            worst = order[:k].tolist()
            keep = [i for i in range(59) if i not in worst]
            Z0 = x_best[keep]
            new_vecs = []
            Zcur = Z0.copy()
            for j in range(k):
                hv = _hole_vector(Zcur, rng, n_cand=3000, beta=120.0, steps=300, lr=0.25)
                new_vecs.append(hv)
                Zcur = np.vstack([Zcur, hv.reshape(1, -1)])
            for j, idx in enumerate(worst):
                Xt[idx] = new_vecs[j]
            # tiny noise on the rest to allow adjustment
            Xt += 0.004 * rng.standard_normal(Xt.shape)
            Xt = _normalize(Xt)
            b0 = 30.0 + (trial % 3) * 10.0
            b1 = 300.0 + (trial % 4) * 50.0
            y, v = _adam_refine(Xt, steps=2200, beta0=b0, beta1=b1, lr0=0.007, lr1=0.0007)
            y2, v2 = _contact_refine(y, steps=800, step0=0.004, tol=0.005)
            y, v = (y2, v2) if v2 < v else (y, v)
        elif mode == 2:
            # block shake of worst 6-10 points with large sigma
            k = 6 + (trial % 3) * 2
            worst = order[:k]
            noise = rng.standard_normal((59, 22))
            Xt[worst] += 0.30 * noise[worst]
            rest = np.ones(59, dtype=bool)
            rest[worst] = False
            Xt[rest] += 0.006 * noise[rest]
            Xt = _normalize(Xt)
            y, v = _adam_refine(Xt, steps=2200, beta0=40.0, beta1=450.0, lr0=0.008, lr1=0.0007)
            y2, v2 = _contact_refine(y, steps=800, step0=0.004, tol=0.005)
            y, v = (y2, v2) if v2 < v else (y, v)
        else:
            sigma = sigmas[trial % len(sigmas)]
            # bias noise toward overloaded points slightly
            noise = rng.standard_normal((59, 22))
            Xt = x_best + sigma * noise
            Xt = _normalize(Xt)
            b0 = 25.0 + (trial % 3) * 12.0
            b1 = 250.0 + (trial % 4) * 60.0
            y, v = _adam_refine(Xt, steps=2000, beta0=b0, beta1=b1, lr0=0.008, lr1=0.0008)
            y2, v2 = _contact_refine(y, steps=700, step0=0.004, tol=0.005)
            y, v = (y2, v2) if v2 < v else (y, v)
        if v < best_val:
            best_val = v
            x_best = y.copy()
        trial += 1

    # final ultra-fine polish
    if time.time() < deadline - 8.0:
        try:
            y, v = _adam_refine(x_best, steps=1500, beta0=150.0, beta1=600.0, lr0=0.003, lr1=0.0002)
            if v < best_val:
                best_val = v
                x_best = y.copy()
            y2, v2 = _contact_refine(x_best, steps=1200, step0=0.0015, tol=0.002)
            if v2 < best_val:
                best_val = v2
                x_best = y2.copy()
        except Exception:
            pass
    x_best = _normalize(x_best)
    return x_best
