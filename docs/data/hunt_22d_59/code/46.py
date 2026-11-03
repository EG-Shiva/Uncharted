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

def _adam_lse(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    lb = np.log(beta1 / beta0) / max(steps - 1, 1) if steps > 1 else 0.0
    ll = np.log(lr1 / lr0) / max(steps - 1, 1) if steps > 1 else 0.0
    lbb0 = np.log(beta0); llr0 = np.log(lr0)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        np.subtract(G, gmax, out=G)
        np.multiply(G, beta, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
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

def _contact_polish(Y0, steps, step0, tol):
    Y = Y0.copy()
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    step = float(step0)
    n = Y.shape[0]
    eye = np.eye(n, dtype=np.float64)
    for _ in range(steps):
        Gfull = Y @ Y.T
        m = float(np.max(Gfull - 2.0 * eye))
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
            if step < 1e-12:
                step = 1e-12
    return best_Y, best_val

def _hole_vector(Yothers, rng, n_samples=2500, refine=350):
    d = Yothers.shape[1]
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ Yothers.T
    scores = S.max(axis=1)
    idx = np.argsort(scores)[:4]
    best_x = C[idx[0]].copy()
    best_s = float(scores[idx[0]])
    for b in idx:
        x = C[b].copy()
        lr0 = 0.08
        for it in range(refine):
            dots = Yothers @ x
            j = int(np.argmax(dots))
            cur = float(dots[j])
            g = Yothers[j] - x * cur
            frac = it / max(1, refine - 1)
            lr = lr0 * (1.0 - 0.85 * frac)
            x = x - lr * g
            nx = float(np.linalg.norm(x))
            if nx == 0 or not np.isfinite(nx):
                x = C[b].copy()
                continue
            x /= nx
        final = float(np.max(Yothers @ x))
        if final < best_s:
            best_s = final
            best_x = x.copy()
    return best_x

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0
    X0 = None
    best0 = 1e9
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
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)
    x_best = X0.copy()
    best_val = best0
    # quick polish baseline
    y, v = _adam_lse(x_best, steps=3000, beta0=50.0, beta1=400.0, lr0=0.008, lr1=0.001)
    if v < best_val:
        best_val = v; x_best = y.copy()
    else:
        y = x_best.copy()
    y2, v2 = _contact_polish(y, steps=1000, step0=0.004, tol=0.006)
    if v2 < best_val:
        best_val = v2; x_best = y2.copy()

    sigmas_big = [0.10, 0.18, 0.28, 0.12, 0.22, 0.08, 0.35]
    sigmas_small = [0.008, 0.015, 0.03]
    block_sizes = [4, 6, 9, 12, 5, 8]
    trial = 0
    # main basin-hopping loop
    while True:
        if time.time() > deadline - 30.0:
            break
        trial += 1
        mode = trial % 5
        Xt = x_best.copy()
        if mode == 0 or mode == 3:
            # kill-and-fill worst 1-2 points
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                k = 1 if (trial % 2 == 1) else 2
                kills = order[:k]
                for kk in kills:
                    mask = np.ones(59, dtype=bool); mask[kk] = False
                    Yo = Xt[mask]
                    nx = _hole_vector(Yo, rng, n_samples=2500, refine=300)
                    Xt[kk] = nx
                # light noise on survivors to allow re-jam
                Xt += 0.008 * rng.standard_normal((59, 22))
                Xt = _normalize(Xt)
            except Exception:
                Xt = _normalize(x_best + 0.05 * rng.standard_normal((59, 22)))
        elif mode == 1 or mode == 4:
            # block shake of overloaded / random subset
            try:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowmax = G.max(axis=1)
                bs = block_sizes[(trial // 5) % len(block_sizes)]
                # mostly worst, sometimes random for diversity
                if trial % 2 == 0:
                    order = np.argsort(-rowmax)[:bs]
                    idx = order
                else:
                    idx = rng.choice(59, size=bs, replace=False)
                sigma = sigmas_big[(trial // 5) % len(sigmas_big)]
                noise = np.zeros((59, 22))
                noise[idx] = rng.standard_normal((bs, 22)) * sigma
                # tiny background
                noise += rng.standard_normal((59, 22)) * 0.005
                Xt = _normalize(x_best + noise)
            except Exception:
                Xt = _normalize(x_best + 0.1 * rng.standard_normal((59, 22)))
        else:
            # small global shake to polish within new basin
            sigma = sigmas_small[trial % len(sigmas_small)]
            Xt = _normalize(x_best + sigma * rng.standard_normal((59, 22)))

        # vary potential slightly for diversity
        bscale = float(10.0 ** rng.uniform(-0.25, 0.25))
        beta1 = 600.0 * bscale
        beta0 = 60.0
        if trial % 7 == 6:
            beta0 = 25.0; beta1 = 250.0 * bscale
        steps = 2000
        # shorten if time is low
        if time.time() > deadline - 45.0:
            steps = 1200
        y, v = _adam_lse(Xt, steps=steps, beta0=beta0, beta1=beta1, lr0=0.006, lr1=0.0009)
        y2, v2 = _contact_polish(y, steps=600, step0=0.004, tol=0.005)
        cand_v = v2 if v2 < v else v
        cand_y = y2 if v2 < v else y
        if cand_v < best_val:
            best_val = cand_v
            x_best = cand_y.copy()
        # opportunistic extra: if close (within 2e-4) keep exploring from cand
        # no update otherwise
        if trial > 120:
            break

    if time.time() < deadline - 12.0:
        y, v = _adam_lse(x_best, steps=7000, beta0=150.0, beta1=1100.0, lr0=0.004, lr1=0.0004)
        if v < best_val:
            best_val = v; x_best = y.copy()
        y, v = _contact_polish(x_best, steps=2000, step0=0.003, tol=0.004)
        if v < best_val:
            best_val = v; x_best = y.copy()
        y, v = _contact_polish(x_best, steps=1000, step0=0.001, tol=0.0015)
        if v < best_val:
            best_val = v; x_best = y.copy()
    x_best = _normalize(x_best)
    return x_best
