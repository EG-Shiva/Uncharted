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
    Y = Y0.copy()
    n, d = Y.shape
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
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Y -= lr * (M / (1.0 - b1t)) / (np.sqrt(V / (1.0 - b2t)) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf; best_Y = Y.copy()
    return best_Y, best_val

def _contact_polish(Y0, steps, step0, tol):
    Y = Y0.copy()
    Gc = Y @ Y.T
    np.fill_diagonal(Gc, -2.0)
    best_val = float(Gc.max())
    best_Y = Y.copy()
    step = float(step0)
    n = Y.shape[0]
    I = np.eye(n, dtype=np.float64)
    for _ in range(steps):
        Gfull = Y @ Y.T
        m = float(np.max(Gfull - 2.0 * I))
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
                best_val = mt; best_Y = Y.copy()
            step = min(step * 1.08, 0.02)
        else:
            step *= 0.5
            if step < 1e-12:
                step = 1e-12
    return best_Y, best_val

def _hole_vector(Xfix, rng, n_samp=4000, steps=400):
    # Xfix: (m,d) unit rows already fixed
    d = Xfix.shape[1]
    # sample candidates
    C = rng.standard_normal((n_samp, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    S = C @ Xfix.T
    mx = S.max(axis=1)
    bi = int(np.argmin(mx))
    y = C[bi:bi+1].copy()  # (1,d)
    # projected subgradient on single vector with annealed softmax
    vel = np.zeros_like(y)
    mu = 0.85
    best = float(mx[bi])
    best_y = y.copy()
    for s in range(steps):
        frac = s / max(1, steps-1)
        beta = 30.0 * (400.0/30.0)**frac
        lr = 0.05 * (0.003/0.05)**frac
        g = (y @ Xfix.T).ravel()  # (m,)
        gm = float(g.max())
        w = np.exp(beta * (g - gm))
        w /= (w.sum() + 1e-30)
        grad = (w @ Xfix).reshape(1, -1)  # pull toward fixed points
        # descent: move opposite to grad projected
        gt = grad - (float(grad @ y.T)) * y
        vel = mu * vel + (1.0 - mu) * gt
        y = y - lr * vel
        y /= (np.linalg.norm(y) + 1e-12)
        if (s % 50 == 49) or s == steps-1:
            cur = float((y @ Xfix.T).max())
            if cur < best:
                best = cur; best_y = y.copy()
    return best_y.ravel(), best

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0
    def left(need=10.0):
        return (time.time() < deadline - need)

    X0 = None; best_val0 = 1e9
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
                best_val0 = v; X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best_val0 = _max_cos(X0)

    x_best = X0.copy(); best_val = best_val0
    # secure baseline polish (same basin)
    y, v = _adam_lse(x_best, steps=4000, beta0=50.0, beta1=400.0, lr0=0.008, lr1=0.001)
    if v < best_val:
        best_val = v; x_best = y.copy()
    else:
        y = x_best.copy()
    y2, v2 = _contact_polish(y, steps=1200, step0=0.005, tol=0.006)
    if v2 < best_val:
        best_val = v2; x_best = y2.copy()

    elites = [x_best.copy()]
    elite_vals = [best_val]

    trial = 0
    # main topology-changing loop
    while left(20.0) and trial < 80:
        trial += 1
        # choose base among elites (round robin, bias to best)
        if len(elites) == 1 or (trial % 4 != 0):
            base = x_best
        else:
            base = elites[trial % len(elites)]
        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # contact degree with margin
        deg = np.sum(G > (m - 0.005), axis=1)
        order = np.argsort(-deg)
        mode = trial % 6
        Xt = base.copy()
        if mode == 0:
            # overloaded subset shake
            k = int(rng.integers(4, 9))
            idx = order[:max(12, k*3)]
            sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
            Xt[sel] += 0.65 * rng.standard_normal((len(sel), 22))
            mask = np.ones(59, bool); mask[sel] = False
            Xt[mask] += 0.008 * rng.standard_normal((int(mask.sum()), 22))
        elif mode == 1:
            # kill 2 worst, reinsert holes
            rowmax = G.max(axis=1)
            worst = np.argsort(-rowmax)[:2]
            keep = [i for i in range(59) if i not in set(worst.tolist())]
            Xfix = Xt[keep]
            for w in worst:
                hv, _ = _hole_vector(Xfix, rng, n_samp=3000, steps=300)
                Xt[w] = hv
                Xfix = Xt[keep + [w]] if False else np.vstack([Xfix, hv.reshape(1,-1)])
            # small jitter on rest
            Xt += 0.008 * rng.standard_normal((59,22))
            # restore fixed? no, keep holes; re-jitter holes slightly already optimal
        elif mode == 2:
            # global moderate shake
            sig = 0.18 if (trial % 2 == 0) else 0.28
            Xt = base + sig * rng.standard_normal((59,22))
        elif mode == 3:
            # random subset medium shake
            k = int(rng.integers(2, 6))
            sel = rng.choice(59, size=k, replace=False)
            Xt[sel] += 0.5 * rng.standard_normal((len(sel), 22))
            mask = np.ones(59, bool); mask[sel] = False
            Xt[mask] += 0.01 * rng.standard_normal((int(mask.sum()), 22))
        elif mode == 4:
            # kill 1 worst
            rowmax = G.max(axis=1)
            w = int(np.argmax(rowmax))
            keep_mask = np.ones(59, bool); keep_mask[w] = False
            Xfix = Xt[keep_mask]
            hv, _ = _hole_vector(Xfix, rng, n_samp=4000, steps=350)
            Xt[w] = hv
            Xt += 0.006 * rng.standard_normal((59,22))
            # keep hole close: reduce jitter on hole
            Xt[w] -= 0.006 * rng.standard_normal(22) * 0.0
        else:
            # strong global but keep some structure: orthogonal mix
            sig = 0.35
            Xt = base + sig * rng.standard_normal((59,22))
        Xt = _normalize(Xt)
        # fast formation: short anneal
        y, v = _adam_lse(Xt, steps=1800, beta0=20.0, beta1=300.0, lr0=0.015, lr1=0.002)
        # gate: only polish if promising (allow slack for new basin)
        thresh = best_val + (0.025 if mode in (0,1,3,4) else 0.015)
        if v < thresh and left(18.0):
            y2, v2 = _adam_lse(y, steps=1800, beta0=120.0, beta1=700.0, lr0=0.005, lr1=0.001)
            if v2 < v:
                v = v2; y = y2
            if v < best_val + 0.004:
                y3, v3 = _contact_polish(y, steps=800, step0=0.004, tol=0.005)
                if v3 < v:
                    v = v3; y = y3
        if v < best_val - 1e-12:
            best_val = v; x_best = y.copy()
            elites.append(y.copy()); elite_vals.append(v)
            if len(elites) > 4:
                # drop worst elite
                wi = int(np.argmax(elite_vals))
                elites.pop(wi); elite_vals.pop(wi)
        elif v < best_val + 0.006:
            # keep diverse second basin
            # check diversity: min Frobenius dist after optimal sign? cheap: dist
            div = min(float(np.linalg.norm(y - e)) for e in elites)
            if div > 1.0 and len(elites) < 4:
                elites.append(y.copy()); elite_vals.append(v)
        # occasional random restart injection to discover fresh basin
        if trial % 10 == 9 and left(25.0):
            R = _normalize(rng.standard_normal((59,22)))
            yr, vr = _adam_lse(R, steps=1800, beta0=20.0, beta1=300.0, lr0=0.015, lr1=0.002)
            if vr < best_val + 0.01:
                yr2, vr2 = _adam_lse(yr, steps=1500, beta0=120.0, beta1=600.0, lr0=0.005, lr1=0.001)
                vr, yr = vr2, yr2
                if vr < best_val - 1e-12:
                    best_val = vr; x_best = yr.copy()
                    elites.append(yr.copy()); elite_vals.append(vr)
                    if len(elites) > 4:
                        wi = int(np.argmax(elite_vals)); elites.pop(wi); elite_vals.pop(wi)

    # final ultra polish of best
    if left(5.0):
        y, v = _adam_lse(x_best, steps=5000, beta0=150.0, beta1=1100.0, lr0=0.004, lr1=0.0004)
        if v < best_val:
            best_val = v; x_best = y.copy()
    if left(4.0):
        y, v = _contact_polish(x_best, steps=1500, step0=0.003, tol=0.003)
        if v < best_val:
            best_val = v; x_best = y.copy()
    if left(3.0):
        y, v = _contact_polish(x_best, steps=800, step0=0.001, tol=0.0012)
        if v < best_val:
            best_val = v; x_best = y.copy()
    x_best = _normalize(x_best)
    return x_best
