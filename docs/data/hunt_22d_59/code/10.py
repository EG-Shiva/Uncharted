import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _adam_lse(X0, steps, beta0, beta1, lr0, lr1, t_end):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1
    pow2 = b2
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    den = max(steps - 1, 1)
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_end:
            break
        frac = (t - 1) / den
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1)
        vhat = v / (1.0 - pow2)
        pow1 *= b1
        pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if (t % 25) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur
                bestX = X.copy()
    return bestX, best

def _greedy_polish(X0, iters, step0, step1, t_end, rng=None):
    Xcur = X0.copy()
    n, d = Xcur.shape
    best = _max_cos(Xcur)
    bestX = Xcur.copy()
    decay = (step1 / step0) ** (1.0 / max(iters - 1, 1))
    step = step0
    for t in range(iters):
        if (t & 63) == 0 and time.time() > t_end:
            break
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        delta = max(0.004, 1.5 * step)
        thr = mm - delta
        Mask = G > thr
        W = np.where(Mask, (G - thr + 0.01), 0.0)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Xcur
        dot = (Gr * Xcur).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Xcur
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        has = (nrm[:, 0] > 1e-9)
        upd = np.zeros_like(Xcur)
        upd[has] = Gt[has] / nrm[has]
        if rng is not None and np.any(~has):
            nhas = ~has
            R = rng.standard_normal(size=(int(nhas.sum()), d))
            Xr = Xcur[nhas]
            R = R - (R * Xr).sum(axis=1, keepdims=True) * Xr
            rn2 = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            upd[nhas] = (R / rn2) * 0.3
        Xcur = Xcur - step * upd
        Xcur = _normalize(Xcur)
        cur = _max_cos(Xcur)
        if cur < best:
            best = cur
            bestX = Xcur.copy()
        step *= decay
    return bestX, best

def _refine_single_point(p, Fixed, steps=200, s0=0.05, s1=0.002):
    # minimise max_j <p, Fixed_j> over unit p
    p = p / max(np.linalg.norm(p), 1e-12)
    dec = (s1 / s0) ** (1.0 / max(steps - 1, 1))
    s = s0
    for _ in range(steps):
        dots = Fixed @ p
        j = int(np.argmax(dots))
        g = Fixed[j] - dots[j] * p
        gn = np.linalg.norm(g) + 1e-12
        if gn < 1e-12:
            break
        g = g / gn
        p = p - s * g
        p = p / max(np.linalg.norm(p), 1e-12)
        s *= dec
    return p / max(np.linalg.norm(p), 1e-12)

def _ruin_recreate(Ybest, victims, rng, n_samples=16000):
    n, d = Ybest.shape
    victims = np.array(victims, dtype=int)
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    Fixed0 = Ybest[keep]
    # sample candidates
    S = n_samples
    # batch sampling to limit memory but still fast
    C = rng.standard_normal(size=(S, d))
    C = _normalize(C)
    placed = []
    Ynew = Ybest.copy()
    for idx, vi in enumerate(victims):
        if len(placed) == 0:
            Base = Fixed0
        else:
            Base = np.vstack([Fixed0] + placed)
        # scores of all C vs Base
        # chunk to avoid huge mat
        best_score = 1e9
        best_c = None
        chunk = 4000
        for a in range(0, S, chunk):
            Cb = C[a:a+chunk]
            sc = (Cb @ Base.T).max(axis=1)
            k = int(np.argmin(sc))
            if float(sc[k]) < best_score:
                best_score = float(sc[k])
                best_c = Cb[k].copy()
        # local single-point refine against Base
        best_c = _refine_single_point(best_c, Base, steps=250, s0=0.06, s1=0.002)
        placed.append(best_c.reshape(1, d))
        Ynew[int(vi)] = best_c
    # small orthogonal jitter to break exact symmetry when k>1
    if len(victims) > 1:
        jit = rng.standard_normal(size=(len(victims), d)) * 0.005
        for ii, vi in enumerate(victims):
            p = Ynew[int(vi)] + jit[ii]
            # project tangent
            pn = p / max(np.linalg.norm(p), 1e-12)
            Ynew[int(vi)] = pn
    return _normalize(Ynew)

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = _np.random.default_rng(0)
    n, d = 59, 22

    def rand_X():
        Y = rng.standard_normal(size=(n, d))
        return _normalize(Y)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            if a.shape == (n, d):
                a = _np.array(a, dtype=float)
                if not _np.all(_np.isfinite(a)):
                    continue
                a = _normalize(a)
                if _np.all(_np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(rand_X())
    vals = [_max_cos(c) for c in cands]
    bi = int(_np.argmin(vals))
    gbestX = cands[bi].copy()
    gbest = float(vals[bi])

    # Stage 0: deep polish of incoming best
    try:
        X1, v1 = _adam_lse(gbestX, 2200, 40.0, 260.0, 0.015, 0.006, t_end - 1.0)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
        else:
            X1 = gbestX
        X2, v2 = _adam_lse(X1, 1200, 200.0, 420.0, 0.007, 0.002, t_end - 1.0)
        if v2 < gbest:
            gbest, gbestX = v2, X2.copy()
            X1 = X2.copy()
        X3, v3 = _greedy_polish(X1, 1000, 0.006, 0.0002, t_end - 1.0, rng)
        if v3 < gbest:
            gbest, gbestX = v3, X3.copy()
    except Exception:
        pass

    it = 0
    # main ruin-and-recreate loop
    while True:
        if _time.time() > t_end - 3.0:
            break
        rem = t_end - _time.time()
        if rem < 5.0:
            break
        it += 1
        # analyse contacts of gbest to choose victims
        G = gbestX @ gbestX.T
        _np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        rowmax = G.max(axis=1)
        # contact counts with tolerance
        tol = 0.006
        cnt = (G > (mm - tol)).sum(axis=1)
        order = _np.argsort(-rowmax - 0.0005 * cnt + 0.0001 * rng.standard_normal(n))
        # choose kick size: mostly 1, sometimes 2-3, rarely 4
        r = it % 10
        if r == 0:
            k = 3
        elif r == 3 or r == 7:
            k = 2
        elif r == 9:
            k = 4
        else:
            k = 1
        # occasionally random victim to diversify
        if (it % 7) == 5:
            victims = rng.choice(n, size=k, replace=False)
        else:
            pool = order[:max(6, 2 * k)]
            # pick k among pool: worst + random
            victims = []
            victims.append(int(pool[0]))
            if k > 1:
                rest = rng.choice(pool[1:], size=k - 1, replace=False)
                for q in rest:
                    victims.append(int(q))
            victims = _np.array(victims, dtype=int)
        # build new start
        try:
            Y0 = _ruin_recreate(gbestX, victims, rng, n_samples=16000)
        except Exception:
            amp = 0.08
            Y0 = _normalize(gbestX + amp * rng.standard_normal(size=(n, d)))
        # small shake of untouched points to help re-equilibrate
        shake = 0.008 if (it % 2 == 0) else 0.015
        Y0 = _normalize(Y0 + shake * rng.standard_normal(size=(n, d)) * 0.3)
        # local optimisation: two-stage adam + greedy
        if rem > 25:
            s1, s2, ig = 1600, 900, 900
        elif rem > 12:
            s1, s2, ig = 1100, 600, 600
        else:
            s1, s2, ig = 700, 400, 400
        lim1 = min(_time.time() + rem * 0.5, t_end - 1.5)
        X1, v1 = _adam_lse(Y0, s1, 35.0, 260.0, 0.014, 0.005, lim1)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
            # extra fine polish immediately on improvement
            limf = min(_time.time() + 3.0, t_end - 1.0)
            Xf, vf = _greedy_polish(X1, 700, 0.004, 0.0001, limf, rng)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
            continue
        lim2 = min(_time.time() + rem * 0.3, t_end - 1.0)
        X2, v2 = _adam_lse(X1, s2, 220.0, 420.0, 0.006, 0.002, lim2)
        if v2 < gbest:
            gbest, gbestX = v2, X2.copy()
            continue
        base = X2 if v2 < v1 else X1
        bv = min(v1, v2)
        # greedy polish
        lim3 = min(_time.time() + rem * 0.25, t_end - 0.6)
        X3, v3 = _greedy_polish(base, ig, 0.007, 0.00015, lim3, rng)
        if v3 < gbest:
            gbest, gbestX = v3, X3.copy()
            continue
        # occasional shake-only basin (no ruin) to complement
        if (it % 6) == 0 and rem > 10:
            amp = 0.10
            Ys = _normalize(gbestX + amp * rng.standard_normal(size=(n, d)))
            Xs, vs = _adam_lse(Ys, 900, 40.0, 250.0, 0.012, 0.005, min(_time.time() + 4.0, t_end - 0.8))
            if vs < gbest:
                gbest, gbestX = vs, Xs.copy()
        if it > 200:
            break

    # final ultra-fine polish
    if _time.time() < t_end - 0.8:
        try:
            Xf, vf = _adam_lse(gbestX, 900, 300.0, 900.0, 0.003, 0.0008, t_end - 0.4)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
            Xg, vg = _greedy_polish(gbestX, 800, 0.002, 0.00004, t_end - 0.15, None)
            if vg < gbest:
                gbest, gbestX = vg, Xg.copy()
        except Exception:
            pass
    gbestX = _normalize(gbestX)
    return gbestX
