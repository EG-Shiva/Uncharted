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

def _adam_lse(X0, steps, beta0, beta1, lr, rng, t_end):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    b1p = b1
    b2p = b2
    # precompute powers incrementally
    pow1 = b1
    pow2 = b2
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > t_end:
            break
        beta = beta0 + (beta1 - beta0) * (t / steps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        # stable weights, max subtracted
        W = np.exp(beta * (G - gm))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-16
        D = (W @ X) / rs
        # tangential gradient (move away from D, so Gt points to increase)
        dot = (D * X).sum(axis=1, keepdims=True)
        Gt = D - dot * X
        # Adam
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1)
        vhat = v / (1.0 - pow2)
        pow1 *= b1
        pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip extreme steps
        # limit per-row norm to 0.05
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / (rn + 1e-12))
        # scale is (n,1), broadcast
        X = X - step * scale
        X = _normalize(X)
        if (t % 25) == 0:
            cur = _max_cos(X)
            if cur < best:
                best = cur
                bestX = X.copy()
    return bestX, best

def _greedy_polish(X0, iters, step0, step1, rng, t_end):
    Xcur = X0.copy()
    n, d = Xcur.shape
    best = _max_cos(Xcur)
    bestX = Xcur.copy()
    # exponential decay from step0 to step1
    decay = (step1 / step0) ** (1.0 / max(iters - 1, 1))
    step = step0
    for t in range(iters):
        if (t & 63) == 0 and time.time() > t_end:
            break
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        delta = max(0.004, 1.5 * step)
        thr = m - delta
        # weights: emphasize worst
        Mask = G > thr
        W = np.where(Mask, (G - thr + 0.01), 0.0)
        np.fill_diagonal(W, 0.0)
        Gr = W @ Xcur
        dot = (Gr * Xcur).sum(axis=1, keepdims=True)
        Gt = Gr - dot * Xcur
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        # unit direction where contact exists
        has = (nrm[:, 0] > 1e-9)
        upd = np.zeros_like(Xcur)
        upd[has] = Gt[has] / nrm[has]
        # rattlers: small random tangent exploration
        nhas = ~has
        if np.any(nhas):
            R = rng.standard_normal(size=(int(nhas.sum()), d))
            # project
            Xr = Xcur[nhas]
            R = R - (R * Xr).sum(axis=1, keepdims=True) * Xr
            rn2 = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            R = R / rn2
            upd[nhas] = R * 0.3
        Xcur = Xcur - step * upd
        Xcur = _normalize(Xcur)
        cur = _max_cos(Xcur)
        if cur < best:
            best = cur
            bestX = Xcur.copy()
        step *= decay
    return bestX, best


def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    TIME_BUDGET = 168.0
    t_end_global = t0 + TIME_BUDGET
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
                # fix nan/inf
                if not _np.all(_np.isfinite(a)):
                    continue
                nr = _np.linalg.norm(a, axis=1, keepdims=True)
                nr = _np.maximum(nr, 1e-12)
                a = a / nr
                if _np.all(_np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(rand_X())
    # evaluate
    vals = [_max_cos(c) for c in cands]
    bi = int(_np.argmin(vals))
    gbestX = cands[bi].copy()
    gbest = float(vals[bi])

    # quick initial polish of loaded best if time allows
    # Phase for restart 0: start from gbest directly
    restart = 0
    # estimate per-restart cost adaptively: use smaller steps as time runs out
    while True:
        if _time.time() > t_end_global - 2.0:
            break
        if restart == 0:
            X0 = gbestX.copy()
        elif restart <= 4:
            # small shake of global best
            amp = [0.04, 0.08, 0.12, 0.20][restart - 1]
            X0 = gbestX + amp * rng.standard_normal(size=(n, d))
            X0 = _normalize(X0)
        elif restart <= 10:
            amp = 0.30 + 0.05 * (restart - 5)
            # half from best shake, half fresh
            if restart % 2 == 0:
                X0 = gbestX + amp * rng.standard_normal(size=(n, d))
                X0 = _normalize(X0)
            else:
                X0 = rand_X()
        else:
            # alternate fresh and medium shake
            if restart % 3 == 0:
                amp = 0.35
                X0 = gbestX + amp * rng.standard_normal(size=(n, d))
                X0 = _normalize(X0)
            else:
                X0 = rand_X()

        # allocate steps based on remaining time
        rem = t_end_global - _time.time()
        if rem < 4:
            break
        # two-stage smooth: coarse then fine
        # keep steps modest for speed
        if restart == 0:
            s1, b0, b1, lr = 2500, 40.0, 200.0, 0.015
        else:
            s1, b0, b1, lr = 1800, 35.0, 220.0, 0.015

        t_lim1 = _time.time() + min(rem * 0.6, 8.0)
        # cap steps if little time
        if rem < 20:
            s1 = 800
        X1, v1 = _adam_lse(X0, s1, b0, b1, lr, rng, min(t_lim1, t_end_global - 1.5))
        if v1 < gbest:
            gbest = v1
            gbestX = X1.copy()

        # second anneal with higher beta from X1
        rem2 = t_end_global - _time.time()
        if rem2 < 1.5:
            restart += 1
            continue
        s2 = 1200 if rem2 > 15 else 500
        t_lim2 = _time.time() + min(rem2 * 0.4, 5.0)
        X2, v2 = _adam_lse(X1, s2, 150.0, 300.0, 0.008, rng, min(t_lim2, t_end_global - 1.0))
        if v2 < gbest:
            gbest = v2
            gbestX = X2.copy()

        # greedy contact polish on the better of X1/X2
        base = X2 if v2 < v1 else X1
        rem3 = t_end_global - _time.time()
        if rem3 < 1.0:
            restart += 1
            continue
        itg = 1500 if rem3 > 12 else 600
        X3, v3 = _greedy_polish(base, itg, 0.008, 0.00015, rng, t_end_global - 0.5)
        if v3 < gbest:
            gbest = v3
            gbestX = X3.copy()

        restart += 1
        if restart > 60:
            break

    # final ultra-fine polish on global best with remaining time
    if _time.time() < t_end_global - 0.8:
        try:
            Xf, vf = _greedy_polish(gbestX, 800, 0.002, 0.00005, rng, t_end_global - 0.2)
            if vf < gbest:
                gbest = vf
                gbestX = Xf.copy()
        except Exception:
            pass

    gbestX = _normalize(gbestX)
    return gbestX
