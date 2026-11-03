import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu, check_every=100):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    n = X.shape[0]
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30:
            continue
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        # retract
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nn, 1e-12, out=nn)
        X /= nn
        if (s % check_every == check_every - 1) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _hard_opt(X, steps, lr0, lr1, mu, marg0, marg1, check_every=150):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(float)
        np.fill_diagonal(W, 0.0)
        # normalize to keep step stable
        Grad = (W @ X) / 20.0
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nn, 1e-12, out=nn)
        X /= nn
        if (s % check_every == check_every - 1) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _best_void_vector(Yfix, rng_n, n_cand=4000, refine=250):
    # Yfix: (m,22) fixed points, find v minimizing max_j <v,Yj>
    m, d = Yfix.shape
    C = np.random.randn(n_cand, d)
    C = _normed(C)
    S = C @ Yfix.T
    smax = S.max(axis=1)
    idx = np.argpartition(smax, 5)[:5]
    best_v = None
    best_s = 1e9
    for k in idx:
        v = C[k].copy()
        lr = 0.08
        for t in range(refine):
            dots = Yfix @ v  # (m,)
            mm = float(dots.max())
            beta = 30.0 + (120.0 - 30.0) * (t / max(1, refine - 1))
            w = np.exp(beta * (dots - mm))
            w = w / (w.sum() + 1e-30)
            g = w @ Yfix  # (22,)
            # project onto tangent at v, descend (we minimize max)
            gt = g - (g @ v) * v
            # decay lr
            lrt = lr * (0.2 ** (t / max(1, refine - 1)))
            v = v - lrt * gt
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                v = np.random.randn(d)
                nv = np.linalg.norm(v)
            v = v / nv
        # score
        sc = float((Yfix @ v).max())
        if sc < best_s:
            best_s = sc
            best_v = v.copy()
    return best_v

def solve():
    import time
    import numpy as np
    t0 = time.time()
    np.random.seed(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(np.random.randn(59, 22))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    def tl(limit):
        return (time.time() - t0) < limit

    # quick initial light polish already near optimum; keep short
    try:
        if tl(155.0):
            Xp = _smooth_opt(best_X, steps=1200, beta0=40.0, beta1=300.0, lr0=0.01, lr1=0.0008, mu=0.9)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp; best_X = Xp
    except Exception:
        pass

    trial = 0
    # main basin-hopping with topology changes
    while tl(140.0) and trial < 120:
        mode = trial % 10
        Xt = None
        try:
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            mcur = float(G.max())
            rowmax = G.max(axis=1)
            deg = (G >= (mcur - 0.001)).sum(axis=1)
        except Exception:
            mcur = best_val; rowmax = None; deg = None
        if mode <= 2:
            sigmas = [0.02, 0.045, 0.085]
            sigma = sigmas[mode]
            Xt = best_X + sigma * np.random.randn(59, 22)
            Xt = _normed(Xt)
        elif mode <= 6:
            # kill-and-fill: remove 1-3 most overloaded points, reinsert in voids
            k = 1 + (trial % 3)  # 1,2,3
            try:
                order = np.argsort(-deg + 0.2 * np.random.randn(59))
                kill = order[:k].tolist()
            except Exception:
                kill = [int(np.random.randint(0, 59)) for _ in range(k)]
            Xt = best_X.copy()
            # small shake of survivors to open holes
            Xt = Xt + 0.006 * np.random.randn(59, 22)
            Xt = _normed(Xt)
            for kk in kill:
                mask = np.ones(59, dtype=bool); mask[kk] = False
                Yfix = Xt[mask]
                vnew = _best_void_vector(Yfix, None, n_cand=3000, refine=200)
                Xt[kk] = vnew
                # renormalize (already unit)
            # extra tiny shake
            Xt = Xt + 0.004 * np.random.randn(59, 22)
            Xt = _normed(Xt)
        elif mode <= 8:
            # block randomize 4-5 points
            nb = 4 + (trial % 2)
            kill = np.random.choice(59, size=nb, replace=False)
            Xt = best_X.copy() + 0.008 * np.random.randn(59, 22)
            for kk in kill:
                Xt[kk] = np.random.randn(22)
            Xt = _normed(Xt)
        else:
            # large global kick
            Xt = best_X + 0.18 * np.random.randn(59, 22)
            Xt = _normed(Xt)

        # anneal params vary with trial
        b0 = 25.0 + (trial % 3) * 10.0
        b1 = 220.0 + (trial % 4) * 50.0
        try:
            Xo = _smooth_opt(Xt, steps=1700, beta0=b0, beta1=b1, lr0=0.014, lr1=0.0007, mu=0.9, check_every=200)
            Xo = _hard_opt(Xo, steps=700, lr0=0.0035, lr1=0.0002, mu=0.85, marg0=0.004, marg1=0.0005, check_every=200)
        except Exception:
            trial += 1
            continue
        v = _max_cos(Xo)
        if v < best_val:
            best_val = v
            best_X = Xo.copy()
        trial += 1

    # final intensive polish of best
    if tl(158.0):
        try:
            Xf = _hard_opt(best_X, steps=1800, lr0=0.0012, lr1=0.00007, mu=0.8, marg0=0.0018, marg1=0.00035, check_every=300)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass
    if tl(165.0):
        try:
            Xf = _smooth_opt(best_X, steps=1500, beta0=150.0, beta1=600.0, lr0=0.0025, lr1=0.00015, mu=0.9, check_every=300)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
            Xf = _hard_opt(best_X, steps=1000, lr0=0.0006, lr1=0.00005, mu=0.8, marg0=0.0009, marg1=0.00028, check_every=250)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
