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

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
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
        X = _normed(X)
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _hard_opt(X, steps, lr0, lr1, mu, marg0, marg1):
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
        deg = W.sum(axis=1, keepdims=True)
        Grad = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Grad[zm, :] = 0.0
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _find_voids(Xrest, k, rng, n_cand=6000):
    d = Xrest.shape[1]
    R = rng.standard_normal((n_cand, d))
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    S = R @ Xrest.T
    rowmax = S.max(axis=1)
    order = np.argsort(rowmax)
    picked = []
    for idx in order:
        cand = R[idx]
        ok = True
        for p in picked:
            if float(cand @ p) > 0.75:
                ok = False
                break
        if ok:
            picked.append(cand)
            if len(picked) >= k:
                break
    while len(picked) < k:
        v = rng.standard_normal(d)
        v = v / max(np.linalg.norm(v), 1e-12)
        picked.append(v)
    return np.array(picked)

def solve():
    import time
    import numpy as np
    t0 = time.time()
    rng = np.random.default_rng(0)
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
        best_X = _normed(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    def tl(limit):
        return (time.time() - t0) < limit

    # Stage 0: polish start point
    try:
        X1 = _smooth_opt(best_X, steps=4000, beta0=30.0, beta1=300.0, lr0=0.02, lr1=0.001, mu=0.9)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1; best_X = X1
        X1 = _hard_opt(best_X, steps=2000, lr0=0.005, lr1=0.0002, mu=0.85, marg0=0.006, marg1=0.0006)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1; best_X = X1
    except Exception:
        pass

    sigmas = [0.006, 0.014, 0.028, 0.055, 0.10, 0.18, 0.01, 0.04, 0.08]
    trial = 0
    while tl(148.0) and trial < 300:
        mode = trial % 10
        Xt = None
        sigma = sigmas[trial % len(sigmas)]
        if mode in (2, 5, 8):
            # overload-targeted void reinsertion
            try:
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                worst = np.argsort(-rowmax)
                if mode == 5:
                    k = 3
                elif mode == 8:
                    k = 1
                else:
                    k = 2
                # occasionally also include high-degree overloaded points
                # degree at tight tolerance
                m = float(G.max())
                deg = (G >= (m - 0.001)).sum(axis=1)
                # boost worst by degree
                alt = np.argsort(-(rowmax + 0.002 * deg))
                # mix: take union, keep k
                cand_idx = list(dict.fromkeys(list(worst[:6]) + list(alt[:6])))[:k]
                # if duplicate handling, just use worst[:k] when small
                rep_idx = np.array(cand_idx[:k])
                if len(rep_idx) < k:
                    rep_idx = worst[:k]
                mask = np.ones(59, dtype=bool)
                mask[rep_idx] = False
                Xrest = best_X[mask]
                voids = _find_voids(Xrest, k, rng, n_cand=6000)
                Xt = best_X.copy()
                for ii, ri in enumerate(rep_idx):
                    Xt[ri] = voids[ii]
                # small jitter on the rest to break symmetry
                Xt = Xt + 0.006 * rng.standard_normal((59, 22))
            except Exception:
                Xt = best_X + sigma * rng.standard_normal((59, 22))
        elif mode == 9:
            # big global shake to new basin
            Xt = best_X + 0.22 * rng.standard_normal((59, 22))
        else:
            # gaussian perturb, occasionally subspace rotation of a subset
            Xt = best_X + sigma * rng.standard_normal((59, 22))
            if mode == 4:
                # random rotation of 6 points in a random 2D subspace
                try:
                    sub = rng.choice(59, size=6, replace=False)
                    a = rng.standard_normal(22); a /= np.linalg.norm(a)
                    b = rng.standard_normal(22); b -= (b @ a) * a; b /= max(np.linalg.norm(b), 1e-12)
                    ang = rng.uniform(-0.3, 0.3)
                    ca, sa = np.cos(ang), np.sin(ang)
                    for ri in sub:
                        x = Xt[ri]
                        xa = float(x @ a); xb = float(x @ b)
                        x = x - xa * a - xb * b
                        na = ca * xa - sa * xb
                        nb = sa * xa + ca * xb
                        Xt[ri] = x + na * a + nb * b
                except Exception:
                    pass
        Xt = _normed(Xt)
        b0 = 22.0 + (trial % 5) * 8.0
        b1 = 220.0 + (trial % 4) * 70.0
        lr0 = 0.015 if (trial % 2 == 0) else 0.010
        try:
            Xo = _smooth_opt(Xt, steps=3000, beta0=b0, beta1=b1, lr0=lr0, lr1=0.0008, mu=0.9)
            Xo = _hard_opt(Xo, steps=1200, lr0=0.004, lr1=0.0002, mu=0.85, marg0=0.005, marg1=0.0005)
        except Exception:
            trial += 1
            continue
        v = _max_cos(Xo)
        if v < best_val:
            best_val = v
            best_X = Xo.copy()
        trial += 1

    if tl(158.0):
        try:
            Xf = _hard_opt(best_X, steps=2500, lr0=0.0012, lr1=0.00008, mu=0.8, marg0=0.002, marg1=0.0004)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass
    if tl(165.0):
        try:
            Xf = _smooth_opt(best_X, steps=1500, beta0=200.0, beta1=600.0, lr0=0.003, lr1=0.0002, mu=0.9)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
            Xf = _hard_opt(best_X, steps=1200, lr0=0.0006, lr1=0.00005, mu=0.8, marg0=0.001, marg1=0.0003)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
