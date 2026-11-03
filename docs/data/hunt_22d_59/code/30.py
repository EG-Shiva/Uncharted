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

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu=0.9):
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
        if (s % 200 == 199) or (s == steps - 1):
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
        if (s % 200 == 199) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _hole_vector(Xkeep, rng, n_sample=6000):
    # largest empty cap: minimise max_j <v, x_j>
    d = Xkeep.shape[1]
    C = _normed(rng.standard_normal((n_sample, d)))
    S = C @ Xkeep.T
    mx = S.max(axis=1)
    i0 = int(np.argmin(mx))
    v = C[i0].copy()
    # refine single vector with LSE projected gradient
    beta0, beta1 = 40.0, 350.0
    lr0, lr1 = 0.05, 0.002
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    steps = 250
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        g = Xkeep @ v
        m = float(g.max())
        w = np.exp(beta * (g - m))
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        grad = (w @ Xkeep) / sw
        gt = grad - float(grad @ v) * v
        v = v - lr * gt
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            v = _normed(rng.standard_normal((1, d)))[0]
        else:
            v = v / nv
    return v

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

    def tl(lim):
        return (time.time() - t0) < lim

    # Stage 0: polish incumbent a bit (cheap)
    try:
        X1 = _smooth_opt(best_X, steps=2500, beta0=30.0, beta1=300.0, lr0=0.02, lr1=0.001, mu=0.9)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1; best_X = X1
        X1 = _hard_opt(best_X, steps=1200, lr0=0.004, lr1=0.0002, mu=0.85, marg0=0.005, marg1=0.0005)
        v1 = _max_cos(X1)
        if v1 < best_val:
            best_val = v1; best_X = X1
    except Exception:
        pass

    trial = 0
    # main topology-changing basin hopping
    while tl(148.0) and trial < 120:
        mode = trial % 5
        Xt = None
        try:
            if mode == 0:
                # large global kick
                sigmas = [0.08, 0.15, 0.25, 0.35]
                sigma = sigmas[(trial // 5) % len(sigmas)]
                Xt = _normed(best_X + sigma * rng.standard_normal((59, 22)))
            elif mode == 1:
                # regional block shake: k nearest neighbours of random centre
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                centre = int(rng.integers(0, 59))
                order = np.argsort(-G[centre])
                k = 6 + ((trial // 5) % 5) * 2  # 6,8,10,12,14
                idx = order[:k]
                Xt = best_X.copy()
                Xt[idx] = Xt[idx] + 0.65 * rng.standard_normal((k, 22))
                mask = np.ones(59, dtype=bool); mask[idx] = False
                Xt[mask] = Xt[mask] + 0.012 * rng.standard_normal((int(mask.sum()), 22))
                Xt = _normed(Xt)
            elif mode == 2:
                # kill overloaded points and fill holes
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                tol = 0.002
                deg = (G >= (m - tol)).sum(axis=1)
                nk = 2 + ((trial // 5) % 2)  # 2 or 3
                kill = np.argsort(-deg)[:nk]
                keep_idx = [i for i in range(59) if i not in set(kill.tolist())]
                Xkeep = best_X[keep_idx]
                Xt = best_X.copy()
                # sequential hole fill (second hole sees first)
                cur_keep = Xkeep.copy()
                for kk in kill:
                    v = _hole_vector(cur_keep, rng, n_sample=6000)
                    Xt[kk] = v
                    cur_keep = np.vstack([cur_keep, v[None, :]])
                # tiny noise on survivors
                Xt = _normed(Xt + 0.008 * rng.standard_normal((59, 22)))
            elif mode == 3:
                # single worst kill + moderate global noise
                G = best_X @ best_X.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G >= (m - 0.002)).sum(axis=1)
                kill = int(np.argmax(deg))
                keep_idx = [i for i in range(59) if i != kill]
                v = _hole_vector(best_X[keep_idx], rng, n_sample=8000)
                Xt = best_X + 0.02 * rng.standard_normal((59, 22))
                Xt[kill] = v * 1.0
                Xt = _normed(Xt)
            else:
                # random subset replacement
                k = 3 + ((trial // 5) % 3)  # 3,4,5
                idx = rng.choice(59, size=k, replace=False)
                Xt = best_X + 0.015 * rng.standard_normal((59, 22))
                Xt[idx] = rng.standard_normal((k, 22))
                Xt = _normed(Xt)
        except Exception:
            trial += 1
            continue
        # short evaluation of new basin
        try:
            b0 = 25.0 + (trial % 3) * 10.0
            b1 = 220.0 + (trial % 4) * 50.0
            Xo = _smooth_opt(Xt, steps=1800, beta0=b0, beta1=b1, lr0=0.018, lr1=0.0008, mu=0.9)
            Xo = _hard_opt(Xo, steps=700, lr0=0.0035, lr1=0.0002, mu=0.85, marg0=0.005, marg1=0.0005)
        except Exception:
            trial += 1
            continue
        v = _max_cos(Xo)
        # two-stage: if close, extend polish
        if v < best_val + 0.004 and tl(150.0):
            try:
                Xo2 = _smooth_opt(Xo, steps=1500, beta0=120.0, beta1=420.0, lr0=0.004, lr1=0.0002, mu=0.9)
                Xo2 = _hard_opt(Xo2, steps=800, lr0=0.001, lr1=0.00008, mu=0.8, marg0=0.002, marg1=0.0004)
                v2 = _max_cos(Xo2)
                if v2 < v:
                    v = v2; Xo = Xo2
            except Exception:
                pass
        if v < best_val:
            best_val = v
            best_X = Xo.copy()
        trial += 1

    # final fine polish of champion
    if tl(168.0):
        try:
            Xf = _hard_opt(best_X, steps=2000, lr0=0.001, lr1=0.00006, mu=0.8, marg0=0.0018, marg1=0.0003)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
        except Exception:
            pass
    if tl(172.0):
        try:
            Xf = _smooth_opt(best_X, steps=1500, beta0=200.0, beta1=550.0, lr0=0.0025, lr1=0.00015, mu=0.9)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf; best_X = Xf
            Xf2 = _hard_opt(best_X, steps=1000, lr0=0.0005, lr1=0.00004, mu=0.8, marg0=0.0009, marg1=0.00025)
            vf2 = _max_cos(Xf2)
            if vf2 < best_val:
                best_val = vf2; best_X = Xf2
        except Exception:
            pass
    best_X = _normed(best_X)
    return best_X
