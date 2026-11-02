import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    import time
    import numpy as np
    t0 = time.time()
    BUDGET = 170.0
    def expired():
        return (time.time() - t0) > BUDGET

    rng = np.random.default_rng(0)

    def normed(a):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    fb = None; fbv = 1e9
    if xb is not None and xb.shape == (76, 15):
        try:
            xb = normed(xb); v = _maxcos(xb); fb = xb.copy(); fbv = v
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            xr = normed(xr); v = _maxcos(xr)
            if v < fbv:
                fb = xr.copy(); fbv = v
        except Exception:
            pass
    if fb is None:
        fb = normed(rng.normal(size=(76, 15))); fbv = _maxcos(fb)

    n, d = 76, 15
    bestX = fb.copy(); bestV = float(fbv)

    # ---------- helpers ----------
    def refine_hole(v, Y, iters=180, T0=0.008, lr0=0.10):
        v = v / (np.linalg.norm(v) + 1e-12)
        for k in range(iters):
            f = k / max(1, iters - 1)
            Tk = T0 * (0.15 ** f)
            lrk = lr0 * (0.25 ** f)
            s = Y @ v
            m = float(s.max())
            w = np.exp((s - m) / Tk)
            sw = float(w.sum())
            if sw <= 0 or not np.isfinite(sw):
                break
            w /= sw
            g = w @ Y
            g = g - float(g @ v) * v
            ng = float(np.linalg.norm(g))
            if ng > 1e-12:
                # normalized capped step
                scale = min(1.0, ng * 5.0) / (ng + 1e-12)
                g = g * scale
            v = v - lrk * g
            nv = float(np.linalg.norm(v))
            if nv == 0 or not np.isfinite(nv):
                break
            v = v / nv
        return v

    def anneal(X, n_iter, T0, T1, lr0, lr1, lam0=0.0, lam1=0.0):
        # minimize softmax + lam*||X'X/n - I/d||^2, projected SGD
        lT = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
        ll = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
        lam_log = False
        try:
            if lam0 > 0 or lam1 > 0:
                lam_log = True
                # linear interpolation for lam (not log, may hit 0)
        except Exception:
            pass
        I15 = np.eye(d) / float(d)
        for k in range(n_iter):
            if (k & 127) == 0 and expired():
                break
            f = k / max(1, n_iter - 1)
            T = T0 * np.exp(lT * f)
            lr = lr0 * np.exp(ll * f)
            lam = lam0 + (lam1 - lam0) * f
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            E = np.exp((G - gm) / T)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0 or not np.isfinite(S):
                break
            E /= S
            Gr = E @ X
            # isotropy gradient
            if lam != 0.0:
                M = (X.T @ X) / float(n)
                D = M - I15
                Gi = (4.0 * lam / float(n)) * (X @ D)
                Gr = Gr + Gi * 0.5  # scale to comparable magnitude
            # tangent projection
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gt = Gr - dot * X
            # clip per-row huge grads
            # simple global clip
            gn = float(np.linalg.norm(Gt))
            if gn > 5.0:
                Gt *= (5.0 / gn)
            X = X - lr * Gt
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X = X / nn
        return X

    def mild_whiten(X, alpha):
        try:
            M = (X.T @ X) / float(n)
            # eigendecomposition
            w, V = np.linalg.eigh(M)
            w = np.maximum(w, 1e-6)
            # W = M^{-1/2}/sqrt(15)
            inv = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
            W = inv / np.sqrt(float(d))
            Wa = (1.0 - alpha) * np.eye(d) + alpha * W
            X2 = X @ Wa
            return _norm(X2)
        except Exception:
            return X

    # precompute hub order of best
    def hub_and_slack(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        row_max = G.max(axis=1)
        slack = M - row_max
        # hubs: count above M-0.015
        thr = M - 0.015
        G2 = X @ X.T
        np.fill_diagonal(G2, -2.0)
        counts = (G2 > thr).sum(axis=1)
        hub_order = np.argsort(-counts, kind='stable')
        rattler_order = np.argsort(slack, kind='stable')  # small slack first, large last
        return hub_order, rattler_order, M

    # quick initial polish of best with short pure anneal if time allows (to get baseline tight)
    # don't waste time: one short polish
    try:
        if not expired():
            Xp = anneal(bestX.copy(), n_iter=1200, T0=0.002, T1=0.0003, lr0=0.6, lr1=0.05)
            vp = _maxcos(Xp)
            if vp < bestV:
                bestX = Xp; bestV = vp
    except Exception:
        pass

    trial = 0
    # sigma / kre cycles
    sigmas = [0.015, 0.04, 0.09, 0.18, 0.03, 0.07]
    while not expired():
        trial += 1
        if expired():
            break
        # refresh hub order every 5 trials from bestX
        try:
            hub_order, rattler_order, Mcur = hub_and_slack(bestX)
        except Exception:
            hub_order = np.arange(n); rattler_order = np.arange(n)
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # number to reinsert
        if trial % 4 == 0:
            k_re = 3
        elif trial % 2 == 0:
            k_re = 2
        else:
            k_re = 1
        # larger shake trials delete more
        if sigma > 0.12:
            k_re = 3 if k_re < 3 else 4

        Xt = bestX.copy()
        # choose deletion indices: mix hub + random + rattler
        del_idx = []
        del_idx.append(int(hub_order[(trial - 1) % n]))
        # second: random
        if k_re >= 2:
            del_idx.append(int(rng.integers(0, n)))
        if k_re >= 3:
            # a rattler (large slack) to relocate
            del_idx.append(int(rattler_order[n - 1 - ((trial - 1) % 12)]))
        if k_re >= 4:
            del_idx.append(int(rng.integers(0, n)))
        # dedup
        del_idx = list(dict.fromkeys(del_idx))

        # sequential hole-optimized reinsertion
        for idx in del_idx:
            if expired():
                break
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Y = Xt[mask]
            NS = 3000 if not expired() else 800
            C = rng.normal(size=(NS, d))
            C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            v0 = C[bi]
            v0 = refine_hole(v0, Y, iters=160, T0=0.008, lr0=0.12)
            Xt[idx] = v0

        # gaussian shake on all except just-reinserted (lighter on them)
        noise = rng.normal(size=(n, d))
        # reduce shake on reinserted
        wgt = np.ones((n, 1))
        for idx in del_idx:
            wgt[idx, 0] = 0.25
        Xt = Xt + sigma * wgt * noise
        Xt = _norm(Xt)

        # occasional mild whitening to enforce isotropy / balance
        if trial % 3 == 0:
            Xt = mild_whiten(Xt, alpha=0.25)
            if expired():
                pass

        # two-phase anneal: balanced explore then pure exploit
        # adapt lengths to remaining time
        remain = BUDGET - (time.time() - t0)
        if remain < 12.0:
            # final polish only
            break
        if sigma < 0.05:
            nA, nB = 900, 1400
            T0A, T1A = 0.014, 0.002
            lr0A, lr1A = 3.0, 0.3
        elif sigma < 0.12:
            nA, nB = 1100, 1600
            T0A, T1A = 0.018, 0.0015
            lr0A, lr1A = 4.0, 0.25
        else:
            nA, nB = 1300, 1800
            T0A, T1A = 0.025, 0.0018
            lr0A, lr1A = 5.0, 0.25

        try:
            Xt = anneal(Xt, n_iter=nA, T0=T0A, T1=T1A, lr0=lr0A, lr1=lr1A, lam0=1.5, lam1=0.0)
            if expired():
                v = _maxcos(Xt)
                if v < bestV:
                    bestX = Xt.copy(); bestV = v
                break
            Xt = anneal(Xt, n_iter=nB, T0=0.002, T1=0.00025, lr0=0.7, lr1=0.06, lam0=0.0, lam1=0.0)
        except Exception:
            continue
        try:
            v = _maxcos(Xt)
        except Exception:
            continue
        if v < bestV - 1e-12:
            bestX = Xt.copy(); bestV = v
            # extra greedy polish on improvement if time permits
            if not expired():
                try:
                    Xp2 = anneal(bestX.copy(), n_iter=1200, T0=0.0008, T1=0.00015, lr0=0.4, lr1=0.04)
                    v2 = _maxcos(Xp2)
                    if v2 < bestV:
                        bestX = Xp2; bestV = v2
                except Exception:
                    pass
        # stop early if we beat record by clear margin and time is short? continue searching
        if trial > 60:
            # limit trials to keep final polish time
            remain2 = BUDGET - (time.time() - t0)
            if remain2 < 15.0:
                break

    # ---------- final polish ----------
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 4.0:
            nit = int(min(6000, max(1500, remain * 300)))
            Xf = anneal(bestX.copy(), n_iter=nit, T0=0.0012, T1=0.00012, lr0=0.5, lr1=0.03)
            vf = _maxcos(Xf)
            if vf < bestV:
                bestX = Xf; bestV = vf
    except Exception:
        pass

    bestX = _norm(bestX)
    # tiny shrink guard (no-op, ensures finite)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    # replace any zero row by random
    nr = np.linalg.norm(bestX, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-12:
            v = rng.normal(size=(d,))
            v /= (np.linalg.norm(v) + 1e-12)
            bestX[i] = v
    bestX = _norm(bestX)
    return bestX
