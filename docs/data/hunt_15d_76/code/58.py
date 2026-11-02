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

def _anneal(X, n_iter, T0, T1, lr0, lr1, deadline, check_every=200):
    n, d = X.shape
    t0t = time.time()
    for k in range(n_iter):
        if k % check_every == 0 and time.time() > deadline:
            break
        f = k / max(1, n_iter - 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = GX - dot * X
        X = X - lr * Gt
        X = _norm(X)
    return X

def _refine_hole(v, Y, n_iter=200, T0=0.008, lr0=0.08, deadline=None):
    v = v / np.linalg.norm(v)
    for k in range(n_iter):
        if deadline is not None and k % 50 == 0 and time.time() > deadline:
            break
        f = k / max(1, n_iter - 1)
        Tk = T0 * (0.2 ** f)
        lrk = lr0 * (0.2 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w = w / sw
        g = w @ Y
        g = g - np.dot(g, v) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-14:
            break
        # limit step
        g = g / (ng + 1e-12) * min(1.0, ng * 5.0)
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v = v / nv
    return v

def _whiten(X):
    n, d = X.shape
    C = (X.T @ X) / n
    # eig
    try:
        w, V = np.linalg.eigh(C)
    except Exception:
        return X
    w = np.maximum(w, 1e-6)
    # A = C^{-1/2} / sqrt(d) * sqrt(trace?) trace=1 so target I/d
    # A = V diag(1/sqrt(w*d)) V^T
    inv = 1.0 / np.sqrt(w * d)
    # clamp to avoid extreme
    inv = np.clip(inv, 0.5, 2.0)
    A = (V * inv[None, :]) @ V.T
    Y = X @ A
    return _norm(Y)

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 172.0
    # load
    def try_load(fn):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                # remove nan
                if np.all(np.isfinite(a)):
                    return a
        except Exception:
            pass
        return None
    xb = try_load("best.npy")
    xr = try_load("record.npy")
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        cands.append(_norm(rng.normal(size=(76, 15))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])

    n, d = 76, 15
    trial = 0
    # quick contact helper threshold for rattler detection
    while time.time() < deadline - 12.0:
        trial += 1
        # contacts with moderate tol to find loose points
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        # two tolerances: tight for hubs, loose for rattlers
        thr_loose = 2e-4
        cnt = np.sum(G > (mcur - thr_loose), axis=1)
        order_hi = np.argsort(-cnt)
        order_lo = np.argsort(cnt)
        sigmas = [0.004, 0.01, 0.025, 0.06, 0.12]
        sigma = sigmas[(trial - 1) % len(sigmas)]
        k_re = 2
        if trial % 3 == 0:
            k_re = 3
        elif trial % 2 == 1:
            k_re = 1
            if trial % 7 == 1:
                k_re = 2
        # build index list: mix lowest and highest
        idxs = []
        # always include a rattler/loosest
        idxs.append(int(order_lo[(trial - 1) % n]))
        if k_re >= 2:
            idxs.append(int(order_hi[(trial - 1) % n]))
        if k_re >= 3:
            idxs.append(int(rng.integers(0, n)))
        # dedup
        idxs = list(dict.fromkeys(idxs))

        Xt = Xbest.copy()
        # occasional whitening jump to fix anisotropy (every 6th trial)
        if trial % 6 == 0:
            # whiten then add noise
            Xt = _whiten(Xt)
            # keep best anchor? blend
            Xt = _norm(Xt + 0.02 * rng.normal(size=(n, d)))

        # hole-optimized reinsertion
        for idx in idxs:
            if time.time() > deadline - 11.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            NS = 2500
            C = _norm(rng.normal(size=(NS, d)))
            # scores = max dot
            scores = (C @ Y.T).max(axis=1)
            # pick best few and choose randomly among top to diversify
            # deterministic via trial
            k_top = 8
            part = np.argpartition(scores, k_top - 1)[:k_top]
            # choose one with round-robin
            pick = part[(trial - 1) % k_top]
            # but ensure best is often used
            if trial % 2 == 0:
                pick = part[int(np.argmin(scores[part]))]
            v0 = C[int(pick)]
            v0 = _refine_hole(v0, Y, n_iter=200, T0=0.008, lr0=0.08, deadline=deadline - 9.0)
            Xt[idx] = v0

        # gaussian shake on rest
        Xt = Xt + sigma * rng.normal(size=(n, d))
        Xt = _norm(Xt)

        # anneal params by sigma
        if sigma < 0.02:
            Xt = _anneal(Xt, n_iter=2200, T0=0.012, T1=0.0010, lr0=4.0, lr1=0.15, deadline=deadline - 9.0)
        elif sigma < 0.07:
            Xt = _anneal(Xt, n_iter=2800, T0=0.016, T1=0.0009, lr0=5.0, lr1=0.12, deadline=deadline - 9.0)
        else:
            Xt = _anneal(Xt, n_iter=3200, T0=0.022, T1=0.0012, lr0=6.0, lr1=0.15, deadline=deadline - 9.0)

        v = _maxcos(Xt)
        if v < best - 1e-12 and np.all(np.isfinite(Xt)):
            best = v
            Xbest = Xt.copy()
            # extra low-T polish on improvement if time
            if time.time() < deadline - 11.0:
                Xp = _anneal(Xbest.copy(), n_iter=2000, T0=0.0012, T1=0.00025, lr0=0.8, lr1=0.05, deadline=deadline - 9.0)
                vp = _maxcos(Xp)
                if vp < best:
                    best = vp
                    Xbest = Xp.copy()
        # safety: if time very low break
        if time.time() > deadline - 10.0:
            break

    # final gentle polish on best
    if time.time() < deadline - 8.0:
        Xp = _anneal(Xbest.copy(), n_iter=1500, T0=0.0010, T1=0.0003, lr0=0.5, lr1=0.05, deadline=deadline - 2.0)
        vp = _maxcos(Xp)
        if vp < best:
            best = vp
            Xbest = Xp.copy()

    Xbest = _norm(Xbest)
    return Xbest
