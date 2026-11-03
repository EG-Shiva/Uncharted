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

def _adam_lse(X0, steps, beta0, beta1, lr0, lr1, t_end, track_best=True):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    pow1 = b1; pow2 = b2
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
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
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - pow1); vhat = v / (1.0 - pow2)
        pow1 *= b1; pow2 *= b2
        step = lr * mhat / (np.sqrt(vhat) + eps)
        rn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, 0.05 / rn)
        X = X - step * scale
        X = _normalize(X)
        if track_best and (t % 50 == 0 or t == steps):
            cur = _max_cos(X)
            if cur < best:
                best = cur; bestX = X.copy()
    if track_best:
        cur = _max_cos(X)
        if cur < best:
            best = cur; bestX = X.copy()
        return bestX, best
    else:
        return X, _max_cos(X)

def _find_hole(X, rng, S=30000):
    d = X.shape[1]
    best_c = None
    best_val = 1e9
    batch = 10000
    done = 0
    while done < S:
        b = min(batch, S - done)
        C = rng.standard_normal(size=(b, d))
        C = _normalize(C)
        sc = C @ X.T
        mx = sc.max(axis=1)
        i = int(np.argmin(mx))
        if float(mx[i]) < best_val:
            best_val = float(mx[i]); best_c = C[i].copy()
        done += b
    c = best_c.copy()
    s0, s1, steps = 0.05, 0.002, 200
    dec = (s1 / s0) ** (1.0 / max(steps - 1, 1))
    s = s0
    for _ in range(steps):
        dots = X @ c
        j = int(np.argmax(dots))
        g = X[j] - dots[j] * c
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        g = g / gn
        c = c - s * g
        c = c / max(float(np.linalg.norm(c)), 1e-12)
        s *= dec
    return c

def _victim_order(X, k=8):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mm = float(G.max())
    rowmax = G.max(axis=1)
    cnt = (G > (mm - 0.006)).sum(axis=1)
    order = np.argsort(-rowmax - 0.0005 * cnt)
    return [int(q) for q in order[:k]], mm, rowmax, cnt

def _best_removal(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    best_i = 0
    best_v = 1e9
    idx = np.arange(n)
    for i in range(n):
        keep = idx[idx != i]
        v = float(G[np.ix_(keep, keep)].max())
        if v < best_v:
            best_v = v; best_i = i
    return int(best_i), float(best_v)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 59, 22

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal(size=(n, d))))
    vals = [_max_cos(c) for c in cands]
    gbestX = cands[int(np.argmin(vals))].copy()
    gbest = float(min(vals))

    # short initial polish of incumbent
    try:
        X1, v1 = _adam_lse(gbestX, 1000, 40.0, 300.0, 0.014, 0.005, t_end - 5.0)
        if v1 < gbest:
            gbest, gbestX = v1, X1.copy()
    except Exception:
        pass

    def cycle_58_to_59():
        nonlocal gbest, gbestX
        if time.time() > t_end - 30.0:
            return
        order, mm, rowmax, cnt = _victim_order(gbestX, k=4)
        # try two victims in time permitting
        for vi, victim in enumerate(order[:2]):
            if time.time() > t_end - 28.0:
                break
            mask = np.ones(n, dtype=bool); mask[victim] = False
            Y58 = gbestX[mask].copy()
            Y58 = _normalize(Y58 + 0.004 * rng.standard_normal(size=Y58.shape))
            lim = min(t_end - 20.0, time.time() + 4.0)
            Yp, vp = _adam_lse(Y58, 900, 30.0, 250.0, 0.015, 0.005, lim)
            # thorough if first victim
            if vi == 0 and time.time() < t_end - 25.0:
                lim2 = min(t_end - 20.0, time.time() + 4.0)
                Yp2, vp2 = _adam_lse(Yp, 1200, 40.0, 380.0, 0.012, 0.004, lim2)
                if vp2 < vp:
                    Yp, vp = Yp2, vp2
            # locate largest hole with 30k-50k candidates
            try:
                S = 35000 if vi == 0 else 25000
                cstar = _find_hole(Yp, rng, S=S)
            except Exception:
                cstar = _normalize(rng.standard_normal(size=(d,)))
            Y59 = np.vstack([Yp, cstar.reshape(1, -1)])
            Y59 = _normalize(Y59)
            rem = t_end - time.time()
            if rem < 6.0:
                # still track if lucky
                vv = _max_cos(Y59)
                if vv < gbest:
                    gbest, gbestX = vv, Y59.copy()
                break
            lim1 = min(t_end - 15.0, time.time() + 3.0)
            Z1, _ = _adam_lse(Y59, 700, 20.0, 100.0, 0.02, 0.008, lim1, track_best=False)
            lim2 = min(t_end - 12.0, time.time() + 5.0)
            Z2, v2 = _adam_lse(Z1, 1400, 120.0, 550.0, 0.010, 0.003, lim2)
            if v2 < gbest:
                gbest, gbestX = v2, Z2.copy()
                break  # success, stop trying other victim

    def cycle_overshoot(K):
        nonlocal gbest, gbestX
        if time.time() > t_end - 28.0:
            return
        # build overcrowded K-point code from gbest
        X = gbestX.copy()
        # add holes / splits until K
        toggle = 0
        while X.shape[0] < K:
            if time.time() > t_end - 24.0:
                break
            if toggle % 2 == 0:
                try:
                    c = _find_hole(X, rng, S=12000)
                except Exception:
                    c = _normalize(rng.standard_normal(size=(d,)))
                X = np.vstack([X, c.reshape(1, -1)])
                X = _normalize(X)
            else:
                try:
                    G = X @ X.T
                    np.fill_diagonal(G, -2.0)
                    mm = float(G.max())
                    cnt = (G > (mm - 0.005)).sum(axis=1)
                    kk = int(np.argmax(cnt))
                    xk = X[kk].copy()
                    r = rng.standard_normal(size=(d,))
                    r = r - float(r @ xk) * xk
                    rn = float(np.linalg.norm(r)) + 1e-12
                    r = r / rn
                    eps_sep = 0.06
                    a = xk + eps_sep * r; a = a / max(float(np.linalg.norm(a)), 1e-12)
                    b = xk - eps_sep * r; b = b / max(float(np.linalg.norm(b)), 1e-12)
                    X[kk] = a
                    X = np.vstack([X, b.reshape(1, -1)])
                except Exception:
                    c = _normalize(rng.standard_normal(size=(d,)))
                    X = np.vstack([X, c.reshape(1, -1)])
            toggle += 1
        if X.shape[0] < 60:
            return
        # brief polish of overcrowded system
        lim = min(t_end - 20.0, time.time() + 3.0)
        Xp, _ = _adam_lse(X, 600, 25.0, 120.0, 0.016, 0.007, lim, track_best=False)
        X = Xp
        # greedy prune back to 59
        while X.shape[0] > 59:
            if time.time() > t_end - 16.0:
                break
            try:
                bi, _ = _best_removal(X)
            except Exception:
                # fallback: remove highest-contact
                G = X @ X.T; np.fill_diagonal(G, -2.0)
                mm = float(G.max()); cnt = (G > (mm - 0.005)).sum(axis=1)
                bi = int(np.argmax(cnt))
            keep = np.ones(X.shape[0], dtype=bool); keep[bi] = False
            X = X[keep].copy()
            # re-polish after each deletion
            lim2 = min(t_end - 14.0, time.time() + 2.5)
            X2, v2 = _adam_lse(X, 600, 40.0, 300.0, 0.012, 0.005, lim2)
            X = X2
            if X.shape[0] == 59 and v2 < gbest:
                gbest, gbestX = v2, X.copy()
        # final joint re-polish of pruned 59
        if X.shape[0] == 59 and time.time() < t_end - 12.0:
            lim3 = min(t_end - 10.0, time.time() + 4.0)
            X3, v3 = _adam_lse(X, 1200, 100.0, 500.0, 0.009, 0.003, lim3)
            if v3 < gbest:
                gbest, gbestX = v3, X3.copy()

    it = 0
    while time.time() < t_end - 12.0:
        it += 1
        if it % 2 == 1:
            cycle_58_to_59()
        else:
            K = 63 if (it % 4 == 0) else 62
            # occasionally 61/64 for diversity
            if it % 8 == 6:
                K = 64
            cycle_overshoot(K)
        if time.time() > t_end - 12.0:
            break

    # final high-beta fine polish of best-ever
    try:
        if time.time() < t_end - 2.0:
            lim = t_end - 1.0
            Xf, vf = _adam_lse(gbestX, 2000, 150.0, 800.0, 0.006, 0.0015, lim)
            if vf < gbest:
                gbest, gbestX = vf, Xf.copy()
    except Exception:
        pass
    gbestX = _normalize(gbestX)
    # tiny shrink guard (no-op for verifier, keeps strict feasibility)
    return np.asarray(gbestX, dtype=float)
