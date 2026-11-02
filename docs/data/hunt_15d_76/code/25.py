import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos_row(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    r = G.max(axis=1)
    return m, r, G

def _anneal_balanced(X, steps, T0, T1, lr0, lr1, lam_uni, gamma, mom, t0, tlim):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        if (s & 255) == 0 and (time.time() - t0) > tlim:
            break
        f = s / max(1, steps - 1)
        T = np.exp(lT0 + (lT1 - lT0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # softmax pair weights
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        # per-point worst-row weighting to equalize load
        r = E.max(axis=1)  # not needed; use rowmax of G
        # rowmax of G
        rm = G.max(axis=1)
        # weight in [1-gamma, 1+gamma]-ish, pushes worst points harder
        w = np.exp((rm - m) / max(T * 2.0, 1e-6))
        # normalize w to mean 1
        w = w / (w.mean() + 1e-12)
        w = 1.0 + gamma * (w - 1.0)
        Gr = (E @ X) / S
        # uniform spreading term: gradient of mean G^2 (all pairs)
        if lam_uni > 0:
            G2 = G.copy()
            G2[G2 < -1.0] = 0.0  # diag already -2 -> zeroed
            # G2 in [-1,1]; square-ish repulsion: use G2*|G2|
            Gu = G2 * np.abs(G2)
            np.fill_diagonal(Gu, 0.0)
            Gr2 = (Gu @ X) / (n * n)
            # scale to comparable magnitude
            Gr = Gr + lam_uni * Gr2
        # per-point scaling
        Gr = Gr * w[:, None]
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        rms = float(np.sqrt((Gt * Gt).sum() / n)) + 1e-12
        Gt /= rms
        V = mom * V + (1.0 - mom) * Gt
        X = X - lr * V
        X = _norm(X)
    return X

def _refine_hole_vec(v, Y, steps=120, T0=0.008, lr0=0.1):
    v = v / np.linalg.norm(v)
    for k in range(steps):
        f = k / max(1, steps - 1)
        Tk = T0 * (0.15 ** f)
        lrk = lr0 * (0.25 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        g = g - float(np.dot(g, v)) * v
        ng = float(np.linalg.norm(g))
        if ng < 1e-12:
            break
        # normalize
        g /= (ng + 1e-12)
        g *= min(1.0, ng * 4.0)
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if not np.isfinite(nv) or nv == 0:
            break
        v /= nv
    return v

def _hole_fill(X, victims, rng, NS=3500):
    n, d = X.shape
    for idx in victims:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        Y = X[mask]
        C = rng.standard_normal(size=(NS, d))
        C = _norm(C)
        sc = C @ Y.T
        sm = sc.max(axis=1)
        bi = int(np.argmin(sm))
        v0 = C[bi]
        v0 = _refine_hole_vec(v0, Y, steps=120, T0=0.008, lr0=0.12)
        X[idx] = v0
    return X

def solve():
    t0 = time.time()
    TLIM = 165.0
    rng = np.random.default_rng(42)
    n, d = 76, 15
    try:
        Xb = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        Xb = None
    try:
        Xr = np.asarray(np.load("record.npy"), dtype=np.float64)
    except Exception:
        Xr = None
    cands = []
    if Xb is not None and Xb.shape == (n, d):
        cands.append(_norm(Xb))
    if Xr is not None and Xr.shape == (n, d):
        cands.append(_norm(Xr))
    if not cands:
        cands.append(_norm(rng.standard_normal(size=(n, d))))
    def mc(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    vals = [mc(c) for c in cands]
    bi = int(np.argmin(vals))
    Xbest = cands[bi].copy()
    best = float(vals[bi])
    Xcur = Xbest.copy()
    cur = best

    # quick evaluation of contacts for victim selection
    def pick_victims(X, trial, rng):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        rm = G.max(axis=1)
        cnt = (G >= m - 0.005).sum(axis=1)
        order_bad = np.argsort(-rm)  # worst rowmax first
        order_cnt = np.argsort(-cnt)
        ratt = np.where(cnt == 0)[0]
        # number to replace: cycle 2..6
        k = 2 + (trial % 5)  # 2,3,4,5,6
        v = []
        # 1-2 worst rowmax
        v.append(int(order_bad[0]))
        if k >= 3:
            v.append(int(order_bad[1]))
        # 1 rattler if exists
        if len(ratt) > 0 and k >= 3:
            # pick rattler with largest rowmax (least slack) or random
            rrm = rm[ratt]
            ri = int(ratt[int(np.argmax(rrm))]) if (trial % 2 == 0) else int(ratt[int(rng.integers(0, len(ratt)))])
            if ri not in v:
                v.append(ri)
        # fill rest: prefer high-contact then random
        p = 0
        while len(v) < k:
            cand = int(order_cnt[p % n]) if (p % 3 != 2) else int(rng.integers(0, n))
            p += 1
            if cand not in v:
                v.append(cand)
            if p > 3 * n:
                break
        return v[:k]

    sigmas = [0.03, 0.08, 0.18, 0.32, 0.05, 0.12, 0.25, 0.02, 0.10, 0.22]
    trial = 0
    # wandering basin-hopping
    while True:
        if time.time() - t0 > TLIM - 14:
            break
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        # occasional fresh Thomson start to seek balanced basin
        if trial % 17 == 0 and (time.time() - t0) < TLIM - 40:
            Xt = _norm(rng.standard_normal(size=(n, d)))
            Xt = _anneal_balanced(Xt, steps=2200, T0=0.03, T1=0.002, lr0=5.0, lr1=0.2,
                                  lam_uni=0.6, gamma=1.0, mom=0.85, t0=t0, tlim=TLIM - 12)
        else:
            Xt = Xcur.copy()
            victims = pick_victims(Xt, trial, rng)
            # hole-optimized reinsertion (sequential)
            try:
                NS = 3000 if sigma < 0.15 else 2000
                Xt = _hole_fill(Xt, victims, rng, NS=NS)
            except Exception:
                pass
            # gaussian shake (stronger on victims' neighbourhood is implicit)
            if sigma > 0:
                Xt = Xt + sigma * rng.standard_normal(size=(n, d))
                Xt = _norm(Xt)
            # anneal params adapt to sigma
            if sigma < 0.06:
                Xt = _anneal_balanced(Xt, steps=1600, T0=0.014, T1=0.0012, lr0=3.0, lr1=0.15,
                                      lam_uni=0.25, gamma=1.2, mom=0.88, t0=t0, tlim=TLIM - 12)
            elif sigma < 0.15:
                Xt = _anneal_balanced(Xt, steps=2000, T0=0.020, T1=0.0010, lr0=4.5, lr1=0.12,
                                      lam_uni=0.35, gamma=1.0, mom=0.86, t0=t0, tlim=TLIM - 12)
            else:
                Xt = _anneal_balanced(Xt, steps=2400, T0=0.028, T1=0.0015, lr0=6.0, lr1=0.15,
                                      lam_uni=0.45, gamma=0.8, mom=0.85, t0=t0, tlim=TLIM - 12)
        v = mc(Xt)
        if v < best - 1e-15:
            best = v
            Xbest = Xt.copy()
            Xcur = Xt.copy()
            cur = v
            # extra greedy low-T squeeze on improvement if time allows
            if time.time() - t0 < TLIM - 12:
                Xsq = _anneal_balanced(Xbest.copy(), steps=1200, T0=0.0025, T1=0.0003,
                                       lr0=0.7, lr1=0.05, lam_uni=0.05, gamma=1.5,
                                       mom=0.90, t0=t0, tlim=TLIM - 10)
                vs = mc(Xsq)
                if vs < best:
                    best = vs
                    Xbest = Xsq
                    Xcur = Xsq.copy()
                    cur = vs
        else:
            # sideways acceptance to keep wandering (decaying tolerance)
            elapsed = time.time() - t0
            tol = 0.004 * (1.0 - elapsed / TLIM) + 0.0002
            if v < cur + tol and v < cur + 0.01:
                # accept worse to change basin; bias to accept if more balanced
                Xcur = Xt.copy()
                cur = v
            elif trial % 7 == 0 and (time.time() - t0) < TLIM - 20:
                # occasional restart from best to not drift too far
                Xcur = Xbest.copy()
                cur = best
        # early thorough check: keep time for final polish
        if trial > 60 and (time.time() - t0) > TLIM - 20:
            break

    # final low-temperature polish of best
    if time.time() - t0 < TLIM - 2:
        try:
            Xbest = _anneal_balanced(Xbest, steps=1500, T0=0.0018, T1=0.00025,
                                     lr0=0.5, lr1=0.03, lam_uni=0.0, gamma=1.5,
                                     mom=0.90, t0=t0, tlim=TLIM)
        except Exception:
            pass
    Xbest = _norm(np.nan_to_num(Xbest))
    return Xbest
