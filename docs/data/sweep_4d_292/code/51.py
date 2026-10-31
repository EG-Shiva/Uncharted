import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    I, J = np.where(np.triu(G >= m - tol, 1))
    deg = np.zeros(X.shape[0], dtype=int)
    for i, j in zip(I, J):
        deg[i] += 1
        deg[j] += 1
    return m, I, J, deg

def _anneal(X, steps, beta0, beta1, lr0, mu, rng=None):
    X = _norm(X.copy())
    n = X.shape[0]
    V = np.zeros_like(X)
    if beta1 > beta0:
        lb = np.log(beta1 / beta0)
    else:
        lb = 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / S
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        V = mu * V + Gt
        vd = (V * X).sum(axis=1, keepdims=True)
        V = V - vd * X
        X = X - lr * V
        X = _norm(X)
    return X

def _lp_polish_deadline(X, deadline, r0=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(np.asarray(X, dtype=float).copy())
    n, d = X.shape
    best = _max_cos(X)
    Xb = X.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if time.time() >= deadline:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            ci = I[:, None] * d + np.arange(d)
            cj = J[:, None] * d + np.arange(d)
            cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; Xb = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return Xb

def _find_holes(X, k, rng, nsamp=12000):
    S = _norm(rng.standard_normal((nsamp, 4)))
    # max cosine to config: smaller = deeper hole
    M = S @ X.T
    m = M.max(axis=1)
    order = np.argsort(m)
    holes = []
    for idx in order:
        s = S[idx]
        ok = True
        for h in holes:
            if float(s @ h) > 0.90:
                ok = False
                break
        if ok:
            holes.append(s)
            if len(holes) >= k:
                break
    if len(holes) < k:
        for idx in order:
            holes.append(S[idx].copy())
            if len(holes) >= k:
                break
    return np.array(holes)

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(292794)
    n, d = 292, 4

    def _load(fn):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                if np.all(np.linalg.norm(a, axis=1) > 1e-12):
                    return a
        except Exception:
            pass
        return None

    best = _load("best.npy")
    rec = _load("record.npy")
    if best is None and rec is None:
        best = _norm(rng.standard_normal((n, d)))
    elif best is None:
        best = rec.copy()
    if rec is not None and _max_cos(rec) < _max_cos(best):
        best = rec.copy()
    best_val = _max_cos(best)
    Xbest = best.copy()

    # quick initial short polish of incumbent if time allows (2s)
    # skip long polish now; do it at end. Just ensure best is locally tight.
    it = 0
    # main basin-hopping
    while time.time() < DEAD - 18.0:
        it += 1
        mode = it % 4
        Xc = Xbest.copy()
        m_cur, Ic, Jc, deg = _contacts(Xc, tol=2e-7)
        order_hi = np.argsort(-deg)
        order_lo = np.argsort(deg)
        if mode == 0:
            # global tangent shake, small
            sigma = 0.012 if (it % 8 == 0) else 0.006
            noise = rng.standard_normal((n, d)) * sigma
            # project tangent
            noise -= (noise * Xc).sum(axis=1, keepdims=True) * Xc
            Xc = _norm(Xc + noise)
            Xc = _anneal(Xc, steps=300, beta0=60.0, beta1=500.0, lr0=0.025, mu=0.82)
        elif mode == 1 or mode == 2:
            # transplant overloaded + rattlers to holes
            k = int(rng.integers(4, 11))  # 4..10 points
            # sources: prefer rattlers/low then high-degree
            src = []
            # always include lowest-degree point
            src.append(int(order_lo[0]))
            # include up to 2 highest
            src.append(int(order_hi[0]))
            if k > 2:
                # mix of high and random
                pool = list(order_hi[:40]) + list(rng.integers(0, n, size=k * 2))
                for c in pool:
                    c = int(c) % n
                    if c not in src:
                        src.append(c)
                    if len(src) >= k:
                        break
            src = src[:k]
            holes = _find_holes(Xc, len(src), rng, nsamp=12000)
            for s, h in zip(src, holes):
                # small jitter on hole to avoid exact symmetry
                j = rng.standard_normal(4) * 0.002
                j -= float(j @ h) * h
                Xc[s] = _norm((h + j).reshape(1, -1))[0]
            # anneal a bit longer to integrate transplants
            Xc = _anneal(Xc, steps=380, beta0=40.0, beta1=700.0, lr0=0.030, mu=0.80)
        else:
            # medium shake on a random subset (cluster surgery)
            k = int(rng.integers(12, 30))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.06
            noise = np.zeros((n, d))
            noise[idx] = rng.standard_normal((k, d)) * sigma
            noise -= (noise * Xc).sum(axis=1, keepdims=True) * Xc
            Xc = _norm(Xc + noise)
            Xc = _anneal(Xc, steps=320, beta0=80.0, beta1=600.0, lr0=0.022, mu=0.85)

        v = _max_cos(Xc)
        # only LP-polish if promising (within 2e-3 of best or better)
        if v < best_val + 2e-3 and time.time() < DEAD - 18.0:
            # short polish budget 1.2s
            Xp = _lp_polish_deadline(Xc, min(time.time() + 1.2, DEAD - 18.0), r0=8e-4)
            vp = _max_cos(Xp)
        else:
            Xp, vp = Xc, v
        if vp < best_val - 1e-12:
            Xbest = Xp.copy()
            best_val = vp
        # also opportunistic: if anneal alone beat best without polish, keep polishing a bit more
        if time.time() >= DEAD - 18.0:
            break

    # final long polish of the champion
    remain = DEAD - time.time()
    if remain > 2.0:
        Xbest = _lp_polish_deadline(Xbest, time.time() + min(remain - 1.0, 16.0), r0=1e-3)
        best_val = _max_cos(Xbest)
    Xbest = _norm(Xbest)
    return Xbest
