import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _contacts(X, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    rowmax = np.max(G, axis=1)
    return m, cnt, rowmax, G

def _find_hole(Rem, rng, n_samples=8000):
    # Rem: (m,d) normalized fixed points, find vector minimizing max dot
    d = Rem.shape[1]
    S = rng.standard_normal(size=(n_samples, d))
    S = _normalize(S)
    D = S @ Rem.T
    md = np.max(D, axis=1)
    idx = int(np.argmin(md))
    return S[idx], float(md[idx])

def _adam_relax(X0, steps, a0, a1, lr0):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eye = np.eye(n)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        # max off-diagonal
        gm = float(np.max(G - 2.0 * eye))
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        Grad = alpha * (W @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        m2 = b2 * m2 + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # safety clip
        # limit per-row step norm to 0.1
        nr = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.minimum(1.0, 0.1 / (nr + 1e-12))
        step = step * scale
        X = X - step
        X = _normalize(X)
    return X

def _lp_polish(X, seconds, t_end=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _normalize(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(7)
    # load start
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            X = np.load(fn)
            X = np.asarray(X, dtype=float)
            if X.shape != (123, 13):
                continue
            X = _normalize(X)
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best = X.copy()
        except Exception:
            continue
    if best is None:
        X = _normalize(rng.standard_normal(size=(123, 13)))
        best = X
        best_val = _max_cos(X)

    # quick LP squeeze of start if time allows (2s)
    # keep elitist
    try:
        if HARD_END - time.time() > 20:
            Xp = _lp_polish(best, seconds=2.0, t_end=HARD_END - 5)
            vp = _max_cos(Xp)
            if vp < best_val:
                best, best_val = Xp, vp
    except Exception:
        pass

    trial = 0
    # main basin-hopping loop
    while time.time() < HARD_END - 18:
        trial += 1
        remaining = HARD_END - time.time()
        if remaining < 8:
            break
        Xc = best.copy()
        n, d = Xc.shape
        mode = trial % 4
        try:
            m, cnt, rowmax, G = _contacts(Xc, tol=1e-4)
        except Exception:
            m, cnt, rowmax = best_val, np.zeros(n), np.zeros(n)
        if mode == 0:
            # global gaussian shake, varied magnitude
            sigmas = [0.03, 0.07, 0.15, 0.28]
            sigma = sigmas[trial % 4] * (0.7 + 0.6 * rng.random())
            # boost shakeness of overloaded points
            w = (cnt / (np.max(cnt) + 1e-9))
            boost = 1.0 + 1.5 * w
            noise = rng.standard_normal(size=Xc.shape) * (sigma * boost[:, None])
            Xc = _normalize(Xc + noise)
            steps = 1400 + int(rng.integers(0, 800))
            a0, a1, lr0 = 30.0, 260.0, 0.008
        elif mode == 1:
            # kill worst k and hole-reseed
            k = 3 + int(rng.integers(0, 5))  # 3..7
            order = np.argsort(-rowmax)  # worst first
            # include most-contact points
            order2 = np.argsort(-cnt)
            cand = list(order[:k]) + list(order2[:2])
            # unique
            kill = []
            for idx in cand:
                if idx not in kill:
                    kill.append(idx)
                if len(kill) >= k:
                    break
            # also kill rattlers (rowmax far below m)
            ratt = np.where(rowmax < m - 1e-6)[0]
            for idx in ratt[:2]:
                if idx not in kill:
                    kill.append(idx)
            fixed_mask = np.ones(n, dtype=bool)
            fixed_mask[kill] = False
            Rem = Xc[fixed_mask]
            for idx in kill:
                hole, _ = _find_hole(Rem, rng, n_samples=6000)
                hole = _normalize((hole + 0.02 * rng.standard_normal(d)).reshape(1, -1))[0]
                Xc[idx] = hole
                Rem = np.vstack([Rem, hole.reshape(1, -1)])
            # small shake of survivors
            Xc = _normalize(Xc + 0.01 * rng.standard_normal(size=Xc.shape))
            steps = 1800 + int(rng.integers(0, 800))
            a0, a1, lr0 = 25.0, 320.0, 0.009
        elif mode == 2:
            # relocate rattlers / low-contact to holes + cluster shake
            tol = 1e-7
            _, cnt0, rowmax0, _ = _contacts(Xc, tol=tol)
            low = np.where(cnt0 <= np.median(cnt0))[0]
            # pick random cap region to shake
            u = _normalize(rng.standard_normal(size=(1, d)))[0]
            dots = Xc @ u
            cap = np.argsort(-dots)[:12 + int(rng.integers(0, 10))]
            Xc[cap] += 0.18 * rng.standard_normal(size=(len(cap), d))
            # move rattlers to holes
            ratt = np.where(cnt0 == 0)[0]
            if len(ratt) == 0:
                ratt = np.argsort(cnt0)[:2]
            Rem_all = _normalize(Xc.copy())
            # build Rem without rattlers for hole search
            mask = np.ones(n, dtype=bool)
            mask[ratt] = False
            Rem = Xc[mask]
            for idx in ratt:
                hole, _ = _find_hole(Rem, rng, n_samples=8000)
                Xc[idx] = hole
                Rem = np.vstack([Rem, hole.reshape(1, -1)])
            Xc = _normalize(Xc)
            steps = 1600 + int(rng.integers(0, 800))
            a0, a1, lr0 = 35.0, 280.0, 0.007
        else:
            # random subset strong shake
            k = 10 + int(rng.integers(0, 12))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.25
            Xc[idx] = _normalize(Xc[idx] + sigma * rng.standard_normal(size=(k, d)))
            # tiny global jitter
            Xc = _normalize(Xc + 0.008 * rng.standard_normal(size=Xc.shape))
            steps = 1200 + int(rng.integers(0, 600))
            a0, a1, lr0 = 20.0, 220.0, 0.010

        # annealed smooth relaxation
        # adapt steps to remaining time
        if remaining < 25:
            steps = min(steps, 900)
        try:
            Xr = _adam_relax(Xc, steps=steps, a0=a0, a1=a1, lr0=lr0)
        except Exception:
            continue
        # short LP polish
        lp_sec = 2.2
        if remaining < 25:
            lp_sec = 1.2
        try:
            Xp = _lp_polish(Xr, seconds=lp_sec, t_end=HARD_END - 14)
        except Exception:
            continue
        vp = _max_cos(Xp)
        if vp < best_val - 1e-12:
            best, best_val = Xp, vp
        # opportunistic: also try raw relaxed without LP
        # (already evaluated via LP, so skip)

    # final long polish of best
    try:
        fsec = max(1.0, min(14.0, HARD_END - time.time() - 1.0))
        if fsec > 0.5:
            Xf = _lp_polish(best, seconds=fsec)
            vf = _max_cos(Xf)
            if vf < best_val:
                best, best_val = Xf, vf
    except Exception:
        pass
    best = _normalize(best)
    return best
