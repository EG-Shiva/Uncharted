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

def _contacts_deg(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    I, J = np.where(np.triu(G >= m - tol, 1))
    deg = np.zeros(X.shape[0], dtype=int)
    for a, b in zip(I, J):
        deg[a] += 1
        deg[b] += 1
    return m, deg

def _anneal_joint(X, steps, beta0, beta1, lr0, mu):
    X = _norm(X.copy())
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

def _anneal_frozen(X, src, steps, beta0, beta1, lr0, mu):
    X = _norm(X.copy())
    n = X.shape[0]
    mask = np.zeros(n)
    mask[np.asarray(src, dtype=int)] = 1.0
    mask = mask[:, None]
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
        # avoid overflow: beta*(G-gmax) <=0
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / S
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        V = mu * V + Gt
        vd = (V * X).sum(axis=1, keepdims=True)
        V = V - vd * X
        X = X - lr * mask * V
        X = _norm(X)
    return X

def _lp_polish_deadline(X, deadline, r0=8e-4):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = _norm(np.asarray(X, dtype=float).copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    Xb = X.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() >= deadline:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap problem size for speed
        if p > 6000:
            # keep most tight pairs
            order = np.argsort(-c[sel])[:6000]
            I = I[order]; J = J[order]
            svals = c[sel][order]
            p = len(I)
        else:
            svals = c[sel]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-svals, A_eq=A_eq, b_eq=np.zeros(n),
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

def _find_holes(base_pts, k, rng, nsamp=15000, sep=0.88):
    S = _norm(rng.standard_normal((nsamp, 4)))
    M = S @ base_pts.T
    m = M.max(axis=1)
    order = np.argsort(m)
    holes = []
    for idx in order:
        s = S[idx]
        ok = True
        for h in holes:
            if float(s @ h) > sep:
                ok = False
                break
        if ok:
            holes.append(s.copy())
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
    HOP_END = t0 + 154.0
    rng = np.random.default_rng(29251)
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
    if rec is not None:
        try:
            if _max_cos(rec) < _max_cos(best):
                best = rec.copy()
        except Exception:
            pass
    best_val = _max_cos(best)
    Xbest = best.copy()

    Xcur = Xbest.copy()
    cur_val = best_val

    T0, T1 = 8e-4, 2e-6

    it = 0
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - t0
        total = max(1.0, HOP_END - t0)
        frac = min(1.0, max(0.0, elapsed / total))
        # exponential cooling
        Temp = T0 * ((T1 / T0) ** frac)

        Xc = Xcur.copy()
        mode = it % 5
        if mode == 0:
            sigma = 0.018 if (it % 10 == 0) else 0.008
            noise = rng.standard_normal((n, d)) * sigma
            noise -= (noise * Xc).sum(axis=1, keepdims=True) * Xc
            Xc = _norm(Xc + noise)
            Xc = _anneal_joint(Xc, steps=300, beta0=60.0, beta1=550.0, lr0=0.025, mu=0.82)
        else:
            try:
                _, deg = _contacts_deg(Xc, tol=1e-7)
            except Exception:
                deg = np.zeros(n, dtype=int)
            order_hi = np.argsort(-deg, kind="stable")
            order_lo = np.argsort(deg, kind="stable")
            k = int(rng.integers(5, 12))
            src = []
            src.append(int(order_lo[0]))
            if n > 1:
                src.append(int(order_hi[0]))
            # add second low if distinct
            for c in order_lo[1:4]:
                if len(src) >= 3:
                    break
                c = int(c)
                if c not in src and rng.random() < 0.7:
                    src.append(c)
            pool = list(order_hi[:40]) + list(rng.integers(0, n, size=k * 3))
            for c in pool:
                c = int(c) % n
                if c not in src:
                    src.append(c)
                if len(src) >= k:
                    break
            src = src[:k]
            # holes w.r.t. remaining points
            mask_keep = np.ones(n, dtype=bool)
            mask_keep[np.asarray(src, dtype=int)] = False
            remaining = Xc[mask_keep]
            try:
                holes = _find_holes(remaining, len(src), rng, nsamp=15000, sep=0.86)
            except Exception:
                holes = _norm(rng.standard_normal((len(src), 4)))
            for s, h in zip(src, holes):
                j = rng.standard_normal(4) * 0.002
                j -= float(j @ h) * h
                Xc[s] = h + j
            Xc = _norm(Xc)
            # stage 1: frozen (only src moves)
            b0 = float(rng.uniform(30.0, 60.0))
            Xc = _anneal_frozen(Xc, src, steps=180, beta0=b0, beta1=350.0, lr0=0.05, mu=0.75)
            # stage 2: joint
            Xc = _anneal_joint(Xc, steps=240, beta0=50.0, beta1=700.0, lr0=0.028, mu=0.80)

        try:
            v = _max_cos(Xc)
        except Exception:
            continue

        # short LP evaluation if promising
        Xp = Xc
        vp = v
        if v < cur_val + 3e-3 and time.time() < HOP_END - 1.0:
            dl = min(time.time() + 0.8, HOP_END)
            try:
                Xp = _lp_polish_deadline(Xc, dl, r0=8e-4)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xc, v

        if vp < best_val - 1e-12:
            Xbest = Xp.copy()
            best_val = vp

        # Metropolis acceptance to cur
        if vp <= cur_val:
            Xcur = Xp.copy()
            cur_val = vp
        else:
            delta = vp - cur_val
            # cap to avoid underflow
            prob = np.exp(-delta / max(Temp, 1e-12))
            if rng.random() < prob:
                Xcur = Xp.copy()
                cur_val = vp
        # occasional restart from best if wandering too far
        if cur_val > best_val + 4e-3 and rng.random() < 0.3:
            Xcur = Xbest.copy()
            cur_val = best_val

        if time.time() >= HOP_END:
            break

    remain = DEAD - time.time()
    if remain > 2.5:
        try:
            Xbest = _lp_polish_deadline(Xbest, time.time() + min(remain - 1.0, 16.0), r0=1e-3)
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
