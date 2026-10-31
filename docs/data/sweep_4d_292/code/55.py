import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = x.shape
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac if beta1 > 0 else beta0
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        A = (W @ x) / np.maximum(r, 1e-30)
        Ad = np.sum(A * x, axis=1, keepdims=True)
        At = A - Ad * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        x = _norm(x)
    return x

def _lp_polish(x, deadline, r0=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    rng_c = np.arange(n * d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if not np.any(sel):
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + ar
        cj = J[:, None] * d + ar
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (rep, rng_c)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return xb

def _contacts_deg(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    sel = c >= m - tol
    I = iu[0][sel]; J = iu[1][sel]
    deg = np.zeros(X.shape[0], dtype=int)
    for ii, jj in zip(I, J):
        deg[ii] += 1; deg[jj] += 1
    return deg, m, I, J

def _deep_holes(X, nholes, nsample, rng):
    n, d = X.shape
    S = _norm(rng.standard_normal((nsample, d)))
    # chunk to save memory
    best_vals = np.empty(nsample)
    chunk = 4000
    for a in range(0, nsample, chunk):
        b = min(nsample, a + chunk)
        D = S[a:b] @ X.T
        best_vals[a:b] = D.max(axis=1)
    order = np.argsort(best_vals)
    holes = []
    for idx in order:
        v = S[idx]
        ok = True
        for h in holes:
            if float(v @ h) > 0.90:
                ok = False; break
        if ok:
            holes.append(v.copy())
            if len(holes) >= nholes:
                break
    if len(holes) < nholes:
        for idx in order:
            holes.append(S[idx].copy())
            if len(holes) >= nholes:
                break
    return np.array(holes)

def solve():
    t0 = time.time()
    TLIM = 170.0
    deadline_final = t0 + TLIM
    rng = np.random.default_rng(292146)
    n, d = 292, 4

    def _load(fn):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
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
    best_val = _max_cos(best)
    if rec is not None:
        rv = _max_cos(rec)
        if rv < best_val:
            best = rec.copy(); best_val = rv

    # initial polish of incumbent
    try:
        cand = _lp_polish(best, t0 + 10.0, r0=1e-3)
        cv = _max_cos(cand)
        if cv < best_val:
            best, best_val = cand, cv
    except Exception:
        pass

    it = 0
    # reserve last ~12s for final polish
    while time.time() < deadline_final - 13.0:
        it += 1
        if time.time() > deadline_final - 14.0:
            break
        Xc = best.copy()
        mode = it % 6
        try:
            deg, m, I, J = _contacts_deg(Xc, tol=2e-9)
        except Exception:
            deg = np.full(n, 6)
        order_hi = np.argsort(-deg)
        order_lo = np.argsort(deg)
        if mode == 0:
            # global small shake
            sigma = 0.012 if (it % 12 != 0) else 0.035
            Xc = _norm(Xc + sigma * rng.standard_normal((n, d)))
            steps, b0, b1 = 350, 40.0, 700.0
            l0, l1 = 0.03, 0.003
        elif mode == 1 or mode == 3:
            # transplant overloaded hubs + rattlers to deep holes
            k = 4 if mode == 1 else 7
            k = int(min(k + (it % 3), 10))
            hi = order_hi[:k * 2]
            # pick k among top-2k randomly
            sel_idx = rng.choice(hi, size=min(k, len(hi)), replace=False)
            # also include rattlers
            ratt = np.where(deg <= 1)[0]
            if len(ratt) > 0:
                extra = ratt[:min(2, len(ratt))]
                sel_idx = np.unique(np.concatenate([sel_idx, extra]))
            nh = len(sel_idx)
            holes = _deep_holes(Xc, nh, 22000, rng)
            for t, idx in enumerate(sel_idx):
                j = rng.integers(0, 1000000) % nh if nh > 0 else 0
                # use hole t with small jitter
                v = holes[t % nh] + 0.01 * rng.standard_normal(d)
                Xc[idx] = v / max(np.linalg.norm(v), 1e-12)
            # small global jitter
            Xc = _norm(Xc + 0.004 * rng.standard_normal((n, d)))
            steps, b0, b1 = 500, 30.0, 900.0
            l0, l1 = 0.035, 0.003
        elif mode == 2:
            # subset reinit random
            k = int(rng.integers(4, 11))
            sel_idx = rng.choice(n, size=k, replace=False)
            for idx in sel_idx:
                v = rng.standard_normal(d)
                Xc[idx] = v / max(np.linalg.norm(v), 1e-12)
            Xc = _norm(Xc + 0.006 * rng.standard_normal((n, d)))
            steps, b0, b1 = 450, 35.0, 800.0
            l0, l1 = 0.032, 0.003
        elif mode == 4:
            # cluster surgery: pick random point, move its contacts
            center = int(rng.integers(0, n))
            dists = Xc @ Xc[center]
            nbrs = np.argsort(-dists)[:8]
            holes = _deep_holes(Xc, len(nbrs), 16000, rng)
            for t, idx in enumerate(nbrs[1:]):
                if rng.random() < 0.7:
                    v = holes[t % len(holes)] + 0.015 * rng.standard_normal(d)
                    Xc[idx] = v / max(np.linalg.norm(v), 1e-12)
            steps, b0, b1 = 500, 30.0, 1000.0
            l0, l1 = 0.035, 0.0025
        else:
            # medium shake with hole-biased single fill
            sigma = 0.02
            Xc = _norm(Xc + sigma * rng.standard_normal((n, d)))
            # move worst hub to deepest hole
            try:
                holes = _deep_holes(Xc, 2, 12000, rng)
                Xc[int(order_hi[0])] = holes[0] / max(np.linalg.norm(holes[0]), 1e-12)
            except Exception:
                pass
            steps, b0, b1 = 400, 40.0, 700.0
            l0, l1 = 0.03, 0.003

        try:
            Xc = _anneal(Xc, steps, b0, b1, l0, l1, mu=0.85)
        except Exception:
            continue
        # quick LP
        try:
            Xc = _lp_polish(Xc, min(time.time() + 1.6, deadline_final - 12.5), r0=8e-4)
        except Exception:
            pass
        try:
            cv = _max_cos(Xc)
        except Exception:
            continue
        if cv < best_val:
            best, best_val = Xc, cv

    # final long polish
    try:
        cand = _lp_polish(best, deadline_final - 0.5, r0=1e-3)
        cv = _max_cos(cand)
        if cv < best_val:
            best, best_val = cand, cv
    except Exception:
        pass
    best = _norm(best)
    return best
