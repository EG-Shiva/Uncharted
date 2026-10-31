import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x, thresh):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = (g > m - thresh).sum(axis=1)
    return m, cnt, g

def _anneal(X, steps, b0, b1, lr0, lr1, mu, noise=0.0, rng=None, deadline=None):
    n, d = X.shape
    v = np.zeros_like(X)
    lb = np.log(b1 / b0) if b1 > b0 else 0.0
    for t in range(steps):
        if deadline is not None and (t & 127) == 0 and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        beta = b0 * np.exp(lb * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ X) / S
        dot = (A * X).sum(axis=1, keepdims=True)
        Gt = A - dot * X
        if noise > 0.0:
            N = rng.standard_normal((n, d))
            Nd = (N * X).sum(axis=1, keepdims=True)
            N -= Nd * X
            Gt = Gt + noise * (1.0 - frac) * N
        v = mu * v + Gt
        vd = (v * X).sum(axis=1, keepdims=True)
        v = v - vd * X
        X = X - lr * v
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X /= nr
    return X

def _greedy_holes(X_rem, k, rng, n_cand=15000):
    Xa = X_rem.copy()
    for _ in range(k):
        cand = rng.standard_normal((n_cand, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # batch to limit memory
        best_s = 2.0
        best_p = None
        bs = 3000
        for a in range(0, n_cand, bs):
            cc = cand[a:a+bs]
            dd = cc @ Xa.T
            mm = dd.max(axis=1)
            j = int(np.argmin(mm))
            if float(mm[j]) < best_s:
                best_s = float(mm[j])
                best_p = cc[j]
        jp = best_p + 0.005 * rng.standard_normal(4)
        jp /= max(1e-12, np.linalg.norm(jp))
        Xa = np.vstack([Xa, jp.reshape(1, 4)])
    return Xa

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = c.max()
        sel = c > mval - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TL = 172.0
    explore_end = t0 + 134.0
    rng = np.random.default_rng(20261031)
    n, d = 292, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)) and nr.min() > 1e-9:
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        Xb = rng.standard_normal((n, d)); Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    else:
        Xb = cands[int(np.argmin([_max_cos(c) for c in cands]))].copy()
    best = _max_cos(Xb)
    Xbest = Xb.copy()
    it = 0
    while time.time() < explore_end:
        it += 1
        rem = explore_end - time.time()
        if rem < 1.2 and it > 5:
            break
        X = Xbest.copy()
        u = rng.random()
        if u < 0.45:
            # jammed-cluster excision
            try:
                m, cnt, G = _contact_counts(X, 1e-7)
            except Exception:
                cnt = np.zeros(n)
            # seed: most loaded point (random among top)
            order = np.argsort(-cnt + rng.random(n) * 0.3)
            k = int(rng.integers(6, 17))
            seed = int(order[0])
            dots = (X @ X[seed])
            # k nearest neighbours of seed (largest dots)
            idx = np.argsort(-dots)[:k]
            keep = np.ones(n, dtype=bool); keep[idx] = False
            Xrem = X[keep]
            Xnew = _greedy_holes(Xrem, k, rng, n_cand=12000)
            # shuffle order back: greedy appends at end, scatter
            # put new points at excised indices
            X[idx] = Xnew[-k:]
            # small shake of neighbours
            X = X + 0.004 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = int(rng.choice([1000, 1400, 2000]))
            b0 = float(rng.choice([15.0, 25.0, 40.0]))
            b1 = float(rng.choice([300.0, 450.0, 650.0]))
            lr0 = float(rng.choice([0.05, 0.08, 0.12]))
            lr1 = lr0 * 0.5
            mu = float(rng.choice([0.82, 0.86, 0.90]))
        elif u < 0.70:
            # rattler / worst hole move + global shake
            try:
                m, cnt, G = _contact_counts(X, 1e-7)
            except Exception:
                cnt = np.zeros(n)
            # kill worst k points (largest individual max)
            Gm = G.max(axis=1)
            k = int(rng.integers(2, 9))
            idx = np.argsort(-Gm + rng.random(n) * 0.01)[:k]
            R = rng.standard_normal((k, d)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            # farthest-hole candidates for those indices
            keep = np.ones(n, dtype=bool); keep[idx] = False
            Xrem = X[keep]
            Xnew = _greedy_holes(Xrem, k, rng, n_cand=10000)
            X[idx] = Xnew[-k:]
            sig = float(rng.choice([0.008, 0.02, 0.05]))
            X = X + sig * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = int(rng.choice([900, 1300, 1800]))
            b0, b1 = 20.0, 400.0
            lr0, lr1, mu = 0.08, 0.04, 0.86
            if rng.random() < 0.3:
                b0, b1 = 10.0, 200.0
                lr0, lr1, mu = 0.12, 0.06, 0.88
        else:
            # pure global shake + Langevin anneal (different basin)
            sig = float(rng.choice([0.01, 0.03, 0.08, 0.18]))
            X = X + sig * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            if rng.random() < 0.25:
                # occasional cap rotation for topology change
                ax = rng.standard_normal(4); ax /= np.linalg.norm(ax)
                proj = X @ ax
                th = float(rng.choice([0.85, 0.90, 0.95]))
                sel = proj > th
                if sel.sum() >= 2:
                    # random small rotation
                    A = rng.standard_normal((4, 4)) * 0.15
                    A = A - A.T
                    # exp approx I+A
                    Rm = np.eye(4) + A
                    # project to SO(4) via QR
                    Q, _ = np.linalg.qr(Rm)
                    if np.linalg.det(Q) < 0:
                        Q[:, 0] = -Q[:, 0]
                    X[sel] = X[sel] @ Q.T
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
            steps = int(rng.choice([1100, 1600, 2200]))
            b0 = float(rng.choice([12.0, 25.0, 35.0]))
            b1 = float(rng.choice([250.0, 400.0, 600.0]))
            lr0 = float(rng.choice([0.06, 0.10, 0.14]))
            lr1 = lr0 * 0.5
            mu = float(rng.choice([0.80, 0.86, 0.90]))
        # adapt steps to time left
        est = steps * 0.00022 + 0.05
        if est > rem - 0.5:
            steps = max(350, int((rem - 0.6) / 0.00022))
            if steps < 300:
                break
        nz = 0.0
        if rng.random() < 0.3:
            nz = 0.15
        X = _anneal(X, steps, b0, b1, lr0, lr1, mu, noise=nz, rng=rng, deadline=explore_end - 0.3)
        # short high-beta squeeze
        if time.time() < explore_end - 0.5:
            X = _anneal(X, 250, 300.0, 800.0, 0.02, 0.01, 0.85, deadline=explore_end - 0.2)
        val = _max_cos(X)
        if np.isfinite(val) and val < best:
            best = val
            Xbest = X.copy()
    # final squeeze + LP polish
    try:
        Xbest = _anneal(Xbest, 600, 200.0, 900.0, 0.02, 0.008, 0.85)
    except Exception:
        pass
    lp_deadline = t0 + TL - 1.0
    try:
        Xbest = _lp_polish(Xbest, lp_deadline)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xb
    return Xbest
