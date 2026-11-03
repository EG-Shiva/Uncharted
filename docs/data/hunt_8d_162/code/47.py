import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_anneal(X, iters, a0, a1, lr0, lr1, noise0, noise1, mu, rng, stress_exp=0.7):
    n, d = X.shape
    V = np.zeros_like(X)
    for t in range(iters):
        f = t / max(1, iters - 1)
        a = a0 * ((a1 / a0) ** f) if a1 != a0 else a0
        lr = lr0 * ((lr1 / lr0) ** f) if lr1 != lr0 else lr0
        if noise0 <= 0:
            nm = 0.0
        else:
            if noise1 is None or noise1 <= 0:
                nm = noise0 * (1.0 - f)
            else:
                nm = noise0 * ((noise1 / noise0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(a * (G - m))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True)
        rmean = float(row.mean()) + 1e-12
        H = W @ X
        WG = (W * G).sum(axis=1, keepdims=True)
        Dir = H - WG * X
        Dir_avg = Dir / np.maximum(row, 1e-30)
        w = (row / rmean) ** stress_exp
        Target = Dir_avg * w
        # sanitize
        if not np.all(np.isfinite(Target)):
            Target = np.zeros_like(X)
        V = mu * V + Target
        vmx = float(np.linalg.norm(V, axis=1).max())
        if vmx > 1e-14 and np.isfinite(vmx):
            X = X - V * (lr / vmx)
        if nm > 0:
            Nr = rng.randn(n, d)
            nd = (Nr * X).sum(axis=1, keepdims=True)
            Nr = Nr - nd * X
            X = X + nm * Nr
        X = _norm(X)
    return X

def _stress(X, a=60.0):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(G.max())
    W = np.exp(a * (G - m))
    np.fill_diagonal(W, 0.0)
    row = W.sum(axis=1)
    maxd = G.max(axis=1)
    return row, maxd, G, m

def _largest_holes(X_fixed, k, rng, n_cand=30000, sep=0.60):
    d = X_fixed.shape[1]
    C = _norm(rng.randn(n_cand, d))
    dots = C @ X_fixed.T
    scores = dots.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= k:
            break
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > sep:
                ok = False
                break
        if ok:
            picked.append(v.copy())
    while len(picked) < k:
        v = rng.randn(d)
        v = v / (np.linalg.norm(v) + 1e-12)
        picked.append(v)
    P = np.array(picked)
    # local refinement: push away from fixed set
    for _ in range(60):
        D = P @ X_fixed.T
        mrow = D.max(axis=1, keepdims=True)
        W = np.exp(40.0 * (D - mrow))
        Ws = W.sum(axis=1, keepdims=True) + 1e-12
        Gdir = (W @ X_fixed) / Ws
        # tangent component, move away
        dot = (Gdir * P).sum(axis=1, keepdims=True)
        Gt = Gdir - dot * P
        P = _norm(P - 0.05 * Gt)
    return P

def _lp_polish(X, seconds, t_deadline=None, r0=0.005, rmax=0.05, cap=3000):
    X = _norm(np.array(X, dtype=float))
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iuI, iuJ]
        m = float(cf.max())
        if m < best:
            best = m
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            # keep tightest cap
            idx = np.argpartition(-cf, cap)[:cap]
            I = iuI[idx]; J = iuJ[idx]
            c = cf[idx]
        else:
            I = iuI[sel]; J = iuJ[sel]
            c = cf[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y; best = v
            radius = min(radius * 1.5, rmax)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.RandomState(20261103)
    Xb = None; sb = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (162, 8):
                v = _norm(v)
                s = _max_cos(v)
                if s < sb:
                    sb = s; Xb = v.copy()
        except Exception:
            pass
    if Xb is None:
        Xb = _norm(rng.randn(162, 8))
        sb = _max_cos(Xb)
    n, d = 162, 8
    X_best = Xb.copy(); s_best = float(sb)
    X_cur = Xb.copy(); s_cur = float(sb)

    trial = 0
    # exploration with smooth annealing only + occasional short LP
    while time.time() < deadline - 38.0 and trial < 600:
        trial += 1
        mode = trial % 7
        Xt = None
        try:
            base = X_best if (trial % 3 != 0) else X_cur
            Xt = base.copy()
            if mode == 0:  # small global shake
                Xt = _norm(Xt + rng.randn(n, d) * 0.012)
                Xt = _smooth_anneal(Xt, 700, 25, 350, 0.025, 0.002, 0.004, 0.0, 0.85, rng)
                Xt = _smooth_anneal(Xt, 400, 100, 900, 0.008, 0.001, 0.0, 0.0, 0.88, rng)
            elif mode == 1:  # medium global shake
                Xt = _norm(Xt + rng.randn(n, d) * 0.030)
                Xt = _smooth_anneal(Xt, 800, 20, 300, 0.030, 0.002, 0.006, 0.0, 0.85, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 1000, 0.008, 0.0008, 0.0, 0.0, 0.88, rng)
            elif mode == 2:  # overload -> hole relocation small k
                row, maxd, _, _ = _stress(Xt, a=60.0)
                k = int(rng.choice([3, 4, 5, 6]))
                over = np.argsort(-row)[:max(12, 2*k)]
                sel = rng.choice(over, size=k, replace=False)
                keep = np.ones(n, dtype=bool); keep[sel] = False
                Xf = Xt[keep]
                holes = _largest_holes(Xf, k, rng, n_cand=30000, sep=0.60)
                Xt[sel] = holes
                Xt = _norm(Xt)
                Xt = _smooth_anneal(Xt, 900, 18, 280, 0.030, 0.002, 0.005, 0.0, 0.85, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 1000, 0.007, 0.0008, 0.0, 0.0, 0.88, rng)
            elif mode == 3:  # overload -> hole larger k
                row, maxd, _, _ = _stress(Xt, a=60.0)
                k = int(rng.choice([8, 10, 12]))
                over = np.argsort(-row)[:max(20, 2*k)]
                sel = rng.choice(over, size=k, replace=False)
                keep = np.ones(n, dtype=bool); keep[sel] = False
                Xf = Xt[keep]
                holes = _largest_holes(Xf, k, rng, n_cand=25000, sep=0.55)
                Xt[sel] = holes
                Xt = _norm(Xt)
                Xt = _smooth_anneal(Xt, 1000, 15, 250, 0.035, 0.002, 0.007, 0.0, 0.84, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 900, 0.007, 0.0008, 0.0, 0.0, 0.88, rng)
            elif mode == 4:  # cluster shake around overloaded center
                row, maxd, _, _ = _stress(Xt, a=50.0)
                probs = row / (row.sum() + 1e-12)
                c = rng.choice(n, p=probs)
                dots = Xt @ Xt[c]
                nn = np.argsort(-dots)[:int(rng.choice([10, 14, 18]))]
                sig = float(rng.choice([0.08, 0.12, 0.16]))
                Xt[nn] = _norm(Xt[nn] + rng.randn(len(nn), d) * sig)
                Xt = _smooth_anneal(Xt, 800, 20, 320, 0.028, 0.002, 0.004, 0.0, 0.85, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 1000, 0.007, 0.0008, 0.0, 0.0, 0.88, rng)
            elif mode == 5:  # low-contact + rattler reshuffle
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                mx = G.max(axis=1)
                # low maxd = isolated, high stress = overloaded; mix
                row, _, _, _ = _stress(Xt, a=60.0)
                low = np.argsort(mx)[:10]
                high = np.argsort(-row)[:10]
                sel = np.unique(np.concatenate([rng.choice(low, size=3, replace=False), rng.choice(high, size=3, replace=False)]))
                Xt[sel] = _norm(Xt[sel] + rng.randn(len(sel), d) * 0.10)
                Xt = _smooth_anneal(Xt, 800, 22, 300, 0.028, 0.002, 0.005, 0.0, 0.85, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 1000, 0.007, 0.0008, 0.0, 0.0, 0.88, rng)
            else:  # random subset heavy shake
                k = int(rng.choice([6, 9, 13]))
                sel = rng.choice(n, size=k, replace=False)
                Xt[sel] = _norm(Xt[sel] + rng.randn(k, d) * 0.18)
                Xt = _smooth_anneal(Xt, 900, 15, 250, 0.032, 0.002, 0.006, 0.0, 0.84, rng)
                Xt = _smooth_anneal(Xt, 400, 120, 900, 0.007, 0.0008, 0.0, 0.0, 0.88, rng)
        except Exception:
            continue
        if Xt is None:
            continue
        try:
            s = _max_cos(Xt)
        except Exception:
            continue
        # Metropolis move for cur to diversify basins
        if s < s_cur or rng.rand() < np.exp(-(s - s_cur) / 0.004):
            X_cur = Xt.copy(); s_cur = float(s)
        if s < s_best + 0.004 and time.time() < deadline - 40.0:
            # short LP to probe basin true potential
            try:
                Xp = _lp_polish(Xt, seconds=1.2, t_deadline=deadline - 36.0, r0=0.004, rmax=0.03, cap=2500)
                sp = _max_cos(Xp)
                if sp < s_best:
                    X_best = Xp.copy(); s_best = float(sp)
                    X_cur = Xp.copy(); s_cur = float(sp)
            except Exception:
                if s < s_best:
                    X_best = Xt.copy(); s_best = float(s)
        else:
            if s < s_best:
                X_best = Xt.copy(); s_best = float(s)
        if time.time() > deadline - 38.0:
            break
    # final thorough polish of best
    remain = deadline - time.time() - 1.0
    if remain > 5.0:
        try:
            X_best = _lp_polish(X_best, seconds=min(remain, 34.0), t_deadline=deadline - 0.5, r0=0.003, rmax=0.05, cap=3500)
        except Exception:
            pass
    X_best = _norm(X_best)
    return X_best
