import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if rownorm:
            S = W.sum(axis=1, keepdims=True) + 1e-12
            Grad = (W @ X) / S
        else:
            Z = W.sum()
            if Z == 0:
                continue
            Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        mhat = m / bc1; vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _hard_descent(X, steps=300, lr0=0.004, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        M = (G > (mx - tol)).astype(np.float64)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        active = (cnt[:, 0] > 0.5)
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=200, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        mx = g.max()
        w = np.exp(beta * (g - mx))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _find_void(Y, rng, n_probe=10000, n_top=8):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 4000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    cands = []
    for idx in order[:max(n_top * 4, 20)]:
        p = P[idx].copy()
        ok = True
        for q in cands:
            if abs(float(p @ q)) > 0.92:
                ok = False
                break
        if ok:
            cands.append(p)
        if len(cands) >= n_top:
            break
    best_p = cands[0]
    best_v = 1e9
    for p in cands:
        q = _refine_hole(Y, p, iters=200, beta=80.0 + rng.rand() * 40.0,
                         lr0=0.10 + rng.rand() * 0.06, lr1=0.003)
        vv = float((Y @ q).max())
        if vv < best_v:
            best_v = vv
            best_p = q
    return best_p

def _lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 175.0
    PHASE1 = 132.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(7)
        best = r0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    rng = np.random.RandomState(20240517)
    elite_vals = [best_val]
    elite_arrs = [best.copy()]
    def elite_add(v, X):
        for ev in elite_vals:
            if abs(v - ev) < 1e-12:
                return
        if len(elite_vals) < 4:
            elite_vals.append(v); elite_arrs.append(X.copy())
        else:
            k = int(np.argmax(elite_vals))
            if v < elite_vals[k]:
                elite_vals[k] = v; elite_arrs[k] = X.copy()
    it = 0
    stall = 0
    deadline1 = t0 + PHASE1
    while time.time() < deadline1:
        if time.time() + 2.5 > t0 + BUDGET:
            break
        it += 1
        if rng.rand() < 0.85:
            X = best.copy()
        else:
            X = elite_arrs[rng.randint(len(elite_arrs))].copy()
        r = rng.rand()
        if r < 0.28:
            sigmas = (0.006, 0.012, 0.025, 0.05)
            sigma = sigmas[rng.randint(len(sigmas))]
            if stall > 25:
                sigma *= 1.7
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 30.0 + rng.rand() * 35.0
            beta1 = 250.0 + rng.rand() * 550.0
            steps = 900 + int(rng.randint(900))
            lr0 = 0.003 + rng.rand() * 0.005
            rownorm = bool(rng.rand() < 0.5)
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.1, rownorm)
            Xq = _hard_descent(Xq, steps=220, lr0=0.003, lr1=0.0002)
        elif r < 0.62:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = 1 + int(rng.randint(3))
            if k >= 3:
                k = 3 if rng.rand() < 0.7 else 2
            u = rng.rand()
            if u < 0.50:
                idx = np.argsort(rowmax)[-k:]
            elif u < 0.78:
                idx = np.argsort(rowmax)[:k]
                if k > 1 and rng.rand() < 0.5:
                    idx[0] = int(np.argmax(rowmax))
            else:
                idx = rng.choice(n, k, replace=False)
            np.random.shuffle(idx) if hasattr(np.random, 'shuffle') else None
            for j in idx:
                others = np.delete(X, j, axis=0)
                y = _find_void(others, rng, n_probe=8000 + int(rng.randint(8000)), n_top=6)
                X[j] = y
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 40.0
            beta1 = 350.0 + rng.rand() * 450.0
            steps = 1000 + int(rng.randint(900))
            lr0 = 0.004 + rng.rand() * 0.004
            rownorm = bool(rng.rand() < 0.6)
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.12, rownorm)
            Xq = _hard_descent(Xq, steps=280, lr0=0.0035, lr1=0.00015)
        elif r < 0.85:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            thr = m - 0.002
            deg = (G > thr).sum(axis=1)
            hub = int(np.argmax(deg))
            if deg[hub] < 3:
                hub = int(np.argmax(G.max(axis=1)))
            order = np.argsort(-G[hub])
            nb = int(min(n - 1, max(4, deg[hub] + 2)))
            idx = order[:nb]
            idx = np.unique(np.concatenate([[hub], idx]))
            sigma = 0.03 + rng.rand() * 0.05
            noise = np.zeros((n, d))
            noise[idx] = rng.randn(len(idx), d) * sigma
            # occasionally also relocate a rattler to a void
            if rng.rand() < 0.35:
                rm = G.max(axis=1)
                jr = int(np.argmin(rm))
                others = np.delete(X, jr, axis=0)
                try:
                    y = _find_void(others, rng, n_probe=6000, n_top=4)
                    X[jr] = y
                except Exception:
                    pass
            X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 35.0 + rng.rand() * 25.0
            beta1 = 300.0 + rng.rand() * 500.0
            steps = 1100 + int(rng.randint(800))
            lr0 = 0.004 + rng.rand() * 0.004
            rownorm = bool(rng.rand() < 0.5)
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.1, rownorm)
            Xq = _hard_descent(Xq, steps=250, lr0=0.003, lr1=0.00015)
        else:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            # top 3 pairs, pick one at random to split
            flat = np.argpartition(G.ravel(), -6)[-6:]
            pick = flat[rng.randint(len(flat))]
            ii, jj = pick // n, pick % n
            for a, b in ((ii, jj), (jj, ii)):
                xa = X[a]; xb = X[b]
                dot = float(xa @ xb)
                direction = -(xb - dot * xa)
                nn = np.linalg.norm(direction)
                if nn > 1e-12:
                    direction /= nn
                    amp = 0.015 + rng.rand() * 0.03
                    X[a] = X[a] + direction * amp + rng.randn(d) * 0.008
            # small global jitter
            X = X + rng.randn(n, d) * 0.004
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 45.0
            beta1 = 400.0 + rng.rand() * 400.0
            steps = 1000 + int(rng.randint(800))
            lr0 = 0.004 + rng.rand() * 0.003
            Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.1, bool(rng.rand() < 0.5))
            Xq = _hard_descent(Xq, steps=250, lr0=0.003, lr1=0.00012)
        if not np.all(np.isfinite(Xq)):
            stall += 1
            continue
        vq = _max_cos(Xq)
        if vq < best_val + 2e-4 and vq > best_val - 5e-6:
            if (it % 9 == 0) and (time.time() + 4.0 < t0 + BUDGET) and (time.time() < deadline1 - 5):
                try:
                    Xs = _lp_polish(Xq, 1.0, r0=5e-4)
                    vs = _max_cos(Xs)
                    if np.all(np.isfinite(Xs)) and vs < vq:
                        Xq, vq = Xs, vs
                except Exception:
                    pass
        if vq + 1e-12 < best_val:
            best = Xq
            best_val = vq
            elite_add(vq, Xq)
            stall = 0
        else:
            if vq < max(elite_vals) + 1e-6:
                elite_add(vq, Xq)
            stall += 1
    # final hard polish of elites then LP on best
    try:
        for k in range(len(elite_arrs)):
            H = _hard_descent(elite_arrs[k].copy(), steps=350, lr0=0.002, lr1=0.00008, tol0=2e-4, tol1=1e-6)
            vh = _max_cos(H)
            if vh < elite_vals[k] and np.all(np.isfinite(H)):
                elite_vals[k] = vh; elite_arrs[k] = H
                if vh < best_val:
                    best_val = vh; best = H.copy()
    except Exception:
        pass
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 38.0)
        try:
            polished = _lp_polish(best, secs, r0=1e-3)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(int(bad.sum()), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
