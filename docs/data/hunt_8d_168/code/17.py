import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm, rng=None):
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

def _hard_polish(X, steps=350, lr0=0.004, lr1=0.00015, tol0=3e-4, tol1=2e-6):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        M = G > (m - tol)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        Grad = (M.astype(np.float64) @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize per-row to avoid overloaded domination
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        # only move points that are active
        active = (cnt[:, 0] > 0.5)
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=220, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        m = g.max()
        w = np.exp(beta * (g - m))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _find_holes(Y, k, rng, d, n_probe=16000):
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 4000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    sel = []
    for idx in order:
        if len(sel) >= k:
            break
        p = P[idx]
        ok = True
        for q in sel:
            if abs(float(p @ q)) > 0.90:
                ok = False
                break
        if ok:
            sel.append(p.copy())
        if len(sel) < k and idx > 5000:
            # relax
            pass
    # if too strict, fill with best remaining
    j = 0
    while len(sel) < k:
        sel.append(P[order[j]].copy()); j += 1
    return sel

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
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
    n, d = 168, 8
    rng = np.random.RandomState(20261103)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    elites = [(best_val, best.copy())]
    it = 0
    deadline1 = t0 + PHASE1
    while time.time() < deadline1:
        it += 1
        if time.time() + 2.5 > t0 + BUDGET:
            break
        # pick source: mostly best, sometimes other elite
        if len(elites) > 1 and rng.rand() < 0.22:
            si = rng.randint(len(elites))
            src = elites[si][1]
        else:
            src = best
        X = src.copy()
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        m0 = G0.max()
        rowmax = G0.max(axis=1)
        u = rng.rand()
        if u < 0.34:
            # global Gaussian kick, scale random
            sigmas = (0.006, 0.012, 0.025, 0.045)
            sigma = sigmas[rng.randint(len(sigmas))]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif u < 0.66:
            # worst-to-hole reinsertion
            k = 1 + int(rng.randint(4))  # 1..4
            # choose worst by rowmax, mostly worst, sometimes random jammed
            if rng.rand() < 0.7:
                idx = np.argsort(rowmax)[-k:]
            else:
                # jammed = above m0-2e-3
                jam = np.where(rowmax > m0 - 2e-3)[0]
                if len(jam) >= k:
                    idx = rng.choice(jam, k, replace=False)
                else:
                    idx = np.argsort(rowmax)[-k:]
            mask = np.ones(n, dtype=bool); mask[idx] = False
            Y = X[mask]
            holes = _find_holes(Y, k, rng, d, n_probe=16000)
            for t2, j in enumerate(idx):
                h = _refine_hole(Y, holes[t2], iters=220, beta=70.0 + rng.rand()*60.0)
                X[j] = h
            # small jitter to neighbours to open space
            X += rng.randn(n, d) * 0.003
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # contact-break kick + rattler shuffle
            tol = 1.5e-3
            I_, J_ = np.where(G0 > m0 - tol)
            # keep i<j
            pairs = [(int(a), int(b)) for a, b in zip(I_, J_) if a < b]
            if len(pairs) == 0:
                pairs = [(int(np.argmax(G0.max(axis=1))), 0)]
            nc = 1 + int(rng.randint(2))  # 1..2 pairs
            chosen = [pairs[rng.randint(len(pairs))] for _ in range(nc)]
            inv = set()
            for (a, b) in chosen:
                inv.add(a); inv.add(b)
            inv = list(inv)
            for a, b in chosen:
                # push apart along difference
                push = 0.02 + rng.rand() * 0.03
                Xa = X[a] - X[b]; Xb = X[b] - X[a]
                # make tangent
                Xa = Xa - (Xa @ X[a]) * X[a]
                Xb = Xb - (Xb @ X[b]) * X[b]
                na = np.linalg.norm(Xa)+1e-12; nb = np.linalg.norm(Xb)+1e-12
                X[a] = X[a] + (push/na)*Xa + rng.randn(d)*0.008
                X[b] = X[b] + (push/nb)*Xb + rng.randn(d)*0.008
            # shuffle a few rattlers (lowest rowmax) to random spots
            nr = 1 + int(rng.randint(3))
            ridx = np.argsort(rowmax)[:nr]
            # with prob move rattlers to holes
            if rng.rand() < 0.5:
                mask2 = np.ones(n, dtype=bool); mask2[ridx] = False
                Y2 = X[mask2]
                holes2 = _find_holes(Y2, nr, rng, d, n_probe=12000)
                for t2, j in enumerate(ridx):
                    if rng.rand() < 0.7:
                        h = _refine_hole(Y2, holes2[t2], iters=150, beta=80.0)
                        X[j] = h
                    else:
                        v = rng.randn(d); v /= np.linalg.norm(v)
                        X[j] = v
            else:
                for j in ridx:
                    if rng.rand() < 0.5:
                        v = rng.randn(d); v /= np.linalg.norm(v)
                        X[j] = 0.7*X[j] + 0.3*v
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # extra tiny global noise
            X = X + rng.randn(n, d) * 0.004
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        # quench params randomized
        beta0 = 25.0 + rng.rand() * 35.0
        beta1 = 250.0 + rng.rand() * 350.0
        steps = 700 + int(rng.randint(800))
        lr0 = 0.003 + rng.rand() * 0.005
        lr1 = lr0 * (0.08 + rng.rand()*0.08)
        rownorm = (rng.rand() < 0.6)
        Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm)
        # hard subgradient polish for accurate basin value
        Xq = _hard_polish(Xq, steps=300, lr0=0.0035, lr1=0.00012)
        vq = _max_cos(Xq)
        if np.all(np.isfinite(Xq)) and vq + 1e-12 < best_val:
            best = Xq; best_val = vq
            elites.append((vq, Xq.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:4]
        elif np.all(np.isfinite(Xq)) and len(elites) < 4 and vq < best_val + 5e-5 and rng.rand() < 0.05:
            elites.append((vq, Xq.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:4]
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 40.0)
        try:
            polished = _lp_polish(best, secs)
            vp = _max_cos(polished)
            if vp < best_val and np.all(np.isfinite(polished)):
                best = polished; best_val = vp
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.RandomState(1).randn(bad.sum(), d)
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best[bad] = rr
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
