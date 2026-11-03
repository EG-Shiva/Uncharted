import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _balanced_quench(X0, steps, b0, b1, lr0, lr1, lam=0.14, bl_frac=0.25, deadline=1e18):
    n, d = X0.shape
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    mad = np.zeros_like(X)
    vad = np.zeros_like(X)
    b1a = 0.9; b2a = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > deadline:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_m:
            best_m = gmax
            best = X.copy()
        E = np.exp(beta * (G - gmax))
        np.fill_diagonal(E, 0.0)
        Rs = E.sum(axis=1, keepdims=True) + 1e-30
        Grad_g = (E @ X) / Rs
        mx = G.max(axis=1, keepdims=True)
        bl = beta * bl_frac
        if bl < 8.0:
            bl = 8.0
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs2 = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs2
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 3.0 / nn)
        Gt = Gt * sc
        mad *= b1a; mad += (1.0 - b1a) * Gt
        vad *= b2a; vad += (1.0 - b2a) * (Gt * Gt)
        bc1 = 1.0 - b1a ** t; bc2 = 1.0 - b2a ** t
        step = (mad / bc1) / (np.sqrt(vad / bc2) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _hard_polish(X0, steps=250, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=2e-6):
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        M = (G > (mm - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, deadline, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            idx = np.argsort(c[sel])[-2800:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_hole(h, X, iters=40):
    h = h / np.linalg.norm(h)
    for k in range(iters):
        dots = X @ h
        m = float(dots.max())
        W = np.exp(35.0 * (dots - m))
        s = W.sum() + 1e-30
        g = (W @ X) / s
        dot = float(g @ h)
        gt = g - dot * h
        lr = 0.08 * (0.2 / 0.08) ** (k / max(iters - 1, 1))
        h = h - lr * gt
        nrm = np.linalg.norm(h)
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        h /= nrm
    return h

def _find_holes(X, k, rng, n_probe=25000):
    n, d = X.shape
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    S = P @ X.T
    cov = S.max(axis=1)
    order = np.argsort(cov)
    holes = []
    for idx in order:
        h = P[idx].copy()
        ok = True
        for q in holes:
            if float(h @ q) > 0.92:
                ok = False
                break
        if not ok:
            continue
        h = _refine_hole(h, X, iters=40)
        holes.append(h)
        if len(holes) >= k:
            break
    if len(holes) < k:
        extra = rng.randn(k - len(holes), d)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        for e in extra:
            holes.append(_refine_hole(e, X, iters=30))
    return np.array(holes[:k])

def solve():
    t0 = time.time()
    BUDGET = 175.0
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
    incumbent = best.copy(); inc_val = best_val

    def note(X):
        nonlocal best, best_val
        try:
            v = _max_cos(X)
        except Exception:
            return False
        if v < best_val and np.all(np.isfinite(X)):
            best_val = v; best = X.copy()
            return True
        return False

    # Phase 1: perturbed re-jams with balanced quench
    sigmas = [0.004, 0.012, 0.03, 0.06]
    for si, sigma in enumerate(sigmas):
        if time.time() > t0 + BUDGET - 95:
            break
        Xb = incumbent + rng.randn(n, d) * sigma
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
        dl = t0 + BUDGET - 60
        Xq = _balanced_quench(Xb, steps=600, b0=22.0, b1=520.0,
                              lr0=0.007, lr1=0.00045, lam=0.14, deadline=dl)
        note(Xq)
        Xh = _hard_polish(Xq, steps=180, lr0=0.0028, lr1=0.00012)
        note(Xh)

    # Phase 2: hole surgery on current best / incumbent
    bases = []
    try:
        bases.append(best.copy())
    except Exception:
        pass
    bases.append(incumbent.copy())
    for bi, base in enumerate(bases):
        if time.time() > t0 + BUDGET - 70:
            break
        for k in ([8, 4] if bi == 0 else [12, 6]):
            if time.time() > t0 + BUDGET - 60:
                break
            try:
                pm = _per_max(base)
                worst = np.argsort(-pm)[:k]
                holes = _find_holes(base, k, rng, n_probe=22000)
                Xh0 = base.copy()
                Xh0[worst] = holes
                Xh0 /= np.linalg.norm(Xh0, axis=1, keepdims=True)
            except Exception:
                continue
            dl = t0 + BUDGET - 45
            Xq = _balanced_quench(Xh0, steps=700, b0=25.0, b1=560.0,
                                  lr0=0.007, lr1=0.0004, lam=0.16, deadline=dl)
            note(Xq)
            # second jolt: small noise re-jam from this new basin
            if time.time() > t0 + BUDGET - 50:
                continue
            try:
                Xb2 = Xq + rng.randn(n, d) * 0.006
                Xb2 /= np.linalg.norm(Xb2, axis=1, keepdims=True)
                Xq2 = _balanced_quench(Xb2, steps=500, b0=25.0, b1=520.0,
                                       lr0=0.006, lr1=0.0004, lam=0.14, deadline=dl)
                note(Xq2)
            except Exception:
                pass

    # Phase 3: concentrate on best with longer quench + hard polish
    if time.time() < t0 + BUDGET - 35:
        try:
            cur = best.copy()
        except Exception:
            cur = incumbent.copy()
        for rep in range(2):
            if time.time() > t0 + BUDGET - 32:
                break
            sigma = 0.005 if rep == 0 else 0.010
            Xb = cur + rng.randn(n, d) * sigma
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
            Xq = _balanced_quench(Xb, steps=800, b0=28.0, b1=600.0,
                                  lr0=0.006, lr1=0.00035, lam=0.13,
                                  deadline=t0 + BUDGET - 26)
            note(Xq)
            try:
                Xhp = _hard_polish(Xq, steps=220, lr0=0.0026, lr1=0.00010)
                note(Xhp)
            except Exception:
                pass

    # Final LP squeeze
    remain = (t0 + BUDGET - 3.0) - time.time()
    if remain > 5:
        try:
            Xlp = _lp_polish(best, deadline=t0 + BUDGET - 3.0, r0=1.2e-3)
            note(Xlp)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    try:
        if _max_cos(best) > inc_val and np.all(np.isfinite(incumbent)):
            best = incumbent.copy()
    except Exception:
        pass
    return best
