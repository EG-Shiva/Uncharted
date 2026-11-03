import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = (G > m - tol)
    cnt = C.sum(axis=1)
    imax = G.max(axis=1)
    return cnt, imax, m, G

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, per_row, t0, t_lim):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        if (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if per_row:
            S = W.sum(axis=1, keepdims=True) + 1e-18
            Grad = (W @ X) / np.maximum(S, 1e-18)
            # normalize per-row average: already averaged
        else:
            Z = W.sum() + 1e-18
            Grad = (W @ X) / Z
            Grad = Grad * n  # match scale
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.sqrt((X * X).sum(axis=1, keepdims=True))
    return X

def _power_quench(X, steps, p0, p1, lr0, lr1, t0, t_lim):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    for t in range(1, steps + 1):
        if (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        p = p0 + (p1 - p0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, 0.0)
        np.maximum(G, 0.0, out=G)
        # weight p*G^{p-1}, avoid overflow: clip
        # for large p, G^p underflows for small G -> good (sparse)
        with np.errstate(over='ignore', under='ignore'):
            W = np.power(G, p - 1.0, out=np.zeros_like(G), where=G > 1e-12)
        W *= p
        np.fill_diagonal(W, 0.0)
        Grad = (W @ X) / max(n * 0.5, 1.0)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        # clip step norm to avoid blowup
        X = X - lr * step
        X /= np.sqrt((X * X).sum(axis=1, keepdims=True))
    return X

def _refine_hole(Y, p, iters=220, beta=90.0, lr0=0.2, lr1=0.005):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-18
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

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
    BUDGET = 174.0
    t_lim1 = t0 + 132.0
    n, d = 168, 8
    rng = np.random.RandomState(12345)
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
        best = rng.randn(n, d); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # quick LP-touch on start is NOT done; keep basin hopping first

    it = 0
    while time.time() < t_lim1:
        if time.time() + 3.0 > t0 + BUDGET:
            break
        it += 1
        X = best.copy()
        mode = rng.rand()
        inner_lim = t_lim1
        if mode < 0.30:
            # global kick + two-stage: power then softmax
            sigma = [0.006, 0.012, 0.025, 0.05][rng.randint(4)]
            X = X + rng.randn(n, d) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            p0 = 4.0 + rng.rand() * 4.0
            p1 = 18.0 + rng.rand() * 22.0
            X = _power_quench(X, 450 + int(rng.randint(350)), p0, p1, 0.02, 0.004, t0, inner_lim)
            beta0 = 25.0 + rng.rand() * 30.0
            beta1 = 220.0 + rng.rand() * 380.0
            per_row = bool(rng.rand() < 0.5)
            lr0 = 0.004 + rng.rand() * 0.005
            X = _adam_quench(X, 900 + int(rng.randint(700)), beta0, beta1, lr0, lr0 * 0.1, per_row, t0, inner_lim)
        elif mode < 0.62:
            # hole surgery: delete overloaded / worst points, reinsert into holes
            cnt, imax, mcur, Gcur = _contacts(X, tol=5e-7)
            k = 1 + int(rng.randint(5))  # 1..5
            # victims: mostly worst + high contacts
            order_worst = np.argsort(imax)
            order_cnt = np.argsort(cnt)
            victims = set()
            for _ in range(k):
                r = rng.rand()
                if r < 0.55:
                    # worst remaining
                    for idx in order_worst[::-1]:
                        if idx not in victims:
                            victims.add(int(idx)); break
                elif r < 0.85:
                    for idx in order_cnt[::-1]:
                        if idx not in victims:
                            victims.add(int(idx)); break
                else:
                    v = int(rng.randint(n))
                    while v in victims:
                        v = int(rng.randint(n))
                    victims.add(v)
            victims = sorted(victims)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Y = X[keep]
            # find holes w.r.t Y
            n_probe = 16000
            P = rng.randn(n_probe, d); P /= np.linalg.norm(P, axis=1, keepdims=True)
            CH = 4000
            bs = np.empty(n_probe)
            for a0 in range(0, n_probe, CH):
                b0 = min(a0 + CH, n_probe)
                D = P[a0:b0] @ Y.T
                bs[a0:b0] = D.max(axis=1)
            order = np.argsort(bs)
            holes = []
            for idx in order:
                p = P[idx]
                ok = True
                for q in holes:
                    if abs(float(p @ q)) > 0.90:
                        ok = False; break
                if ok:
                    # refine against Y + already chosen holes
                    Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
                    pr = _refine_hole(Yext, p, iters=220, beta=70.0 + rng.rand() * 60.0)
                    holes.append(pr)
                    if len(holes) >= k:
                        break
            for vi, h in zip(victims, holes):
                X[vi] = h
            # small jitter on neighbours to open space
            X = X + rng.randn(n, d) * 0.004
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            beta0 = 35.0 + rng.rand() * 25.0
            beta1 = 280.0 + rng.rand() * 320.0
            per_row = bool(rng.rand() < 0.6)
            lr0 = 0.004 + rng.rand() * 0.004
            X = _adam_quench(X, 1100 + int(rng.randint(700)), beta0, beta1, lr0, lr0 * 0.12, per_row, t0, inner_lim)
        elif mode < 0.82:
            # rattler-swarm relocation
            cnt, imax, mcur, Gcur = _contacts(X, tol=1e-6)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) < 3:
                # fallback: lowest-contact points
                ratt = np.argsort(cnt)[:12]
            # shake rattlers hard, jammed core soft
            noise = rng.randn(n, d) * 0.006
            rr = np.zeros(n)
            rr[ratt] = 0.05 + rng.rand() * 0.06
            # also random subset of others small
            noise = noise + (rng.randn(n, d).T * rr).T * 2.0
            # with some prob replace a few rattlers by fresh holes
            if rng.rand() < 0.5 and len(ratt) >= 2:
                k2 = min(3, len(ratt))
                sel = rng.choice(ratt, k2, replace=False)
                Y = np.delete(X, sel, axis=0)
                P = rng.randn(9000, d); P /= np.linalg.norm(P, axis=1, keepdims=True)
                D = P @ Y.T
                bs = D.max(axis=1)
                for vi in sel:
                    pi = int(np.argmin(bs))
                    h = _refine_hole(Y, P[pi], iters=180)
                    X[vi] = h
                    # mask nearby probes
                    bs = np.minimum(bs, 10.0 * np.abs(P @ h))
            else:
                X = X + noise
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # quench: power brief then softmax
            X = _power_quench(X, 350, 6.0, 25.0, 0.015, 0.004, t0, inner_lim)
            beta0 = 30.0 + rng.rand() * 30.0
            beta1 = 250.0 + rng.rand() * 350.0
            X = _adam_quench(X, 1000 + int(rng.randint(600)), beta0, beta1, 0.005, 0.0005, True, t0, inner_lim)
        else:
            # local cap shake + per-row sharpening quench
            c = rng.randn(d); c /= np.linalg.norm(c)
            dots = X @ c
            k = 10 + int(rng.randint(14))
            idx = np.argsort(dots)[-k:]
            sigma = 0.04 + rng.rand() * 0.06
            Xn = X.copy()
            Xn[idx] = Xn[idx] + rng.randn(k, d) * sigma
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            X = Xn
            beta0 = 40.0; beta1 = 350.0 + rng.rand() * 250.0
            per_row = True
            X = _adam_quench(X, 1300 + int(rng.randint(500)), beta0, beta1, 0.006, 0.0006, per_row, t0, inner_lim)
        if not np.all(np.isfinite(X)):
            continue
        vq = _max_cos(X)
        if vq + 1e-12 < best_val:
            best = X / np.linalg.norm(X, axis=1, keepdims=True)
            best_val = vq
    remaining = (t0 + BUDGET) - time.time()
    if remaining > 7.0:
        secs = min(remaining - 2.0, 40.0)
        try:
            pol = _lp_polish(best, secs)
            vp = _max_cos(pol)
            if vp < best_val and np.all(np.isfinite(pol)):
                best = pol; best_val = vp
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
