import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_softmax(X, steps, a0, a1, lr0):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        step = (m / c1) / (np.sqrt(v / c2) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def _adam_riesz(X, steps, s, lr0):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        lr = lr0 * (1.0 - 0.6 * t / steps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        D = 2.0 - 2.0 * G
        D[D < 1e-9] = 1e-9
        # W = s * D^(-s/2-1), clip to avoid overflow
        pw = -s / 2.0 - 1.0
        # cap D small values
        with np.errstate(over='ignore', divide='ignore'):
            W = s * np.power(D, pw)
        np.fill_diagonal(W, 0.0)
        # normalize to avoid huge grads
        med = np.median(W[W > 0]) if np.any(W > 0) else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0
        W = W / med
        W[W > 50.0] = 50.0
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize Gt scale
        sc = np.mean(np.linalg.norm(Gt, axis=1)) + 1e-12
        Gt = Gt / sc
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        step = (m / c1) / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        if not np.all(np.isfinite(X)):
            break
    return X

def _push(X, iters=100, eta0=0.015, band=0.012):
    X = X.copy()
    for it in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        W = (G > M - band).astype(float)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        eta = eta0 * (1.0 - 0.8 * it / max(iters, 1))
        X = X - eta * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _plane_rotation(d, a, b, theta):
    # Q = I + (cos-1)(aaT+bbT) + sin(baT-abT)
    c = np.cos(theta) - 1.0
    s = np.sin(theta)
    Q = np.eye(d)
    Q += c * (np.outer(a, a) + np.outer(b, b))
    Q += s * (np.outer(b, a) - np.outer(a, b))
    return Q

def _hemisphere_rotate(X, rng, theta):
    n, d = X.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    dots = X @ u
    thr = np.median(dots)
    mask = dots > thr
    if mask.sum() == 0 or mask.sum() == n:
        return X
    # random plane
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(d)
    b = b - np.dot(b, a) * a
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        return X
    b /= nb
    Q = _plane_rotation(d, a, b, theta)
    Y = X.copy()
    Y[mask] = (Q @ Y[mask].T).T
    return _norm(Y)

def _cap_regrow(X, rng, K):
    n, d = X.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    dots = X @ u
    kill = np.argsort(-dots)[:K]
    keep = np.ones(n, dtype=bool)
    keep[kill] = False
    cur = X[keep].copy()
    # greedy reinsert
    for _ in range(K):
        C = 6000
        cand = rng.standard_normal((C, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ cur.T
        md = D.max(axis=1)
        # pick among 8 best randomly
        idx = np.argpartition(md, 8)[:8]
        best8 = idx[np.argsort(md[idx])]
        pick = best8[rng.integers(8)] if rng.random() < 0.6 else best8[0]
        cur = np.vstack([cur, cand[pick:pick+1]])
    return _norm(cur + 0.008 * rng.standard_normal((n, d)))

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 6000:
            # subsample worst to keep LP cheap
            order = np.argsort(-c[sel])[:6000]
            I, J = I[order], J[order]
            p = 6000
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel][:p] if p != len(sel) else -c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    EXPL_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    Xbest = best.copy()
    cur = best.copy()
    cur_val = best_val
    trial = 0
    while time.time() < EXPL_END and time.time() < DEAD - 25:
        trial += 1
        elapsed = time.time() - t0
        frac = elapsed / max(EXPL_END - t0, 1.0)
        T_acc = 0.0022 * (1.0 - frac) + 0.0002
        r = rng.random()
        try:
            if r < 0.28:
                sigma = float(rng.uniform(0.10, 0.35))
                Xtry = cur + sigma * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
            elif r < 0.50:
                K = int(rng.integers(8, 21))
                Xtry = _cap_regrow(cur, rng, K)
            elif r < 0.72:
                theta = float(rng.uniform(0.15, 0.75))
                Xtry = _hemisphere_rotate(cur, rng, theta)
                Xtry = Xtry + 0.01 * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
            elif r < 0.82:
                # fresh random start
                Xtry = _norm(rng.standard_normal((n, d)))
            else:
                # tight-cluster kill
                Gc = cur @ cur.T
                np.fill_diagonal(Gc, -2.0)
                rowmax = np.max(Gc, axis=1)
                order = np.argsort(rowmax)
                K = int(rng.integers(5, 12))
                idx = order[-K:]
                Xtry = cur.copy()
                Xtry[idx] = rng.standard_normal((K, d))
                Xtry += 0.012 * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
        except Exception:
            continue
        # choose refiner
        try:
            if r < 0.5 and (trial % 2 == 0):
                s = float(rng.uniform(10.0, 24.0))
                steps = int(rng.integers(500, 950))
                lr0 = float(rng.uniform(0.008, 0.022))
                Xref = _adam_riesz(Xtry, steps, s, lr0)
            else:
                a0 = float(rng.uniform(18.0, 35.0))
                a1 = float(rng.uniform(400.0, 650.0))
                steps = int(rng.integers(600, 1100))
                lr0 = float(rng.uniform(0.010, 0.020))
                Xref = _adam_softmax(Xtry, steps, a0, a1, lr0)
            Xref = _push(Xref, iters=90, eta0=0.015, band=0.012)
            val = _max_cos(Xref)
        except Exception:
            continue
        if val < best_val - 1e-12:
            Xbest = Xref.copy()
            best_val = val
        # wandering acceptance on cur
        if val < cur_val or rng.random() < np.exp((cur_val - val) / max(T_acc, 1e-6)):
            cur = Xref.copy()
            cur_val = val
        # occasional reheat of cur to best region
        if trial % 40 == 0 and rng.random() < 0.3:
            # jump cur back near best with medium shake to intensify
            Xb = Xbest + 0.08 * rng.standard_normal((n, d))
            cur = _norm(Xb)
            cur_val = _max_cos(cur)
        if time.time() > EXPL_END:
            break
    # final polish of best
    remaining = DEAD - time.time() - 1.5
    if remaining > 4:
        try:
            Xbest = _lp_polish(Xbest, min(remaining - 2.0, 12.0), deadline=DEAD - 2.0)
            best_val = _max_cos(Xbest)
        except Exception:
            pass
    t_left = DEAD - time.time() - 0.5
    if t_left > 5:
        for _ in range(3):
            if time.time() > DEAD - 2.0:
                break
            try:
                Xt = Xbest + 3e-4 * rng.standard_normal((n, d))
                Xt = _norm(Xt)
                Xt = _lp_polish(Xt, 1.2, deadline=DEAD - 1.0)
                vt = _max_cos(Xt)
                if vt < best_val:
                    Xbest, best_val = Xt, vt
            except Exception:
                break
    Xbest = _norm(Xbest)
    return Xbest
