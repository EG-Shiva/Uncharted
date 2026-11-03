import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def normed(X):
    n = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
    n[n == 0] = 1.0
    return X / n

def refine_exp_mom(X, steps, T0, T1, lr0, lr1, mom=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(lt0 * (1.0 - frac) + lt1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        E = G - m
        E /= T
        np.clip(E, -50.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            continue
        E /= S
        Gr = E @ X
        Gr *= 2.0
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gr -= dot * X
        V *= mom
        V -= lr * Gr
        X += V
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X

def refine_adam_exp(X, steps, T0, T1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    M = np.zeros_like(X)
    Vc = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(lt0 * (1.0 - frac) + lt1 * frac)
        lr = lr0 * (1.0 - frac) + lr1 * frac
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = np.max(G)
        E = G - m
        E /= T
        np.clip(E, -50.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S == 0:
            continue
        E /= S
        Gt = (E @ X) * 2.0
        dot = np.einsum('ij,ij->i', Gt, X)[:, None]
        Gt -= dot * X
        M = b1 * M + (1 - b1) * Gt
        Vc = b2 * Vc + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        Mh = M / bc1
        Vh = Vc / bc2
        step = lr * Mh / (np.sqrt(Vh) + eps)
        step -= np.einsum('ij,ij->i', step, X)[:, None] * X
        X += -step
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X

_IU = None
def triu_idx(n):
    global _IU
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    return _IU

def refine_topk_adam(X, steps, k0, k1, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    iu = triu_idx(n)
    I0, J0 = iu
    M = np.zeros_like(X)
    Vc = np.zeros_like(X)
    P = I0.shape[0]
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        lr = lr0 * (1.0 - frac) + lr1 * frac
        k = int(round(k0 * (1.0 - frac) + k1 * frac))
        k = max(1, min(P, k))
        G = X @ X.T
        c = G[I0, J0]
        if k >= P:
            sel = np.arange(P)
        else:
            part = np.argpartition(-c, k - 1)[:k]
            sel = part
        Is = I0[sel]; Js = J0[sel]
        Gt = np.zeros_like(X)
        # accumulate mean gradient of selected cosines
        # Gt[i] += X[j]/k
        np.add.at(Gt, Is, X[Js])
        np.add.at(Gt, Js, X[Is])
        Gt /= float(k)
        dot = np.einsum('ij,ij->i', Gt, X)[:, None]
        Gt -= dot * X
        M = b1 * M + (1 - b1) * Gt
        Vc = b2 * Vc + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = lr * (M / bc1) / (np.sqrt(Vc / bc2) + eps)
        step -= np.einsum('ij,ij->i', step, X)[:, None] * X
        X -= step
        nr = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        nr[nr == 0] = 1.0
        X /= nr
    return X

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def best_hole_candidates(Xfrozen, n_try, rng, T=0.005, refine_iters=0):
    # return single point minimizing max dot to Xfrozen (with optional tiny smooth refine)
    d = Xfrozen.shape[1]
    Rc = rng.randn(n_try, d)
    Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
    scores = np.max(Rc @ Xfrozen.T, axis=1)
    bi = int(np.argmin(scores))
    q = Rc[bi].copy()
    # quick hill climb on sphere to minimize softmax to frozen
    for _ in range(refine_iters):
        dots = Xfrozen @ q
        mm = dots.max()
        w = np.exp((dots - mm) / T)
        ws = w.sum()
        if ws == 0 or not np.isfinite(ws):
            break
        w /= ws
        g = w @ Xfrozen  # (d,)
        g = g - np.dot(g, q) * q
        q = q - 0.05 * g
        q /= np.linalg.norm(q)
    return q

def solve():
    t0 = time.time()
    rng = np.random.RandomState(7)
    X0 = None; R0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = None
    try:
        R0 = np.load("record.npy")
    except Exception:
        R0 = None
    bestX = None; bestV = np.inf
    for c in (X0, R0):
        if c is None:
            continue
        try:
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = normed(c)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < bestV:
                bestV = v; bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12); bestX = normed(bestX); bestV = max_cos(bestX)
    n, d = 102, 12

    def contact_counts(X, tol=0.005):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        return (G > m - tol).sum(axis=1), m

    trial = 0
    # exploration until ~136s
    while time.time() - t0 < 136.0:
        trial += 1
        r = rng.rand()
        Xt = None
        mode = -1
        if r < 0.32:
            sigma = 10 ** rng.uniform(-2.4, -0.65)
            Xt = bestX + sigma * rng.randn(n, d)
            Xt = normed(Xt)
            mode = 0
        elif r < 0.52:
            # worst-point hole reinsertion (1-4 points)
            Xt = bestX.copy()
            cnt, m = contact_counts(bestX, 0.005)
            order = np.argsort(-cnt)
            # bias to overloaded + random
            if rng.rand() < 0.7:
                pool = order[:22]
            else:
                pool = np.arange(n)
                # weight by cnt
            k = 1 if rng.rand() < 0.5 else (2 if rng.rand() < 0.7 else (3 if rng.rand() < 0.8 else 4))
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            mask = np.ones(n, dtype=bool); mask[chosen] = False
            Xcur = Xt[mask].copy()
            # sequential greedy insertion
            new_pts = []
            for _ in chosen:
                q = best_hole_candidates(Xcur if len(new_pts) == 0 else np.vstack([Xcur] + new_pts), 4000, rng, refine_iters=60)
                new_pts.append(q[None, :])
            for idx, qp in zip(chosen, new_pts):
                Xt[idx] = qp[0]
            mode = 1
        elif r < 0.72:
            # subset shake
            Xt = bestX.copy()
            k = int(rng.randint(6, 24))
            idxs = rng.choice(n, size=k, replace=False)
            sigma = 10 ** rng.uniform(-1.6, -0.45)
            Xt[idxs] += sigma * rng.randn(k, d)
            Xt = normed(Xt)
            mode = 2
        elif r < 0.86:
            # coherent cap rotation
            Xt = bestX.copy()
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = bestX @ u
            k = int(rng.randint(8, 28))
            if rng.rand() < 0.5:
                idxs = np.argsort(-proj)[:k]
            else:
                idxs = np.argsort(proj)[:k]
            A = rng.randn(d, d)
            skew = (A - A.T) * (10 ** rng.uniform(-2.0, -0.8))
            Q = np.eye(d) + skew
            # orthonormalize small rotation approx via QR to keep near-orthogonal
            # use exp approx: keep as is then renormalize rows
            Xt[idxs] = Xt[idxs] @ Q.T
            # add small noise too
            Xt[idxs] += 10 ** rng.uniform(-2.5, -1.2) * rng.randn(k, d)
            Xt = normed(Xt)
            mode = 3
        else:
            # partial reinit: freeze core, randomize k points then optimize holes
            Xt = bestX.copy()
            k = int(rng.randint(4, 14))
            cnt, m = contact_counts(bestX, 0.005)
            order = np.argsort(-cnt)
            # pick mostly worst
            nw = min(k, 10)
            idxs = np.concatenate([order[:nw], rng.choice(n, size=k - nw, replace=False)]) if k > nw else order[:k]
            # randomize them to random holes sequentially
            mask = np.ones(n, dtype=bool); mask[idxs] = False
            Xf = Xt[mask]
            new_pts = []
            for _ in idxs:
                cur = Xf if len(new_pts) == 0 else np.vstack([Xf] + new_pts)
                q = best_hole_candidates(cur, 3000, rng, refine_iters=30)
                new_pts.append(q[None, :])
            for idx, qp in zip(idxs, new_pts):
                Xt[idx] = qp[0]
            mode = 4

        # choose refiner
        rr = rng.rand()
        try:
            if mode == 0:
                if rr < 0.5:
                    Xt = refine_exp_mom(Xt, 550, 0.014, 0.0011, 0.024, 0.002, 0.85 + rng.rand() * 0.08)
                    Xt = refine_exp_mom(Xt, 150, 0.0007, 0.00035, 0.001, 0.0005, 0.9)
                elif rr < 0.8:
                    Xt = refine_adam_exp(Xt, 550, 0.014, 0.001, 0.02, 0.002)
                    Xt = refine_topk_adam(Xt, 150, 400, 80, 0.004, 0.0008)
                else:
                    Xt = refine_topk_adam(Xt, 600, 1200, 120, 0.008, 0.001)
                    Xt = refine_exp_mom(Xt, 150, 0.0008, 0.00035, 0.001, 0.0004, 0.9)
            elif mode == 1:
                if rr < 0.6:
                    Xt = refine_adam_exp(Xt, 700, 0.011, 0.0008, 0.018, 0.0012)
                    Xt = refine_exp_mom(Xt, 150, 0.0007, 0.0003, 0.001, 0.0004, 0.9)
                else:
                    Xt = refine_exp_mom(Xt, 700, 0.013, 0.0009, 0.022, 0.0015, 0.88)
                    Xt = refine_topk_adam(Xt, 120, 300, 60, 0.003, 0.0006)
            elif mode == 2:
                if rr < 0.5:
                    Xt = refine_exp_mom(Xt, 650, 0.016, 0.001, 0.03, 0.002, 0.90)
                else:
                    Xt = refine_topk_adam(Xt, 650, 800, 100, 0.006, 0.0008)
                    Xt = refine_exp_mom(Xt, 120, 0.0007, 0.0003, 0.001, 0.0004, 0.9)
            elif mode == 3:
                Xt = refine_exp_mom(Xt, 650, 0.013, 0.0009, 0.024, 0.0016, 0.87 + rng.rand() * 0.06)
                if rng.rand() < 0.5:
                    Xt = refine_topk_adam(Xt, 120, 350, 70, 0.003, 0.0006)
            else:
                Xt = refine_adam_exp(Xt, 750, 0.012, 0.0008, 0.02, 0.0012)
                Xt = refine_exp_mom(Xt, 150, 0.0007, 0.0003, 0.0012, 0.0004, 0.9)
        except Exception:
            continue
        if not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(Xt)
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()
        if time.time() - t0 > 135.5:
            break

    # intensify around best
    if time.time() - t0 < 160:
        for _ in range(3):
            if time.time() - t0 > 158:
                break
            try:
                Xt = refine_adam_exp(bestX.copy(), 600, 0.007, 0.0004, 0.007, 0.0004)
                v = max_cos(Xt)
                if v < bestV and np.all(np.isfinite(Xt)):
                    bestV = v; bestX = Xt.copy()
            except Exception:
                pass
            try:
                Xt = refine_topk_adam(bestX.copy(), 300, 250, 50, 0.0025, 0.0004)
                v = max_cos(Xt)
                if v < bestV and np.all(np.isfinite(Xt)):
                    bestV = v; bestX = Xt.copy()
            except Exception:
                pass

    remaining = 178.0 - (time.time() - t0)
    if remaining > 7:
        try:
            bestX = lp_polish(bestX, min(remaining - 2.0, 24.0))
        except Exception:
            pass
    bestX = normed(bestX)
    return bestX
