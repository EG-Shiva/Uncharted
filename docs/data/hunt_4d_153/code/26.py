import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import orthogonal_procrustes

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if budget <= 0:
        return xb, best
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            xb = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb, best

def _anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    x = x.copy()
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    if steps <= 1:
        return x
    lrb = np.log(lr1 / lr0) / (steps - 1)
    beb = np.log(beta1 / beta0) / (steps - 1)
    for s in range(steps):
        beta = beta0 * np.exp(beb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        if not np.isfinite(Z) or Z == 0:
            break
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W += W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _mask(x, tol=1e-6):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    return (g > m - tol)

def _hamming(a, b):
    return int(np.count_nonzero(a != b))

def _fix_size(cand, n, rng):
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    # remove excess by deleting one end of closest pair
    while len(cand) > n:
        G = cand @ cand.T
        np.fill_diagonal(G, -2)
        idx = np.argmax(G)
        i, j = idx // len(cand), idx % len(cand)
        si = G[i].sum(); sj = G[j].sum()
        drop = i if si > sj else j
        cand = np.delete(cand, drop, axis=0)
    while len(cand) < n:
        # largest void fill: sample dirs, pick farthest from set
        S = rng.normal(size=(2000, 4))
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        dots = S @ cand.T
        md = dots.max(axis=1)
        best = np.argmin(md)
        cand = np.vstack([cand, S[best:best+1]])
    return cand

def _crossover(A, B, rng):
    n = A.shape[0]
    try:
        R, _ = orthogonal_procrustes(B, A)
    except Exception:
        R = np.eye(4)
    Bb = B @ R
    u = rng.normal(size=4); u /= np.linalg.norm(u)
    pa = A @ u; pb = Bb @ u
    Apos = A[pa > 0]; Bneg = Bb[pb <= 0]
    if len(Apos) == 0:
        Apos = A[:n//2]
    if len(Bneg) == 0:
        Bneg = Bb[:n//2]
    cand = np.vstack([Apos, Bneg])
    # jitter exact duplicates slightly
    cand += rng.normal(0, 1e-4, size=cand.shape)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cand = _fix_size(cand, n, rng)
    return cand

def _delete_to_n(x, n, tol=1e-5):
    # delete highest-load points
    while len(x) > n:
        G = x @ x.T
        np.fill_diagonal(G, -2)
        m = G.max()
        tight = G > m - tol
        cnt = tight.sum(axis=1)
        # tie-break by sum of dots
        s = G.sum(axis=1)
        score = cnt * 10 + (s - s.min()) / max(s.max()-s.min(),1e-12)
        drop = int(np.argmax(score))
        x = np.delete(x, drop, axis=0)
    return x

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        rec = np.load("best.npy")
    except Exception:
        rec = None
    try:
        pub = np.load("record.npy")
    except Exception:
        pub = None
    if rec is None and pub is None:
        rec = rng.normal(size=(n, d))
    base0 = rec if rec is not None else pub
    base0 = np.asarray(base0, dtype=float).reshape(n, d)
    base0 /= np.linalg.norm(base0, axis=1, keepdims=True)
    if pub is None:
        pub = base0.copy()

    pool_X = []
    pool_V = []
    pool_M = []
    best_val = 1e9
    best_X = base0.copy()

    def try_add(X, v):
        nonlocal best_val, best_X
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        if v < best_val:
            best_val = v
            best_X = X.copy()
        if v > 0.86:
            return False
        m = _mask(X, 1e-6)
        for k, om in enumerate(pool_M):
            if _hamming(m, om) < 80:
                if v < pool_V[k]:
                    pool_X[k] = X.copy(); pool_V[k] = v; pool_M[k] = m
                    return True
                return False
        pool_X.append(X.copy()); pool_V.append(v); pool_M.append(m)
        return True

    # seed pool with record (quick polish tiny)
    v0 = _max_cos(base0)
    try_add(base0, v0)
    # build diverse pool
    sigmas = [0.15, 0.2, 0.25, 0.18]
    # 1) large perturbations of record
    for si, sg in enumerate(sigmas):
        if time.time() - t0 > 60:
            break
        y = np.asarray(pub).reshape(n, d).copy()
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        y = y + rng.normal(0, sg, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _anneal(y, 350, 15, 350, 0.05, 0.005)
        y, v = _lp_polish(y, 2.0)
        try_add(y, v)
        if time.time() - t0 > HARD - 90:
            break
    # 2) independent random starts
    for k in range(4):
        if time.time() - t0 > 90:
            break
        if len(pool_X) >= 8:
            break
        y = rng.normal(size=(n, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _anneal(y, 450, 10, 400, 0.06, 0.004)
        y, v = _lp_polish(y, 2.0)
        try_add(y, v)
    # 3) overloaded squeeze-out
    for ov in [156, 158, 160]:
        if time.time() - t0 > 110:
            break
        if len(pool_X) >= 10:
            break
        y = rng.normal(size=(ov, d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _anneal(y, 300, 15, 300, 0.05, 0.005)
        y = _delete_to_n(y, n)
        y = _anneal(y, 200, 20, 350, 0.03, 0.005)
        y, v = _lp_polish(y, 1.5)
        try_add(y, v)

    # ensure pool sorted, keep best 10
    if len(pool_X) == 0:
        pool_X.append(best_X.copy()); pool_V.append(best_val); pool_M.append(_mask(best_X))
    # selection-crossover-polish
    it = 0
    while time.time() - t0 < HARD - 12:
        it += 1
        if len(pool_X) < 2:
            break
        # tournament selection
        idx = np.argsort(np.array(pool_V))
        # keep pool bounded
        if len(pool_X) > 10:
            keep = idx[:10]
            pool_X = [pool_X[i] for i in keep]
            pool_V = [pool_V[i] for i in keep]
            pool_M = [pool_M[i] for i in keep]
            idx = np.argsort(np.array(pool_V))
        # pick parents: bias to best but ensure diversity
        # parent1 from top-4, parent2 random
        a = int(rng.choice(idx[:min(4, len(idx))]))
        b = int(rng.integers(len(pool_X)))
        if b == a:
            b = int(rng.integers(len(pool_X)))
        A = pool_X[a]; B = pool_X[b]
        # alternate overload child occasionally
        if it % 7 == 0:
            ov = 157 + int(rng.integers(3))
            y = rng.normal(size=(ov, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _anneal(y, 200, 15, 250, 0.05, 0.008)
            y = _delete_to_n(y, n)
            child = y
        else:
            child = _crossover(A, B, rng)
        # anneal child lightly then polish
        # randomize anneal length a bit
        child = _anneal(child, 220, 20, 400, 0.04, 0.004)
        # small extra kick to avoid immediate fallback
        child += rng.normal(0, 0.003, size=child.shape)
        child /= np.linalg.norm(child, axis=1, keepdims=True)
        # short LP budget adapting to remaining time
        remain = HARD - 12 - (time.time() - t0)
        if remain < 1:
            break
        budg = min(1.8, remain / 3)
        child_p, v = _lp_polish(child, budg)
        try_add(child_p, v)
        # also consider raw child if polish worsened? no
        if best_val < 0.8485:
            break

    # final polish best
    remain = HARD - (time.time() - t0)
    if remain > 2:
        best_X, best_val = _lp_polish(best_X, min(remain - 1, 6.0))
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
