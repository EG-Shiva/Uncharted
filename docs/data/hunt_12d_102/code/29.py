import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def contact_set(X, tol=2e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    sel = (G[iu] > m - tol)
    return set(np.where(sel)[0].tolist()), m


def edit_dist(a, b):
    return len(a.symmetric_difference(b))


def smooth_refine(X, steps, T0, T1, lr0, lr1, mom=0.88):
    n, d = X.shape
    V = np.zeros_like(X)
    logT0 = np.log(T0)
    logT1 = np.log(T1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        T = np.exp(logT0 * (1.0 - frac) + logT1 * frac)
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


def build_soft_modes(X, K=40, Th=0.02, seed=0):
    n, d = X.shape
    N = n * d
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    np.fill_diagonal(G, 0.0)
    E = np.exp(np.clip((G - m) / Th, -50, 0))
    np.fill_diagonal(E, 0.0)
    Emat = E / Th
    Fmat = E / (Th * Th)
    H = np.zeros((N, N), dtype=np.float64)
    xa_xb = X.T  # not needed
    # fill blocks
    for a in range(d):
        xa = X[:, a]
        for b in range(d):
            xb = X[:, b]
            M = xb[:, None] * Fmat * xa[None, :]
            s = Fmat @ (xa * xb)
            # add diag s
            # M diagonal is 0, add s
            # use flat index add
            M = M.copy()
            np.fill_diagonal(M, s)
            if a == b:
                M += Emat
            rows = a + np.arange(n) * d
            cols = b + np.arange(n) * d
            H[np.ix_(rows, cols)] = M
    # curvature correction: subtract radial grad
    G2 = X @ X.T
    r = np.sum(Emat * G2, axis=1)
    for i in range(n):
        for a in range(d):
            p = i * d + a
            H[p, p] -= r[i]
    # tangent basis B
    rng0 = np.random.RandomState(seed)
    Zr = rng0.randn(d, d - 1)
    # orthonormalize Zr once
    Zr, _ = np.linalg.qr(Zr)
    NT = n * (d - 1)
    B = np.zeros((N, NT), dtype=np.float64)
    for i in range(n):
        x = X[i]
        proj = x @ Zr  # (d-1,)
        Y = Zr - np.outer(x, proj)
        Q, _ = np.linalg.qr(Y)
        B[i * d:(i + 1) * d, i * (d - 1):(i + 1) * (d - 1)] = Q
    tmp = H @ B
    Ht = B.T @ tmp
    # symmetrize
    Ht = 0.5 * (Ht + Ht.T)
    w, V = np.linalg.eigh(Ht)
    K = min(K, V.shape[1])
    Vlow = V[:, :K]
    Ulow = B @ Vlow  # N x K ambient tangent modes, norm 1
    return Ulow


def random_rotation12(rng, angle):
    # random 2-plane rotation in R^12
    A = rng.randn(12, 12)
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]
    v = Q[:, 1]
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.eye(12) + (c - 1.0) * (np.outer(u, u) + np.outer(v, v)) + s * (np.outer(v, u) - np.outer(u, v))
    return R


def kmeans_assign(X, ncl, rng, iters=8):
    n = X.shape[0]
    idx = rng.choice(n, size=ncl, replace=False)
    C = X[idx].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for k in range(ncl):
            m = assign == k
            if np.any(m):
                c = X[m].mean(axis=0)
                nn = np.linalg.norm(c)
                if nn > 0:
                    c /= nn
                C[k] = c
            else:
                C[k] = X[rng.randint(n)]
                C[k] /= np.linalg.norm(C[k])
    return assign


def solve():
    t0 = time.time()
    rng = np.random.RandomState(0)
    X0 = None
    R0 = None
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = None
    try:
        R0 = np.load("record.npy")
    except Exception:
        R0 = None
    bestX = None
    bestV = np.inf
    for c in (X0, R0):
        if c is None:
            continue
        try:
            c = np.asarray(c, dtype=np.float64)
            if c.shape != (102, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            v = max_cos(c)
            if v < bestV:
                bestV = v
                bestX = c.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.randn(102, 12)
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = max_cos(bestX)
    n, d = 102, 12
    best_edges, _ = contact_set(bestX)
    Ulow = None
    pool = []  # list of (v, X)
    trial = 0
    HOP_END = t0 + 138.0
    # initial soft modes
    try:
        if time.time() - t0 < 120:
            Ulow = build_soft_modes(bestX, K=40, Th=0.02, seed=0)
    except Exception:
        Ulow = None

    while time.time() < HOP_END:
        trial += 1
        time_left = HOP_END - time.time()
        if time_left < 1.0:
            break
        use_spectral = (trial % 2 == 1)
        # occasionally restart from pool diverse member
        baseX = bestX
        baseV = bestV
        if len(pool) > 0 and rng.rand() < 0.15:
            # pick random pool member within 0.002
            pi = rng.randint(len(pool))
            baseX = pool[pi][1]
        Xt = None
        if use_spectral and Ulow is not None:
            try:
                Kuse = int(rng.randint(20, 41))
                Kuse = min(Kuse, Ulow.shape[1])
                coeff = rng.randn(Kuse)
                nn = np.linalg.norm(coeff)
                if nn == 0:
                    coeff[0] = 1.0
                    nn = 1.0
                coeff /= nn
                U = Ulow[:, :Kuse] @ coeff  # N,
                amp = float(rng.uniform(0.02, 0.15))
                # 10% larger jumps
                if rng.rand() < 0.1:
                    amp = float(rng.uniform(0.15, 0.25))
                dx = amp * np.sqrt(n) * U
                Xt = baseX + dx.reshape(n, d)
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            except Exception:
                Xt = None
            # refresh Hessian periodically or after improvement
            if trial % 8 == 0:
                try:
                    Ulow = build_soft_modes(bestX, K=40, Th=0.02, seed=trial)
                except Exception:
                    pass
        if Xt is None and not use_spectral:
            # cluster rotation move
            try:
                # choose split: half PC1, half k-means
                if rng.rand() < 0.3:
                    # PC1 split
                    Xc = baseX - baseX.mean(axis=0, keepdims=True)
                    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                    pc1 = Xc @ Vt[0]
                    med = np.median(pc1)
                    mask = pc1 > med
                    # randomly pick side
                    if rng.rand() < 0.5:
                        cl_mask = mask
                    else:
                        cl_mask = ~mask
                    # ensure nontrivial
                    if cl_mask.sum() < 5 or cl_mask.sum() > 97:
                        raise ValueError("bad split")
                    # treat as cluster
                    ang = float(rng.uniform(0.03, 0.2))
                    if rng.rand() < 0.5:
                        # coordinate plane
                        a, b = rng.choice(d, size=2, replace=False)
                        R2 = np.eye(2) * np.cos(ang)
                        # build rotation on coords
                        Xt = baseX.copy()
                        ca, sa = np.cos(ang), np.sin(ang)
                        xa = Xt[cl_mask, a].copy()
                        xb = Xt[cl_mask, b].copy()
                        Xt[cl_mask, a] = ca * xa - sa * xb
                        Xt[cl_mask, b] = sa * xa + ca * xb
                    else:
                        R = random_rotation12(rng, ang)
                        Xt = baseX.copy()
                        Xt[cl_mask] = Xt[cl_mask] @ R.T
                        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                else:
                    ncl = int(rng.randint(4, 7))
                    assign = kmeans_assign(baseX, ncl, rng, iters=8)
                    kpick = int(rng.randint(ncl))
                    cl_mask = assign == kpick
                    if cl_mask.sum() < 4 or cl_mask.sum() > 90:
                        # fallback random subset
                        k = int(rng.randint(8, 30))
                        cl_mask = np.zeros(n, dtype=bool)
                        cl_mask[rng.choice(n, size=k, replace=False)] = True
                    ang = float(rng.uniform(0.03, 0.2))
                    if rng.rand() < 0.4:
                        a, b = rng.choice(d, size=2, replace=False)
                        Xt = baseX.copy()
                        ca, sa = np.cos(ang), np.sin(ang)
                        xa = Xt[cl_mask, a].copy()
                        xb = Xt[cl_mask, b].copy()
                        Xt[cl_mask, a] = ca * xa - sa * xb
                        Xt[cl_mask, b] = sa * xa + ca * xb
                    else:
                        R = random_rotation12(rng, ang)
                        Xt = baseX.copy()
                        Xt[cl_mask] = Xt[cl_mask] @ R.T
                    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            except Exception:
                Xt = None
        if Xt is None:
            # fallback small gaussian kick
            sigma = 10 ** float(rng.uniform(-2.0, -1.0))
            Xt = baseX + sigma * rng.randn(n, d)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        # quench with Adam-like momentum
        try:
            steps = 450
            # vary annealing slightly
            T0p = float(10 ** rng.uniform(-1.9, -1.6))
            T1p = 0.001
            lr0 = float(rng.uniform(0.02, 0.035))
            lr1 = 0.002
            mom = 0.88 + rng.rand() * 0.05
            Xt = smooth_refine(Xt, steps, T0p, T1p, lr0, lr1, mom)
        except Exception:
            continue
        if not np.all(np.isfinite(Xt)):
            continue
        v = max_cos(Xt)
        # short LP only for promising
        if v < 0.269 and v < bestV + 0.0015:
            rem = HOP_END - time.time()
            if rem > 3.0:
                try:
                    Xt2 = lp_polish(Xt, min(1.2, rem - 1.0))
                    v2 = max_cos(Xt2)
                    if np.all(np.isfinite(Xt2)) and v2 < v:
                        Xt, v = Xt2, v2
                except Exception:
                    pass
        if v < bestV - 1e-12:
            bestV = v
            bestX = Xt.copy()
            best_edges, _ = contact_set(bestX)
            # refresh soft modes at new basin
            try:
                if time.time() < HOP_END - 5:
                    Ulow = build_soft_modes(bestX, K=40, Th=0.02, seed=1000 + trial)
            except Exception:
                pass
            # clear pool far above?
            pool = [p for p in pool if p[0] < bestV + 0.002]
        elif v < bestV + 0.002:
            try:
                es, _ = contact_set(Xt)
                ed = edit_dist(es, best_edges)
                if ed > 40:  # substantially different topology
                    pool.append((v, Xt.copy()))
                    if len(pool) > 6:
                        pool.sort(key=lambda z: z[0])
                        pool = pool[:6]
            except Exception:
                pass

    # final polish
    try:
        if time.time() - t0 < 165:
            Xt = smooth_refine(bestX.copy(), 600, 0.008, 0.0005, 0.008, 0.0005, 0.90)
            v = max_cos(Xt)
            if v < bestV and np.all(np.isfinite(Xt)):
                bestV = v
                bestX = Xt
    except Exception:
        pass
    remaining = 178.0 - (time.time() - t0)
    if remaining > 5:
        try:
            bestX = lp_polish(bestX, min(remaining - 2.0, 20.0))
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
