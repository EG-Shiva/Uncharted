import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _quench(X0, steps, beta0, beta1, lr0, lr1, mu=0.85, rng=None, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = np.max(G)
        # avoid overflow: beta*(G-m) <= 0
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Gw = W @ X
        dot = np.sum(Gw * X, axis=1, keepdims=True)
        Gt = Gw - dot * X
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        if noise > 0 and rng is not None:
            nz = rng.standard_normal(X.shape)
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            nn = np.linalg.norm(nz, axis=1, keepdims=True) + 1e-12
            nz /= nn
            Gt = (1.0 - noise) * Gt + noise * nz
            # renormalize rows
            nn2 = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
            Gt = Gt / nn2
        vel = mu * vel + (1.0 - mu) * Gt
        # normalize velocity rows for stability
        vn = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
        step = vel / vn
        X = X - lr * step
        X = _nrm(X)
    return X

def _hole(Y, rng, n_cand=5000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _nrm(cand)
    dots = cand @ Y.T
    md = np.max(dots, axis=1)
    return cand[np.argmin(md)]

def _lp_polish(x, deadline, radius_init=1e-3):
    x = _nrm(x)
    n, d = x.shape
    best = _mc(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
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
        y = _nrm(y)
        v = _mc(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 172.0
    SEARCH_END = t0 + 128.0
    POLISH_END = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    # load incumbents
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _nrm(a)
                s = _mc(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _nrm(rng.standard_normal((n, d)))
        best_score = _mc(best)
    x_best = best.copy()
    # regular simplex in 21D (22 vertices) for structured seeds
    # build once: 22 x 22 simplex then project to 21D via QR null
    try:
        S22 = np.eye(22) - np.ones((22, 22)) / 22.0
        # orthonormal basis of subspace sum=0
        Q, _ = np.linalg.qr(np.ones((22, 1)))
        # null space basis
        U, _, _ = np.linalg.svd(S22)
        simplex22 = U[:, :21]  # not exact but orthonormal rows? use standard construction
        # better standard: vertices v_i in R^21
        # Use explicit: take 22 vectors in R22 orthogonal to 1, normalize, drop last coord via basis
        Z = np.eye(22) - np.ones((22, 22)) / 22.0
        # get basis B 22x21 of null(ones)
        q, _ = np.linalg.qr(np.column_stack([np.ones(22) / np.sqrt(22), rng.standard_normal((22, 21))]))
        # actually use SVD of projector
        # fallback: simple simplex via iterative
        simplex = None
    except Exception:
        simplex = None
    # precompute a clean 22-simplex in R21: v_i = (e_i - c) projected
    # c = mean; use Householder to map ones to e0
    v = np.zeros((22, 21))
    # construction: take 22 x 21 matrix with rows orthonormal-ish
    # Use known formula: G = I* (1+1/21) - J*(1/21) ... scaled
    # Simplex Gram: 1 on diag, -1/21 off diag. Cholesky it.
    Gsimp = np.full((22, 22), -1.0 / 21.0) + np.eye(22) * (1.0 + 1.0 / 21.0)
    try:
        L = np.linalg.cholesky(Gsimp + 1e-12 * np.eye(22))
        # L is 22x22 rank 21; take first 21 cols and orthonormalize rows? Use SVD to embed in 21D
        Uu, Ss, _ = np.linalg.svd(L)
        simp21 = Uu[:, :21] * np.sqrt(22.0 / 21.0)  # approx; will normalize
        # fix norms
        simp21 = _nrm(simp21[:22])
    except Exception:
        simp21 = None

    trial = 0
    # fast search loop, no LP inside
    while time.time() < SEARCH_END:
        trial += 1
        remain = SEARCH_END - time.time()
        if remain < 1.0:
            break
        mode = trial % 8
        X0 = None
        if mode == 0:
            # pure random Gaussian (new topology)
            X0 = _nrm(rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 700, 80.0, 900.0, 0.02, 0.004
            nz = 0.05
        elif mode == 1:
            # binary +-1 balanced seed
            X0 = rng.integers(0, 2, size=(n, d)) * 2.0 - 1.0
            X0 = _nrm(X0 + 0.15 * rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 700, 60.0, 800.0, 0.02, 0.004
            nz = 0.03
        elif mode == 2:
            # centrosymmetric: 27 + antipartners + 1
            half = _nrm(rng.standard_normal((27, d)))
            X0 = np.vstack([half, -half, _nrm(rng.standard_normal((1, d)))])
            # shuffle
            perm = rng.permutation(n)
            X0 = X0[perm]
            X0 = _nrm(X0 + 0.05 * rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 800, 100.0, 1200.0, 0.015, 0.003
            nz = 0.02
        elif mode == 3:
            # simplex + random rest (algebraic block)
            X0 = _nrm(rng.standard_normal((n, d)))
            if simp21 is not None:
                # random rotation of simplex block
                R, _ = np.linalg.qr(rng.standard_normal((d, d)))
                block = simp21 @ R.T
                X0[:22] = block
                X0 = _nrm(X0 + 0.02 * rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 800, 80.0, 1000.0, 0.015, 0.003
            nz = 0.02
        elif mode == 4:
            # big kick from incumbent to new basin
            sigma = 0.35 + 0.3 * rng.random()
            X0 = _nrm(x_best + sigma * rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 600, 120.0, 1000.0, 0.012, 0.003
            nz = 0.02
        elif mode == 5:
            # hole reinsertion of worst points from best
            X0 = x_best.copy()
            G = X0 @ X0.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            nr = 3 + int(rng.integers(0, 4))  # 3-6
            victims = order[:nr]
            keep_mask = np.ones(n, dtype=bool)
            # insert sequentially
            Ybase = X0[np.array([i for i in range(n) if i not in victims])]
            for w in victims:
                h = _hole(np.vstack([Ybase, X0[np.array([i for i in victims if i != w and i in range(n)])]]) if False else Ybase, rng, n_cand=4000)
                X0[w] = h
                Ybase = np.vstack([Ybase, h[None, :]])
            X0 = _nrm(X0 + 0.01 * rng.standard_normal((n, d)))
            steps, b0, b1, l0, l1 = 600, 100.0, 1100.0, 0.012, 0.003
            nz = 0.02
        elif mode == 6:
            # random orthogonal frames stacked
            R, _ = np.linalg.qr(rng.standard_normal((d, d)))
            # take 21 rows + 21 rows of another rotation + 13 random
            R2, _ = np.linalg.qr(rng.standard_normal((d, d)))
            A = np.vstack([R, R2[:21], _nrm(rng.standard_normal((13, d)))])
            # A is 55x21? R is 21x21, so rows =21+21+13=55
            X0 = _nrm(A + 0.1 * rng.standard_normal((55, 21)))
            steps, b0, b1, l0, l1 = 700, 70.0, 900.0, 0.018, 0.004
            nz = 0.03
        else:
            # subset shake of best
            X0 = x_best.copy()
            k = 12 + int(rng.integers(0, 12))
            subset = rng.choice(n, size=k, replace=False)
            sigma = 0.25 + 0.3 * rng.random()
            X0[subset] += sigma * rng.standard_normal((k, d))
            X0 = _nrm(X0)
            steps, b0, b1, l0, l1 = 550, 120.0, 900.0, 0.012, 0.003
            nz = 0.02
        # adapt steps to remaining time
        if remain < 6:
            steps = min(steps, 250)
        Xq = _quench(X0, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1, mu=0.85, rng=rng, noise=nz)
        s = _mc(Xq)
        if s < best_score - 1e-12:
            best_score = s
            x_best = Xq.copy()

    # final polish only on global best (cheap, time-guarded)
    if time.time() < POLISH_END - 2.0:
        # short extra quench at high beta to tighten
        try:
            x_best = _quench(x_best, steps=300, beta0=800.0, beta1=2500.0, lr0=0.004, lr1=0.0008, mu=0.9, rng=rng, noise=0.0)
            s = _mc(x_best)
            if s < best_score:
                best_score = s
        except Exception:
            pass
        try:
            x_best = _lp_polish(x_best, deadline=POLISH_END - 0.5, radius_init=1e-3)
        except Exception:
            pass
    x_best = _nrm(x_best)
    return x_best
