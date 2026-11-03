import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _tangent_noise(X, sigma, rng):
    N = rng.standard_normal(X.shape)
    d = np.sum(N * X, axis=1, keepdims=True)
    Nt = N - d * X
    return _norm(X + sigma * Nt)

def _lse_grad(X, beta):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    S = float(E.sum() * 0.5) + 1e-300
    W = E / S
    Gr = W @ X
    return m, Gr

def _adam_free(X0, betas, iters_list, lrs, deadline):
    n, d = X0.shape
    Y = X0.copy()
    m1 = np.zeros_like(Y); m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for beta, iters, lr in zip(betas, iters_list, lrs):
        for _ in range(int(iters)):
            if time.time() > deadline:
                return _norm(Y)
            nn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
            X = Y / nn
            _, Gx = _lse_grad(X, beta)
            dot = np.sum(Gx * X, axis=1, keepdims=True)
            Gy = (Gx - dot * X) / nn
            step += 1
            m1 = b1 * m1 + (1 - b1) * Gy
            m2 = b2 * m2 + (1 - b2) * Gy * Gy
            mh = m1 / (1 - b1 ** step)
            vh = m2 / (1 - b2 ** step)
            Y -= lr * mh / (np.sqrt(vh) + eps)
    return _norm(Y)

def _largest_hole(X, rng, ntry=2000, exclude=-1):
    n, d = X.shape
    C = _norm(rng.standard_normal((ntry, d)))
    if exclude >= 0:
        Xr = np.delete(X, exclude, axis=0)
    else:
        Xr = X
    # max dot of each candidate to set
    D = C @ Xr.T
    mx = D.max(axis=1)
    return C[int(np.argmin(mx))]

def _void_reinsert(X, rng, nrep=2, ntry=1500):
    X = X.copy()
    n, d = X.shape
    for _ in range(nrep):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        worst = int(np.argmax(rowmax))
        cand = _largest_hole(X, rng, ntry=ntry, exclude=worst)
        X[worst] = cand
        # also try second worst if still jammed
        G2 = X @ X.T
        np.fill_diagonal(G2, -2.0)
        # greedy accept only if global max improves or stays close
        # accept regardless to change topology (basin jump)
    return _norm(X)

def _lp_polish(X, deadline, radius0=6e-3, cap=2200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gf = X @ X.T
        c = Gf[iu]
        m = float(c.max())
        thr = m - max(3.0 * radius, 5e-4)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            X, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    n, d = 138, 8
    rng = np.random.default_rng(7)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    base_best = best.copy()

    cands = [(best_val, base_best.copy())]

    # schedules
    betasA = [40.0, 130.0, 320.0]
    itersA = [500, 500, 700]
    lrsA = [0.02, 0.012, 0.006]
    betasB = [150.0, 350.0]
    itersB = [300, 400]
    lrsB = [0.01, 0.005]

    sigmas = [0.03, 0.08, 0.15, 0.25, 0.4, 0.6]
    # include pure random starts for truly different basins
    explore_deadline = t0 + 118.0

    si = 0
    # interleave kicked starts and random starts
    starts = []
    for s in sigmas:
        starts.append(('kick', s))
    for _ in range(4):
        starts.append(('rand', 0.0))
    for s in [0.1, 0.2, 0.35]:
        starts.append(('kick', s))

    for kind, sig in starts:
        if time.time() > explore_deadline:
            break
        if kind == 'kick':
            X0 = _tangent_noise(base_best, sig, rng)
            # disperse hubs: extra noise on currently high-contact points
            try:
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowmax = G.max(axis=1)
                hub = np.where(rowmax > m - 2e-4)[0]
                if len(hub) > 0:
                    Nz = rng.standard_normal((len(hub), d))
                    dz = np.sum(Nz * X0[hub], axis=1, keepdims=True)
                    Nz -= dz * X0[hub]
                    X0[hub] = _norm(X0[hub] + (sig * 0.7) * Nz)
            except Exception:
                pass
        else:
            X0 = _norm(rng.standard_normal((n, d)))
        try:
            X1 = _adam_free(X0, betasA, itersA, lrsA, explore_deadline)
        except Exception:
            continue
        v1 = _max_cos(X1)
        # void reinsert worst into largest hole + short relax (basin jump)
        for rep in range(2):
            if time.time() > explore_deadline:
                break
            try:
                Xr = _void_reinsert(X1, rng, nrep=2, ntry=1200)
                Xr = _adam_free(Xr, betasB, itersB, lrsB, explore_deadline)
            except Exception:
                break
            vr = _max_cos(Xr)
            # keep the relaxed version if competitive (allow small uphill to keep diversity)
            if vr < v1 + 0.005:
                X1, v1 = Xr, vr
            else:
                break
        cands.append((v1, X1.copy()))
        cands.sort(key=lambda t: t[0])
        cands = cands[:4]
        if v1 < best_val - 1e-12:
            best_val = v1; best = X1.copy()
        si += 1

    # LP polish top candidates with remaining time
    cands.sort(key=lambda t: t[0])
    # always include global best
    ordered = []
    seen = set()
    for v, Xc in cands[:4]:
        ordered.append((v, Xc))
    # polish in order, best first
    lp_deadline = deadline - 3.0
    for v, Xc in ordered:
        if time.time() > lp_deadline - 5.0:
            break
        # small shake before polish for non-best to try new contact topology
        for trial in range(2):
            if time.time() > lp_deadline:
                break
            if trial == 0:
                Xt = Xc.copy()
            else:
                Xt = _tangent_noise(Xc, 0.008, rng)
                Xt = _adam_free(Xt, [200.0], [250], [0.006], lp_deadline)
            Xp = _lp_polish(Xt, min(lp_deadline, time.time() + 28.0))
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp; best = Xp.copy()
            if vp < v - 1e-9 and trial == 0:
                break
        if time.time() > lp_deadline:
            break

    # final short LP on best if time remains
    if time.time() < deadline - 8.0:
        try:
            Xf = _lp_polish(best, deadline - 2.0, radius0=2e-3, cap=2500)
            vf = _max_cos(Xf)
            if vf < best_val:
                best = Xf; best_val = vf
        except Exception:
            pass
    return _norm(best)
