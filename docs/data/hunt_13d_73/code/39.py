import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam_smooth(X0, steps, s0, s1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    n = X.shape[0]
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        se = E.sum()
        if not np.isfinite(se) or se <= 0:
            continue
        W = E / se
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m *= b1
        m += (1.0 - b1) * gt
        v *= b2
        v += (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        X -= lr * (mh / (np.sqrt(vh) + eps))
        nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _find_holes(Xkeep, rng, k, ntry=2500):
    d = Xkeep.shape[1]
    holes = []
    cur = Xkeep.copy()
    for _ in range(k):
        C = rng.randn(ntry, d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ cur.T
        md = D.max(axis=1)
        bi = int(np.argmin(md))
        h = C[bi]
        holes.append(h)
        cur = np.vstack([cur, h[None, :]])
    return np.array(holes)

def _lp_polish(X, seconds, t_end):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = X @ X.T
        c = G[iu0]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        # cap constraints for speed
        if len(I) > 2500:
            idx = np.argsort(-c[sel])[:2500]
            I, J = I[idx], J[idx]
            ssel = c[sel][idx]
        else:
            ssel = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-ssel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(7)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    Xcur = xb.copy()
    cur_val = best_val

    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30, 0.45]
    trial = 0
    while time.time() < SEARCH_END and time.time() < DEADLINE - 35.0:
        trial += 1
        # base selection: mostly best, sometimes current for diversity
        if rng.rand() < 0.75:
            base = Xbest
            base_val = best_val
        else:
            base = Xcur
            base_val = cur_val
        mode = trial % 7
        Xtry = None
        try:
            if mode in (0, 1, 2, 5):
                # kill worst k and reinsert into holes (balanced topology move)
                k = 1
                r = rng.rand()
                if mode == 0:
                    k = 3
                elif mode == 1:
                    k = 2
                elif mode == 5:
                    k = 4 if r < 0.3 else 2
                else:
                    k = 1
                rm = _row_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                # occasionally also kill the rattler (smallest row-max) to move it
                if rng.rand() < 0.35:
                    ratt = int(np.argmin(rm))
                    victims[0] = ratt
                elif rng.rand() < 0.25:
                    victims[rng.integers(0, k)] = int(rng.integers(0, n))
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                Xkeep = base[keep_mask]
                holes = _find_holes(Xkeep, rng, k, ntry=2500)
                Xtry = base.copy()
                for vi, h in zip(victims, holes):
                    # small jitter on hole
                    hj = h + 0.005 * rng.randn(d)
                    hj /= np.linalg.norm(hj)
                    Xtry[vi] = hj
            elif mode == 3:
                # coherent cap shake: pick direction, move cap points together
                u = rng.randn(d); u /= np.linalg.norm(u)
                dots = base @ u
                thr = np.quantile(dots, 0.85)
                idx = np.where(dots >= thr)[0]
                Xtry = base.copy()
                pert = rng.randn(d); pert -= (pert @ u) * u
                pn = np.linalg.norm(pert)
                if pn > 0:
                    pert /= pn
                ang = 0.05 + 0.15 * rng.rand()
                # rotate cap points slightly toward pert
                for ii in idx:
                    v = Xtry[ii] + ang * pert + 0.01 * rng.randn(d)
                    v /= np.linalg.norm(v)
                    Xtry[ii] = v
            else:
                # global Gaussian kick
                s = sigmas[rng.integers(0, len(sigmas))]
                Xtry = base + s * rng.randn(n, d)
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        except Exception:
            continue
        # cheap screen + annealed Adam (two scales)
        try:
            # quick coarse stage
            Xref = _adam_smooth(Xtry, 260, 60.0, 160.0, 0.02, 0.004)
            # sharper stage for promising only or periodically
            v0 = _max_cos(Xref)
            # accept filter: only sharpen if within margin of best
            if v0 < best_val + 0.015 or (trial % 5 == 0):
                Xref2 = _adam_smooth(Xref, 260, 160.0, 420.0, 0.006, 0.0015)
            else:
                Xref2 = Xref
            v = _max_cos(Xref2)
        except Exception:
            continue
        # Metropolis for cur to keep diversity
        if v < cur_val or rng.rand() < np.exp(-(v - cur_val) / 0.0015):
            Xcur, cur_val = Xref2, v
        if v < best_val:
            best_val = v
            Xbest = Xref2.copy()
        if time.time() > SEARCH_END:
            break

    # final polish of incumbent only (time-safe)
    try:
        Xpol = _lp_polish(Xbest, 25.0, DEADLINE - 3.0)
        vp = _max_cos(Xpol)
        if vp < best_val:
            Xbest, best_val = Xpol, vp
        else:
            # keep polished version if equal within 1e-12 (more balanced)
            if vp <= best_val + 1e-12:
                # prefer fewer contacts? keep original best to be safe
                pass
    except Exception:
        pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
