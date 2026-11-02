import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contact_counts(X, tol=1e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    return (G >= m - tol).sum(axis=1), m

def _smooth(X, iters, k0=70.0, k1=320.0, lr0=0.028, lr1=0.004):
    X = X.copy()
    for it in range(iters):
        f = it / max(1, iters - 1)
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp((G - m) * k)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        s = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - s * X
        rn = np.linalg.norm(Gt, axis=1).max() + 1e-12
        X = X - (lr / rn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _lp_budget(x, seconds, radius_init=1e-3, cap=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-13:
        Gf = x @ x.T
        c = Gf[iu0, iu1]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            if time.time() - start >= seconds:
                break
        else:
            radius *= 0.5
    return x, best

def _void(X, rng, ns=5000):
    n, d = X.shape
    R = rng.standard_normal(size=(ns, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    dots = R @ X.T
    worst = dots.max(axis=1)
    return R[int(np.argmin(worst))]

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    SEARCH_END = 152.0
    FINAL_END = 176.0
    n, d = 73, 13
    X0 = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < bv:
                        bv = c
                        X0 = v.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        bv = _max_cos(X0)
    bestX = X0.copy()
    best = bv
    try:
        bestX, best = _lp_budget(bestX, seconds=2.0, radius_init=1e-3)
    except Exception:
        pass
    cur = bestX.copy()
    cur_val = best
    it = 0
    while time.time() - t0 < SEARCH_END:
        it += 1
        if time.time() - t0 > SEARCH_END - 1.0:
            break
        T = 0.0022 * (0.997 ** it) + 0.0002
        try:
            ppm = _per_point_max(cur)
        except Exception:
            ppm = None
        try:
            cc, _ = _contact_counts(cur, tol=1e-3)
        except Exception:
            cc = None
        Xtry = cur.copy()
        r = rng.random()
        if r < 0.28:
            sigmas = (0.006, 0.014, 0.028, 0.055)
            sg = sigmas[it % len(sigmas)]
            Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * sg
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.52:
            ksub = int(rng.integers(4, 12))
            if ppm is not None and cc is not None:
                score = ppm + 0.0005 * cc
                order = np.argsort(-score)
                pool = order[:22]
                idx = rng.choice(pool, size=min(ksub, len(pool)), replace=False)
            elif ppm is not None:
                order = np.argsort(-ppm)
                idx = rng.choice(order[:20], size=ksub, replace=False)
            else:
                idx = rng.choice(n, size=ksub, replace=False)
            Xtry[idx] += rng.standard_normal(size=(len(idx), d)) * 0.085
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.003
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif r < 0.74:
            try:
                vv = _void(cur, rng, ns=5000)
            except Exception:
                vv = rng.standard_normal(size=(d,))
                vv /= np.linalg.norm(vv)
            if ppm is not None:
                if rng.random() < 0.65:
                    idx = int(np.argmax(ppm))
                else:
                    if rng.random() < 0.5:
                        idx = int(np.argmin(ppm))
                    else:
                        idx = int(rng.integers(0, n))
            else:
                idx = int(rng.integers(0, n))
            # sometimes kill two points into two voids
            if rng.random() < 0.30 and ppm is not None:
                order = np.argsort(-ppm)
                idx2 = int(order[1])
                try:
                    vv2 = _void(cur, rng, ns=3000)
                except Exception:
                    vv2 = rng.standard_normal(size=(d,)); vv2 /= np.linalg.norm(vv2)
                Xtry[idx2] = vv2 + rng.standard_normal(size=(d,)) * 0.01
                Xtry[idx2] /= np.linalg.norm(Xtry[idx2])
            Xtry[idx] = vv + rng.standard_normal(size=(d,)) * 0.008
            Xtry[idx] /= np.linalg.norm(Xtry[idx])
            Xtry += rng.standard_normal(size=Xtry.shape) * 0.0035
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            try:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                a, b = np.unravel_index(int(np.argmax(G)), G.shape)
                Xtry[a] = Xtry[a] - 0.035 * Xtry[b] + rng.standard_normal(size=(d,)) * 0.012
                Xtry[b] = Xtry[b] - 0.035 * Xtry[a] + rng.standard_normal(size=(d,)) * 0.012
                Xtry += rng.standard_normal(size=Xtry.shape) * 0.004
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            except Exception:
                Xtry = Xtry + rng.standard_normal(size=Xtry.shape) * 0.02
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        try:
            ni = 320 + int(rng.integers(0, 260))
            Xtry = _smooth(Xtry, iters=ni, k0=70.0, k1=330.0, lr0=0.028, lr1=0.004)
        except Exception:
            pass
        try:
            vq = _max_cos(Xtry)
        except Exception:
            continue
        # filter: skip LP if far worse to save time
        if vq > cur_val + 0.012 and vq > best + 0.008:
            # still allow rare uphill to keep diversity
            if rng.random() > 0.05:
                continue
            else:
                # accept smooth directly with small prob, no LP
                if rng.random() < np.exp((cur_val - vq) / max(T, 1e-6)) * 0.2:
                    cur, cur_val = Xtry, vq
                continue
        tl = SEARCH_END - (time.time() - t0)
        if tl < 0.9:
            # no time for LP, just greedy on smooth value
            if vq < best - 1e-12:
                best, bestX = vq, Xtry.copy()
                cur, cur_val = Xtry.copy(), vq
            elif vq < cur_val:
                cur, cur_val = Xtry.copy(), vq
            continue
        budget = 0.75 if (it % 4) else 1.2
        budget = min(budget, tl)
        try:
            Xpol, v = _lp_budget(Xtry, seconds=budget, radius_init=1.4e-3)
        except Exception:
            continue
        if not np.all(np.isfinite(Xpol)):
            continue
        if v < best - 1e-12:
            best, bestX = v, Xpol.copy()
            cur, cur_val = Xpol.copy(), v
        elif v < cur_val:
            cur, cur_val = Xpol.copy(), v
        else:
            try:
                if rng.random() < np.exp((cur_val - v) / max(T, 1e-6)):
                    cur, cur_val = Xpol.copy(), v
            except Exception:
                pass
        if it % 25 == 0 and cur_val > best + 0.015:
            cur, cur_val = bestX.copy(), best
    try:
        rem = FINAL_END - (time.time() - t0) - 0.8
        if rem > 2.0:
            bestX, best = _lp_budget(bestX, seconds=min(rem, 15.0), radius_init=9e-4)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    if not np.all(np.isfinite(bestX)):
        bestX = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    return bestX
