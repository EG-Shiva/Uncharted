import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam(X0, steps, T0, T1, lr0, lr1, deadline):
    X = _norm(X0.copy())
    bv = _max_cos(X)
    bx = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    glob = 0
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob += 1
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1.0 - b1 ** glob)
        m2h = m2 / (1.0 - b2 ** glob)
        step = m1h / (np.sqrt(m2h) + eps)
        dn = np.sum(step * X, axis=1, keepdims=True)
        step = step - dn * X
        nm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 2.0 / nm)
        X = X - lr * step * sc
        X = _norm(X)
    v = _max_cos(X)
    if v < bv:
        bv = v
        bx = X.copy()
    return bx, bv

def _greedy(X0, deadline, T=6e-05, iters=600):
    X = _norm(X0.copy())
    bv = _max_cos(X)
    bx = X.copy()
    lr = 0.0015
    n = X.shape[0]
    for it in range(iters):
        if (it & 31) == 0 and time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            break
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # normalize per-point
        nm = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True)) + 1e-12
        Gt = Gt / np.maximum(1.0, nm * 2.0)
        Y = _norm(X - lr * Gt)
        v = _max_cos(Y)
        if v < bv:
            bv = v
            bx = Y.copy()
            X = Y
            lr = min(lr * 1.08, 0.003)
        else:
            # try smaller
            Y2 = _norm(X - lr * 0.3 * Gt)
            v2 = _max_cos(Y2)
            if v2 < bv:
                bv = v2
                bx = Y2.copy()
                X = Y2
            else:
                lr *= 0.92
                if lr < 1e-05:
                    lr = 1e-05
                # still move to equalize even if max up slightly
                if v < bv + 5e-04:
                    X = Y
        # anneal T slightly
        if (it % 200) == 199:
            T *= 0.8
    return bx, bv

def _lp_jam(x0, best, deadline, r0=4e-04, pmax=1500, max_solves=25):
    x = _norm(x0.copy())
    bx = x.copy()
    v0 = _max_cos(x)
    if v0 < best:
        best = v0
        bx = x.copy()
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    radius = r0
    solves = 0
    while time.time() < deadline and radius > 2e-13 and solves < max_solves:
        Gf = x @ x.T
        cf = Gf[iu0, iu1]
        m = float(cf.max())
        thresh = m - max(15.0 * radius, 3e-07)
        sel = cf > thresh
        p = int(np.count_nonzero(sel))
        if p < 500:
            k = min(800, cf.size)
            idx = np.argpartition(-cf, k - 1)[:k]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
            p = k
        if p > pmax:
            idx = np.where(sel)[0]
            order = np.argsort(cf[idx])[-pmax:]
            ns = np.zeros_like(sel, dtype=bool)
            ns[idx[order]] = True
            sel = ns
            p = pmax
        I = iu0[sel]; J = iu1[sel]
        cs = cf[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            tlim = max(0.5, min(2.5, deadline - time.time() - 0.1))
            if tlim <= 0.1:
                break
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tlim})
            solves += 1
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v
                bx = y.copy()
                x = y
                radius = min(radius * 1.3, 8e-04)
                improved = True
                break
        if not improved:
            radius *= 0.5
        if time.time() > deadline:
            break
    return bx, best

def _hole_vector(X, exclude, K=12000, seed=0):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    R = rng.randn(K, d)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = R @ X.T
    if 0 <= exclude < n:
        S[:, exclude] = -2.0
    sm = S.max(axis=1)
    k = int(np.argmin(sm))
    return R[k], float(sm[k])

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t_start)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.RandomState(0).randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        rngf = np.random.RandomState(1)
        y = rngf.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182); mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    n, d = 182, 9

    # Stage A: sharpen incumbent
    if rem() > 40:
        dl = time.time() + min(10.0, rem() - 32)
        xb, vb = _adam(x_best, steps=3200, T0=0.0010, T1=0.00008,
                       lr0=0.0020, lr1=0.00030, deadline=dl)
        if vb < best_val:
            best_val = vb; x_best = xb

    # Stage B: basin hops
    sigs = [0.004, 0.007, 0.0025, 0.010, 0.005, 0.0018, 0.008]
    for trial, sig in enumerate(sigs):
        if rem() < 36:
            break
        rng = np.random.RandomState(3000 + trial * 177)
        G = x_best @ x_best.T
        np.fill_diagonal(G, -10.0)
        iu0, iu1 = np.triu_indices(n, 1)
        flat = G[iu0, iu1]
        k2 = int(np.argmax(flat))
        a, b = int(iu0[k2]), int(iu1[k2])
        # near-bulk contact counts for weighting
        thr = np.sort(flat)[-1500]
        msk = flat > thr
        cnt = np.zeros(n)
        for aa, bb in zip(iu0[msk], iu1[msk]):
            cnt[aa] += 1; cnt[bb] += 1
        mx = cnt.max() if cnt.max() > 0 else 1.0
        cnt = cnt / mx
        Xp = x_best.copy()
        # deterministic split of worst pair
        try:
            xa = x_best[a]; xb_ = x_best[b]
            dot = float(xa @ xb_)
            da = xb_ - dot * xa
            db = xa - dot * xb_
            na = float(np.linalg.norm(da)); nb = float(np.linalg.norm(db))
            if na > 1e-12 and nb > 1e-12:
                da /= na; db /= nb
                Xp[a] = xa - sig * 2.0 * da
                Xp[b] = xb_ - sig * 2.0 * db
        except Exception:
            pass
        # hole reinsertion on even trials: replace worst point
        if (trial % 2) == 1:
            try:
                # worst point = larger row-max
                rm = G.max(axis=1)
                w = int(np.argmax(rm))
                hv, _ = _hole_vector(x_best, exclude=w, K=8000, seed=5000 + trial)
                # mix hole with old to keep continuity
                Xp[w] = _norm((hv[None, :] + 0.25 * rng.randn(1, d)))[0]
            except Exception:
                pass
        noise = rng.randn(n, 9)
        scale = (sig * (0.35 + 1.0 * cnt))[:, None]
        Xp = Xp + scale * noise
        Xp = _norm(Xp)
        dl = time.time() + min(7.0, rem() - 28)
        T0 = 0.0007 if trial % 2 == 0 else 0.0011
        xb2, vb2 = _adam(Xp, steps=2200, T0=T0, T1=0.00005,
                         lr0=0.0016, lr1=0.00020, deadline=dl)
        # quick greedy equalization if promising
        if vb2 < best_val + 0.002 and rem() > 30:
            dl2 = time.time() + min(1.5, rem() - 28)
            try:
                xg, vg = _greedy(xb2, deadline=dl2, T=6e-05, iters=300)
                if vg < vb2:
                    xb2, vb2 = xg, vg
            except Exception:
                pass
        if vb2 < best_val - 1e-12:
            best_val = vb2; x_best = xb2
            # short LP to jam new basin
            if rem() > 30:
                try:
                    xl2, vl2 = _lp_jam(xb2, vb2, deadline=time.time() + min(2.5, rem() - 28), r0=4e-04, pmax=1300, max_solves=8)
                    if vl2 < best_val:
                        best_val = vl2; x_best = xl2
                except Exception:
                    pass

    # Stage C: cold equalization
    if rem() > 16:
        dl = time.time() + min(10.0, rem() - 8)
        xb, vb = _adam(x_best, steps=4000, T0=0.00025, T1=0.000028,
                       lr0=0.00080, lr1=0.00012, deadline=dl)
        if vb < best_val:
            best_val = vb; x_best = xb
        # greedy
        if rem() > 10:
            dl2 = time.time() + min(2.0, rem() - 8)
            try:
                xg, vg = _greedy(x_best, deadline=dl2, T=5e-05, iters=400)
                if vg < best_val:
                    best_val = vg; x_best = xg
            except Exception:
                pass
    # Stage D: final LP jamming
    r = rem() - 1.0
    if r > 3:
        r = min(r, 22.0)
        try:
            xl, vl = _lp_jam(x_best, best_val, deadline=time.time() + r, r0=3.5e-04, pmax=1500, max_solves=25)
            if vl < best_val:
                best_val = vl; x_best = xl
        except Exception:
            pass
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
