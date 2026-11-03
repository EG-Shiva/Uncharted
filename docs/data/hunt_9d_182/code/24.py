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
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
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
        if (it % 100) == 0 and m < bv:
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
        # clip huge steps
        nm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 2.0 / nm)
        X = X - lr * step * sc
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m
        bx = X.copy()
    return bx, bv

def _lp(X0, seconds, r0=6e-4, pmax=3000):
    t_end = time.time() + seconds
    x = _norm(X0.copy())
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    iu0, iu1 = np.triu_indices(n, 1)
    radius = r0
    while time.time() < t_end and radius > 1e-13:
        Gf = x @ x.T
        cf = Gf[iu0, iu1]
        m = float(cf.max())
        thresh = m - max(20.0 * radius, 4e-7)
        sel = cf > thresh
        p = int(np.count_nonzero(sel))
        if p < 600:
            k = min(900, cf.size)
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
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            y = _norm(x + alpha * dx)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v
                bx = y.copy()
                x = y
                radius = min(radius * 1.4, 1.2e-3)
                improved = True
                break
        if not improved:
            radius *= 0.5
        if time.time() > t_end:
            break
    return bx, best

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

    # Stage A: quick sharpen from current best
    if rem() > 30:
        dl = time.time() + min(14.0, rem() - 22)
        xb, vb = _adam(x_best, steps=4000, T0=0.0012, T1=0.00010,
                       lr0=0.0020, lr1=0.00030, deadline=dl)
        if vb < best_val:
            best_val = vb; x_best = xb
    if rem() > 28:
        try:
            xl, vl = _lp(x_best, seconds=min(7.0, rem() - 22), r0=6e-4)
            if vl < best_val:
                best_val = vl; x_best = xl
        except Exception:
            pass

    # Stage B: basin hops targeting the isolated top pair
    sigs = [0.004, 0.007, 0.0025, 0.010, 0.005, 0.0015]
    for trial, sig in enumerate(sigs):
        if rem() < 30:
            break
        rng = np.random.RandomState(2000 + trial * 131)
        G = x_best @ x_best.T
        np.fill_diagonal(G, -10.0)
        iu0, iu1 = np.triu_indices(n, 1)
        flat = G[iu0, iu1]
        k2 = int(np.argmax(flat))
        a, b = int(iu0[k2]), int(iu1[k2])
        # contact counts in near bulk (top 1500)
        thr = np.sort(flat)[-1500]
        msk = flat > thr
        cnt = np.zeros(n)
        for aa, bb in zip(iu0[msk], iu1[msk]):
            cnt[aa] += 1; cnt[bb] += 1
        mx = cnt.max() if cnt.max() > 0 else 1.0
        cnt = cnt / mx
        Xp = x_best.copy()
        # deterministic split of worst pair along tangent
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
        # contact-weighted noise elsewhere
        noise = rng.randn(n, 9)
        # remove radial component later via norm; scale
        scale = (sig * (0.35 + 1.0 * cnt))[:, None]
        Xp = Xp + scale * noise
        # extra jitter orthogonal for a,b
        Xp = _norm(Xp)
        dl = time.time() + min(9.0, rem() - 24)
        # anneal from moderate to ultra-low T to equalize bulk
        T0 = 0.0006 if trial % 2 == 0 else 0.0010
        xb2, vb2 = _adam(Xp, steps=2600, T0=T0, T1=0.00006,
                         lr0=0.0016, lr1=0.00020, deadline=dl)
        # judge with short LP only if promising
        if vb2 < best_val + 0.0015 and rem() > 26:
            try:
                xl2, vl2 = _lp(xb2, seconds=min(3.0, rem() - 24), r0=5e-4)
            except Exception:
                xl2, vl2 = xb2, vb2
            if vl2 < best_val - 1e-12:
                best_val = vl2; x_best = xl2
            elif vb2 < best_val - 1e-12:
                best_val = vb2; x_best = xb2
        else:
            if vb2 < best_val - 1e-12:
                best_val = vb2; x_best = xb2

    # Stage C: ultra-low-T equalization
    if rem() > 14:
        dl = time.time() + min(12.0, rem() - 9)
        xb, vb = _adam(x_best, steps=4500, T0=0.00028, T1=0.000030,
                       lr0=0.00080, lr1=0.00012, deadline=dl)
        if vb < best_val:
            best_val = vb; x_best = xb
    # Stage D: final LP jamming
    r = rem() - 1.2
    if r > 3:
        r = min(r, 40.0)
        try:
            xl, vl = _lp(x_best, seconds=r, r0=4e-4)
            if vl < best_val:
                best_val = vl; x_best = xl
        except Exception:
            pass
    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
