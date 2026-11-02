import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _worst_info(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    idx = int(np.argmax(G))
    n = X.shape[0]
    i = idx // n
    j = idx % n
    m = float(G[i, j])
    # per-point worst
    pw = G.max(axis=1)
    return m, i, j, pw, G

def _adam_lse(X0, s0, s1, steps, lr0, lr1, rng=None, noise=0.0, t_lim=None):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros((n, d))
    v = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    l0, l1 = np.log(lr0), np.log(lr1)
    ls0, ls1 = np.log(s0), np.log(s1)
    WX = np.empty((n, d))
    for k in range(steps):
        if t_lim is not None and (k & 63) == 0 and time.time() > t_lim:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(l0 + (l1 - l0) * frac)
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        # log-sum-exp weights
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            # fallback to max pair only
            w = np.zeros_like(c)
            w[int(np.argmax(c))] = 1.0
        else:
            w = e / se
        WX.fill(0.0)
        np.add.at(WX, iu0, w[:, None] * X[iu1])
        np.add.at(WX, iu1, w[:, None] * X[iu0])
        sprod = np.sum(WX * X, axis=1)
        Gt = WX - sprod[:, None] * X
        m *= b1
        m += (1 - b1) * Gt
        v *= b2
        v += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # limit step size
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 3.0
        over = sn > cap
        if np.any(over):
            step = np.where(over, step * (cap / np.maximum(sn, 1e-12)), step)
        X = X - lr * step
        if noise > 0 and rng is not None:
            X += rng.standard_normal(size=(n, d)) * (noise * lr * 0.1)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _hole_for(X, rng, n_try=300):
    # find direction minimizing max dot to X
    n, d = X.shape
    C = rng.standard_normal(size=(n_try, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X.T
    m = S.max(axis=1)
    return C[int(np.argmin(m))]

def _lp_polish(X, best, t_deadline, max_iters=12):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    radius = 4e-4
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while it < max_iters and time.time() < t_deadline and radius > 1e-14:
        it += 1
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(25.0 * radius, 1.5e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 1800:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:1800]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
        if vv < best - 1e-12:
            X, best = y, vv
            radius = min(radius * 1.6, 0.01)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    SEARCH_END = t0 + 142.0
    LP_END = t0 + 164.0
    rng = np.random.default_rng(7)
    n, d = 73, 13
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc
                        Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    # Stage A: equalizing anneal from incumbent (balanced, moderate s)
    if time.time() < SEARCH_END:
        try:
            Xa = _adam_lse(Xbest, 60.0, 350.0, 900, 0.02, 0.002, t_lim=SEARCH_END)
            va = _max_cos(Xa)
            if va < best and np.all(np.isfinite(Xa)):
                best, Xbest = va, Xa.copy()
            # second push to higher s to equalize second tier
            Xb = _adam_lse(Xbest, 300.0, 900.0, 900, 0.008, 0.0012, t_lim=SEARCH_END)
            vb = _max_cos(Xb)
            if vb < best and np.all(np.isfinite(Xb)):
                best, Xbest = vb, Xb.copy()
        except Exception:
            pass

    # Stage B: cheap screened basin hops: micro-kicks + hole reinsertion
    # keep iterations cheap (400-600 steps) to avoid timeout
    it = 0
    sigmas = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3]
    while time.time() < SEARCH_END:
        it += 1
        r = rng.random()
        base = Xbest
        X0 = None
        if r < 0.45:
            # gaussian micro kick
            sigma = sigmas[rng.integers(0, len(sigmas))]
            if rng.random() < 0.25:
                sigma = float(10 ** rng.uniform(-4.2, -2.4))
            X0 = base + sigma * rng.standard_normal(size=(n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif r < 0.75:
            # worst-pair targeted kick: push the max pair apart + small noise
            try:
                _, ii, jj, pw, _ = _worst_info(base)
            except Exception:
                ii, jj = 0, 1
            X0 = base.copy()
            # tangent push apart
            for (a, b) in ((ii, jj), (jj, ii)):
                dot = float(np.dot(X0[a], X0[b]))
                tang = X0[b] - dot * X0[a]
                tn = float(np.linalg.norm(tang))
                if tn > 1e-12:
                    tang /= tn
                    amp = float(10 ** rng.uniform(-4.0, -2.6))
                    X0[a] = X0[a] - amp * tang
            X0 += rng.standard_normal(size=(n, d)) * 3e-4
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # kill worst point and reinsert into hole
            try:
                _, _, _, pw, _ = _worst_info(base)
                victim = int(np.argmax(pw))
            except Exception:
                victim = int(rng.integers(0, n))
            X0 = base.copy()
            # hole direction w.r.t. others
            mask = np.ones(n, dtype=bool)
            mask[victim] = False
            hole = _hole_for(X0[mask], rng, n_try=300)
            # blend with random for diversity
            if rng.random() < 0.4:
                hole = hole + 0.05 * rng.standard_normal(d)
                hole /= np.linalg.norm(hole)
            X0[victim] = hole
            # small shake of neighbours
            X0 += rng.standard_normal(size=(n, d)) * 2e-4
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # cheap screen
        s0 = float(10 ** rng.uniform(1.7, 2.1))
        s1 = float(10 ** rng.uniform(2.5, 2.95))
        if s1 < s0:
            s0, s1 = s1, s0
        steps = int(rng.choice([400, 500, 650]))
        lr0 = float(10 ** rng.uniform(-1.9, -1.5))
        lr1 = float(10 ** rng.uniform(-2.9, -2.4))
        try:
            Xc = _adam_lse(X0, s0, s1, steps, lr0, lr1, t_lim=SEARCH_END + 2)
        except Exception:
            continue
        if not np.all(np.isfinite(Xc)):
            continue
        vc = _max_cos(Xc)
        if vc < best:
            best, Xbest = vc, Xc.copy()
            # opportunistic short refine at higher s
            if time.time() < SEARCH_END - 5:
                try:
                    Xr = _adam_lse(Xbest, 500.0, 1000.0, 500, 0.004, 0.001, t_lim=SEARCH_END + 2)
                    vr = _max_cos(Xr)
                    if vr < best and np.all(np.isfinite(Xr)):
                        best, Xbest = vr, Xr.copy()
                except Exception:
                    pass
        # periodic check to escape infinite loop
        if it > 200:
            break

    # Stage C: final long equalizing polish on best
    if time.time() < LP_END - 25:
        try:
            Xp = _adam_lse(Xbest, 200.0, 1200.0, 1200, 0.006, 0.0008, t_lim=LP_END - 22)
            vp = _max_cos(Xp)
            if vp < best and np.all(np.isfinite(Xp)):
                best, Xbest = vp, Xp.copy()
        except Exception:
            pass

    # Stage D: brief LP squeeze
    if time.time() < LP_END - 3:
        try:
            Xl, bl = _lp_polish(Xbest, best, LP_END)
            if bl < best and np.all(np.isfinite(Xl)):
                best, Xbest = bl, Xl.copy()
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
