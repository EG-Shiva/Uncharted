import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 20:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * radius, 6e-5)
        sel = c > (mval - margin)
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]
            J = J[part]
            csel = csel[part]
            pnum = cap
        rows = np.repeat(np.arange(pnum), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((pnum, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pnum, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(pnum, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
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
        val = _max_cos(y)
        if val < best:
            x = y
            bx = y.copy()
            best = val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return bx, best

def _push_bottleneck(x, eps, rng, jitter):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = min(8, n)
    # find top few pairs
    flat = np.argpartition(G.ravel(), -k)[-k:]
    pairs = [(int(f // n), int(f % n)) for f in flat]
    y = x.copy()
    for (i, j) in pairs:
        if i == j:
            continue
        # push i away from j and j away from i along tangent
        vi = x[i] - float(np.dot(x[i], x[j])) * x[j]
        vj = x[j] - float(np.dot(x[j], x[i])) * x[i]
        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        if ni > 1e-12:
            y[i] = y[i] - eps * vi / ni
        if nj > 1e-12:
            y[j] = y[j] - eps * vj / nj
    if jitter > 0:
        z = rng.normal(size=y.shape)
        z -= np.sum(z * y, axis=1, keepdims=True) * y
        y = y + jitter * z
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _void_replace(x, rng, nrep=1):
    y = x.copy()
    n = x.shape[0]
    rm = _row_max(x)
    for _ in range(nrep):
        worst = int(np.argmax(rm))
        cand = rng.normal(size=(4000, 6))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # score = max dot to existing (excluding worst)
        rest = np.delete(y, worst, axis=0)
        dots = cand @ rest.T
        md = dots.max(axis=1)
        pick = int(np.argmin(md))
        y[worst] = cand[pick]
        # update rm roughly
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
    return y

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        rng0 = np.random.default_rng(0)
        x_best = rng0.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    # short initial LP squeeze with different margin
    try:
        if time.time() < T_END - 60:
            bx, bv = _lp_polish(x_best, deadline=min(T_END - 55, time.time() + 9), r0=1.0e-3, cap=2600)
            if bv < best and np.all(np.isfinite(bx)):
                best = bv
                x_best = bx.copy()
    except Exception:
        pass
    trial = 0
    rng_master = np.random.default_rng(777)
    sigmas = [0.002, 0.004, 0.007, 0.012, 0.02, 0.035]
    while time.time() < T_END - 32:
        trial += 1
        seed = 2000 + trial * 131
        rng = np.random.default_rng(seed)
        mode = trial % 4
        sigma = sigmas[(trial // 4) % len(sigmas)]
        eps = float(rng_master.uniform(0.002, 0.012))
        if mode == 0:
            x0 = _push_bottleneck(x_best, eps, rng, jitter=sigma * 0.5)
        elif mode == 1:
            # global jitter
            z = rng.normal(size=x_best.shape)
            z -= np.sum(z * x_best, axis=1, keepdims=True) * x_best
            x0 = x_best + sigma * z
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode == 2:
            x0 = _void_replace(x_best, rng, nrep=int(rng.integers(1, 3)))
            z = rng.normal(size=x0.shape)
            z -= np.sum(z * x0, axis=1, keepdims=True) * x0
            x0 = x0 + (sigma * 0.5) * z
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # subset shake of high row-max points
            rm = _row_max(x_best)
            order = np.argsort(-rm)
            nshake = int(rng.integers(8, 40))
            x0 = x_best.copy()
            z = rng.normal(size=(nshake, 6))
            idx = order[:nshake]
            # tangent
            z -= np.sum(z * x0[idx], axis=1, keepdims=True) * x0[idx]
            x0[idx] = x0[idx] + sigma * 1.5 * z
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        K0 = 120.0
        K1 = 4500.0
        lr = 0.015 if (trial % 2 == 0) else 0.008
        steps = 650
        dl = min(T_END - 32, time.time() + 4.0)
        try:
            xa, va = _adam_lse(x0, steps=steps, K0=K0, K1=K1, lr=lr, deadline=dl)
        except Exception:
            continue
        if va < best:
            # mini LP to fully relax
            try:
                xb, vb = _lp_polish(xa, deadline=min(T_END - 30, time.time() + 3.0), r0=8e-4, cap=2200)
                if vb < va and np.all(np.isfinite(xb)):
                    xa, va = xb, vb
            except Exception:
                pass
            if va < best:
                best = va
                x_best = xa.copy()
        # early stop if time low
        if time.time() > T_END - 34:
            break
    # final polish: Adam fine + LP
    try:
        if time.time() < T_END - 6:
            xa, va = _adam_lse(x_best, steps=500, K0=300.0, K1=12000.0, lr=0.006, deadline=min(T_END - 22, time.time() + 6))
            if va < best:
                best = va
                x_best = xa.copy()
    except Exception:
        pass
    try:
        if time.time() < T_END - 2:
            xb, vb = _lp_polish(x_best, deadline=T_END - 1.5, r0=1.5e-3, cap=3000)
            if vb < best and np.all(np.isfinite(xb)):
                best = vb
                x_best = xb.copy()
    except Exception:
        pass
    x_best = np.asarray(x_best, float)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    bad = ~np.all(np.isfinite(x_best), axis=1)
    if np.any(bad):
        rng = np.random.default_rng(999)
        for i in np.where(bad)[0]:
            v = rng.normal(size=6)
            v /= np.linalg.norm(v)
            x_best[i] = v
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best /= nr
    return x_best
