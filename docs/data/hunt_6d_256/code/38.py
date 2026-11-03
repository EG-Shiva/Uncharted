import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_riesz(x0, steps, p, lr, deadline=1e18):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    expc = float(p * 0.5 - 1.0)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        umax = (1.0 + gmax) * 0.5
        if umax <= 1e-12:
            umax = 0.5
        logUmax = np.log(umax)
        U = (1.0 + G) * 0.5
        np.fill_diagonal(U, 1.0)
        logU = np.log(np.maximum(U, 1e-300))
        E = expc * (logU - logUmax)
        np.fill_diagonal(E, -100.0)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _adam_balanced(x0, steps, K0, K1, lam0, lam1, topM, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    iu = np.triu_indices(n, 1)
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 10:
            break
        frac = t / steps
        K = float(np.exp(la0 + (la1 - la0) * frac))
        lam = float(lam0 + (lam1 - lam0) * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if topM > 0 and lam > 0:
            c = G[iu[0], iu[1]]
            if topM < c.size:
                thr = float(np.partition(c, c.size - topM)[c.size - topM])
            else:
                thr = float(c.min())
            Mask = (G >= thr).astype(float)
            np.fill_diagonal(Mask, 0.0)
            # symmetrize already symmetric; add uniform plateau weight
            W = W + lam * Mask
            np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    return _adam_balanced(x0, steps, K0, K1, 0.0, 0.0, 0, lr, deadline)

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (mval - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
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
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
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
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _tangent_shake(x, sigma, rng):
    z = rng.normal(size=x.shape)
    z -= np.sum(z * x, axis=1, keepdims=True) * x
    x2 = x + sigma * z
    x2 /= np.linalg.norm(x2, axis=1, keepdims=True)
    return x2

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    CENSUS_END = t0 + 105.0
    SHARP_END = t0 + 150.0
    rng0 = np.random.default_rng(0)
    x_best = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng0.normal(size=(256, 6))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    incumbent = x_best.copy()

    cands = []
    # de novo basins with equalized loss
    b = 0
    while time.time() < CENSUS_END:
        if b >= 7:
            break
        rng = np.random.default_rng(2000 + b)
        x = rng.normal(size=(256, 6))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        try:
            p = 12 if (b % 2 == 0) else 24
            lr_r = 0.012 if (b % 3 == 0) else (0.010 if (b % 3 == 1) else 0.014)
            x, _ = _adam_riesz(x, 180, p, lr_r, deadline=CENSUS_END)
            if time.time() > CENSUS_END:
                cands.append((_max_cos(x), x.copy())); break
            # equalized plateau phase
            topM = 1800 if (b % 2 == 0) else 1200
            x, _ = _adam_balanced(x, 550, 150.0, 2500.0, 0.6, 0.05, topM, 0.008, deadline=CENSUS_END)
            if time.time() > CENSUS_END:
                cands.append((_max_cos(x), x.copy())); break
            x, vv = _adam_balanced(x, 200, 3000.0, 5000.0, 0.0, 0.0, 0, 0.005, deadline=CENSUS_END)
            cands.append((vv, x.copy()))
            if vv < best:
                best = vv; x_best = x.copy()
        except Exception:
            pass
        b += 1
    # rattler-shake basins from incumbent (topology change)
    sigmas = [0.03, 0.06, 0.10, 0.16, 0.05, 0.12, 0.20, 0.08]
    for si, sigma in enumerate(sigmas):
        if time.time() > SHARP_END:
            break
        try:
            rng = np.random.default_rng(5000 + si)
            x = _tangent_shake(incumbent, sigma, rng)
            # short spread then equalized
            x, _ = _adam_riesz(x, 80, 16, 0.010, deadline=SHARP_END)
            if time.time() > SHARP_END:
                vv = _max_cos(x)
                cands.append((vv, x.copy()))
                if vv < best:
                    best = vv; x_best = x.copy()
                break
            topM = 2000 if (si % 2 == 0) else 1500
            x, _ = _adam_balanced(x, 450, 120.0, 2200.0, 0.8, 0.05, topM, 0.008, deadline=SHARP_END)
            if time.time() > SHARP_END:
                vv = _max_cos(x)
                cands.append((vv, x.copy()))
                if vv < best:
                    best = vv; x_best = x.copy()
                break
            x, vv = _adam_balanced(x, 220, 2500.0, 6000.0, 0.02, 0.0, 1000, 0.005, deadline=SHARP_END)
            cands.append((vv, x.copy()))
            if vv < best and np.all(np.isfinite(x)):
                best = vv; x_best = x.copy()
        except Exception:
            continue
    # polish top 3 distinct
    cands_sorted = sorted(cands, key=lambda t: t[0])[:3]
    # always include current best
    pool = []
    if x_best is not None:
        pool.append(x_best.copy())
    for vv, xx in cands_sorted:
        if vv < best + 5e-4:
            pool.append(xx.copy())
        if len(pool) >= 3:
            break
    # deduplicate pool lightly by value
    uniq = []
    for xx in pool:
        try:
            vv = _max_cos(xx)
        except Exception:
            continue
        dup = False
        for yy in uniq:
            try:
                if abs(_max_cos(yy) - vv) < 1e-12:
                    dup = True; break
            except Exception:
                pass
        if not dup:
            uniq.append(xx)
    for xx in uniq:
        if time.time() > T_END - 8:
            break
        try:
            # final sharp + LP
            xx2, vv2 = _adam_lse(xx, 250, 800.0, 6000.0, 0.004, deadline=T_END - 8)
            if vv2 < best:
                best = vv2; x_best = xx2.copy()
                xx = xx2
            xxp = _lp_polish(xx, deadline=min(time.time() + 12, T_END - 2))
            vvp = _max_cos(xxp)
            if vvp < best and np.all(np.isfinite(xxp)):
                best = vvp; x_best = xxp.copy()
        except Exception:
            continue
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
