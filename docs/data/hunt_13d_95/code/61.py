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

def _full_gd(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        V = mu * V + Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _block_gd(Y, B, idx, mask, steps, T0, T1, lr0, lr1, mu=0.9):
    # Y: (k,d) variable, B: (m,d) fixed, idx/mask map into full n
    n = len(mask)
    d = Y.shape[1]
    k = Y.shape[0]
    V = np.zeros_like(Y)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    Xf = np.empty((n, d))
    Xf[mask] = B
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        Xf[idx] = Y
        G = Xf @ Xf.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        GradFull = W @ Xf
        Grad = GradFull[idx]
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        V = mu * V + Gt
        Y = Y - lr * V
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y /= nr
    Xf[idx] = Y
    # normalize (B already unit)
    nr = np.linalg.norm(Xf, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xf /= nr
    return Xf

def _fast_hole(others, rng, ns=8000):
    d = others.shape[1]
    C = rng.standard_normal((ns, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    md = (C @ others.T).max(axis=1)
    j = int(np.argmin(md))
    return C[j]

def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if seconds - (time.time() - start) < 0.1:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mk = np.zeros_like(sel, dtype=bool)
            mk[idx] = True
            sel = mk
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    EXPL_END = t0 + 128.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None and xr.shape == (95, 13):
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if not cands:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = best.copy()
    cur_val = best_val
    n, d = 95, 13
    sigmas = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    it = 0
    Tb0 = 0.0025
    while time.time() < EXPL_END and time.time() < HARD_END - 40:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (EXPL_END - t0))
        Tb = Tb0 * (1.0 - 0.85 * min(1.0, max(0.0, frac)))
        Tb = max(Tb, 2e-4)
        u = rng.random()
        if u < 0.06:
            # fresh random long anneal (rare, new basin)
            Xt = rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _full_gd(Xt, 700, 0.020, 0.0009, 0.16, 0.012, mu=0.9)
        elif u < 0.38:
            # subset shake
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            Xt = cur.copy()
            if rng.random() < 0.5:
                ksub = int(rng.integers(4, 16))
                idx = rng.choice(n, size=ksub, replace=False)
                Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
            else:
                Xt += sigma * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            if sigma > 0.08:
                Xt = _full_gd(Xt, 350, 0.014, 0.0009, 0.13, 0.010, mu=0.9)
            elif sigma > 0.02:
                Xt = _full_gd(Xt, 280, 0.010, 0.0008, 0.11, 0.010, mu=0.9)
            else:
                Xt = _full_gd(Xt, 220, 0.007, 0.0007, 0.09, 0.010, mu=0.9)
        elif u < 0.55:
            # single hole re-insertion of a worst point
            ppm = _per_point_max(cur)
            order = np.argsort(ppm)
            if rng.random() < 0.7:
                pool = order[-10:]
                j = int(pool[int(rng.integers(0, len(pool)))])
            else:
                j = int(rng.integers(0, n))
            mask = np.ones(n, dtype=bool); mask[j] = False
            others = cur[mask]
            h = _fast_hole(others, rng, ns=8000)
            Xt = cur.copy()
            Xt[j] = h
            Xt += 0.003 * rng.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = _full_gd(Xt, 250, 0.009, 0.0008, 0.11, 0.010, mu=0.9)
        else:
            # joint cluster block resampling: hub neighbourhood or random cap
            if rng.random() < 0.6:
                ppm = _per_point_max(cur)
                order = np.argsort(ppm)
                top = order[-8:]
                i0 = int(top[int(rng.integers(0, len(top)))])
                dots = cur @ cur[i0]
                dots[i0] = -2.0
                k = int(rng.integers(4, 11))  # total block size
                nn = np.argsort(dots)[-(k-1):]
                idx = np.array([i0] + nn.tolist(), dtype=int)
            else:
                uu = rng.standard_normal(d); uu /= np.linalg.norm(uu)
                proj = cur @ uu
                k = int(rng.integers(4, 11))
                idx = np.argsort(proj)[-k:]
            mask = np.ones(n, dtype=bool); mask[idx] = False
            B = cur[mask]
            ksz = len(idx)
            # init block: 50% perturbed old, 50% fresh biased
            if rng.random() < 0.5:
                Y0 = cur[idx] + 0.35 * rng.standard_normal((ksz, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            else:
                # fresh random, but bias toward cap centre if random cap
                Y0 = rng.standard_normal((ksz, d))
                Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
                if rng.random() < 0.5:
                    # pull toward mean of removed block to stay in region
                    ctr = cur[idx].mean(axis=0); ctr /= np.linalg.norm(ctr)
                    Y0 = Y0 + 0.8 * ctr
                    Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            Xt = _block_gd(Y0, B, idx, mask, 300, 0.012, 0.0009, 0.14, 0.010, mu=0.9)
            # short full relaxation to knit boundary
            Xt = _full_gd(Xt, 120, 0.005, 0.0007, 0.07, 0.008, mu=0.9)
        tv = _max_cos(Xt)
        if tv < best_val:
            best_val = tv
            best = Xt.copy()
        if tv < cur_val or rng.random() < np.exp((cur_val - tv) / Tb):
            cur = Xt.copy()
            cur_val = tv
        if it % 70 == 0:
            cur = best.copy()
            cur_val = best_val
        if best_val < 0.238:
            break
    remain = HARD_END - time.time() - 1.0
    if remain > 4:
        lp_time = min(max(6.0, remain * 0.85), 38.0)
        best = _lp_polish(best, lp_time, deadline=HARD_END - 1.5)
        best_val = _max_cos(best)
        left = HARD_END - time.time() - 0.6
        if left > 3.0:
            tmp = _full_gd(best.copy(), 250, 0.002, 0.0004, 0.03, 0.005, mu=0.9)
            tv = _max_cos(tmp)
            if tv < best_val:
                best = tmp
                best_val = tv
                l2 = HARD_END - time.time() - 0.4
                if l2 > 2.0:
                    best = _lp_polish(best, min(l2, 6.0), deadline=HARD_END - 0.2)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
