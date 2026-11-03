import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def gd_refine(X, steps, T0, T1, lr0, lr1):
    # annealed soft-max repulsion with Adam on tangent space
    X = X.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v2 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    denom = np.empty_like(X)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        # softmax weights
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        ss = W.sum()
        if not np.isfinite(ss) or ss == 0:
            break
        W /= ss
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # Adam moments
        m *= b1
        m += (1.0 - b1) * Gt
        v2 *= b2
        v2 += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - (b1 ** (s + 1))
        bc2 = 1.0 - (b2 ** (s + 1))
        # denom = sqrt(v2/bc2)+eps
        np.multiply(v2, 1.0 / bc2, out=denom)
        np.sqrt(denom, out=denom)
        denom += eps
        upd = (lr / bc1) * (m / denom)
        # re-project adaptive step to tangent
        d2 = np.sum(upd * X, axis=1, keepdims=True)
        upd -= d2 * X
        X -= upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
        if not np.all(np.isfinite(X)):
            break
    return X


def lp_polish(x, seconds, radius_init=1e-3, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_END = t0 + 120.0
    DEADLINE = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 95, 13

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
    if len(cands) == 0:
        z = rng.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val

    sigmas = [0.03, 0.07, 0.15, 0.02, 0.05, 0.12, 0.25, 0.04]
    trial = 0
    while time.time() < EXPL_END and time.time() < DEADLINE - 40:
        trial += 1
        r = rng.random()
        try:
            if r < 0.45:
                sigma = sigmas[rng.integers(0, len(sigmas))]
                Xt = cur.copy()
                if rng.random() < 0.4:
                    ksub = int(rng.integers(4, 20))
                    idx = rng.choice(n, size=ksub, replace=False)
                    Xt[idx] += sigma * 2.0 * rng.standard_normal((ksub, d))
                else:
                    Xt += sigma * rng.standard_normal((n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 350
                T0, T1 = 0.03, 0.003
                lr0, lr1 = 0.003, 0.0003
                Xp = gd_refine(Xt, steps, T0, T1, lr0, lr1)
            else:
                rk = rng.random()
                if rk < 0.4:
                    k = 1
                elif rk < 0.7:
                    k = 2
                elif rk < 0.9:
                    k = 3
                else:
                    k = 4
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                if rng.random() < 0.7:
                    idx = order[:k].copy()
                    if k > 1 and rng.random() < 0.4:
                        idx[-1] = int(rng.integers(0, n))
                else:
                    idx = rng.choice(n, size=k, replace=False)
                Xt = cur.copy()
                for jj in idx:
                    mask = np.ones(n, dtype=bool)
                    mask[int(jj)] = False
                    others = Xt[mask]
                    cand = rng.standard_normal((3000, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    dots = cand @ others.T
                    md = dots.max(axis=1)
                    jbest = int(np.argmin(md))
                    Xt[int(jj)] = cand[jbest]
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = 450 if k > 1 else 380
                T0, T1 = 0.025, 0.0025
                lr0, lr1 = 0.003, 0.00025
                Xp = gd_refine(Xt, steps, T0, T1, lr0, lr1)
        except Exception:
            continue
        try:
            v = max_cos(Xp)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < best_val - 1e-12:
            x_best = Xp.copy()
            best_val = v
            cur = Xp.copy()
            cur_val = v
        elif v < cur_val:
            cur = Xp.copy()
            cur_val = v
        else:
            if rng.random() < 0.03 and v < cur_val + 0.008:
                cur = Xp.copy()
                cur_val = v
            if trial % 25 == 0 and rng.random() < 0.35:
                cur = x_best.copy()
                cur_val = best_val

    # final LP squeeze of best (and cur if distinct and good)
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 3.0:
            fsec = min(remain * 0.7, 30.0)
            x_best = lp_polish(x_best, seconds=fsec, radius_init=1e-3, deadline=DEADLINE - 1.0)
            best_val = max_cos(x_best)
            # polish cur as alternative if close
            if max_cos(cur) < best_val + 0.002:
                remain2 = DEADLINE - time.time() - 1.0
                if remain2 > 4.0:
                    cur_p = lp_polish(cur, seconds=min(8.0, remain2 * 0.4), radius_init=8e-4, deadline=DEADLINE - 1.0)
                    if max_cos(cur_p) < best_val:
                        x_best = cur_p
                        best_val = max_cos(x_best)
            remain3 = DEADLINE - time.time() - 1.0
            if remain3 > 2.0:
                x_best = lp_polish(x_best, seconds=min(remain3, 15.0), radius_init=5e-4, deadline=DEADLINE - 0.5)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: if NaN or bad, fall back to loaded best
    try:
        if not np.all(np.isfinite(x_best)) or x_best.shape != (95, 13):
            raise ValueError("bad")
        _ = max_cos(x_best)
    except Exception:
        z = cands[bi].copy()
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    return x_best
