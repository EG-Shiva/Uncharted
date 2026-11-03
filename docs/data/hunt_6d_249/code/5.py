import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_adam(x, steps, tau0, tau1, lr, deadline=None):
    n, d = x.shape
    mt = np.zeros_like(x)
    vt = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        frac = t / steps
        tau = tau0 + (tau1 - tau0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        W = np.exp((G - mval) / tau)
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S < 1e-200:
            break
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        b1p *= b1
        b2p *= b2
        mt = b1 * mt + (1.0 - b1) * gtan
        vt = b2 * vt + (1.0 - b2) * (gtan * gtan)
        mhat = mt / (1.0 - b1p)
        vhat = vt / (1.0 - b2p)
        upd = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _lp_polish(x, deadline):
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.5e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thresh)[0]
        if sel.size == 0:
            break
        if sel.size > 3000:
            part = np.argpartition(-c_all, 3000)[:3000]
            sel = part
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = sel.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_block = np.concatenate(
            [(I[:, None] * d + ar), (J[:, None] * d + ar),
             np.full((p, 1), n * d)], axis=1
        ).ravel()
        vals_block = np.concatenate(
            [x[J], x[I], -np.ones((p, 1))], axis=1
        ).ravel()
        try:
            A_ub = coo_matrix((vals_block, (rows, cols_block)),
                              shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (rep, np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            bounds = [(-radius, radius)] * (n * d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    dead_hop = t0 + 110.0
    dead_lp = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    nr = np.linalg.norm(xb, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    xb /= nr
    n, d = xb.shape
    best = _max_cos(xb)

    # initial smooth squeeze without perturbation
    try:
        x0 = _smooth_adam(xb.copy(), steps=2000, tau0=0.008,
                          tau1=0.002, lr=0.005, deadline=dead_hop)
        v0 = _max_cos(x0)
        if v0 < best:
            xb, best = x0, v0
    except Exception:
        pass

    sigmas = [0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
    ks = [3, 5, 8, 12, 20]
    it = 0
    while time.time() < dead_hop:
        it += 1
        sigma = sigmas[it % len(sigmas)]
        k = ks[(it // len(sigmas)) % len(ks)]
        # worst points
        try:
            G = xb @ xb.T
            np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            idx_worst = np.argsort(-w)
            top30 = idx_worst[:30]
        except Exception:
            top30 = np.arange(min(30, n))
        k1 = min(k // 2 + 1, 30, k)
        k2 = k - k1
        sel = set()
        if k1 > 0:
            pick1 = rng.choice(top30, size=k1, replace=False)
            for v in pick1:
                sel.add(int(v))
        if k2 > 0:
            # random complement
            cand = rng.choice(n, size=min(n, k2 * 3), replace=False)
            c = 0
            for v in cand:
                if int(v) not in sel:
                    sel.add(int(v))
                    c += 1
                    if c >= k2:
                        break
        sel = list(sel)
        tau0 = float(rng.uniform(0.005, 0.012))
        tau1 = 0.0015
        lr = float(rng.uniform(0.003, 0.008))
        steps = 800 if sigma < 0.05 else 1500
        x_try = xb.copy()
        if len(sel) > 0:
            x_try[sel] += rng.normal(scale=sigma, size=(len(sel), d))
        # tiny global jitter to break symmetry
        x_try += rng.normal(scale=0.0015, size=(n, d))
        nn = np.linalg.norm(x_try, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        x_try /= nn
        try:
            x_try = _smooth_adam(x_try, steps=steps, tau0=tau0,
                                 tau1=tau1, lr=lr, deadline=dead_hop)
        except Exception:
            continue
        v = _max_cos(x_try)
        if v < best:
            xb, best = x_try, v

    # final LP trust-region squeeze
    try:
        if time.time() < dead_lp:
            xb = _lp_polish(xb, dead_lp)
    except Exception:
        pass
    nn = np.linalg.norm(xb, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    xb /= nn
    xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
    # safety: replace any zero row
    nn2 = np.linalg.norm(xb, axis=1)
    for i in np.where(nn2 < 1e-12)[0]:
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        xb[i] = v
    return xb
