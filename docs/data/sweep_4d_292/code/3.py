import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.arange(d)
        cols = np.concatenate(
            [I[:, None] * d + cc, J[:, None] * d + cc,
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": max(0.5, seconds - (time.time() - start))})
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


def smooth_adam(Y0, steps, tau0, tau1, lr):
    Y = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        tau = tau0 * (tau1 / tau0) ** frac
        lrt = lr * (1.0 - 0.7 * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = (G - M) / tau
        mask = D > -15.0
        W = np.zeros_like(G)
        W[mask] = np.exp(D[mask])
        s = float(np.sum(W))
        if s <= 0 or not np.isfinite(s):
            break
        W /= s
        grad = 2.0 * (W @ Y)
        r = np.sum(grad * Y, axis=1, keepdims=True)
        g = grad - r * Y
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * g
        vv = b2 * vv + (1.0 - b2) * (g * g)
        mh = m / (1.0 - b1p)
        vh = vv / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        # clip per-point update to avoid explosions
        un = np.linalg.norm(upd, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 3.0 / un)
        upd = upd * clip
        Y = Y - lrt * upd
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y


def reinsert_rattlers(xb, rng):
    n, d = xb.shape
    G = xb @ xb.T
    np.fill_diagonal(G, -2.0)
    M = float(np.max(G))
    contacts = np.sum(G > M - 1e-7, axis=1)
    ridx = np.where(contacts == 0)[0]
    if len(ridx) == 0:
        ridx = np.argsort(contacts)[:3]
    k = len(ridx)
    fixed = np.delete(xb, ridx, axis=0)
    ns = 20000
    cand = rng.randn(ns, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # chunked scoring to limit memory
    best_dot = np.empty(ns)
    chunk = 5000
    for s in range(0, ns, chunk):
        e = min(ns, s + chunk)
        best_dot[s:e] = np.max(cand[s:e] @ fixed.T, axis=1)
    order = np.argsort(best_dot)
    chosen = []
    for idx in order:
        c = cand[idx]
        ok = True
        for j in chosen:
            if float(c @ cand[j]) > 0.90:
                ok = False
                break
        if ok:
            chosen.append(idx)
            if len(chosen) >= k:
                break
    xn = xb.copy()
    for r, ci in zip(ridx, chosen):
        xn[r] = cand[ci]
    return xn


def solve():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    t0 = time.time()
    BUDGET = 175.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (292, 4):
        # try to reshape / truncate defensively
        xb = xb.reshape(292, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    it = 0
    # reserve ~20s for final polish
    while time.time() - t0 < 148.0:
        it += 1
        mode = it % 5
        if mode == 1:
            sigma = 0.008
            xt = x_best + sigma * rng.randn(*x_best.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = smooth_adam(xt, steps=700, tau0=0.02, tau1=0.002, lr=0.002)
        elif mode == 2:
            sigma = 0.022
            xt = x_best + sigma * rng.randn(*x_best.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = smooth_adam(xt, steps=900, tau0=0.015, tau1=0.001, lr=0.0025)
        elif mode == 3:
            xt = reinsert_rattlers(x_best, rng)
            # small shake of the rest to break symmetry
            xt = xt + 0.003 * rng.randn(*xt.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = smooth_adam(xt, steps=800, tau0=0.012, tau1=0.0008, lr=0.002)
        elif mode == 4:
            # shake overloaded hubs harder
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            contacts = np.sum(G > M - 1e-7, axis=1)
            order = np.argsort(-contacts)
            nhub = 20
            hubs = order[:nhub]
            xt = x_best.copy()
            noise = rng.randn(*xt.shape) * 0.004
            noise[hubs] *= 5.0  # ~0.02 on hubs
            xt = xt + noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            xt = smooth_adam(xt, steps=700, tau0=0.018, tau1=0.0015, lr=0.002)
        else:
            # random subset teleport: move 4 random points to holes
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            contacts = np.sum(G > M - 1e-7, axis=1)
            worst = np.argsort(-contacts)[:8]
            pick = worst[rng.choice(len(worst), size=min(4, len(worst)), replace=False)]
            ns = 10000
            cand = rng.randn(ns, 4)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            fixed = np.delete(x_best, pick, axis=0)
            bd = np.max(cand @ fixed.T, axis=1)
            order = np.argsort(bd)
            chosen = []
            for idx in order:
                c = cand[idx]
                if all(float(c @ cand[j]) < 0.90 for j in chosen):
                    chosen.append(idx)
                    if len(chosen) >= len(pick):
                        break
            xt = x_best.copy()
            for r, ci in zip(pick, chosen):
                xt[r] = cand[ci]
            xt = smooth_adam(xt, steps=800, tau0=0.015, tau1=0.001, lr=0.002)

        v = max_cos(xt)
        # snap promising basins with short LP
        if v < best_val + 0.004:
            remain = BUDGET - (time.time() - t0)
            if remain < 3:
                break
            secs = min(3.0, remain - 1.0)
            try:
                xt2 = lp_polish(xt, secs)
            except Exception:
                xt2 = xt
            v2 = max_cos(xt2)
            if v2 < best_val:
                x_best = xt2
                best_val = v2
        else:
            if v < best_val:
                x_best = xt
                best_val = v
        if time.time() - t0 > BUDGET - 25:
            break

    # final polish of champion
    remain = BUDGET - (time.time() - t0)
    if remain > 4:
        try:
            x_best = lp_polish(x_best, min(remain - 1.0, 20.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
