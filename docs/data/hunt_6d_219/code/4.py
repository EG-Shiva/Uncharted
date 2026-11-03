import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    # precompute iu arrays
    while time.time() - t0 < seconds and time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        thr = max(20.0 * radius, 1e-9)
        sel = cfull > m - thr
        # cap constraints for speed
        if int(sel.sum()) > 5000:
            # keep top 5000
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:5000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        c = cfull[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_adam(x, iters, tau0, tau1, lr):
    n, d = x.shape
    mmt = np.zeros_like(x)
    vmt = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ltau0 = np.log(tau0)
    ltau1 = np.log(tau1)
    for it in range(1, iters + 1):
        tau = float(np.exp(ltau0 + (ltau1 - ltau0) * (it / iters)))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m0 = float(G.max())
        Gs = (G - m0) / tau
        np.clip(Gs, -50.0, 0.0, out=Gs)
        E = np.exp(Gs)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        grad = W @ x
        # tangent projection (ascent dir)
        g = grad - (np.sum(grad * x, axis=1, keepdims=True)) * x
        mmt = b1 * mmt + (1.0 - b1) * g
        vmt = b2 * vmt + (1.0 - b2) * (g * g)
        mhat = mmt / (1.0 - b1 ** it)
        vhat = vmt / (1.0 - b2 ** it)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        x = x - step
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def solve():
    t_start = time.time()
    deadline = t_start + 172.0
    rng = np.random.default_rng(0)

    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        xb = np.load("record.npy").astype(np.float64)
    if xb.shape != (219, 6):
        # fallback: truncate / pad deterministically
        flat = xb.reshape(-1, 6)
        if flat.shape[0] >= 219:
            xb = flat[:219].copy()
        else:
            extra = rng.standard_normal(size=(219 - flat.shape[0], 6))
            xb = np.vstack([flat, extra])
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    vb = _max_cos(xb)
    x_best = xb.copy()
    v_best = vb

    def remaining():
        return deadline - time.time()

    # Phase A: direct squeeze of loaded best
    if remaining() > 25:
        xa = _smooth_adam(x_best.copy(), iters=1200, tau0=0.004, tau1=0.0007, lr=0.006)
        va = _max_cos(xa)
        # short LP only if promising
        if va < v_best + 0.002 and remaining() > 15:
            xa = _lp_polish(xa, seconds=min(10.0, remaining() - 8.0), deadline=deadline)
            va = _max_cos(xa)
        if va < v_best:
            x_best, v_best = xa, va

    # Phase B: iterated kicks + reinsertion + smooth + LP
    sigmas = [0.004, 0.008, 0.002, 0.012, 0.006]
    trial = 0
    while remaining() > 30:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        xt = x_best + sigma * rng.standard_normal(size=x_best.shape)
        xt = xt / np.linalg.norm(xt, axis=1, keepdims=True)

        # every 4th trial: greedy worst-point reinsertion
        if trial % 4 == 0:
            try:
                G = xt @ xt.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                w = int(np.argmax(rowmax))
                keep = np.delete(xt, w, axis=0)
                ncand = 20000
                cand = rng.standard_normal(size=(ncand, 6))
                cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
                D = cand @ keep.T
                scores = D.max(axis=1)
                bi = int(np.argmin(scores))
                xt[w] = cand[bi]
            except Exception:
                pass

        # smooth descent (cheaper when little time left)
        if remaining() < 60:
            iters = 800
        else:
            iters = 1500
        xt = _smooth_adam(xt, iters=iters, tau0=0.005, tau1=0.0007, lr=0.008)
        vs = _max_cos(xt)
        if vs > v_best + 0.002:
            continue
        if remaining() < 12:
            if vs < v_best:
                x_best, v_best = xt, vs
            continue
        xt = _lp_polish(xt, seconds=min(6.0, remaining() - 10.0), deadline=deadline)
        v = _max_cos(xt)
        if v < v_best:
            x_best, v_best = xt, v

    # Phase C: final equalization polish
    if remaining() > 5:
        x_best = _lp_polish(x_best, seconds=max(1.0, remaining() - 2.0), deadline=deadline)

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
