import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _adam_smooth(X0, steps, tau0, tau1, lr0, lr1, rng, noise0=0.0,
                 b1=0.9, b2=0.999, eps=1e-8):
    X = _norm(X0.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    ltau = np.log(tau0)
    ltau1 = np.log(tau1)
    llr0 = np.log(lr0)
    llr1 = np.log(lr1)
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        tau = float(np.exp(ltau + (ltau1 - ltau) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        S = G / tau
        m0 = float(S.max())
        # exp in place
        E = np.exp(S - m0)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum())
        if s <= 0:
            s = 1.0
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - sdot * X
        if noise0 > 0:
            nz = rng.standard_normal(size=X.shape)
            nd = np.sum(nz * X, axis=1, keepdims=True)
            nz -= nd * X
            step_t += (noise0 * (1.0 - frac) * lr) * nz
        X -= step_t
        X = _norm(X)
    return X


def _refine_hole(u0, Xr, steps, tau0, tau1, lr0, lr1, rng):
    u = u0 / (np.linalg.norm(u0) + 1e-300)
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    ltau, ltau1 = np.log(tau0), np.log(tau1)
    llr0, llr1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        tau = float(np.exp(ltau + (ltau1 - ltau) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        s = (Xr @ u) / tau
        m0 = float(s.max())
        e = np.exp(s - m0)
        w = e / (float(e.sum()) + 1e-300)
        g = (w[:, None] * Xr).sum(axis=0)
        gt = g - float(np.dot(g, u)) * u
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - float(np.dot(step, u)) * u
        u = u - step
        u = u / (np.linalg.norm(u) + 1e-300)
    return u


def _best_hole(Xr, rng, n_sample=12000):
    d = Xr.shape[1]
    C = rng.standard_normal(size=(n_sample, d))
    C = _norm(C)
    D = C @ Xr.T
    scores = D.max(axis=1)
    idx = np.argsort(scores)[:8]
    best_u = None
    best_s = 1e9
    for k in idx:
        u0 = C[k]
        u = _refine_hole(u0, Xr, steps=250, tau0=0.015, tau1=0.002,
                         lr0=0.01, lr1=0.001, rng=rng)
        s = float((Xr @ u).max())
        if s < best_s:
            best_s = s
            best_u = u
    return best_u, best_s


def _contact_counts(X, thresh=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - thresh).sum(axis=1)
    return m, cnt, G


def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 173, 5

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = _norm(a)
                vv = _max_cos(an)
                if vv < best_val:
                    best_val = vv
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)

    # quick annealed polish of the incumbent to ensure local opt baseline
    try:
        if deadline - time.time() > 20:
            cand = _adam_smooth(best, steps=900, tau0=0.02, tau1=0.002,
                                lr0=0.006, lr1=0.0008, rng=rng)
            vv = _max_cos(cand)
            if vv < best_val:
                best_val = vv
                best = cand.copy()
    except Exception:
        pass

    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 12.0:
            break
        # adaptive polish length
        if rem < 30:
            steps = 600
        elif rem < 80:
            steps = 900
        else:
            steps = 1100
        mode = trial % 8
        y = best.copy()
        try:
            _, cnt, Gfull = _contact_counts(best)
            order_desc = np.argsort(-cnt)
            order_asc = np.argsort(cnt)
            if mode == 0:
                y = y + 0.012 * rng.standard_normal(size=y.shape)
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.025, tau1=0.002,
                                    lr0=0.007, lr1=0.0007, rng=rng)
            elif mode == 1:
                y = y + 0.035 * rng.standard_normal(size=y.shape)
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.03, tau1=0.002,
                                    lr0=0.008, lr1=0.0007, rng=rng, noise0=0.6)
            elif mode == 2:
                # kick jammed points hard
                k = 8
                victims = order_desc[:k]
                y[victims] += 0.12 * rng.standard_normal(size=(k, d))
                # small global shake
                y += 0.008 * rng.standard_normal(size=y.shape)
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.03, tau1=0.0018,
                                    lr0=0.008, lr1=0.0007, rng=rng)
            elif mode == 3:
                # relocate single worst point to largest hole
                victim = int(order_desc[0])
                keep = np.ones(n, dtype=bool)
                keep[victim] = False
                Xr = best[keep]
                u, _ = _best_hole(Xr, rng, n_sample=12000)
                y[victim] = u + 0.005 * rng.standard_normal(size=(d,))
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.02, tau1=0.0015,
                                    lr0=0.006, lr1=0.0006, rng=rng)
            elif mode == 4:
                # relocate two worst to holes sequentially
                victims = [int(order_desc[0]), int(order_desc[1])]
                keep = np.ones(n, dtype=bool)
                keep[victims] = False
                Xr = best[keep]
                u1, _ = _best_hole(Xr, rng, n_sample=10000)
                # add u1 to set for second hole
                Xr2 = np.vstack([Xr, u1[None, :]])
                u2, _ = _best_hole(Xr2, rng, n_sample=10000)
                y[victims[0]] = u1
                y[victims[1]] = u2
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.02, tau1=0.0015,
                                    lr0=0.006, lr1=0.0006, rng=rng)
            elif mode == 5:
                # reinit 5 worst randomly
                victims = order_desc[:5]
                y[victims] = rng.standard_normal(size=(5, d))
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps + 300, tau0=0.035, tau1=0.002,
                                    lr0=0.009, lr1=0.0007, rng=rng)
            elif mode == 6:
                # fresh random restart with long anneal
                y = _norm(rng.standard_normal(size=(n, d)))
                cand = _adam_smooth(y, steps=min(steps + 800, 1900),
                                    tau0=0.04, tau1=0.002,
                                    lr0=0.010, lr1=0.0008, rng=rng)
            else:
                # large global shake
                y = y + 0.08 * rng.standard_normal(size=y.shape)
                y = _norm(y)
                cand = _adam_smooth(y, steps=steps, tau0=0.03, tau1=0.002,
                                    lr0=0.008, lr1=0.0007, rng=rng, noise0=0.3)
            vv = _max_cos(cand)
            if vv < best_val - 1e-12:
                best_val = vv
                best = cand.copy()
        except Exception:
            pass
        trial += 1

    # final snap to local optimum
    rem = deadline - time.time()
    if rem > 3.0:
        try:
            # short fine Adam with tiny tau then LP
            ft = min(4.0, max(1.0, rem - 6.0))
            # use part of remaining for Adam
            cand = _adam_smooth(best, steps=500, tau0=0.006, tau1=0.0009,
                                lr0=0.002, lr1=0.0003, rng=rng)
            vv = _max_cos(cand)
            if vv < best_val:
                best_val = vv
                best = cand.copy()
        except Exception:
            pass
    rem = deadline - time.time()
    if rem > 2.0:
        try:
            best = _lp_polish(best, seconds=max(0.5, rem - 0.8))
            best_val = _max_cos(best)
        except Exception:
            pass

    best = _norm(best)
    return best
