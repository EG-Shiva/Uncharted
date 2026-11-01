import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish_deadline(x, deadline, init_radius=0.004):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        ns = int(np.sum(sel))
        if ns == 0:
            radius *= 0.5
            continue
        if ns > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[-4000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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

def _smooth_anneal(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    X = x.copy()
    # geometric T schedule, linear lr schedule
    lT0, lT1 = np.log(T0), np.log(T1)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(lr0 * (1 - frac) + lr1 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # stable exp
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        Z = float(np.sum(E))
        if not np.isfinite(Z) or Z <= 0:
            break
        W = E / Z
        grad = 2.0 * (W @ X)
        # tangent projection
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        # control step size: limit max row norm
        nr = np.linalg.norm(gt, axis=1)
        mx = float(np.max(nr))
        if mx > 1e-12:
            # scale so that lr*mx ~ target displacement
            X = X - lr * gt
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        # occasional re-center not needed
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _best_hole(Y, rng, n_try=12000, refine=250, lr0=0.05):
    # Y: (m,5) existing points, find point minimizing max dot
    d = Y.shape[1]
    cand = rng.normal(size=(n_try, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Y.T
    worst = np.max(dots, axis=1)
    bi = int(np.argmin(worst))
    c = cand[bi].copy()
    lr = lr0
    for it in range(refine):
        dv = Y @ c  # (m,)
        j = int(np.argmax(dv))
        g = Y[j] - float(dv[j]) * c
        # also soften with top-k average for stability
        # every 50 steps use softmax gradient
        if it % 50 == 49:
            # softmax step
            mval = float(np.max(dv))
            w = np.exp((dv - mval) / 0.005)
            w /= np.sum(w)
            g2 = w @ Y - float(w @ Y @ c) * c if False else (w @ Y)
            # project
            g2 = g2 - float(np.dot(g2, c)) * c
            g = g2
        c = c - lr * g
        nrm = float(np.linalg.norm(c))
        if nrm == 0 or not np.isfinite(nrm):
            c = cand[bi].copy()
            lr *= 0.5
            continue
        c /= nrm
        lr *= (1.0 - 0.002)
    return c / np.linalg.norm(c)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline_all = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    def contact_counts(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnt = np.sum(G >= m - tol, axis=1)
        return m, cnt, G

    trial = 0
    while True:
        rem_all = deadline_all - time.time()
        if rem_all < 9.0:
            break
        mode = trial % 7
        _, cnt, Gcur = contact_counts(best, tol=1e-7)
        order_desc = np.argsort(-cnt, kind="stable")
        order_asc = np.argsort(cnt, kind="stable")
        Y = best.copy()
        if mode == 0:
            # single worst kill -> hole
            victim = int(order_desc[0])
            keep = np.delete(np.arange(n), victim)
            hole = _best_hole(Y[keep], rng, n_try=15000, refine=300)
            Y[victim] = hole
            # jitter neighbours of victim slightly
            neigh = np.where(Gcur[victim] > float(np.max(Gcur[victim])) - 0.03)[0]
            neigh = neigh[neigh != victim]
            if len(neigh) > 0:
                Y[neigh] += 0.02 * rng.normal(size=(len(neigh), d))
        elif mode == 1:
            # double worst kill sequential holes
            victims = [int(order_desc[0]), int(order_desc[1])]
            for vi in victims:
                keep = np.delete(np.arange(n), vi)
                # hole w.r.t. current Y without vi (use updated Y)
                Ytmp = np.delete(Y, vi, axis=0)
                hole = _best_hole(Ytmp, rng, n_try=10000, refine=200)
                Y[vi] = hole
            # small shake of top-8 jammed
            sub = order_desc[:8]
            Y[sub] += 0.015 * rng.normal(size=(len(sub), d))
        elif mode == 2:
            # rattler relocation + random hole
            ratt = np.where(cnt == 0)[0]
            if len(ratt) == 0:
                ratt = order_asc[:2]
            for vi in ratt[:2]:
                vi = int(vi)
                Ytmp = np.delete(Y, vi, axis=0)
                hole = _best_hole(Ytmp, rng, n_try=15000, refine=300)
                Y[vi] = hole
            # also move one random point to second hole
            vr = int(rng.choice(n))
            Ytmp = np.delete(Y, vr, axis=0)
            hole2 = _best_hole(Ytmp, rng, n_try=8000, refine=150)
            # mix to keep diversity
            if rng.random() < 0.5:
                Y[vr] = hole2
        elif mode == 3:
            # jammed cluster shake medium
            sub = order_desc[:12]
            Y[sub] += 0.05 * rng.normal(size=(len(sub), d))
            sub2 = rng.choice(n, 6, replace=False)
            Y[sub2] += 0.08 * rng.normal(size=(6, d))
        elif mode == 4:
            # global small shake (different basin via annealing)
            Y += 0.012 * rng.normal(size=(n, d))
        elif mode == 5:
            # broad 25-point moderate shake
            sub = rng.choice(n, 25, replace=False)
            Y[sub] += 0.035 * rng.normal(size=(len(sub), d))
        else:
            # random single reinit + neighbour shake (strong topology change)
            victim = int(rng.choice(n))
            # bias to worst half
            if rng.random() < 0.7:
                victim = int(rng.choice(order_desc[:40]))
            Y[victim] = rng.normal(size=(d,))
            Y[victim] /= np.linalg.norm(Y[victim])
            G2 = Y @ Y.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            neigh = np.where(G2[victim] > m2 - 0.05)[0]
            if len(neigh) > 0:
                Y[neigh] += 0.04 * rng.normal(size=(len(neigh), d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # smooth anneal params per mode
        if mode in (0, 1, 2):
            Ys = _smooth_anneal(Y, iters=500, T0=0.012, T1=0.0025, lr0=4.0, lr1=0.6)
        elif mode == 4:
            Ys = _smooth_anneal(Y, iters=700, T0=0.02, T1=0.002, lr0=5.0, lr1=0.5)
        else:
            Ys = _smooth_anneal(Y, iters=400, T0=0.01, T1=0.002, lr0=3.0, lr1=0.5)
        # LP polish with limited time
        rem_all2 = deadline_all - time.time()
        if rem_all2 < 7.0:
            break
        pt = 2.2
        if rem_all2 < 12:
            pt = max(0.8, rem_all2 - 6.5)
        lp_dead = time.time() + pt
        init_r = 0.006 if mode in (0, 1, 2, 6) else 0.003
        Yp = _lp_polish_deadline(Ys, lp_dead, init_radius=init_r)
        v = _max_cos(Yp)
        if v < best_val - 1e-12:
            best = Yp
            best_val = v
        trial += 1

    rem = deadline_all - time.time()
    if rem > 1.5:
        best = _lp_polish_deadline(best, deadline_all - 0.5, init_radius=0.003)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
