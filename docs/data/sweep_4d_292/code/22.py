import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    pm = G.max(axis=1)
    m = pm.max()
    deg = (G >= m - 1e-7).sum(axis=1)
    return m, pm, deg, G

def _anneal_row(x, steps, beta0, beta1, lr0, mu, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v -= vd * x
        x = x - lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _anneal_global(x, steps, beta0, beta1, lr0):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lrb = np.log(beta1 / beta0) if beta0 > 0 else 0.0
    for k in range(steps):
        beta = beta0 * np.exp(lrb * (k / max(steps - 1, 1)))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S < 1e-30:
            S = 1e-30
        Gr = (W @ x) / S
        dot = (Gr * x).sum(axis=1, keepdims=True)
        Gp = Gr - dot * x
        m1 = b1 * m1 + (1 - b1) * Gp
        v1 = b2 * v1 + (1 - b2) * (Gp * Gp)
        mh = m1 / (1 - b1 ** (k + 1))
        vh = v1 / (1 - b2 ** (k + 1))
        step = lr0 * mh / (np.sqrt(vh) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.where(sn > 0.05, np.minimum(1.0, 0.05 / sn), 1.0)
        x = x - step
        x = _norm(x)
    return x

def _cheap_descent(x, steps=140, lr0=0.01, tol0=0.004):
    n, d = x.shape
    lr = lr0
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        band = tol0 * (1.0 - 0.8 * t / max(1, steps - 1))
        Mask = G > m - band
        deg = Mask.sum(axis=1)
        has = deg > 0
        if not np.any(has):
            break
        W = Mask.astype(float)
        D = np.maximum(deg, 1)[:, None]
        A = (W @ x) / D
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        Gt[~has] = 0.0
        x = x - lr * Gt
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        lr *= 0.998
    return x

def _lp_polish(x, seconds, hard_deadline=1e100):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < hard_deadline:
        if radius < 1e-14:
            break
        C = (x @ x.T)[iu]
        m = C.max()
        sel = C > m - max(20 * radius, 1e-9)
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
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _find_holes(Xkeep, k, rng, n_cand=24000):
    cand = _norm(rng.normal(size=(n_cand, 4)))
    best = np.empty(n_cand)
    cs = 5000
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    idx = np.argsort(best)
    sel = []
    for i in idx:
        if len(sel) >= k:
            break
        c = cand[i]
        ok = True
        for s in sel:
            if float(c @ s) > 0.90:
                ok = False
                break
        if ok:
            sel.append(c)
    if len(sel) < k:
        for i in idx:
            if len(sel) >= k:
                break
            c = cand[i]
            dup = False
            for s in sel:
                if float(c @ s) > 0.999:
                    dup = True
                    break
            if not dup:
                sel.append(c)
    return np.array(sel)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    explore_end = t0 + 132.0
    rng = np.random.default_rng(7)
    xb = None
    bv = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (292, 4):
                a = _norm(a)
                v = _max_cos(a)
                if v < bv:
                    bv = v
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.normal(size=(292, 4)))
        bv = _max_cos(xb)
    x_best = xb.copy()
    best = float(bv)
    n, d = 292, 4
    # cached contact info of best
    try:
        _, permax_best, deg_best, _ = _per_point(x_best)
    except Exception:
        permax_best = np.zeros(n)
        deg_best = np.zeros(n, dtype=int)
    order_worst = np.argsort(-permax_best)
    order_deg_hi = np.argsort(-deg_best)
    order_deg_lo = np.argsort(deg_best)

    sigmas = [0.004, 0.012, 0.03, 0.07, 0.15, 0.30]
    it = 0
    while time.time() < explore_end:
        it += 1
        if deadline - time.time() < 9:
            break
        X = x_best.copy()
        # global small shake
        sigma = sigmas[(it - 1) % len(sigmas)] if (it % 3 != 0) else rng.choice(sigmas)
        # randomize sigma a bit
        sigma = float(sigma * (0.7 + 0.6 * rng.random()))
        X = _norm(X + sigma * rng.normal(size=X.shape))
        r = rng.random()
        k = 0
        if r < 0.62:
            # kill-and-fill
            if rng.random() < 0.35:
                k = int(rng.integers(1, 4))
            elif rng.random() < 0.7:
                k = int(rng.integers(4, 9))
            else:
                k = int(rng.integers(9, 15))
            k = min(k, 14)
            victims = set()
            # always include worst point and max-degree point
            victims.add(int(order_worst[0]))
            victims.add(int(order_deg_hi[0]))
            # rattlers (0 contacts) prioritized
            ratt = np.where(deg_best == 0)[0]
            for rr in ratt:
                if len(victims) >= k:
                    break
                victims.add(int(rr))
            # fill rest: mix of hi-deg, lo-deg, worst, random
            while len(victims) < k:
                u = rng.random()
                if u < 0.3:
                    victims.add(int(order_deg_hi[int(rng.integers(0, min(n, 30)))]))
                elif u < 0.55:
                    victims.add(int(order_deg_lo[int(rng.integers(0, min(n, 30)))]))
                elif u < 0.75:
                    victims.add(int(order_worst[int(rng.integers(0, min(n, 30)))]))
                else:
                    victims.add(int(rng.integers(0, n)))
            victims = np.array(sorted(victims)[:k], dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = X[keep_mask]
            try:
                holes = _find_holes(Xkeep, len(victims), rng, n_cand=24000)
            except Exception:
                holes = None
            if holes is not None and holes.shape[0] == len(victims):
                X[victims] = holes + 0.003 * rng.normal(size=(len(victims), 4))
                X = _norm(X)
            else:
                X[victims] = _norm(rng.normal(size=(len(victims), 4)))
                X = _norm(X)
        elif r < 0.82:
            # jammed neighbourhood shake (no kill)
            try:
                w = int(order_deg_hi[0]) if rng.random() < 0.5 else int(order_worst[0])
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m = G.max()
                neigh = np.where(G[w] > m - 0.004)[0]
                # include second order a bit
                X[w] += 0.12 * rng.normal(size=(4,))
                if len(neigh) > 0:
                    X[neigh] += 0.08 * rng.normal(size=(len(neigh), 4))
                # a few random others small kick
                ri = rng.choice(n, size=min(6, n), replace=False)
                X[ri] += 0.02 * rng.normal(size=(len(ri), 4))
                X = _norm(X)
            except Exception:
                pass
        # else pure shake already done
        # choose anneal hyperparams based on sigma/k
        if sigma > 0.1 or k >= 9:
            steps, b0, b1, lr0, mu = 850, 15.0, 500.0, 0.0045, 0.88
            use_global = False
        elif sigma > 0.04 or k >= 4:
            steps, b0, b1, lr0, mu = 700, 20.0, 450.0, 0.004, 0.87
            use_global = (it % 2 == 0)
        else:
            steps, b0, b1, lr0, mu = 550, 25.0, 400.0, 0.0035, 0.85
            use_global = (it % 3 == 0)
        try:
            if use_global:
                Xopt = _anneal_global(X, steps, b0, b1, lr0)
            else:
                Xopt = _anneal_row(X, steps, b0, b1, lr0, mu)
            Xopt = _cheap_descent(Xopt, steps=130, lr0=0.010, tol0=0.005)
        except Exception:
            continue
        try:
            v = _max_cos(Xopt)
        except Exception:
            continue
        if v < best - 2e-6:
            rem = min(6.0, max(1.0, deadline - time.time() - 28.0))
            try:
                Xpol = _lp_polish(Xopt, seconds=rem, hard_deadline=deadline - 24.0)
                vp = _max_cos(Xpol)
            except Exception:
                Xpol, vp = Xopt, v
            if vp < best:
                x_best, best = Xpol, vp
                try:
                    _, permax_best, deg_best, _ = _per_point(x_best)
                    order_worst = np.argsort(-permax_best)
                    order_deg_hi = np.argsort(-deg_best)
                    order_deg_lo = np.argsort(deg_best)
                except Exception:
                    pass
        elif v < best + 0.001 and (it % 6 == 0) and time.time() < explore_end - 8:
            rem = min(2.2, max(0.7, deadline - time.time() - 28.0))
            if rem >= 0.7:
                try:
                    Xpol = _lp_polish(Xopt, seconds=rem, hard_deadline=deadline - 24.0)
                    vp = _max_cos(Xpol)
                    if vp < best:
                        x_best, best = Xpol, vp
                        try:
                            _, permax_best, deg_best, _ = _per_point(x_best)
                            order_worst = np.argsort(-permax_best)
                            order_deg_hi = np.argsort(-deg_best)
                            order_deg_lo = np.argsort(deg_best)
                        except Exception:
                            pass
                except Exception:
                    pass
        if time.time() > explore_end:
            break
    rem = deadline - time.time()
    if rem > 3:
        try:
            x_best = _lp_polish(x_best, seconds=min(rem - 1.0, 38.0), hard_deadline=deadline - 1.0)
        except Exception:
            pass
    x_best = _norm(np.nan_to_num(x_best, nan=0.0, posinf=1.0, neginf=-1.0))
    x_best = _norm(x_best)
    return x_best
