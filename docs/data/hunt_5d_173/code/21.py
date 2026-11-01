import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, seconds, r0=8e-3):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cfull = G[iu]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        cnt_sel = int(sel.sum())
        if cnt_sel == 0:
            radius *= 0.5
            continue
        if cnt_sel > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = cfull[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth_polish(X, iters=350, a0=25.0, a1=140.0, lr=0.03):
    X = _norm(X)
    n, d = X.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        alpha = a0 + (a1 - a0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        # stabilize: subtract row-wise max? use global max
        gm = G.max()
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        grad = alpha * (W @ X)
        # tangent projection
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        # adaptive step: normalize by max norm
        gn = np.linalg.norm(gt, axis=1, keepdims=True).max()
        if gn < 1e-14 or not np.isfinite(gn):
            continue
        st = lr / (1.0 + alpha * 0.02)
        X = X - st * gt / (gn + 1e-12) * np.sqrt(n)
        X = _norm(X)
    return X

def _find_void(X, exclude, rng, ns=25000):
    n, d = X.shape
    Z = rng.normal(size=(ns, d))
    Z = _norm(Z)
    if exclude is not None:
        mask = np.ones(n, dtype=bool)
        mask[exclude] = False
        Xo = X[mask]
    else:
        Xo = X
    dots = Z @ Xo.T
    md = dots.max(axis=1)
    bi = int(np.argmin(md))
    z = Z[bi].copy()
    # local refine: push away from nearest points
    for _ in range(120):
        dp = Xo @ z
        m = dp.max()
        # average of near-max points
        sel = dp > m - 1e-3
        if int(sel.sum()) == 0:
            break
        c = Xo[sel].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        g = c - (c @ z) * z
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            # random tangent step
            r = rng.normal(size=(d,))
            r = r - (r @ z) * z
            r = r / (np.linalg.norm(r) + 1e-12)
            z = z + 0.02 * r
        else:
            z = z - 0.06 * g / gn
        z = z / np.linalg.norm(z)
    return z

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    # quick confirm polish (short, already optimal usually)
    if deadline - time.time() > 12:
        best = _lp_polish(best, 2.0)
        best_val = _max_cos(best)

    def contacts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G >= m - 1e-9
        cnt = mask.sum(axis=1)
        return m, cnt, G

    trial = 0
    stuck = 0
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        trial += 1
        m, cnt, G = contacts(best)
        order_desc = np.argsort(-cnt)
        order_asc = np.argsort(cnt)
        mode = trial % 7
        y = best.copy()
        if mode == 0:
            # rattler to void
            ratt = np.where(cnt == 0)[0]
            victim = int(ratt[0]) if len(ratt) > 0 else int(order_asc[0])
            # re-pick random rattler if many
            if len(ratt) > 1:
                victim = int(ratt[rng.integers(len(ratt))])
            z = _find_void(best, exclude=victim, rng=rng, ns=25000)
            y[victim] = z
            # shake neighbours of void slightly? no
        elif mode == 1:
            # kill most jammed -> void
            victim = int(order_desc[0])
            z = _find_void(best, exclude=victim, rng=rng, ns=25000)
            y[victim] = z
            # also jitter its old neighbours
            nb = np.where(G[victim] > m - 1e-7)[0]
            if len(nb) > 0:
                k = min(len(nb), 4)
                pick = nb[rng.choice(len(nb), k, replace=False)]
                y[pick] += 0.04 * rng.normal(size=(k, d))
        elif mode == 2:
            # cluster kick around jammed core
            center = int(order_desc[rng.integers(min(5, n))])
            dp = best @ best[center]
            nn = np.argsort(-dp)[:8]
            sigma = 0.10 + 0.05 * min(stuck / 10.0, 2.0)
            y[nn] += sigma * rng.normal(size=(len(nn), d))
        elif mode == 3:
            # random subset medium kick
            k = 6 + int(rng.integers(0, 6))
            idx = rng.choice(n, k, replace=False)
            sigma = 0.20 if (trial % 14 == 3) else 0.07
            y[idx] += sigma * rng.normal(size=(k, d))
        elif mode == 4:
            # small global shake
            sigma = 0.012 + 0.004 * (stuck / 20.0 if stuck < 40 else 2.0)
            y += sigma * rng.normal(size=(n, d))
        elif mode == 5:
            # two-point swap + void: pick two high-contact, reinit one to void, jitter other
            v1 = int(order_desc[0])
            v2 = int(order_desc[1])
            z = _find_void(best, exclude=v1, rng=rng, ns=15000)
            y[v1] = z + 0.02 * rng.normal(size=(d,))
            y[v2] += 0.08 * rng.normal(size=(d,))
        else:
            # large shake of random 12
            k = 12
            idx = rng.choice(n, k, replace=False)
            y[idx] += 0.35 * rng.normal(size=(k, d))
        y = _norm(y)
        # smooth repulsion pre-polish (fast, no LP)
        pt_smooth = 300 if rem > 30 else 150
        y = _smooth_polish(y, iters=pt_smooth, a0=30.0, a1=150.0, lr=0.05)
        # LP polish
        pt = 1.6
        if rem < 25:
            pt = max(0.7, rem - 8.5)
        pt = min(pt, rem - 8.0)
        if pt < 0.4:
            break
        y = _lp_polish(y, pt)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            best = y
            best_val = v
            stuck = 0
        else:
            stuck += 1
            # sideways move if very close to allow topology drift
            if v < best_val + 1e-10 and (trial % 11 == 0) and stuck > 8:
                best = y
                best_val = v
                stuck = 0
    # final polish on remaining time
    rem = deadline - time.time()
    if rem > 2.0:
        best = _lp_polish(best, max(1.0, rem - 1.0), r0=5e-3)
    best = _norm(best)
    return best
