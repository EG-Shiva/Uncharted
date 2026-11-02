import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _adam_quench(X_init, steps, K0, K1, lr):
    X = X_init.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        K = K0 * (K1 / K0) ** frac
        lr_t = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        gmax = G.max()
        A = np.exp(K * (G - gmax))
        np.fill_diagonal(A, 0.0)
        S = A.sum()
        if not np.isfinite(S) or S <= 0:
            break
        P = A / S
        grad = 2.0 * (P @ X)
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * vv + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        step = lr_t * mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / sn)
        step = step * clip
        X = X - step
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-18)
    return X

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csv = c[sel]
        p = len(I)
        if p > 900:
            idx = np.argpartition(-csv, 900)[:900]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 900
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _best_void(X_rem, K, rng):
    d = X_rem.shape[1]
    C = rng.randn(K, d)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D = C @ X_rem.T
    rm = D.max(axis=1)
    return C[int(np.argmin(rm))]

def solve():
    t0 = time.time()
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.asarray(c, dtype=np.float64)
                if a.shape != (54, 21):
                    flat = a.ravel()
                    need = 54 * 21
                    if flat.size < need:
                        flat = np.concatenate([flat, rng.randn(need - flat.size)])
                    a = flat[:need].reshape(54, 21)
                a = _proj(a)
                cands.append(a)
            except Exception:
                pass
    if not cands:
        cands.append(_proj(rng.randn(54, 21)))
    best = min(cands, key=_max_cos)
    best_val = _max_cos(best)
    n, d = 54, 21

    # quick contact stats helper inline
    it = 0
    # elite pool
    elites = [best.copy()]
    elite_vals = [best_val]

    K1s = [700.0, 1000.0, 1400.0, 600.0]
    lrs = [0.004, 0.006, 0.009]

    # hopping budget: leave ~25s for final LP
    while True:
        if time.time() - t0 > 132:
            break
        if it > 400:
            break
        # choose base: mostly best, sometimes 2nd elite for diversity
        if len(elites) > 1 and (it % 6 == 5):
            base = elites[1]
        else:
            base = best
        mode = it % 8

        G = base @ base.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(rowmax)  # ascending, worst last

        Y = base.copy()
        if mode == 0:
            # large global shake
            Y = Y + 0.30 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 900, K1s[it % 4], 0.006
        elif mode == 1:
            # collective cluster shift: move kNN cluster together
            seed = int(order[-1 - (it % 5)]) if it % 2 == 0 else int(rng.randint(n))
            dots = base[seed] @ base.T
            k = 8
            cl = np.argsort(-dots)[:k]
            common = 0.28 * rng.randn(d)
            Y[cl] += common
            Y[cl] += 0.03 * rng.randn(len(cl), d)
            mask = np.ones(n, bool); mask[cl] = False
            Y[mask] += 0.012 * rng.randn(mask.sum(), d)
            Y = _proj(Y)
            steps, K1, lr = 1000, 900.0, 0.005
        elif mode == 2:
            # delete 3 worst, refill at voids
            k = 3
            victims = order[-k:]
            # sequential greedy refill
            keep = np.ones(n, bool); keep[victims] = False
            for v in victims:
                rem = Y[~np.eye(n, dtype=bool)[v].__invert__() if False else keep]
                # build remaining set: all except v (using current Y)
                idx = np.ones(n, bool); idx[v] = False
                Xrem = Y[idx]
                nv = _best_void(Xrem, 3000, rng)
                Y[v] = nv
                keep[v] = True  # now filled, next uses updated
                keep[:] = True
                # mark remaining victims as still to fill? just continue
            Y += 0.015 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 1100, K1s[(it + 1) % 4], lrs[it % 3]
        elif mode == 3:
            Y = Y + 0.065 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 700, 1200.0, 0.004
        elif mode == 4:
            # correlated subset move, size 12
            k = 12
            perm = rng.permutation(n)[:k]
            common = 0.20 * rng.randn(d)
            Y[perm] += common + 0.05 * rng.randn(k, d)
            Y = _proj(Y)
            steps, K1, lr = 1000, 800.0, 0.006
        elif mode == 5:
            # delete 5: 3 worst + 2 random, refill
            victims = list(order[-3:]) + list(rng.choice(n, 2, replace=False))
            for v in victims:
                idx = np.ones(n, bool); idx[v] = False
                Xrem = Y[idx]
                nv = _best_void(Xrem, 2000, rng)
                Y[v] = nv
            Y += 0.02 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 1100, 1000.0, 0.005
        elif mode == 6:
            # bridge to random for new basin
            R = rng.randn(n, d); R = _proj(R)
            alpha = 0.45
            Y = (1 - alpha) * base + alpha * R + 0.05 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 1200, 700.0, 0.007
        else:
            # single worst best-hole reinsertion
            worst = int(order[-1])
            idx = np.ones(n, bool); idx[worst] = False
            Xrem = Y[idx]
            Y[worst] = _best_void(Xrem, 5000, rng)
            Y += 0.012 * rng.randn(*Y.shape)
            Y = _proj(Y)
            steps, K1, lr = 800, 1300.0, 0.004

        Y2 = _adam_quench(Y, steps, 70.0, K1, lr)
        v = _max_cos(Y2)
        if v < best_val:
            # confirm with short LP to bottom of basin
            if time.time() - t0 < 125:
                Y3 = _lp_polish(Y2, 2.0)
                v3 = _max_cos(Y3)
                if v3 < best_val:
                    best, best_val = Y3, v3
                    elites.append(best.copy()); elite_vals.append(best_val)
                elif v < best_val:
                    best, best_val = Y2, v
                    elites.append(best.copy()); elite_vals.append(best_val)
            else:
                best, best_val = Y2, v
                elites.append(best.copy()); elite_vals.append(best_val)
            # keep elite pool small, sorted
            o = np.argsort(np.array(elite_vals))
            elites = [elites[i] for i in o[:4]]
            elite_vals = [elite_vals[i] for i in o[:4]]
        else:
            # keep diversity: store near-best distinct configs
            if v < (elite_vals[-1] if len(elite_vals) >= 4 else best_val + 0.004):
                if len(elites) < 4:
                    elites.append(Y2.copy()); elite_vals.append(v)
                else:
                    if v < max(elite_vals):
                        j = int(np.argmax(elite_vals))
                        elites[j] = Y2.copy(); elite_vals[j] = v
        it += 1

    remain = 172 - (time.time() - t0)
    if remain > 6:
        budget = min(remain - 2, 28)
        best = _lp_polish(best, budget)
        best_val = _max_cos(best)
    best = _proj(best)
    return best
