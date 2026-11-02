import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, b0, b1, lr0, lr1, clip=0.04):
    n, d = X.shape
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        step = lr * Gt
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        # scale only over-limit rows
        if np.any(over):
            step = np.where(over, step * (clip / sn), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _single_opt(xv, Y, steps=120, b0=200.0, b1=2500.0, lr0=0.08, lr1=0.01):
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    x = xv.copy()
    nx = np.linalg.norm(x)
    if nx == 0:
        x = Y[0] * -1.0
        nx = np.linalg.norm(x)
    x = x / nx
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ x
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        g = w @ Y
        gd = float(g @ x)
        gt = g - gd * x
        x = x - lr * gt
        n2 = np.linalg.norm(x)
        if n2 == 0 or not np.isfinite(n2):
            break
        x = x / n2
    return x

def _lp_capped(X, budget, deadline, max_pairs=450, max_iters=10):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= budget:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > max_iters:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c, kind='stable')
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < max_pairs:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= max_pairs:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list, dtype=np.int64)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    TRIAL_DEADLINE = t0 + 132.0
    FINAL_END = t0 + 168.0
    rng = np.random.default_rng(0)
    n, d = 58, 22
    incumbent = None
    inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (n, d):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < inc_val:
                inc_val = c
                incumbent = v.copy()
        except Exception:
            continue
    if incumbent is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        incumbent = z
        inc_val = _max_cos(z)
    best = incumbent.copy()
    best_val = float(inc_val)
    Ks = [3, 4, 5, 6, 8]
    trial = 0
    # precompute nothing heavy
    while time.time() < TRIAL_DEADLINE:
        trial += 1
        if time.time() > FINAL_END - 22.0:
            break
        k = Ks[int(rng.integers(len(Ks)))]
        mode = int(rng.integers(3))  # 0 random, 1 hubs, 2 cluster
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            gap = 0.004
            deg = (G0 > m0 - gap).sum(axis=1)
            if mode == 0:
                rem = rng.choice(n, size=k, replace=False)
            elif mode == 1:
                order = np.argsort(-deg, kind='stable')
                # take top-k with small random perturbation to vary
                jitter = rng.standard_normal(n) * 0.3
                score = deg.astype(float) + jitter
                rem = np.argsort(-score, kind='stable')[:k]
            else:
                seed = int(rng.integers(n))
                sims = G0[seed].copy()
                sims[seed] = -2.0
                nn = np.argsort(-sims, kind='stable')[:k-1]
                rem = np.concatenate([[seed], nn])
            rem = np.array(sorted(set(int(v) for v in rem)), dtype=int)
            # if dedup shrank, fill randomly
            while len(rem) < k:
                c = int(rng.integers(n))
                if c not in rem:
                    rem = np.append(rem, c)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[rem] = False
            Y = best[keep_mask].copy()  # (n-k, d)
            # sequential reinsertion
            for _r in range(k):
                if time.time() >= TRIAL_DEADLINE:
                    break
                n_cand = 8192
                C = rng.standard_normal((n_cand, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
                dots = C @ Y.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                xv = C[bi].copy()
                # local optimize inserted point with others fixed
                xv = _single_opt(xv, Y, steps=100, b0=250.0, b1=3000.0, lr0=0.06, lr1=0.008)
                Y = np.vstack([Y, xv[None, :]])
            if Y.shape[0] != n:
                continue
            Xtr = Y.copy()
            # short all-point soft-max quench
            Xtr = _quench(Xtr, steps=180, b0=150.0, b1=1800.0, lr0=0.025, lr1=0.004)
            # capped-pair LP polish, short budget
            remain = TRIAL_DEADLINE - time.time()
            if remain < 0.4:
                # skip LP, just evaluate
                v = _max_cos(Xtr)
                if v < best_val - 1e-12:
                    best = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
                    best_val = float(v)
                continue
            lb = min(1.2, max(0.4, remain * 0.25))
            Xtr = _lp_capped(Xtr, budget=lb, deadline=TRIAL_DEADLINE, max_pairs=420, max_iters=8)
            v = _max_cos(Xtr)
            if v < best_val - 5e-13:
                best = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
                best_val = float(v)
        except Exception:
            continue
    # reserve final ~20s for LP polishing of best
    try:
        fb = max(0.5, FINAL_END - time.time())
        fb = min(fb, 22.0)
        if fb > 0.6:
            best = _lp_capped(best, budget=fb, deadline=FINAL_END, max_pairs=650, max_iters=30)
            bv = _max_cos(best)
            # keep fallback guarantee: if polish worsened, revert to incumbent
            if bv > best_val + 1e-12:
                best = incumbent.copy()
            else:
                best_val = float(bv)
                if best_val > inc_val - 1e-12:
                    # no strict improvement: return incumbent to guarantee fallback
                    # but return the slightly polished best if it is not worse
                    pass
    except Exception:
        pass
    # guaranteed fallback
    try:
        bv = _max_cos(best)
        if not np.isfinite(bv) or bv > inc_val:
            return incumbent.copy()
    except Exception:
        return incumbent.copy()
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
