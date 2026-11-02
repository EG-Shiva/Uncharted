import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, iters, T0, T1, lr0, lr1):
    X = X.copy()
    n, d = X.shape
    for it in range(iters):
        f = it / max(1, iters - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        # ignore diagonal for max
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        # soft-max weights
        # (G-m)/T <=0, safe
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S == 0:
            S = 1.0
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        cap = 0.5
        factor = np.minimum(1.0, cap / gn)
        Gt = Gt * factor
        X = X - lr * 2.0 * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_reinsert(X, idx, rng, n_samples=6000):
    n, d = X.shape
    Y = np.delete(X, idx, axis=0)
    Z = rng.randn(n_samples, d)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    dots = Z @ Y.T
    worst = dots.max(axis=1)
    b = int(worst.argmin())
    Xn = X.copy()
    Xn[idx] = Z[b]
    return Xn

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for very long highs call
        if time.time() - start > seconds - 0.2:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
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
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    np.random.seed(0)
    rng = np.random.RandomState(0)
    t0 = time.time()
    HARD_LIMIT = 172.0
    # load start
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.randn(78, 15)
    if Xb.shape != (78, 15):
        # try to reshape / truncate
        tmp = rng.randn(78, 15)
        m = min(Xb.shape[0], 78)
        tmp[:m, :min(Xb.shape[1], 15)] = Xb[:m, :min(Xb.shape[1], 15)]
        Xb = tmp
    Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)
    try:
        Xr = np.load("record.npy").astype(float)
        Xr = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12)
        if _max_cos(Xr) < _max_cos(Xb):
            Xb = Xr
    except Exception:
        pass
    n, d = Xb.shape
    best_val = _max_cos(Xb)
    Xbest = Xb.copy()

    # quick contact info for bias (recomputed occasionally)
    trial = 0
    # exploration budget: leave ~38s for final polish
    explore_until = t0 + 128.0
    # if time already large, skip exploration
    while time.time() < explore_until and time.time() - t0 < HARD_LIMIT - 45:
        trial += 1
        u = rng.rand()
        Xt = Xbest.copy()
        if u < 0.40:
            sigma = 0.012 + 0.028 * rng.rand()  # 0.012-0.04
            Xt = Xt + rng.randn(n, d) * sigma
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
            T0 = 0.012 + 0.018 * rng.rand()
            T1 = 0.0004 + 0.0012 * rng.rand()
            lr0 = 0.03 + 0.02 * rng.rand()
            lr1 = 0.003 + 0.004 * rng.rand()
            iters = 650 + int(rng.randint(350))
            Xq = _quench(Xt, iters, T0, T1, lr0, lr1)
        elif u < 0.72:
            # single hole reinsertion, bias to random index
            # occasionally bias to worst / loosest
            if rng.rand() < 0.5:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                pw = G.max(axis=1)
                # pick among 12 worst or 12 best
                if rng.rand() < 0.5:
                    order = np.argsort(pw)[-12:]
                else:
                    order = np.argsort(pw)[:12]
                idx = int(order[rng.randint(len(order))])
            else:
                idx = int(rng.randint(n))
            Xt = _hole_reinsert(Xbest, idx, rng, n_samples=6000)
            # small shake of others
            Xt = Xt + rng.randn(n, d) * 0.006
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
            # re-enforce hole position a bit (don't shake it as much)
            T0 = 0.010 + 0.015 * rng.rand()
            T1 = 0.0003 + 0.001 * rng.rand()
            lr0 = 0.025 + 0.02 * rng.rand()
            lr1 = 0.003 + 0.003 * rng.rand()
            iters = 700 + int(rng.randint(400))
            Xq = _quench(Xt, iters, T0, T1, lr0, lr1)
        else:
            # block shake / multi-hole
            k = 4 + int(rng.randint(9))  # 4-12
            idxs = rng.choice(n, size=k, replace=False)
            # 50% hole reinsert block, 50% gaussian block
            if rng.rand() < 0.5:
                for idx in idxs:
                    Xt = _hole_reinsert(Xt, int(idx), rng, n_samples=3000)
                Xt = Xt + rng.randn(n, d) * 0.008
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
            else:
                Xt[idxs] = Xt[idxs] + rng.randn(k, d) * 0.06
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
                Xt = Xt + rng.randn(n, d) * 0.005
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
            T0 = 0.015 + 0.020 * rng.rand()
            T1 = 0.0005 + 0.0015 * rng.rand()
            lr0 = 0.035 + 0.02 * rng.rand()
            lr1 = 0.004 + 0.004 * rng.rand()
            iters = 800 + int(rng.randint(400))
            Xq = _quench(Xt, iters, T0, T1, lr0, lr1)

        v = _max_cos(Xq)
        if v < best_val:
            # squeeze promising candidate with short LP
            remain = HARD_LIMIT - (time.time() - t0)
            # only if enough time
            if remain > 50:
                Xq2 = _lp_polish(Xq, seconds=3.0)
                v2 = _max_cos(Xq2)
                if v2 < v:
                    Xq, v = Xq2, v2
            if v < best_val:
                Xbest = Xq
                best_val = v
                # early exit if big win? continue to push further
        # occasional time check for loop
        if trial > 800:
            break

    # final polish of best with remaining time
    elapsed = time.time() - t0
    remain = HARD_LIMIT - elapsed - 3.0
    if remain > 5:
        final_sec = min(remain, 40.0)
        Xbest = _lp_polish(Xbest, seconds=final_sec)
        # one more short quench+polish cycle to escape tiny rattler slack
        # quick rattler hole attempt if time left
        elapsed2 = time.time() - t0
        if HARD_LIMIT - elapsed2 > 12:
            # try moving the current loosest point once more
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            idx = int(np.argmin(pw))
            Xt = _hole_reinsert(Xbest, idx, rng, n_samples=8000)
            Xq = _quench(Xt, 700, 0.012, 0.0005, 0.03, 0.004)
            if _max_cos(Xq) < _max_cos(Xbest):
                Xbest = _lp_polish(Xq, seconds=5.0)
            # final squeeze
            if time.time() - t0 < HARD_LIMIT - 4:
                Xbest = _lp_polish(Xbest, seconds=max(2.0, HARD_LIMIT - (time.time()-t0) - 2.0))

    Xbest = Xbest / (np.linalg.norm(Xbest, axis=1, keepdims=True) + 1e-12)
    # safety: replace NaN
    Xbest[~np.isfinite(Xbest)] = 0.0
    # renormalize after safety
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
