import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thresh=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - thresh, axis=1)
    worst = g.max(axis=1)
    return cnt, worst, m

def _lp_polish(x, seconds, max_sol=40):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    nsol = 0
    while time.time() - start < seconds and radius > 1e-14 and nsol < max_sol:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        nsol += 1
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

def _adam_opt(X, T_init, steps, lr0, lr1, power=1):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    o1, o2 = 1.0 - b1, 1.0 - b2
    if steps > 1:
        decay = (lr1 / lr0) ** (1.0 / steps)
    else:
        decay = 1.0
    lr = lr0
    T = float(T_init)
    for t in range(1, steps + 1):
        G = X @ X.T
        W = G - T
        np.maximum(W, 0, out=W)
        np.fill_diagonal(W, 0.0)
        if power == 2:
            W *= W
        wmax = W.max()
        # adapt target to keep pressure focused
        if wmax == 0.0:
            T -= 0.0008
            G2 = X @ X.T
            W = G2 - T
            np.maximum(W, 0, out=W)
            np.fill_diagonal(W, 0.0)
            if power == 2:
                W *= W
            if W.max() == 0.0:
                lr *= decay; b1p *= b1; b2p *= b2
                continue
        elif (t % 400) == 0:
            # recenter T near current max to focus on worst pairs
            cur = float(G.max())
            # want T slightly below cur
            t_new = cur - 0.0025
            # move T partway
            if t_new > T:
                T = 0.5 * (T + t_new)
            else:
                T = t_new
        grad = W @ X
        s = np.sum(grad * X, axis=1, keepdims=True)
        g = grad - s * X
        m *= b1; m += o1 * g
        v *= b2; v += o2 * (g * g)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        upd = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * upd
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        lr *= decay; b1p *= b1; b2p *= b2
    return X

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        best = np.load("best.npy")
    except Exception:
        best = np.load("record.npy")
    best = np.asarray(best, dtype=np.float64)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    n, d = best.shape
    if best.shape != (180, 9):
        tmp = rng.standard_normal(size=(180, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        mm = min(n, 180)
        dd = min(d, 9)
        tmp[:mm, :dd] = best[:mm, :dd]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        best = tmp
        n, d = best.shape
    best_val = _max_cos(best)
    cnt, worst, _ = _contacts(best)
    it = 0
    HOP_END = t0 + 138.0
    while time.time() < HOP_END:
        it += 1
        if time.time() > t0 + 136.0:
            break
        # refresh contact info from current best every few iters
        if it % 5 == 1:
            cnt, worst, _ = _contacts(best)
            rattlers = np.where(cnt == 0)[0]
            order_worst = np.argsort(worst)  # ascending
            order_cnt = np.argsort(cnt)
        r = rng.random()
        if r < 0.45:
            # kick overloaded points (high contacts / high worst)
            k = int(rng.integers(4, 13))
            # mix of worst and high-count
            k1 = k // 2
            k2 = k - k1
            idx1 = np.argpartition(worst, -k1)[-k1:] if k1 > 0 else np.empty(0, dtype=int)
            idx2 = np.argpartition(cnt, -k2)[-k2:] if k2 > 0 else np.empty(0, dtype=int)
            idx = np.unique(np.concatenate([idx1, idx2]))
            k = len(idx)
            X0 = best.copy()
            X0 += (10.0 ** rng.uniform(-2.2, -1.4)) * rng.standard_normal(size=(n, d))
            X0[idx] = rng.standard_normal(size=(k, d))
            # move one rattler too if exists
            if len(rattlers) > 0 and (it % 2 == 0):
                rr = rng.choice(rattlers, size=min(1, len(rattlers)), replace=False)
                X0[rr] = rng.standard_normal(size=(len(rr), d))
        elif r < 0.7:
            # void re-insertion: remove few points, place in voids
            k = int(rng.integers(2, 6))
            idx = np.argpartition(worst, -k)[-k:]
            X0 = best.copy()
            X0 += (10.0 ** rng.uniform(-2.5, -1.8)) * rng.standard_normal(size=(n, d))
            # find voids: sample candidates, pick farthest from others
            for j in idx:
                cand = rng.standard_normal(size=(64, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                # max dot to all except j
                rest = np.delete(X0, j, axis=0)
                dots = cand @ rest.T
                md = dots.max(axis=1)
                X0[j] = cand[np.argmin(md)]
        elif r < 0.85:
            # coherent cluster shift of jammed neighbourhood
            k = int(rng.integers(8, 24))
            seed = int(np.argmax(worst)) if rng.random() < 0.7 else int(rng.integers(0, n))
            dots = best[seed] @ best.T
            idx = np.argsort(dots)[-k:]
            X0 = best.copy()
            shift = rng.standard_normal(size=(d,)) * float(rng.uniform(0.02, 0.12))
            X0[idx] += shift + 0.01 * rng.standard_normal(size=(k, d))
            X0 += 0.002 * rng.standard_normal(size=(n, d))
        else:
            # global small shake
            X0 = best + float(10.0 ** rng.uniform(-2.6, -1.9)) * rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        delta = float(rng.uniform(0.0012, 0.0045))
        T_init = best_val - delta
        steps = int(rng.integers(1800, 4200))
        lr0 = float(10.0 ** rng.uniform(-2.3, -1.8))
        lr1 = lr0 * float(10.0 ** rng.uniform(-1.5, -0.7))
        power = 2 if (it % 4 == 0) else 1
        Xopt = _adam_opt(X0, T_init, steps, lr0, lr1, power=power)
        val = _max_cos(Xopt)
        if val < best_val - 4e-9:
            if time.time() + 3.0 < t0 + 150.0:
                Xopt = _lp_polish(Xopt, 2.5, max_sol=30)
                val = _max_cos(Xopt)
            if val < best_val:
                best = Xopt / np.linalg.norm(Xopt, axis=1, keepdims=True)
                best_val = val
                cnt, worst, _ = _contacts(best)
                rattlers = np.where(cnt == 0)[0]
        if it % 22 == 0 and time.time() < HOP_END - 6:
            Xs = best + 0.004 * rng.standard_normal(size=(n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = _adam_opt(Xs, best_val - 0.001, 2000, 0.004, 0.0003, power=1)
            vs = _max_cos(Xs)
            if vs < best_val - 4e-9:
                if time.time() + 2.0 < t0 + 150.0:
                    Xs = _lp_polish(Xs, 1.5, max_sol=20)
                    vs = _max_cos(Xs)
                if vs < best_val:
                    best = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
                    best_val = vs
                    cnt, worst, _ = _contacts(best)
                    rattlers = np.where(cnt == 0)[0]
    now = time.time()
    remaining = 178.0 - (now - t0)
    if remaining > 3.0:
        remaining = min(remaining, 30.0)
        best = _lp_polish(best, remaining, max_sol=60)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
