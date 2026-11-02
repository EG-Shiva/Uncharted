import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, a0, a1, lr0, lr1):
    n, d = X.shape
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1).mean() + 1e-12
        X = X - (lr / gn) * Gt
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X

def _hole_replace(X, idxs, rng, n_cand=3000):
    n, d = X.shape
    if np.isscalar(idxs):
        idxs = [int(idxs)]
    else:
        idxs = [int(i) for i in idxs]
    X2 = X.copy()
    for idx in idxs:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        Y = X2[mask]
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        dots = C @ Y.T
        worst = dots.max(axis=1)
        best_c = C[np.argmin(worst)]
        X2[idx] = best_c
    return X2

def _greedy_push(X, steps, eps0, eps1):
    n = X.shape[0]
    for s in range(steps):
        f = s / max(1, steps - 1)
        eps = eps0 * (eps1 / eps0) ** f if eps0 != eps1 else eps0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        g = float(G[i, j])
        xi = X[i].copy()
        xj = X[j].copy()
        ti = xj - g * xi
        tj = xi - g * xj
        ni = float(np.linalg.norm(ti)) + 1e-12
        nj = float(np.linalg.norm(tj)) + 1e-12
        X[i] = xi - eps * ti / ni
        X[j] = xj - eps * tj / nj
        X[i] /= np.linalg.norm(X[i]) + 1e-16
        X[j] /= np.linalg.norm(X[j]) + 1e-16
    return X

def _lp_polish(x, seconds, t_deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        # cap size to keep HiGHS fast
        if np.count_nonzero(sel) > 900:
            # keep top 900
            idx = np.argpartition(-c, 900)[:900]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        if np.count_nonzero(sel) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(7)
    # load starts
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn).astype(float)
            if c.shape != (58, 22):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < best_val:
                best_val = v
                best = c.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((58, 22))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best = z
        best_val = _max_cos(best)

    n, d = 58, 22
    presets = [
        (80, 800, 0.020, 0.0020, 800),
        (150, 1200, 0.015, 0.0015, 900),
        (60, 600, 0.025, 0.0025, 700),
        (120, 1500, 0.012, 0.0012, 1000),
        (90, 900, 0.018, 0.0018, 800),
    ]
    trial = 0
    # precompute contact threshold helper
    while time.time() < HOP_END and time.time() < T_END - 30 and trial < 200:
        trial += 1
        a0, a1, lr0, lr1, steps = presets[trial % len(presets)]
        mode = trial % 5
        y = best.copy()
        if mode == 0:
            sig = 0.008 + 0.004 * ((trial // 5) % 3)
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            k = 5 + (trial % 4)  # 5..8
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += 0.05 * rng.standard_normal((k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 2:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 0.004).sum(axis=1)
            order = np.argsort(-cnt, kind="stable")
            k = 1 + (trial % 2)  # 1 or 2 worst hubs
            victims = order[:k]
            # randomize slightly among top 5
            if rng.random() < 0.5:
                victims = rng.choice(order[:6], size=k, replace=False)
            y = _hole_replace(best, victims, rng, n_cand=3000)
            y += 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 3:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // n; j = k % n
            victim = i if rng.random() < 0.5 else j
            y = _hole_replace(best, victim, rng, n_cand=4000)
            # shake partner a bit
            other = j if victim == i else i
            y[other] += 0.01 * rng.standard_normal(d)
            y += 0.002 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = rng.standard_normal((k, d))
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # quench, adapt steps to remaining time
        remain = min(HOP_END - time.time(), T_END - 30 - time.time())
        if remain < 1.0:
            break
        # shorten if very late
        if remain < 10 and steps > 600:
            steps = 600
        try:
            q = _quench(y, steps, a0, a1, lr0, lr1)
        except Exception:
            continue
        # greedy refinement (cheap)
        try:
            q = _greedy_push(q, 150, 0.003, 0.0003)
        except Exception:
            pass
        try:
            v = _max_cos(q)
        except Exception:
            continue
        if v < best_val:
            best = q
            best_val = v
            # intensify around new best: quick extra quench with sharp alpha
            if time.time() < HOP_END - 2:
                try:
                    q2 = _quench(best.copy(), 400, 200, 1500, 0.008, 0.001)
                    v2 = _max_cos(q2)
                    if v2 < best_val:
                        best = q2
                        best_val = v2
                except Exception:
                    pass

    # final LP polish with remaining budget (max 28s)
    try:
        budget = min(28.0, max(5.0, T_END - time.time() - 2.0))
        if budget > 2.0:
            best = _lp_polish(best, budget, T_END - 1.0)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best = np.where(np.isfinite(best), best, 0.0)
    # fix any degenerate row
    for i in range(best.shape[0]):
        nr = float(np.linalg.norm(best[i]))
        if not np.isfinite(nr) or nr < 1e-12:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            best[i] = v
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
