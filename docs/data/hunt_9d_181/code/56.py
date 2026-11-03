import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 12:
            break
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            cs = c[sel]
            idx = np.argpartition(-cs, 1500)[:1500]
            I = I[idx]; J = J[idx]
            cs2 = cs[idx]
        else:
            cs2 = c[sel]
            p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs2, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(X, steps, k0=30.0, k1=220.0, lr0=0.02, lr1=0.002, t_dead=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    for s in range(steps):
        if t_dead is not None and s % 20 == 0 and time.time() >= t_dead:
            break
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # exp weights, diff<=0 so no overflow
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = float(np.linalg.norm(Gt, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _deepest_hole(K, rng, n_cand=20000):
    d = K.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ K.T
    mx = S.max(axis=1)
    # deepest = minimal max-dot; break ties randomly among top few by re-check
    i = int(np.argmin(mx))
    # small refinement: local repulsion push of winner (few projected steps)
    v = C[i].copy()
    # 30 steps of repulsion from K to centre in hole
    for _ in range(40):
        dots = K @ v
        # weights focused on nearest
        m = dots.max()
        w = np.exp(60.0 * (dots - m))
        g = w @ K
        g = g - np.dot(g, v) * v
        nrm = np.linalg.norm(g)
        if nrm < 1e-12:
            break
        v = v - (0.05 / (1.0 + nrm)) * g
        v /= np.linalg.norm(v)
    return v

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_dead = t0 + BUDGET
    # load incumbent
    cands = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9) and np.all(np.isfinite(a)):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        a = rng0.standard_normal((181, 9))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    scores = [_max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    x_best = cands[bi].copy()
    b_best = float(scores[bi])

    del_sizes = [18, 22, 15, 25, 20, 17, 24]
    n_restart = 0
    # quick initial LP to lock baseline if time allows (short)
    if t_dead - time.time() > 25:
        x_best = _lp_polish(x_best, budget=2.0, t_dead=t_dead - 20)
        b_best = _max_cos(x_best)

    while True:
        rem = t_dead - time.time()
        if rem < 14.0:
            break
        if n_restart > 12:
            break
        kdel = del_sizes[n_restart % len(del_sizes)]
        rng = np.random.default_rng(1000 + n_restart * 137)
        n_restart += 1
        # contact degrees with loose tol to find hubs
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            cnt = (G > m0 - 5e-4).sum(axis=1)
        except Exception:
            cnt = np.zeros(181)
        hub_pool = np.argsort(-cnt)[:40]
        n_hub = kdel // 2
        n_rnd = kdel - n_hub
        hub_choice = rng.choice(hub_pool, size=n_hub, replace=False)
        remaining = np.setdiff1d(np.arange(181), hub_choice)
        rnd_choice = rng.choice(remaining, size=n_rnd, replace=False)
        del_idx = np.concatenate([hub_choice, rnd_choice])
        keep_mask = np.ones(181, dtype=bool)
        keep_mask[del_idx] = False
        K = x_best[keep_mask].copy()
        # hard optimize smaller code
        # a touch of noise to leave basin
        K = K + rng.standard_normal(K.shape) * 0.002
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        inner_dead = t_dead - (kdel * 0.35 + 8.0)
        if inner_dead < time.time() + 1:
            inner_dead = time.time() + 1
        K = _smooth(K, steps=350, k0=30.0, k1=200.0, lr0=0.02, lr1=0.003, t_dead=inner_dead)
        if time.time() >= t_dead - 10:
            # still evaluate K growth? skip
            pass
        else:
            K = _lp_polish(K, budget=min(1.2, max(0.4, (t_dead - time.time()) * 0.08)), t_dead=t_dead - 8)
        # regrow one at a time
        cur = K
        ok = True
        for j in range(kdel):
            if time.time() >= t_dead - 6:
                ok = False
                break
            v = _deepest_hole(cur, rng, n_cand=20000)
            cur = np.vstack([cur, v[None, :]])
            # 150 global smoothing steps
            cur = _smooth(cur, steps=150, k0=40.0, k1=220.0, lr0=0.015, lr1=0.002, t_dead=t_dead - 2)
            if (j + 1) % 4 == 0:
                if time.time() >= t_dead - 6:
                    ok = False
                    break
                cur = _lp_polish(cur, budget=1.0, t_dead=t_dead - 4)
        if not ok:
            # incomplete growth: skip unless full size
            if cur.shape[0] != 181:
                continue
        if cur.shape[0] != 181:
            continue
        # final polish if time
        if t_dead - time.time() > 4:
            cur = _smooth(cur, steps=120, k0=60.0, k1=260.0, lr0=0.008, lr1=0.001, t_dead=t_dead - 2)
        if t_dead - time.time() > 3:
            cur = _lp_polish(cur, budget=min(2.0, t_dead - time.time() - 0.5), t_dead=t_dead - 0.3)
        try:
            v = _max_cos(cur)
        except Exception:
            continue
        if v < b_best:
            x_best = cur / np.linalg.norm(cur, axis=1, keepdims=True)
            b_best = float(v)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
