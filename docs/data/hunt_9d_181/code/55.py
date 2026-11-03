import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth(X, steps, k0, k1, lr0, lr1, mu=0.75):
    n, d = X.shape
    V = np.zeros_like(X)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.clip(k * (G - m), -30.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = float(np.linalg.norm(Gt, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-12:
            V *= 0.5
            continue
        V = mu * V + (lr / nrm) * Gt
        X = X - V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hole_scores(R, B):
    S = R @ B.T
    return S.max(axis=1)

def _refine_vec(v, B, steps=60, k=60, alpha=0.08):
    v = v / np.linalg.norm(v)
    for s in range(steps):
        dots = B @ v
        m = float(dots.max())
        w = np.exp(np.clip(k * (dots - m), -30, 0))
        w /= (w.sum() + 1e-300)
        g = w @ B
        gt = g - float(np.dot(g, v)) * v
        n = float(np.linalg.norm(gt))
        if not np.isfinite(n) or n < 1e-12:
            break
        a = alpha * (1.0 - 0.6 * s / max(1, steps - 1))
        v = v - (a / (n + 1e-12)) * gt * n
        # above is a*unit step; simplify to fixed tangential step
        # actually (a/(n))*gt*n = a*gt, so just:
        # v already updated; renormalize
        v /= (np.linalg.norm(v) + 1e-300)
    return v / np.linalg.norm(v)

def _best_hole(B, rng, n_try=12000, n_ref=3, kref=60):
    d = B.shape[1]
    R = rng.standard_normal((n_try, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    sc = _hole_scores(R, B)
    order = np.argpartition(sc, n_ref)[:n_ref]
    bv = None; bs = 1e9
    for oi in order:
        v = _refine_vec(R[oi], B, steps=50, k=kref)
        s = float((B @ v).max())
        if s < bs:
            bs = s; bv = v
    return bv

def _greedy_block(B, k, rng, pool=6000):
    d = B.shape[1]
    C = rng.standard_normal((pool, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    base = C @ B.T
    base_max = base.max(axis=1)
    picked = []
    cur_max = base_max.copy()
    P = []
    for _ in range(k):
        idx = int(np.argmin(cur_max))
        v = C[idx]
        # refine against B + already picked
        if len(P) == 0:
            BG = B
        else:
            BG = np.vstack([B] + P)
        v = _refine_vec(v, BG, steps=40, k=60)
        P.append(v.copy())
        # update cur_max with new point
        s = C @ v
        cur_max = np.maximum(cur_max, s)
        # forbid repicking same region: penalize near picks
        cur_max[idx] = 1e9
    return np.array(P)

def _lp_polish(X, budget, init_radius=1e-3, t_dead=None):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = c[sel]
        if cnt > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I = I[idx]; J = J[idx]; cs = cs[idx]
            cnt = 1500
        nv = n * d + 1
        cols_mat = np.empty((cnt, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(cnt), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((cnt, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(cnt, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_DEAD = t0 + 128.0
    rng = np.random.default_rng(12345)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        tmp = rng.standard_normal((181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [_max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy(); bb = float(scores[bi])
    xc = xb.copy(); bc = bb
    n, d = xb.shape
    T = 4e-05
    it = 0
    iu = np.triu_indices(n, 1)
    while time.time() < HOP_DEAD:
        rem = BUDGET - (time.time() - t0)
        if rem < 8.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 7
        if mode == 0:
            y = y + rng.standard_normal(y.shape) * 0.010
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=260, k0=30, k1=220, lr0=0.020, lr1=0.004)
        elif mode == 1:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            np.fill_diagonal(G, 1.0)
            cnt = np.sum(G > m - 1e-7, axis=1) - 1
            if rng.random() < 0.5:
                r = int(np.argmax(cnt))
            else:
                r = int(np.argmin(cnt))
            B = np.delete(y, r, axis=0)
            v = _best_hole(B, rng, n_try=12000, n_ref=3)
            y[r] = v
            y = y + rng.standard_normal(y.shape) * 0.004
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=300, k0=30, k1=260, lr0=0.020, lr1=0.003)
        elif mode == 2:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            np.fill_diagonal(G, 1.0)
            cnt = np.sum(G > m - 1e-7, axis=1) - 1
            order = np.argsort(-cnt)
            k = int(rng.integers(3, 6))
            victims = order[:k].copy()
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            B = y[keep_mask]
            P = _greedy_block(B, k, rng, pool=6000)
            y[victims] = P
            y = y + rng.standard_normal(y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=320, k0=25, k1=240, lr0=0.022, lr1=0.003)
        elif mode == 3:
            vdir = rng.standard_normal(d); vdir /= np.linalg.norm(vdir)
            dots = y @ vdir
            k = int(rng.integers(8, 14))
            victims = np.argsort(-dots)[:k]
            keep_mask = np.ones(n, dtype=bool); keep_mask[victims] = False
            B = y[keep_mask]
            P = _greedy_block(B, k, rng, pool=6000)
            # bias block toward cap is automatic via hole; just place
            y[victims] = P
            y = y + rng.standard_normal(y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=320, k0=25, k1=220, lr0=0.022, lr1=0.003)
        elif mode == 4:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            c = G[iu]
            o = np.argpartition(-c, 40)[:40]
            pick = o[int(rng.integers(0, 40))]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            diff = y[i0] - y[j0]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[i0] = y[i0] + diff * 0.07 + rng.standard_normal(d) * 0.02
            y[j0] = y[j0] - diff * 0.07 + rng.standard_normal(d) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=280, k0=35, k1=300, lr0=0.018, lr1=0.003)
        elif mode == 5:
            y = y + rng.standard_normal(y.shape) * 0.035
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=300, k0=25, k1=200, lr0=0.022, lr1=0.004)
        else:
            y = y + rng.standard_normal(y.shape) * 0.10
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=340, k0=20, k1=180, lr0=0.025, lr1=0.004)
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < bb - 1e-14:
            xb = y.copy(); bb = v
            xc = y.copy(); bc = v
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-9)):
                xc = y.copy(); bc = v
        T *= 0.9992
        if T < 6e-06:
            T = 6e-06
        # occasional restart of walker to best if stuck
        if it % 60 == 0 and bc > bb + 0.004:
            xc = xb.copy(); bc = bb
    # final elite polish (time-capped, small number of LP solves)
    rem = BUDGET - (time.time() - t0)
    if rem > 4.0:
        try:
            # polish current best walker too if distinct and promising
            cands = [(xb, bb)]
            if bc < bb + 0.002:
                cands.append((xc.copy(), bc))
            polished = []
            per = min(10.0, max(3.0, (rem - 1.5) / len(cands)))
            for X0, s0 in cands:
                if time.time() - t0 > BUDGET - 1.0:
                    break
                # short extra smoothing before LP
                X1 = _smooth(X0.copy(), steps=120, k0=60, k1=400, lr0=0.010, lr1=0.002, mu=0.6)
                v1 = _max_cos(X1)
                if v1 < s0:
                    X0, s0 = X1, v1
                Xp = _lp_polish(X0, budget=per, init_radius=0.012, t_dead=t0 + BUDGET - 0.6)
                vp = _max_cos(Xp)
                polished.append((Xp, vp))
            polished.sort(key=lambda t: t[1])
            if polished and polished[0][1] < bb:
                xb, bb = polished[0][0].copy(), float(polished[0][1])
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
