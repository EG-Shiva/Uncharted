import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth(X, steps, T0, T1, lr0, lr1, mom=0.82):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # (G-m)<=0 so exp stable
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W @ X
        dot = np.sum(S * X, axis=1, keepdims=True)
        Gr = S - dot * X
        s = float(np.sum(W)) / n + 1e-12
        Gr /= s
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _find_hole(Xk, rng, ns=12000):
    d = Xk.shape[1]
    Z = rng.standard_normal((ns, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    z = Z[idx].copy()
    # refine hole: minimise soft-max of dots
    T = 0.006
    lr = 0.06
    for _ in range(180):
        dots = Xk @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W /= (W.sum() + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z /= nz
    return z

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = x @ x.T
        call = Gf[iu0]
        m = float(call.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = call > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-call[idx])[:4500]
            msk = np.zeros_like(sel)
            msk[idx[order]] = True
            sel = msk
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = call[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    hop_deadline = t0 + 128.0
    final_deadline = t0 + 172.0
    it = 0
    while time.time() < hop_deadline:
        it += 1
        elapsed = time.time() - t0
        prog = elapsed / 128.0
        Tann = 0.003 * (1.0 - prog) + 0.00015
        mode = it % 9
        Y = cur.copy()
        if mode == 0:
            Y += 0.010 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 500, 0.015, 0.002, 0.025, 0.006)
        elif mode == 1:
            Y += 0.030 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 600, 0.020, 0.002, 0.030, 0.005)
        elif mode == 2:
            # kick worst + random hubs
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            k = 3
            idx = rng.choice(n, k, replace=False).tolist()
            if w not in idx:
                idx[0] = w
            Y[idx] += 0.22 * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 700, 0.020, 0.0015, 0.030, 0.005)
        elif mode == 3:
            # single worst-point hole reinsertion
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            mask = np.ones(n, dtype=bool)
            mask[w] = False
            z = _find_hole(Y[mask], rng, ns=10000)
            Y[w] = z
            Y = _smooth(Y, 700, 0.015, 0.0012, 0.025, 0.004)
        elif mode == 4:
            # 3 worst hole reinsertions
            G = Y @ Y.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            ws = np.argsort(-per)[:3]
            keep = np.ones(n, dtype=bool)
            keep[ws] = False
            Xk = Y[keep]
            for w in ws:
                z = _find_hole(Xk, rng, ns=8000)
                Y[w] = z
                Xk = np.vstack([Xk, z[None, :]])
            Y = _smooth(Y, 800, 0.018, 0.0012, 0.028, 0.004)
        elif mode == 5:
            # random subset replacement (big topology jump)
            k = 5
            idx = rng.choice(n, k, replace=False)
            R = rng.standard_normal((k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            Y[idx] = R
            Y = _smooth(Y, 800, 0.022, 0.0015, 0.032, 0.005)
        elif mode == 6:
            # global shake + one hub big kick
            Y += 0.012 * rng.standard_normal(Y.shape)
            idx = rng.choice(n, 1, replace=False)
            Y[idx] += 0.30 * rng.standard_normal((1, d))
            Y = _norm(Y)
            Y = _smooth(Y, 700, 0.020, 0.0015, 0.030, 0.005)
        elif mode == 7:
            # fresh random restart with long anneal (explore far basin)
            Y = _norm(rng.standard_normal((n, d)))
            Y = _smooth(Y, 1400, 0.025, 0.0012, 0.035, 0.004, mom=0.85)
        else:
            # perturb best directly (elitist kick)
            Y = best.copy()
            Y += 0.020 * rng.standard_normal(Y.shape)
            k = 2
            idx = rng.choice(n, k, replace=False)
            Y[idx] += 0.18 * rng.standard_normal((k, d))
            Y = _norm(Y)
            Y = _smooth(Y, 650, 0.018, 0.0015, 0.028, 0.005)
        v = _max_cos(Y)
        if v < cur_val:
            cur, cur_val = Y, v
            if v < best_val:
                best, best_val = Y.copy(), v
        else:
            # Metropolis move on cur to wander basins
            if rng.random() < np.exp((cur_val - v) / max(Tann, 1e-6)):
                cur, cur_val = Y, v
            # if cur drifted too far, reset to best
            if cur_val > best_val + 0.012:
                cur, cur_val = best.copy(), best_val
        # occasional reset of cur to best for intensification
        if it % 25 == 0 and time.time() < hop_deadline - 5:
            cur, cur_val = best.copy(), best_val
    # final squeeze
    try:
        best = _lp_polish(best, min(final_deadline, t0 + 172.0))
    except Exception:
        pass
    best = _norm(best)
    return best
