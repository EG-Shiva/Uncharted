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

def _smooth(X, steps, T0, T1, lr0, lr1, mom=0.85):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 * (1.0 - f) + T1 * f
        lr = lr0 * (1.0 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights, max pair = 1
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        S = W @ X
        dot = np.sum(S * X, axis=1, keepdims=True)
        Gr = S - dot * X
        s = float(np.sum(W)) / n + 1e-12
        Gr = Gr / s
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X = _norm(X)
    return X

def _refine_hole_vec(z, Xk, steps=200, T=0.008, lr=0.06):
    for _ in range(steps):
        dots = Xk @ z
        m = float(np.max(dots))
        W = np.exp((dots - m) / T)
        W = W / (np.sum(W) + 1e-12)
        g = W @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            break
        z = z / nz
    return z

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = x @ x.T
        c_all = Gf[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

    it = 0
    # main hopping until ~132s, reserve ~40s for final squeeze
    while time.time() - t0 < 132:
        it += 1
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        mv = float(np.max(G))
        deg = np.sum(G >= (mv - 1e-7), axis=1)
        ratt = np.where(deg == 0)[0]
        Y = best.copy()
        do_hole = (len(ratt) > 0 and (it % 3 == 0)) or (it % 6 == 0)
        if do_hole:
            if len(ratt) > 0:
                tgt = ratt[int(rng.integers(len(ratt)))]
            else:
                tgt = int(np.argmin(deg))
            Xk = np.delete(best, tgt, axis=0)
            # sample hole
            Z = rng.standard_normal((12000, d))
            Z = _norm(Z)
            D = Z @ Xk.T
            worst = np.max(D, axis=1)
            zi = int(np.argmin(worst))
            z = Z[zi].copy()
            z = _refine_hole_vec(z, Xk, steps=180, T=0.008, lr=0.06)
            Y[tgt] = z
            Y = _norm(Y)
            Y = _smooth(Y, steps=650, T0=0.022, T1=0.0022, lr0=0.07, lr1=0.005, mom=0.85)
        else:
            w = (deg.astype(float) + 1.0) ** 1.5
            probs = w / (np.sum(w) + 1e-12)
            cyc = it % 4
            if cyc == 1:
                Y = Y + 0.010 * rng.standard_normal(Y.shape)
                k, sig = 2, 0.13
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += sig * rng.standard_normal((k, d))
            elif cyc == 2:
                Y = Y + 0.018 * rng.standard_normal(Y.shape)
                k, sig = 1, 0.28
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += sig * rng.standard_normal((k, d))
            elif cyc == 3:
                Y = Y + 0.007 * rng.standard_normal(Y.shape)
                k, sig = 3, 0.11
                idx = rng.choice(n, k, replace=False, p=probs)
                Y[idx] += sig * rng.standard_normal((k, d))
            else:
                Y = Y + 0.014 * rng.standard_normal(Y.shape)
            Y = _norm(Y)
            # broad (balance) then narrow schedule
            Y = _smooth(Y, steps=550, T0=0.025, T1=0.0025, lr0=0.06, lr1=0.004, mom=0.82)
        vf = _max_cos(Y)
        # only LP-polish promising candidates to save time
        if vf < best_val + 0.0004:
            dl = min(time.time() + 2.2, t0 + 134)
            if dl - time.time() < 0.4:
                break
            Y2 = _lp_polish_deadline(Y, dl)
            v2 = _max_cos(Y2)
            if v2 < best_val - 1e-12:
                best = Y2
                best_val = v2
        if time.time() - t0 >= 133:
            break

    # final squeeze with remaining budget (cap ~38s, hard stop 176s)
    final_dl = min(t0 + 176, time.time() + 38)
    best = _lp_polish_deadline(best, final_dl)
    best = _norm(best)
    return best
