import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _smooth_polish(X, steps, T0, T1, lr0, lr1, mom=0.82):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        # log interp for T and lr
        T = T0 * ((T1 / T0) ** f) if T0 > 0 and T1 > 0 else T0 * (1 - f) + T1 * f
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # exp weights
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        S = W @ X
        dot = np.sum(S * X, axis=1, keepdims=True)
        rs = np.sum(W, axis=1, keepdims=True)
        F = (S - dot * X) / np.maximum(1.0, rs)
        V = mom * V + (1.0 - mom) * F
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _largest_voids(X_keep, k, rng, n_probe=15000):
    d = X_keep.shape[1]
    P = rng.standard_normal((n_probe, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    b = (P @ X_keep.T).max(axis=1)
    order = np.argsort(b)
    chosen = []
    for idx in order:
        if len(chosen) >= k:
            break
        p = P[idx]
        ok = True
        for q in chosen:
            if float(p @ q) > 0.70:
                ok = False
                break
        if ok:
            chosen.append(p)
    if len(chosen) < k:
        for idx in order:
            p = P[idx]
            dup = False
            for q in chosen:
                if np.allclose(p, q):
                    dup = True
                    break
            if not dup:
                chosen.append(p)
                if len(chosen) >= k:
                    break
    return np.array(chosen)

def _refine_void(z, X_keep, steps=120, T=0.006, lr=0.03):
    for _ in range(steps):
        dots = X_keep @ z
        m = float(dots.max())
        W = np.exp((dots - m) / T)
        W = W / (W.sum() + 1e-12)
        g = W @ X_keep
        g = g - float(np.dot(g, z)) * z
        nrm = float(np.linalg.norm(g))
        if nrm < 1e-12:
            break
        z = z - lr * g / max(1.0, nrm * 2.0)
        z = z / np.linalg.norm(z)
    return z

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
        best = rng.standard_normal((n, d))
        best = _norm(best)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    BUDGET = 138.0

    def contact_deg(X, thr=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = G >= (m - thr)
        return C.sum(axis=1), m

    def load_deg(X, margin=0.005):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = G >= (m - margin)
        return C.sum(axis=1), m

    it = 0
    while time.time() - t0 < BUDGET:
        it += 1
        elapsed = time.time() - t0
        prog = elapsed / BUDGET
        # annealing temperature for sideways moves
        Tann = 0.0018 * (1 - prog) + 0.00015
        # choose base: mostly cur for wandering, sometimes best
        if rng.random() < 0.30:
            base = best
            base_val = best_val
        else:
            base = cur
            base_val = cur_val
        cdeg, _ = contact_deg(base, 1e-9)
        ldeg, _ = load_deg(base, 0.005)
        rattlers = np.where(cdeg == 0)[0]
        # hub order by load
        order_load = np.argsort(ldeg)
        Y = base.copy()
        r = rng.random()
        if r < 0.32:
            # coordinated hub excision -> voids
            if rng.random() < 0.5:
                k = 2
            elif rng.random() < 0.75:
                k = 3
            else:
                k = 4
            # pick k hubs among top 15 by load
            pool = order_load[-15:]
            pw = (ldeg[pool].astype(float) + 1.0)
            pw = pw / pw.sum()
            idx = rng.choice(pool, size=min(k, len(pool)), replace=False, p=pw)
            k = len(idx)
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            X_keep = base[mask]
            V = _largest_voids(X_keep, k, rng, n_probe=15000)
            for t_i, vi in enumerate(V):
                # refine each void against keep + other voids already placed
                if len(chosen_sofar := []) == 0:
                    pass
                # refine against keep
                vi = _refine_void(vi, X_keep, steps=100, T=0.006, lr=0.03)
                V[t_i] = vi
            # joint repulsion among new points: push apart if too close
            # simple: if any pair >0.5, jitter
            Y[idx] = V + 0.006 * rng.standard_normal((k, d))
            Y += 0.003 * rng.standard_normal(Y.shape)
        elif r < 0.45:
            # hub-rattler swap + jitter
            if len(rattlers) > 0:
                nh = int(rng.integers(1, 3))
                hubs = order_load[-10:]
                hi = rng.choice(hubs, size=min(nh, len(hubs)), replace=False)
                ri = rng.choice(rattlers, size=min(nh, len(rattlers)), replace=False)
                # swap with noise
                tmp_h = Y[hi].copy()
                tmp_r = Y[ri].copy()
                Y[hi] = tmp_r + 0.02 * rng.standard_normal((len(hi), d))
                # move rattlers to old hub spots with larger kick to escape jam
                Y[ri] = tmp_h + 0.05 * rng.standard_normal((len(ri), d))
                # extra jitter on neighbors of hubs
                Y += 0.003 * rng.standard_normal(Y.shape)
            else:
                sig = float(rng.uniform(0.006, 0.02))
                # degree-weighted jitter: hubs move more
                w = (ldeg.astype(float) + 1.0)
                w = w / w.mean()
                Y += (sig * w[:, None]) * rng.standard_normal(Y.shape)
        elif r < 0.65:
            # degree-weighted global shake
            sig = float(rng.uniform(0.008, 0.025))
            w = (ldeg.astype(float) + 1.0) ** 0.7
            w = w / w.mean()
            Y += (sig * w[:, None]) * rng.standard_normal(Y.shape)
        elif r < 0.82:
            # cap shake: clustered region
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = base @ u
            thr = float(rng.uniform(0.25, 0.75))
            idx = np.where(proj > thr)[0]
            if len(idx) < 3 or len(idx) > 35:
                k = int(rng.integers(6, 18))
                idx = rng.choice(n, size=k, replace=False)
            sig = float(rng.uniform(0.06, 0.18))
            Y[idx] += sig * rng.standard_normal((len(idx), d))
            Y += 0.004 * rng.standard_normal(Y.shape)
        else:
            # big kick on 1-2 load hubs
            k = int(rng.integers(1, 3))
            pool = order_load[-20:]
            pw = (ldeg[pool].astype(float) + 1.0)
            pw = pw / pw.sum()
            idx = rng.choice(pool, size=k, replace=False, p=pw)
            sig = float(rng.uniform(0.12, 0.38))
            Y[idx] += sig * rng.standard_normal((k, d))
            Y += 0.007 * rng.standard_normal(Y.shape)
        Y = _norm(Y)
        # screening smooth polish
        lr0 = float(rng.uniform(0.014, 0.030))
        lr1 = float(rng.uniform(0.0003, 0.0009))
        T0 = float(rng.uniform(0.008, 0.014))
        T1 = float(rng.uniform(0.0008, 0.002))
        Ys = _smooth_polish(Y, 450, T0, T1, lr0, lr1, mom=0.82)
        vs = _max_cos(Ys)
        # decide extension
        improved_best = vs < best_val - 1e-12
        near = vs < min(cur_val + 0.004, best_val + 0.004)
        if improved_best or near:
            extra = int(rng.integers(900, 1600))
            # cooler second stage
            Ys = _smooth_polish(Ys, extra, 0.005, 0.0007, lr0 * 0.5, 0.00025, mom=0.85)
            Ys = _smooth_polish(Ys, 300, 0.0012, 0.0005, 0.0007, 0.00015, mom=0.80)
            v = _max_cos(Ys)
        else:
            v = vs
        # update best
        if v < best_val - 1e-12:
            best = Ys.copy()
            best_val = v
            cur = Ys.copy()
            cur_val = v
        else:
            # Metropolis for cur
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Tann, 1e-12)):
                cur = Ys.copy()
                cur_val = v
        # occasional reset cur to best to avoid drift
        if it % 25 == 0 and cur_val > best_val + 0.003:
            cur = best.copy()
            cur_val = best_val
        if time.time() - t0 > BUDGET:
            break

    # final intensification on best
    try:
        if time.time() - t0 < 172:
            best = _smooth_polish(best, 800, 0.004, 0.0006, 0.008, 0.0002, mom=0.85)
    except Exception:
        pass
    remain = 178 - (time.time() - t0)
    if remain > 6:
        try:
            best = _lp_polish(best, min(remain - 2, 28))
        except Exception:
            pass
    best = _norm(best)
    return best
