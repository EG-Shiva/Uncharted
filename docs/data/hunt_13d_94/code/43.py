import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_refine(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - np.eye(n) * 2.0))
        # stable exp
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
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
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
    return x, best

def _build_cluster_greedy(x, seed, K):
    n = x.shape[0]
    G = x @ x.T
    inset = {int(seed)}
    while len(inset) < K:
        mem = list(inset)
        best_i = -1
        best_v = -1e9
        for i in range(n):
            if i in inset:
                continue
            v = float(G[i, mem].max())
            if v > best_v:
                best_v = v
                best_i = i
        inset.add(int(best_i))
    return sorted(inset)

def solve():
    T0 = time.time()
    BUDGET = 170.0
    HOP_END = T0 + 132.0
    LP_END = T0 + BUDGET - 2.0
    rng = np.random.default_rng(0)
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    glob = best.copy()
    gval = best_val
    cur = best.copy()
    cval = best_val

    it = 0
    no_improve = 0
    # temperature schedule for plateau walk
    while time.time() < HOP_END:
        it += 1
        remain = HOP_END - (time.time() - T0) - 0 if False else HOP_END - time.time()
        if remain < 2:
            break
        # choose move
        u = rng.random()
        # sigma cycle
        sigmas = [0.03, 0.06, 0.02, 0.10, 0.045, 0.15, 0.03, 0.08]
        sigma = sigmas[it % len(sigmas)] * rng.uniform(0.7, 1.3)
        X0 = cur.copy()
        if u < 0.55:
            # global tangent shake
            N = rng.standard_normal((n, d)) * sigma
            dot = (N * X0).sum(axis=1, keepdims=True)
            Nt = N - dot * X0
            X0 = X0 + Nt
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        elif u < 0.75:
            # worst-pair split + small global shake
            try:
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G.max(axis=1)))
                # find partner
                row = X0 @ X0[ij]
                row[ij] = -2.0
                jj = int(np.argmax(row))
                # push apart
                eps = rng.uniform(0.03, 0.08)
                xi = X0[ij].copy(); xj = X0[jj].copy()
                dij = float(xi @ xj)
                # tangent components
                pix = xj - dij * xi
                pjx = xi - dij * xj
                pix /= (np.linalg.norm(pix) + 1e-12)
                pjx /= (np.linalg.norm(pjx) + 1e-12)
                X0[ij] = xi - eps * pix
                X0[jj] = xj - eps * pjx
                X0[ij] /= np.linalg.norm(X0[ij])
                X0[jj] /= np.linalg.norm(X0[jj])
            except Exception:
                pass
            N = rng.standard_normal((n, d)) * (sigma * 0.5)
            dot = (N * X0).sum(axis=1, keepdims=True)
            X0 = X0 + (N - dot * X0)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # connected cluster uniform reinsertion (topology change)
            try:
                G = X0 @ X0.T
                np.fill_diagonal(G, -2.0)
                worst = G.max(axis=1)
                # random among top-12 worst for diversity
                order = np.argsort(-worst)
                cand = order[:12]
                seed = int(cand[rng.integers(0, len(cand))])
                K = int(rng.choice([8, 10, 12, 14, 16]))
                cl = _build_cluster_greedy(X0, seed, K)
                R = rng.standard_normal((K, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                X0[cl] = R
            except Exception:
                N = rng.standard_normal((n, d)) * 0.1
                dot = (N * X0).sum(axis=1, keepdims=True)
                X0 = X0 + (N - dot * X0)
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        # occasional restart from global best to avoid drift
        if no_improve > 25 and rng.random() < 0.5:
            # shake global best instead
            X0 = glob.copy()
            N = rng.standard_normal((n, d)) * 0.04
            dot = (N * X0).sum(axis=1, keepdims=True)
            X0 = X0 + (N - dot * X0)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)

        # adapt steps to remaining time
        tleft = HOP_END - time.time()
        if tleft < 10:
            steps1, steps2 = 300, 150
        elif tleft < 40:
            steps1, steps2 = 450, 200
        else:
            steps1, steps2 = 650, 300

        try:
            X1 = _adam_refine(X0, steps=steps1, k0=25.0, k1=320.0, lr0=0.025, lr1=0.0025)
        except Exception:
            no_improve += 1
            continue
        v1 = _max_cos(X1)
        # only sharp-polish promising/plateau configs
        if v1 < cval + 8e-4:
            try:
                X1 = _adam_refine(X1, steps=steps2, k0=120.0, k1=600.0, lr0=0.006, lr1=0.001)
            except Exception:
                pass
            v1 = _max_cos(X1)
        else:
            # discard far-worse without extra work
            pass

        # update global
        if v1 < gval - 1e-12:
            gval = v1
            glob = X1.copy()
            cur = X1.copy()
            cval = v1
            no_improve = 0
            continue

        # Metropolis on plateau: accept worse with temp
        # decay temp from 3e-4 to 2e-5
        frac_done = 1.0 - max(0.0, (HOP_END - time.time()) / max(1e-9, (HOP_END - T0)))
        Temp = 3e-4 * (1 - frac_done) + 2e-5 * frac_done
        if v1 <= cval:
            # accept equal/better walker move (diversify even if not global)
            cur = X1.copy()
            cval = v1
            no_improve = 0 if v1 < cval + 1e-12 else no_improve + 1
        else:
            gap = v1 - cval
            prob = float(np.exp(-gap / max(1e-12, Temp)))
            if rng.random() < prob:
                cur = X1.copy()
                cval = v1
                no_improve += 1
            else:
                no_improve += 1

    # final: start LP from global best
    try:
        glob, gval2 = _lp_polish_deadline(glob, LP_END)
        gval = min(gval, gval2)
    except Exception:
        pass
    glob /= np.linalg.norm(glob, axis=1, keepdims=True)
    return glob
