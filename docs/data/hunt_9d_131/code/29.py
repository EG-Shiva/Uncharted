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

def _lse(X, T=0.01):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(np.max(c))
    s = float(np.sum(np.exp((c - m) / T)))
    return m + T * np.log(s + 1e-300), m

def fast_polish(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        th = m - margin
        Mask = G >= th
        W = np.zeros_like(G)
        if margin > 1e-12:
            W[Mask] = 0.2 + 0.8 * (G[Mask] - th) / margin
        else:
            W[Mask] = 1.0
        np.fill_diagonal(W, 0.0)
        S = W @ X
        deg = np.sum(W, axis=1, keepdims=True)
        dot = np.sum(S * X, axis=1, keepdims=True)
        F = S - dot * X
        F /= np.maximum(1.0, deg)
        X = X - lr * F
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def lp_polish_deadline(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:4000]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu[0][sel]; J = iu[1][sel]
        c = cfull[sel]
        p = len(I); nv = n * d + 1
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

def sph_kmeans(X, K, rng, restarts=6, iters=12):
    n, d = X.shape
    best_score = -1e100
    best_lab = None
    for _ in range(restarts):
        si = rng.choice(n, K, replace=False)
        C = X[si] + 0.01 * rng.standard_normal((K, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        lab = np.zeros(n, dtype=int)
        for _it in range(iters):
            S = X @ C.T
            lab = np.argmax(S, axis=1)
            Cn = np.zeros_like(C)
            for k in range(K):
                pts = X[lab == k]
                if len(pts) == 0:
                    Cn[k] = rng.standard_normal(d)
                else:
                    Cn[k] = pts.mean(axis=0)
            Cn /= (np.linalg.norm(Cn, axis=1, keepdims=True) + 1e-12)
            if np.allclose(Cn, C):
                C = Cn
                break
            C = Cn
        S = X @ C.T
        lab = np.argmax(S, axis=1)
        sc = float(np.sum(S[np.arange(n), lab]))
        if sc > best_score:
            best_score = sc
            best_lab = lab.copy()
    return best_lab

def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c; best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d)); best = _norm(best)
        best_val = _max_cos(best)
    best_lse, _ = _lse(best, 0.01)
    # pool of 5 elites: list of dicts sorted by hard max
    pool = [(best_val, best_lse, best.copy())]
    HOP_END = t0 + 132
    it = 0
    while time.time() < HOP_END:
        it += 1
        # pick base
        if len(pool) == 1 or rng.random() < 0.6:
            bi = 0
        else:
            bi = int(rng.integers(len(pool)))
        bval, blse, B = pool[bi]
        Y = None
        r = rng.random()
        if r < 0.68:
            K = int(rng.integers(8, 13))
            try:
                labels = sph_kmeans(B, K, rng, restarts=6, iters=12)
            except Exception:
                labels = None
            if labels is None:
                sz = int(rng.integers(5, 19))
                cl_idx = rng.choice(n, sz, replace=False)
            else:
                uniq, cnts = np.unique(labels, return_counts=True)
                elig = [u for u, c in zip(uniq, cnts) if 5 <= c <= 18]
                if len(elig) == 0:
                    sz = int(rng.integers(5, 19))
                    cl_idx = rng.choice(n, sz, replace=False)
                else:
                    ch = elig[int(rng.integers(len(elig)))]
                    cl_idx = np.where(labels == ch)[0]
            Y = B.copy()
            if rng.random() < 0.75:
                theta = float(rng.choice(np.array([0.15, 0.3, 0.6])))
                if rng.random() < 0.5:
                    theta = -theta
                cent = Y[cl_idx].mean(axis=0)
                nn = np.linalg.norm(cent)
                if nn < 1e-12:
                    cent = rng.standard_normal(d)
                    nn = np.linalg.norm(cent)
                u = cent / nn
                rv = rng.standard_normal(d)
                rv = rv - np.dot(rv, u) * u
                rn = np.linalg.norm(rv)
                if rn < 1e-12:
                    rv = rng.standard_normal(d)
                    rv = rv - np.dot(rv, u) * u
                    rn = np.linalg.norm(rv) + 1e-12
                v = rv / rn
                Xc = Y[cl_idx]
                a = Xc @ u; b = Xc @ v
                ct = np.cos(theta); st = np.sin(theta)
                da = a * (ct - 1.0) - b * st
                db = a * st + b * (ct - 1.0)
                Y[cl_idx] = Xc + da[:, None] * u + db[:, None] * v
                Y[cl_idx] /= (np.linalg.norm(Y[cl_idx], axis=1, keepdims=True) + 1e-12)
            else:
                nnv = rng.standard_normal(d)
                nnv /= (np.linalg.norm(nnv) + 1e-12)
                Xc = Y[cl_idx]
                Y[cl_idx] = Xc - 2.0 * np.outer(Xc @ nnv, nnv)
        else:
            Y = B.copy()
            if rng.random() < 0.5:
                Y = Y + 0.012 * rng.standard_normal(Y.shape)
            else:
                k = int(rng.integers(1, 4))
                idx = rng.choice(n, k, replace=False)
                Y[idx] = Y[idx] + 0.15 * rng.standard_normal((k, d))
            Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        steps = int(rng.integers(1200, 2001))
        lr0 = float(rng.uniform(0.012, 0.028))
        lr1 = float(rng.uniform(0.0002, 0.0008))
        m0 = float(rng.uniform(0.02, 0.05))
        m1 = float(rng.uniform(0.0015, 0.004))
        Y = fast_polish(Y, steps, lr0, lr1, m0, m1)
        Y = fast_polish(Y, 300, 0.0006, 0.00015, 0.002, 0.001)
        v = _max_cos(Y)
        lse_c, _ = _lse(Y, 0.01)
        improved_lse = lse_c < blse
        improved_hard = v < best_val - 1e-12
        # LP polish only if within 1e-4 of incumbent and not yet better by big margin
        if (v < best_val + 1e-4) and (v >= best_val - 1e-12) and (time.time() < HOP_END - 2):
            # short squeeze 1.2s
            Y2 = lp_polish_deadline(Y, min(time.time() + 1.2, HOP_END))
            v2 = _max_cos(Y2)
            lse2, _ = _lse(Y2, 0.01)
            if v2 < v:
                Y, v, lse_c = Y2, v2, lse2
                improved_hard = v < best_val - 1e-12
                improved_lse = lse_c < blse
        if improved_hard:
            best_val = v
            best = Y.copy()
            best_lse = lse_c
        if improved_lse or improved_hard:
            # insert into pool, keep 5 best by hard max, dedup
            pool.append((v, lse_c, Y.copy()))
            # dedup: sort by val
            pool = sorted(pool, key=lambda t: t[0])
            filt = []
            for t in pool:
                keep = True
                for t2 in filt:
                    if abs(t[0] - t2[0]) < 1e-11:
                        keep = False
                        break
                if keep:
                    filt.append(t)
            pool = filt[:5]
        if time.time() > HOP_END:
            break
    # final polish of incumbent
    remain = 178 - (time.time() - t0)
    if remain > 4:
        try:
            best = lp_polish_deadline(best, time.time() + min(remain - 1, 30))
        except Exception:
            pass
    best = _norm(best)
    return best
