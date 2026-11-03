import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal_full(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum() + 1e-300
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        if np.count_nonzero(sel) > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)
            ).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _build_groups(d=8):
    groups = {}
    # cyc8
    perms = np.zeros((8, d), dtype=int)
    signs = np.ones((8, d))
    for j in range(8):
        for i in range(d):
            perms[j, i] = (i + j) % d
    groups['cyc8'] = (perms, signs)
    # dih16 : cyclic + reversed cyclic
    perms16 = np.zeros((16, d), dtype=int)
    signs16 = np.ones((16, d))
    for j in range(8):
        for i in range(d):
            perms16[j, i] = (i + j) % d
            perms16[8 + j, i] = (j - i) % d
    groups['dih16'] = (perms16, signs16)
    # sign16 : cyclic with one-coord flip for second half
    perms_s = np.zeros((16, d), dtype=int)
    signs_s = np.ones((16, d))
    for j in range(8):
        for i in range(d):
            perms_s[j, i] = (i + j) % d
            perms_s[8 + j, i] = (i + j) % d
    signs_s[8:, 0] = -1.0
    groups['sign16'] = (perms_s, signs_s)
    # perm24 : S4 on first 4 coords
    lst = list(itertools.permutations(range(4)))
    perms24 = np.zeros((len(lst), d), dtype=int)
    for j, p in enumerate(lst):
        perms24[j, 0:4] = list(p)
        perms24[j, 4:8] = [4, 5, 6, 7]
    signs24 = np.ones((len(lst), d))
    groups['perm24'] = (perms24, signs24)
    # had64 : 8 shifts x 8 hadamard sign rows
    H2 = np.array([[1, 1], [1, -1]], dtype=float)
    H4 = np.kron(H2, H2)
    H8 = np.kron(H4, H2)
    perms64 = np.zeros((64, d), dtype=int)
    signs64 = np.zeros((64, d))
    idx = 0
    for q in range(8):
        for pp in range(8):
            for i in range(d):
                perms64[idx, i] = (i + pp) % d
                signs64[idx, i] = H8[q, i]
            idx += 1
    groups['had64'] = (perms64, signs64)
    # convert to matrices M (m,d,d)
    mats = {}
    for k, (pm, sn) in groups.items():
        m = pm.shape[0]
        M = np.zeros((m, d, d))
        ar = np.arange(d)
        for j in range(m):
            M[j, ar, pm[j]] = sn[j]
        mats[k] = M
    return mats


def _seed_adam(Y0, M, stages):
    Y = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
    k, d = Y.shape
    m = M.shape[0]
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    t = 0
    for beta, steps, lr in stages:
        for s in range(steps):
            t += 1
            X3 = np.einsum('ap,jip->aji', Y, M)
            N = k * m
            X = X3.reshape(N, d)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            E = np.exp(beta * (G - mx))
            np.fill_diagonal(E, 0.0)
            S = E.sum() + 1e-300
            Gx = (E @ X) / S
            Gx3 = Gx.reshape(k, m, d)
            Gy = np.einsum('aji,jip->ap', Gx3, M)
            dot = (Gy * Y).sum(axis=1, keepdims=True)
            Gt = Gy - dot * Y
            m1 = 0.9 * m1 + 0.1 * Gt
            v1 = 0.999 * v1 + 0.001 * Gt * Gt
            mhat = m1 / (1.0 - 0.9 ** t)
            vhat = v1 / (1.0 - 0.999 ** t)
            Y = Y - lr * mhat / (np.sqrt(vhat) + 1e-8)
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _adjust_to_138(Xexp, rng, n_target=138):
    N = Xexp.shape[0]
    d = Xexp.shape[1]
    if N == n_target:
        return Xexp.copy()
    if N > n_target:
        cur = Xexp.copy()
        while cur.shape[0] > n_target:
            ncur = cur.shape[0]
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            flat = np.argmax(G)
            ii, jj = flat // ncur, flat % ncur
            # max to others excluding the pair
            row_i = G[ii].copy()
            row_j = G[jj].copy()
            row_i[jj] = -2.0
            row_j[ii] = -2.0
            mi = row_i.max()
            mj = row_j.max()
            victim = ii if mi > mj else jj
            cur = np.delete(cur, victim, axis=0)
        return cur
    else:
        cur = Xexp.copy()
        while cur.shape[0] < n_target:
            Ncand = 20000
            cand = rng.standard_normal(size=(Ncand, d))
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            smax = (cand @ cur.T).max(axis=1)
            bi = int(np.argmin(smax))
            v0 = cand[bi]
            # refine new point against existing set
            v = v0.copy()
            beta = 120.0
            lr = 0.08
            for tt in range(120):
                dots = cur @ v
                mv = dots.max()
                W = np.exp(beta * (dots - mv))
                S = W.sum() + 1e-300
                g = (W @ cur) / S
                gt = g - np.dot(g, v) * v
                st = lr / (1.0 + tt * 0.02)
                v = v - st * gt
                v = v / (np.linalg.norm(v) + 1e-30)
            cur = np.vstack([cur, v[None, :]])
        return cur


def solve():
    rng = np.random.default_rng(777)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    mats = _build_groups(d=d)
    configs = [
        ('sign16', 9),
        ('dih16', 9),
        ('perm24', 6),
        ('had64', 2),
        ('cyc8', 17),
        ('cyc8', 18),
        ('sign16', 8),
        ('perm24', 5),
    ]
    stages = [
        (30.0, 220, 0.03),
        (80.0, 220, 0.02),
        (150.0, 220, 0.015),
        (250.0, 220, 0.01),
        (400.0, 250, 0.008),
    ]
    for gname, kk in configs:
        if time.time() > deadline - 18.0:
            break
        try:
            M = mats[gname]
        except Exception:
            continue
        Y0 = rng.standard_normal(size=(kk, d))
        Y0 = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
        try:
            Yopt = _seed_adam(Y0, M, stages)
            X3 = np.einsum('ap,jip->aji', Yopt, M)
            Xexp = X3.reshape((-1, d))
            Xexp = Xexp / np.linalg.norm(Xexp, axis=1, keepdims=True)
            X0 = _adjust_to_138(Xexp, rng, n_target=n)
            X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            continue
        try:
            Xcur = _anneal_full(X0, steps=500, beta0=20.0, beta1=200.0, lr0=0.02, lr1=0.003)
            for amp in [0.01, 0.03]:
                if time.time() > deadline - 14.0:
                    break
                noise = rng.standard_normal(size=Xcur.shape)
                proj = (noise * Xcur).sum(axis=1, keepdims=True)
                noise = noise - proj * Xcur
                nn = np.linalg.norm(noise, axis=1, keepdims=True) + 1e-30
                noise = noise / nn
                Xcur = Xcur + amp * noise
                Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
                Xcur = _anneal_full(Xcur, steps=350, beta0=30.0, beta1=300.0, lr0=0.015, lr1=0.002)
            Xcur = _lp_polish(Xcur, min(deadline - 6.0, time.time() + 2.5))
            v = _max_cos(Xcur)
            if v < best_val - 1e-12:
                best_val = v
                best = Xcur.copy()
        except Exception:
            continue

    try:
        if time.time() < deadline - 0.5:
            best = _lp_polish(best, deadline - 0.3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
