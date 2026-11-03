import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
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

def _build_cluster(x, seed, K):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    in_cl = set([int(seed)])
    while len(in_cl) < K:
        best_i = -1
        best_v = -1e9
        # score = max dot to cluster
        for j in range(n):
            if j in in_cl:
                continue
            mv = -1e9
            for i in in_cl:
                v = G[j, i]
                if v > mv:
                    mv = v
            if mv > best_v:
                best_v = mv
                best_i = j
        in_cl.add(int(best_i))
    return sorted(list(in_cl))

def _cavity_batch(Fixed, K, center, rng, B, steps, k0, k1, lr0, lr1):
    F, d = Fixed.shape
    Ft = Fixed.T
    sigmas = rng.uniform(0.30, 0.80, size=B)
    X = np.empty((B, K, d))
    for b in range(B):
        nz = rng.standard_normal((K, d)) * sigmas[b]
        init = center[None, :] + nz
        nr = np.linalg.norm(init, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X[b] = init / nr
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eyeK = np.eye(K)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        Df = np.matmul(X, Ft)
        Dv = np.matmul(X, np.transpose(X, (0, 2, 1)))
        Dv_off = Dv - eyeK[None, :, :] * 2.0
        mDf = Df.max(axis=(1, 2))
        mDv = Dv_off.max(axis=(1, 2))
        m = np.maximum(mDf, mDv)
        Wf = np.exp(kk * (Df - m[:, None, None]))
        Wv = np.exp(kk * (Dv_off - m[:, None, None]))
        Wv = Wv * (1.0 - eyeK[None, :, :])
        sumF = Wf.sum(axis=(1, 2))
        sumVall = Wv.sum(axis=(1, 2))
        S = sumF + 0.5 * sumVall
        S[S <= 0] = 1.0
        # avoid overflow: S finite
        Gf = np.matmul(Wf, Fixed)
        Gv = np.matmul(Wv, X)
        Grad = (Gf + Gv) / S[:, None, None]
        dot = (Grad * X).sum(axis=2, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / bc1
        vhat = v1 / bc2
        upd = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=2, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _global_adam(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = np.max(G - np.eye(n) * 2.0)
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = W.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
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
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if not np.all(np.isfinite(x)):
            break
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    END = T0 + BUDGET
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((94, 13))
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        tmp = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(tmp):
            xb = xr.copy()
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    n, d = xb.shape

    # seed selection: high contact degree
    G = xb @ xb.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    deg_tight = np.sum(G > m0 - 1e-9, axis=1)
    deg_near = np.sum(G > m0 - 5e-7, axis=1)
    # secondary: sum of top-4 dots
    Gs = np.sort(G, axis=1)[:, -4:].sum(axis=1)
    score = deg_tight * 1000.0 + deg_near * 10.0 + (Gs - Gs.min()) / max(1e-12, (Gs.max() - Gs.min() + 1e-12))
    order = np.argsort(-score, kind="stable")
    seeds = []
    for idx in order:
        idx = int(idx)
        if len(seeds) >= 4:
            break
        # enforce separation: dot < 0.97 to accepted seeds
        ok = True
        for s in seeds:
            if float(xb[idx] @ xb[s]) > 0.97:
                ok = False
                break
        if ok:
            seeds.append(idx)
    if len(seeds) == 0:
        seeds = [int(order[0])]
    # cluster sizes per spec
    Ks = [12, 8, 16]

    cands = []  # (val, dist, full, seed, K)
    # cavity phase deadline
    CAV_END = T0 + 85.0
    for si, seed in enumerate(seeds):
        for K in Ks:
            if time.time() > CAV_END or time.time() > END - 70:
                break
            cl = _build_cluster(xb, seed, K)
            mask = np.ones(n, dtype=bool)
            mask[cl] = False
            Fixed = xb[mask]
            OldC = xb[cl]
            cen = OldC.mean(axis=0)
            nn = np.linalg.norm(cen)
            if nn < 1e-12:
                cen = xb[seed].copy()
            else:
                cen = cen / nn
            B = 32
            # fewer steps for larger K to save time
            steps = 550 if K <= 12 else 450
            try:
                Xb = _cavity_batch(Fixed, K, cen, rng, B, steps, 25.0, 380.0, 0.025, 0.002)
            except Exception:
                continue
            for b in range(B):
                Xnew = Xb[b]
                Full = np.empty_like(xb)
                Full[mask] = Fixed
                # place new points in cluster positions (order as cl)
                Full[np.array(cl)] = Xnew
                v = _max_cos(Full)
                # distance from original cluster via assignment
                try:
                    cost = -(Xnew @ OldC.T)
                    ri, ci = linear_sum_assignment(cost)
                    diff = Xnew[ri] - OldC[ci]
                    dist = float(np.linalg.norm(diff))
                except Exception:
                    dist = float(np.linalg.norm(Xnew - OldC))
                cands.append((v, dist, Full.copy(), seed, K))
            # early keep best
            # sort to allow quick check
    if len(cands) == 0:
        return xb

    cands.sort(key=lambda t: t[0])
    # update global best if cavity alone improved
    state_x = xb.copy()
    state_v = best_val
    if cands[0][0] < state_v - 1e-15:
        state_x = cands[0][2].copy()
        state_v = cands[0][0]

    # pick diverse best 5: rank by val, require distance
    pool = cands[:24]  # top by val
    # also ensure some far ones: sort pool by val, greedy diverse
    picked = []
    for v, dist, full, sd, K in pool:
        if len(picked) >= 5:
            break
        # diversity vs already picked (assignment on full? use cheap frobenius after ordering? use max-cos config distance approx)
        div_ok = True
        # require either good val or large dist from original: accept all in val order but enforce pairwise diversity
        for _, _, pf, _, _ in picked:
            # minimal alignment distance approximated by mean? use direct (labels correspond: fixed part same order, variable part order may differ but both occupy same indices cl; Hungarian already inside cavity but across different cl sets labels differ)
            # simple: Frobenius distance
            if float(np.linalg.norm(full - pf)) < 0.4:
                div_ok = False
                break
        # also prefer far from original: if dist < 0.3 and we already have near ones, skip unless very good
        if dist < 0.25 and len(picked) >= 2 and v > pool[0][0] + 2e-6:
            continue
        if div_ok:
            picked.append((v, dist, full, sd, K))
    if len(picked) == 0:
        picked = pool[:3]

    # unfreeze + refine best picked with global Adam
    REF_END = T0 + 125.0
    refined = []
    for (v, dist, full, sd, K) in picked:
        if time.time() > REF_END or time.time() > END - 45:
            break
        x = full.copy()
        try:
            x = _global_adam(x, steps=380, k0=30.0, k1=420.0, lr0=0.015, lr1=0.002)
        except Exception:
            pass
        nv = _max_cos(x)
        refined.append((nv, dist, x, sd, K))
        if nv < state_v - 1e-15 and np.all(np.isfinite(x)):
            state_x = x.copy()
            state_v = nv
    if len(refined) > 0:
        refined.sort(key=lambda t: t[0])
        # LP polish top 2
        for ii in range(min(2, len(refined))):
            if time.time() > END - 8:
                break
            nv, dist, x, sd, K = refined[ii]
            try:
                xp, vp = _lp_polish_deadline(x, deadline=min(END - 2.0, time.time() + 18.0))
                if vp < state_v and np.all(np.isfinite(xp)):
                    state_x = xp.copy()
                    state_v = vp
            except Exception:
                pass
    else:
        # polish state directly
        try:
            xp, vp = _lp_polish_deadline(state_x, deadline=END - 2.0)
            if vp < state_v:
                state_x, state_v = xp, vp
        except Exception:
            pass

    # final safety: never worse than input best.npy
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(state_x) > _max_cos(x0) + 1e-12:
            state_x = x0
    except Exception:
        pass
    state_x = state_x / np.linalg.norm(state_x, axis=1, keepdims=True)
    return state_x
