import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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
    in_cl = set([int(seed)])
    while len(in_cl) < K:
        best_i = -1
        best_v = -1e9
        mem = list(in_cl)
        for i in range(n):
            if i in in_cl:
                continue
            v = float(np.max(G[i, mem]))
            if v > best_v:
                best_v = v
                best_i = i
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
        mval = float(np.max(G - np.eye(n) * 2.0))
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
            xb = rng.standard_normal(size=(94, 13))
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xn):
            xb = xr.copy()
    except Exception:
        pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    state_x = xb.copy()
    state_v = _max_cos(state_x)
    init_v = state_v

    # seed ranking: contact degree + nearest sums
    G0 = state_x @ state_x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    contact_thr = m0 - 1e-9
    # if too few contacts (parent-like), use top-5 sums
    degs = (G0 > contact_thr).sum(axis=1)
    # top-5 neighbor sum for tie-break
    Gs = np.sort(G0, axis=1)[:, -5:]
    sums = Gs.sum(axis=1)
    order = np.lexsort((-sums, -degs))
    seeds = list(order[:4])

    Ks = [12, 16, 8]
    B = 32
    # iterate seed x K
    for seed in seeds:
        for K in Ks:
            if time.time() > END - 22.0:
                break
            # rebuild cluster from current best state
            try:
                cl = _build_cluster(state_x, int(seed), int(K))
            except Exception:
                continue
            mask = np.ones(n, dtype=bool)
            mask[np.array(cl, dtype=int)] = False
            Fixed = state_x[mask]
            Orig = state_x[np.array(cl, dtype=int)]
            ctr = Orig.mean(axis=0)
            nn = np.linalg.norm(ctr)
            if nn == 0 or not np.isfinite(nn):
                ctr = Orig[0].copy()
            else:
                ctr = ctr / nn
            # fixed-fixed max (constant)
            Gf = Fixed @ Fixed.T
            np.fill_diagonal(Gf, -2.0)
            ffmax = float(Gf.max())
            # batched cavity optimization
            steps_c = 900 if K <= 12 else 700
            try:
                Xb = _cavity_batch(Fixed, K, ctr, rng, B, steps_c, 25.0, 600.0, 0.03, 0.003)
            except Exception:
                continue
            # exact scoring
            vals = np.empty(B)
            dmax_f = np.empty(B)
            for b in range(B):
                Xc = Xb[b]
                Df = Xc @ Fixed.T
                Gv = Xc @ Xc.T
                np.fill_diagonal(Gv, -2.0)
                vals[b] = max(ffmax, float(Df.max()), float(Gv.max()))
            idx = np.argsort(vals)
            # diversity distance via assignment for top 10
            cand_pool = idx[:10]
            dists = np.zeros(len(cand_pool))
            for pi, b in enumerate(cand_pool):
                try:
                    C = -(Xb[b] @ Orig.T)
                    r, c = linear_sum_assignment(C)
                    s = float((-C[r, c]).sum())
                    s = max(-K, min(K, s))
                    dists[pi] = float(np.sqrt(max(0.0, 2 * K - 2 * s)))
                except Exception:
                    dists[pi] = 0.0
            # choose best 4: primary vals, prefer far if tie within 2e-4
            # sort pool by vals, then filter very-close duplicates
            pool_vals = vals[cand_pool]
            # keep those not essentially identical to original (dist>1e-3) unless all close
            keep = []
            for pi, b in enumerate(cand_pool):
                keep.append((pool_vals[pi], -dists[pi], int(b)))
            keep.sort()
            # take top 5, then rerank to favor distant among near-equal
            topBs = [k[2] for k in keep[:5]]
            # refine best 3-4 unfrozen
            nref = 4 if K <= 12 else 3
            topBs = topBs[:nref]
            for b in topBs:
                if time.time() > END - 18.0:
                    break
                full = np.empty_like(state_x)
                full[mask] = Fixed
                full[np.array(cl, dtype=int)] = Xb[b]
                full /= np.linalg.norm(full, axis=1, keepdims=True)
                # short global Adam from this completion
                try:
                    gsteps = 450
                    # adapt steps to remaining time
                    if END - time.time() < 25:
                        gsteps = 250
                    full2 = _global_adam(full.copy(), gsteps, 80.0, 800.0, 0.008, 0.001)
                except Exception:
                    full2 = full
                v2 = _max_cos(full2)
                if v2 < state_v - 1e-15 and np.all(np.isfinite(full2)):
                    state_x = full2.copy()
                    state_v = v2
                    # update fixed-fixed etc? keep going, also update G0 ordering? no
            # small LP polish on current state if improved notably? defer to end
        if time.time() > END - 22.0:
            break

    # final short LP polish on best
    remain = END - time.time() - 1.5
    if remain > 4:
        try:
            px, pv = _lp_polish_deadline(state_x.copy(), END - 1.0)
            if pv < state_v and np.all(np.isfinite(px)):
                state_x, state_v = px, pv
        except Exception:
            pass
    state_x /= np.linalg.norm(state_x, axis=1, keepdims=True)
    # safety never worse than input
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(state_x) > _max_cos(x0) + 1e-12:
            # return original if we failed (but keep new if tie-or-better)
            # check time: if we improved over init, keep; else revert
            if state_v > init_v + 1e-12:
                state_x = x0
    except Exception:
        pass
    return state_x
