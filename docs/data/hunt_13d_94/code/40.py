import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 9e9
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            cmat = (x @ x.T)[iu]
        except Exception:
            break
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

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
        S[~np.isfinite(S)] = 1.0
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
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _build_cluster(x, seed, K):
    n = x.shape[0]
    G = x @ x.T
    in_cl = set([int(seed)])
    while len(in_cl) < K:
        best_i = -1
        best_v = -1e100
        members = list(in_cl)
        for i in range(n):
            if i in in_cl:
                continue
            v = float(np.max(G[i, members]))
            if v > best_v:
                best_v = v
                best_i = i
        in_cl.add(int(best_i))
    return np.array(sorted(list(in_cl)), dtype=int)

def solve():
    T0 = time.time()
    BUDGET = 168.0
    HOP_END = T0 + 118.0
    rng = np.random.default_rng(20261102)
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        x_best = np.load("record.npy").astype(float)
    try:
        x_rec = np.load("record.npy").astype(float)
        x_rec = x_rec / np.linalg.norm(x_rec, axis=1, keepdims=True)
        xb_n = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        if _max_cos(x_rec) < _max_cos(xb_n):
            x_best = x_rec.copy()
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    cur_x = x_best.copy()
    cur_val = best_val
    n, d = cur_x.shape
    Ks = [8, 12, 16]
    it = 0
    while time.time() < HOP_END and time.time() - T0 < BUDGET - 50:
        it += 1
        if time.time() - T0 > BUDGET - 55:
            break
        # stats on cur
        G = cur_x @ cur_x.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        worst = G.max(axis=1)
        deg = (G > mx - 5e-4).sum(axis=1)
        # seed: high contact degree
        order_hi = np.argsort(-deg)
        pool_hi = order_hi[:12]
        seed = int(pool_hi[rng.integers(len(pool_hi))])
        # occasionally use worst point as seed
        if rng.random() < 0.2:
            order_w = np.argsort(-worst)
            seed = int(order_w[rng.integers(0, 6)])
        K = Ks[it % 3]
        # vary K slightly sometimes
        if rng.random() < 0.25:
            K = int(rng.choice([8, 12, 16]))
        K = min(K, n - 2)
        idx = _build_cluster(cur_x, seed, K)
        Fixed = np.delete(cur_x, idx, axis=0)
        Orig = cur_x[idx].copy()
        # cavity centre: normalized mean of removed cluster
        cm = Orig.mean(axis=0)
        nn = np.linalg.norm(cm)
        if nn == 0 or not np.isfinite(nn):
            cm = cur_x[seed] / np.linalg.norm(cur_x[seed])
        else:
            cm = cm / nn
        # fixed-fixed max (constant)
        Gf0 = Fixed @ Fixed.T
        np.fill_diagonal(Gf0, -2.0)
        fixed_max = float(Gf0.max())
        B = 32
        # anneal params slightly varied per iteration
        k0 = 18.0 + 8.0 * rng.random()
        k1 = 300.0 + 100.0 * rng.random()
        lr0 = 0.025 + 0.01 * rng.random()
        lr1 = 0.004 + 0.003 * rng.random()
        steps_cav = 450
        # reduce steps if time short
        remain = HOP_END - time.time()
        if remain < 30:
            steps_cav = 300
            B = 32
        try:
            Xb = _cavity_batch(Fixed, K, cm, rng, B, steps_cav, k0, k1, lr0, lr1)
        except Exception:
            continue
        if not np.all(np.isfinite(Xb)):
            continue
        # rank by exact max cosine
        try:
            Ft = Fixed.T
            Df = np.matmul(Xb, Ft)
            Dv = np.matmul(Xb, np.transpose(Xb, (0, 2, 1)))
            eyeK = np.eye(K)
            Dv_off = Dv - eyeK[None, :, :] * 2.0
            cross_max = Df.max(axis=(1, 2))
            within_max = Dv_off.max(axis=(1, 2))
            over = np.maximum(cross_max, within_max)
            total = np.maximum(over, fixed_max)
        except Exception:
            continue
        idx_sorted = np.argsort(total)
        # distance from original cluster (assignment)
        pool_n = min(12, B)
        pool = idx_sorted[:pool_n]
        dists = np.zeros(len(pool))
        for pi, b in enumerate(pool):
            Cb = Xb[b]
            M = Cb @ Orig.T
            try:
                r, c = linear_sum_assignment(-M)
                md = float(M[r, c].mean())
            except Exception:
                md = float(M.max(axis=1).mean())
            # chordal distance: larger = farther
            dists[pi] = 1.0 - md
        # select best 3-5 to refine: 2 best by value + 2 most distant within tolerance
        best_over = float(total[idx_sorted[0]])
        tol = 0.006
        mask = total[pool] <= best_over + tol
        # always keep top2
        chosen = []
        chosen.append(int(pool[0]))
        if len(pool) > 1:
            chosen.append(int(pool[1]))
        # among remaining pool with mask, pick most distant
        order_dist = np.argsort(-dists)
        for oi in order_dist:
            b = int(pool[oi])
            if b in chosen:
                continue
            if total[b] <= best_over + tol or len(chosen) < 3:
                chosen.append(b)
            if len(chosen) >= 4:
                break
        # ensure 4
        for b in idx_sorted:
            if len(chosen) >= 4:
                break
            if int(b) not in chosen:
                chosen.append(int(b))
        chosen = chosen[:4]
        # unfreeze and refine each chosen
        proposals = []
        for b in chosen:
            y = cur_x.copy()
            y[idx] = Xb[b]
            # tiny jitter to break symmetry before global refine
            y = y + 0.0015 * rng.standard_normal(size=y.shape)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            try:
                y_ref = _global_adam(y, steps=350, k0=25.0, k1=380.0, lr0=0.018, lr1=0.004)
            except Exception:
                y_ref = y
            if not np.all(np.isfinite(y_ref)):
                continue
            v = _max_cos(y_ref)
            # distance proxy for ranking
            proposals.append((v, b, y_ref))
        if len(proposals) == 0:
            continue
        proposals.sort(key=lambda t: t[0])
        prop_val, _, prop_x = proposals[0]
        # short LP polish only if promising (near or better than cur)
        if prop_val < cur_val + 0.002 and time.time() < HOP_END - 4:
            try:
                lp_dead = min(time.time() + 1.2, HOP_END)
                ylp, vlp = _lp_polish_deadline(prop_x, lp_dead)
                if np.all(np.isfinite(ylp)) and vlp < prop_val:
                    prop_val = vlp
                    prop_x = ylp
            except Exception:
                pass
        # update best
        if prop_val < best_val - 1e-15:
            x_best = prop_x.copy()
            best_val = prop_val
            cur_x = prop_x.copy()
            cur_val = prop_val
        else:
            elapsed = time.time() - T0
            frac = elapsed / max(1e-9, (HOP_END - T0))
            Temp = 0.002 * max(0.08, 1.0 - frac)
            Temp = max(Temp, 0.00015)
            if prop_val < cur_val or rng.random() < np.exp((cur_val - prop_val) / Temp):
                cur_x = prop_x.copy()
                cur_val = prop_val
            if rng.random() < 0.03:
                cur_x = x_best.copy() + 0.02 * rng.standard_normal(size=x_best.shape)
                cur_x = cur_x / np.linalg.norm(cur_x, axis=1, keepdims=True)
                cur_val = _max_cos(cur_x)
    # final refine
    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 6:
        try:
            x_best = _global_adam(x_best, steps=300, k0=80.0, k1=650.0, lr0=0.008, lr1=0.002)
            nv = _max_cos(x_best)
            if nv < best_val and np.all(np.isfinite(x_best)):
                best_val = nv
        except Exception:
            pass
        try:
            x_pol, pv = _lp_polish_deadline(x_best, time.time() + max(2.0, remain - 1.0))
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = pv
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0):
            # keep ours only if better, else revert to stored best
            if _max_cos(x0) < best_val:
                x_best = x0
    except Exception:
        pass
    return x_best
