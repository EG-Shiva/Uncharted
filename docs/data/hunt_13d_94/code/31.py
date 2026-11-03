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

def _cavity_batch(Fixed, K, center, rng, B, steps, k0, k1, lr0, lr1):
    # Fixed: (F,d), center: (d,), returns X (B,K,d)
    F, d = Fixed.shape
    Ft = Fixed.T  # (d,F)
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
        # dots
        Df = np.matmul(X, Ft)  # (B,K,F)
        Dv = np.matmul(X, np.transpose(X, (0, 2, 1)))  # (B,K,K)
        Dv_off = Dv - eyeK[None, :, :] * 2.0
        mDf = Df.max(axis=(1, 2))
        mDv = Dv_off.max(axis=(1, 2))
        m = np.maximum(mDf, mDv)  # (B,)
        # weights
        # exp may underflow -> 0 fine
        Wf = np.exp(kk * (Df - m[:, None, None]))
        Wv = np.exp(kk * (Dv_off - m[:, None, None]))
        # zero diagonal (already ~ exp(-2k) tiny but force 0)
        Wv = Wv * (1.0 - eyeK[None, :, :])
        sumF = Wf.sum(axis=(1, 2))  # (B,)
        sumVall = Wv.sum(axis=(1, 2))  # sum over i!=j
        S = sumF + 0.5 * sumVall
        S[S <= 0] = 1.0
        Gf = np.matmul(Wf, Fixed)  # (B,K,d)
        Gv = np.matmul(Wv, X)  # (B,K,d)
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
        mval = np.max(G - np.eye(n) * 2.0)
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        # unordered denominator: sum upper = 0.5*sum all
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

def _build_cluster(x, seed, K):
    n = x.shape[0]
    in_cl = {int(seed)}
    G = x @ x.T
    while len(in_cl) < K:
        best_i, best_v = -1, -1e9
        members = list(in_cl)
        # for each outside point, score = max dot to cluster
        for i in range(n):
            if i in in_cl:
                continue
            v = G[i, members].max()
            if v > best_v:
                best_v = v
                best_i = i
        in_cl.add(int(best_i))
    return sorted(in_cl)

def solve():
    T0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(0)
    # load incumbent
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
    cur_best = best.copy()
    cur_val = best_val

    # seed ranking: worst max per point
    G0 = cur_best @ cur_best.T
    np.fill_diagonal(G0, -2.0)
    worst = G0.max(axis=1)
    seed_order = np.argsort(-worst)  # descending worst first

    # trial plan: (seed_idx, K)
    trials = []
    # high contact degree seeds, sizes 8,12,16
    cand_seeds = [int(seed_order[0]), int(seed_order[1]), int(seed_order[2]), int(seed_order[5])]
    for sd in cand_seeds[:2]:
        trials.append((sd, 12))
    trials.append((cand_seeds[2], 12))
    trials.append((cand_seeds[0], 16))
    trials.append((cand_seeds[1], 8))
    trials.append((cand_seeds[3], 12))
    trials.append((cand_seeds[0], 8))

    B = 32
    for ti, (seed, K) in enumerate(trials):
        if time.time() - T0 > 118.0:
            break
        # rebuild seed order from current best (topology may have changed)
        try:
            cl = _build_cluster(cur_best, seed % n, K)
        except Exception:
            continue
        cl = sorted(set(int(c) % n for c in cl))
        if len(cl) != K:
            continue
        mask = np.ones(n, dtype=bool)
        mask[cl] = False
        Fixed = cur_best[mask].copy()  # (F,d)
        C0 = cur_best[cl].copy()  # (K,d)
        ff = Fixed @ Fixed.T
        np.fill_diagonal(ff, -2.0)
        ff_max = float(ff.max())
        cav_center = C0.mean(axis=0)
        nc = np.linalg.norm(cav_center)
        if nc == 0 or not np.isfinite(nc):
            cav_center = cur_best[seed % n].copy()
        else:
            cav_center = cav_center / nc
        # steps adapt to remaining time and K
        remain = 118.0 - (time.time() - T0)
        if remain < 8:
            break
        steps = 1100
        if K >= 16:
            steps = 1300
        elif K <= 8:
            steps = 900
        # shrink steps if many trials left and little time
        if remain < 40:
            steps = min(steps, 700)
        try:
            Xb = _cavity_batch(Fixed, K, cav_center, rng, B=B, steps=steps,
                               k0=25.0, k1=350.0, lr0=0.03, lr1=0.003)
        except Exception:
            continue
        # rank completions by exact max and distance
        cross_max = np.empty(B)
        inner_max = np.empty(B)
        dists = np.empty(B)
        C0n = C0 / np.linalg.norm(C0, axis=1, keepdims=True)
        for b in range(B):
            X = Xb[b]
            df = X @ Fixed.T
            cross_max[b] = df.max()
            dv = X @ X.T
            np.fill_diagonal(dv, -2.0)
            inner_max[b] = dv.max()
            # matched distance via Hungarian (maximize dot)
            try:
                dots = C0n @ X.T  # (K,K)
                ri, ci = linear_sum_assignment(-dots)
                mdot = dots[ri, ci].mean()
                dists[b] = 1.0 - float(mdot)
            except Exception:
                dists[b] = 0.0
        glob = np.maximum(np.maximum(cross_max, inner_max), ff_max)
        order = np.argsort(glob, kind="stable")
        # take top 8 by max, then prefer distant for diversity
        top = order[:8]
        # sort top by distance descending, keep best-max among close ties
        # choose 3-5: best overall + farthest + second farthest
        sel = []
        sel.append(int(top[0]) if int(order[0]) == int(top[np.argsort(glob[top])[0]]) else int(order[0]))
        # actually ensure best is included
        if int(order[0]) not in sel:
            sel = [int(order[0])] + sel
        # add farthest from top
        far_order = top[np.argsort(-dists[top], kind="stable")]
        for c in far_order:
            if int(c) not in sel:
                sel.append(int(c))
            if len(sel) >= 4:
                break
        # unfreeze + refine best 3-4
        for b in sel[:4]:
            if time.time() - T0 > 130.0:
                break
            full = np.empty_like(cur_best)
            full[mask] = Fixed
            full[~mask if False else np.array([False]*n)] = 0  # noop
            # reconstruct: place Xb at cl positions
            full = cur_best.copy()
            full[cl] = Xb[b]
            full /= np.linalg.norm(full, axis=1, keepdims=True)
            # short global annealed refine (unfrozen)
            try:
                ref = _global_adam(full, steps=450, k0=60.0, k1=280.0, lr0=0.008, lr1=0.0015)
            except Exception:
                continue
            v = _max_cos(ref)
            if v < cur_val - 1e-12:
                cur_val = v
                cur_best = ref.copy()
            elif v < cur_val + 2e-6:
                # still consider if topologically different and close: keep exploring from it briefly?
                # only update if strictly better with small tol to preserve diversity pool
                pass
        # also try one global shake-refine from current best to exploit new basin hint
        if time.time() - T0 > 130.0:
            break

    # final LP polish with remaining time
    deadline = T0 + BUDGET - 2.0
    # give at least 10s to LP if possible
    try:
        cur_best, cur_val2 = _lp_polish_deadline(cur_best, deadline)
        cur_val = min(cur_val, cur_val2)
    except Exception:
        pass
    cur_best /= np.linalg.norm(cur_best, axis=1, keepdims=True)
    return cur_best
