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

def adam_continuation(X, betas, steps_list, lr0, lr1, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    total = int(np.sum(steps_list))
    if total <= 0:
        return X
    gstep = 0
    t = 0
    b1p = 1.0
    b2p = 1.0
    for beta, steps in zip(betas, steps_list):
        beta = float(beta)
        for _ in range(int(steps)):
            frac = gstep / total
            lr = lr1 + 0.5 * (lr0 - lr1) * (1.0 + np.cos(np.pi * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mval = float(np.max(G))
            # exp(beta*(G-mval)) in place
            np.subtract(G, mval, out=G)
            np.multiply(G, beta, out=G)
            # clip to avoid underflow warning (exp(-100)->0 fine)
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G) * 0.5)
            if S < 1e-300:
                S = 1e-300
            G /= S
            Gr = G @ X
            dot = np.sum(Gr * X, axis=1, keepdims=True)
            Gr -= dot * X
            t += 1
            b1p *= b1
            b2p *= b2
            # m = b1*m + (1-b1)*Gr ; v similarly
            m *= b1
            m += (1.0 - b1) * Gr
            v *= b2
            v += (1.0 - b2) * (Gr * Gr)
            mhat_d = 1.0 - b1p
            vhat_d = 1.0 - b2p
            # avoid div by zero
            step = (m / mhat_d) / (np.sqrt(v / max(vhat_d, 1e-300)) + eps)
            X -= lr * step
            nrm = np.linalg.norm(X, axis=1, keepdims=True)
            np.maximum(nrm, 1e-12, out=nrm)
            X /= nrm
            gstep += 1
    return X

def fast_tail(X, steps, lr0, lr1, m0, m1):
    n, d = X.shape
    for s in range(int(steps)):
        f = s / max(1, int(steps) - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        margin = m0 * (1 - f) + m1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
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
        X = _norm(X)
    return X

def signature_top(X, k=400):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    k = min(k, c.size)
    idx = np.argpartition(-c, k - 1)[:k]
    s = c[idx]
    s = np.sort(s)[::-1]
    return s

def lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        c_all = (x @ x.T)[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4500]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(20261102)
    rng = np.random.default_rng(20261102)
    n, d = 131, 9
    # fallback best
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
    fallback = best.copy()
    fallback_val = best_val

    # ---- Phase A: many independent short continuations ----
    NSEED = 28
    b0_choices = [12.0, 20.0, 30.0]
    short_results = []  # (val, X)
    for si in range(NSEED):
        if time.time() > DEADLINE - 60:
            break
        b0 = b0_choices[si % len(b0_choices)]
        # vary initial beta slightly deterministically
        b0 = b0 * float(rng.uniform(0.9, 1.1))
        betas = [b0, b0 * 2.0]
        steps_list = [550, 550]
        lr0 = float(0.015 + 0.010 * rng.random())  # 0.015-0.025
        lr1 = 0.005
        X0 = _norm(rng.standard_normal((n, d)))
        X = adam_continuation(X0, betas, steps_list, lr0, lr1)
        val = _max_cos(X)
        short_results.append((val, X))
    if len(short_results) == 0:
        return _norm(fallback)
    short_results.sort(key=lambda t: t[0])
    # keep top K survivors
    K = min(8, len(short_results))
    survivors = [short_results[i][1] for i in range(K)]
    surv_vals = [short_results[i][0] for i in range(K)]

    # ---- cluster survivors by sorted pair-cosine spectra ----
    sigs = [signature_top(X, k=400) for X in survivors]
    clusters = []  # list of list of indices
    thresh = 0.004
    for i in range(K):
        assigned = False
        for cl in clusters:
            rep = cl[0]
            dist = float(np.linalg.norm(sigs[i] - sigs[rep]))
            if dist < thresh:
                cl.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
    # best rep per cluster (survivors already sorted, first in cluster is best)
    reps = []
    for cl in clusters:
        # cl members refer to survivors order (sorted by val), so min index is best
        bi = min(cl)
        reps.append((surv_vals[bi], survivors[bi]))
    reps.sort(key=lambda t: t[0])
    # limit to at most 5 clusters to save time
    reps = reps[:5]

    # ---- Phase B: longer continuation for each distinct rep ----
    extended = []
    for ri, (rv, RX) in enumerate(reps):
        if time.time() > DEADLINE - 40:
            break
        X = RX.copy()
        # geometric continuation up to ~400
        betas2 = [70.0, 140.0, 280.0, 400.0]
        steps2 = [600, 600, 600, 700]
        # smaller lr for late enforcement
        X = adam_continuation(X, betas2, steps2, 0.008, 0.0008)
        # fast tail to enforce equioscillation
        X = fast_tail(X, 800, 0.004, 0.0004, 0.015, 0.002)
        val = _max_cos(X)
        extended.append((val, X))
    # also consider short survivors directly polished (in case Phase B diverged)
    candidates = extended + [(surv_vals[i], survivors[i]) for i in range(min(3, K))]
    candidates.sort(key=lambda t: t[0])
    # polish top 2 candidates more
    polished = []
    for ci in range(min(2, len(candidates))):
        if time.time() > DEADLINE - 25:
            break
        val, X = candidates[ci]
        X = X.copy()
        # extra high-beta Adam + fine tail
        remain = DEADLINE - time.time()
        if remain > 12:
            X = adam_continuation(X, [400.0, 500.0], [500, 500], 0.004, 0.0005)
            X = fast_tail(X, 1200, 0.002, 0.0002, 0.008, 0.001)
        val2 = _max_cos(X)
        polished.append((val2, X))
    all_cands = polished + extended + short_results[:3]
    all_cands.sort(key=lambda t: t[0])
    cur_best = None
    cur_val = 1e9
    if len(all_cands) > 0:
        cur_val, cur_best = all_cands[0][0], all_cands[0][1].copy()
    else:
        cur_best, cur_val = fallback.copy(), fallback_val

    # keep fallback if random search worse
    if cur_val > fallback_val:
        cur_best, cur_val = fallback.copy(), fallback_val
    else:
        # we beat fallback: keep it, else still keep fallback as safety? keep better
        pass

    # ---- final LP squeeze if time remains ----
    remain = DEADLINE - time.time()
    if remain > 6:
        try:
            lp_dead = time.time() + min(remain - 1.5, 12.0)
            cur_best = lp_polish(cur_best, lp_dead)
            cur_val = _max_cos(cur_best)
            if cur_val > min(fallback_val, all_cands[0][0]) + 1e-12:
                # revert if LP hurt (should not)
                if fallback_val <= cur_val:
                    cur_best = fallback.copy()
        except Exception:
            pass
        # final safety: choose best among cur and fallback
        try:
            fv = fallback_val
            cv = _max_cos(cur_best)
            if fv < cv:
                cur_best = fallback.copy()
        except Exception:
            pass

    cur_best = _norm(cur_best)
    return cur_best
