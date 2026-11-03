import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def adam_Y_stage(Y, alpha, lr, steps, t_end):
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        if t_end is not None and (s % 50 == 0) and time.time() > t_end:
            break
        G = Y @ Y.T
        # max over off-diag G and -G
        # compute max efficiently
        np.fill_diagonal(G, -10.0)
        mx1 = float(np.max(G))
        mx2 = float(np.max(-G))  # = -min, diag becomes 10 -> max 10? careful
        # Above diag fill corrupts -G max: diagonal of -G would be 10, must exclude
        # Recompute correctly: min off-diag
        # G diag is -10, so -G diag is 10 -> mx2=10 spurious. Fix:
        # use masked min/max
        # Let's recompute mx correctly
        # G with diag -10: mx1 ok. For mx2 we need -min_off = max(-G_off)
        # -G_off max = -(min_off). min_off = min of original G_off.
        # We have G with diag -10, min would be min_off unless min_off>-10 (always true since dots>-1). So get min:
        mn = float(np.min(G))  # this is min_off if min_off < -10? no, -10 is very low, so min is -10 spuriously
        # So masking approach failed. Do it properly with fill 0 and mask.
        # Fallback: compute full G2
        # To avoid confusion, recompute Gfull:
        Gf = Y @ Y.T
        # off diag max and min
        # use upper triangle
        # quick:
        mmax = -2.0
        mmin = 2.0
        # vectorized via masked array
        # create copy with diag handling
        # Instead use: max off-diag
        # set diag to -2 for max, +2 for min
        tmp_max = Gf.copy()
        np.fill_diagonal(tmp_max, -2.0)
        mx1 = float(tmp_max.max())
        tmp_min = Gf.copy()
        np.fill_diagonal(tmp_min, 2.0)
        mn_off = float(tmp_min.min())
        mx_neg = float(-mn_off)
        M = mx1 if mx1 > mx_neg else mx_neg
        # weights
        # E+ = exp(alpha*(Gf-M)) off-diag, E- = exp(alpha*(-Gf-M)) off-diag
        # subtract M for stability
        Eplus = np.exp(alpha * (Gf - M))
        Eminus = np.exp(alpha * (-Gf - M))
        np.fill_diagonal(Eplus, 0.0)
        np.fill_diagonal(Eminus, 0.0)
        S = float(Eplus.sum() + Eminus.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Ediff = (Eplus - Eminus) / S
        grad = Ediff @ Y
        # tangent projection
        nd = (grad * Y).sum(axis=1, keepdims=True)
        gt = grad - nd * Y
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * Y).sum(axis=1, keepdims=True)
        delta = delta - dn * Y
        # clip
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, 0.05 / rn)
        delta = delta * sc
        Y = Y - delta
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def adam_X_steps(X, alpha, lr, steps, t_end=None, clip=0.05):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        if t_end is not None and (s % 50 == 0) and time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ X) / S
        nd = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - nd * X
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * X).sum(axis=1, keepdims=True)
        delta = delta - dn * X
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        sc = np.minimum(1.0, clip / rn)
        delta = delta * sc
        X = X - delta
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def anneal_X_timed(X, seconds):
    start = time.time()
    end = start + seconds
    # split into 4 sub-stages
    sched = [(30.0, 0.005), (100.0, 0.003), (300.0, 0.0015), (800.0, 0.0007)]
    # allocate time proportionally: 20%,20%,25%,35%
    fracs = [0.20, 0.20, 0.25, 0.35]
    for (alpha, lr), fr in zip(sched, fracs):
        if time.time() >= end:
            break
        sub_end = start + seconds * sum(fracs[:fracs.index(fr) + 1]) if False else None
        # simpler: each stage gets fr*seconds from now proportionally remaining? use fixed chunk
        # compute stage end time
        # we track stage start
        st = time.time()
        stage_budget = fr * seconds
        # adjust for overtime
        remaining = end - st
        if remaining <= 0.05:
            break
        stage_budget = min(stage_budget, remaining)
        # if last stage, use all remaining
        if (alpha, lr) == sched[-1]:
            stage_budget = remaining
        stage_end = st + stage_budget
        # large step count, time-capped
        X = adam_X_steps(X, alpha, lr, 100000, t_end=stage_end)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = float(c_full.max())
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 6000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:6000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-8):
            radius *= 0.5
            continue
        y /= yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def solve():
    np.random.seed(12345)
    t0 = time.time()
    deadline = t0 + 176.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        r = np.random.RandomState(12345)
        best = r.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    fallback = best.copy()
    fallback_val = best_val

    # --- antipodal ansatz ---
    rng = np.random.RandomState(2026)
    Y = rng.randn(90, 9)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    # timed stages totaling at most 60s
    anti_start = time.time()
    anti_budget = 60.0
    # reserve time for branches: need at least 4*(2s) if near deadline; but try full
    # if little time left, shrink
    remain = deadline - time.time() - 5.0
    if remain < anti_budget + 20:
        anti_budget = max(5.0, min(anti_budget, remain - 20.0))
    anti_end = anti_start + anti_budget
    # per-stage budgets equal split, time-checked
    alphas = [30.0, 100.0, 300.0, 800.0]
    lrs = [0.005, 0.003, 0.0015, 0.0007]
    per_stage = anti_budget / 4.0
    for ai, lr in zip(alphas, lrs):
        if time.time() >= anti_end - 0.2:
            break
        stage_end = min(time.time() + per_stage, anti_end)
        # steps large, capped by time
        Y = adam_Y_stage(Y, ai, lr, 100000, t_end=stage_end)
        if time.time() >= deadline - 30:
            break
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    try:
        Xanti = np.vstack([Y, -Y])
        Xanti /= np.linalg.norm(Xanti, axis=1, keepdims=True)
        v_anti = max_cos(Xanti)
    except Exception:
        Xanti = None
        v_anti = 1e9

    # --- break symmetry branches ---
    sigmas = [0.0, 1e-4, 1e-3, 3e-3]
    # time left divided: each branch up to 15s anneal + 10s LP
    for bi, sigma in enumerate(sigmas):
        if Xanti is None:
            break
        # check global time: need at least a few seconds; keep 2s margin
        if time.time() >= deadline - 3.0:
            break
        branches_left = len(sigmas) - bi
        remaining = deadline - time.time() - 1.0
        # reserve equally: cap anneal and lp per branch
        # to fit, allow at most remaining/branches_left per branch
        per_branch = remaining / branches_left
        anneal_secs = min(15.0, max(1.0, per_branch - 2.0))
        # if per_branch small, reduce lp too
        lp_secs = min(10.0, max(0.8, per_branch - anneal_secs - 0.3))
        if per_branch < 2.0:
            anneal_secs = max(0.5, per_branch * 0.6)
            lp_secs = max(0.5, per_branch - anneal_secs - 0.2)
        rng_b = np.random.RandomState(5000 + bi)
        Xb = Xanti.copy()
        if sigma > 0:
            N = rng_b.randn(180, 9)
            proj = (N * Xb).sum(axis=1, keepdims=True)
            Nt = N - proj * Xb
            Xb = Xb + sigma * Nt
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
        # anneal
        try:
            # don't exceed deadline
            anneal_secs = min(anneal_secs, deadline - time.time() - lp_secs - 1.0)
            if anneal_secs > 0.3:
                Xb = anneal_X_timed(Xb, anneal_secs)
        except Exception:
            pass
        try:
            vb = max_cos(Xb)
        except Exception:
            continue
        # LP polish
        try:
            lp_allow = min(lp_secs, deadline - time.time() - 0.8)
            if lp_allow > 0.4:
                # smaller radius for tiny sigma branches, larger for big sigma
                rinit = 0.004 if sigma >= 1e-3 else 0.002
                Xb = lp_polish(Xb, lp_allow, radius_init=rinit)
                vb = max_cos(Xb)
        except Exception:
            pass
        try:
            if vb < best_val and np.all(np.isfinite(Xb)):
                best_val = vb
                best = Xb.copy()
        except Exception:
            pass
        if time.time() >= deadline - 2.0:
            break

    # timed fallback: if nothing beat fallback, return fallback (already best)
    # ensure best is valid
    try:
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        _ = max_cos(best)
    except Exception:
        best = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
    return best
