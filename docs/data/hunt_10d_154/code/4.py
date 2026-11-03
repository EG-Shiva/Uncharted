import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_maxs(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        # time guard for linprog overrun
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_opt(X, steps, alpha0, alpha1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mask = np.ones((n, n), dtype=float)
    np.fill_diagonal(mask, 0.0)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = alpha0 * (alpha1 / alpha0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        # exclude diag
        np.fill_diagonal(G, -10.0)
        cmax = G.max()
        E = np.exp(alpha * (G - cmax)) * mask
        S = E.sum()
        if S <= 0 or not np.isfinite(S):
            break
        W = E / S
        Grad = W @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = (step * X).sum(axis=1, keepdims=True)
        step_t = step - sdot * X
        # clip huge steps
        nm = np.linalg.norm(step_t, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / nm)
        step_t = step_t * clip
        X = X - step_t
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def hole_for_set(Xfixed, rng, n_try=6000, refine=250):
    # Xfixed: (m,d) normalized
    d = Xfixed.shape[1]
    Z = rng.normal(size=(n_try, d))
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30)
    dm = (Z @ Xfixed.T).max(axis=1)
    y = Z[int(np.argmin(dm))].copy()
    # refine: minimize softmax max dot
    b1, b2, eps = 0.9, 0.999, 1e-8
    mm = np.zeros(d); vvv = np.zeros(d)
    for t in range(1, refine + 1):
        frac = t / refine
        alpha = 30.0 * (500.0 / 30.0) ** frac
        lr = 0.01 * (0.002 / 0.01) ** frac
        s = Xfixed @ y  # (m,)
        smax = s.max()
        e = np.exp(alpha * (s - smax))
        w = e / (e.sum() + 1e-30)
        g = w @ Xfixed  # (d,)
        gt = g - (g @ y) * y
        mm = b1 * mm + (1 - b1) * gt
        vvv = b2 * vvv + (1 - b2) * (gt * gt)
        mh = mm / (1 - b1 ** t)
        vh = vvv / (1 - b2 ** t)
        st = lr * mh / (np.sqrt(vh) + eps)
        st = st - (st @ y) * y
        nn = np.linalg.norm(st) + 1e-12
        if nn > 0.05:
            st *= 0.05 / nn
        y = y - st
        y /= (np.linalg.norm(y) + 1e-30)
    return y

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 176.0
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None:
            try:
                z = np.asarray(z, dtype=float)
                if z.shape == (154, 10):
                    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-30)
                    if np.all(np.isfinite(z)):
                        cands.append(z)
            except Exception:
                pass
    if len(cands) == 0:
        rng0 = np.random.RandomState(0)
        z = rng0.normal(size=(154, 10))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [max_cos(z) for z in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    # quick evaluation, keep best
    rng = np.random.RandomState(12345)
    n, d = best.shape
    it = 0
    # exploration until reserve for final polish
    while time.time() - t0 < BUDGET - 28.0:
        it += 1
        elapsed = time.time() - t0
        remain = BUDGET - elapsed
        if remain < 10:
            break
        # choose strategy
        cand = best.copy()
        r = rng.rand()
        if r < 0.65:
            sigma_all = rng.uniform(0.008, 0.045)
            cand = cand + rng.normal(scale=sigma_all, size=cand.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            # find worst k in cand
            try:
                rm = row_maxs(cand)
            except Exception:
                rm = row_maxs(best)
            k = int(rng.randint(1, 4))
            worst = np.argsort(rm)[-k:]
            # sequential hole replacement
            for idx in worst:
                # fixed set without idx
                Xf = np.delete(cand, idx, axis=0)
                y = hole_for_set(Xf, rng, n_try=4000, refine=200)
                cand[idx] = y
        elif r < 0.85:
            # shake random subset strongly
            sz = int(rng.randint(8, 26))
            subset = rng.choice(n, size=sz, replace=False)
            sigma = rng.uniform(0.07, 0.22)
            cand[subset] += rng.normal(scale=sigma, size=(sz, d))
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
            # also reinsert single worst hole sometimes
            if rng.rand() < 0.5:
                rm = row_maxs(cand)
                idx = int(np.argmax(rm))
                Xf = np.delete(cand, idx, axis=0)
                cand[idx] = hole_for_set(Xf, rng, n_try=4000, refine=200)
        else:
            # full moderate shake
            sigma = rng.uniform(0.03, 0.09)
            cand = best + rng.normal(scale=sigma, size=best.shape)
            cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)

        # smooth anneal; adapt steps to remaining time
        # keep steps modest for throughput
        steps = 700
        if remain < 60:
            steps = 500
        a0 = 40.0
        a1 = rng.uniform(250.0, 500.0)
        lr0 = rng.uniform(0.006, 0.014)
        lr1 = 0.0015
        try:
            cand2 = smooth_opt(cand, steps=steps, alpha0=a0, alpha1=a1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        try:
            v2 = max_cos(cand2)
        except Exception:
            continue
        # decide to LP polish
        # LP only if promising to save time
        need_lp = (v2 < best_val + 0.0025)
        # occasionally polish even if slightly worse to escape? rarely
        if not need_lp and rng.rand() < 0.05:
            need_lp = (v2 < best_val + 0.006)
        if need_lp:
            # short LP budget
            sec = 2.0
            if v2 < best_val - 0.0002:
                sec = 3.0
            # guard remaining
            if remain - sec < 28.0:
                sec = max(0.7, remain - 28.5)
            try:
                cand3 = lp_polish(cand2, seconds=sec)
                v3 = max_cos(cand3)
            except Exception:
                cand3, v3 = cand2, v2
            if v3 < best_val - 1e-12:
                best, best_val = cand3, v3
                # print progress (allowed)
                # print(f"iter {it} new best {best_val:.10f}")
            elif v2 < best_val - 1e-12:
                best, best_val = cand2, v2
        else:
            if v2 < best_val - 1e-12:
                # still accept without LP (will be polished later)
                # quick LP? just accept smooth
                best, best_val = cand2, v2
        # time check hard
        if time.time() - t0 >= BUDGET - 28.0:
            break

    # final squeeze with remaining time
    remain = BUDGET - (time.time() - t0)
    # leave 2 sec margin
    final_sec = max(1.0, remain - 2.0)
    final_sec = min(final_sec, 40.0)
    try:
        best = lp_polish(best, seconds=final_sec)
        best_val = max_cos(best)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-30)
    # tiny shrink not needed; ensure finite
    best = np.where(np.isfinite(best), best, 0.0)
    # renormalize after nan guard
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-30
    best = best / nr
    return best
