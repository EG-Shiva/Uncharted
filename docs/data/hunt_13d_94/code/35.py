import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_relax(X, steps, a0, a1, lr0, lr1, tabu_I=None, tabu_J=None, boost0=0.0, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    has_tabu = (tabu_I is not None) and (len(tabu_I) > 0) and (boost0 > 0)
    for s in range(steps):
        if deadline is not None and (s % 300 == 0) and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        # geometric schedules
        alpha = a0 * (a1 / a0) ** frac if a1 != a0 else a0
        lr = lr0 * (lr1 / lr0) ** frac if lr1 != lr0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mval = float(G.max())
        # avoid overflow: G-mval <= ~0.5
        W = np.exp(alpha * (G - mval))
        np.fill_diagonal(W, 0.0)
        if has_tabu:
            boost = boost0 * (1.0 - frac)
            if boost > 0.01:
                # symmetric boost
                W[tabu_I, tabu_J] *= (1.0 + boost)
                W[tabu_J, tabu_I] *= (1.0 + boost)
        S = float(W.sum())
        if S == 0.0 or not np.isfinite(S):
            break
        Grad = W @ X / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        # Adam
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mhat = m1 / bc1
        vhat = v1 / bc2
        upd = lr * mhat / (np.sqrt(vhat) + eps)
        X = X - upd
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
    return x

def solve():
    T0 = time.time()
    BUDGET = 168.0
    HOP_END = T0 + 128.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    # load best
    try:
        x_best = np.load("best.npy").astype(float)
    except Exception:
        try:
            x_best = np.load("record.npy").astype(float)
        except Exception:
            x_best = rng.standard_normal(size=(94, 13))
    x_best = x_best.astype(float)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    best_val = _max_cos(x_best)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best_val:
            x_best = xr.copy(); best_val = vr
    except Exception:
        pass
    n, d = x_best.shape
    cur_x = x_best.copy()
    cur_val = best_val

    def tabu_pairs_of(X, nkeep=600):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(n, 1)
        c = G[iu]
        # keep pairs within 5e-4 of max, up to nkeep tightest
        order = np.argsort(-c)
        take = min(nkeep, len(order))
        cand = order[:take]
        # filter to those reasonably close
        mask = c[cand] > m - 5e-4
        cand = cand[mask]
        if len(cand) == 0:
            cand = order[:50]
        return iu[0][cand].astype(np.int64), iu[1][cand].astype(np.int64)

    def cap_perturb(base):
        Y = base.copy()
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        dots = base @ u
        k = int(rng.integers(6, 17))
        idx_top = np.argsort(-dots)[:k]
        sigma_common = float(rng.uniform(0.08, 0.24))
        common = rng.standard_normal(d); common /= np.linalg.norm(common)
        common *= sigma_common
        # coordinated shear: top cap one way, bottom cap opposite (half the time)
        if rng.random() < 0.5:
            k2 = int(rng.integers(4, 12))
            idx_bot = np.argsort(dots)[:k2]
            Y[idx_top] += common
            Y[idx_bot] -= common * float(rng.uniform(0.4, 1.0))
            Y[idx_top] += rng.standard_normal((k, d)) * 0.015
            Y[idx_bot] += rng.standard_normal((k2, d)) * 0.015
        else:
            Y[idx_top] += common
            Y[idx_top] += rng.standard_normal((k, d)) * 0.015
        Y += rng.standard_normal((n, d)) * 0.003
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    trial = 0
    while time.time() < HOP_END and time.time() - T0 < BUDGET - 45:
        trial += 1
        r = rng.random()
        elapsed = time.time() - T0
        # pick base
        if rng.random() < 0.55:
            base = cur_x
        else:
            base = x_best
        if r < 0.60:
            # coordinated cap move + tabu repulsion (main topology changer)
            tI, tJ = tabu_pairs_of(base, nkeep=700)
            Y = cap_perturb(base)
            boost = float(rng.uniform(3.0, 9.0))
            # two-stage: tabu-forced then unbiased
            Y = _adam_relax(Y, steps=500, a0=25, a1=250, lr0=0.012, lr1=0.003,
                            tabu_I=tI, tabu_J=tJ, boost0=boost, deadline=HOP_END)
            Y = _adam_relax(Y, steps=900, a0=30, a1=450, lr0=0.008, lr1=0.0015,
                            tabu_I=None, tabu_J=None, boost0=0.0, deadline=HOP_END)
        elif r < 0.75:
            # strong global shake to new basin
            sigma = float(rng.choice([0.06, 0.10, 0.16]))
            Y = base + sigma * rng.standard_normal(size=base.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            tI, tJ = tabu_pairs_of(base, nkeep=400)
            boost = float(rng.uniform(1.0, 4.0))
            Y = _adam_relax(Y, steps=1300, a0=20, a1=350, lr0=0.012, lr1=0.002,
                            tabu_I=tI, tabu_J=tJ, boost0=boost, deadline=HOP_END)
        elif r < 0.88:
            # far random restart, thorough relax (diversity)
            Y = rng.standard_normal(size=base.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            # greedy pre-screen: quick low-sharpness relax then sharp
            Y = _adam_relax(Y, steps=900, a0=12, a1=200, lr0=0.015, lr1=0.003, deadline=HOP_END)
            Y = _adam_relax(Y, steps=900, a0=30, a1=500, lr0=0.008, lr1=0.0015, deadline=HOP_END)
        else:
            # medium subset shake + sharp polish (exploitation)
            kpick = int(rng.integers(6, 14))
            idx = rng.choice(n, size=kpick, replace=False)
            Y = base.copy()
            Y[idx] += 0.10 * rng.standard_normal(size=(kpick, d))
            Y += rng.standard_normal(size=Y.shape) * 0.002
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _adam_relax(Y, steps=1000, a0=35, a1=500, lr0=0.007, lr1=0.0012, deadline=HOP_END)

        if not np.all(np.isfinite(Y)):
            continue
        try:
            new_val = _max_cos(Y)
        except Exception:
            continue
        if not np.isfinite(new_val):
            continue
        if new_val < best_val - 1e-15:
            x_best = Y.copy(); best_val = new_val
            cur_x = Y.copy(); cur_val = new_val
        else:
            frac_e = (time.time() - T0) / max(1e-9, (HOP_END - T0))
            Temp = 0.0015 * max(0.08, 1.0 - frac_e)
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / Temp):
                cur_x = Y.copy(); cur_val = new_val
            # occasional restart cur from best if wandering too high
            if cur_val > best_val + 0.008 and rng.random() < 0.3:
                cur_x = x_best.copy(); cur_val = best_val

    # final tightening + LP polish with remaining time
    remain = BUDGET - (time.time() - T0) - 2.0
    if remain > 6:
        try:
            # short sharp Adam tighten
            x_best = _adam_relax(x_best, steps=400, a0=80, a1=800, lr0=0.004, lr1=0.001,
                                 deadline=T0 + BUDGET - 30)
            nv = _max_cos(x_best)
            if nv < best_val and np.all(np.isfinite(x_best)):
                best_val = nv
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=max(5.0, remain - 3.0), t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol; best_val = pv
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0) + 1e-12:
            # return better of the two to avoid regression
            if _max_cos(x0) < _max_cos(x_best):
                x_best = x0
    except Exception:
        pass
    return x_best
