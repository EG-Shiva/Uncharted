import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def softmax_adam(Y0, steps, T0, T1, lr, deadline):
    Y = Y0.copy()
    n, d = Y.shape
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_local = Y.copy()
    best_val = max_cos(Y)
    b1p, b2p = 1.0, 1.0
    for s in range(1, steps + 1):
        if time.time() > deadline:
            break
        frac = s / steps
        T = T0 * ((T1 / T0) ** frac)
        lr_s = lr / (1.0 + 1.0 * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        M = G.max()
        # scaled exp, stable (<=1)
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ Y) / S
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        upd = mhat / (np.sqrt(vhat) + eps)
        upd = upd - np.sum(upd * Y, axis=1, keepdims=True) * Y
        # clip per-point step
        un = np.linalg.norm(upd, axis=1, keepdims=True)
        cap = 1.0
        scale = np.minimum(1.0, cap / (un + 1e-12))
        upd = upd * scale
        Y = Y - lr_s * upd
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
        if (s % 50 == 0) or (s == steps):
            cur = max_cos(Y)
            if cur < best_val:
                best_val = cur
                best_local = Y.copy()
    return best_local, best_val


def greedy_relocate(Y, trials):
    n, d = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    cur_global = G.max()
    order = np.argsort(-rowmax)
    for idx in order[:12]:
        cur_row = rowmax[idx]
        C = np.random.randn(trials, d)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Y.T
        D[:, idx] = -2.0
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        if cm[k] < cur_row - 1e-12:
            Y2 = Y.copy()
            Y2[idx] = C[k]
            ng = max_cos(Y2)
            if ng < cur_global - 1e-12:
                return Y2, ng, True
    return Y, cur_global, False


def lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        # cap constraints for speed
        idx = np.where(sel)[0]
        if len(idx) > 3000:
            # keep largest
            order = np.argsort(-c_all[idx])[:3000]
            idx = idx[order]
            sel = np.zeros_like(sel)
            sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # track best even if x drifts
        if best < max_cos(best_x):
            best_x = x.copy()
    return best_x / np.linalg.norm(best_x, axis=1, keepdims=True)


def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    np.random.seed(0)

    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.randn(121, 8)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (121, 8):
        x0 = np.random.randn(121, 8)
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)

    best_x = x0.copy()
    best = max_cos(best_x)

    # 1) greedy relocations on current best (fast basin jumps)
    for _ in range(30):
        if time.time() > deadline - 120:
            break
        Y2, v2, ok = greedy_relocate(best_x, trials=4000)
        if ok and v2 < best:
            best_x, best = Y2, v2
        else:
            break

    # 2) annealed softmax runs
    # run A: exploit best basin
    configs = [
        (best_x, 1800, 0.02, 0.0025, 0.008),
        (best_x, 1200, 0.01, 0.0012, 0.006),
    ]
    # add perturbed copies
    rng_pert = [0.015, 0.03, 0.05]
    for sig in rng_pert:
        yp = best_x + sig * np.random.randn(*best_x.shape)
        yp /= np.linalg.norm(yp, axis=1, keepdims=True)
        configs.append((yp, 1200, 0.02, 0.002, 0.008))

    # add pure random starts (short)
    for _ in range(2):
        yr = np.random.randn(121, 8)
        yr /= np.linalg.norm(yr, axis=1, keepdims=True)
        configs.append((yr, 1000, 0.05, 0.004, 0.010))

    for (init, steps, T0, T1, lr) in configs:
        if time.time() > deadline - 60:
            break
        Yb, vb = softmax_adam(init, steps, T0, T1, lr, deadline - 45)
        # quick greedy clean of result
        for _ in range(5):
            Yb2, v2, ok = greedy_relocate(Yb, trials=2000)
            if ok and v2 < vb:
                Yb, vb = Yb2, v2
            else:
                break
        if vb < best:
            best, best_x = vb, Yb.copy()

    # 3) kick-and-refine loop
    for it in range(25):
        if time.time() > deadline - 45:
            break
        sig = 0.01 if (it % 2 == 0) else 0.025
        yp = best_x + sig * np.random.randn(*best_x.shape)
        yp /= np.linalg.norm(yp, axis=1, keepdims=True)
        Yb, vb = softmax_adam(yp, 600, 0.008, 0.001, 0.005, deadline - 35)
        if vb < best:
            best, best_x = vb, Yb.copy()

    # 4) final LP polish on best
    if time.time() < deadline - 5:
        best_x = lp_polish(best_x, deadline - 1.0)
        best = max_cos(best_x)

    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
