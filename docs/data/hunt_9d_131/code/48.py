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


def _adam_tau(X_init, tau, steps, lr0, lr1, T0, T1):
    X = X_init.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1 - f) + T1 * f
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        U = (G - tau) / T
        Sp = np.logaddexp(0, U)
        Uc = np.clip(U, -50, 50)
        Sigm = 1.0 / (1.0 + np.exp(-Uc))
        W = (2.0 * Sp * Sigm) / T
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step_t = step - sdot * X
        X = X - step_t
        X = _norm(X)
    return X


def _refine_hole(z, Xk, steps=250, T=0.008, lr=0.03):
    for _ in range(steps):
        dots = Xk @ z
        mv = float(dots.max())
        E = np.exp((dots - mv) / T)
        s = float(E.sum()) + 1e-300
        w = E / s
        g = w @ Xk
        g = g - float(np.dot(g, z)) * z
        z = z - lr * g
        nz = float(np.linalg.norm(z))
        if nz < 1e-12:
            break
        z = z / nz
    return z


def _find_best_hole(Xk, rng, n_samples=20000):
    d = Xk.shape[1]
    Z = rng.standard_normal((n_samples, d))
    Z = _norm(Z)
    D = Z @ Xk.T
    worst = D.max(axis=1)
    idx = int(np.argmin(worst))
    return Z[idx].copy(), float(worst[idx])


def _try_tau(curX, curVal, tau, rng, core_steps=2200, joint_steps=1200):
    n, d = curX.shape
    G = curX @ curX.T
    np.fill_diagonal(G, -2.0)
    deg = np.sum(G >= (curVal - 1e-7), axis=1)
    order = np.argsort(deg)  # lowest first
    # remove 9 lowest (rattlers + loosest)
    drop = order[:9]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[drop] = False
    core_idx = np.where(keep_mask)[0]
    Xc = curX[core_idx].copy()
    # tangential Adam on 122 core under tau penalty, T 3e-3 -> 2e-4
    Xc = _adam_tau(Xc, tau, core_steps, 0.01, 0.001, 3e-3, 2e-4)
    # reinsert 9 points one by one
    Xlist = [Xc]
    cur_full = Xc
    for _ in range(9):
        z, _ = _find_best_hole(cur_full, rng, n_samples=20000)
        z = _refine_hole(z, cur_full, steps=250, T=0.008, lr=0.03)
        cur_full = np.vstack([cur_full, z[None, :]])
    Xcand = cur_full  # 131 x 9
    # jointly re-polish all 131 under tau penalty
    Xcand = _adam_tau(Xcand, tau, joint_steps, 0.005, 0.0008, 2e-3, 2e-4)
    candVal = _max_cos(Xcand)
    return Xcand, candVal


def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(c_all.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:5000]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
    return x


def solve():
    t0 = time.time()
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 131, 9
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
    curX = best.copy()
    curVal = float(best_val)
    feas_end = t0 + 150.0
    final_end = t0 + 177.0
    # successive targets tau = 0.4220, 0.4215, ... adaptively
    tau = 0.4220
    # if curVal already below 0.4220, start 5e-4 below it
    if tau >= curVal:
        tau = curVal - 0.0004
    fails = 0
    it = 0
    while time.time() < feas_end:
        it += 1
        if tau < 0.415:
            break
        # time left check: need at least ~2s for a try
        if feas_end - time.time() < 2.0:
            break
        # shrink steps if little time left
        if feas_end - time.time() < 15:
            cs, js = 1200, 600
        else:
            cs, js = 2200, 1200
        try:
            candX, candVal = _try_tau(curX, curVal, tau, rng, core_steps=cs, joint_steps=js)
        except Exception:
            tau = (tau + curVal) / 2.0
            fails += 1
            if fails > 12:
                break
            continue
        if candVal < curVal:
            curX = candX
            curVal = float(candVal)
            fails = 0
            # next harder target: 5e-4 below new best (matches 0.4220,0.4215 grid)
            tau = curVal - 0.0005
            # also ensure we don't skip the fixed grid: go to min with grid
            # snap tau down to next 0.0005 grid point if close
            # e.g. grid ... 0.4220,0.4215: pick min
            # keep as is for pressure
        else:
            fails += 1
            mid = (tau + curVal) / 2.0
            # if step already tiny, stop pushing
            if curVal - mid < 1.5e-05:
                if fails > 4:
                    break
                # try a very small push
                tau = curVal - 2e-05
                # if that also fails repeatedly, break next time
            else:
                tau = mid
            if fails > 14:
                break
        # safety: tau must stay below curVal to exert pressure
        if tau >= curVal:
            tau = curVal - 2e-05
    # reserve final ~20s for LP polishing
    lp_deadline = min(final_end, time.time() + 20.0)
    # ensure at least a couple seconds of LP if time remains
    if time.time() < final_end - 1.0:
        try:
            curX = _lp_polish(curX, lp_deadline)
        except Exception:
            pass
    curX = _norm(curX)
    return curX
