import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, t_start=None):
    if t_start is None:
        t_start = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - t_start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            cnt = int(np.sum(sel))
        I = iu[0][sel]
        J = iu[1][sel]
        c_sel = c_all[sel]
        p = cnt
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.concatenate([cols_I, cols_J, np.full(p, n * d)])
        # interleave per row: need order I_d, J_d, t per row
        # Build correctly row-wise
        cols_mat = np.empty((p, 2 * d + 1), dtype=int)
        cols_mat[:, :d] = I[:, None] * d + np.arange(d)
        cols_mat[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols_mat[:, 2 * d] = n * d
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _active_gd(x, steps, lr, m0, m1, mu=0.85):
    v = np.zeros_like(x)
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        margin = m0 + (m1 - m0) * (s / max(steps - 1, 1))
        thr = m - margin
        Mask = G > thr
        W = Mask.astype(np.float64)
        deg = W.sum(axis=1, keepdims=True)
        deg = np.maximum(deg, 1.0)
        Rw = W @ x
        dsum = (W * G).sum(axis=1, keepdims=True)
        grad = (Rw - dsum * x) / deg
        v = mu * v - lr * grad
        x = x + v
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _find_hole(Y, n_samples=3000, refine_steps=150):
    n, d = Y.shape
    R = np.random.randn(n_samples, d)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ Y.T
    am = D.max(axis=1)
    idx = np.argpartition(am, 2)[:2]
    best_y = R[idx[0]] / np.linalg.norm(R[idx[0]])
    best_v = 1e9
    for k in idx:
        y = R[k].copy()
        y = y / np.linalg.norm(y)
        lr = 0.05
        for s in range(refine_steps):
            dots = Y @ y
            top = np.argpartition(dots, -3)[-3:]
            beta = 100.0
            mx = float(np.max(dots[top]))
            w = np.exp(beta * (dots[top] - mx))
            w = w / w.sum()
            c = (w[:, None] * Y[top]).sum(axis=0)
            dotc = float(y @ c)
            grad = c - dotc * y
            y = y - lr * grad
            nr = np.linalg.norm(y)
            if nr < 1e-12:
                y = R[k].copy()
            else:
                y = y / nr
            lr *= 0.995
        vv = float(np.max(Y @ y))
        if vv < best_v:
            best_v = vv
            best_y = y
    return best_y


def solve():
    np.random.seed(42)
    t0 = time.time()
    TIME_BUDGET = 175.0
    try:
        X0 = np.load("best.npy")
    except Exception:
        X0 = np.load("record.npy")
    X0 = np.asarray(X0, dtype=np.float64)
    # ensure shape (189,12)
    if X0.shape != (189, 12):
        # try transpose or truncate/pad deterministically
        if X0.size == 189 * 12:
            X0 = X0.reshape(189, 12)
        else:
            R = np.random.randn(189, 12)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            X0 = R
    X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    best_val = _max_cos(X0)
    bestX = X0.copy()
    curX = X0.copy()
    curVal = best_val

    it = 0
    # exploration phase until ~120s
    while time.time() - t0 < 120.0:
        it += 1
        if it % 10 == 0:
            base = bestX
        else:
            base = curX
        trial = base.copy()
        n, d = trial.shape
        move = it % 4
        if move == 0:
            sigma = 0.012
            trial = trial + sigma * np.random.randn(n, d)
            trial = trial / np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _active_gd(trial, steps=250, lr=0.015, m0=0.08, m1=0.003, mu=0.8)
        elif move == 1:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            degs = (G > m - 0.005).sum(axis=1)
            h = int(np.argmax(degs))
            neigh = np.where(G[h] > m - 0.03)[0]
            idx = np.unique(np.concatenate([np.array([h]), neigh]))[:30]
            sigma = 0.08
            trial[idx] = trial[idx] + sigma * np.random.randn(len(idx), d)
            trial[idx] = trial[idx] / np.linalg.norm(trial[idx], axis=1, keepdims=True)
            trial = _active_gd(trial, steps=250, lr=0.018, m0=0.06, m1=0.003, mu=0.85)
        elif move == 2:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # rattlers: no contact within 1e-7
            cont = (G > m - 1e-7).sum(axis=1)
            ratt = np.where(cont == 0)[0]
            if len(ratt) > 0:
                r = int(ratt[np.random.randint(len(ratt))])
                mask = np.ones(n, dtype=bool)
                mask[r] = False
                Y = trial[mask]
                hole = _find_hole(Y, n_samples=3000, refine_steps=120)
                trial[r] = hole
                trial = _active_gd(trial, steps=200, lr=0.012, m0=0.05, m1=0.002, mu=0.8)
            else:
                sigma = 0.05
                trial = trial + sigma * np.random.randn(n, d)
                trial = trial / np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _active_gd(trial, steps=250, lr=0.015, m0=0.08, m1=0.003, mu=0.8)
        else:
            G = trial @ trial.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            degs = (G > m - 0.005).sum(axis=1)
            h = int(np.argmax(degs))
            mask = np.ones(n, dtype=bool)
            mask[h] = False
            Y = trial[mask]
            hole = _find_hole(Y, n_samples=3000, refine_steps=120)
            # small random tilt to break symmetry
            hole = hole + 1e-3 * np.random.randn(d)
            hole = hole / np.linalg.norm(hole)
            trial[h] = hole
            trial = _active_gd(trial, steps=250, lr=0.015, m0=0.06, m1=0.003, mu=0.85)

        v = _max_cos(trial)
        if v < best_val - 1e-12:
            # squeeze with short LP
            t_rem = TIME_BUDGET - (time.time() - t0)
            sec = 2.0 if t_rem > 30 else 1.0
            trial2 = _lp_polish(trial, seconds=sec)
            v2 = _max_cos(trial2)
            if v2 < best_val:
                best_val = v2
                bestX = trial2.copy()
                curX = trial2.copy()
                curVal = v2
            else:
                if v < curVal:
                    curX = trial.copy()
                    curVal = v
        elif v < curVal + 0.005:
            curX = trial.copy()
            curVal = v
            if curVal > best_val + 0.01:
                curX = bestX.copy()
                curVal = best_val
        # hard time guard
        if TIME_BUDGET - (time.time() - t0) < 25.0:
            break

    remaining = TIME_BUDGET - (time.time() - t0) - 2.0
    if remaining > 5.0:
        bestX = _lp_polish(bestX, seconds=remaining)
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # safety: remove NaN
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    bestX = bestX / nr
    return bestX
