import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, time_limit, deadline, thr_floor=1e-6):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    t0 = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - t0 < time_limit and time.time() < deadline:
        if radius < 1e-13:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        thr = max(20.0 * radius, thr_floor)
        sel = c_all > m - thr
        # cap constraints for speed
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        I = iu0[sel]
        J = iu1[sel]
        c_sel = c_all[sel]
        if p > 3500:
            # keep tightest 3500
            idx = np.argpartition(-c_sel, 3500)[:3500]
            I = I[idx]
            J = J[idx]
            c_sel = c_sel[idx]
            p = 3500
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([
            I[:, None] * d + np.arange(d),
            J[:, None] * d + np.arange(d),
            np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
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
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best


def _smooth_opt(X0, T, maxiter):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    flat0 = X0.ravel().copy()

    def fg(flat):
        X = flat.reshape(n, d)
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Y = X / rn
        C = Y @ Y.T
        c = C[iu0, iu1]
        cm = c.max()
        e = np.exp((c - cm) / T)
        Z = e.sum()
        f = cm + T * np.log(Z)
        w = e / Z
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GY = W @ Y
        dot = (GY * Y).sum(axis=1, keepdims=True)
        GX = (GY - Y * dot) / rn
        lam = 0.01
        diff = rn - 1.0
        f = f + lam * np.mean(diff ** 2)
        GX = GX + (2.0 * lam / n) * diff * Y
        return float(f), GX.ravel()

    res = minimize(fg, flat0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})
    X = res.x.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _anneal(X, Ts, iters):
    for T, it in zip(Ts, iters):
        X = _smooth_opt(X, T, it)
    return X


def solve():
    t_start = time.time()
    deadline = t_start + 168.0
    rng = np.random.default_rng(7)
    np.random.seed(7)

    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None

    cands = []
    if xb is not None:
        xb = xb.astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    if xr is not None:
        xr = xr.astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    if not cands:
        tmp = rng.standard_normal((219, 6))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)

    best_x = min(cands, key=_max_cos).copy()
    best_v = _max_cos(best_x)
    n, d = best_x.shape

    # Stage 1: balanced LP on incumbent (fixes ping-pong with 1 contact)
    if time.time() < deadline - 20:
        remain = min(12.0, deadline - 20 - time.time())
        if remain > 1:
            nx, nv = _lp_polish(best_x, remain, deadline - 12, thr_floor=1e-6)
            if nv < best_v:
                best_x, best_v = nx, nv

    # Stage 2: basin hopping
    Ts = [0.012, 0.005, 0.002, 0.0009]
    Iters = [120, 120, 120, 150]
    trial = 0
    sigmas = [0.025, 0.07, 0.012, 0.045]

    while time.time() < deadline - 12 and trial < 200:
        trial += 1
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)

        if trial % 4 == 0:
            # remove-and-reinsert worst point
            k = int(order[0])
            others = np.delete(best_x, k, axis=0)
            cand = rng.standard_normal((2500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            x_try = best_x.copy()
            x_try[k] = cand[bi]
        else:
            K = 14 + (trial % 12)
            cul = order[:K]
            sigma = sigmas[trial % len(sigmas)]
            x_try = best_x.copy()
            x_try[cul] += rng.standard_normal((K, d)) * sigma
            # tiny global jitter
            x_try += rng.standard_normal((n, d)) * 0.003
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        if time.time() > deadline - 12:
            break
        try:
            x_try = _anneal(x_try, Ts, Iters)
        except Exception:
            continue
        if time.time() > deadline - 11:
            # skip LP, just evaluate
            v = _max_cos(x_try)
            if v < best_v:
                best_x, best_v = x_try, v
            continue
        try:
            x_try, v = _lp_polish(x_try, 2.5, deadline - 10, thr_floor=8e-7)
        except Exception:
            v = _max_cos(x_try)
        if v < best_v - 1e-12:
            best_x, best_v = x_try, v

    # Final squeeze
    if time.time() < deadline - 2:
        try:
            nx, nv = _lp_polish(best_x, max(0.5, deadline - 2 - time.time()),
                                deadline - 0.5, thr_floor=7e-7)
            if nv < best_v:
                best_x, best_v = nx, nv
        except Exception:
            pass

    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
