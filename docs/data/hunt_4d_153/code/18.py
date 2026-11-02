import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, deadline, max_solves=100):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _softmax_quench(x, steps, T0, T1, lrfac, mu):
    n, d = x.shape
    vel = np.zeros_like(x)
    vmax = 0.06
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        vel = mu * vel - lr * gt
        # clip per-row velocity
        vn = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, vmax / vn)
        vel *= scale
        x = x + vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _riesz_quench(x, steps, s_pow, lr0, mu):
    n, d = x.shape
    vel = np.zeros_like(x)
    vmax = 0.05
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        D2 = np.maximum(2.0 - 2.0 * G, 1e-6)
        W = D2 ** (-s_pow / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        # avoid extreme weights
        # normalize by median to keep stable
        med = np.median(W[W > 0]) + 1e-12
        W /= med
        W = np.minimum(W, 20.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lr0 * (1.0 - 0.7 * frac)
        vel = mu * vel - lr * gt
        vn = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, vmax / vn)
        vel *= scale
        x = x + vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _find_void(x_keep, rng, ns=15000):
    R = rng.standard_normal(size=(ns, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x_keep.T
    m = S.max(axis=1)
    return R[int(np.argmin(m))]


def solve():
    t0 = time.time()
    deadline = t0 + 176.0
    explore_end = t0 + 138.0
    rng = np.random.default_rng(20261102)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = best_x.copy()
    cur_val = best_val

    n = 153
    trial = 0
    T_accept = 0.0012
    no_improve = 0

    while time.time() < explore_end:
        trial += 1
        if trial % 12 == 0:
            base = best_x
        else:
            base = cur_x
        if cur_val > best_val + 0.012 and (trial % 5 == 0):
            base = best_x
        # stronger diversification if stuck
        stuck = no_improve > 25
        mode = trial % 7
        if stuck and (trial % 3 == 0):
            mode = 6
        y = base.copy()

        if mode == 0:
            sig = 0.035 if stuck else 0.018
            y = y + sig * rng.standard_normal(y.shape)
        elif mode == 1:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            idx = np.argsort(-per)[:10]
            y[idx] = y[idx] + 0.09 * rng.standard_normal((10, 4))
            y = y + 0.004 * rng.standard_normal(y.shape)
        elif mode == 2:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            w = int(np.argmax(per))
            keep = np.delete(base, w, axis=0)
            v = _find_void(keep, rng, ns=15000)
            y[w] = v + 0.008 * rng.standard_normal(4)
            y = y + 0.003 * rng.standard_normal(y.shape)
        elif mode == 3:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            ws = np.argsort(-per)[:2]
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=9000)
                y[w] = v + 0.006 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y = y + 0.003 * rng.standard_normal(y.shape)
        elif mode == 4:
            # coherent patch move
            c = rng.standard_normal(4)
            c /= np.linalg.norm(c)
            dots = base @ c
            k = int(rng.integers(12, 22))
            sel = np.argsort(-dots)[:k]
            delta = 0.07 * rng.standard_normal(4)
            delta -= (delta @ c) * c
            y[sel] = y[sel] + delta + 0.015 * rng.standard_normal((k, 4))
            y = y + 0.002 * rng.standard_normal(y.shape)
        elif mode == 5:
            # explicit edge split of worst pairs
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            # find worst 3 pairs
            iu0 = np.triu_indices(n, 1)
            order = np.argsort(-G[iu0])
            for t in range(3):
                ii = iu0[0][order[t]]
                jj = iu0[1][order[t]]
                d = y[ii] - y[jj]
                d /= (np.linalg.norm(d) + 1e-12)
                y[ii] = y[ii] + 0.02 * d + 0.008 * rng.standard_normal(4)
                y[jj] = y[jj] - 0.02 * d + 0.008 * rng.standard_normal(4)
            y = y + 0.003 * rng.standard_normal(y.shape)
        else:
            sig = 0.13 if (trial % 2 == 0) else 0.06
            y = y + sig * rng.standard_normal(y.shape)

        y /= np.linalg.norm(y, axis=1, keepdims=True)

        # choose quench kernel
        r = rng.random()
        if r < 0.65:
            T0 = float(rng.uniform(0.018, 0.042))
            T1 = float(rng.uniform(0.0012, 0.0032))
            steps = int(rng.integers(480, 720))
            mu = float(rng.uniform(0.80, 0.90))
            lrfac = float(rng.uniform(0.5, 0.9))
            y = _softmax_quench(y, steps, T0, T1, lrfac, mu)
            # short hard tail
            y = _softmax_quench(y, 110, 0.0035, 0.0012, 0.7, 0.8)
        else:
            s_pow = float(rng.uniform(6.0, 14.0))
            steps = int(rng.integers(350, 550))
            mu = float(rng.uniform(0.75, 0.88))
            lr0 = float(rng.uniform(0.010, 0.022))
            y = _riesz_quench(y, steps, s_pow, lr0, mu)
            y = _softmax_quench(y, 130, 0.004, 0.0012, 0.7, 0.8)

        v = _max_cos(y)

        if v < best_val - 1e-12:
            best_x, best_val = y, v
            cur_x, cur_val = y.copy(), v
            no_improve = 0
            continue
        no_improve += 1
        if v < cur_val:
            cur_x, cur_val = y, v
        else:
            if rng.random() < np.exp((cur_val - v) / T_accept):
                cur_x, cur_val = y, v

        # occasional LP intensification on wandering state
        if (trial % 8 == 0) and time.time() < explore_end - 5.0 and v < best_val + 0.0025:
            y2, v2 = _lp_polish(y, time.time() + 1.0, max_solves=10)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                cur_x, cur_val = y2.copy(), v2
                no_improve = 0
            elif v2 < cur_val:
                cur_x, cur_val = y2, v2

    # final thorough squeeze of best
    if time.time() < deadline - 1.0:
        best_x, best_val = _lp_polish(best_x, deadline - 0.5, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
