import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

_TRIU = None

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _make_fg(n, d, k, iu0, iu1):
    def fg(yflat):
        y = yflat.reshape(n, d)
        r = np.sqrt(np.sum(y * y, axis=1, keepdims=True))
        r = np.maximum(r, 1e-12)
        X = y / r
        G = X @ X.T
        gvec = G[iu0, iu1]
        M = np.max(gvec)
        e = np.exp(k * (gvec - M))
        S = np.sum(e)
        f = M + np.log(S) / k
        pvec = e / S
        P = np.zeros((n, n))
        P[iu0, iu1] = pvec
        P = P + P.T
        V = P @ X
        D = np.sum(P * G, axis=1, keepdims=True)
        grad_y = (V - D * X) / r
        return float(f), grad_y.ravel()
    return fg

def _anneal(x0, ks, maxiters, t_deadline):
    n, d = x0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    y0 = x0.ravel().copy()
    for k, mi in zip(ks, maxiters):
        if time.time() > t_deadline:
            break
        fg = _make_fg(n, d, float(k), iu0, iu1)
        try:
            res = minimize(fg, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mi), "ftol": 1e-14, "gtol": 1e-09, "maxfun": 100000})
            y0 = res.x
        except Exception:
            break
    y = y0.reshape(n, d)
    y = _norm(y)
    return y

def _lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.5:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.RandomState(0)
    np.random.seed(0)

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
        cands.append(_norm(np.asarray(xb, dtype=float)))
    if xr is not None:
        cands.append(_norm(np.asarray(xr, dtype=float)))
    if len(cands) == 0:
        cands.append(_norm(rng.randn(73, 13)))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = best.shape

    it = 0
    # quick initial LP to ensure local optimality baseline (short)
    # skip long polish now, go to hopping
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() > DEADLINE - 35:
            break
        # choose start
        if it % 9 == 0:
            # cold random restart
            x0 = _norm(rng.randn(n, d))
            ks = [60, 150, 350, 700]
            mis = [120, 150, 180, 180]
        else:
            # perturb best
            ppm, gm = _per_point_max(best)
            slack = gm - ppm
            # loosest (rattlers) first
            order = np.argsort(-slack)  # largest slack first
            sigma = float(rng.choice([0.01, 0.03, 0.07, 0.15, 0.25, 0.35]))
            x0 = best + rng.randn(n, d) * sigma
            nkick = int(rng.randint(1, 5))
            kick = set()
            # include a rattler with high prob
            if rng.rand() < 0.8:
                kick.add(int(order[rng.randint(0, min(5, n))]))
            while len(kick) < nkick:
                kick.add(int(rng.randint(0, n)))
            kick = list(kick)
            if rng.rand() < 0.45:
                # full reinsertion of kicked points
                for idx in kick:
                    v = rng.randn(d)
                    v /= np.linalg.norm(v) + 1e-12
                    # place near random direction, not near best
                    x0[idx] = v
            else:
                for idx in kick:
                    x0[idx] += rng.randn(d) * float(rng.choice([0.4, 0.7, 1.0]))
            x0 = _norm(x0)
            # annealing sharpness depends on sigma: small sigma -> sharper
            if sigma < 0.05:
                ks = [150, 400, 800]
                mis = [100, 120, 150]
            elif sigma < 0.15:
                ks = [60, 150, 400, 800]
                mis = [80, 100, 120, 120]
            else:
                ks = [40, 100, 250, 600]
                mis = [80, 100, 130, 150]

        x1 = _anneal(x0, ks, mis, DEADLINE - 32)
        v1 = _max_cos(x1)
        if v1 < best_val:
            best = x1
            best_val = v1
        # opportunistic: if very close and we have time, try short LP to see true local opt
        # only for promising candidates to save time
        if v1 < best_val + 0.004 and (it % 7 == 0) and (time.time() < EXPLORE_END - 20):
            # very short LP probe (2 sec) on x1
            try:
                xp = _lp_polish(x1, 2.0)
                vp = _max_cos(xp)
                if vp < best_val:
                    best = xp
                    best_val = vp
            except Exception:
                pass

    remain = DEADLINE - time.time() - 1.5
    if remain > 4:
        try:
            best = _lp_polish(best, min(remain, 32.0))
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    # final safety: remove NaN/zero
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best = best / nr
    return best
