import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_run(X, steps, beta0, beta1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0 = np.log(beta0)
    lb1 = np.log(beta1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        # E = exp(beta*(G-M)) clipped
        G -= M
        G *= beta
        np.clip(G, -20.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-300:
            continue
        H = (G @ X) * (2.0 / S)
        dot = np.sum(H * X, axis=1, keepdims=True)
        H -= dot * X  # tangent gradient Gt
        V *= mu
        V += (1.0 - mu) * H
        vn = np.sqrt(np.sum(V * V, axis=1))
        mv = float(np.max(vn))
        if mv < 1e-15:
            continue
        X -= V * (lr / mv)
        nrm = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    # load start
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((182, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (182, 9):
        # try to reshape / truncate
        tmp = np.zeros((182, 9))
        m = min(xb.shape[0], 182)
        tmp[:m, :min(xb.shape[1], 9)] = xb[:m, :min(xb.shape[1], 9)]
        # fill rest random
        if m < 182:
            rr = rng.standard_normal((182 - m, 9))
            tmp[m:] = rr
        xb = tmp
    nrm = np.linalg.norm(xb, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    xb /= nrm
    best_val = max_cos(xb)
    best_X = xb.copy()

    n, d = 182, 9
    # time budget: explore until ~125s, then squeeze
    explore_deadline = t0 + 125.0
    hard_deadline = t0 + 170.0
    cycle = 0
    sigmas = [0.005, 0.01, 0.02, 0.035, 0.06, 0.09]
    while time.time() < explore_deadline and time.time() < hard_deadline - 30:
        cycle += 1
        sigma = sigmas[cycle % len(sigmas)]
        # alternate perturbation modes
        mode = cycle % 4
        X = best_X.copy()
        if mode == 3:
            # delete-and-reinsert worst k points
            k = int(rng.integers(2, 5))  # 2-4
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            idx = np.argsort(-rowmax)[:k]
            # keep others fixed initially, greedy best-of-many
            others_mask = np.ones(n, dtype=bool)
            others_mask[idx] = False
            cur = X[others_mask]  # (n-k, d)
            for ii in idx:
                cand = rng.standard_normal((2000, d))
                cn = np.linalg.norm(cand, axis=1, keepdims=True)
                cn[cn == 0] = 1.0
                cand /= cn
                # score = max dot to cur
                dots = cand @ cur.T
                scores = np.max(dots, axis=1)
                # also penalize closeness among new? cur grows
                best_c = cand[np.argmin(scores)]
                X[ii] = best_c
                cur = np.vstack([cur, best_c[None, :]])
            # small noise on the rest
            X[others_mask] += 0.005 * rng.standard_normal((n - k, d))
        elif mode == 2:
            # targeted kick of worst points + global noise
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            rowmax = np.max(G, axis=1)
            idx = np.argsort(-rowmax)[:8]
            X += sigma * 0.5 * rng.standard_normal((n, d))
            X[idx] += (sigma * 1.5) * rng.standard_normal((len(idx), d))
        else:
            X += sigma * rng.standard_normal((n, d))
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X /= nn

        # stage 1 anneal: broad
        if sigma > 0.04:
            X = smooth_run(X, steps=2200, beta0=30.0, beta1=180.0, lr0=0.025, lr1=0.002, mu=0.82)
        else:
            X = smooth_run(X, steps=2200, beta0=40.0, beta1=350.0, lr0=0.018, lr1=0.0012, mu=0.86)
        v = max_cos(X)
        if v < best_val:
            best_val = v
            best_X = X.copy()
            # immediate high-beta refinement
            if time.time() < hard_deadline - 35:
                Xr = smooth_run(X.copy(), steps=900, beta0=350.0, beta1=650.0, lr0=0.006, lr1=0.0008, mu=0.9)
                vr = max_cos(Xr)
                if vr < best_val:
                    best_val = vr
                    best_X = Xr.copy()
        else:
            # try a sharp refinement anyway occasionally to squeeze
            if cycle % 5 == 0 and time.time() < hard_deadline - 40:
                Xr = smooth_run(X.copy(), steps=800, beta0=300.0, beta1=600.0, lr0=0.005, lr1=0.0008, mu=0.9)
                vr = max_cos(Xr)
                if vr < best_val:
                    best_val = vr
                    best_X = Xr.copy()
        # safety: if we already beat by good margin, keep exploring
        if time.time() > explore_deadline:
            break

    # pre-squeeze with high-beta smooth
    try:
        if time.time() < hard_deadline - 8:
            best_X = smooth_run(best_X.copy(), steps=1200, beta0=400.0, beta1=800.0, lr0=0.004, lr1=0.0006, mu=0.9)
            best_val = max_cos(best_X)
    except Exception:
        pass

    # final LP squeeze with remaining time
    rem = hard_deadline - time.time() - 1.0
    if rem > 3:
        try:
            polished = lp_polish(best_X.copy(), min(rem, 30.0))
            pv = max_cos(polished)
            if pv < best_val:
                best_val = pv
                best_X = polished
        except Exception:
            pass

    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # strict validity: replace NaN/inf if any
    best_X = np.nan_to_num(best_X, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(best_X, axis=1, keepdims=True)
    zero_rows = (nn[:, 0] < 1e-12)
    if np.any(zero_rows):
        rr = rng.standard_normal((int(np.sum(zero_rows)), 9))
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        best_X[zero_rows] = rr
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
