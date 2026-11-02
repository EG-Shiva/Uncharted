import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # cache iu arrays
    iu_r = iu0[0]
    iu_c = iu0[1]
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds or radius < 1e-14:
            break
        # gram and selection
        G = x @ x.T
        c_all = G[iu_r, iu_c]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        # cap constraints for speed
        if np.count_nonzero(sel) > 1400:
            # keep tightest 1400
            idx = np.argsort(c_all)[-1400:]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I = iu_r[sel]
        J = iu_c[sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        cols_t = np.full(p, n * d)
        cols = np.concatenate([
            cols_I.reshape(p, d).reshape(p, d) if False else cols_I,
        ])
        # build cols vectorised: [I*d+d, J*d+d, t]
        cols_full = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols_full[0::(2 * d + 1)] = None  # placeholder not used
        # simpler: construct via concatenate of 2D blocks
        block = np.concatenate([ (I[:, None] * d + ar), (J[:, None] * d + ar), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows2, block)), shape=(p, nv)).tocsr()
            # A_eq: x_i . dx_i = 0
            Ae_rows = np.repeat(np.arange(n), d)
            Ae_cols = np.arange(n * d)
            Ae_vals = x.ravel()
            A_eq = coo_matrix((Ae_vals, (Ae_rows, Ae_cols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            remain = seconds - (time.time() - start)
            if remain <= 0.05:
                break
            tl = min(max(remain, 0.5), 5.0)
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl, "presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_hole_candidate(x, exclude, rng, K=20000):
    n, d = x.shape
    mask = np.ones(n, dtype=bool)
    mask[exclude] = False
    Xo = x[mask]  # (n-1,d)
    # sample in chunks to bound memory/time
    best_c = None
    best_s = np.inf
    chunk = 5000
    for _ in range(0, K, chunk):
        kk = min(chunk, K - _)
        C = rng.standard_normal((kk, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ Xo.T
        s = S.max(axis=1)
        j = int(np.argmin(s))
        if float(s[j]) < best_s:
            best_s = float(s[j])
            best_c = C[j].copy()
    return best_c


def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    rng = np.random.default_rng(12345)
    # load starts
    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (58, 22):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        x_best = rng.standard_normal((58, 22))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    else:
        vals = [max_cos(a) for a in cand]
        x_best = cand[int(np.argmin(vals))]
    n, d = 58, 22
    best_val = max_cos(x_best)

    # short initial quench to ensure local optimum (cheap)
    if TIME_BUDGET - (time.time() - t0) > 20:
        x_best = lp_polish(x_best, 3.0, radius0=1e-3)
        best_val = max_cos(x_best)

    it = 0
    # precompute per-point quality for victim choice
    while time.time() - t0 < TIME_BUDGET - 12.0:
        it += 1
        time_left = TIME_BUDGET - 12.0 - (time.time() - t0)
        if time_left <= 1.0:
            break
        budget = min(4.0, max(1.2, time_left / 8.0))
        # choose perturbation type
        mode = it % 4
        y = None
        try:
            if mode == 0 or mode == 3:
                # global kick with varying sigma
                sig = [0.008, 0.02, 0.05, 0.09][rng.integers(0, 4)]
                y = x_best + sig * rng.standard_normal((n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            elif mode == 1:
                # worst-point hole reinsertion (1 or 2 points)
                G = x_best @ x_best.T
                np.fill_diagonal(G, -np.inf)
                per_max = G.max(axis=1)
                # victims: worst few
                order = np.argsort(per_max)[::-1]
                nvictim = 1 if rng.random() < 0.7 else 2
                victims = order[:5]
                # pick randomly among top-5
                chosen = rng.choice(victims, size=nvictim, replace=False)
                y = x_best.copy()
                for v in chosen:
                    bc = best_hole_candidate(y, int(v), rng, K=15000)
                    y[int(v)] = bc
            else:
                # subset strong shake
                m = int(rng.integers(1, 4))
                # bias to high-contact points
                G = x_best @ x_best.T
                np.fill_diagonal(G, -np.inf)
                per_max = G.max(axis=1)
                order = np.argsort(per_max)[::-1]
                pool = order[:12]
                idx = rng.choice(pool, size=min(m, len(pool)), replace=False)
                y = x_best.copy()
                y[idx] += 0.35 * rng.standard_normal((len(idx), d))
                y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
        except Exception:
            continue
        # quench
        try:
            yq = lp_polish(y, budget, radius0=2e-3)
            v = max_cos(yq)
        except Exception:
            continue
        if v < best_val - 1e-12 and np.isfinite(v):
            x_best = yq
            best_val = v
        # opportunistic: if very close, keep exploring from new point occasionally
        # (no acceptance of worse to stay hill-climbing)

    # final polish with remaining time
    remain = TIME_BUDGET - (time.time() - t0)
    if remain > 1.0:
        try:
            x_best = lp_polish(x_best, min(remain, 10.0), radius0=8e-4)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
