import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def smooth_refine(x, schedule, deadline=None, rng=None):
    """Adam on sphere for loss sum exp(alpha*(dot-Gmax)). schedule=(alpha,steps,lr)."""
    n, d = x.shape
    for alpha, steps, lr in schedule:
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            if deadline is not None and (t & 63) == 0 and time.time() > deadline:
                return x
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            gmax = float(np.max(G))
            # weights focused on top pairs
            # G- gmax <= 0
            W = np.exp(alpha * (G - gmax))
            # G diag was -2, fix diag weight to 0
            # recompute correctly: use full gram for W then zero diag
            # redo with true gram to avoid exp(-big):
            # Actually W diag currently exp(alpha*(-2-gmax)) ~0, fine, set 0
            np.fill_diagonal(W, 0.0)
            grad = W @ x
            # tangential projection
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            # Adam
            m = b1 * m + (1.0 - b1) * gt
            vv = gt * gt
            v = b2 * v + (1.0 - b2) * vv
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # safety clip per-row norm
            x = x - step
            x = normalize(x)
    return x


def lp_polish_deadline(x, deadline):
    x = normalize(x)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    iu0 = np.triu_indices(n, 1)
    # preallocate? simple loop
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        K = 1200
        win = max(30.0 * radius, 3e-7)
        if len(c_all) > K:
            thr = np.partition(c_all, -K)[-K]
            sel = c_all >= min(thr, m - win) - 1e-15
            # also ensure all within win are included (already if thr lower)
            # if thr > m-win, need union
            if thr > m - win:
                sel = (c_all >= thr - 1e-15) | (c_all > m - win)
        else:
            sel = c_all > m - win
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        # columns
        ar = np.arange(d)
        cols_I = (I[:, None] * d + ar).ravel()
        cols_J = (J[:, None] * d + ar).ravel()
        # interleave? build vals aligned with rows
        # rows order: for each k: d entries of I, d entries of J, 1 entry of t
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        # fill via loop over d? vectorized:
        # create blocks
        cols_blocks_I = (I[:, None] * d + ar)
        cols_blocks_J = (J[:, None] * d + ar)
        for k in range(p):
            base = k * (2 * d + 1)
            cols[base:base + d] = cols_blocks_I[k]
            vals[base:base + d] = x[J[k]]
            cols[base + d:base + 2 * d] = cols_blocks_J[k]
            vals[base + d:base + 2 * d] = x[I[k]]
            cols[base + 2 * d] = n * d
            vals[base + 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        b_ub = -c_all[sel]
        # equality: x_i . dx_i = 0
        erows = np.repeat(np.arange(n), d)
        ecols = np.arange(n * d)
        e vals = x.ravel()
        A_eq = coo_matrix((evals := x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = normalize(y)
        vv = max_cos(y)
        if vv < best - 1e-15:
            x, best = y, vv
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
            # if stuck with tiny radius, jitter radius up once?
        # early stop if time
    return x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = normalize(np.asarray(xb, dtype=np.float64))
    xr = normalize(np.asarray(xr, dtype=np.float64))
    if xb.shape != (93, 13):
        xb = xr
    best_x = xb.copy()
    best_v = max_cos(best_x)
    rv = max_cos(xr)
    if rv < best_v:
        best_x, best_v = xr.copy(), rv

    # quick fine tune of starting point with short smooth run
    # Phase 1: shake + smooth annealing
    phase1_end = t0 + 118.0
    sigmas = [0.003, 0.007, 0.015, 0.04, 0.008, 0.02]
    it = 0
    # schedules
    coarse = [(120.0, 200, 0.002), (350.0, 300, 0.001), (900.0, 300, 0.0005)]
    while time.time() < phase1_end:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        x0 = best_x + sigma * rng.standard_normal(best_x.shape)
        # every 6th: targeted kick of current max pair
        if it % 6 == 0:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            nn = best_x.shape[0]
            a, b = divmod(flat, nn)
            x0 = best_x.copy()
            x0[a] += 3.0 * sigma * rng.standard_normal(13)
            x0[b] += 3.0 * sigma * rng.standard_normal(13)
            # also small global noise
            x0 += 0.3 * sigma * rng.standard_normal(x0.shape)
        x0 = normalize(x0)
        # vary alpha slightly deterministically
        alpha_jit = 1.0 + 0.2 * ((it % 3) - 1) * 0.5
        sched = [(a * alpha_jit, s, l) for (a, s, l) in coarse]
        x1 = smooth_refine(x0, sched, deadline=phase1_end + 2.0)
        v1 = max_cos(x1)
        if v1 < best_v:
            best_x, best_v = x1, v1
        # opportunistic hole reinsertion when stuck: try replacing one of max pair
        if it % 10 == 0 and time.time() < phase1_end:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            nn = best_x.shape[0]
            a, b = divmod(flat, nn)
            for victim in (a, b):
                mask = np.ones(nn, dtype=bool)
                mask[victim] = False
                rest = best_x[mask]
                cand = normalize(rng.standard_normal((4000, 13)))
                scores = np.max(cand @ rest.T, axis=1)
                bi = int(np.argmin(scores))
                xt = best_x.copy()
                xt[victim] = cand[bi]
                # local refine victim only via few smooth steps on full set
                xt = smooth_refine(xt, [(400.0, 150, 0.001)], deadline=phase1_end + 2.0)
                vt = max_cos(xt)
                if vt < best_v:
                    best_x, best_v = xt, vt
                    break
        # keep loop responsive to time
    # Phase 2: fine smooth on best then LP polish
    if time.time() < deadline - 5:
        try:
            best_x = smooth_refine(best_x, [(500.0, 600, 0.0004), (1200.0, 600, 0.0002)], deadline=deadline - 35)
            best_v = max_cos(best_x)
        except Exception:
            pass
    if time.time() < deadline - 2:
        try:
            best_x = lp_polish_deadline(best_x, deadline - 0.5)
            best_v = max_cos(best_x)
        except Exception:
            pass
    best_x = normalize(best_x)
    return best_x
