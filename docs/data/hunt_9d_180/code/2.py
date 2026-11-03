import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, seconds, t_deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _repel(x, iters, beta0, beta1, lr0, lr1, cut=0.25):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        S = G - gmax
        W = np.zeros_like(G)
        mk = S > -cut
        # exp weights, max is 1
        W[mk] = np.exp(beta * S[mk])
        np.fill_diagonal(W, 0.0)
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        x = x - lr * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    rng = np.random.RandomState(0)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    # ensure shape (180,9)
    if xb.shape != (180, 9):
        # fallback: tile/truncate deterministically
        tmp = rng.randn(180, 9)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m = min(n, 180)
        tmp[:m] = xb[:m, :9] if d >= 9 else np.concatenate([xb[:m], np.zeros((m, 9 - d))], 1)
        xb = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
        n, d = xb.shape

    best_val = _max_cos(xb)
    x_best = xb.copy()

    sigmas = [0.16, 0.10, 0.06, 0.035, 0.02, 0.012]
    beta_pairs = [(30.0, 150.0), (30.0, 300.0), (50.0, 400.0), (20.0, 120.0)]
    trial = 0
    # reserve ~22s for final polish
    while time.time() < DEADLINE - 24.0 and trial < 200:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        b0, b1 = beta_pairs[(trial - 1) % len(beta_pairs)]
        x_try = x_best.copy()

        if trial % 6 == 0:
            # hole reinsertion: remove worst point, put in largest hole
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            idx = int(np.argmax(rowmax))
            cand = rng.randn(4000, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            others = np.delete(x_try, idx, axis=0)
            dots = cand @ others.T
            m = dots.max(axis=1)
            x_try[idx] = cand[int(np.argmin(m))]
            x_try += rng.randn(n, d) * (sigma * 0.2)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            G = x_try @ x_try.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            deg = np.sum(G > gmax - 0.01, axis=1)
            probs = (deg + 1).astype(float)
            probs /= probs.sum()
            k = int(rng.choice(np.array([2, 3, 4, 6, 8])))
            k = min(k, n)
            idx = rng.choice(n, size=k, replace=False, p=probs)
            x_try[idx] += rng.randn(k, d) * sigma
            x_try += rng.randn(n, d) * (sigma * 0.05)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # fast repulsion annealing
        iters = 700 if sigma > 0.03 else 500
        lr0 = 0.012 if sigma > 0.05 else 0.008
        lr1 = 0.0015
        x_try = _repel(x_try, iters=iters, beta0=b0, beta1=b1, lr0=lr0, lr1=lr1)

        v_raw = _max_cos(x_try)
        if v_raw > best_val + 0.03:
            continue
        if time.time() > DEADLINE - 24.0:
            break
        # short LP snap
        x_try, v_pol = _lp_polish(x_try, seconds=2.6, t_deadline=DEADLINE - 22.0)
        if v_pol < best_val - 1e-12:
            x_best = x_try
            best_val = v_pol
            # extra exploitation: small jitter + repel around new best
            # continue to next trial (which uses smaller sigma cyclically)

    # final long polish of the best found
    remain = DEADLINE - time.time() - 1.0
    if remain > 2.0:
        x_best, best_val = _lp_polish(x_best, seconds=min(remain, 25.0), t_deadline=DEADLINE - 0.5)

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # strict cleanup
    x_best = np.nan_to_num(x_best, nan=1.0, posinf=1.0, neginf=-1.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best /= nr
    return x_best
