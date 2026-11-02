import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t0=None):
    """Trust-region LP polish."""
    if t0 is None:
        t0 = time.time()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - t0 > seconds:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_anneal(Xn, alphas, iters_list, lrs, iu0, iu1, t_deadline):
    n, d = Xn.shape
    m = np.zeros_like(Xn)
    v = np.zeros_like(Xn)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    # ensure normalized
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    for ai, alpha in enumerate(alphas):
        iters = iters_list[ai]
        lr = lrs[ai]
        for k in range(iters):
            if (k & 127) == 0 and time.time() > t_deadline:
                return Xn
            G = Xn @ Xn.T
            g = G[iu0, iu1]
            mx = g.max()
            e = np.exp(alpha * (g - mx))
            s = e.sum()
            w = e / s
            W = np.zeros((n, n), dtype=Xn.dtype)
            W[iu0, iu1] = w
            W += W.T
            grad = W @ Xn
            r = (grad * Xn).sum(axis=1, keepdims=True)
            grad_t = grad - r * Xn
            b1p *= beta1
            b2p *= beta2
            m *= beta1
            m += (1 - beta1) * grad_t
            v *= beta2
            v += (1 - beta2) * (grad_t * grad_t)
            mhat = m / (1.0 - b1p)
            vhat = v / (1.0 - b2p)
            step = mhat / (np.sqrt(vhat) + eps)
            rs = (step * Xn).sum(axis=1, keepdims=True)
            step_t = step - rs * Xn
            Xn = Xn - lr * step_t
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    return Xn

def solve():
    t0 = time.time()
    HARD = 172.0  # must finish before 180
    rng = np.random.default_rng(12345)
    n, d = 74, 13
    # load start
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                va = max_cos(a)
                if va < best_val:
                    best_val = va
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    iu0, iu1 = np.triu_indices(n, 1)
    alphas = [120.0, 350.0, 900.0, 2200.0]
    iters_list = [500, 600, 700, 900]
    lrs = [0.010, 0.008, 0.006, 0.004]
    sigmas = [0.008, 0.015, 0.03, 0.06, 0.12]
    trial = 0
    # global search until ~145s, reserve rest for LP
    while time.time() - t0 < 143.0:
        if time.time() - t0 > HARD - 30:
            break
        trial += 1
        t_deadline = min(t0 + 143.0, HARD - 30)
        if trial % 6 == 0:
            # largest-hole reinsertion of most crowded point
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            k = int(np.argmax(rowmax))
            Rc = rng.standard_normal((2500, d))
            Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)
            dots = Rc @ best.T
            dots[:, k] = -2.0
            cmax = dots.max(axis=1)
            bc = Rc[int(np.argmin(cmax))]
            Xt = best.copy()
            Xt[k] = bc
            # small jitter others to break symmetry
            Xt += 0.003 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        else:
            sigma = sigmas[(trial - 1) % len(sigmas)]
            # occasionally big escape
            if trial % 23 == 0:
                sigma = 0.25
            Xt = best + sigma * rng.standard_normal(best.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # half the time push worst pair apart explicitly
            if trial % 2 == 0:
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                idx = int(np.argmax(G))
                ii, jj = idx // n, idx % n
                # need correct 2d index for upper; brute force
                # find pair via argmax
                flat = np.argmax(G)
                wi, wj = flat // n, flat % n
                push = 0.5 * sigma
                Xt[wi] -= push * Xt[wj]
                Xt[wj] -= push * Xt[wi]
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        Xt2 = smooth_anneal(Xt, alphas, iters_list, lrs, iu0, iu1, t_deadline)
        val = max_cos(Xt2)
        if val < best_val:
            best = Xt2
            best_val = val
        # also try cheap greedy acceptance of intermediate? keep best only
    # final squeeze: short high-alpha refine then LP
    try:
        if time.time() - t0 < HARD - 28:
            best = smooth_anneal(best, [2500.0], [1200], [0.003], iu0, iu1, t0 + HARD - 25)
            best_val = max_cos(best)
    except Exception:
        pass
    try:
        remain = (t0 + HARD - 3) - time.time()
        if remain > 5:
            best = lp_polish(best, t0 + HARD - 3, time.time())
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
