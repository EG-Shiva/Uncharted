import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds):
    """Trust-region LP: move points tangentially to lower the largest cosine."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def adam_smooth(x_init, steps, beta0, beta1, lr0, lr1):
    """Projected globally-adaptive repulsion on soft-max(beta*dot)."""
    x = x_init.copy()
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_gl = 1e-12
    b1, b2, eps = 0.9, 0.999, 1e-8
    pow1, pow2 = 1.0, 1.0
    lrat = beta1 / max(beta0, 1e-12)
    lrrat = lr1 / max(lr0, 1e-12)
    for t in range(1, steps + 1):
        frac = (t - 1) / (steps - 1) if steps > 1 else 1.0
        beta = beta0 * (lrat ** frac)
        lr = lr0 * (lrrat ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = np.max(G)
        # all D<=0 so no overflow
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        m_ad = b1 * m_ad + (1.0 - b1) * grad_t
        gv = float(np.mean(grad_t * grad_t))
        v_gl = b2 * v_gl + (1.0 - b2) * gv
        pow1 *= b1
        pow2 *= b2
        mhat = m_ad / (1.0 - pow1)
        vhat = v_gl / (1.0 - pow2)
        step = (lr / (np.sqrt(vhat) + eps)) * mhat
        # safety cap per point
        sn = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        cap = 0.05
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        # scale is (n,1), broadcast
        step = step * scale
        x = x - step
        x /= np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x


def reinsert_worst(x, rng, k_remove=2, n_cand=4000):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    row_max = np.max(G, axis=1)
    worst = np.argsort(-row_max)[:k_remove]
    x_new = x.copy()
    for idx in worst:
        cand = rng.randn(n_cand, 4)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        D = cand @ x_new.T
        D[:, idx] = -2.0
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        x_new[idx] = cand[bi]
    return x_new


def solve():
    t0 = time.time()
    TIME_BUDGET = 170.0
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()

    trial = 0
    # main basin-hopping loop, reserve time for final polish
    while time.time() - t0 < 148.0:
        trial += 1
        op = rng.rand()
        if op < 0.40:
            # global shake
            sigma = float([0.02, 0.04, 0.07, 0.10, 0.15][rng.randint(5)] * (0.8 + 0.4 * rng.rand()))
            noise = rng.randn(*x_best.shape)
            xt = x_best + sigma * noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        elif op < 0.70:
            # cluster shake around random center
            center = int(rng.randint(x_best.shape[0]))
            dots = x_best @ x_best[center]
            k = int([10, 15, 25, 40][rng.randint(4)])
            idx = np.argsort(-dots)[:k]
            s_big = float([0.05, 0.10, 0.18, 0.25][rng.randint(4)])
            s_small = s_big * 0.15
            noise = rng.randn(*x_best.shape) * s_small
            noise[idx] = rng.randn(len(idx), 4) * s_big
            xt = x_best + noise
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
        else:
            # worst-point greedy reinsertion + small shake
            k_remove = int([1, 2, 3][rng.randint(3)])
            n_cand = int([2000, 4000, 6000][rng.randint(3)])
            xt = reinsert_worst(x_best, rng, k_remove=k_remove, n_cand=n_cand)
            sigma = float(0.01 + 0.02 * rng.rand())
            xt = xt + sigma * rng.randn(*xt.shape)
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)

        steps = int([1200, 1800, 2500][rng.randint(3)])
        beta0 = 30.0
        beta1 = float([200.0, 300.0, 450.0, 600.0][rng.randint(4)])
        lr0 = float([0.008, 0.012, 0.02][rng.randint(3)])
        lr1 = lr0 * 0.1
        try:
            xs = adam_smooth(xt, steps, beta0, beta1, lr0, lr1)
        except Exception:
            continue
        sc = max_cos(xs)
        if sc < best:
            # immediate improvement without LP
            best = sc
            x_best = xs.copy()
            # refine it a bit
            remain = TIME_BUDGET - (time.time() - t0) - 3.0
            if remain > 1.0:
                try:
                    xp = lp_polish(x_best, min(2.0, remain))
                    sp = max_cos(xp)
                    if sp < best:
                        best = sp
                        x_best = xp
                except Exception:
                    pass
        elif sc < best + 0.002:
            remain = TIME_BUDGET - (time.time() - t0) - 5.0
            if remain < 1.0:
                continue
            try:
                xp = lp_polish(xs, min(2.0, remain))
            except Exception:
                continue
            sp = max_cos(xp)
            if sp < best:
                best = sp
                x_best = xp.copy()
        # early exit if time low
        if TIME_BUDGET - (time.time() - t0) < 6.0:
            break

    remain = TIME_BUDGET - (time.time() - t0)
    if remain > 2.0:
        try:
            x_best = lp_polish(x_best, min(remain - 0.5, 25.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
