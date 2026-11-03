import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def contacts_per_point(x, thresh=1e-4):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    cnt = np.sum(g > m - thresh, axis=1)
    return m, cnt


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    Gfull = None
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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


def smooth_refine(x0, taus=(0.04, 0.015), maxiters=(120, 150)):
    n, d = x0.shape
    flat = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()

    for tau, mi in zip(taus, maxiters):
        def fun(f):
            X = f.reshape(n, d)
            norms = np.linalg.norm(X, axis=1)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms[:, None]
            G = Xn @ Xn.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0:
                S = 1e-300
            val = m + tau * np.log(S / 2.0 + 1e-300)
            W = E / S
            Gw = W @ Xn
            grad_Xn = 2.0 * Gw
            dot = (grad_Xn * Xn).sum(axis=1, keepdims=True)
            grad_X = (grad_Xn - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            grad_X = grad_X + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, grad_X.ravel()

        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": mi * 3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def random_sphere(rng, k, d):
    a = rng.randn(k, d)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def solve():
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    t0 = time.time()
    DEADLINE = t0 + 172.0

    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = random_sphere(rng, 162, 8)
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (162, 8):
        # try to fix shape if transposed etc
        xb = xb.reshape(162, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    # quick analysis of best for targeting
    # precompute contact order
    trial = 0
    # small warm final budget reserve
    while time.time() < DEADLINE - 13.0:
        trial += 1
        if time.time() > DEADLINE - 13.0:
            break
        m, cnt = contacts_per_point(x_best, thresh=2e-3)
        order = np.argsort(cnt)  # ascending
        # choose strategy
        mode = trial % 6
        x0 = x_best.copy()
        n, d = x0.shape
        if mode == 0:
            # global small jitter
            sigma = float(rng.uniform(0.008, 0.03))
            x0 = x0 + sigma * rng.randn(n, d)
        elif mode == 1:
            # shake overloaded top points hard
            k = int(rng.randint(6, 16))
            idx = np.argsort(cnt)[-k:]  # most contacts
            sigma = float(rng.uniform(0.06, 0.18))
            x0[idx] = x0[idx] + sigma * rng.randn(k, d)
            # also tiny global
            x0 = x0 + 0.005 * rng.randn(n, d)
        elif mode == 2:
            # random subset re-init / large kick
            k = int(rng.randint(1, 5))
            idx = rng.choice(n, size=k, replace=False)
            sigma = float(rng.uniform(0.15, 0.35))
            # half replace, half kick
            if rng.rand() < 0.5:
                x0[idx] = random_sphere(rng, k, d)
            else:
                x0[idx] = x0[idx] + sigma * rng.randn(k, d)
        elif mode == 3:
            # rattler / hole reinsertion
            # find rattlers (low contacts with tight thresh)
            _, cnt_tight = contacts_per_point(x_best, thresh=1e-7)
            ratt = np.where(cnt_tight <= 2)[0]
            if len(ratt) == 0:
                ratt = order[:2]
            # pick one rattler to relocate to best hole
            ri = int(rng.choice(ratt))
            # candidate search for hole: maximize min distance = minimize max dot
            cand = random_sphere(rng, 3000, d)
            others = np.delete(x_best, ri, axis=0)
            dots = cand @ others.T
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            x0[ri] = cand[bi]
            # jitter neighbours of overloaded a bit
            k2 = 4
            idx2 = np.argsort(cnt)[-k2:]
            x0[idx2] = x0[idx2] + 0.04 * rng.randn(k2, d)
            x0 = x0 + 0.004 * rng.randn(n, d)
        elif mode == 4:
            # medium global + pair-split worst pair
            sigma = float(rng.uniform(0.03, 0.09))
            x0 = x0 + sigma * rng.randn(n, d)
        else:
            # low-contact shake + global
            k = int(rng.randint(4, 12))
            # mix of low and high
            idx_h = np.argsort(cnt)[-k//2:] if k//2 > 0 else np.array([], dtype=int)
            idx_l = np.argsort(cnt)[:k-len(idx_h)]
            idx = np.concatenate([idx_h, idx_l])
            sigma = float(rng.uniform(0.05, 0.15))
            x0[idx] = x0[idx] + sigma * rng.randn(len(idx), d)

        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        # smooth stage with varying tau
        # vary tau slightly deterministically
        tau1 = float(rng.uniform(0.03, 0.06))
        tau2 = float(rng.uniform(0.008, 0.02))
        try:
            xs = smooth_refine(x0, taus=(tau1, tau2), maxiters=(80, 120))
        except Exception:
            continue
        vs = max_cos(xs)
        # filter: only LP if promising
        if vs > best_val + 0.02:
            # not promising, but keep tiny chance: skip LP
            if vs < best_val:
                x_best = xs
                best_val = vs
            continue
        if time.time() > DEADLINE - 13.0:
            # no time for LP, just check smooth
            if vs < best_val - 1e-12:
                x_best = xs
                best_val = vs
            continue
        # short LP snap
        budget = 1.4
        # adjust budget if little time
        remain = DEADLINE - 13.0 - time.time()
        if remain < budget:
            budget = max(0.4, remain)
        try:
            xl = lp_polish(xs, budget)
        except Exception:
            continue
        vl = max_cos(xl)
        if vl < best_val - 1e-12:
            x_best = xl
            best_val = vl
        elif vs < best_val - 1e-12:
            x_best = xs
            best_val = vs

    # final long polish of best
    try:
        remain = DEADLINE - time.time() - 1.0
        if remain > 2.0:
            x_best = lp_polish(x_best, min(remain, 11.0))
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
