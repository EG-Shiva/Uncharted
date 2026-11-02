import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def contact_counts(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    cnt = np.sum(g > m - tol, axis=1)
    return cnt, m


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1.5e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if np.sum(sel) > 2500:
            idx = np.argpartition(-c, 2500)[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
        I = iu[0][sel]
        J = iu[1][sel]
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
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def quench(x, steps=280, beta0=12.0, beta1=130.0, lr0=0.025, lr1=0.0025, mu=0.72):
    n, d = x.shape
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta0 > 0 else 0.0
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        scale = float(np.sum(W)) / n
        if scale < 1e-12 or not np.isfinite(scale):
            scale = 1.0
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        grad = grad / scale
        rn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad = grad * np.minimum(1.0, 1.0 / rn)
        vel = mu * vel - lr * grad
        vdot = np.sum(vel * x, axis=1, keepdims=True)
        vel = vel - vdot * x
        x = x + vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
    return x


def best_hole(others, rng, n_cand=3000):
    d = others.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = np.max(dots, axis=1)
    return cand[np.argmin(worst)]


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 82, 14
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal((n, d))
    if xb.shape != (n, d):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    X_best = xb.copy()
    # light initial polish if time allows (already polished, keep short)
    if time.time() < DEADLINE - 60:
        X_best = lp_polish(X_best, seconds=2.0, deadline=DEADLINE - 40)
        best_val = max_cos(X_best)

    trial = 0
    while time.time() < DEADLINE - 14:
        trial += 1
        time_left = DEADLINE - time.time()
        if time_left < 6:
            break
        Y = X_best.copy()
        cnt, mcur = contact_counts(Y, tol=2e-3)
        order_loose = np.argsort(cnt)
        ratt0, _ = contact_counts(Y, tol=1e-9)
        rattlers = np.where(ratt0 == 0)[0]
        mode = trial % 4
        if mode == 0:
            # re-insert 2 rattlers / loosest into holes
            k = 2 if len(rattlers) >= 2 else 2
            if len(rattlers) >= 1:
                base = list(rattlers[:k])
                if len(base) < k:
                    for o in order_loose:
                        if o not in base:
                            base.append(int(o))
                            if len(base) >= k:
                                break
            else:
                base = [int(o) for o in order_loose[:k]]
            for idx in base:
                others = np.delete(Y, idx, axis=0)
                Y[idx] = best_hole(others, rng, n_cand=3000)
            # tiny shake of overloaded hubs
            hub = np.argsort(-cnt)[:2]
            Y[hub] += 0.015 * rng.standard_normal((len(hub), d))
        elif mode == 1:
            # single rattler re-insert + small global noise
            if len(rattlers) > 0:
                idx = int(rattlers[trial % len(rattlers)])
            else:
                idx = int(order_loose[0])
            others = np.delete(Y, idx, axis=0)
            Y[idx] = best_hole(others, rng, n_cand=5000)
            Y += 0.008 * rng.standard_normal(Y.shape)
        elif mode == 2:
            # shake overloaded cluster + one loose re-insert
            hub = np.argsort(-cnt)[:4]
            Y[hub] += 0.05 * rng.standard_normal((len(hub), d))
            idx = int(order_loose[0])
            others = np.delete(Y, idx, axis=0)
            Y[idx] = best_hole(others, rng, n_cand=2500)
        else:
            # 3-point hole re-insert (aggressive basin change)
            base = [int(o) for o in order_loose[:3]]
            for idx in base:
                others = np.delete(Y, idx, axis=0)
                Y[idx] = best_hole(others, rng, n_cand=2000)
            Y += 0.005 * rng.standard_normal(Y.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # fast quench exploration (no LP yet)
        steps = 260 if mode != 3 else 320
        Y = quench(Y, steps=steps)
        vq = max_cos(Y)
        # decide to polish: always polish if promising or every few trials
        do_polish = (vq < best_val + 0.004) or (trial % 5 == 0)
        if not do_polish:
            continue
        if time.time() > DEADLINE - 14:
            break
        # short polish
        budget = 1.4 if vq > best_val else 2.5
        budget = min(budget, max(0.6, (DEADLINE - 14 - time.time()) / 2.0))
        if budget < 0.5:
            break
        Yp = lp_polish(Y, seconds=budget, deadline=DEADLINE - 10)
        vp = max_cos(Yp)
        if vp < best_val - 1e-12:
            X_best = Yp
            best_val = vp
        # opportunistic early exit if clearly new record with margin
        # continue hopping regardless to push further
    # final polish with remaining time
    rem = DEADLINE - time.time() - 1.0
    if rem > 2.0:
        X_best = lp_polish(X_best, seconds=min(rem, 12.0), deadline=DEADLINE - 0.5)
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
