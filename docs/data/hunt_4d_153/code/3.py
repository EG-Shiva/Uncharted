import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def random_sphere(n, d, rng):
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def smooth_adam(X, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    for t in range(1, steps + 1):
        if t % 400 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (arat ** frac)
        lr = lr0 * (lrat ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = np.max(G)
        # normalized weights, max=1
        W = np.exp(alpha * (G - m0))
        np.fill_diagonal(W, 0.0)
        grad_raw = W @ X
        dot = np.sum(grad_raw * X, axis=1, keepdims=True)
        grad_t = grad_raw - dot * X
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        b1t = 1.0 - b1 ** t
        b2t = 1.0 - b2 ** t
        mhat = m / b1t
        vhat = v / b2t
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step -= sdot * X
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.where(nrm > 0.02, 0.02 / (nrm + 1e-12), 1.0)
        step *= scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    # precompute iu arrays
    I0 = iu[0]
    J0 = iu[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        # cap constraints for speed
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            # keep top 3000
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 3000
        I = I0[sel]
        J = J0[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = cols_I[0::d]
        # build cols vectorized simpler:
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        vv = max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            # also accept sideways move of x if it improved linearized but not true? keep x
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x


def contact_counts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - 1e-9)
    return B.sum(axis=1), m


def solve():
    t0 = time.time()
    DEADLINE = 178.0  # leave margin for 180s limit
    rng = np.random.default_rng(12345)
    # load starts
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
    if len(cands) == 0:
        cands.append(random_sphere(153, 4, rng))
    bestX = None
    bestV = 1e30
    for c in cands:
        v = max_cos(c)
        if v < bestV:
            bestV = v
            bestX = c.copy()
    n, d = bestX.shape
    # quick initial squeeze if time allows (short)
    # exploration budget until 150s
    EXPLORE_END = t0 + 150.0
    trial = 0
    # precompute contacts for biased kill
    cc, _ = contact_counts(bestX)
    order_hi = np.argsort(-cc)
    while time.time() < EXPLORE_END and time.time() - t0 < DEADLINE - 25:
        trial += 1
        mode = trial % 5
        Xtry = None
        if mode == 0:
            sigma = 0.008
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=1800, a0=50, a1=320, lr0=0.004, lr1=0.0004, t_deadline=EXPLORE_END)
        elif mode == 1:
            sigma = 0.030
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2000, a0=40, a1=280, lr0=0.005, lr1=0.0005, t_deadline=EXPLORE_END)
        elif mode == 2:
            sigma = 0.10
            Xs = bestX + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2200, a0=30, a1=250, lr0=0.006, lr1=0.0006, t_deadline=EXPLORE_END)
        elif mode == 3:
            # kill-and-refill jammed points
            k = int(rng.integers(2, 6))
            # bias to high contacts 50%, random 50%
            idx = set()
            nh = k // 2 + 1
            # pick from top 30 overloaded
            top = order_hi[:30]
            perm = rng.permutation(top)[:nh]
            for a in perm:
                idx.add(int(a))
            while len(idx) < k:
                idx.add(int(rng.integers(0, n)))
            idx = np.array(sorted(idx), dtype=int)
            Xs = bestX.copy()
            Xs[idx] = random_sphere(len(idx), d, rng)
            # small shake rest
            Xs += 0.012 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xtry = smooth_adam(Xs, steps=2200, a0=40, a1=300, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
        else:
            if trial % 20 == 0:
                # full random restart
                Xs = random_sphere(n, d, rng)
                Xtry = smooth_adam(Xs, steps=3000, a0=30, a1=280, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
            else:
                sigma = 0.015
                Xs = bestX + sigma * rng.standard_normal((n, d))
                Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
                Xtry = smooth_adam(Xs, steps=1600, a0=60, a1=350, lr0=0.003, lr1=0.0003, t_deadline=EXPLORE_END)
        if Xtry is None:
            continue
        v = max_cos(Xtry)
        # decide to LP-polish
        need_lp = False
        lp_time = 1.0
        if v < bestV - 1e-12:
            need_lp = True
            lp_time = 2.0
        elif v < bestV + 0.0025:
            # promising: polish occasionally to save time
            if mode in (0, 3, 4) or (trial % 2 == 0):
                need_lp = True
                lp_time = 1.0
        if need_lp:
            # cap by remaining explore time
            remain = EXPLORE_END - time.time()
            if remain < 0.5:
                break
            lp_time = min(lp_time, remain - 0.1)
            if lp_time < 0.3:
                # skip LP, just check
                if v < bestV:
                    bestV = v
                    bestX = Xtry.copy()
                    cc, _ = contact_counts(bestX)
                    order_hi = np.argsort(-cc)
                continue
            Xp = lp_polish(Xtry, lp_time)
            vp = max_cos(Xp)
            if vp < bestV:
                bestV = vp
                bestX = Xp.copy()
                cc, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
            elif v < bestV:
                bestV = v
                bestX = Xtry.copy()
                cc, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
        else:
            if v < bestV:
                bestV = v
                bestX = Xtry.copy()
                cc, _ = contact_counts(bestX)
                order_hi = np.argsort(-cc)
    # final thorough polish with remaining time
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 1.0:
        bestX = lp_polish(bestX, min(remain, 25.0))
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
