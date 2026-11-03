import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, t0=None, limit=1e30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() - t0 > limit:
            break
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq,
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


def _smooth_stage(x, alpha, steps, lr, mu, t0, deadline):
    n, d = x.shape
    v = np.zeros_like(x)
    # linear decay of lr
    for s in range(steps):
        if (s & 63) == 0 and time.time() > deadline:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-12)
        Grad = W @ x
        Gav = Grad / rs
        dn = np.sum(Gav * x, axis=1, keepdims=True)
        Gt = Gav - dn * x
        cur_lr = lr * (1.0 - 0.7 * s / max(steps, 1))
        v = mu * v - cur_lr * Gt
        # clip velocity to avoid blow up
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        cap = 0.1
        over = vn > cap
        if np.any(over):
            v = np.where(over, v * (cap / np.maximum(vn, 1e-12)), v)
        x = x + v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _reinsert_hole(x, rng):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    pt_max = G.max(axis=1)
    order = np.argsort(pt_max)  # ascending
    # pick among top-5 deterministically via rng
    top5 = order[-5:]
    victim = int(top5[rng.randint(0, 5)])
    Xo = np.delete(x, victim, axis=0)  # (n-1,d)
    cand = rng.standard_normal((5000, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = cand @ Xo.T
    mscores = scores.max(axis=1)
    best_idx = np.argsort(mscores)[:5]
    best_c = None
    best_m = 1e9
    for b in best_idx:
        c = cand[int(b)].copy()
        for it in range(300):
            dots = Xo @ c
            mm = float(dots.max())
            w = np.exp(150.0 * (dots - mm))
            sw = w.sum()
            g = (w @ Xo) / max(sw, 1e-12)
            gt = g - float(np.dot(g, c)) * c
            c = c - 0.02 * gt
            nc = np.linalg.norm(c)
            if nc < 1e-12:
                c = cand[int(b)]
                break
            c /= nc
        mm = float((Xo @ c).max())
        if mm < best_m:
            best_m = mm
            best_c = c.copy()
    x_new = x.copy()
    x_new[victim] = best_c
    return x_new


def solve():
    t0 = time.time()
    HARD = 172.0  # must return before 180
    rng = np.random.RandomState(12345)
    n, d = 99, 13
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        # try to reshape / fallback
        tmp = rng.standard_normal((n, d))
        m = min(xb.size, n * d)
        tmp.ravel()[:m] = xb.ravel()[:m]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    # guard NaN
    xb = np.where(np.isfinite(xb), xb, 0.0)
    nn = np.linalg.norm(xb, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    xb = xb / nn
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass

    best_x = xb.copy()
    best_v = _max_cos(best_x)

    sigmas = [0.008, 0.015, 0.03, 0.06, 0.12, 0.02, 0.045]
    stages = [(60.0, 500, 0.030, 0.80), (150.0, 500, 0.015, 0.85), (350.0, 350, 0.008, 0.90)]

    trial = 0
    # exploration budget ~ up to 150s, reserve rest for final polish
    while time.time() - t0 < 148.0:
        trial += 1
        if trial % 4 == 0:
            x_try = _reinsert_hole(best_x, rng)
            # small shake after reinsert to break symmetry
            x_try = x_try + 0.005 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            sig = sigmas[(trial - 1) % len(sigmas)]
            x_try = best_x + sig * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        deadline = min(t0 + 148.0, t0 + HARD - 8.0)
        for (alpha, steps, lr, mu) in stages:
            if time.time() > deadline:
                break
            x_try = _smooth_stage(x_try, alpha, steps, lr, mu, t0, deadline)
        v_smooth = _max_cos(x_try)
        # only LP-polish if promising (within 0.015 of best)
        if v_smooth < best_v + 0.015:
            remain = min(3.0, max(0.5, (t0 + 150.0 - time.time()) * 0.15))
            if time.time() - t0 > 145.0:
                remain = 0.8
            x_pol, v_pol = _lp_polish(x_try, seconds=remain, t0=t0, limit=150.0)
            if v_pol < best_v:
                best_v = float(v_pol)
                best_x = x_pol.copy()
                # early exit if big win
                if best_v < 0.2435:
                    break
        else:
            # quick check if smooth alone already beats (rare)
            if v_smooth < best_v:
                x_pol, v_pol = _lp_polish(x_try, seconds=1.5, t0=t0, limit=150.0)
                if v_pol < best_v:
                    best_v = float(v_pol)
                    best_x = x_pol.copy()
        if time.time() - t0 > 150.0:
            break
        if trial > 400:
            break

    # final polish of best with remaining time
    rem = HARD - (time.time() - t0) - 1.0
    if rem > 0.5:
        rem = min(rem, 20.0)
        best_x, best_v = _lp_polish(best_x, seconds=rem)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
