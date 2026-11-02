import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def quench_softmax(x, steps, a0, a1, lr0, lr1, rng=None):
    n, d = x.shape
    # geometric schedules
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        alpha = float(np.exp(la0 + (la1 - la0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights, avoid overflow
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        # optional sparsify tiny weights to speed / sharpen
        # keep all for smoothness
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # normalize step by mean norm to make lr predictable
        gn = np.linalg.norm(Gt, axis=1).mean() + 1e-12
        # scale: want displacement ~ lr
        x = x - (lr / gn) * Gt * 2.0
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

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
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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

def solve():
    t0 = time.time()
    TIME_LIMIT = 168.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            z = rng.standard_normal((58, 22))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            return z
    # ensure shape (58,22)
    if xb.shape != (58, 22):
        # try to reshape/crop
        z = rng.standard_normal((58, 22))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        # copy overlap
        m = min(xb.shape[0], 58)
        d = min(xb.shape[1], 22)
        z[:m, :d] = xb[:m, :d]
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        xb = z
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best = xb.copy()

    # also consider record
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < best_val:
            best_val = vr
            best = xr.copy()
    except Exception:
        pass

    n, d = 58, 22
    # fast basin hopping until ~130s, reserve time for LP
    HOP_END = t0 + 130.0
    it = 0
    # presets for quench
    presets = [
        (60, 600, 0.03, 0.004, 1500),
        (100, 900, 0.025, 0.003, 1500),
        (40, 400, 0.04, 0.005, 1200),
        (150, 1500, 0.02, 0.002, 1800),
        (80, 1200, 0.03, 0.002, 2000),
    ]
    while time.time() < HOP_END and time.time() - t0 < TIME_LIMIT - 30:
        it += 1
        a0, a1, lr0, lr1, steps = presets[it % len(presets)]
        # perturb best
        y = best.copy()
        mode = it % 4
        if mode == 0:
            sig = 0.015 + 0.01 * ((it // 4) % 3)
            y = y + sig * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            k = 2 + (it % 6)  # 2..7 points reinserted
            idx = rng.choice(n, size=k, replace=False)
            # global small jitter + reinit subset
            y = y + 0.005 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r
        elif mode == 2:
            # kick worst pair + neighbors
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            # find worst pair
            ij = np.argmax(G)
            i0, j0 = ij // n, ij % n
            # neighbors of i0,j0 (top dots)
            # kick i0,j0 strongly + their top-6 neighbors weakly
            cand = set([int(i0), int(j0)])
            # add random 2-4 extra
            extra = rng.choice(n, size=2 + (it % 3), replace=False)
            for e in extra:
                cand.add(int(e))
            cand = list(cand)
            sig = 0.08 + 0.04 * ((it % 3) == 0)
            y[cand] = y[cand] + sig * rng.standard_normal((len(cand), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # single-point hole reinsertion + tiny jitter
            k = 1 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            y = y + 0.008 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            r = rng.standard_normal((k, d))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            y[idx] = r * 1.0
            # also jitter a few others
            idx2 = rng.choice(n, size=6, replace=False)
            y[idx2] = y[idx2] + 0.02 * rng.standard_normal((6, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # quench
        # adapt steps to remaining time
        if time.time() + 0.5 > HOP_END:
            break
        y = quench_softmax(y, steps, a0, a1, lr0, lr1)
        # extra sharpening tail with high alpha small lr
        y = quench_softmax(y, 300, 1200, 2000, 0.003, 0.001)
        v = max_cos(y)
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
        # opportunistic: also try keeping near-miss with different topology
        # if close, perturb from y instead of best occasionally
        if v < best_val + 0.002 and (it % 17 == 0):
            # short extra kick from y
            y2 = y + 0.01 * rng.standard_normal(y.shape)
            y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
            y2 = quench_softmax(y2, 800, 100, 800, 0.02, 0.003)
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best_val = v2
                best = y2.copy()
        # safety time check
        if it > 300:
            break

    # final sharpen: a bit of softmax descent on best
    try:
        if time.time() - t0 < TIME_LIMIT - 32:
            best = quench_softmax(best, 800, 800, 2500, 0.004, 0.001)
            best_val = max_cos(best)
    except Exception:
        pass

    # final LP polish with remaining budget (cap 25s)
    remain = TIME_LIMIT - (time.time() - t0) - 2.0
    if remain > 3:
        lp_sec = min(remain, 25.0)
        try:
            polished = lp_polish(best, lp_sec)
            vp = max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # guard NaN
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
