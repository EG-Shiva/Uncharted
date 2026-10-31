import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _counts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    c9 = int(((G > m - 1e-9).sum()) // 2)
    c4 = int(((G > m - 1e-4).sum()) // 2)
    c2 = int(((G > 0.8965).sum()) // 2)
    return m, c9, c4, c2

def lp_polish(x, seconds):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def riemann_adam_hinge(x0, t, tau, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
    x = _norm(x0.copy())
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    for s in range(1, steps + 1):
        G = x @ x.T
        # upper computation including diag then zero out
        U = (G - t) / tau
        np.clip(U, -50.0, 50.0, out=U)
        # softplus stable: logaddexp(0,U)
        SP = np.logaddexp(0.0, U)
        H = tau * SP
        # kill diagonal (self pairs c=1 must not contribute)
        np.fill_diagonal(H, 0.0)
        np.fill_diagonal(U, -50.0)
        Sig = 1.0 / (1.0 + np.exp(-U))
        np.fill_diagonal(Sig, 0.0)
        W = 2.0 * H * Sig
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # Adam moments on tangent grad
        m1 = b1 * m1 + (1.0 - b1) * gt
        m2 = b2 * m2 + (1.0 - b2) * (gt * gt)
        b1p *= b1
        b2p *= b2
        mhat = m1 / (1.0 - b1p)
        vhat = m2 / (1.0 - b2p)
        upd = mhat / (np.sqrt(vhat) + eps)
        # project update to tangent
        d = np.sum(upd * x, axis=1, keepdims=True)
        updt = upd - d * x
        # clip global norm to avoid blowup
        un = np.linalg.norm(updt) + 1e-12
        maxn = 10.0 * np.sqrt(float(n))
        if un > maxn:
            updt *= (maxn / un)
        # decay schedule
        lr_s = lr * (1.0 - 0.5 * (s - 1) / max(steps - 1, 1))
        x = x - lr_s * updt
        x = _norm(x)
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(272, 4))
    xb = np.asarray(xb, dtype=np.float64)
    xb = _norm(xb)
    if xb.shape != (272, 4):
        n0 = xb.shape[0] if xb.ndim == 2 else 0
        if xb.ndim != 2 or xb.shape[1] != 4:
            xb = _norm(rng.standard_normal(size=(272, 4)))
        elif n0 > 272:
            xb = _norm(xb[:272])
        else:
            extra = _norm(rng.standard_normal(size=(272 - n0, 4)))
            xb = np.vstack([xb, extra])
            xb = _norm(xb)
    best = xb.copy()
    best_val = _max_cos(best)
    m0, c9, c4, c2 = _counts(best)
    sparse_best = best.copy()
    sparse_val = best_val
    sparse_c4 = c4
    sparse_c9 = c9
    thresholds = [0.8980, 0.8975, 0.8970, 0.8965]
    taus = [1.0e-4, 1.2e-4, 0.8e-4, 1.5e-4]
    sigmas = [0.006, 0.012, 0.004, 0.020, 0.009]
    it = 0
    # exploration, reserve ~30s for final LP
    while time.time() - t0 < TIME_LIMIT - 32.0:
        it += 1
        k = (it - 1) % len(thresholds)
        t_thr = thresholds[k]
        tau = taus[(it - 1) % len(taus)]
        # choose base: alternate best and sparse_best
        if it % 3 == 0 and sparse_val < 0.8985:
            base = sparse_best
        else:
            base = best
        sigma = sigmas[(it - 1) % len(sigmas)]
        if it % 17 == 0:
            sigma = 0.035
        x0 = base + sigma * rng.standard_normal(size=base.shape)
        x0 = _norm(x0)
        # surrogate stage: Riemannian Adam on hinge^2
        steps = 500
        # smaller steps for lower thresholds to save time
        if k >= 2:
            steps = 400
        lr = 0.0025 if sigma < 0.015 else 0.0020
        try:
            x1 = riemann_adam_hinge(x0, t_thr, tau, steps, lr)
        except Exception:
            continue
        v1 = _max_cos(x1)
        # histogram after stage
        try:
            _, n9, n4, _ = _counts(x1)
        except Exception:
            continue
        improved_sparse = False
        # retain if sparser near-contact set while max below 0.8985
        if v1 < 0.8985 and n4 < sparse_c4:
            sparse_best = x1.copy()
            sparse_val = v1
            sparse_c4 = n4
            sparse_c9 = n9
            improved_sparse = True
        elif v1 < 0.8985 and n4 == sparse_c4 and v1 < sparse_val:
            sparse_best = x1.copy()
            sparse_val = v1
            improved_sparse = True
        # direct best improvement (even without LP)
        if v1 < best_val:
            best = x1.copy()
            best_val = v1
            # also update sparse tracking
            if v1 < 0.8985 and n4 <= sparse_c4:
                sparse_best = x1.copy()
                sparse_val = v1
                sparse_c4 = n4
                sparse_c9 = n9
            # quick LP lock-in on clear win
            if time.time() - t0 < TIME_LIMIT - 55.0:
                try:
                    y = lp_polish(best.copy(), 2.0)
                    vy = _max_cos(y)
                    if vy < best_val:
                        best = y
                        best_val = vy
                except Exception:
                    pass
            continue
        # alternate each surrogate stage with LP polish if sparse improved
        # or periodically to try to drop max from sparser graph
        do_lp = False
        if improved_sparse:
            do_lp = True
        elif it % 4 == 0 and v1 < 0.8985:
            do_lp = True
        if do_lp:
            if time.time() - t0 > TIME_LIMIT - 38.0:
                continue
            try:
                y = lp_polish(x1.copy(), 2.5)
                vy = _max_cos(y)
                if vy < best_val:
                    best = y
                    best_val = vy
                    _, bn9, bn4, _ = _counts(best)
                    if vy < 0.8985 and bn4 <= sparse_c4:
                        sparse_best = best.copy()
                        sparse_val = vy
                        sparse_c4 = bn4
                        sparse_c9 = bn9
                elif vy < 0.8985:
                    _, yn9, yn4, _ = _counts(y)
                    if yn4 < sparse_c4:
                        sparse_best = y.copy()
                        sparse_val = vy
                        sparse_c4 = yn4
                        sparse_c9 = yn9
            except Exception:
                pass
        if time.time() - t0 > TIME_LIMIT - 32.0:
            break
    remain = TIME_LIMIT - (time.time() - t0) - 1.0
    if remain > 5.0:
        try:
            lp_time = min(remain, 28.0)
            y = lp_polish(best.copy(), lp_time)
            vy = _max_cos(y)
            if vy < best_val:
                best = y
                best_val = vy
            # also try polishing sparse_best if different and compare
            if sparse_c4 < 900:
                rt = remain - lp_time
                if rt is not None and time.time() - t0 < TIME_LIMIT - 4.0:
                    try:
                        r2 = min(max(TIME_LIMIT - (time.time() - t0) - 1.0, 0.0), 6.0)
                        if r2 > 1.5:
                            y2 = lp_polish(sparse_best.copy(), r2)
                            v2 = _max_cos(y2)
                            if v2 < best_val:
                                best = y2
                                best_val = v2
                    except Exception:
                        pass
        except Exception:
            pass
    best = _norm(best)
    best = np.nan_to_num(best, nan=1.0, posinf=1.0, neginf=-1.0)
    best = _norm(best)
    return best
