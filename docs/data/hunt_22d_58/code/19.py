import time
import numpy as np

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def softmax_quench(x, steps, beta0, beta1, s0, s1):
    n, d = x.shape
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ls0 = np.log(s0); ls1 = np.log(s1)
    for t in range(steps):
        f = t / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * f)
        s = np.exp(ls0 + (ls1 - ls0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        # weights exp(beta*(G-m)), in (0,1]
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        tot = np.sum(W)
        if tot == 0 or not np.isfinite(tot):
            continue
        W /= tot
        Gr = W @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        rms = np.sqrt(np.mean(Gt * Gt))
        if rms < 1e-14 or not np.isfinite(rms):
            continue
        x = x - (s / rms) * Gt
        nr = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def greedy_refine(x, steps=350, step0=0.004, step1=0.0002, tol0=0.005, tol1=0.0004):
    l0 = np.log(step0); l1 = np.log(step1)
    lt0 = np.log(tol0); lt1 = np.log(tol1)
    for t in range(steps):
        f = t / (steps - 1) if steps > 1 else 1.0
        step = np.exp(l0 + (l1 - l0) * f)
        tol = np.exp(lt0 + (lt1 - lt0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        Wb = G > (m - tol)
        Wf = Wb.astype(np.float64)
        np.fill_diagonal(Wf, 0.0)
        if np.sum(Wf) == 0:
            continue
        Gr = Wf @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        rn = np.sqrt(np.sum(Gt * Gt, axis=1, keepdims=True))
        mx = np.max(rn)
        if mx < 1e-12 or not np.isfinite(mx):
            break
        x = x - step * Gt / mx
        nr = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def lp_polish(x, seconds):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
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

def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(1)
    np.random.seed(1)
    # load start
    x0 = None
    for fn in ["best.npy", "record.npy"]:
        try:
            x0 = np.load(fn).astype(np.float64)
            if x0.shape == (58, 22):
                break
        except Exception:
            x0 = None
    if x0 is None or x0.shape != (58, 22):
        r = rng.standard_normal((58, 22))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        x0 = r
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best_val = max_cos(x0)
    x_keep = x0.copy()
    keep_val = best_val
    # candidate pool: keep top few quenched
    pool = [(keep_val, x0.copy())]
    trial = 0
    EXPL_END = t0 + 108.0
    # precompute degrees of x_keep for targeted moves (update occasionally)
    while time.time() < EXPL_END and time.time() - t0 < HARD - 50:
        trial += 1
        pat = trial % 6
        n, d = x_keep.shape
        x = x_keep.copy()
        if pat == 0:
            sig = 0.022
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 1600, 60.0, 2500.0, 0.02, 0.0003
        elif pat == 1:
            sig = 0.06
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 2000, 50.0, 3500.0, 0.03, 0.0003
        elif pat == 2:
            sig = 0.13
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 2400, 40.0, 4000.0, 0.04, 0.0004
        elif pat == 3:
            # worst-point reinsert
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            permax = np.max(G, axis=1)
            k = int(rng.integers(2, 5))
            idx = np.argsort(permax)[-k:]
            rnew = rng.standard_normal((k, d))
            rnew /= np.linalg.norm(rnew, axis=1, keepdims=True)
            x[idx] = rnew
            x = x + 0.012 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 2400, 50.0, 4000.0, 0.035, 0.0003
        elif pat == 4:
            k = int(rng.integers(4, 9))
            idx = rng.choice(n, k, replace=False)
            x[idx] = x[idx] + 0.35 * rng.standard_normal((k, d))
            x = x + 0.008 * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 2200, 55.0, 3500.0, 0.03, 0.0003
        else:
            sig = 0.09
            x = x + sig * rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, b0, b1, s0, s1 = 2000, 80.0, 3000.0, 0.025, 0.00025
        # quench
        try:
            xq = softmax_quench(x, steps, b0, b1, s0, s1)
            if not np.all(np.isfinite(xq)):
                continue
            xq = greedy_refine(xq, steps=320, step0=0.004, step1=0.0002)
        except Exception:
            continue
        if not np.all(np.isfinite(xq)):
            continue
        v = max_cos(xq)
        if not np.isfinite(v):
            continue
        # maintain pool of best 4
        pool.append((v, xq.copy()))
        pool = sorted(pool, key=lambda z: z[0])[:4]
        if v < keep_val - 1e-12:
            keep_val = v
            x_keep = xq.copy()
            # early exit if clearly record beating by margin? keep exploring to improve further
        # adapt: if time low, break
        if trial > 600:
            break
    # final selection + LP polish within remaining budget
    pool = sorted(pool, key=lambda z: z[0])[:3]
    # ensure x_keep included
    best_x = pool[0][1].copy()
    best_v = pool[0][0]
    remain = HARD - (time.time() - t0)
    # short polish each candidate
    short_sec = 4.0
    polished = []
    for (v, xc) in pool:
        if time.time() - t0 > HARD - 30:
            polished.append((v, xc))
            continue
        try:
            xp = lp_polish(xc.copy(), min(short_sec, max(1.0, HARD - 32 - (time.time() - t0))))
            vp = max_cos(xp)
            polished.append((vp, xp))
        except Exception:
            polished.append((v, xc))
    if len(polished) > 0:
        polished = sorted(polished, key=lambda z: z[0])
        best_v, best_x = polished[0][0], polished[0][1].copy()
    # long final polish with leftover
    leftover = HARD - (time.time() - t0) - 2.0
    if leftover > 3:
        try:
            best_x = lp_polish(best_x.copy(), min(leftover, 45.0))
            best_v = max_cos(best_x)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
