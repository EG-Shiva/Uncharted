import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def smooth_refine(v, alpha, steps, lr, beta=0.9):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        # anneal temperature from 0.6*alpha to 1.4*alpha
        a = alpha * (0.6 + 0.8 * s / max(steps - 1, 1))
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = a * (G - m)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - cur_lr * step
        nrm = np.linalg.norm(cur, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        cur /= nrm
        if (s % 200 == 199) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def refine_single_point(vec, others, alpha=80.0, steps=250, lr=0.08, beta=0.9):
    cur = vec.copy() / np.linalg.norm(vec)
    mom = np.zeros_like(cur)
    best = cur.copy()
    # true objective for best tracking
    def true_max(c):
        return float((others @ c).max())
    best_s = true_max(cur)
    for s in range(steps):
        dots = others @ cur
        m = dots.max()
        E = alpha * (dots - m)
        W = np.exp(E)
        grad = (W[:, None] * others).sum(axis=0) / (W.sum() + 1e-12)
        gt = grad - float(grad @ cur) * cur
        mom = beta * mom + (1.0 - beta) * gt
        bc = 1.0 - beta ** (s + 1)
        cur = cur - lr * (mom / bc)
        nrm = np.linalg.norm(cur)
        if nrm == 0 or not np.isfinite(nrm):
            cur = best.copy()
            mom[:] = 0
            continue
        cur /= nrm
        if s % 50 == 49 or s == steps - 1:
            sc = true_max(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def deepest_reinsert(base, idx, rng, n_cand=4000):
    x0 = base.copy()
    n, d = base.shape
    for k in list(idx):
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        order = np.argsort(md)
        # take top 4 candidates and locally refine single point
        best_v = None
        best_s = 1e9
        for t in range(min(4, len(order))):
            v0 = cand[order[t]]
            # randomize alpha slightly
            al = float(rng.choice([60.0, 80.0, 110.0]))
            vr, sr = refine_single_point(v0, others, alpha=al, steps=220, lr=0.07)
            if sr < best_s:
                best_s = sr
                best_v = vr
        x0[k] = best_v
    return x0

def far_reinsert_simple(base, idx, rng, n_cand=2000):
    x0 = base.copy()
    for k in list(idx):
        cand = rng.standard_normal(size=(n_cand, base.shape[1]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap problem size for speed
        if p > 3000:
            # keep worst 3000
            idx = np.argsort(c[sel])[-3000:]
            I, J = I[idx], J[idx]
            p = len(I)
            sel_idx = np.where(sel)[0][idx]
            csub = c[sel_idx]
        else:
            csub = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE = 128.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        try:
            if xb.ndim == 2:
                m0 = min(xb.shape[0], 99)
                m1 = min(xb.shape[1], 13)
                tmp[:m0, :m1] = xb[:m0, :m1]
        except Exception:
            pass
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    # pick better of best/record if both exist
    best_x = xb.copy()
    best_val = max_cos(best_x)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        rv = max_cos(xr)
        if rv < best_val:
            best_val = rv
            best_x = xr.copy()
    except Exception:
        pass
    cur_x = best_x.copy()
    cur_val = best_val
    sigmas = [0.012, 0.035, 0.08, 0.15, 0.26, 0.42]
    alphas = [25.0, 40.0, 60.0, 90.0, 130.0, 180.0]
    it = 0
    noimp = 0
    while time.time() - t0 < EXPLORE and it < 500:
        it += 1
        remain = EXPLORE - (time.time() - t0)
        if remain < 2:
            break
        progress = (time.time() - t0) / EXPLORE
        Temp = 4e-4 * (1.0 - progress) + 1e-5
        # choose source: mostly wander state, sometimes best
        if rng.random() < 0.3:
            base = best_x
        else:
            base = cur_x
        mode = rng.random()
        try:
            if mode < 0.30:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                x0 = base + rng.standard_normal(size=(n, d)) * sigma
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                alpha = float(alphas[int(rng.integers(0, len(alphas)))])
                steps = int(rng.integers(400, 1000))
                lr = float(rng.uniform(0.03, 0.12))
            elif mode < 0.55:
                m = int(rng.integers(2, 10))
                idx = rng.choice(n, size=m, replace=False)
                sigma = float(rng.uniform(0.1, 0.5))
                x0 = base.copy()
                x0[idx] += rng.standard_normal(size=(m, d)) * sigma
                x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
                alpha = float(alphas[int(rng.integers(0, len(alphas)))])
                steps = int(rng.integers(500, 1100))
                lr = float(rng.uniform(0.03, 0.12))
            elif mode < 0.85:
                ppm = per_point_max(base)
                q = int(rng.integers(1, 4))
                idx = np.argsort(ppm)[-q:]
                # occasionally add a random rattler to change topology
                if rng.random() < 0.3:
                    r = int(rng.integers(0, n))
                    idx = np.unique(np.append(idx, r))
                if rng.random() < 0.6:
                    x0 = deepest_reinsert(base, idx, rng, n_cand=3000)
                else:
                    x0 = far_reinsert_simple(base, idx, rng, n_cand=2000)
                x0 += rng.standard_normal(size=(n, d)) * 0.008
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                alpha = float(rng.choice([40.0, 60.0, 90.0, 130.0]))
                steps = int(rng.integers(600, 1200))
                lr = float(rng.uniform(0.04, 0.10))
            else:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                thr = G.max() - 1e-3
                cnt = (G > thr).sum(axis=1)
                q = int(rng.integers(3, 9))
                idx = np.argsort(cnt)[-q:]
                x0 = base.copy()
                sigma = float(rng.uniform(0.15, 0.4))
                x0[idx] += rng.standard_normal(size=(len(idx), d)) * sigma
                x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
                alpha = float(alphas[int(rng.integers(0, len(alphas)))])
                steps = int(rng.integers(500, 1000))
                lr = float(rng.uniform(0.03, 0.12))
            cand, sc = smooth_refine(x0, alpha, steps, lr)
        except Exception:
            continue
        # update best
        if sc < best_val - 1e-12:
            best_val = sc
            best_x = cand.copy()
            noimp = 0
            cur_x = cand.copy()
            cur_val = sc
        else:
            noimp += 1
            # wandering acceptance (simulated annealing)
            if sc < cur_val or rng.random() < np.exp((cur_val - sc) / max(Temp, 1e-9)):
                cur_x = cand.copy()
                cur_val = sc
            # periodic restart of wanderer to best if stuck
            if noimp > 40 and rng.random() < 0.2:
                cur_x = best_x.copy()
                cur_val = best_val
                noimp = 0
    # final intensive polish with remaining time
    rem = BUDGET - (time.time() - t0)
    rem = max(6.0, min(rem, 44.0))
    # short high-alpha smoothing before LP
    try:
        tmp, tv = smooth_refine(best_x, 160.0, 800, 0.05)
        if tv < best_val:
            best_x = tmp
            best_val = tv
    except Exception:
        pass
    try:
        polished = lp_polish(best_x, rem - 1.0)
        pv = max_cos(polished)
        if pv <= best_val:
            best_x = polished
            best_val = pv
        elif pv < best_val + 1e-9:
            best_x = polished
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
