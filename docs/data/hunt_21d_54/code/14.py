import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=4e-3, cap=700):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < 60:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        # select worst pairs, cap to most violated
        if c.size > cap:
            # partial selection of top-cap
            part = np.argpartition(c, -cap)[-cap:]
            # also require thresh, but if max is tight keep top-cap anyway
            # filter to thresh to reduce size further
            sel_idx = part[c[part] > thresh]
            if sel_idx.size < 50:
                # ensure at least 50 constraints around max
                sel_idx = part[np.argsort(c[part])[-50:]]
        else:
            sel_idx = np.where(c > thresh)[0]
        if sel_idx.size == 0:
            break
        I = iuI[sel_idx]; J = iuJ[sel_idx]
        p = len(I)
        csel = c[sel_idx]
        nv = n * d + 1
        # build sparse A_ub: row k has x[J] at I-block, x[I] at J-block, -1 at t
        rows = np.repeat(np.arange(p), 2 * d + 1)
        Ccol = np.empty((p, 2 * d + 1), dtype=int)
        Ccol[:, :d] = I[:, None] * d + np.arange(d)
        Ccol[:, d:2 * d] = J[:, None] * d + np.arange(d)
        Ccol[:, 2 * d] = n * d
        cols = Ccol.ravel()
        Arow = np.empty((p, 2 * d + 1))
        Arow[:, :d] = x[J]
        Arow[:, d:2 * d] = x[I]
        Arow[:, 2 * d] = -1.0
        vals = Arow.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        # early stop if time nearly up
        if time.time() - start >= seconds:
            break
    return x

def _gd_repel(x, steps, T0=0.02, T1=0.002, lr0=0.03, lr1=0.003, mu=0.6, rng=None, noise=0.0):
    x = _norm(x.copy())
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        lr = lr0 * (1 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nr = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / nr
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        if noise > 0:
            x = x + noise * np.random.randn(n, d) * 0.01
            # use global rng via np.random? use passed rng
        x = _norm(x)
    return x

def _contacts(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol).sum(axis=1)
    # per-point worst
    np.fill_diagonal(G, -np.inf)
    pw = G.max(axis=1)
    return C, m, pw

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    FINAL_RESERVE = 22.0
    rng = np.random.RandomState(20261101)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(54, 21)
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (54, 21):
        tmp = rng.randn(54, 21)
        # try to salvage if transposed or different
        try:
            xb = _norm(np.asarray(xb, dtype=float).reshape(54, 21))
        except Exception:
            xb = _norm(tmp)
    best = xb.copy()
    best_val = _max_cos(best)
    sigmas = [0.07, 0.03, 0.015, 0.10, 0.025, 0.05, 0.008, 0.15]
    trial = 0
    # precompute iu not needed here
    while time.time() - t0 < DEADLINE - FINAL_RESERVE:
        trial += 1
        remain = DEADLINE - FINAL_RESERVE - (time.time() - t0)
        if remain < 2.0:
            break
        mode = trial % 4
        sigma = sigmas[(trial - 1) % len(sigmas)]
        cand = best.copy()
        if mode == 1:
            # global shake + quench
            cand = cand + sigma * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=420, T0=0.02, T1=0.0025, lr0=0.04, lr1=0.004, mu=0.5)
        elif mode == 2:
            # kill worst point into largest hole
            C, m, pw = _contacts(best)
            idx = int(np.argmax(C + 0.1 * (pw - pw.min())))
            keep = np.delete(best, idx, axis=0)
            S = _norm(rng.randn(2500, 21))
            dots = S @ keep.T
            md = dots.max(axis=1)
            hole = S[int(np.argmin(md))]
            cand[idx] = hole + 0.005 * rng.randn(21)
            cand = _norm(cand)
            # jitter neighbors slightly
            cand = cand + (sigma * 0.2) * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=350, T0=0.012, T1=0.002, lr0=0.025, lr1=0.003, mu=0.5)
        elif mode == 3:
            # subset surgery on overloaded hubs
            C, m, pw = _contacts(best)
            order = np.argsort(-C)
            k = 6 + (trial % 3)  # 6-8 points
            subset = order[:k]
            cand[subset] = cand[subset] + (sigma * 1.5) * rng.randn(k, 21)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=380, T0=0.015, T1=0.002, lr0=0.03, lr1=0.003, mu=0.6)
        else:
            # worst-pair split: push max pair apart
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G.ravel()))
            ii, jj = divmod(ij, 54)
            # push apart along tangent difference
            vi = best[ii] - best[jj]
            vj = -vi
            # remove radial component
            vi = vi - np.dot(vi, best[ii]) * best[ii]
            vj = vj - np.dot(vj, best[jj]) * best[jj]
            ni = np.linalg.norm(vi) + 1e-12
            nj = np.linalg.norm(vj) + 1e-12
            cand[ii] = cand[ii] + (sigma * 1.2) * vi / ni + 0.01 * rng.randn(21)
            cand[jj] = cand[jj] + (sigma * 1.2) * vj / nj + 0.01 * rng.randn(21)
            cand = cand + (sigma * 0.15) * rng.randn(*cand.shape)
            cand = _norm(cand)
            cand = _gd_repel(cand, steps=400, T0=0.018, T1=0.002, lr0=0.035, lr1=0.004, mu=0.5)
        v_pre = _max_cos(cand)
        # filter: skip LP if far worse than best (save time)
        # allow larger tolerance for large sigma (more exploration already quenched)
        if v_pre > best_val + 0.025:
            continue
        budget = min(1.8, remain - 0.2)
        if budget < 0.6:
            break
        r0 = min(0.015, max(0.0025, sigma * 0.35))
        cand_p = _lp_polish(cand, seconds=budget, r0=r0, cap=650)
        v = _max_cos(cand_p)
        if v < best_val - 1e-12:
            best_val = v
            best = cand_p.copy()
        if time.time() - t0 > DEADLINE - FINAL_RESERVE - 0.5:
            break
    remain = DEADLINE - (time.time() - t0)
    if remain > 3:
        best = _lp_polish(best, seconds=max(3.0, remain - 1.0), r0=2e-3, cap=800)
    best = _norm(best)
    return best
