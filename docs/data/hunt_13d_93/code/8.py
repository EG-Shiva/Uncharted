import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _refine(x, alpha, steps, lr, t0=1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps+1):
        tt = t0 + t
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        mg = G.max()
        # stable exp(alpha*(G-mg))
        E = np.exp(alpha * (G - mg))
        np.fill_diagonal(E, 0.0)
        s = E.sum()
        if s == 0 or not np.isfinite(s):
            break
        Wn = E / s
        Grad = Wn @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** tt)
        vhat = v / (1 - b2 ** tt)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # safety clip
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # avoid huge jumps
        clip = 0.01
        too = sn > clip
        if np.any(too):
            step = np.where(too, step / (sn / clip), step)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _greedy_pair_push(x, iters, eps0):
    n, d = x.shape
    best = _max_cos(x)
    eps = eps0
    iu0 = np.triu_indices(n, 1)
    for it in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = np.argmax(G)
        ii = k // n
        jj = k % n
        if ii == jj:
            # fallback
            flat = G[iu0].argmax()
            ii, jj = iu0[0][flat], iu0[1][flat]
        # tangent components
        xi = x[ii].copy()
        xj = x[jj].copy()
        # move xi away from xj, xj away from xi (tangential)
        di = xj - (xj @ xi) * xi
        dj = xi - (xi @ xj) * xj
        ni = np.linalg.norm(di)
        nj = np.linalg.norm(dj)
        if ni < 1e-12 or nj < 1e-12:
            eps *= 0.5
            continue
        di /= ni
        dj /= nj
        y = x.copy()
        y[ii] = xi - eps * di
        y[jj] = xj - eps * dj
        y[ii] /= np.linalg.norm(y[ii])
        y[jj] /= np.linalg.norm(y[jj])
        val = _max_cos(y)
        if val < best:
            x = y
            best = val
            eps *= 1.05
        else:
            eps *= 0.7
            if eps < 1e-12:
                eps = eps0 * 0.1
        if it % 50 == 49:
            eps = min(eps, eps0)
    return x

def _lp_polish(x, seconds, r0=2e-4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-15:
        it += 1
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-7)
        sel = c_all > thresh
        # cap to top 4000 to keep LP fast
        cnt = int(sel.sum())
        if cnt > 4000:
            idx = np.argpartition(-c_all, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 4000
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        # time guard for remaining: break if very little time
        if time.time() - start > seconds - 0.5:
            break
    return x

def solve():
    np.random.seed(0)
    t_start = time.time()
    LIMIT = 170.0
    # load seeds
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (93, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(np.random.randn(93, 13))
        cands[0] /= np.linalg.norm(cands[0], axis=1, keepdims=True)
    # pick best seed
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    # stage 0: quick squeeze of seed with sharp smooth + greedy
    # small budget
    best = _refine(best, alpha=50000.0, steps=800, lr=5e-5, t0=1)
    best = _greedy_pair_push(best, iters=200, eps0=1e-4)
    v = _max_cos(best)
    if v < best_val:
        best_val = v
    else:
        # revert if worsened (refine should not worsen much, keep best known)
        pass

    # basin hopping with annealed softmax
    rng = np.random.RandomState(0)
    # reserve ~55s for final LP
    hop_deadline = t_start + LIMIT - 55.0
    trial = 0
    sigmas = [0.008, 0.005, 0.012, 0.003, 0.007, 0.015, 0.004, 0.006, 0.010, 0.002]
    while time.time() < hop_deadline and trial < 30:
        sig = sigmas[trial % len(sigmas)]
        trial += 1
        y = best + sig * rng.randn(*best.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # annealed schedule: coarse -> fine
        # adapt steps to remaining time
        remain = hop_deadline - time.time()
        if remain < 3:
            break
        y = _refine(y, alpha=4000.0, steps=600, lr=2e-4, t0=trial*1000)
        y = _refine(y, alpha=40000.0, steps=600, lr=8e-5, t0=trial*1000+600)
        y = _refine(y, alpha=300000.0, steps=500, lr=3e-5, t0=trial*1000+1200)
        # tiny greedy to break single pair
        y = _greedy_pair_push(y, iters=100, eps0=5e-5)
        v = _max_cos(y)
        if v < best_val and np.isfinite(v):
            best = y
            best_val = v
        # also try pure small perturbation refinement of best without kick
        if trial % 3 == 0 and time.time() < hop_deadline:
            z = _refine(best.copy(), alpha=500000.0, steps=400, lr=2e-5, t0=9999+trial)
            vz = _max_cos(z)
            if vz < best_val:
                best = z
                best_val = vz

    # final intensive polish on best
    remain = LIMIT - (time.time() - t_start)
    if remain > 8:
        # medium-sharp refine before LP
        use = min(8.0, remain - 6.0)
        # quick refine with large alpha, small lr
        best = _refine(best, alpha=800000.0, steps=600, lr=1.5e-5, t0=50000)
        best_val = _max_cos(best)
        remain = LIMIT - (time.time() - t_start)
    if remain > 5:
        lp_time = remain - 1.0
        lp_time = min(lp_time, 50.0)
        best = _lp_polish(best, seconds=lp_time, r0=1.5e-4)
        best_val = _max_cos(best)
        # second LP pass with even smaller radius to tie exactly
        remain2 = LIMIT - (time.time() - t_start)
        if remain2 > 6:
            best = _lp_polish(best, seconds=remain2-1.0, r0=5e-5)

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # tiny shrink guard + finite check
    best = np.where(np.isfinite(best), best, rng.randn(*best.shape))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
