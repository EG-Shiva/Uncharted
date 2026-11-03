import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _max_pair(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    n = X.shape[0]
    return k // n, k % n

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _adam_anneal(X, steps, tau0, tau1, lr0, t0time, deadline):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
    for s in range(steps):
        if time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltaU * frac) if False else tau0 * np.exp(ltau * frac)
        lr = lr0 * (0.35 + 0.65 * (1.0 - frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mg = float(np.max(G))
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        t += 1
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** t)
        vh = v / (1.0 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        clip = 0.012
        too = sn > clip
        if np.any(too):
            step = np.where(too, step / (sn / clip + 1e-12), step)
        X = X - step
        X = _norm(X)
    return X

def _hard_descent(X, iters, t0time, deadline):
    cur = _maxcos(X)
    ss = 0.003
    for _ in range(iters):
        if time.time() > deadline:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        margin = 0.0015
        mask = G > (cmax - margin)
        grad = mask.astype(np.float64) @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        rn = np.linalg.norm(gt, axis=1, keepdims=True)
        nz = (rn[:, 0] > 1e-12)
        if not np.any(nz):
            break
        direc = np.zeros_like(gt)
        direc[nz] = gt[nz] / rn[nz]
        Xt = _norm(X - ss * direc)
        vt = _maxcos(Xt)
        if vt < cmax:
            X = Xt
            cur = vt
            ss = min(ss * 1.2, 0.005)
        else:
            ss *= 0.5
            if ss < 1e-7:
                break
    return X

def _find_void(Xkeep, rng, K=20000):
    n, d = Xkeep.shape
    C = _norm(rng.standard_normal((K, d)))
    # max dot per candidate
    D = C @ Xkeep.T
    m = D.max(axis=1)
    bi = int(np.argmin(m))
    return C[bi]

def _lp_polish(X, seconds, t0time, deadline, r0=4e-4):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    radius = r0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            idx = np.argpartition(-c_all, 3500)[:3500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = 3500
        I = iu0[sel]; J = iu1[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cb = np.empty((p, 2 * d + 1), dtype=np.int64)
        cb[:, :d] = I[:, None] * d + np.arange(d)
        cb[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cb[:, 2 * d] = n * d
        cols = cb.ravel()
        vals2 = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals2[:, :d] = X[J]
        vals2[:, d:2 * d] = X[I]
        vals2[:, 2 * d] = -1.0
        vals = vals2.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start > seconds - 0.4:
            break
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 93, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (93, 13):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(best)

    def local_run(X):
        dl = DEADLINE - 0.5
        X = _adam_anneal(X, 450, 0.02, 0.0025, 0.004, t0, dl)
        X = _adam_anneal(X, 350, 0.0025, 0.0004, 0.002, t0, dl)
        X = _hard_descent(X, 80, t0, dl)
        return X

    # initial squeeze of seed lightly (don't waste time)
    if time.time() < HOP_END:
        tmp = _adam_anneal(best.copy(), 250, 0.008, 0.001, 0.003, t0, HOP_END)
        tmp = _hard_descent(tmp, 40, t0, HOP_END)
        v = _maxcos(tmp)
        if v < best_val:
            best, best_val = tmp, v

    trial = 0
    # precompute sigma cycle
    while time.time() < HOP_END and trial < 300:
        trial += 1
        mode = trial % 7
        y = best.copy()
        if mode == 0:
            sig = 0.02 if (trial % 2 == 0) else 0.035
            y = y + sig * rng.standard_normal(y.shape)
            y = _norm(y)
        elif mode == 1:
            # rattler shake: big kick to loose points
            rm, _ = _row_max(y)
            gmax = float(rm.max())
            loose = rm < (gmax - 0.001)
            noise = rng.standard_normal(y.shape)
            scale = np.where(loose[:, None], 0.09, 0.006)
            # decay loose scale with trial to fine-tune later
            y = y + scale * noise
            y = _norm(y)
        elif mode == 2:
            # hole reinsertion of worst point
            rm, Gfull = _row_max(y)
            wi = int(np.argmax(rm))
            keep = np.delete(y, wi, axis=0)
            hole = _find_void(keep, rng, K=15000)
            # small jitter around hole
            hole = hole + 0.005 * rng.standard_normal(d)
            hole = hole / np.linalg.norm(hole)
            y[wi] = hole
            # jitter neighbours slightly
            y = y + 0.004 * rng.standard_normal(y.shape)
            y = _norm(y)
        elif mode == 3:
            sig = 0.06
            y = y + sig * rng.standard_normal(y.shape)
            y = _norm(y)
        elif mode == 4:
            # pair split: push max pair apart
            ii, jj = _max_pair(y)
            xi = y[ii].copy(); xj = y[jj].copy()
            di = xj - (xj @ xi) * xi
            dj = xi - (xi @ xj) * xj
            ni = np.linalg.norm(di); nj = np.linalg.norm(dj)
            eps = 0.03 + 0.02 * rng.random()
            if ni > 1e-12:
                di /= ni
                y[ii] = xi - eps * di
            if nj > 1e-12:
                dj /= nj
                y[jj] = xj - eps * dj
            y = y + 0.006 * rng.standard_normal(y.shape)
            y = _norm(y)
        elif mode == 5:
            # double hole reinsertion
            rm, _ = _row_max(y)
            order = np.argsort(-rm)
            for k in range(2):
                wi = int(order[k])
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[wi] = True
                # approximate keep without updated points: use current y
                keep = y[keep_mask]
                hole = _find_void(keep, rng, K=12000)
                y[wi] = hole
            y = y + 0.003 * rng.standard_normal(y.shape)
            y = _norm(y)
        else:
            sig = 0.12
            y = y + sig * rng.standard_normal(y.shape)
            y = _norm(y)

        y = local_run(y)
        if time.time() > HOP_END:
            break
        v = _maxcos(y)
        if np.isfinite(v) and v < best_val:
            best = y
            best_val = v

    # final sharpen + LP
    remain = DEADLINE - time.time()
    if remain > 8:
        # sharp anneal on best
        best = _adam_anneal(best, 400, 0.0015, 0.00025, 0.0015, t0, DEADLINE - 30)
        best = _hard_descent(best, 80, t0, DEADLINE - 28)
        best_val = _maxcos(best)
    remain = DEADLINE - time.time()
    if remain > 6:
        # allocate most to LP
        lp_t = remain - 1.5
        lp_t = min(lp_t, 32.0)
        best = _lp_polish(best, lp_t, t0, DEADLINE - 1.0, r0=3e-4)
        best_val = _maxcos(best)
        r2 = DEADLINE - time.time() - 1.0
        if r2 > 5:
            best = _lp_polish(best, r2, t0, DEADLINE - 0.5, r0=6e-5)
    best = _norm(best)
    return best
