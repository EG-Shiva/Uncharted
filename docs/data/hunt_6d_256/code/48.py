import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_softmax(x0, steps, K0, K1, lr, deadline):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(float(K0)), np.log(float(K1))
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 20:
            break
        frac = t / float(steps)
        K = float(np.exp(la0 + (la1 - la0) * frac))
        lrt = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = K * (G - gmax)
        np.fill_diagonal(E, -100.0)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * gt
        v *= b2
        v += (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = lrt * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step -= sdot * x
        x -= step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if (t % 100 == 0) or (t == steps):
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _tangent_bases(x):
    n, d = x.shape
    dd = d - 1
    Q = np.empty((n, d, dd))
    amax = np.argmax(np.abs(x), axis=1)
    E = np.eye(d)
    for i in range(n):
        k = int(amax[i])
        xi = x[i]
        cols = [l for l in range(d) if l != k]
        F = np.empty((d, dd))
        for ci, l in enumerate(cols):
            e = E[l]
            F[:, ci] = e - xi[l] * xi
        Qi, _ = np.linalg.qr(F)
        Q[i] = Qi
    return Q

def _lp_minmax(cur, radius, cap, timelimit):
    n, d = cur.shape
    dd = d - 1
    NV = n * dd
    G = cur @ cur.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iuI, iuJ = np.triu_indices(n, 1)
    c_all = G[iuI, iuJ]
    thr = m - max(3e-4, 20.0 * radius)
    sel = c_all > thr
    I = iuI[sel]; J = iuJ[sel]; csel = c_all[sel]
    p = len(I)
    if p < 400:
        k = min(800, len(c_all))
        part = np.argpartition(-c_all, k)[:k]
        I = iuI[part]; J = iuJ[part]; csel = c_all[part]
        p = k
    if p > cap:
        part = np.argpartition(-csel, cap)[:cap]
        I = I[part]; J = J[part]; csel = csel[part]
        p = cap
    try:
        Q = _tangent_bases(cur)
    except Exception:
        return None
    xI = cur[I]
    xJ = cur[J]
    QI = Q[I]
    QJ = Q[J]
    AI = np.sum(QI * xJ[:, :, None], axis=1)
    AJ = np.sum(QJ * xI[:, :, None], axis=1)
    rows = np.repeat(np.arange(p), 2 * dd + 1)
    cols = np.empty(p * (2 * dd + 1), dtype=int)
    vals = np.empty(p * (2 * dd + 1), dtype=float)
    ar = np.arange(dd)
    for k in range(p):
        base = k * (2 * dd + 1)
        vals[base:base + dd] = AI[k]
        cols[base:base + dd] = int(I[k]) * dd + ar
        vals[base + dd:base + 2 * dd] = AJ[k]
        cols[base + dd:base + 2 * dd] = int(J[k]) * dd + ar
        vals[base + 2 * dd] = -1.0
        cols[base + 2 * dd] = NV
    try:
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
    except Exception:
        return None
    obj = np.zeros(NV + 1)
    obj[-1] = 1.0
    bnds = [(-radius, radius)] * NV + [(None, None)]
    try:
        res = linprog(obj, A_ub=Aub, b_ub=-csel, bounds=bnds, method="highs",
                      options={"time_limit": float(timelimit), "presolve": True})
    except Exception:
        return None
    if res is None or getattr(res, "status", 1) != 0 or res.x is None:
        return None
    try:
        y = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
        if not np.all(np.isfinite(y)):
            return None
        dx = np.einsum("ndk,nk->nd", Q, y)
        trial = cur + dx
        nr = np.linalg.norm(trial, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        trial /= nr
        if not np.all(np.isfinite(trial)):
            return None
        return trial
    except Exception:
        return None

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 256, 6
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    # Phase 1: equalizing anneal from incumbent (collective plateau push)
    try:
        rem = HOP_END - time.time()
        if rem > 8.0:
            bx, bv = _adam_softmax(best_x, steps=3500, K0=2500.0, K1=25000.0,
                                   lr=0.004, deadline=min(time.time() + 14.0, HOP_END))
            if bv < best_val:
                best_val = bv
                best_x = bx.copy()
    except Exception:
        pass
    try:
        rem = HOP_END - time.time()
        if rem > 8.0:
            bx, bv = _adam_softmax(best_x, steps=2500, K0=12000.0, K1=4000.0,
                                   lr=0.003, deadline=min(time.time() + 10.0, HOP_END))
            if bv < best_val:
                best_val = bv
                best_x = bx.copy()
    except Exception:
        pass
    # Phase 2: hub-targeted kick-and-reoptimize hopping for topology change
    it = 0
    # a few de novo random basins early to seed diversity
    n_random_seeds = 2
    for rs in range(n_random_seeds):
        if time.time() > HOP_END - 5.0:
            break
        try:
            r0 = rng.normal(size=(n, d))
            r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
            bx, bv = _adam_softmax(r0, steps=1500, K0=3000.0, K1=18000.0,
                                   lr=0.005, deadline=min(time.time() + 6.0, HOP_END))
            if bv < best_val - 1e-12:
                best_val = bv
                best_x = bx.copy()
        except Exception:
            continue
    # main hopping from best
    while time.time() < HOP_END - 2.0:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        try:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            cycle = (it - 1) % 12
            if cycle < 3:
                h = 6
                sigma = 0.016
            elif cycle < 6:
                h = 12
                sigma = 0.008
            elif cycle < 9:
                h = 24
                sigma = 0.0035
            else:
                h = 48
                sigma = 0.0015
            # occasionally kick random rattlers instead of hubs for void refill
            if (it % 7) == 0:
                idx = rng.choice(n, size=h, replace=False)
            else:
                # top-h hubs plus a few random to diversify
                nh = min(h, n)
                idx_top = order[:nh]
                # replace 1/4 with random
                nrand = max(1, nh // 4)
                idx_rand = rng.choice(n, size=nrand, replace=False)
                idx = np.unique(np.concatenate([idx_top[:nh - nrand], idx_rand]))
                h = len(idx)
            start = best_x.copy()
            noise = rng.normal(size=(h, d)) * sigma
            # project noise to tangent for stability
            for kk, ii in enumerate(idx):
                v = noise[kk]
                v = v - np.dot(v, start[ii]) * start[ii]
                start[ii] = start[ii] + v
            start[idx] /= np.linalg.norm(start[idx], axis=1, keepdims=True)
            # short equalizing re-optimization
            K0 = 4000.0 if (it % 2 == 0) else 8000.0
            K1 = 16000.0 if (it % 3 != 0) else 5000.0
            steps = 700 if h <= 12 else 900
            dl = min(time.time() + 3.2, HOP_END)
            bx, bv = _adam_softmax(start, steps=steps, K0=K0, K1=K1, lr=0.0035, deadline=dl)
            if bv < best_val - 1e-12:
                best_val = bv
                best_x = bx.copy()
            # opportunistic void refill: try one greedy reinsert of worst hub
            if (it % 5) == 0 and time.time() < HOP_END - 4.0:
                try:
                    Gw = best_x @ best_x.T
                    np.fill_diagonal(Gw, -2.0)
                    rw = Gw.max(axis=1)
                    wi = int(np.argmax(rw))
                    # sample candidates on sphere, pick most isolated vs others
                    cand = rng.normal(size=(4000, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    rest = np.delete(best_x, wi, axis=0)
                    scores = (cand @ rest.T).max(axis=1)
                    bi = int(np.argmin(scores))
                    trial = best_x.copy()
                    trial[wi] = cand[bi]
                    tv = _max_cos(trial)
                    if tv < best_val:
                        # refine around it
                        bx2, bv2 = _adam_softmax(trial, steps=600, K0=5000.0, K1=15000.0,
                                                 lr=0.003, deadline=min(time.time() + 2.5, HOP_END))
                        if bv2 < best_val:
                            best_val = bv2
                            best_x = bx2.copy()
                        elif tv < best_val:
                            best_val = tv
                            best_x = trial.copy()
                except Exception:
                    pass
        except Exception:
            continue
        if it > 120:
            break
    # Phase 3: tangent-LP min-max polishing with cycling radii
    radii = [4e-4, 2e-4, 1e-4, 5e-5, 2e-4, 1e-4]
    ri = 0
    while time.time() < T_END - 1.5:
        r = radii[ri % len(radii)]
        ri += 1
        rem = T_END - time.time()
        if rem < 2.0:
            break
        try:
            trial = _lp_minmax(best_x, radius=r, cap=2600, timelimit=min(5.0, max(1.0, rem - 0.8)))
        except Exception:
            trial = None
        if trial is None:
            continue
        try:
            tv = _max_cos(trial)
        except Exception:
            continue
        if tv < best_val - 1e-15:
            best_val = tv
            best_x = trial.copy()
        # if LP stalls, do a tiny Adam nudge to change active set
        if ri % 6 == 0 and time.time() < T_END - 6.0:
            try:
                pert = best_x + rng.normal(size=(n, d)) * 2e-5
                pert /= np.linalg.norm(pert, axis=1, keepdims=True)
                bx, bv = _adam_softmax(pert, steps=300, K0=10000.0, K1=10000.0,
                                       lr=0.001, deadline=min(time.time() + 1.5, T_END - 1.0))
                if bv < best_val - 1e-15:
                    best_val = bv
                    best_x = bx.copy()
            except Exception:
                pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return np.asarray(best_x, dtype=float)
