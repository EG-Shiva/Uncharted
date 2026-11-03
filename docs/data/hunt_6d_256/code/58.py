import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_cnt(X, gap=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    c = int(np.sum(G >= m - gap) // 2)
    return m, c

def _tangent_bases(X):
    n, d = X.shape
    dd = d - 1
    Q = np.empty((n, d, dd))
    E = np.eye(d)
    amax = np.argmax(np.abs(X), axis=1)
    for i in range(n):
        k = int(amax[i])
        xi = X[i]
        cols = [l for l in range(d) if l != k]
        F = np.empty((d, dd))
        for ci, l in enumerate(cols):
            e = E[l]
            F[:, ci] = e - xi[l] * xi
        Qi, _ = np.linalg.qr(F)
        Q[i] = Qi
    return Q

def _single_refine(p0, Y, K=2500.0, lr=0.02, steps=75):
    x = p0.copy() / np.linalg.norm(p0)
    m = np.zeros(6)
    v = np.zeros(6)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        g = Y @ x
        gmax = float(g.max())
        e = np.exp(K * (g - gmax))
        s = float(e.sum())
        if not np.finite(s) or s <= 0:
            break
        w = e / s
        grad = w @ Y
        d = float(grad @ x)
        gt = grad - d * x
        b1p *= b1; b2p *= b2
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - float(step @ x) * x
        x = x - step
        nr = float(np.linalg.norm(x))
        if nr == 0 or not np.finite(nr):
            break
        x = x / nr
    return x

def _cluster_joint(Xc0, Xf, K=2000.0, lr=0.015, steps=250):
    Xc = Xc0.copy()
    c = Xc.shape[0]
    m = np.zeros_like(Xc)
    v = np.zeros_like(Xc)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        Gcf = Xc @ Xf.T
        Gcc = Xc @ Xc.T
        # max over relevant
        m1 = float(Gcf.max())
        # exclude diag for Gcc
        tmp = Gcc.copy()
        np.fill_diagonal(tmp, -2.0)
        m2 = float(tmp.max())
        gmax = m1 if m1 > m2 else m2
        Wcf = np.exp(K * (Gcf - gmax))
        Wcc = np.exp(K * (Gcc - gmax))
        np.fill_diagonal(Wcc, 0.0)
        S = float(Wcf.sum() + Wcc.sum() * 0.5)
        if not np.finite(S) or S <= 0:
            break
        grad = (Wcf @ Xf + Wcc @ Xc) / S
        dot = np.sum(grad * Xc, axis=1, keepdims=True)
        gt = grad - dot * Xc
        b1p *= b1; b2p *= b2
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * Xc, axis=1, keepdims=True)
        step = step - sdot * Xc
        Xc = Xc - step
        Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc

def _lp_polish_once(X, radius=3e-4, cap=1800, tl=4.0):
    n, d = X.shape
    dd = d - 1
    NV = n * dd
    iuI, iuJ = np.triu_indices(n, 1)
    G = X @ X.T
    c_all = G[iuI, iuJ]
    mval = float(c_all.max())
    sel = c_all > (mval - max(20.0 * radius, 1.2e-4))
    I = iuI[sel]; J = iuJ[sel]; csel = c_all[sel]
    p = len(I)
    if p == 0:
        return X, mval
    if p > cap:
        part = np.argpartition(-csel, cap)[:cap]
        I = I[part]; J = J[part]; csel = csel[part]
        p = cap
    try:
        Q = _tangent_bases(X)
    except Exception:
        return X, mval
    xI = X[I]; xJ = X[J]
    QI = Q[I]; QJ = Q[J]
    AI = np.sum(QI * xJ[:, :, None], axis=1)
    AJ = np.sum(QJ * xI[:, :, None], axis=1)
    rows = np.repeat(np.arange(p), 2 * dd + 1)
    cols = np.empty(p * (2 * dd + 1), dtype=int)
    vals = np.empty(p * (2 * dd + 1), dtype=float)
    ar = np.arange(dd)
    for k in range(p):
        b = k * (2 * dd + 1)
        vals[b:b + dd] = AI[k]
        cols[b:b + dd] = I[k] * dd + ar
        vals[b + dd:b + 2 * dd] = AJ[k]
        cols[b + dd:b + 2 * dd] = J[k] * dd + ar
        vals[b + 2 * dd] = -1.0
        cols[b + 2 * dd] = NV
    try:
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
    except Exception:
        return X, mval
    obj = np.zeros(NV + 1); obj[-1] = 1.0
    bnds = [(-radius, radius)] * NV + [(None, None)]
    try:
        res = linprog(obj, A_ub=Aub, b_ub=-csel, bounds=bnds, method="highs",
                      options={"time_limit": tl, "presolve": True})
    except Exception:
        return X, mval
    if res.status != 0 or res.x is None:
        return X, mval
    try:
        y = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
        if not np.all(np.isfinite(y)):
            return X, mval
        dx = np.einsum("ndk,nk->nd", Q, y)
        Y = X + dx
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y, _max_cos(Y)
    except Exception:
        return X, mval

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LOOP_END = t0 + 155.0
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
                vv = _max_cos(a)
                if vv < best_val:
                    best_val = vv
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        rng0 = np.random.default_rng(0)
        best_x = rng0.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    glob = best_x.copy()
    glob_c0, glob_cnt = _max_cnt(glob, 1e-4)
    sizes = [12, 20, 32]
    nseeds = 26
    for seed in range(nseeds):
        if time.time() > LOOP_END:
            break
        csize = sizes[seed % len(sizes)]
        # attaining pair of current global best
        try:
            Gg = glob @ glob.T
            np.fill_diagonal(Gg, -2.0)
            flat = int(np.argmax(Gg))
            ii = flat // n; jj = flat % n
            if ii == jj:
                iuI, iuJ = np.triu_indices(n, 1)
                k = int(np.argmax(Gg[iuI, iuJ]))
                ii, jj = int(iuI[k]), int(iuJ[k])
        except Exception:
            continue
        # neighbourhood ranking by max(cos to i/j)
        try:
            Gi = Gg[:, ii].copy(); Gj = Gg[:, jj].copy()
            Gi[ii] = -2; Gi[jj] = -2; Gj[ii] = -2; Gj[jj] = -2
            s = np.maximum(Gi, Gj)
            order = np.argsort(-s)
            need = csize - 2
            if need < 0:
                need = 0
            C = [int(ii), int(jj)] + [int(order[t]) for t in range(need)]
            C = np.array(C, dtype=int)
        except Exception:
            continue
        mask = np.ones(n, dtype=bool); mask[C] = False
        frozen_idx = np.where(mask)[0]
        Xf0 = glob[mask].copy()
        orig_del = glob[C].copy()
        rng = np.random.default_rng(1000 + seed * 7919)
        # pool 200k + perturbations
        try:
            P = rng.normal(size=(200000, d))
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            # tangent perturbations of deleted originals
            per = 200
            noise = rng.normal(size=(len(C) * per, d)) * 0.04
            base = np.repeat(orig_del, per, axis=0)
            Qp = base + noise
            Qp /= np.linalg.norm(Qp, axis=1, keepdims=True)
            # second scale
            noise2 = rng.normal(size=(len(C) * 50, d)) * 0.12
            base2 = np.repeat(orig_del, 50, axis=0)
            Qp2 = base2 + noise2
            Qp2 /= np.linalg.norm(Qp2, axis=1, keepdims=True)
            pool = np.vstack([P, Qp, Qp2])
        except Exception:
            continue
        # sequential greedy reinsert
        Yaug = Xf0.copy()
        R = []
        ok = True
        CH = 20000
        for pos in range(len(C)):
            if time.time() > LOOP_END:
                ok = False
                break
            best_pv = 1e9
            best_pi = -1
            # chunked search
            try:
                mnum = Yaug.shape[0]
                # iterate chunks
                for st in range(0, pool.shape[0], CH):
                    ch = pool[st:st + CH]
                    M = ch @ Yaug.T
                    rm = M.max(axis=1)
                    li = int(np.argmin(rm))
                    lv = float(rm[li])
                    if lv < best_pv:
                        best_pv = lv
                        best_pi = st + li
                if best_pi < 0:
                    ok = False
                    break
                p0 = pool[best_pi].copy()
                pr = _single_refine(p0, Yaug, K=2500.0, lr=0.02, steps=75)
                R.append(pr)
                Yaug = np.vstack([Yaug, pr[None, :]])
            except Exception:
                ok = False
                break
        if not ok:
            continue
        R = np.array(R)
        # build trial
        trial = glob.copy()
        trial[C] = R
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        # cluster-only joint Adam
        try:
            Xc0 = trial[C].copy()
            Xf = trial[mask].copy()
            Xc1 = _cluster_joint(Xc0, Xf, K=2000.0, lr=0.015, steps=250)
            trial[C] = Xc1
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        except Exception:
            pass
        # full LP polish (1-2 iters)
        try:
            rem = T_END - time.time()
            if rem > 8.0:
                trial, _ = _lp_polish_once(trial, radius=4e-4, cap=1800, tl=min(4.0, max(1.0, rem - 4.0)))
                # second smaller radius if time
                if time.time() < LOOP_END:
                    trial2, _ = _lp_polish_once(trial, radius=1.2e-4, cap=1800, tl=2.0)
                    # keep better of the two by max cos
                    if _max_cos(trial2) < _max_cos(trial) + 1e-12:
                        trial = trial2
        except Exception:
            pass
        try:
            tc0, tcnt = _max_cnt(trial, 1e-4)
        except Exception:
            continue
        if tc0 < glob_c0 - 1e-12 and tcnt < glob_cnt:
            glob = trial.copy()
            glob_c0, glob_cnt = tc0, tcnt
        # also allow tiny time check
        if time.time() > LOOP_END:
            break
    glob /= np.linalg.norm(glob, axis=1, keepdims=True)
    return np.asarray(glob, dtype=float)
