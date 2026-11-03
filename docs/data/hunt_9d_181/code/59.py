import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, budget, init_r=8e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_r
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    for _it in range(14):
        if time.time() - t0 >= budget:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c[sel]
        if cnt > 1200:
            idx = np.argpartition(-csel, 1200)[:1200]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            cnt = 1200
        nv = n * d + 1
        # build sparse A_ub
        cols_mat = np.empty((cnt, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(cnt), 2 * d + 1)
        cols = cols_mat.ravel()
        vals = np.empty((cnt, 2 * d + 1))
        vals[:, :d] = x[J]
        vals[:, d:2 * d] = x[I]
        vals[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals.ravel(), (rows, cols)), shape=(cnt, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.04)
        else:
            radius *= 0.5
    return x

def _smooth(X, steps, k0, k1, lr0, lr1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp(np.clip(k * (G - gm), -50, 0.0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        nrm = float(np.linalg.norm(Gt, axis=1).max())
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        X = X - (lr / nrm) * Gt
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _hole(K, Ddone, rng, nt=2500):
    d = K.shape[1]
    C = rng.standard_normal((nt, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ K.T
    mx = S.max(axis=1)
    if Ddone is not None and len(Ddone) > 0:
        S2 = C @ Ddone.T
        mx2 = S2.max(axis=1)
        mx = np.maximum(mx, mx2)
    return C[int(np.argmin(mx))]

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(42)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        tmp = rng.standard_normal((181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [_max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = scores[bi]
    n, d = xb.shape
    # quick tighten of starting point if time allows
    if BUDGET - (time.time() - t0) > 15:
        xb = _lp_polish(xb, budget=2.0, init_r=1.5e-3)
        bb = _max_cos(xb)
    xc = xb.copy(); bc = bb
    T = 2.0e-05
    sizes = [10, 16, 22, 14]
    it = 0
    iu_all = np.triu_indices(n, 1)
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 5.0:
            break
        if it > 120:
            break
        it += 1
        y = xc.copy()
        mode = it % 4
        if mode in (0, 1, 2):
            # --- collective surgery ---
            kdel = sizes[it % len(sizes)]
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            np.fill_diagonal(G, 1.0)
            cnt = np.sum(G > m - 1e-7, axis=1) - 1
            ratt = np.where(cnt <= 0)[0].tolist()
            order_hi = np.argsort(-cnt)
            delset = []
            for r in ratt:
                if len(delset) < kdel:
                    delset.append(int(r))
            nhub = min(max(kdel // 3, 3), n)
            for idx in order_hi:
                if len(delset) >= kdel:
                    break
                ii = int(idx)
                if ii not in delset:
                    # take hubs first, then fill randomly
                    if cnt[ii] >= np.median(cnt) or len(delset) >= kdel * 2 // 3:
                        delset.append(ii)
            while len(delset) < kdel:
                r = int(rng.integers(0, n))
                if r not in delset:
                    delset.append(r)
            delset = np.array(delset, dtype=int)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[delset] = False
            K = y[keep_mask].copy()
            # relax remainder briefly to open holes (collective move)
            K = _smooth(K, steps=40, k0=50.0, k1=150.0, lr0=0.012, lr1=0.004)
            # greedy deepest-hole reinsertion
            D = np.empty((0, d))
            for j in range(kdel):
                h = _hole(K, D if len(D) > 0 else None, rng, nt=2500)
                # small jitter to break symmetry
                h = h + rng.standard_normal(d) * 0.002
                h /= np.linalg.norm(h)
                D = np.vstack([D, h[None, :]]) if len(D) > 0 else h[None, :]
            y[delset] = D
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # joint anneal: soft then sharp
            y = _smooth(y, steps=160, k0=45.0, k1=160.0, lr0=0.016, lr1=0.004)
            if it % 2 == 0:
                y = _smooth(y, steps=80, k0=160.0, k1=320.0, lr0=0.004, lr1=0.0015)
            init_r = 0.012
            lp_bud = 1.6
        else:
            # --- global shake for diversity ---
            sig = 0.05 if (it % 8 == 7) else 0.025
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth(y, steps=150, k0=60.0, k1=200.0, lr0=0.014, lr1=0.003)
            init_r = 0.012
            lp_bud = 1.4
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 3.5:
            break
        lp_bud = min(lp_bud, rem2 - 1.5)
        if lp_bud < 0.6:
            lp_bud = max(0.6, rem2 - 0.8)
        y2 = _lp_polish(y, budget=float(lp_bud), init_r=init_r)
        v = _max_cos(y2)
        if v < bb - 1e-15:
            xb = y2.copy(); bb = v
            xc = y2.copy(); bc = v
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-9)):
                xc = y2.copy(); bc = v
        T *= 0.9985
        if T < 6e-06:
            T = 6e-06
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = _lp_polish(xb, budget=min(remf - 0.5, 5.0), init_r=6e-4)
            vf = _max_cos(yf)
            if vf < bb:
                xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
