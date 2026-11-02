import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    SEARCH_END = t0 + 145.0
    n, d = 73, 13
    rng = np.random.default_rng(0)

    # load incumbent
    Xrec = None
    best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=np.float64)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_file:
                        best_file = c
                        Xrec = v.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy()
    best = float(best_file)

    iu0, iu1 = np.triu_indices(n, 1)
    ar12 = np.arange(12)
    idx12 = (ar12[None, :] - ar12[:, None]) % 12  # (12,12) roll by k
    # for aggregation: acc[l] = sum_k Gblk[k,(l+k)%12]
    agg12 = [(ar12 + k) % 12 for k in range(12)]
    ar13 = np.arange(13)
    idx13 = (ar13[None, :] - ar13[:, None]) % 13
    agg13 = [(ar13 + k) % 13 for k in range(13)]
    e13 = np.zeros(d); e13[-1] = 1.0

    def expand_c12(Tn):
        X = np.empty((73, 13))
        for j in range(6):
            X[j*12:(j+1)*12, :12] = Tn[j, :12][idx12]
            X[j*12:(j+1)*12, 12] = Tn[j, 12]
        X[72] = e13
        return X

    def expand_c13(Tn5, F8):
        X = np.empty((73, 13))
        for j in range(5):
            X[j*13:(j+1)*13, :] = Tn5[j][idx13]
        X[65:73] = F8
        return X

    def adam_c12(Y0, steps, s0, s1, lr0, lr1):
        Y = Y0.copy()  # (6,13) unit rows
        m = np.zeros_like(Y); vv = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        best_v = 1e9; best_T = Y.copy()
        W = np.empty((73, 73))
        for k in range(steps):
            frac = k / max(steps-1, 1)
            s = s0 + (s1-s0)*frac
            lr = np.exp(l0 + (l1-l0)*frac)
            X = expand_c12(Y)
            G = X @ X.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            if mmax < best_v:
                best_v = mmax; best_T = Y.copy()
            e = np.exp(s*(c-mmax))
            se = float(e.sum())
            w = e / se if np.isfinite(se) and se > 0 else np.ones_like(e)/len(e)
            W.fill(0.0)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            np.fill_diagonal(W, 0.0)
            GX = W @ X  # 73x13
            gT = np.empty((6, 13))
            for j in range(6):
                Gb = GX[j*12:(j+1)*12, :]  # 12x13
                G12 = Gb[:, :12]
                acc = np.zeros(12)
                for kk in range(12):
                    acc += G12[kk, agg12[kk]]
                gT[j, :12] = acc
                gT[j, 12] = float(Gb[:, 12].sum())
            dot = np.sum(gT*Y, axis=1, keepdims=True)
            Gt = gT - dot*Y
            m = b1*m + (1-b1)*Gt
            vv = b2*vv + (1-b2)*(Gt*Gt)
            mh = m/(1-b1p); vh = vv/(1-b2p)
            b1p *= b1; b2p *= b2
            step = mh/(np.sqrt(vh)+eps)
            Y = Y - lr*step
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # final eval
        Xf = expand_c12(Y)
        vf = _max_cos(Xf)
        if vf < best_v:
            best_v = vf; best_T = Y.copy()
        return best_T, best_v, Y

    def adam_c13(Y0, steps, s0, s1, lr0, lr1):
        # Y0 (13,13): first 5 templates, last 8 free; unit rows
        Y = Y0.copy()
        m = np.zeros_like(Y); vv = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        best_v = 1e9; best_Y = Y.copy()
        W = np.empty((73, 73))
        for k in range(steps):
            frac = k / max(steps-1, 1)
            s = s0 + (s1-s0)*frac
            lr = np.exp(l0 + (l1-l0)*frac)
            T5 = Y[:5]; F8 = Y[5:]
            X = expand_c13(T5, F8)
            G = X @ X.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            if mmax < best_v:
                best_v = mmax; best_Y = Y.copy()
            e = np.exp(s*(c-mmax))
            se = float(e.sum())
            w = e / se if np.isfinite(se) and se > 0 else np.ones_like(e)/len(e)
            W.fill(0.0)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            np.fill_diagonal(W, 0.0)
            GX = W @ X
            gY = np.empty_like(Y)
            for j in range(5):
                Gb = GX[j*13:(j+1)*13, :]  # 13x13
                acc = np.zeros(13)
                for kk in range(13):
                    acc += Gb[kk, agg13[kk]]
                gY[j] = acc
            gY[5:] = GX[65:73, :]
            dot = np.sum(gY*Y, axis=1, keepdims=True)
            Gt = gY - dot*Y
            m = b1*m + (1-b1)*Gt
            vv = b2*vv + (1-b2)*(Gt*Gt)
            mh = m/(1-b1p); vh = vv/(1-b2p)
            b1p *= b1; b2p *= b2
            step = mh/(np.sqrt(vh)+eps)
            Y = Y - lr*step
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        Xf = expand_c13(Y[:5], Y[5:])
        vf = _max_cos(Xf)
        if vf < best_v:
            best_v = vf; best_Y = Y.copy()
        return best_Y, best_v

    def adam_full(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m = np.zeros_like(X); vv = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        l0, l1 = np.log(lr0), np.log(lr1)
        W = np.empty((n, n))
        for k in range(steps):
            if time.time() > SEARCH_END:
                break
            frac = k / max(steps-1, 1)
            s = s0 + (s1-s0)*frac
            lr = np.exp(l0 + (l1-l0)*frac)
            G = X @ X.T
            c = G[iu0, iu1]
            mmax = float(c.max())
            e = np.exp(s*(c-mmax))
            se = float(e.sum())
            w = e / se if np.isfinite(se) and se > 0 else np.ones_like(e)/len(e)
            W.fill(0.0)
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            WX = W @ X
            sprod = np.sum(WX*X, axis=1, keepdims=True)
            Gt = WX - sprod*X
            m = b1*m + (1-b1)*Gt
            vv = b2*vv + (1-b2)*(Gt*Gt)
            mh = m/(1-b1p); vh = vv/(1-b2p)
            b1p *= b1; b2p *= b2
            step = mh/(np.sqrt(vh)+eps)
            X = X - lr*step
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # ---- symmetric search ----
    sym_cands = []  # (val, X full)
    # C12 seeds
    try:
        for si in range(6):
            if time.time() > t0 + 60 or time.time() > SEARCH_END:
                break
            rs = np.random.default_rng(1000+si)
            Y0 = rs.standard_normal(size=(6, 13))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            bt, bv, _ = adam_c12(Y0, 600, 50.0, 500.0, 0.03, 0.005)
            Xf = expand_c12(bt)
            vv = _max_cos(Xf)
            sym_cands.append((vv, Xf))
            if vv < best:
                best = vv; Xbest = Xf.copy()
    except Exception:
        pass
    # C13 seeds
    try:
        for si in range(4):
            if time.time() > t0 + 100 or time.time() > SEARCH_END:
                break
            rs = np.random.default_rng(2000+si)
            Y0 = rs.standard_normal(size=(13, 13))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            bY, bv = adam_c13(Y0, 600, 50.0, 500.0, 0.03, 0.005)
            Xf = expand_c13(bY[:5], bY[5:])
            vv = _max_cos(Xf)
            sym_cands.append((vv, Xf))
            if vv < best:
                best = vv; Xbest = Xf.copy()
    except Exception:
        pass

    sym_cands.sort(key=lambda t: t[0])
    # ---- break symmetry ----
    try:
        nref = min(3, len(sym_cands))
        sigmas = [0.001, 0.003, 0.01]
        for ri in range(nref):
            if time.time() > SEARCH_END:
                break
            _, Xs = sym_cands[ri]
            sig = sigmas[ri % len(sigmas)]
            rs = np.random.default_rng(3000+ri)
            X0 = Xs + rs.standard_normal(size=(n, d)) * sig
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xr = adam_full(X0, 1000, 80.0, 700.0, 0.02, 0.002)
            vr = _max_cos(Xr)
            if vr < best and np.all(np.isfinite(Xr)):
                best = vr; Xbest = Xr.copy()
            # second anneal at higher sharpness
            if time.time() > SEARCH_END:
                break
            Xr2 = adam_full(Xr, 500, 400.0, 900.0, 0.008, 0.0015)
            vr2 = _max_cos(Xr2)
            if vr2 < best and np.all(np.isfinite(Xr2)):
                best = vr2; Xbest = Xr2.copy()
    except Exception:
        pass

    # ---- shake incumbent once (different basin near record) ----
    if time.time() < SEARCH_END - 5:
        try:
            rs = np.random.default_rng(999)
            X0 = Xrec + rs.standard_normal(size=(n, d)) * 0.004
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Xr = adam_full(X0, 700, 100.0, 600.0, 0.02, 0.002)
            vr = _max_cos(Xr)
            if vr < best and np.all(np.isfinite(Xr)):
                best = vr; Xbest = Xr.copy()
        except Exception:
            pass

    # ---- final LP polish (short) ----
    if time.time() < T_END - 3:
        try:
            remain = min(T_END - time.time() - 0.5, 18.0)
            start = time.time()
            radius = 1e-3
            X = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
            cur = _max_cos(X)
            while time.time() - start < remain and time.time() < T_END - 0.5 and radius > 1e-14:
                Cfull = X @ X.T
                c = Cfull[iu0, iu1]
                mmax = float(c.max())
                sel = c > mmax - max(20.0*radius, 1e-9)
                if np.count_nonzero(sel) > 2200:
                    idx = np.where(sel)[0]
                    order = np.argsort(-c[idx])[:2200]
                    mask = np.zeros_like(sel); mask[idx[order]] = True
                    sel = mask
                I, J = iu0[sel], iu1[sel]
                p = len(I)
                if p == 0:
                    radius *= 0.5; continue
                csel = c[sel]
                nv = n*d+1
                rows = np.repeat(np.arange(p), 2*d+1)
                cI = I[:, None]*d + np.arange(d)
                cJ = J[:, None]*d + np.arange(d)
                ct = np.full((p, 1), n*d)
                cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
                vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
                try:
                    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                    A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
                    obj = np.zeros(nv); obj[-1] = 1.0
                    res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                                  bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
                except Exception:
                    radius *= 0.5; continue
                if res.status != 0:
                    radius *= 0.5; continue
                y = X + res.x[:-1].reshape(n, d)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                vy = _max_cos(y)
                if vy < cur:
                    X, cur = y, vy
                    if cur < best:
                        best = cur; Xbest = X.copy()
                    radius = min(radius*1.5, 0.05)
                else:
                    radius *= 0.5
        except Exception:
            pass

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = Xrec / np.linalg.norm(Xrec, axis=1, keepdims=True)
    return Xbest
