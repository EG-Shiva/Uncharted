import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    np.random.seed(0)
    t0 = time.time()
    LIMIT = 172.0
    SQP_BUDGET = 150.0
    n, d = 93, 13
    dm = d - 1
    N = n * dm  # 1116
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                a = _norm(a)
                v = _maxcos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        rng = np.random.RandomState(0)
        best = _norm(rng.randn(n, d))
        best_val = _maxcos(best)
    X = best.copy()
    cur = float(best_val)
    bestX = X.copy()
    # pair indices (all 4278)
    iu0, iu1 = np.triu_indices(n, 1)
    Pfull = len(iu0)
    ar12 = np.arange(dm)
    # D0 delta matrix (d, dm): D0[a,b]=1 if a==b+1
    D0 = np.zeros((d, dm))
    for b in range(dm):
        D0[b + 1, b] = 1.0
    radius = 3e-3
    e1 = np.zeros(d)
    e1[0] = 1.0
    it = 0
    stall = 0
    # preallocate col template helpers
    while True:
        now = time.time()
        if now - t0 > SQP_BUDGET or now - t0 > LIMIT - 12:
            break
        it += 1
        # --- Gram and active set info ---
        G = X @ X.T
        c_all = G[iu0, iu1]
        cur = float(c_all.max())
        # tracking: counts at slacks
        slack = cur - c_all
        try:
            n10 = int((slack < 1e-10).sum())
            n8 = int((slack < 1e-8).sum())
            n6 = int((slack < 1e-6).sum())
        except Exception:
            n10 = n8 = n6 = 0
        # KKT proxy for top 100 pairs: mean-gradient tangent norm
        try:
            Kk = min(100, Pfull)
            idxk = np.argpartition(-c_all, Kk - 1)[:Kk]
            Ik = iu0[idxk]
            Jk = iu1[idxk]
            grad = np.zeros_like(X)
            np.add.at(grad, Ik, X[Jk])
            np.add.at(grad, Jk, X[Ik])
            grad /= float(Kk)
            dotk = np.einsum('ij,ij->i', grad, X)[:, None]
            gtk = grad - dotk * X
            kkt = float(np.linalg.norm(gtk))
        except Exception:
            kkt = float('nan')
        # decide pair subset: keep all unless HiGHS too slow -> drop slack>1e-3
        # we keep all by default; if iteration slow we filter next time
        # For speed, if Pfull large and radius tiny, still keep all (cloud relevant)
        sel_all = True
        # allow filtering if previous solves were slow (stall with large P)
        # simple: use full set; filtering only if it>50 and n6>2000? keep full anyway
        # Build tangent bases via Householder mapping x -> s*e1
        s = np.sign(X[:, 0])
        s[s == 0] = 1.0
        U = X.copy()
        U[:, 0] -= s
        denom = np.einsum('ij,ij->i', U, U)  # (n,)
        Barr = np.empty((n, d, dm), dtype=np.float64)
        # broadcast D0
        Barr[:] = D0[None, :, :]
        mask = denom > 1e-12
        if np.any(mask):
            Um = U[mask]  # (nm, d)
            denm = denom[mask]  # (nm,)
            Uc = Um[:, 1:]  # (nm, dm)
            # product (nm, d, dm) = Um[:,:,None]*Uc[:,None,:]
            prod = Um[:, :, None] * Uc[:, None, :]
            Barr[mask] -= 2.0 * prod / denm[:, None, None]
        # else degenerate already D0
        # Build linear coefficients Ai = B_I^T x_J, Aj = B_J^T x_I
        # vectorized via fancy indexing (P ~4278, manageable)
        try:
            BI = Barr[iu0]  # (P,d,dm)
            BJ = Barr[iu1]
        except Exception:
            radius *= 0.5
            continue
        XJ = X[iu1]  # (P,d)
        XI = X[iu0]
        Ai = np.einsum('pab,pa->pb', BI, XJ)  # (P,dm)
        Aj = np.einsum('pab,pa->pb', BJ, XI)
        # optionally drop slack>1e-3 if HiGHS slow: we check time per solve adaptively
        # keep all for precision
        use_idx = None
        Pc = Pfull
        Aic = Ai
        Ajc = Aj
        cc = c_all
        Ic = iu0
        Jc = iu1
        # Build sparse A_ub: [Ai Aj -1] y - t <= -c
        try:
            cols = np.empty((Pc, 2 * dm + 1), dtype=np.int64)
            cols[:, :dm] = Ic[:, None] * dm + ar12[None, :]
            cols[:, dm:2 * dm] = Jc[:, None] * dm + ar12[None, :]
            cols[:, 2 * dm] = N
            vals = np.empty((Pc, 2 * dm + 1), dtype=np.float64)
            vals[:, :dm] = Aic
            vals[:, dm:2 * dm] = Ajc
            vals[:, 2 * dm] = -1.0
            rows = np.repeat(np.arange(Pc), 2 * dm + 1)
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(Pc, N + 1)).tocsr()
            b_ub = -cc
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(N + 1)
        obj[-1] = 1.0
        bounds = [(-radius, radius)] * N + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                          options={"presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            radius *= 0.5
            if radius < 1e-11:
                # tiny jitter to change active set
                rngj = np.random.RandomState(it)
                Xj = X + 1e-7 * rngj.randn(n, d)
                Xj = _norm(Xj)
                vj = _maxcos(Xj)
                if vj < cur + 2e-8:
                    X = Xj
                radius = 3e-6
                stall = 0
            continue
        yv = res.x[:-1]
        t_pred = float(res.x[-1])
        pred_red = cur - t_pred
        if not np.isfinite(pred_red):
            radius *= 0.5
            continue
        if pred_red <= 1e-13:
            # first-order optimal at this radius; shrink to refine ties
            radius *= 0.5
            stall += 1
            if radius < 1e-10:
                # try micro jitter of worst points
                G2 = X @ X.T
                np.fill_diagonal(G2, -2.0)
                rm = G2.max(axis=1)
                worst = np.argsort(rm)[-8:]
                rngj = np.random.RandomState(1000 + it)
                Xj = X.copy()
                Xj[worst] += 2e-7 * rngj.randn(len(worst), d)
                Xj = _norm(Xj)
                vj = _maxcos(Xj)
                # accept as iterate if not much worse to move basin slightly
                if vj < cur + 1e-9:
                    X = Xj
                radius = 2e-6
                stall = 0
            continue
        Y = yv.reshape(n, dm)
        maxabs = float(np.max(np.abs(yv))) if yv.size else 0.0
        # displacement D = B @ Y
        Ddir = np.einsum('iab,ib->ia', Barr, Y)
        # exact trial
        Xt = X + Ddir
        Xt = _norm(Xt)
        actual = _maxcos(Xt)
        if actual < cur - 1e-15:
            rho = (cur - actual) / pred_red if pred_red > 1e-14 else 0.0
            X = Xt
            cur = actual
            if actual < best_val - 1e-16:
                best_val = actual
                bestX = Xt.copy()
            if rho > 0.7 and maxabs > 0.9 * radius:
                radius = min(radius * 1.5, 1e-2)
            elif rho < 0.1:
                radius = max(radius * 0.9, 1e-12)
            else:
                # mild expansion if good agreement
                if rho > 0.5:
                    radius = min(radius * 1.2, 8e-3)
            stall = 0
        else:
            # exact max-cosine backtracking along Ddir
            found = False
            for alpha in (0.5, 0.25, 0.125, 0.0625):
                Xb = _norm(X + alpha * Ddir)
                vb = _maxcos(Xb)
                if vb < cur - 1e-15:
                    rho_b = (cur - vb) / (pred_red * alpha if pred_red > 0 else 1.0)
                    X = Xb
                    cur = vb
                    if vb < best_val:
                        best_val = vb
                        bestX = Xb.copy()
                    radius = max(radius * 0.7, 1e-12)
                    found = True
                    stall = 0
                    break
            if not found:
                radius *= 0.5
                stall += 1
                if radius < 5e-11:
                    radius = 2e-6
                    if stall > 10:
                        rngj = np.random.RandomState(2000 + it)
                        Xj = X + 5e-7 * rngj.randn(n, d)
                        Xj = _norm(Xj)
                        vj = _maxcos(Xj)
                        if vj < cur + 5e-9:
                            X = Xj
                        stall = 0
        # safety: always retain best
        # loop time check handled at top
    bestX = _norm(bestX)
    return bestX
