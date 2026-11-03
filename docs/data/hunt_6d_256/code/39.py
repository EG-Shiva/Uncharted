import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_and_tied(x, tol=1e-8):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = int(np.sum(G >= m - tol))
    return m, cnt, G

def _tangent_bases(x):
    n, d = x.shape
    Q = np.empty((n, d, d - 1))
    amax = np.argmax(np.abs(x), axis=1)
    Elist = np.eye(d)
    for i in range(n):
        k = int(amax[i])
        xi = x[i]
        cols = [l for l in range(d) if l != k]
        F = np.empty((d, d - 1))
        for ci, l in enumerate(cols):
            e = Elist[l]
            F[:, ci] = e - xi[l] * xi
        Qi, _ = np.linalg.qr(F)
        Q[i] = Qi
    return Q

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EQ_END = t0 + 157.0
    rng = np.random.default_rng(0)
    n, d, dd = 256, 6, 5
    NV = n * dd
    # load incumbent
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    cur = x_best.copy()
    cur_max, cur_tied, _ = _max_and_tied(cur, tol=1e-8)
    best_x = cur.copy()
    best_max = cur_max
    best_tied = cur_tied
    radii = [3e-4, 1e-4, 3e-5, 1e-5]
    ridx = 0
    iuI, iuJ = np.triu_indices(n, 1)
    obj1 = np.zeros(NV + 1)
    obj1[-1] = 1.0
    bnd1 = None
    while time.time() < EQ_END:
        rem = EQ_END - time.time()
        if rem < 3.0:
            break
        r = radii[ridx % len(radii)]
        ridx += 1
        # active set: slack < 3e-4
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        c_all = G[iuI, iuJ]
        thr = m - 3e-4
        sel = c_all > thr
        I = iuI[sel]; J = iuJ[sel]
        csel = c_all[sel]
        p = len(I)
        if p < 10:
            # expand to at least top 500 to keep LP meaningful
            k = min(500, len(c_all))
            part = np.argpartition(-c_all, k)[:k]
            I = iuI[part]; J = iuJ[part]; csel = c_all[part]
            p = k
        if p > 2200:
            part = np.argpartition(-csel, 2200)[:2200]
            I = I[part]; J = J[part]; csel = csel[part]
            p = 2200
        # tangent bases
        try:
            Q = _tangent_bases(cur)
        except Exception:
            continue
        xI = cur[I]  # p,6
        xJ = cur[J]
        QI = Q[I]  # p,6,5
        QJ = Q[J]
        # a_I = QI^T xJ , a_J = QJ^T xI
        AI = np.sum(QI * xJ[:, :, None], axis=1)  # p,5
        AJ = np.sum(QJ * xI[:, :, None], axis=1)
        # build sparse A_ub for LP1: p x (NV+1)
        rows = np.repeat(np.arange(p), 2 * dd + 1)
        colI = (I[:, None] * dd + np.arange(dd)).ravel()
        colJ = (J[:, None] * dd + np.arange(dd)).ravel()
        # interleave per row: [I-block(5), J-block(5), t]
        cols = np.empty(p * (2 * dd + 1), dtype=int)
        vals = np.empty(p * (2 * dd + 1), dtype=float)
        # fill row by row via vectorized blocks
        # positions within each row chunk
        for k in range(p):
            base = k * (2 * dd + 1)
            vals[base:base + dd] = AI[k]
            cols[base:base + dd] = I[k] * dd + np.arange(dd)
            vals[base + dd:base + 2 * dd] = AJ[k]
            cols[base + dd:base + 2 * dd] = J[k] * dd + np.arange(dd)
            vals[base + 2 * dd] = -1.0
            cols[base + 2 * dd] = NV
        bub = -csel
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
        except Exception:
            continue
        if bnd1 is None or True:
            bnds1 = [(-r, r)] * NV + [(None, None)]
        else:
            bnds1 = bnd1
        # LP1
        t_opt = None
        y1 = None
        try:
            tl = min(6.0, max(1.0, rem - 1.0))
            res = linprog(obj1, A_ub=Aub, b_ub=bub, bounds=bnds1, method="highs",
                          options={"time_limit": tl, "presolve": True})
        except Exception:
            res = None
        if res is not None and getattr(res, "status", 1) == 0 and res.x is not None:
            try:
                y1 = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
                t_opt = float(res.x[-1])
                if not np.all(np.isfinite(y1)) or not np.isfinite(t_opt):
                    y1 = None
                    t_opt = None
            except Exception:
                y1 = None
                t_opt = None
        cand_list = []
        if y1 is not None:
            try:
                dx = np.sum(Q * y1[:, None, :], axis=2)  # n,6? Q n,6,5 * y n,5
                # Q: n,6,5 ; y1: n,5 -> dx[i,d]=sum_k Q[i,d,k]*y[i,k]
                # above sum axis wrong; recompute correctly
                dx = np.einsum("ndk,nk->nd", Q, y1)
                trial = cur + dx
                nr = np.linalg.norm(trial, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                trial /= nr
                cand_list.append(trial)
            except Exception:
                pass
        # LP2 : lexicographic min sum at fixed t_opt
        if t_opt is not None and time.time() < EQ_END - 2.0:
            # use top 1500 pairs for sum
            if p > 1500:
                o = np.argsort(-csel)[:1500]
                I2 = I[o]; J2 = J[o]; cs2 = csel[o]
                AI2 = AI[o]; AJ2 = AJ[o]
                p2 = 1500
            else:
                I2 = I; J2 = J; cs2 = csel
                AI2 = AI; AJ2 = AJ
                p2 = p
            # objective coeffs
            coeff = np.zeros((n, dd))
            np.add.at(coeff, I2, AI2)
            np.add.at(coeff, J2, AJ2)
            c2 = coeff.ravel()
            # A_y
            rows2 = np.repeat(np.arange(p2), 2 * dd)
            cols2 = np.empty(p2 * 2 * dd, dtype=int)
            vals2 = np.empty(p2 * 2 * dd, dtype=float)
            for k in range(p2):
                base = k * 2 * dd
                vals2[base:base + dd] = AI2[k]
                cols2[base:base + dd] = I2[k] * dd + np.arange(dd)
                vals2[base + dd:base + 2 * dd] = AJ2[k]
                cols2[base + dd:base + 2 * dd] = J2[k] * dd + np.arange(dd)
            bub2 = -cs2 + t_opt + 1e-12
            try:
                Ay = coo_matrix((vals2, (rows2, cols2)), shape=(p2, NV)).tocsr()
                bnds2 = [(-r, r)] * NV
                tl2 = min(5.0, max(1.0, EQ_END - time.time() - 0.5))
                res2 = linprog(c2, A_ub=Ay, b_ub=bub2, bounds=bnds2, method="highs",
                               options={"time_limit": tl2, "presolve": True})
            except Exception:
                res2 = None
            if res2 is not None and getattr(res2, "status", 1) == 0 and res2.x is not None:
                try:
                    y2 = np.asarray(res2.x, dtype=float).reshape(n, dd)
                    if np.all(np.isfinite(y2)):
                        dx2 = np.einsum("ndk,nk->nd", Q, y2)
                        trial2 = cur + dx2
                        nr2 = np.linalg.norm(trial2, axis=1, keepdims=True)
                        nr2[nr2 == 0] = 1.0
                        trial2 /= nr2
                        cand_list.append(trial2)
                except Exception:
                    pass
        # evaluate candidates
        for trial in cand_list:
            try:
                nm, nt, _ = _max_and_tied(trial, tol=1e-8)
            except Exception:
                continue
            if not np.isfinite(nm):
                continue
            if nm < best_max - 1e-15:
                best_max = nm
                best_tied = nt
                best_x = trial.copy()
            # accept to cur?
            improved = False
            if nm < cur_max - 1e-12:
                improved = True
            elif nm <= cur_max + 1e-12 and nt > cur_tied:
                improved = True
            if improved:
                cur = trial.copy()
                cur_max = nm
                cur_tied = nt
        # if no candidate accepted, cur stays; cycle continues (topology change via different radius)
        # safety: if time is plenty but stuck with tiny radii, occasionally keep best as cur
        # do nothing else
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    # tiny shrink guard not needed; return best
    return np.asarray(best_x, dtype=float)
