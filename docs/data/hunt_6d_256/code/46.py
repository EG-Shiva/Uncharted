import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_ties(x, tol=1e-8):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu[0], iu[1]]
    m = float(c.max())
    ties = int(np.sum(c >= m - tol))
    return m, ties, c, iu

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n = X.shape[0]
    B = np.empty((n, 6, 5), dtype=float)
    e0 = np.zeros(6); e0[0] = 1.0
    e1 = np.zeros(6); e1[1] = 1.0
    I6 = np.eye(6)
    for i in range(n):
        x = X[i]
        e = e0 if abs(float(x[0])) < 0.99 else e1
        u = e - x
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            B[i] = I6[:, 1:]
        else:
            u = u / nu
            H = I6 - 2.0 * np.outer(u, u)
            B[i] = H[:, 1:]
    return B

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EQ_END = t0 + 152.0
    n, d = 256, 6
    NV = n * 5
    # load start
    x_cur = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    x_cur = a.copy()
        except Exception:
            continue
    if x_cur is None:
        rng = np.random.default_rng(0)
        x_cur = rng.normal(size=(256, 6))
        x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
        best_val = _max_cos(x_cur)
    x_best = x_cur.copy()
    cur_max, cur_ties, _, _ = _max_cos_ties(x_cur, tol=1e-8)
    best_max = cur_max
    # trust cycle
    cycle = [3e-4, 1e-4, 3e-5, 1e-5]
    it = 0
    rng_acc = np.random.default_rng(1)
    while time.time() < EQ_END:
        it += 1
        r = cycle[(it - 1) % len(cycle)]
        if time.time() > EQ_END - 1.0:
            break
        # active set: slack < 3e-4, cap top 1500
        G = x_cur @ x_cur.T
        iuI_all = None
        try:
            iu0 = np.triu_indices(n, 1)
            c_all = G[iu0[0], iu0[1]]
            mval = float(c_all.max())
            cur_max = mval
            sel = c_all > (mval - 3e-4)
            I = iu0[0][sel]; J = iu0[1][sel]
            csel = c_all[sel]
            if len(I) > 1500:
                part = np.argpartition(-csel, 1500)[:1500]
                I = I[part]; J = J[part]; csel = csel[part]
            # sort descending for determinism (top first)
            order = np.argsort(-csel)
            I = I[order]; J = J[order]; csel = csel[order]
            p = len(I)
            if p < 2:
                continue
            cur_ties = int(np.sum(c_all >= mval - 1e-8))
        except Exception:
            continue
        # tangent bases at cur
        try:
            B = _tangent_bases(x_cur)
        except Exception:
            continue
        # coefficients Bi^T xj
        try:
            BI = B[I]  # (p,6,5)
            BJ = B[J]
            XJ = x_cur[J]  # (p,6)
            XI = x_cur[I]
            coeffI = np.einsum('kij,kj->ki', BI, XJ)  # (p,5)
            coeffJ = np.einsum('kij,kj->ki', BJ, XI)
        except Exception:
            continue
        # Build LP1: min t s.t. a^T y - t <= -csel, |y|<=r
        t_star = None
        y1 = None
        try:
            rows_I = np.repeat(np.arange(p), 5)
            Irep = np.repeat(I, 5)
            off = np.tile(np.arange(5), p)
            cols_I = Irep * 5 + off
            vals_I = coeffI.ravel()
            rows_J = np.repeat(np.arange(p), 5)
            Jrep = np.repeat(J, 5)
            cols_J = Jrep * 5 + off
            vals_J = coeffJ.ravel()
            rows_T = np.arange(p)
            cols_T = np.full(p, NV, dtype=int)
            vals_T = np.full(p, -1.0)
            rows = np.concatenate([rows_I, rows_J, rows_T])
            cols = np.concatenate([cols_I, cols_J, cols_T])
            vals = np.concatenate([vals_I, vals_J, vals_T])
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
            b_ub = -csel
            c_obj = np.zeros(NV + 1); c_obj[-1] = 1.0
            bounds = [(-r, r)] * NV + [(None, mval)]
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.status == 0 and np.all(np.isfinite(res.x)):
                t_star = float(res.x[-1])
                y1 = res.x[:-1].copy()
            else:
                t_star = None
                y1 = None
        except Exception:
            t_star = None
            y1 = None
        # evaluate LP1 candidate
        cand_list = []
        if y1 is not None and t_star is not None:
            try:
                Y = y1.reshape(n, 5)
                dX = np.einsum('nij,nj->ni', B, Y)
                Xn = x_cur + dX
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                m1, ties1, _, _ = _max_cos_ties(Xn, tol=1e-8)
                cand_list.append((m1, ties1, Xn))
            except Exception:
                pass
        # Lexicographic second LP: min sum linearized at fixed t
        if t_star is not None and y1 is not None:
            if time.time() > EQ_END - 1.0:
                pass
            else:
                try:
                    t_fixed = t_star + 5e-7
                    if t_fixed > mval - 1e-9:
                        t_fixed = mval - 1e-9
                    # objective: sum of rows
                    c2 = np.zeros(NV, dtype=float)
                    # accumulate with add.at on (n,5) view
                    c2m = c2.reshape(n, 5)
                    np.add.at(c2m, I, coeffI)
                    np.add.at(c2m, J, coeffJ)
                    # scale to avoid huge/ tiny? normalize
                    sc = float(np.max(np.abs(c2)))
                    if sc > 0 and np.isfinite(sc):
                        c2 = c2 / sc
                    rows2 = np.concatenate([rows_I, rows_J])
                    cols2 = np.concatenate([cols_I, cols_J])
                    vals2 = np.concatenate([vals_I, vals_J])
                    A2 = coo_matrix((vals2, (rows2, cols2)), shape=(p, NV)).tocsr()
                    b2 = t_fixed - csel
                    bounds2 = [(-r, r)] * NV
                    res2 = linprog(c2, A_ub=A2, b_ub=b2, bounds=bounds2, method="highs")
                    if res2.status == 0 and np.all(np.isfinite(res2.x)):
                        Y2 = res2.x.reshape(n, 5)
                        dX2 = np.einsum('nij,nj->ni', B, Y2)
                        Xn2 = x_cur + dX2
                        Xn2 /= np.linalg.norm(Xn2, axis=1, keepdims=True)
                        m2, ties2, _, _ = _max_cos_ties(Xn2, tol=1e-8)
                        cand_list.append((m2, ties2, Xn2))
                except Exception:
                    pass
        # acceptance: prefer decrease, else tie increase without increase
        accepted = False
        # sort candidates by (max, -ties)
        cand_list.sort(key=lambda t: (t[0], -t[1]))
        for (m_new, ties_new, Xn) in cand_list:
            if not np.all(np.isfinite(Xn)):
                continue
            if m_new < cur_max - 1e-12:
                x_cur = Xn
                cur_max = m_new
                cur_ties = ties_new
                accepted = True
                if m_new < best_max - 1e-12:
                    best_max = m_new
                    x_best = Xn.copy()
                break
            elif m_new <= cur_max + 2e-12 and ties_new > cur_ties:
                x_cur = Xn
                cur_max = m_new
                cur_ties = ties_new
                accepted = True
                if m_new < best_max - 1e-12:
                    best_max = m_new
                    x_best = Xn.copy()
                break
            else:
                # also track best even if not accepted as move
                if m_new < best_max - 1e-12:
                    best_max = m_new
                    x_best = Xn.copy()
        # if nothing accepted, keep cur (cycle radius automatically); add tiny deterministic jitter every ~25 iters to avoid stall
        if (not accepted) and (it % 40 == 0):
            try:
                if time.time() < EQ_END - 2.0:
                    # small random tangent step of size 3e-5 to probe new basin, keep if not worse by >1e-6
                    amp = 3e-5
                    R = rng_acc.normal(size=(n, 5)) * amp
                    # project: need bases (reuse B)
                    dX = np.einsum('nij,nj->ni', B, R)
                    Xt = x_cur + dX
                    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                    mt, tiest, _, _ = _max_cos_ties(Xt, tol=1e-8)
                    if mt <= cur_max + 1e-6:
                        x_cur = Xt
                        cur_max = mt
                        cur_ties = tiest
                        if mt < best_max:
                            best_max = mt
                            x_best = Xt.copy()
            except Exception:
                pass
        # safety: refresh cur_max/ties periodically from x_cur (already), continue
        if time.time() > DEADLINE - 20.0:
            break
    # return best
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
