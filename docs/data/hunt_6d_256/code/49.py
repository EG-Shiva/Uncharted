import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_ties(x, tol=1e-8):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu[0], iu[1]]
    m = float(c.max())
    t = int(np.sum(c > m - tol))
    return m, t, c

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_basis(x):
    n = x.shape[0]
    B = np.empty((n, 6, 5), dtype=float)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        V = np.zeros((6, 5), dtype=float)
        col = 0
        inv = 1.0 / xi[k]
        for j in range(6):
            if j == k:
                continue
            v = np.zeros(6, dtype=float)
            v[j] = 1.0
            v[k] = -xi[j] * inv
            V[:, col] = v
            col += 1
        Q, _ = np.linalg.qr(V)
        B[i] = Q
    return B

def solve():
    t0 = time.time()
    T_STOP = t0 + 150.0
    # load incumbent
    x_cur = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
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
    x_cur = x_cur / np.linalg.norm(x_cur, axis=1, keepdims=True)
    cur_val, cur_ties, _ = _max_cos_ties(x_cur, tol=1e-8)
    x_best = x_cur.copy()
    best_val = cur_val
    best_ties = cur_ties

    n, d = 256, 6
    NV = n * 5
    radii = [3e-4, 1e-4, 3e-5, 1e-5]
    iuI_all, iuJ_all = np.triu_indices(n, 1)
    it = 0

    while time.time() < T_STOP:
        r = radii[it % 4]
        it += 1
        # current max and active set slack < 3e-4
        try:
            G = x_cur @ x_cur.T
        except Exception:
            break
        iu_c = G[iuI_all, iuJ_all]
        mval = float(iu_c.max())
        # sync cur_val
        cur_val = mval
        # active set
        thresh = mval - 3e-4
        mask = iu_c > thresh
        I = iuI_all[mask]
        J = iuJ_all[mask]
        csel = iu_c[mask]
        p = len(I)
        if p < 2 or p > 8000:
            # cap to top 2000 if huge
            if p > 8000:
                part = np.argpartition(-csel, 2000)[:2000]
                I = I[part]; J = J[part]; csel = csel[part]
                p = 2000
            if p < 2:
                continue
        if p > 2000:
            part = np.argpartition(-csel, 2000)[:2000]
            I = I[part]; J = J[part]; csel = csel[part]
            p = 2000
        # order by largest first (for second LP top1500)
        order = np.argsort(-csel)
        I = I[order]; J = J[order]; csel = csel[order]

        # tangent basis at x_cur
        try:
            B = _tangent_basis(x_cur)
        except Exception:
            continue
        # coefficients ai = B_i^T x_j , aj = B_j^T x_i  (p,5)
        Ai = np.empty((p, 5), dtype=float)
        Aj = np.empty((p, 5), dtype=float)
        # loop (1400*2 dots, cheap)
        for k in range(p):
            ik = int(I[k]); jk = int(J[k])
            Ai[k] = B[ik].T @ x_cur[jk]
            Aj[k] = B[jk].T @ x_cur[ik]

        # ---- LP1: min t s.t. ai^T yi + aj^T yj - t <= -c ----
        # build sparse A_ub (p, NV+1)
        # each row: 5 entries at ik*5:(ik+1)*5, 5 at jk*5, 1 at NV
        try:
            rows1 = np.repeat(np.arange(p), 11)
            cols1 = np.empty(p * 11, dtype=int)
            vals1 = np.empty(p * 11, dtype=float)
            # fill vectorized
            # positions 0..4 -> i block, 5..9 -> j block, 10 -> t
            bi = (I.astype(np.int64) * 5)
            bj = (J.astype(np.int64) * 5)
            # build cols
            for q in range(5):
                cols1[q::11] = bi + q
                vals1[q::11] = Ai[:, q]
            for q in range(5):
                cols1[5 + q::11] = bj + q
                vals1[5 + q::11] = Aj[:, q]
            # wait above slicing wrong: rows1 interleaved, need per-row block
            # redo with explicit reshape approach for correctness
            cols1 = np.empty((p, 11), dtype=int)
            vals1 = np.empty((p, 11), dtype=float)
            cols1[:, 0:5] = bi[:, None] + np.arange(5)[None, :]
            vals1[:, 0:5] = Ai
            cols1[:, 5:10] = bj[:, None] + np.arange(5)[None, :]
            vals1[:, 5:10] = Aj
            cols1[:, 10] = NV
            vals1[:, 10] = -1.0
            A1 = coo_matrix((vals1.ravel(), (np.repeat(np.arange(p), 11), cols1.ravel())), shape=(p, NV + 1)).tocsr()
            b1 = -csel
            obj1 = np.zeros(NV + 1, dtype=float)
            obj1[-1] = 1.0
            bounds1 = [(-r, r)] * NV + [(None, None)]
            res1 = linprog(obj1, A_ub=A1, b_ub=b1, bounds=bounds1, method="highs")
        except Exception:
            continue
        if res1.status != 0 or res1.x is None:
            continue
        try:
            y1 = np.asarray(res1.x[:-1], dtype=float).reshape(n, 5)
            t_pred = float(res1.x[-1])
        except Exception:
            continue
        if not np.all(np.isfinite(y1)) or not np.isfinite(t_pred):
            continue

        # ---- LP2 lexicographic: fix t_fix, min sum linearized ----
        # top 1500
        p2 = min(p, 1500)
        I2 = I[:p2]; J2 = J[:p2]; c2 = csel[:p2]
        A2i = Ai[:p2]; A2j = Aj[:p2]
        t_fix = t_pred + 5e-7
        # objective: sum over pairs of linearized = sum_i coeff_i^T y_i
        # accumulate
        obj2 = np.zeros(NV, dtype=float)
        # for speed use np.add.at
        # build index arrays
        # Ai contributions
        for q in range(5):
            np.add.at(obj2, (I2 * 5 + q), A2i[:, q])
            np.add.at(obj2, (J2 * 5 + q), A2j[:, q])
        try:
            cols2 = np.empty((p2, 10), dtype=int)
            vals2 = np.empty((p2, 10), dtype=float)
            bi2 = (I2.astype(np.int64) * 5)
            bj2 = (J2.astype(np.int64) * 5)
            cols2[:, 0:5] = bi2[:, None] + np.arange(5)[None, :]
            vals2[:, 0:5] = A2i
            cols2[:, 5:10] = bj2[:, None] + np.arange(5)[None, :]
            vals2[:, 5:10] = A2j
            A2 = coo_matrix((vals2.ravel(), (np.repeat(np.arange(p2), 10), cols2.ravel())), shape=(p2, NV)).tocsr()
            b2 = t_fix - c2
            bounds2 = [(-r, r)] * NV
            res2 = linprog(obj2, A_ub=A2, b_ub=b2, bounds=bounds2, method="highs")
        except Exception:
            res2 = None
        cands = []
        # candidate from LP1
        try:
            dx1 = np.einsum('ijk,ik->ij', B, y1)
            y1x = x_cur + dx1
            y1x /= np.linalg.norm(y1x, axis=1, keepdims=True)
            cands.append(y1x)
        except Exception:
            pass
        if res2 is not None and getattr(res2, 'status', 1) == 0 and res2.x is not None:
            try:
                y2 = np.asarray(res2.x, dtype=float).reshape(n, 5)
                if np.all(np.isfinite(y2)):
                    dx2 = np.einsum('ijk,ik->ij', B, y2)
                    y2x = x_cur + dx2
                    y2x /= np.linalg.norm(y2x, axis=1, keepdims=True)
                    # put lexicographic first (preferred)
                    cands.insert(0, y2x)
            except Exception:
                pass
        accepted = False
        for cand in cands:
            try:
                vm, tm, _ = _max_cos_ties(cand, tol=1e-8)
            except Exception:
                continue
            if not np.isfinite(vm):
                continue
            # accept if true max decreases
            if vm < cur_val - 1e-12:
                x_cur = cand
                cur_val = vm
                cur_ties = tm
                if vm < best_val - 1e-12:
                    best_val = vm
                    best_ties = tm
                    x_best = cand.copy()
                elif abs(vm - best_val) <= 1e-12 and tm > best_ties:
                    best_ties = tm
                    x_best = cand.copy()
                accepted = True
                break
            elif vm <= cur_val + 5e-12 and tm > cur_ties:
                # equalized without increasing max
                x_cur = cand
                cur_ties = tm
                # cur_val stays ~same (update to vm)
                cur_val = vm
                if vm < best_val - 1e-12:
                    best_val = vm
                    best_ties = tm
                    x_best = cand.copy()
                accepted = True
                break
        # if neither accepted, stay at x_cur (try next radius)
        # update cur_ties/val occasionally to avoid drift
        if not accepted:
            # recompute ties for cur (in case mval drift)
            _, cur_ties, _ = _max_cos_ties(x_cur, tol=1e-8)
            cur_val = float(mval)
        if time.time() >= T_STOP:
            break

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
