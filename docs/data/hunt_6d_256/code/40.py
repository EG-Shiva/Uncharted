import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_tied(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    tied = int(np.sum(c > m - 1e-8))
    return m, tied, G

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(x):
    n, d = x.shape
    B = np.empty((n, d, 5))
    Eye = np.eye(d)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        cols = [j for j in range(d) if j != k]
        A = Eye[:, cols]
        coeff = xi @ A
        Aproj = A - xi[:, None] * coeff[None, :]
        Q, _ = np.linalg.qr(Aproj)
        B[i] = Q
    return B

def _build_A(I, J, c, B, x, N):
    p = len(I)
    rows = []
    cols = []
    data = []
    # precompute coeffs loop
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        ci = B[i].T @ x[j]
        cj = B[j].T @ x[i]
        bi = i * 5
        bj = j * 5
        for a in range(5):
            rows.append(k); cols.append(bi + a); data.append(float(ci[a]))
            rows.append(k); cols.append(bj + a); data.append(float(cj[a]))
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    data = np.array(data, dtype=float)
    Ay = coo_matrix((data, (rows, cols)), shape=(p, N)).tocsr()
    return Ay

def _y_to_x(x, B, y):
    n = x.shape[0]
    Y = y.reshape(n, 5)
    # dx_i = B_i @ y_i
    # einsum: dx[n,6] = sum_a B[n,6,a]*Y[n,a]
    dx = np.einsum('nda,na->nd', B, Y)
    xn = x + dx
    nr = np.linalg.norm(xn, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    xn = xn / nr
    return xn

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LOOP_END = t0 + 152.0
    n, d = 256, 6
    N = n * 5
    # load start
    x_best = None
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
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        rng0 = np.random.default_rng(0)
        x_best = rng0.normal(size=(n, d))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = _max_cos(x_best)
    x_cur = x_best.copy()
    m_cur, tied_cur, _ = _max_cos_tied(x_cur)
    best_x = x_cur.copy()
    best_m = m_cur
    best_tied = tied_cur

    iuI_all, iuJ_all = np.triu_indices(n, 1)

    radii_cycle = [3e-4, 1e-4, 3e-5, 1e-5]
    it = 0
    # cache G for selection
    while time.time() < LOOP_END:
        it += 1
        r = radii_cycle[(it - 1) % len(radii_cycle)]
        if time.time() > LOOP_END - 2.0:
            break
        # current stats
        G = x_cur @ x_cur.T
        # use triu values
        c_all = G[iuI_all, iuJ_all]
        m_cur = float(c_all.max())
        tied_cur = int(np.sum(c_all > m_cur - 1e-8))
        # LP1 selection: slack < 3e-4
        mask = c_all > (m_cur - 3e-4)
        I1 = iuI_all[mask]
        J1 = iuJ_all[mask]
        c1 = c_all[mask]
        if len(c1) == 0:
            continue
        if len(c1) > 2500:
            part = np.argpartition(-c1, 2500)[:2500]
            I1 = I1[part]; J1 = J1[part]; c1 = c1[part]
        # order descending for stability
        od = np.argsort(-c1)
        I1 = I1[od]; J1 = J1[od]; c1 = c1[od]
        p1 = len(I1)

        # tangent bases at x_cur
        try:
            B = _tangent_bases(x_cur)
        except Exception:
            continue

        # build Ay1
        try:
            Ay1 = _build_A(I1, J1, c1, B, x_cur, N)
        except Exception:
            continue

        # LP1: min t s.t. Ay1 y - t <= -c1, |y|<=r
        p = p1
        # A_ub = [Ay1, -1]
        try:
            coo1 = Ay1.tocoo()
            r1 = coo1.row.astype(int)
            c1c = coo1.col.astype(int)
            d1 = coo1.data
            rows_ub = np.concatenate([r1, np.arange(p, dtype=int)])
            cols_ub = np.concatenate([c1c, np.full(p, N, dtype=int)])
            vals_ub = np.concatenate([d1, -np.ones(p)])
            A_ub1 = coo_matrix((vals_ub, (rows_ub, cols_ub)), shape=(p, N + 1)).tocsr()
            b_ub1 = -c1
            obj1 = np.zeros(N + 1)
            obj1[-1] = 1.0
            bnds1 = [(-r, r)] * N + [(None, None)]
            res1 = linprog(obj1, A_ub=A_ub1, b_ub=b_ub1, bounds=bnds1, method="highs",
                           options={"time_limit": 6.0})
        except Exception:
            continue
        if res1.status != 0:
            continue
        y1 = np.asarray(res1.x[:-1], dtype=float)
        t_star = float(res1.x[-1])
        if not np.all(np.isfinite(y1)) or not np.isfinite(t_star):
            continue

        cand_list = []
        # candidate from LP1
        try:
            xc1 = _y_to_x(x_cur, B, y1)
            m1 = _max_cos(xc1)
            # tied count
            Gc = xc1 @ xc1.T
            np.fill_diagonal(Gc, -2.0)
            t1 = int(np.sum(Gc[iuI_all, iuJ_all] > m1 - 1e-8))
            cand_list.append((xc1, m1, t1))
        except Exception:
            pass

        # LP2: top 1500 pairs, fixed near-optimal t
        try:
            K2 = 1500
            if len(c_all) > K2:
                part2 = np.argpartition(-c_all, K2)[:K2]
                I2 = iuI_all[part2]; J2 = iuJ_all[part2]; c2 = c_all[part2]
            else:
                I2 = iuI_all; J2 = iuJ_all; c2 = c_all
            # sort descending
            od2 = np.argsort(-c2)
            I2 = I2[od2]; J2 = J2[od2]; c2 = c2[od2]
            Ay2 = _build_A(I2, J2, c2, B, x_cur, N)
            # near-optimal fixed t: t_star + eps
            eps_fix = 1e-7
            # guard: t_star could be far below m_cur; keep t_fix <= m_cur
            t_fix = t_star + eps_fix
            # if t_fix > m_cur: clamp slightly below m_cur? keep as is but must be feasible (>=t_star)
            # ensure feasible: t_fix >= t_star
            b_ub2 = t_fix - c2
            # objective sum
            S = np.asarray(Ay2.sum(axis=0)).ravel()
            # normalize scale to avoid huge/zero
            sn = np.linalg.norm(S)
            if sn > 0:
                S = S / sn
            bnds2 = [(-r, r)] * N
            res2 = linprog(S, A_ub=Ay2, b_ub=b_ub2, bounds=bnds2, method="highs",
                           options={"time_limit": 6.0})
            if res2.status == 0:
                y2 = np.asarray(res2.x, dtype=float)
                if np.all(np.isfinite(y2)):
                    xc2 = _y_to_x(x_cur, B, y2)
                    m2 = _max_cos(xc2)
                    Gc2 = xc2 @ xc2.T
                    np.fill_diagonal(Gc2, -2.0)
                    t2 = int(np.sum(Gc2[iuI_all, iuJ_all] > m2 - 1e-8))
                    cand_list.append((xc2, m2, t2))
        except Exception:
            pass

        # choose best candidate by (lower m, higher tied)
        # evaluate acceptance
        accepted = False
        # sort candidates: primary m asc, tied desc
        cand_list.sort(key=lambda z: (z[1], -z[2]))
        for (xc, mc, tc) in cand_list:
            if mc < m_cur - 1e-12:
                x_cur = xc
                m_cur = mc
                tied_cur = tc
                accepted = True
                if mc < best_m - 1e-14:
                    best_m = mc
                    best_x = xc.copy()
                    best_tied = tc
                elif abs(mc - best_m) <= 1e-12 and tc > best_tied:
                    # keep most balanced among equals for final return if equal best?
                    pass
                break
            elif mc <= m_cur + 2e-12 and tc > tied_cur:
                x_cur = xc
                m_cur = mc
                tied_cur = tc
                accepted = True
                break
        # if none accepted, continue cycling (different radius next iter)
        # also update best if somehow candidate beats best but not cur? (should have been accepted)
        # safety: track best among rejected if strictly better due to tolerance
        if not accepted:
            for (xc, mc, tc) in cand_list:
                if mc < best_m - 1e-12:
                    best_m = mc
                    best_x = xc.copy()
                    best_tied = tc
                    x_cur = xc
                    m_cur = mc
                    tied_cur = tc
                    accepted = True
                    break
        # loop; reserve last 20s handled by LOOP_END
        if time.time() >= LOOP_END:
            break

    # final renormalize
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
