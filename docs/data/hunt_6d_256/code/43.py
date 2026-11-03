import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _eval(x):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu[0], iu[1]]
    m = float(c.max())
    ties = int(np.sum(c > m - 1e-8))
    return m, ties, c

def _tangent_bases(x):
    n, d = x.shape
    dt = d - 1
    B = np.empty((n, d, dt))
    for i in range(n):
        a = x[i]
        m = int(np.argmax(np.abs(a)))
        P = np.eye(d) - np.outer(a, a)
        idx = [j for j in range(d) if j != m]
        M = P[:, idx]  # 6x5
        Q, _ = np.linalg.qr(M)
        # ensure tangent: Q^T a ~ 0 (sign irrelevant)
        B[i] = Q
    return B

def _build_lp1(x, B, I, J, csel, r, cur_max):
    p = len(I)
    n = x.shape[0]
    dt = B.shape[2]
    N = n * dt
    # coefficients Ci = B_I^T x_J , Cj = B_J^T x_I
    BI = B[I]  # p,6,5
    BJ = B[J]
    xJ = x[J]
    xI = x[I]
    Ci = np.einsum('pij,pi->pj', BI, xJ)
    Cj = np.einsum('pij,pi->pj', BJ, xI)
    rows = np.repeat(np.arange(p), 2 * dt + 1)
    # columns
    base_I = (I * dt)[:, None] + np.arange(dt)[None, :]
    base_J = (J * dt)[:, None] + np.arange(dt)[None, :]
    cols_I = base_I.ravel()
    cols_J = base_J.ravel()
    cols = np.empty(p * (2 * dt + 1), dtype=int)
    vals = np.empty(p * (2 * dt + 1), dtype=float)
    for k in range(p):
        o = k * (2 * dt + 1)
        cols[o:o+dt] = base_I[k]
        vals[o:o+dt] = Ci[k]
        cols[o+dt:o+2*dt] = base_J[k]
        vals[o+dt:o+2*dt] = Cj[k]
        cols[o+2*dt] = N
        vals[o+2*dt] = -1.0
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, N + 1)).tocsr()
    b_ub = -csel
    obj = np.zeros(N + 1)
    obj[-1] = 1.0
    bounds = [(-r, r)] * N + [(None, cur_max)]
    return A_ub, b_ub, obj, bounds, Ci, Cj

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    LOOP_END = t0 + 150.0  # reserve ~20s
    rng = np.random.default_rng(0)

    # load start
    x_cur = None
    best_val = 1e9
    best_x = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                v = float(G.max())
                if v < best_val:
                    best_val = v
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.normal(size=(256, 6))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        best_val = float(G.max())
    x_cur = best_x.copy()
    cur_max = best_val
    # current ties
    try:
        _, cur_ties, _ = _eval(x_cur)
    except Exception:
        cur_ties = 1
    best_ties = cur_ties

    n, d = x_cur.shape
    dt = d - 1
    N = n * dt
    radii_cycle = [3e-4, 1e-4, 3e-5, 1e-5]
    slack_active = 3e-4
    cyc = 0
    it = 0

    while time.time() < LOOP_END and it < 600:
        it += 1
        r = radii_cycle[cyc % len(radii_cycle)]
        cyc += 1
        if time.time() > LOOP_END - 2:
            break
        # active set
        try:
            G = x_cur @ x_cur.T
            iuI, iuJ = np.triu_indices(n, 1)
            c_all = G[iuI, iuJ]
            mval = float(c_all.max())
            cur_max = mval
            sel = c_all > (mval - slack_active)
            I0 = iuI[sel]; J0 = iuJ[sel]; c0 = c_all[sel]
            p0 = len(I0)
            if p0 == 0:
                continue
            # order by tightness, cap for LP1
            order = np.argsort(-c0)
            cap1 = 2500
            if p0 > cap1:
                order = order[:cap1]
            I = I0[order]; J = J0[order]; csel = c0[order]
        except Exception:
            continue
        # tangent bases
        try:
            B = _tangent_bases(x_cur)
        except Exception:
            continue
        # LP1: minimize t
        u1 = None; t1 = None
        try:
            A_ub, b_ub, obj, bounds, Ci, Cj = _build_lp1(x_cur, B, I, J, csel, r, cur_max)
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.status == 0:
                u1 = res.x[:-1]
                t1 = float(res.x[-1])
            else:
                continue
        except Exception:
            continue
        # evaluate LP1 step
        y1 = None; v1 = None; ties1 = None
        try:
            U1 = u1.reshape(n, dt)
            DX = np.einsum('nij,nj->ni', B, U1)
            y1 = x_cur + DX
            y1 /= np.linalg.norm(y1, axis=1, keepdims=True)
            G1 = y1 @ y1.T
            iu = np.triu_indices(n, 1)
            cc1 = G1[iu[0], iu[1]]
            v1 = float(cc1.max())
            ties1 = int(np.sum(cc1 > v1 - 1e-8))
        except Exception:
            y1 = None

        # LP2: lexicographic spread at fixed t1, minimize sum of top1500
        y2 = None; v2 = None; ties2 = None
        try:
            # top 1500 for spread
            if len(csel) > 1500:
                I2 = I[:1500]; J2 = J[:1500]; c2 = csel[:1500]
                Ci2 = Ci[:1500]; Cj2 = Cj[:1500]
            else:
                I2 = I; J2 = J; c2 = csel
                Ci2 = Ci; Cj2 = Cj
            p2 = len(I2)
            # build A for u-only
            rows2 = np.repeat(np.arange(p2), 2 * dt)
            cols2 = np.empty(p2 * 2 * dt, dtype=int)
            vals2 = np.empty(p2 * 2 * dt, dtype=float)
            base_I2 = (I2 * dt)[:, None] + np.arange(dt)[None, :]
            base_J2 = (J2 * dt)[:, None] + np.arange(dt)[None, :]
            for k in range(p2):
                o = k * 2 * dt
                cols2[o:o+dt] = base_I2[k]
                vals2[o:o+dt] = Ci2[k]
                cols2[o+dt:o+2*dt] = base_J2[k]
                vals2[o+dt:o+2*dt] = Cj2[k]
            A2 = coo_matrix((vals2, (rows2, cols2)), shape=(p2, N)).tocsr()
            t_fixed = t1  # near-optimal fixed
            b2 = t_fixed - c2
            # objective = sum of rows
            g = np.zeros(N)
            # accumulate
            # vectorized via add.at
            np.add.at(g, base_I2.ravel(), Ci2.ravel())
            np.add.at(g, base_J2.ravel(), Cj2.ravel())
            bounds2 = [(-r, r)] * N
            res2 = linprog(g, A_ub=A2, b_ub=b2, bounds=bounds2, method="highs")
            if res2.status == 0:
                u2 = res2.x
                U2 = u2.reshape(n, dt)
                DX2 = np.einsum('nij,nj->ni', B, U2)
                y2 = x_cur + DX2
                y2 /= np.linalg.norm(y2, axis=1, keepdims=True)
                G2 = y2 @ y2.T
                cc2 = G2[iu[0], iu[1]]
                v2 = float(cc2.max())
                ties2 = int(np.sum(cc2 > v2 - 1e-8))
        except Exception:
            y2 = None

        # acceptance: decrease OR more ties without increase
        improved = False
        # collect feasible candidates
        cands = []
        if y1 is not None and np.all(np.isfinite(y1)):
            cands.append((v1, ties1, y1))
        if y2 is not None and np.all(np.isfinite(y2)):
            cands.append((v2, ties2, y2))
        if not cands:
            continue
        # track global best regardless of acceptance
        for (vv, tt, yy) in cands:
            if vv < best_val - 1e-15:
                best_val = vv
                best_x = yy.copy()
                best_ties = tt
        # first try strict decrease
        dec = [(vv, tt, yy) for (vv, tt, yy) in cands if vv < cur_max - 1e-12]
        if dec:
            dec.sort(key=lambda t: t[0])
            vv, tt, yy = dec[0]
            x_cur = yy
            cur_max = vv
            cur_ties = tt
            improved = True
        else:
            # tie increase without increasing max (allow tiny tol)
            tie_cands = [(vv, tt, yy) for (vv, tt, yy) in cands if vv <= cur_max + 2e-12 and tt > cur_ties]
            if tie_cands:
                # prefer most ties, then lowest max
                tie_cands.sort(key=lambda t: (-t[1], t[0]))
                vv, tt, yy = tie_cands[0]
                x_cur = yy
                cur_max = vv
                cur_ties = tt
                improved = True
        # if not improved, just cycle to smaller radius (already cycling)
        # optional: if stuck many iters, keep best as current? do not wander
        # ensure x_cur stays normalized
        # loop continues

        if time.time() > LOOP_END:
            break

    # return best iterate
    out = best_x.copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    # final safety: ensure finite
    if not np.all(np.isfinite(out)):
        out = x_cur / np.linalg.norm(x_cur, axis=1, keepdims=True)
    return out
