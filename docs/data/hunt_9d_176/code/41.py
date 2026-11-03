import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish_deadline(x, deadline, radius_init=4e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    for i in range(n):
        M = np.eye(d)
        M[:, 0] = x[i]
        Q, _ = np.linalg.qr(M)
        # ensure first col aligns with x[i]
        if np.dot(Q[:, 0], x[i]) < 0:
            Q[:, 0] *= -1
            Q[:, 1:] *= -1
        B[i] = Q[:, 1:]
    return B

def solve():
    t0 = time.time()
    PHASE_END = t0 + 140.0
    HARD_END = t0 + 165.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_x = xb.copy()
    best_val = max_cos(best_x)
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    n, d = 176, 9
    dm = d - 1
    Ntan = n * dm
    alphas = [0.01, 0.03, 0.1]
    NK = 30  # null combos per iteration

    outer = 0
    # quick initial squeeze if time permits (small)
    # main floppy loop
    while time.time() < PHASE_END:
        outer += 1
        cur_max = max_cos(best_x)
        # active contacts with threshold max-1e-7
        G0 = best_x @ best_x.T
        iu = np.triu_indices(n, 1)
        c_all = G0[iu]
        thr = cur_max - 1e-7
        sel_idx = np.where(c_all > thr)[0]
        p = len(sel_idx)
        if p == 0:
            break
        # cap to largest 1300 to bound linear system
        if p > 1300:
            order = np.argpartition(-c_all[sel_idx], 1300)[:1300]
            sel_idx = sel_idx[order]
            p = 1300
        Iact = iu[0][sel_idx]
        Jact = iu[1][sel_idx]
        # contact counts for hub weighting
        cntc = np.zeros(n, dtype=float)
        np.add.at(cntc, Iact, 1)
        np.add.at(cntc, Jact, 1)
        # active set for change-ratio (use 1e-9 threshold set)
        thr9 = cur_max - 1e-9
        sel9 = set(np.where(c_all > thr9)[0].tolist())
        n_active9 = max(len(sel9), 1)

        # tangent bases
        try:
            B = tangent_bases(best_x)  # n x 9 x 8
        except Exception:
            break
        # build Jacobian A (p x 1408)
        try:
            A = np.zeros((p, Ntan))
            # Bi^T x_j : for each active pair
            # loop in python (p~1100) is fine
            for k in range(p):
                i = int(Iact[k]); j = int(Jact[k])
                # 8-vectors
                A[k, i*dm:(i+1)*dm] = B[i].T @ best_x[j]
                A[k, j*dm:(j+1)*dm] = B[j].T @ best_x[i]
        except Exception:
            break
        if time.time() >= PHASE_END - 0.5:
            break
        # nullspace sampling via projection: Z = Z0 - A^T (A A^T)^{-1} A Z0
        try:
            Mmat = A @ A.T
            # regularize
            diag_mean = float(np.mean(np.diag(Mmat))) if p > 0 else 1.0
            if not np.isfinite(diag_mean) or diag_mean <= 0:
                diag_mean = 1.0
            reg = 1e-10 * max(diag_mean, 1e-12)
            Mmat = Mmat + reg * np.eye(p)
            Z0 = rng.standard_normal((Ntan, NK))
            # normalize columns
            Z0 /= np.linalg.norm(Z0, axis=0, keepdims=True) + 1e-30
            W = A @ Z0  # p x NK
            # solve M Y = W for Y (p x NK)
            try:
                Y = np.linalg.solve(Mmat, W)
            except Exception:
                Y, *_ = np.linalg.lstsq(Mmat, W, rcond=None)
            Z = Z0 - A.T @ Y
            zn = np.linalg.norm(Z, axis=0, keepdims=True)
            zn[zn < 1e-12] = 1.0
            Z /= zn
            # filter tiny / nan
            if not np.all(np.isfinite(Z)):
                continue
        except Exception:
            continue
        # map null vectors to ambient tangent fields D (NK x n x 9)? build on fly
        # Precompute D fields: NK x (n,9)
        try:
            # Z is Ntan x NK -> reshape to n x dm x NK
            Zr = Z.reshape(n, dm, NK)  # Z[row i*dm:(i+1)*dm, k]
            # D[k,i,:] = B[i] @ Zr[i,:,k]
            # vectorized: D_kin = sum_m B[i,:,m]*Zr[i,m,k]
            # einsum: B: n,d,dm ; Zr: n,dm,NK -> D: n,d,NK
            Dfield = np.einsum('ndm,nmk->ndk', B, Zr)  # n x d x NK
        except Exception:
            continue
        # evaluate candidates for each alpha
        cands = []  # list of (score_tuple, k, alpha, Xc, mnew, brokenHub, violCnt)
        # for ranking need hub weights of active pairs
        hub_w = cntc[Iact] + cntc[Jact]  # p
        c_act_vals = c_all[sel_idx]  # not really needed
        for k in range(NK):
            Dk = Dfield[:, :, k]  # n x d
            # ensure tangent (numerical)
            for alpha in alphas:
                if time.time() >= PHASE_END - 0.3:
                    break
                Xc = best_x + alpha * Dk
                Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
                Gc = Xc @ Xc.T
                # full max (exclude diag)
                # quick max
                np.fill_diagonal(Gc, -2.0)
                mnew = float(np.max(Gc))
                # active-pair new values: gather via Iact,Jact
                # need G without filled diag for those pairs (they are off-diag)
                new_act = Gc[Iact, Jact]
                # broken: dropped below cur_max-1e-4? use 1e-7? use 1e-4 for significance
                broken_mask = new_act < (cur_max - 1e-4)
                broken_hub = float(np.sum(hub_w[broken_mask])) if np.any(broken_mask) else 0.0
                n_broken = int(np.count_nonzero(broken_mask))
                # violations above cur_max+0.002
                viol_thr = cur_max + 0.002
                # count violations over all pairs: use Gc max? need count; approximate via full Gc
                viol_cnt = int(np.count_nonzero(Gc > viol_thr))
                viol_amt = max(0.0, mnew - viol_thr)
                # rank key: smaller viol_cnt, smaller viol_amt, larger broken_hub, smaller mnew
                cands.append((viol_cnt, viol_amt, -broken_hub, -n_broken, mnew, k, alpha, mnew, broken_hub, viol_cnt))
            if time.time() >= PHASE_END - 0.3:
                break
        if not cands:
            continue
        cands.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
        top3 = cands[:3]
        accepted = False
        for (_vc, _va, _bh, _nb, _mn, k, alpha, _mnew2, _bhub, _vc2) in top3:
            if time.time() + 0.45 >= PHASE_END and time.time() < PHASE_END:
                # only allow polish if at least 0.45s left before phase end
                pass
            if time.time() >= PHASE_END:
                break
            Dk = Dfield[:, :, k]
            Xc = best_x + alpha * Dk
            Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
            lp_dead = min(time.time() + 2.0, PHASE_END)
            if lp_dead - time.time() < 0.35:
                break
            # rank uses second-order info already; now polish
            try:
                Xp = lp_polish_deadline(Xc, lp_dead, radius_init=0.01)
            except Exception:
                continue
            v2 = max_cos(Xp)
            if v2 < best_val - 1e-12:
                best_val = v2
                best_x = Xp.copy()
                accepted = True
                break
            elif abs(v2 - best_val) <= 1e-9:
                # check 5% contact change
                try:
                    Gp = Xp @ Xp.T
                    cp = Gp[iu]
                    m2 = float(cp.max())
                    sel9p = set(np.where(cp > m2 - 1e-9)[0].tolist())
                    inter = len(sel9.intersection(sel9p))
                    union_change = (len(sel9) + len(sel9p) - 2*inter) / max(n_active9, 1)
                    # also compare to old threshold set size
                    if union_change >= 0.05:
                        best_x = Xp.copy()
                        # keep best_val as min
                        if v2 < best_val:
                            best_val = v2
                        accepted = True
                        break
                except Exception:
                    pass
            # else reject
            if time.time() >= PHASE_END:
                break
        # if accepted improvement, continue loop (new Jacobian)
        # else continue sampling new random null combos
        # safety: check global hard end
        if time.time() >= PHASE_END:
            break

    # final reserve: one LP polish of global best
    rem = HARD_END - time.time()
    rem = min(max(rem - 0.3, 0.0), 25.0)
    if rem > 0.8:
        try:
            best_x = lp_polish_deadline(best_x, time.time() + rem, radius_init=1e-3)
        except Exception:
            pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
