import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def _get_duals(res):
    try:
        o = res.ineqlin
        try:
            m = o.marginals
            return np.asarray(m, dtype=np.float64).ravel()
        except Exception:
            pass
        try:
            m = o['marginals']
            return np.asarray(m, dtype=np.float64).ravel()
        except Exception:
            pass
    except Exception:
        pass
    try:
        m = res.get('ineqlin', {}).get('marginals', None)
        if m is not None:
            return np.asarray(m, dtype=np.float64).ravel()
    except Exception:
        pass
    return None

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    LP_CUTOFF = t0 + 158.0
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n, d = 453, 5
    Nvar = n * 4

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    iu_r, iu_c = np.triu_indices(n, 1)
    ar4 = np.arange(4)

    def lp_solve_once(X, radius, t_rem):
        G = X @ X.T
        c_all = G[iu_r, iu_c]
        m = float(np.max(c_all))
        slack_all = m - c_all
        tight = np.where(slack_all < 2e-5)[0]
        if tight.size == 0:
            tight = np.argsort(slack_all)[:500]
        if tight.size > 2500:
            part = np.argpartition(slack_all[tight], 2500)[:2500]
            tight = tight[part]
        I = iu_r[tight]; J = iu_c[tight]
        p = int(I.size)
        b_ub = slack_all[tight]
        B = _tangent_bases(X)
        aI = np.einsum('pdk,pd->pk', B[I], X[J])
        aJ = np.einsum('pdk,pd->pk', B[J], X[I])
        cols2 = np.empty((p, 9), dtype=np.int64)
        cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
        cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
        cols2[:, 8] = Nvar
        vals2 = np.empty((p, 9), dtype=np.float64)
        vals2[:, 0:4] = aI
        vals2[:, 4:8] = aJ
        vals2[:, 8] = -1.0
        rows = np.repeat(np.arange(p), 9)
        try:
            A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar + 1)).tocsr()
        except Exception:
            return None
        c_obj = np.zeros(Nvar + 1)
        c_obj[-1] = 1.0
        bounds = [(-radius, radius)] * Nvar + [(None, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs", options={"time_limit": max(0.4, min(8.0, t_rem))})
        except Exception:
            return None
        return {"res": res, "B": B, "I": I, "J": J, "b": b_ub, "m": m,
                "aI": aI, "aJ": aJ, "slack_all": slack_all, "tight": tight, "p": p}

    def apply_step(X, B, u):
        U = u.reshape(n, 4)
        DX = np.einsum('idk,ik->id', B, U)
        Xn = X + DX
        Xn /= (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-30)
        return Xn

    def lp_polish(Xs, init_radius, budget, max_solves=6):
        X = Xs.copy()
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        bX = X.copy(); bV = m0
        radius = float(init_radius)
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 6.0:
                break
            if time.time() > budget_dead[0]:
                break
            if radius < 1e-12:
                break
            t_rem = min(budget_dead[0] - time.time(), LP_CUTOFF - time.time(), T_END - 6.0 - time.time(), 8.0)
            if t_rem < 0.35:
                break
            out = lp_solve_once(X, radius, t_rem)
            if out is None:
                radius *= 0.5
                continue
            res = out["res"]
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            u = np.asarray(res.x[:-1], dtype=np.float64)
            tp = float(res.x[-1])
            if not np.all(np.finite(u)) or not np.isfinite(tp):
                radius *= 0.5
                continue
            if tp >= -1e-13:
                radius *= 0.5
                if radius < 5e-6:
                    break
                continue
            Xn = apply_step(X, out["B"], u)
            mn = _max_cos(Xn)
            mcur = _max_cos(X)
            if mn < mcur - 1e-12:
                X = Xn
                if mn < bV:
                    bV = mn; bX = Xn.copy()
                radius = min(radius * 1.3, 0.02)
            else:
                radius *= 0.45
        return bX, bV

    # rotation generator matrix in tangent coords
    def rotation_basis(B, X):
        # 10 skew pairs
        pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
        R = np.empty((Nvar, 10), dtype=np.float64)
        for k, (a, b) in enumerate(pairs):
            DX = np.zeros_like(X)
            DX[:, a] = X[:, b]
            DX[:, b] = -X[:, a]
            # tangent coords: B_i^T dx_i
            U = np.einsum('idk,id->ik', B, DX)
            R[:, k] = U.ravel()
        # orthonormalize
        Q, _ = np.linalg.qr(R)
        return Q  # Nvar x 10

    def floppy_modes_dense(A_dense, k=20):
        try:
            # dense SVD, smallest vectors = last rows of Vt
            _, S, Vt = np.linalg.svd(A_dense, full_matrices=False)
            Vs = Vt[-k:, :].T.copy()  # Nvar x k
            Ss = S[-k:].copy()
            return Vs, Ss
        except Exception:
            return None, None

    budget_dead = [LP_CUTOFF]
    curX = bestX.copy(); curV = bestV

    # initial quick LP polish to get duals baseline (short)
    try:
        budget_dead[0] = min(LP_CUTOFF, time.time() + 8.0)
        curX, curV = lp_polish(curX, 0.01, budget_dead, max_solves=5)
        if curV < bestV:
            bestV = curV; bestX = curX.copy()
    except Exception:
        pass

    amps = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    n_outer = 0
    while time.time() < LP_CUTOFF - 4.0 and time.time() < T_END - 10.0:
        n_outer += 1
        # build system + one HiGHS LP for duals
        radius0 = 0.008
        t_rem = min(LP_CUTOFF - time.time(), T_END - 10.0 - time.time(), 8.0)
        if t_rem < 0.4:
            break
        out = lp_solve_once(curX, radius0, t_rem)
        if out is None:
            # fallback shake + polish
            try:
                budget_dead[0] = min(LP_CUTOFF, time.time() + 5.0)
                nz = rng.standard_normal((n, d)) * 0.002
                nz -= np.sum(nz * curX, axis=1, keepdims=True) * curX
                Xt = curX + nz
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt, Xv = lp_polish(Xt, 0.005, budget_dead, max_solves=4)
                if Xv < curV:
                    curV = Xv; curX = Xt.copy()
                    if curV < bestV:
                        bestV = curV; bestX = curX.copy()
            except Exception:
                pass
            continue
        res = out["res"]
        B = out["B"]; I = out["I"]; J = out["J"]; m = out["m"]; p = out["p"]
        aI = out["aI"]; aJ = out["aJ"]
        duals = _get_duals(res)
        if duals is not None and duals.size == p:
            w = -duals  # positive weights
            w = np.where(np.isfinite(w), w, 0.0)
            w = np.maximum(w, 0.0)
            mx = float(np.max(w)) if w.size else 0.0
            if mx > 0:
                bbmask = w > 1e-6 * mx
            else:
                bbmask = np.zeros(p, dtype=bool)
        else:
            w = np.zeros(p)
            bbmask = np.zeros(p, dtype=bool)
        # if backbone empty, use tightest contacts as backbone
        if np.sum(bbmask) == 0:
            # use slack < 1e-9 as backbone
            sl = out["b"]
            bbmask = sl < 1e-9
            if np.sum(bbmask) == 0:
                bbmask[:min(p, 500)] = True
        Ibb = I[bbmask]; Jbb = J[bbmask]
        # build dense active matrix A_u (p x Nvar) for nullspace
        try:
            A_dense = np.zeros((p, Nvar), dtype=np.float64)
            # fill row by row (vectorized per block using loops, p~2000 fine)
            for r in range(p):
                A_dense[r, int(I[r]) * 4:int(I[r]) * 4 + 4] = aI[r]
                A_dense[r, int(J[r]) * 4:int(J[r]) * 4 + 4] = aJ[r]
        except Exception:
            continue
        # get floppy modes
        K = 20
        if p < 5 or Nvar - p < 2:
            K = min(12, Nvar - 1)
        K = min(K, max(2, min(p, Nvar) - 1))
        if K < 2:
            continue
        Vs, Ss = floppy_modes_dense(A_dense, k=K)
        if Vs is None:
            continue
        # project out global rotations
        try:
            Qr = rotation_basis(B, curX)
            Vs = Vs - Qr @ (Qr.T @ Vs)
            # renormalize columns
            nn = np.linalg.norm(Vs, axis=0, keepdims=True) + 1e-30
            Vs = Vs / nn
        except Exception:
            pass
        # trial moves
        # precompute backbone cos for penalty reference
        # evaluate modes + random combos
        dirs = []
        for k in range(Vs.shape[1]):
            dirs.append(Vs[:, k])
        # random combos
        for _ in range(14):
            c = rng.standard_normal(Vs.shape[1])
            c /= (np.linalg.norm(c) + 1e-30)
            dirs.append(Vs @ c)
        # also include zero-stress-directed: random sparse on rattlers? add rattler moves
        # find rattlers (0 contacts within 1e-9)
        try:
            Gc = curX @ curX.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(np.max(Gc))
            cnt = np.sum(Gc >= mc - 1e-9, axis=1)
            ratt = np.where(cnt == 0)[0]
        except Exception:
            ratt = np.array([], dtype=int)
        best_cand = None
        best_score = float("inf")
        best_m = float("inf")
        # score each dir/amp by exact max + backbone-break penalty
        for dv in dirs:
            U = dv.reshape(n, 4)
            DX = np.einsum('idk,ik->id', B, U)
            # normalize DX to unit rms tangent norm
            rms = float(np.sqrt(np.mean(DX * DX))) + 1e-30
            DXn = DX / rms
            # rms per-point ~? scale amps as tangent step size
            for amp in amps:
                if time.time() > LP_CUTOFF - 3.0:
                    break
                Xt = curX + amp * DXn
                Xt /= (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-30)
                # exact max
                G = Xt @ Xt.T
                np.fill_diagonal(G, -2.0)
                mn = float(np.max(G))
                # backbone break penalty
                if Ibb.size > 0:
                    cbb = np.sum(Xt[Ibb] * Xt[Jbb], axis=1)
                    broken = np.mean(cbb < m - 2e-5)
                    # mean slack of backbone
                    mslack = float(np.mean(np.maximum(0.0, m - cbb)))
                    pen = 0.002 * broken + 0.0005 * min(mslack * 1000.0, 1.0)
                else:
                    pen = 0.0
                score = mn + pen
                if score < best_score:
                    best_score = score
                    best_m = mn
                    best_cand = (dv.copy(), amp, mn, pen)
            if time.time() > LP_CUTOFF - 3.0:
                break
        if best_cand is None:
            continue
        dvb, ampb, mnb, penb = best_cand
        # accept if not raising max beyond tolerance
        tol = 2e-6 if n_outer < 8 else 5e-6
        cands_to_polish = []
        # collect top few distinct dirs that satisfy tolerance (re-scan cheap top)
        # simple: take best dir + 2 random combos with best amp re-evaluated
        # we already have best; just polish it if within tol
        if mnb <= curV + tol:
            # retracted move
            U = dvb.reshape(n, 4)
            DX = np.einsum('idk,ik->id', B, U)
            rms = float(np.sqrt(np.mean(DX * DX))) + 1e-30
            Xt0 = curX + ampb * (DX / rms)
            Xt0 /= (np.linalg.norm(Xt0, axis=1, keepdims=True) + 1e-30)
            budget_dead[0] = min(LP_CUTOFF, time.time() + 5.0)
            try:
                Xp, Vp = lp_polish(Xt0, 0.006, budget_dead, max_solves=4)
            except Exception:
                Xp, Vp = Xt0, mnb
            if Vp < curV - 1e-12:
                curV = Vp; curX = Xp.copy()
                if curV < bestV:
                    bestV = curV; bestX = curX.copy()
            elif Vp < curV + 5e-7 and ampb >= 3e-3:
                # topology change without gain: accept sideways to new basin occasionally
                if rng.random() < 0.3:
                    curX = Xp.copy(); curV = Vp
            else:
                # try second-best: random larger combo forced polish to force topology change
                pass
        else:
            # no floppy direction keeps max: force a backbone-preserving larger jump + polish
            try:
                c = rng.standard_normal(Vs.shape[1])
                c /= (np.linalg.norm(c) + 1e-30)
                dv2 = Vs @ c
                U2 = dv2.reshape(n, 4)
                DX2 = np.einsum('idk,ik->id', B, U2)
                rms2 = float(np.sqrt(np.mean(DX2 * DX2))) + 1e-30
                Xt0 = curX + 0.01 * (DX2 / rms2)
                # also relocate one rattler to hole if exists
                if ratt.size > 0:
                    # largest hole
                    Cc = rng.standard_normal((4000, d))
                    Cc /= (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-30)
                    D = Cc @ curX.T
                    md = np.max(D, axis=1)
                    h = Cc[int(np.argmin(md))]
                    ri = int(ratt[int(rng.integers(0, len(ratt)))])
                    Xt0[ri] = h
                    Xt0 /= (np.linalg.norm(Xt0, axis=1, keepdims=True) + 1e-30)
                budget_dead[0] = min(LP_CUTOFF, time.time() + 5.0)
                Xp, Vp = lp_polish(Xt0, 0.008, budget_dead, max_solves=4)
                if Vp < curV:
                    curV = Vp; curX = Xp.copy()
                    if curV < bestV:
                        bestV = curV; bestX = curX.copy()
                elif Vp < curV + 8e-6 and rng.random() < 0.25:
                    curX = Xp.copy(); curV = Vp
            except Exception:
                pass
        # keep global best
        if curV < bestV:
            bestV = curV; bestX = curX.copy()

    # final polish on global best
    try:
        budget_dead[0] = min(t0 + 170.0, LP_CUTOFF + 10.0, T_END - 1.0)
        # allow beyond LP_CUTOFF for final: inline loop with extended deadline
        X = bestX.copy()
        radius = 0.008
        for _ in range(8):
            if time.time() > T_END - 1.5:
                break
            t_rem = min(T_END - 1.5 - time.time(), 10.0)
            if t_rem < 0.4:
                break
            out = lp_solve_once(X, radius, t_rem)
            if out is None:
                radius *= 0.5
                continue
            res = out["res"]
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            u = np.asarray(res.x[:-1], dtype=np.float64)
            tp = float(res.x[-1])
            if not np.all(np.finite(u)):
                radius *= 0.5
                continue
            if tp >= -1e-13:
                radius *= 0.5
                if radius < 5e-6:
                    break
                continue
            Xn = apply_step(X, out["B"], u)
            mn = _max_cos(Xn)
            if mn < _max_cos(X) - 1e-12:
                X = Xn
                if mn < bestV:
                    bestV = mn; bestX = Xn.copy()
                radius = min(radius * 1.3, 0.02)
            else:
                radius *= 0.45
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
