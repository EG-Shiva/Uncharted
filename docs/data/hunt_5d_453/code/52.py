import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_full(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G)), G

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

def _lp_polish(X, bestX, bestV, iu_r, iu_c, deadline, max_solves, r_init):
    n = X.shape[0]
    Nvar = n * 4
    c_obj = np.zeros(Nvar + 1, dtype=np.float64)
    c_obj[-1] = 1.0
    ar4 = np.arange(4)
    radius = float(r_init)
    X = X.copy()
    for _ in range(max_solves):
        if time.time() > deadline:
            break
        if radius < 1e-12:
            break
        G = X @ X.T
        c_all = G[iu_r, iu_c]
        m = float(np.max(c_all))
        if m < bestV:
            bestV = m
            bestX = X.copy()
        slack_all = m - c_all
        tight = np.where(slack_all < 2e-5)[0]
        if tight.size == 0:
            tight = np.argsort(slack_all)[:500]
        if tight.size > 2500:
            part = np.argpartition(slack_all[tight], 2500)[:2500]
            tight = tight[part]
        I = iu_r[tight]; J = iu_c[tight]
        p = int(I.size)
        b_ub = slack_all[tight].astype(np.float64)
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
            radius *= 0.5
            continue
        rem = deadline - time.time()
        if rem < 0.6:
            break
        rem = min(rem, 8.0)
        bounds = [(-radius, radius)] * Nvar + [(None, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            continue
        try:
            U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(n, 4)
        except Exception:
            radius *= 0.5
            continue
        if not np.all(np.isfinite(U)):
            radius *= 0.5
            continue
        DX = np.einsum('ndk,nk->nd', B, U)
        Xt = X + DX
        nn = np.sqrt(np.sum(Xt * Xt, axis=1, keepdims=True))
        nn[nn == 0] = 1.0
        Xt = Xt / nn
        mt, _ = _max_cos_full(Xt)
        if mt < m - 1e-13 and np.isfinite(mt):
            X = Xt
            if mt < bestV:
                bestV = mt
                bestX = Xt.copy()
            radius = min(radius * 1.5, 5e-4)
        else:
            radius *= 0.5
    Gf = X @ X.T
    cf = Gf[iu_r, iu_c]
    mf = float(np.max(cf))
    if mf < bestV:
        bestV = mf
        bestX = X.copy()
    return X, bestX, bestV

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EXPLORE_END = t0 + 138.0
    FINAL_END = T_END - 1.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None; bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc*cc, axis=1, keepdims=True)); nn[nn==0]=1.0
            cc = cc/nn
            if not np.all(np.isfinite(cc)):
                continue
            vv,_ = _max_cos_full(cc)
            if vv < bestV:
                bestV = vv; bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n,d)); bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
        bestV,_ = _max_cos_full(bestX)
    iu_r, iu_c = np.triu_indices(n, 1)
    ar4 = np.arange(4)
    Xcur = bestX.copy()
    mcur = bestV

    # ---- second-order null-space exploration ----
    s_grid = [0.002, 0.007, 0.02, 0.06, 0.15]
    n_rounds = 0
    while time.time() < EXPLORE_END and n_rounds < 10:
        n_rounds += 1
        if time.time() > EXPLORE_END - 4.0:
            break
        try:
            G = Xcur @ Xcur.T
            c_all = G[iu_r, iu_c]
            m0 = float(np.max(c_all))
            slack = m0 - c_all
            c_idx = np.where(slack < 1e-9)[0]
            if c_idx.size < 50 or c_idx.size > 4000:
                # fallback: take tightest 1700
                c_idx = np.argsort(slack)[:1700]
                # keep only very tight? still use as contacts
                # if loosest slack too big, skip null round
                if slack[c_idx[-1]] > 1e-7:
                    c_idx = np.where(slack < 1e-9)[0]
                    if c_idx.size < 50:
                        break
            m_c = int(c_idx.size)
            Nvar = n*4
            if m_c >= Nvar - 5:
                break
            shell_mask = (slack >= 5e-7) & (slack < 1e-4)
            shell_idx = np.where(shell_mask)[0]
            if shell_idx.size == 0:
                # widen slightly
                shell_mask = (slack >= 1e-9) & (slack < 1e-4)
                shell_idx = np.where(shell_mask)[0]
                if shell_idx.size == 0:
                    break
            # sort shell by slack, cap 400
            order = np.argsort(slack[shell_idx])[:400]
            shell_idx = shell_idx[order]
            Snum = int(shell_idx.size)
            B = _tangent_bases(Xcur)
            I = iu_r[c_idx]; J = iu_c[c_idx]
            aI = np.einsum('pdk,pd->pk', B[I], Xcur[J])
            aJ = np.einsum('pdk,pd->pk', B[J], Xcur[I])
            R = np.zeros((m_c, Nvar), dtype=np.float64)
            r_rep = np.repeat(np.arange(m_c), 4)
            cI = (np.repeat(I,4)*4 + np.tile(ar4, m_c))
            cJ = (np.repeat(J,4)*4 + np.tile(ar4, m_c))
            R[r_rep, cI] = aI.ravel()
            R[r_rep, cJ] += aJ.ravel()
            # null space via complete QR of R.T
            try:
                Q, _ = np.linalg.qr(R.T, mode='complete')
            except Exception:
                break
            Kdim = Nvar - m_c
            if Kdim <= 0:
                break
            if Kdim > 300:
                Kdim = 300
                Vnull = Q[:, m_c:m_c+Kdim]
            else:
                Vnull = Q[:, m_c:]
                Kdim = int(Vnull.shape[1])
            if Kdim < 5:
                break
            # shell rows
            Is = iu_r[shell_idx]; Js = iu_c[shell_idx]
            aIs = np.einsum('pdk,pd->pk', B[Is], Xcur[Js])
            aJs = np.einsum('pdk,pd->pk', B[Js], Xcur[Is])
            RS = np.zeros((Snum, Nvar), dtype=np.float64)
            rs_rep = np.repeat(np.arange(Snum), 4)
            cIs = (np.repeat(Is,4)*4 + np.tile(ar4, Snum))
            cJs = (np.repeat(Js,4)*4 + np.tile(ar4, Snum))
            RS[rs_rep, cIs] = aIs.ravel()
            RS[rs_rep, cJs] += aJs.ravel()
            Gproj = RS @ Vnull  # S x K
            norms = np.linalg.norm(Gproj, axis=1) + 1e-30
            # build candidate w's in K-dim
            cand_ws = []
            # shell-directed: first up to 60 tightest shell
            lim = min(60, Snum)
            for q in range(lim):
                w = -Gproj[q] / norms[q]
                cand_ws.append(w)
            # random
            for _ in range(60):
                w = rng.standard_normal(Kdim); w/= (np.linalg.norm(w)+1e-30)
                cand_ws.append(w)
            # combos: mix of 3 shells + noise
            for _ in range(20):
                pick = rng.choice(Snum, size=min(3,Snum), replace=False)
                w = np.zeros(Kdim)
                for q in pick:
                    w -= Gproj[q] / norms[q]
                w += 0.5*rng.standard_normal(Kdim)
                w/= (np.linalg.norm(w)+1e-30)
                cand_ws.append(w)
            # evaluate
            best_feas_max = float("inf")
            best_feas_nc = -1
            best_feas_X = None
            best_overall_max = float("inf")
            best_overall_X = None
            old_nc = int(np.sum(slack < 1e-9))
            # precompute DX per candidate on fly
            for w in cand_ws:
                if time.time() > EXPLORE_END - 1.0:
                    break
                u = Vnull @ w  # N
                Umat = u.reshape(n, 4)
                DX = np.einsum('ndk,nk->nd', B, Umat)
                for s in s_grid:
                    Xt = Xcur + s * DX
                    nn = np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)); nn[nn==0]=1.0
                    Xt = Xt/nn
                    Gt = Xt @ Xt.T
                    ct = Gt[iu_r, iu_c]
                    mt = float(np.max(ct))
                    if mt < best_overall_max:
                        best_overall_max = mt; best_overall_X = Xt
                    # feasible if not raising beyond tol
                    if mt <= m0 + 2e-12:
                        nc = int(np.sum((mt - ct) < 1e-9))
                        if mt < best_feas_max - 1e-13 or (abs(mt-best_feas_max)<1e-13 and nc>best_feas_nc):
                            best_feas_max = mt; best_feas_nc = nc; best_feas_X = Xt
                    # early break if raising a lot and s growing will only worsen? not strictly monotonic, but break if mt > m0+1e-4
                    if mt > m0 + 2e-3:
                        break
                # end s
            # decision
            accepted = False
            if best_feas_X is not None:
                if best_feas_max < m0 - 1e-12:
                    Xcur = best_feas_X; mcur = best_feas_max
                    accepted = True
                    if mcur < bestV:
                        bestV = mcur; bestX = Xcur.copy()
                elif best_feas_nc > old_nc:
                    # new contact without raising
                    Xcur = best_feas_X; mcur = best_feas_max
                    accepted = True
                elif best_overall_max < m0 - 1e-12:
                    Xcur = best_overall_X; mcur = best_overall_max
                    accepted = True
                    if mcur < bestV:
                        bestV = mcur; bestX = Xcur.copy()
            else:
                if best_overall_max < m0 - 1e-12:
                    Xcur = best_overall_X; mcur = best_overall_max
                    accepted = True
                    if mcur < bestV:
                        bestV = mcur; bestX = Xcur.copy()
            if accepted:
                # short LP re-jam (time-capped)
                dl = min(time.time()+6.0, EXPLORE_END, T_END-32.0)
                try:
                    Xcur, bestX, bestV = _lp_polish(Xcur, bestX, bestV, iu_r, iu_c, dl, 3, 1e-4)
                    mcur, _ = _max_cos_full(Xcur)
                    # sync best
                    if mcur < bestV:
                        bestV = mcur; bestX = Xcur.copy()
                    else:
                        # keep Xcur near best if drifted worse by >5e-4 revert
                        if mcur > bestV + 5e-4:
                            Xcur = bestX.copy(); mcur = bestV
                except Exception:
                    pass
            else:
                # no feasible topology change: try small random kick in null space to escape? take tiny step along best random that minimizes max
                if best_overall_X is not None and best_overall_max <= m0 + 5e-6:
                    # take smallest-s that gave best_overall? just perturb slightly
                    pass
                # if stuck, break to final polish to save time
                if n_rounds >= 3:
                    # add tiny isotropic jitter to rattlers to diversify next round
                    try:
                        Gb = Xcur @ Xcur.T
                        np.fill_diagonal(Gb, -2.0)
                        rm = np.max(Gb, axis=1)
                        # jitter 5 lowest-rm points
                        idx = np.argsort(rm)[:5]
                        for vv in idx:
                            rr = rng.standard_normal(d); rr -= float(rr @ Xcur[vv])*Xcur[vv]
                            nnr = float(np.linalg.norm(rr))
                            if nnr < 1e-12:
                                continue
                            rr/=nnr
                            Xcur[vv] = Xcur[vv] + 1e-4*rr
                            Xcur[vv]/=np.linalg.norm(Xcur[vv])
                        mcur,_ = _max_cos_full(Xcur)
                    except Exception:
                        pass
                if n_rounds >= 6 and not accepted:
                    break
        except Exception:
            break
    # ---- final polish reserve 30s ----
    try:
        deadline = FINAL_END
        # start from best
        Xstart = bestX.copy()
        Xstart, bestX, bestV = _lp_polish(Xstart, bestX, bestV, iu_r, iu_c, deadline, 40, 1e-4)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
