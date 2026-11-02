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

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    # reserve final 30s for polish/return
    EXPLORE_END = t0 + 140.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    # load
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
            nn = np.sqrt(np.sum(cc*cc, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            cc = cc / nn
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

    def lp_polish(X, cur_best_X, cur_best_V, max_solves, deadline):
        X = X.copy()
        bX = cur_best_X.copy()
        bV = float(cur_best_V)
        radius = 1e-4
        Nvar = n * 4
        c_obj = np.zeros(Nvar + 1)
        c_obj[-1] = 1.0
        ar4 = np.arange(4)
        for _ in range(max_solves):
            if time.time() > deadline or time.time() > T_END - 6.0:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            if m < bV:
                bV = m
                bX = X.copy()
            slack_all = m - c_all
            tight = np.where(slack_all < 2e-5)[0]
            if tight.size == 0:
                tight = np.argsort(slack_all)[:500]
            if tight.size > 2500:
                part = np.argpartition(slack_all[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]; J = iu_c[tight]
            p = I.size
            b_ub = slack_all[tight]
            B = _tangent_bases(X)
            aI = np.einsum('pdk,pd->pk', B[I], X[J])
            aJ = np.einsum('pdk,pd->pk', B[J], X[I])
            cols2 = np.empty((p, 9), dtype=np.int64)
            cols2[:, 0:4] = I[:, None]*4 + ar4[None, :]
            cols2[:, 4:8] = J[:, None]*4 + ar4[None, :]
            cols2[:, 8] = Nvar
            vals2 = np.empty((p, 9), dtype=np.float64)
            vals2[:, 0:4] = aI
            vals2[:, 4:8] = aJ
            vals2[:, 8] = -1.0
            rows = np.repeat(np.arange(p), 9)
            try:
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar+1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(deadline - time.time(), T_END - 6.0 - time.time(), 9.0)
            if rem < 0.5:
                break
            bounds = [(-radius, radius)]*Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            try:
                U = np.asarray(res.x[:Nvar]).reshape(n, 4)
            except Exception:
                radius *= 0.5
                continue
            if not np.all(np.isfinite(U)):
                radius *= 0.5
                continue
            DX = np.einsum('ndk,nk->nd', B, U)
            Xt = X + DX
            Xt /= (np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True)) + 1e-30)
            mt = _max_cos(Xt)
            if mt < m - 1e-13 and np.isfinite(mt):
                X = Xt
                if mt < bV:
                    bV = mt
                    bX = Xt.copy()
                radius = min(radius*1.5, 5e-4)
            else:
                radius *= 0.5
        return bX, bV, X

    # quick initial tiny polish to get accurate contact set (2 solves)
    try:
        bestX, bestV, Xcur = lp_polish(bestX, bestX, bestV, 3, min(EXPLORE_END, T_END-40))
    except Exception:
        Xcur = bestX.copy()

    Xbase = bestX.copy()
    mbase = float(bestV)

    # === second-order null-space exploration ===
    try:
        # build rigidity matrix for contacts slack<1e-9
        G = Xbase @ Xbase.T
        c_all = G[iu_r, iu_c]
        m0 = float(np.max(c_all))
        slack_all = m0 - c_all
        c_idx = np.where(slack_all < 1e-9)[0]
        # cap to tightest 2000
        if c_idx.size > 2000:
            c_idx = c_idx[np.argsort(slack_all[c_idx])[:2000]]
        # shell 5e-7 to 1e-4
        s_idx = np.where((slack_all >= 5e-7) & (slack_all <= 1e-4))[0]
        if s_idx.size > 1200:
            # keep smallest slack
            s_idx = s_idx[np.argsort(slack_all[s_idx])[:1200]]
        Ic = iu_r[c_idx]; Jc = iu_c[c_idx]
        Is = iu_r[s_idx]; Js = iu_c[s_idx]
        B0 = _tangent_bases(Xbase)
        Nvar = n*4
        p = Ic.size
        # build R contact (p x Nvar) dense? p~1700, Nvar 1812 -> ~3M floats ok
        R = np.zeros((p, Nvar), dtype=np.float64)
        if p > 0:
            aI = np.einsum('pdk,pd->pk', B0[Ic], Xbase[Jc])
            aJ = np.einsum('pdk,pd->pk', B0[Jc], Xbase[Ic])
            # fill via advanced indexing
            # row-wise: use loops in blocks for speed/memory
            for k in range(p):
                R[k, Ic[k]*4:Ic[k]*4+4] = aI[k]
                R[k, Jc[k]*4:Jc[k]*4+4] = aJ[k]
        # null space via eigh of R^T R
        K = 100
        if p == 0:
            Null = np.eye(Nvar, min(K, Nvar))
        else:
            # M = R.T @ R
            M = R.T @ R
            # symmetrize
            M = 0.5*(M+M.T)
            w_all, V_all = np.linalg.eigh(M)
            # smallest K
            Null = V_all[:, :K]
            # free mem
            del M
        # projected shell matrix S = R_shell @ Null
        q = Is.size
        Sproj = None
        if q > 0:
            # build R_shell on the fly: R_shell (q x Nvar) then multiply
            # to save memory do block
            Rs = np.zeros((q, Nvar), dtype=np.float64)
            aIs = np.einsum('pdk,pd->pk', B0[Is], Xbase[Js])
            aJs = np.einsum('pdk,pd->pk', B0[Js], Xbase[Is])
            for k in range(q):
                Rs[k, Is[k]*4:Is[k]*4+4] = aIs[k]
                Rs[k, Js[k]*4:Js[k]*4+4] = aJs[k]
            Sproj = Rs @ Null  # q x K
            del Rs
        # candidate directions in w-space
        candW = []
        # random gaussians
        for _ in range(60):
            wv = rng.standard_normal(K)
            nrm = np.linalg.norm(wv)+1e-30
            candW.append(wv/nrm)
        # shell-targeted: rows of Sproj (most promising = largest norm rows among smallest slack)
        if Sproj is not None and q > 0:
            rnorm = np.linalg.norm(Sproj, axis=1)+1e-30
            # order by slack (s_idx already sorted) take first 80
            lim = min(q, 80)
            for k in range(lim):
                v = Sproj[k]
                nv = np.linalg.norm(v)+1e-30
                if nv < 1e-14:
                    continue
                candW.append(v/nv)
            # combos: sum of top-2, top-3 etc random pairs
            for _ in range(40):
                a = rng.integers(0, lim)
                b = rng.integers(0, lim)
                v = Sproj[a]+Sproj[b]
                nv = np.linalg.norm(v)+1e-30
                if nv < 1e-14:
                    continue
                candW.append(v/nv)
        # convert each w to full U0 (max-norm normalized)
        # precompute Null->U mapping per candidate on the fly
        steps = [3e-4, 8e-4, 1.6e-3, 3e-3]
        best_trial_X = None
        best_trial_V = mbase
        best_new_contacts = int(np.sum(slack_all < 1e-9))
        # helper to retract
        for wi, wv in enumerate(candW):
            if time.time() > EXPLORE_END - 5.0:
                break
            u = Null @ wv  # Nvar
            U0 = u.reshape(n, 4)
            # normalize so max per-point tangent norm =1
            pn = np.sqrt(np.sum(U0*U0, axis=1))
            mx = float(np.max(pn))
            if mx < 1e-14 or not np.isfinite(mx):
                continue
            U0 = U0 / mx
            DX0 = np.einsum('ndk,nk->nd', B0, U0)
            # trial retractions at several step sizes; estimate curvature
            for st in steps:
                if time.time() > EXPLORE_END - 2.0:
                    break
                Xt = Xbase + st * DX0
                Xt /= (np.sqrt(np.sum(Xt*Xt, axis=1, keepdims=True))+1e-30)
                if not np.all(np.isfinite(Xt)):
                    continue
                Gt = Xt @ Xt.T
                np.fill_diagonal(Gt, -2.0)
                mt = float(np.max(Gt))
                if not np.isfinite(mt):
                    continue
                # curvature estimate not needed explicitly; acceptance rule:
                # (a) lowers true max, or (b) new contact without raising
                if mt < mbase - 1e-12:
                    # improvement: keep best
                    if mt < best_trial_V:
                        best_trial_V = mt
                        best_trial_X = Xt.copy()
                elif mt <= mbase + 1e-12:
                    # count contacts w.r.t. mbase threshold (absolute)
                    # new contact if a shell pair now within 1e-9 of mbase
                    # quick check: max over shell pairs
                    # compute shell cosines for this Xt
                    if q > 0:
                        cs = np.sum(Xt[Is]*Xt[Js], axis=1)
                        cmax_shell = float(np.max(cs)) if cs.size else -2.0
                        # creates new contact if shell max >= mbase-1e-9 and previously < mbase-1e-9
                        # previously shell max = m0 - min_slack_shell
                        if cmax_shell >= mbase - 1e-9:
                            # verify true max not raised and contact count increased
                            cnt = int(np.sum((mbase - (Gt[iu_r, iu_c])) < 1e-9)) if False else 0
                            # approximate: accept as topology change candidate
                            # prefer larger shell increase (closer to contact)
                            # store if better than current best (which hasn't improved max)
                            if best_trial_X is None and cmax_shell > (mbase - 5e-7):
                                best_trial_X = Xt.copy()
                                best_trial_V = mt
                                break
                    # else no shell, skip
            # if we found strict improvement, we can break early to polish
            if best_trial_X is not None and best_trial_V < mbase - 1e-12:
                break
        # if accepted step found, short LP polish it
        if best_trial_X is not None:
            # check acceptance: either lowers or creates contact without raising
            mt0 = _max_cos(best_trial_X)
            accept = False
            if mt0 < mbase - 1e-12:
                accept = True
            elif mt0 <= mbase + 1e-12:
                # verify new contact: count pairs within 1e-9 of mbase
                Gb0 = Xbase @ Xbase.T
                np.fill_diagonal(Gb0, -2.0)
                # use absolute threshold mbase-1e-9
                c0 = int(np.sum(Gb0 >= mbase - 1e-9)) // 1
                Gt2 = best_trial_X @ best_trial_X.T
                np.fill_diagonal(Gt2, -2.0)
                c1 = int(np.sum(Gt2 >= mbase - 1e-9)) // 1
                if c1 > c0:
                    accept = True
                else:
                    # also accept if shell max moved up significantly without raising max
                    accept = True  # allow polish to decide; still topology change
                    # but require not worse
                    if mt0 > mbase + 5e-10:
                        accept = False
            if accept:
                # short time-capped LP polish from trial point
                deadline = min(EXPLORE_END + 10.0, T_END - 32.0)
                try:
                    bX2, bV2, _ = lp_polish(best_trial_X, bestX if bestV < mt0 else best_trial_X, min(bestV, mt0), 8, deadline)
                    if bV2 < bestV:
                        bestX = bX2.copy()
                        bestV = float(bV2)
                except Exception:
                    pass
    except Exception:
        pass

    # === final 30s: polish/return incumbent ===
    try:
        deadline = T_END - 1.0
        # determine remaining solves based on time
        bestX, bestV, _ = lp_polish(bestX, bestX, bestV, 12, deadline)
    except Exception:
        pass
    bestX = bestX / (np.sqrt(np.sum(bestX*bestX, axis=1, keepdims=True))+1e-30)
    return bestX
