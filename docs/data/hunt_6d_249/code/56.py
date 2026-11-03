import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(1)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy()
            v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = (v / nv).reshape(6, 1)
                H = E - 2.0 * (w @ w.T)
                T[i] = H[:, 1:]
        return T

    def one_lp(X, slack=2e-3, radius=2e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        gfull = G[I0, J0]
        sel = np.where(gfull > m - slack)[0]
        if len(sel) == 0:
            sel = np.argsort(gfull)[-500:]
        if len(sel) > 4000:
            part = np.argpartition(gfull[sel], -4000)[-4000:]
            sel = sel[part]
        Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
        K = len(sel)
        T = tangent_bases(X)
        TI = T[Ic]; TJ = T[Jc]
        XJ = X[Jc]; XI = X[Ic]
        Ai = np.sum(TI * XJ[:, :, None], axis=1)
        Aj = np.sum(TJ * XI[:, :, None], axis=1)
        Nv = n * 5
        r1 = np.repeat(np.arange(K), 5)
        rows = np.concatenate([r1, r1, np.arange(K)])
        cols = np.concatenate([(Ic[:, None]*5+np.arange(5)).ravel(),
                               (Jc[:, None]*5+np.arange(5)).ravel(),
                               np.full(K, Nv)])
        data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
        try:
            Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv+1)).tocsr()
        except Exception:
            return None, None, None, m
        bub = -cc
        cvec = np.zeros(Nv+1); cvec[Nv] = 1.0
        bnds = [(-radius, radius)]*Nv + [(None, None)]
        try:
            res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                          options={'presolve': True})
        except Exception:
            return None, None, None, m
        if res.status != 0 or res.x is None:
            return None, None, None, m
        dual = None
        try:
            marg = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
            if marg is not None:
                dual = -np.asarray(marg, dtype=float).ravel()
                dual = np.maximum(dual, 0.0)
        except Exception:
            dual = None
        return dual, (Ic, Jc, cc), float(res.x[Nv]), m

    def lp_jam(Xinit, deadline, radius_init=2e-3, slack=2e-3, max_it=25):
        nonlocal bestX, bestVal
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        if mcur < bestVal:
            bestVal = mcur; bestX = X.copy()
        radius = float(radius_init)
        it = 0; fails = 0
        Nv = n*5
        while it < max_it and time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - slack)[0]
            if len(sel) == 0:
                sel = np.argsort(gfull)[-500:]
            if len(sel) > 4000:
                part = np.argpartition(gfull[sel], -4000)[-4000:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]; TJ = T[Jc]
            XJ = X[Jc]; XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            r1 = np.repeat(np.arange(K), 5)
            rows = np.concatenate([r1, r1, np.arange(K)])
            cols = np.concatenate([(Ic[:, None]*5+np.arange(5)).ravel(),
                                   (Jc[:, None]*5+np.arange(5)).ravel(),
                                   np.full(K, Nv)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data, (rows, cols)), shape=(K, Nv+1)).tocsr()
            except Exception:
                radius *= 0.5; continue
            bub = -cc
            cvec = np.zeros(Nv+1); cvec[Nv] = 1.0
            bnds = [(-radius, radius)]*Nv + [(None, None)]
            try:
                res = linprog(cvec, A_ub=Aub, b_ub=bub, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                radius *= 0.5; fails += 1
                if fails > 5: break
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5; fails += 1
                if fails > 8: break
                continue
            fails = 0
            y = res.x[:Nv]
            Y = y.reshape(n, 5)
            V = np.einsum('ijk,ik->ij', T, Y)
            Xt = X + V
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            vnew = _max_cos(Xt)
            if vnew < mcur - 1e-12:
                X = Xt; mcur = vnew
                if vnew < bestVal:
                    bestVal = vnew; bestX = Xt.copy()
                radius = min(radius*1.4, 5e-3)
            else:
                radius *= 0.5
        return X, mcur

    def subset_opt(Yinit, F, deadline, taus=(0.015, 0.006, 0.0025), iters=(120, 120, 200)):
        k = Yinit.shape[0]
        nf = F.shape[0]
        # precompute self pair indices
        SI, SJ = np.triu_indices(k, 1)
        ns = len(SI)
        x = Yinit.ravel().copy()
        creg = 0.08
        for tau, mi in zip(taus, iters):
            if time.time() > deadline:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                Y = xf.reshape(k, 6)
                r = np.sqrt(np.sum(Y*Y, axis=1))
                r = np.maximum(r, 1e-12)
                Yn = Y / r[:, None]
                Cmat = Yn @ F.T  # k x nf
                Gself = Yn @ Yn.T
                gs = Gself[SI, SJ]
                g_all_max = -2.0
                # need global max for shift: compute quickly
                gm1 = float(np.max(Cmat)) if Cmat.size else -2.0
                gm2 = float(np.max(gs)) if gs.size else -2.0
                gm = gm1 if gm1 > gm2 else gm2
                e_cross = np.exp((Cmat - gm)/tau)
                e_self = np.exp((gs - gm)/tau) if ns>0 else np.zeros(0)
                S = float(np.sum(e_cross) + np.sum(e_self))
                Lsoft = gm + tau*np.log(max(S,1e-300))
                w_cross = e_cross / S
                w_self = e_self / S if ns>0 else e_self
                # gradient on Yn
                A = w_cross @ F  # k,6
                if ns>0:
                    Ws = np.zeros((k,k))
                    Ws[SI, SJ] = w_self
                    Ws = Ws + Ws.T
                    A = A + Ws @ Yn
                proj = np.sum(A*Yn, axis=1)
                G0 = (A - proj[:,None]*Yn)/r[:,None]
                diff = r - 1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:,None]*Yn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, x, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol':1e-14, 'gtol':1e-7, 'maxcor':8})
                x = res.x
            except Exception:
                break
            # renormalize to keep scale sane
            Ym = x.reshape(k,6)
            rn = np.maximum(np.linalg.norm(Ym,axis=1,keepdims=True),1e-12)
            x = (Ym/rn).ravel()
        Yo = x.reshape(k,6)
        Yo /= np.maximum(np.linalg.norm(Yo,axis=1,keepdims=True),1e-12)
        return Yo

    # initial jam to get to local optimum + duals
    try:
        if time.time() < T_END - 60:
            lp_jam(bestX.copy(), min(T_END-50, time.time()+25), radius_init=2e-3, slack=2e-3, max_it=35)
    except Exception:
        pass

    # main stress-surgery loop
    round_id = 0
    # vary hub counts / thresholds across rounds
    hub_takes = [2, 3, 2, 1, 3, 2, 3, 1, 2, 4]
    tight_tols = [1e-9, 3e-9, 1e-9, 5e-9, 2e-9, 1e-9, 1e-8, 1e-9, 3e-9, 2e-9]
    while time.time() < T_END - 18 and round_id < 12:
        round_id += 1
        try:
            # get dual stresses at current best
            dual, pack, _, mcur = one_lp(bestX, slack=2e-3, radius=2e-3)
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            ttol = tight_tols[(round_id-1) % len(tight_tols)]
            tsel = np.where(gfull > m - ttol)[0]
            TIc = I0[tsel]; TJc = J0[tsel]
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, TIc, 1); np.add.at(deg, TJc, 1)
            stress = np.zeros(n)
            if dual is not None and pack is not None:
                Ic, Jc, cc = pack
                # clip dual
                for q in range(len(Ic)):
                    w = float(dual[q]) if q < len(dual) else 0.0
                    if w > 0 and np.isfinite(w):
                        stress[int(Ic[q])] += w
                        stress[int(Jc[q])] += w
                # if dual all zero, fallback to degree-weighted
                if np.sum(stress) <= 0:
                    stress = deg.astype(float)
            else:
                stress = deg.astype(float)
            ratt = np.where(deg == 0)[0]
            # hubs with >=15 contacts sorted by stress
            hubs = np.where(deg >= 15)[0]
            if len(hubs) == 0:
                hubs = np.argsort(-stress)[:5]
            else:
                hubs = hubs[np.argsort(-stress[hubs])]
            nhub = hub_takes[(round_id-1) % len(hub_takes)]
            # rotate hub choice by round to visit different neighborhoods
            shift = (round_id-1) % max(1,len(hubs))
            hubs_rolled = np.roll(hubs, -shift)
            sel_hubs = hubs_rolled[:min(nhub, len(hubs_rolled))].tolist()
            # neighbor map from tight contacts
            neigh = set()
            adj = {h: set() for h in sel_hubs}
            for a, b in zip(TIc.tolist(), TJc.tolist()):
                if a in adj:
                    adj[a].add(b); neigh.add(b)
                if b in adj:
                    adj[b].add(a); neigh.add(a)
            mov = set(int(x) for x in ratt.tolist()) | set(int(x) for x in sel_hubs) | set(int(x) for x in neigh)
            # ensure frozen max is broken: if frozen-frozen still at m, add endpoints
            # expand / trim to 20-35
            mov = list(mov)
            # score remaining by stress
            if len(mov) < 20:
                order = np.argsort(-stress)
                for idx in order:
                    if int(idx) not in mov:
                        mov.append(int(idx))
                    if len(mov) >= 24:
                        break
            if len(mov) > 35:
                # keep rattlers+hubs, trim neighbors by stress
                keep_must = set(int(x) for x in ratt.tolist()) | set(int(x) for x in sel_hubs)
                rest = [x for x in mov if x not in keep_must]
                rest.sort(key=lambda x: -stress[x])
                mov = list(keep_must) + rest[:35-len(keep_must)]
            mov = sorted(set(int(x) for x in mov))
            # break frozen max: add max-pair endpoints until frozen max < m-1e-7 or size 35
            keep_mask = np.ones(n, dtype=bool); keep_mask[mov] = False
            # compute frozen max pairs
            # quick: G submatrix
            Gkeep = G[np.ix_(np.where(keep_mask)[0], np.where(keep_mask)[0])]
            np.fill_diagonal(Gkeep, -2.0)
            mf = float(np.max(Gkeep)) if Gkeep.size else -2.0
            cand_add_order = np.argsort(-stress)
            ptr = 0
            keep_idx = np.where(keep_mask)[0]
            while mf > m - 1e-7 and len(mov) < 35:
                # find a max pair in frozen
                ii, jj = np.unravel_index(int(np.argmax(Gkeep)), Gkeep.shape)
                a = int(keep_idx[ii]); b = int(keep_idx[jj])
                # add higher-stress one
                add = a if stress[a] > stress[b] else b
                if add not in mov:
                    mov.append(add); mov = sorted(set(mov))
                    keep_mask = np.ones(n, dtype=bool); keep_mask[mov] = False
                    keep_idx = np.where(keep_mask)[0]
                    if len(keep_idx) < 2:
                        break
                    Gkeep = G[np.ix_(keep_idx, keep_idx)]
                    np.fill_diagonal(Gkeep, -2.0)
                    mf = float(np.max(Gkeep))
                else:
                    # add the other
                    other = b if add == a else a
                    if other not in mov:
                        mov.append(other); mov = sorted(set(mov))
                        keep_mask = np.ones(n, dtype=bool); keep_mask[mov] = False
                        keep_idx = np.where(keep_mask)[0]
                        if len(keep_idx) < 2:
                            break
                        Gkeep = G[np.ix_(keep_idx, keep_idx)]
                        np.fill_diagonal(Gkeep, -2.0)
                        mf = float(np.max(Gkeep))
                    else:
                        break
                ptr += 1
                if ptr > 10:
                    break
            k = len(mov)
            if k < 15 or k > 40:
                continue
            keep_idx = np.where(~np.isin(np.arange(n), np.array(mov)))[0]
            F = bestX[keep_idx].copy()
            nf = F.shape[0]
            # refill pool >=100k
            C = 100000
            # generate pool in one go (deterministic stream)
            pool = rng.standard_normal(size=(C, 6))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            # score vs frozen in chunks
            curmax = np.empty(C, dtype=float)
            CH = 20000
            for s in range(0, C, CH):
                e = min(C, s+CH)
                D = pool[s:e] @ F.T
                curmax[s:e] = np.max(D, axis=1)
            used = np.zeros(C, dtype=bool)
            # greedy deep holes
            new_pts = []
            # mask used as inf
            for _ in range(k):
                bi = int(np.argmin(np.where(used, np.inf, curmax)))
                used[bi] = True
                pt = pool[bi]
                new_pts.append(pt.copy())
                dd = pool @ pt
                # update
                np.maximum(curmax, dd, out=curmax)
                curmax[used] = np.inf
                if time.time() > T_END - 15:
                    break
            new_pts = np.stack(new_pts, axis=0)
            # subset softmax optimize (movable only)
            sub_dead = min(T_END - 12, time.time() + 12)
            Yopt = subset_opt(new_pts, F, sub_dead)
            Xcand = bestX.copy()
            Xcand[mov] = Yopt
            Xcand /= np.maximum(np.linalg.norm(Xcand, axis=1, keepdims=True), 1e-12)
            mcand_pre = _max_cos(Xcand)
            # short global LP jam on candidate
            jam_dead = min(T_END - 10, time.time() + 14)
            Xj, vj = lp_jam(Xcand, jam_dead, radius_init=1.2e-3, slack=2e-3, max_it=18)
            # lp_jam already updates bestX/bestVal if improved; nothing else needed
            # if no improvement, continue to next stressed neighborhood
        except Exception:
            continue

    # final polish on best
    try:
        if time.time() < T_END - 4:
            lp_jam(bestX.copy(), T_END - 1.0, radius_init=5e-4, slack=2e-3, max_it=18)
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
