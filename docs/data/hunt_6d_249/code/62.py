import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RESERVE = 22.0
    rng = np.random.default_rng(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
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

    from scipy.optimize import minimize, linprog
    from scipy.sparse import coo_matrix

    def tangent_bases(X):
        T = np.empty((n, 6, 5), dtype=float)
        E = np.eye(6)
        for i in range(n):
            a = X[i]
            s = -1.0 if a[0] >= 0 else 1.0
            v = a.copy(); v[0] -= s
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                T[i] = E[:, 1:]
            else:
                w = (v / nv).reshape(6, 1)
                H = E - 2.0 * (w @ w.T)
                T[i] = H[:, 1:]
        return T

    def lp_jam(Xinit, deadline, max_it=12, radius_init=2e-3, slack=1.5e-3):
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        mcur = _max_cos(X)
        radius = float(radius_init)
        Nv = n * 5
        sl = float(slack)
        it = 0
        fails = 0
        while it < max_it and time.time() < deadline:
            it += 1
            if radius < 1e-12:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            gfull = G[I0, J0]
            sel = np.where(gfull > m - sl)[0]
            if len(sel) < 800:
                sel = np.argsort(gfull)[-1200:]
            if len(sel) > 3200:
                part = np.argpartition(gfull[sel], -3200)[-3200:]
                sel = sel[part]
            Ic = I0[sel]; Jc = J0[sel]; cc = gfull[sel]
            K = len(sel)
            T = tangent_bases(X)
            TI = T[Ic]; TJ = T[Jc]
            XJ = X[Jc]; XI = X[Ic]
            Ai = np.sum(TI * XJ[:, :, None], axis=1)
            Aj = np.sum(TJ * XI[:, :, None], axis=1)
            cols = np.concatenate([(Ic[:, None]*5+np.arange(5)).ravel(),
                                   (Jc[:, None]*5+np.arange(5)).ravel(),
                                   np.full(K, Nv)])
            rows = np.concatenate([np.repeat(np.arange(K),5), np.repeat(np.arange(K),5), np.arange(K)])
            data = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(K)])
            try:
                Aub = coo_matrix((data,(rows,cols)),shape=(K,Nv+1)).tocsr()
            except Exception:
                radius *= 0.5; continue
            bub = -cc
            cvec = np.zeros(Nv+1); cvec[Nv]=1.0
            bnds = [(-radius,radius)]*Nv+[(None,None)]
            try:
                res = linprog(cvec,A_ub=Aub,b_ub=bub,bounds=bnds,method='highs',options={'presolve':True})
            except Exception:
                radius*=0.5; fails+=1
                if fails>4: break
                continue
            if res.status!=0 or res.x is None:
                radius*=0.5; fails+=1
                if fails>6: break
                continue
            fails=0
            y = res.x[:Nv].reshape(n,5)
            tpred=float(res.x[Nv])
            step=np.einsum('nkm,nm->nk',T,y)
            Xn=X+step
            Xn/=np.maximum(np.linalg.norm(Xn,axis=1,keepdims=True),1e-12)
            mnew=_max_cos(Xn)
            if mnew < m+1e-12:
                X=Xn; mcur=min(mcur,mnew)
                if tpred < m-1e-9:
                    radius=min(radius*1.3,6e-3)
                else:
                    radius*=0.7
                if it%6==0:
                    sl=max(sl*0.7,3e-4)
            else:
                radius*=0.5
                if radius<2e-4:
                    break
        return X

    def contact_stress(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # degrees with tight threshold 1e-9 (contacts)
        deg = np.sum(G > m - 1e-9, axis=1).astype(int)
        # dual-stress proxy: softmax over near-active band (LP dual approx)
        g = G[I0, J0]
        band = 2e-3
        msel = g > m - band
        # also try to get LP dual if cheap: use softmax weights
        tau = 1.5e-4
        gs = g[msel]
        Is = I0[msel]; Js = J0[msel]
        e = np.exp((gs - m) / tau)
        S = float(np.sum(e)) + 1e-300
        w = e / S
        stress = np.zeros(n, dtype=float)
        np.add.at(stress, Is, w)
        np.add.at(stress, Js, w)
        # neighbor lists with looser threshold for surgery
        neigh = [np.where(G[i] > m - 1e-6)[0].tolist() for i in range(n)]
        return m, deg, stress, neigh, G

    def select_movable(deg, stress, neigh, k_target):
        rattlers = np.where(deg == 0)[0].tolist()
        # hubs with >=15 contacts, else top degree
        hubs15 = np.where(deg >= 15)[0]
        if len(hubs15) == 0:
            hubs15 = np.where(deg >= 12)[0]
            if len(hubs15) == 0:
                hubs15 = np.argsort(deg)[-6:]
        # order hubs by stress
        order = np.argsort(stress[hubs15])[::-1]
        hubs_sorted = hubs15[order]
        # randomize number of hubs to attack
        nh = int(rng.integers(2, 5))
        nh = min(nh, len(hubs_sorted))
        # sometimes pick random high-stress hub outside top for diversity
        chosen_hubs = hubs_sorted[:nh].tolist()
        if rng.random() < 0.4:
            top20 = np.argsort(stress)[-20:]
            cand = int(rng.choice(top20))
            if cand not in chosen_hubs:
                chosen_hubs.append(cand)
        mov = set(rattlers)
        for h in chosen_hubs:
            mov.add(int(h))
        # add contact neighbors of chosen hubs
        for h in chosen_hubs:
            nb = neigh[int(h)][:]
            rng.shuffle(nb)
            # take up to 6 neighbors per hub
            take = min(len(nb), int(rng.integers(4, 7)))
            for v in nb[:take]:
                mov.add(int(v))
            # also add second-ring with small prob to reach size
        # fill to k_target with highest stress points
        order_all = np.argsort(stress)[::-1]
        for v in order_all:
            if len(mov) >= k_target:
                break
            mov.add(int(v))
        mov = list(mov)
        # trim if too many: keep rattlers+hubs, drop lowest stress others
        if len(mov) > k_target + 2:
            keep = set(rattlers) | set(chosen_hubs)
            rest = [v for v in mov if v not in keep]
            rest_sorted = sorted(rest, key=lambda v: stress[v], reverse=True)
            need = k_target - len(keep)
            if need < 0:
                # keep top stressed among keep+rest
                all_sorted = sorted(mov, key=lambda v: stress[v], reverse=True)
                mov = all_sorted[:k_target]
            else:
                mov = list(keep) + rest_sorted[:need]
        rng.shuffle(mov)
        # enforce 20-35
        if len(mov) < 18:
            for v in order_all:
                if int(v) not in mov:
                    mov.append(int(v))
                if len(mov) >= 20:
                    break
        if len(mov) > 36:
            mov = mov[:36]
        return np.asarray(mov, dtype=int)

    def greedy_refill(Xfrozen, k, pool_size=100000):
        F = Xfrozen.shape[0]
        Cp = rng.standard_normal(size=(pool_size, d))
        Cp /= np.maximum(np.linalg.norm(Cp, axis=1, keepdims=True), 1e-12)
        D = Cp @ Xfrozen.T
        curmax = D.max(axis=1)
        used = np.zeros(pool_size, dtype=bool)
        new_pts = []
        for _ in range(k):
            curmax[used] = np.inf
            bi = int(np.argmin(curmax))
            used[bi] = True
            pt = Cp[bi].copy()
            new_pts.append(pt)
            dd = Cp @ pt
            # maintain max vs frozen and vs already chosen
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
            if time.time() > T_END - FINAL_RESERVE:
                break
        new_pts = np.stack(new_pts, axis=0)
        # quick hole polish of worst new points vs frozen+others
        for s in range(len(new_pts)):
            if time.time() > T_END - FINAL_RESERVE:
                break
            others = np.vstack([Xfrozen, np.delete(new_pts, s, axis=0)])
            best_pt = new_pts[s].copy()
            best_v = float(np.max(others @ best_pt))
            for _ in range(40):
                prop = best_pt + 0.03 * rng.standard_normal(size=(d,))
                prop /= (np.linalg.norm(prop) + 1e-12)
                v = float(np.max(others @ prop))
                if v < best_v:
                    best_v = v
                    best_pt = prop
            new_pts[s] = best_pt
        return new_pts

    def movable_polish(Xfrozen, Yinit, deadline):
        F = Xfrozen.shape[0]
        k = Yinit.shape[0]
        nn = F + k
        II, JJ = np.triu_indices(nn, 1)
        Yf = Yinit.ravel().copy()
        # normalize init
        Ym = Yf.reshape(k, d)
        Ym /= np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
        Yf = Ym.ravel()
        taus = [4e-3, 1.2e-3, 4e-4, 1.4e-4]
        iters = [50, 70, 90, 110]
        creg = 0.05
        for tau, mi in zip(taus, iters):
            if time.time() > deadline:
                break
            tau = float(tau)
            def fg(yflat, tau=tau):
                Y = yflat.reshape(k, d)
                r = np.sqrt(np.sum(Y * Y, axis=1))
                r = np.maximum(r, 1e-12)
                Yn = Y / r[:, None]
                Xfull = np.vstack([Xfrozen, Yn])
                G = Xfull @ Xfull.T
                g = G[II, JJ]
                gm = float(np.max(g))
                e = np.exp((g - gm) / tau)
                S = float(np.sum(e)) + 1e-300
                Lsoft = gm + tau * np.log(S)
                w = e / S
                W = np.zeros((nn, nn), dtype=float)
                W[II, JJ] = w
                W = W + W.T
                A = W @ Xfull
                proj = np.sum(A * Xfull, axis=1)
                # only movable rows
                Am = A[F:]
                Xm = Xfull[F:]
                pm = proj[F:]
                G0n = (Am - pm[:, None] * Xm) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0n + (2.0 * creg * diff)[:, None] * Xm
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Yf, jac=True, method='L-BFGS-B',
                               options={'maxiter': int(mi), 'maxfun': int(mi*4),
                                        'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Yf = res.x
            except Exception:
                break
            try:
                Ym = Yf.reshape(k, d)
                rn = np.maximum(np.linalg.norm(Ym, axis=1, keepdims=True), 1e-12)
                Yf = (Ym / rn).ravel()
            except Exception:
                pass
        Yfinal = Yf.reshape(k, d)
        Yfinal /= np.maximum(np.linalg.norm(Yfinal, axis=1, keepdims=True), 1e-12)
        return Yfinal

    # main stress-surgery loop
    itcount = 0
    # initial jam to get baseline duals consistent
    # (keep best as is; surgeries start from best)
    while time.time() < T_END - FINAL_RESERVE:
        itcount += 1
        # compute stress on incumbent best
        try:
            m0, deg0, stress0, neigh0, _ = contact_stress(bestX)
        except Exception:
            break
        k_target = int(rng.integers(20, 36))  # 20-35
        mov_idx = select_movable(deg0, stress0, neigh0, k_target)
        k = len(mov_idx)
        mask = np.ones(n, dtype=bool)
        mask[mov_idx] = False
        Xfrozen = bestX[mask].copy()
        # refill from >=100k pool
        pool_sz = 100000
        # enlarge pool on early iterations
        if itcount % 3 == 1:
            pool_sz = 120000
        try:
            Ynew = greedy_refill(Xfrozen, k, pool_size=pool_sz)
        except Exception:
            continue
        if Ynew.shape[0] != k:
            continue
        Xt = bestX.copy()
        Xt[mov_idx] = Ynew
        v_greedy = _max_cos(Xt)
        # movable-only softmax polish
        try:
            Ypol = movable_polish(Xfrozen, Ynew, T_END - FINAL_RESERVE)
        except Exception:
            continue
        Xcand = bestX.copy()
        Xcand[mov_idx] = Ypol
        v_pol = _max_cos(Xcand)
        # acceptance threshold for expensive global jam: vary
        thresh_choices = [0.0, 2e-4, 5e-4, 1e-3]
        thr = thresh_choices[itcount % len(thresh_choices)]
        # only jam if promising vs best (transfer stress, don't waste time)
        if min(v_greedy, v_pol) > bestVal + thr + 5e-4 and v_pol > bestVal:
            # skip global jam, try next neighborhood
            continue
        # short global LP jam
        try:
            Xjam = lp_jam(Xcand, T_END - FINAL_RESERVE, max_it=10, radius_init=2e-3, slack=1.5e-3)
        except Exception:
            Xjam = Xcand
        v_jam = _max_cos(Xjam)
        # keep only if exact maximum improves (strict)
        if v_jam < bestVal - 1e-12:
            bestX = Xjam.copy()
            bestVal = v_jam
        elif v_pol < bestVal - 1e-12:
            bestX = Xcand.copy()
            bestVal = v_pol
        # time check: if very little improvement, continue with new random neighborhood
        if time.time() > T_END - FINAL_RESERVE - 2:
            break

    # final short global jam on best if time remains
    if time.time() < T_END - FINAL_RESERVE + 6:
        try:
            Xf = lp_jam(bestX, T_END - 2.0, max_it=14, radius_init=1.5e-3, slack=8e-4)
            vf = _max_cos(Xf)
            if vf < bestVal:
                bestX = Xf; bestVal = vf
        except Exception:
            pass

    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
