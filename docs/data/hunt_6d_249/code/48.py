import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_RES = 22.0
    rng = np.random.default_rng(11)
    n0, d = 249, 6

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n0, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n0, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])
    curX = bestX.copy()
    curVal = bestVal

    I0_full = None  # built per n

    def soft_jam(Xinit, taus, iters, creg=0.08):
        nn = Xinit.shape[0]
        iuI, iuJ = np.triu_indices(nn, 1)
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RES and len(taus) > 1:
                # allow short but break long
                pass
            tau = float(tau)
            def fg(xf):
                X = xf.reshape(nn, d)
                r = np.sqrt(np.sum(X*X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[iuI, iuJ]
                gm = np.max(g)
                e = np.exp((g-gm)/tau)
                S = np.sum(e)
                Lsoft = gm + tau*np.log(S)
                w = e/S
                W = np.zeros((nn, nn))
                W[iuI, iuJ] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn)/r[:, None]
                diff = r-1.0
                L = Lsoft + creg*np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                    options={'maxiter':int(mi),'maxfun':int(mi*4),'ftol':1e-14,'gtol':1e-7,'maxcor':8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(nn, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm/rn).ravel()
            if time.time() > T_END - 2:
                break
        Xo = Xf.reshape(nn, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def exact_polish(Xinit, deadline, step0=1.0e-3):
        cur = Xinit.copy()
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        curValL = _max_cos(cur)
        step = step0
        while time.time() < deadline:
            if step < 5e-14:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0*step, 2e-4)
            mask = G > (m-window)
            F = mask.astype(float) @ cur
            dot = np.sum(F*cur, axis=1)
            Ft = F - dot[:, None]*cur
            trial = cur - step*Ft
            trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            v = _max_cos(trial)
            if v < curValL:
                cur = trial; curValL = v
                step = min(step*1.25, 0.01)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        return cur, curValL

    def top_set(X, K=1500):
        nn = X.shape[0]
        iuI, iuJ = np.triu_indices(nn, 1)
        G = X @ X.T
        g = G[iuI, iuJ]
        if K >= len(g):
            idx = np.arange(len(g))
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = iuI[idx].astype(np.int64)*512 + iuJ[idx].astype(np.int64)
        return set(codes.tolist())

    def jaccard(a, b):
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - inter/union

    archive = []  # list of (val, X, set)
    best_set = top_set(bestX, 1500)
    archive.append((bestVal, bestX.copy(), set(best_set)))

    def try_add_archive(val, X):
        nonlocal archive
        s = top_set(X, 1500)
        # min distance to archive
        md = 1.0
        for _, _, s2 in archive:
            dd = jaccard(s, s2)
            if dd < md:
                md = dd
        if md < 0.08:
            return False, md
        archive.append((val, X.copy(), s))
        # cap size
        if len(archive) > 24:
            # drop worst with small diversity? keep best + diverse
            # sort by val, keep best 8 + most diverse 16
            archive_sorted = sorted(archive, key=lambda t: t[0])
            keep = archive_sorted[:8]
            rest = archive_sorted[8:]
            # compute diversity to best
            bset = top_set(bestX, 1500)
            rest_sorted = sorted(rest, key=lambda t: jaccard(t[2], bset), reverse=True)
            archive = keep + rest_sorted[:16]
        return True, md

    def refine_single(pt, Xfix):
        # single-point soft-max refinement
        p = pt.copy()/np.linalg.norm(pt)
        for tau, steps, lr in [(0.01, 80, 0.3),(0.003, 80, 0.2),(0.001, 80, 0.12)]:
            for _ in range(steps):
                dots = Xfix @ p
                gm = float(np.max(dots))
                e = np.exp((dots-gm)/tau)
                w = e/np.sum(e)
                grad = w @ Xfix
                grad_t = grad - np.dot(grad, p)*p
                gn = np.linalg.norm(grad_t)
                if gn > 1e-12:
                    grad_t /= gn
                    # line try
                    step = lr
                    bestd = float(np.max(Xfix @ p))
                    improved = False
                    for _ls in range(3):
                        pn = p - step*grad_t
                        pn /= np.linalg.norm(pn)
                        nd = float(np.max(Xfix @ pn))
                        if nd < bestd:
                            p = pn; bestd = nd; improved = True
                            break
                        step *= 0.5
                    if not improved:
                        # small step anyway to move
                        pn = p - 0.02*grad_t
                        pn /= np.linalg.norm(pn)
                        # accept if not much worse
                        nd = float(np.max(Xfix @ pn))
                        if nd < bestd + 0.005:
                            p = pn
                else:
                    break
        return p

    def deepest_holes(Xfix, total=100000, topk=20, chunk=10000):
        # return list of refined candidates (topk) sorted by max-dot
        nn = Xfix.shape[0]
        # collect best raw
        pool_vals = []
        pool_pts = []
        nch = total//chunk
        for _ in range(nch):
            C = rng.standard_normal(size=(chunk, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Xfix.T
            cm = D.max(axis=1)
            pool_vals.append(cm)
            pool_pts.append(C)
            if time.time() > T_END - 4:
                break
        if len(pool_vals)==0:
            return []
        V = np.concatenate(pool_vals, axis=0)
        P = np.concatenate(pool_pts, axis=0)
        idx = np.argpartition(V, topk)[:topk]
        # refine each of topk
        refined = []
        for k in idx:
            pt = refine_single(P[k], Xfix)
            v = float(np.max(Xfix @ pt))
            refined.append((v, pt))
        refined.sort(key=lambda t: t[0])
        return refined  # list (val, pt)

    def add_one_hole(X, total=100000):
        rh = deepest_holes(X, total=total, topk=20)
        if not rh:
            # random
            pt = rng.standard_normal(size=(d,))
            pt /= np.linalg.norm(pt)
            return np.vstack([X, pt[None,:]]), 1.0
        v, pt = rh[0]
        return np.vstack([X, pt[None,:]]), v

    def prune_one(X, delta=2e-3, tau=5e-4):
        nn = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - delta
        # count involvement
        mask = G > thr
        cnt = np.sum(mask, axis=1)
        rm, _ = _row_max(X)
        # score: primary cnt, secondary rm
        # pick max cnt; tie max rm
        order = np.lexsort((rm, cnt))
        idx_rm = int(order[-1])
        # if all zero counts (isolated), remove worst rm
        return np.delete(X, idx_rm, axis=0), idx_rm

    def greedy_refill_large(Xkeep, k_need):
        # one 100k pool greedy + light refine
        # generate pool 100k
        total = 100000
        chunk = 10000
        pools = []
        for _ in range(total//chunk):
            C = rng.standard_normal(size=(chunk, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            pools.append(C)
        Pool = np.concatenate(pools, axis=0)  # 100k x6
        # greedy select
        # D vs keep
        # chunked to avoid huge matrix: compute curmax per pool
        # compute in chunks
        curmax = np.empty(Pool.shape[0])
        # process pool in blocks of 10k vs Xkeep
        for s in range(0, Pool.shape[0], 10000):
            e = min(s+10000, Pool.shape[0])
            D = Pool[s:e] @ Xkeep.T
            curmax[s:e] = D.max(axis=1)
        used = np.zeros(Pool.shape[0], dtype=bool)
        new_pts = []
        Xall_extra = []  # for mutual updates
        for _ in range(k_need):
            curmax[used] = np.inf
            bi_ = int(np.argmin(curmax))
            used[bi_] = True
            pt = Pool[bi_].copy()
            # light single refine vs keep+already chosen
            if len(new_pts)==0:
                Xfix = Xkeep
            else:
                Xfix = np.vstack([Xkeep, np.stack(new_pts,axis=0)])
            pt = refine_single(pt, Xfix)
            new_pts.append(pt)
            # update curmax with new pt
            dd = Pool @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
            if time.time() > T_END - 4:
                break
        if len(new_pts)==0:
            return Xkeep
        return np.vstack([Xkeep, np.stack(new_pts,axis=0)])

    # quick jam helpers with time guard
    def jam_short(X):
        # short jam: one tau + exact polish 0.4s
        nn = X.shape[0]
        if nn > 260:
            taus=[0.003]; iters=[40]
        elif nn < 249:
            taus=[0.003,0.0012]; iters=[50,50]
        else:
            taus=[0.002,0.001]; iters=[45,45]
        Y = soft_jam(X, taus, iters)
        Y, v = exact_polish(Y, time.time()+0.35, step0=8e-4)
        return Y, _max_cos(Y)

    def jam_full(X):
        Y = soft_jam(X, [0.004,0.0015,0.0006], [70,70,70])
        Y, v = exact_polish(Y, time.time()+0.8, step0=1e-3)
        v2 = _max_cos(Y)
        if time.time() < T_END - 6:
            Y2 = soft_jam(Y, [0.0004], [50])
            v22 = _max_cos(Y2)
            if v22 < v2:
                Y, v2 = Y2, v22
        return Y, _max_cos(Y)

    it = 0
    # initial small jam of incumbent
    if time.time() < T_END - 60:
        try:
            Y = soft_jam(bestX, [0.002],[40])
            v = _max_cos(Y)
            if v < bestVal:
                bestVal=v; bestX=Y.copy(); curX=Y.copy(); curVal=v
                try_add_archive(v, Y)
        except Exception:
            pass

    while time.time() < T_END - FINAL_RES - 2:
        it += 1
        op = it % 5
        try:
            if op == 0:
                # GROW small -> 249 (cardinality homotopy low side)
                # choose target small size 235-248
                Ns = int(rng.integers(235, 249))
                # base choice: best or diverse archive
                if len(archive)>2 and rng.random()<0.5:
                    # pick diverse member
                    bset = top_set(bestX,1500)
                    divs = sorted(archive, key=lambda t: jaccard(t[2], bset), reverse=True)
                    base = divs[0][1].copy()
                    if base.shape[0]!=249:
                        base = bestX.copy()
                else:
                    base = curX.copy() if rng.random()<0.6 else bestX.copy()
                # delete to Ns
                rm,_ = _row_max(base)
                # delete worst + random mix
                kdel = 249 - Ns
                # worst-biased delete
                order = np.argsort(-rm)
                del_idx = set()
                # take ~70% worst, 30% random
                nw = int(0.7*kdel)
                for k in range(nw):
                    del_idx.add(int(order[k]))
                while len(del_idx)<kdel:
                    del_idx.add(int(rng.integers(0,249)))
                del_idx = sorted(del_idx)
                keep = np.ones(249,bool); keep[del_idx]=False
                Xs = base[keep]
                # jam small briefly
                Xs,_ = jam_short(Xs)
                # grow back one-by-one with deep holes (100k each)
                # limit total holes to keep time: if kdel>10, use greedy large for part
                if kdel > 8:
                    # first bulk greedy then last few deep
                    need_bulk = kdel-4
                    # greedy bulk from pool
                    Xg = greedy_refill_large(Xs, need_bulk)
                    Xs = Xg
                    for _ in range(4):
                        Xs,_ = add_one_hole(Xs, total=100000)
                else:
                    for _ in range(kdel):
                        Xs,_ = add_one_hole(Xs, total=100000)
                        if time.time()>T_END-FINAL_RES-2:
                            break
                Y,v = jam_short(Xs)
            elif op == 1:
                # PRUNE overfull -> 249
                No = int(rng.integers(255, 271))
                kadd = No-249
                base = curX.copy() if rng.random()<0.6 else bestX.copy()
                extra = rng.standard_normal(size=(kadd,d))
                extra /= np.linalg.norm(extra,axis=1,keepdims=True)
                Xo = np.vstack([base, extra])
                Xo,_ = jam_short(Xo)
                # prune back
                while Xo.shape[0] > 249:
                    Xo,_ = prune_one(Xo, delta=2e-3)
                Y,v = jam_short(Xo)
            elif op == 2:
                # 1-for-2 exchange (N+1 then prune)
                base = curX.copy() if rng.random()<0.7 else bestX.copy()
                # choose victim: worst or random rattler
                rm,Gm = _row_max(base)
                mcur = float(np.max(Gm)+0) if False else _max_cos(base)
                # contact degree at 1e-3
                G = base@base.T
                np.fill_diagonal(G,-2.0)
                deg = np.sum(G>(mcur-1e-3),axis=1)
                # prefer high-degree victim
                if rng.random()<0.7:
                    victim = int(np.argmax(rm + 1e-4*deg))
                else:
                    victim = int(rng.integers(0,249))
                Xtmp = np.delete(base, victim, axis=0)  # 248
                # add 2 deep holes
                Xtmp,_ = add_one_hole(Xtmp, total=100000)
                Xtmp,_ = add_one_hole(Xtmp, total=100000)  # 250
                # prune 1 to 249 (severity)
                Xtmp,_ = prune_one(Xtmp)
                Y,v = jam_short(Xtmp)
            elif op == 3:
                # 2-for-1 exchange (N-1 then grow)
                base = curX.copy() if rng.random()<0.7 else bestX.copy()
                rm,_ = _row_max(base)
                order = np.argsort(-rm)
                # remove 2: one worst, one random
                v1 = int(order[0])
                v2 = int(rng.integers(0,249))
                if v2==v1:
                    v2=(v2+1)%249
                keep = np.ones(249,bool); keep[v1]=False; keep[v2]=False
                Xtmp = base[keep]  # 247
                Xtmp,_ = add_one_hole(Xtmp, total=100000)  # 248
                # jam a bit then add final to 249
                Xtmp,_ = jam_short(Xtmp)
                Xtmp,_ = add_one_hole(Xtmp, total=100000)  # 249
                Y,v = jam_short(Xtmp)
            else:
                # large delete-refill avalanche 40-80
                base = curX.copy() if rng.random()<0.5 else bestX.copy()
                kdel = int(rng.integers(40, 81))
                rm,Gm2 = _row_max(base)
                G = base@base.T
                np.fill_diagonal(G,-2.0)
                mcur=_max_cos(base)
                deg = np.sum(G>(mcur-2e-4),axis=1)
                order = np.argsort(-(rm+5e-4*deg))
                del_idx=set()
                nw=int(0.6*kdel)
                for k in range(nw):
                    del_idx.add(int(order[k % len(order)]))
                while len(del_idx)<kdel:
                    del_idx.add(int(rng.integers(0,249)))
                keep=np.ones(249,bool); keep[sorted(del_idx)]=False
                Xkeep=base[keep]
                Xnew=greedy_refill_large(Xkeep,kdel)
                Y,v=jam_short(Xnew)
        except Exception:
            continue
        # acceptance: monotone best + non-monotone diverse to 0.715
        if 'v' not in locals() or 'Y' not in locals():
            continue
        try:
            vv=float(v); YY=Y
        except Exception:
            continue
        if vv < bestVal - 1e-12:
            bestVal=vv; bestX=YY.copy()
            curX=YY.copy(); curVal=vv
            try_add_archive(vv, YY)
        elif vv < 0.715:
            added,_ = try_add_archive(vv, YY)
            # temporarily accept worsened to allow reorganization
            if added and rng.random()<0.6:
                curX=YY.copy(); curVal=vv
            elif vv < curVal and rng.random()<0.3:
                curX=YY.copy(); curVal=vv
        # periodically fully re-jam most diverse survivor
        if it % 5 == 0 and time.time() < T_END-FINAL_RES-4:
            try:
                if len(archive)>=3:
                    bset = top_set(bestX,1500)
                    divs = sorted(archive, key=lambda t: jaccard(t[2], bset), reverse=True)
                    cand = divs[0][1]
                    Y2,v2 = jam_short(cand)
                    v2=float(v2)
                    if v2 < bestVal-1e-12:
                        bestVal=v2; bestX=Y2.copy(); curX=Y2.copy(); curVal=v2
                        try_add_archive(v2,Y2)
            except Exception:
                pass

    # final: fully re-jam most diverse survivors + best
    try:
        bset = top_set(bestX,1500)
        divs = sorted(archive, key=lambda t: jaccard(t[2], bset), reverse=True)
        final_cands = [bestX] + [t[1] for t in divs[:4]]
        # dedup by shape
        for Fc in final_cands:
            if time.time() > T_END - 2:
                break
            if Fc.shape != (249,6):
                continue
            Yf,vf = jam_full(Fc)
            if float(vf) < bestVal:
                bestVal=float(vf); bestX=Yf.copy()
    except Exception:
        pass
    bestX = bestX / np.maximum(np.linalg.norm(bestX,axis=1,keepdims=True),1e-12)
    # tiny shrink not needed; ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    # renormalize zeros if any
    nr = np.linalg.norm(bestX,axis=1,keepdims=True)
    bad = (nr.squeeze()<1e-9)
    if np.any(bad):
        rr = rng.standard_normal(size=(int(np.sum(bad)),d))
        rr/=np.linalg.norm(rr,axis=1,keepdims=True)
        bestX[bad]=rr
    bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX
