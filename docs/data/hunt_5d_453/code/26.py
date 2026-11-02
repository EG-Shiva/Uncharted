import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_HOP = t0 + 125.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
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
        bestX = rng.standard_normal((453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    n, d = bestX.shape

    def anneal(X, steps, b0, b1, lr0, lr1, mu, deadline):
        v = np.zeros_like(X)
        lb = np.log(b1 / b0) if b0 > 0 and b1 > 0 else 0.0
        ll = np.log(lr1 / lr0) if lr0 > 0 and lr1 > 0 else 0.0
        for it in range(steps):
            if time.time() > deadline:
                break
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            m = float(np.max(G))
            frac = it / max(1, steps - 1)
            beta = b0 * np.exp(lb * frac)
            lr = lr0 * np.exp(ll * frac)
            with np.errstate(over='ignore', invalid='ignore'):
                W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            Z = float(np.sum(W) * 0.5)
            if not np.isfinite(Z) or Z <= 0:
                break
            grad = (W @ X) / Z
            dot = np.sum(grad * X, axis=1, keepdims=True)
            Gt = grad - dot * X
            mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if not np.isfinite(mn) or mn < 1e-14:
                break
            Gt /= mn
            v = mu * v + (1.0 - mu) * Gt
            Y = X - lr * v
            yn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
            yn[yn == 0] = 1.0
            X = Y / yn
        return X

    def find_hole(Xkeep, n_cand=8000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Xkeep.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        h = C[bi].copy()
        for _ in range(25):
            dots = Xkeep @ h
            ci = int(np.argmax(dots))
            c = float(dots[ci])
            xc = Xkeep[ci]
            tt = xc - c * h
            tn = float(np.linalg.norm(tt))
            if tn < 1e-12:
                break
            tt /= tn
            h = h - 0.02 * tt
            nh = float(np.linalg.norm(h))
            if nh == 0 or not np.isfinite(nh):
                break
            h /= nh
        return h

    # quick initial shave of current max (few hundred iters, cheap)
    # use anneal short burst first to catch easy gain
    try:
        if time.time() < T_HOP:
            bestX = anneal(bestX, 150, 300.0, 900.0, 0.002, 0.0002, 0.85, min(time.time()+3.0, T_HOP))
            bv = _max_cos(bestX)
            if bv < bestV:
                bestV = bv
            else:
                # reload best if worsened (anneal returns last, not best) -> keep track
                # re-evaluate: if worse, keep old best copy? we overwrote; reload from file
                # instead keep elitist: compare and revert if much worse
                if bv > bestV + 0.002:
                    for fn in ("best.npy",):
                        try:
                            a = np.load(fn)
                            a = a/np.linalg.norm(a,axis=1,keepdims=True)
                            if _max_cos(a) <= bestV+1e-12:
                                bestX = a.copy()
                                break
                        except Exception:
                            pass
                    bestV = _max_cos(bestX)
                else:
                    bestV = bv
    except Exception:
        pass

    betas = [(250.0,1100.0),(400.0,2000.0),(150.0,600.0),(600.0,3000.0)]
    lrs = [(0.0022,0.00022),(0.0028,0.0002),(0.0014,0.00014),(0.0032,0.0003)]
    sigmas = [0.004,0.007,0.002,0.009,0.0012]

    it = 0
    # precompute row max for rattler targeting
    while time.time() < T_HOP:
        it += 1
        if T_HOP - time.time() < 0.8:
            break
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            flat = int(np.argmax(G))
            wi = flat // n; wj = flat % n
            rm = np.max(G, axis=1)
        except Exception:
            break
        inner_end = min(time.time() + 2.2, T_HOP)
        bi = it % len(betas)
        b0, b1 = betas[bi]
        lr0, lr1 = lrs[bi]
        sigma = float(sigmas[it % len(sigmas)])
        op = it % 3
        y = bestX.copy()
        try:
            if op == 0:
                # shake worst neighbourhood
                row = G[wi].copy()
                # top neighbours
                k = 10
                idx = np.argpartition(row, -k)[-k:]
                sset = set([int(wi), int(wj)])
                for tt in idx:
                    sset.add(int(tt))
                # add few random
                for _ in range(4):
                    sset.add(int(rng.integers(0, n)))
                sidx = np.array(list(sset), dtype=int)
                nz = rng.standard_normal((len(sidx), d)) * sigma
                for a2, idx2 in enumerate(sidx):
                    vv = nz[a2] - float(np.dot(nz[a2], y[idx2])) * y[idx2]
                    y[idx2] = y[idx2] + vv
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = anneal(y, 220, b0, b1, lr0, lr1, 0.87, inner_end)
            elif op == 1:
                # hole reinsertion: worst + 1-2 rattlers / random
                k = 2 + (it % 2)  # 2..3
                # rattlers = low row max
                try:
                    order = np.argsort(rm)  # smallest first
                    victims = [int(wi)]
                    if rm[wj] > rm[wi]:
                        victims = [int(wj)]
                    # add most-rattling points
                    for cand in order:
                        if int(cand) not in victims:
                            victims.append(int(cand))
                        if len(victims) >= k:
                            break
                except Exception:
                    victims = [int(wi), int(wj)][:k]
                while len(victims) < k:
                    c = int(rng.integers(0, n))
                    if c not in victims:
                        victims.append(c)
                mask = np.ones(n, dtype=bool)
                mask[np.array(victims, dtype=int)] = False
                Xe = y[mask]
                for vv2 in victims:
                    if time.time() > inner_end - 0.3:
                        break
                    h = find_hole(Xe, n_cand=7000)
                    y[vv2] = h
                    Xe = np.vstack([Xe, h.reshape(1, -1)])
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = anneal(y, 260, b0, b1, lr0, lr1, 0.87, inner_end)
            else:
                # global small shake + anneal (equalize shell)
                nz = rng.standard_normal((n, d)) * (sigma * 0.4)
                dot = np.sum(nz * y, axis=1, keepdims=True)
                nz -= dot * y
                y = y + nz
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = anneal(y, 220, b0, b1, lr0, lr1, 0.88, inner_end)
        except Exception:
            continue
        try:
            nv = _max_cos(y)
        except Exception:
            continue
        if nv < bestV:
            bestV = nv
            bestX = y.copy()

    # final micro-shaving: greedy worst-pair moves, time-boxed
    try:
        eps = 1.0e-7
        # estimate iterations affordable: ~4000
        max_it = 6000
        it2 = 0
        while it2 < max_it and time.time() < T_END - 1.0:
            it2 += 1
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            flat = int(np.argmax(G))
            i = flat // n; j = flat % n
            xi = bestX[i].copy(); xj = bestX[j].copy()
            cij = float(xi @ xj)
            cij = max(-1.0, min(1.0, cij))
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d); ti -= float(ti @ xi)*xi
                ni = float(np.linalg.norm(ti))+1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d); tj -= float(tj @ xj)*xj
                nj = float(np.linalg.norm(tj))+1e-30
            tj /= nj
            G[i,:]=-2.0; G[:,i]=-2.0; G[j,:]=-2.0; G[:,j]=-2.0
            m_rest = float(np.max(G))
            ri = bestX @ xi; rj = bestX @ xj
            ri[i]=-2.0; ri[j]=-2.0; rj[i]=-2.0; rj[j]=-2.0
            m_ri = float(np.max(ri)); m_rj = float(np.max(rj))
            m_rest_i = m_rest if m_rest>m_rj else m_rj
            m_rest_j = m_rest if m_rest>m_ri else m_ri
            best_m2 = m; bv_i=None; bv_j=None; mode=0; sv=None
            for scale in (eps, eps*0.5):
                vi = xi - scale*ti; nv1=float(np.linalg.norm(vi))
                if nv1==0 or not np.isfinite(nv1): continue
                vi/=nv1
                vj = xj - scale*tj; nv2=float(np.linalg.norm(vj))
                if nv2==0 or not np.isfinite(nv2): continue
                vj/=nv2
                di = bestX @ vi; dj = bestX @ vj
                di[i]=-2.0; di[j]=-2.0; dj[i]=-2.0; dj[j]=-2.0
                mc = m_rest
                a=float(np.max(di)); 
                if a>mc: mc=a
                b=float(np.max(dj)); 
                if b>mc: mc=b
                c2=float(vi@vj)
                if c2>mc: mc=c2
                if mc < best_m2-1e-15:
                    best_m2=mc; bv_i=vi; bv_j=vj; mode=0
            for scale in (eps, eps*0.5):
                vi = xi - scale*ti; nv1=float(np.linalg.norm(vi))
                if nv1==0 or not np.isfinite(nv1): continue
                vi/=nv1
                di = bestX @ vi; di[i]=-2.0
                mc = m_rest_i; a=float(np.max(di))
                if a>mc: mc=a
                if mc < best_m2-1e-15:
                    best_m2=mc; mode=1; sv=vi
            for scale in (eps, eps*0.5):
                vj = xj - scale*tj; nv2=float(np.linalg.norm(vj))
                if nv2==0 or not np.isfinite(nv2): continue
                vj/=nv2
                dj = bestX @ vj; dj[j]=-2.0
                mc = m_rest_j; a=float(np.max(dj))
                if a>mc: mc=a
                if mc < best_m2-1e-15:
                    best_m2=mc; mode=2; sv=vj
            if best_m2 < m-1e-15:
                if mode==0:
                    bestX[i]=bv_i; bestX[j]=bv_j
                elif mode==1:
                    bestX[i]=sv
                else:
                    bestX[j]=sv
                bestV=best_m2
                eps *= 0.98
                if eps < 1e-13: eps=1e-13
            else:
                eps *= 0.6
                if eps < 5e-14:
                    break
                if (it2 % 500)==0:
                    k = i if (it2%1000)==0 else j
                    rnd = rng.standard_normal(d)
                    rnd -= float(rnd@bestX[k])*bestX[k]
                    rn=float(np.linalg.norm(rnd))
                    if rn>1e-12:
                        rnd/=rn
                        xn=bestX[k]+1e-12*rnd
                        xn/=np.linalg.norm(xn)
                        bestX[k]=xn
    except Exception:
        pass

    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    # final safety: ensure finite
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nn = np.linalg.norm(bestX, axis=1, keepdims=True)
    nn[nn==0]=1.0
    bestX /= nn
    return bestX
