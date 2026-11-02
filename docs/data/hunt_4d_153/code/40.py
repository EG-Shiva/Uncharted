import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def random_sphere(n, d, rng):
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]; J0 = iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu0]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 3000
        I = I0[sel]; J = J0[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        vv = max_cos(y)
        if vv < best:
            best = vv; best_x = y.copy(); x = y
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x

def get_duals(x, radius=1e-3):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = (x @ x.T)[iu]
    m = float(np.max(G))
    thr = m - max(20.0*radius, 1e-9)
    sel = G > thr
    I = iu[0][sel]; J = iu[1][sel]
    csel = G[sel]
    p = len(I)
    nv = n*d+1
    cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
    rows = np.repeat(np.arange(p), 2*d+1)
    vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
    except Exception:
        return I, J, csel, None
    if res.status != 0:
        return I, J, csel, None
    marg = None
    try:
        # scipy >=1.9: res.ineqlin.marginals
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            try:
                marg = np.asarray(res.ineqlin.marginals, dtype=float)
            except Exception:
                marg = np.asarray(res.ineqlin['marginals'], dtype=float)
        elif isinstance(res, dict) and 'ineqlin' in res:
            marg = np.asarray(res['ineqlin']['marginals'], dtype=float)
        else:
            marg = np.asarray(getattr(res, 'ineqlin_marginals', None), dtype=float)
    except Exception:
        marg = None
    return I, J, csel, marg

def penalized_adam(X0, fI, fJ, rel_mask, t_target, lam, steps, b0, b1, lr0, lr1, rng, t_deadline=1e30):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a=0.9; b2a=0.999; eps=1e-8
    lrat = lr1/lr0 if lr0!=0 else 1.0
    brat = b1/b0 if b0!=0 else 1.0
    # build released boolean matrix for fast scaling (n x n symmetric)
    relMat = None
    if rel_mask is not None and len(fI)>=0:
        pass
    # rel pairs arrays
    rI = None; rJ = None
    if rel_mask is not None:
        rI, rJ = rel_mask
    for t in range(1, steps+1):
        if t % 300 == 0 and time.time() > t_deadline:
            break
        frac = t/steps
        beta = b0 * (brat ** frac)
        lr = lr0 * (lrat ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        # softmax weights, overflow safe
        E = np.exp(beta*(G - m0))
        np.fill_diagonal(E, 0.0)
        if rI is not None and len(rI)>0:
            # down-weight released contacts by 0.1
            E[rI, rJ] *= 0.1
            E[rJ, rI] *= 0.1
        s = float(np.sum(E))
        if not np.isfinite(s) or s<=0:
            s = 1.0
        Wn = E / s
        grad_soft = Wn @ X
        # forced penalty gradient
        if len(fI)>0:
            dots = G[fI, fJ]
            diff = t_target - dots  # >0 needs pull up
            act = diff > 0
            if np.any(act):
                aI = fI[act]; aJ = fJ[act]
                # gradient of lam*relu = -lam*X_other per active pair
                # accumulate
                gs = np.zeros_like(X)
                # each pair contributes to both endpoints
                np.add.at(gs, aI, -X[aJ])
                np.add.at(gs, aJ, -X[aI])
                # scale: diff magnitude? use constant lam (not proportional) -> subgradient
                grad_soft = grad_soft + lam * gs / max(1,len(aI))
                # note: gs already ~ O(1); lam controls balance
        dot = np.sum(grad_soft*X, axis=1, keepdims=True)
        grad_t = grad_soft - dot*X
        m *= b1a; m += (1-b1a)*grad_t
        v *= b2a; v += (1-b2a)*(grad_t*grad_t)
        mhat = m/(1-b1a**t); vhat = v/(1-b2a**t)
        step = lr*mhat/(np.sqrt(vhat)+eps)
        sdot = np.sum(step*X, axis=1, keepdims=True)
        step -= sdot*X
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.where(nrm>0.02, 0.02/(nrm+1e-12), 1.0)
        step *= scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    DEADLINE = 178.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands=[]
    if xb is not None and xb.shape==(153,4):
        cands.append(xb/np.linalg.norm(xb,axis=1,keepdims=True))
    if xr is not None and xr.shape==(153,4):
        cands.append(xr/np.linalg.norm(xr,axis=1,keepdims=True))
    if len(cands)==0:
        cands.append(random_sphere(153,4,rng))
    bestX=None; bestV=1e30
    for c in cands:
        vv=max_cos(c)
        if vv<bestV:
            bestV=vv; bestX=c.copy()
    n,d = bestX.shape
    # polish baseline briefly to get true local optimum + duals
    # use small polish to equalize (2s)
    if time.time()-t0 < 150:
        bestX = lp_polish(bestX, 2.0)
        bestV = max_cos(bestX)
    m0 = bestV
    # contact graph
    Gfull = bestX @ bestX.T
    iu = np.triu_indices(n,1)
    Gall = Gfull[iu]
    order = np.argsort(-Gall)  # descending
    sortedG = Gall[order]
    sortedI = iu[0][order]; sortedJ = iu[1][order]
    # contacts tol 1e-9
    contact_mask = Gall > (m0 - 1e-9)
    cI = iu[0][contact_mask]; cJ = iu[1][contact_mask]
    contact_set = set(zip(cI.tolist(), cJ.tolist()))
    ncontacts = len(contact_set)
    # contact counts
    B = Gfull > (m0 - 1e-9)
    np.fill_diagonal(B, False)
    cc = B.sum(axis=1)
    rattlers = np.where(cc==0)[0]
    lowpts = np.where(cc<=4)[0]
    lowset = set(lowpts.tolist())
    # dual stresses
    LI, LJ, Lsel, marg = get_duals(bestX, radius=1e-3)
    if marg is not None and len(marg)==len(LI):
        stress = -np.asarray(marg, dtype=float)  # >=0 for active
        # clean non-finite
        stress[~np.isfinite(stress)] = 0.0
        stress = np.maximum(stress, 0.0)
        # map to pairs
        # release candidates: smallest stress among true contacts
        # build dict pair->stress (if multiple LP rows same pair keep max)
        # LI/LJ correspond to thresholded set (may be > contacts)
        sorder = np.argsort(stress)  # ascending: low stress first
        release_pool_I = LI[sorder]
        release_pool_J = LJ[sorder]
    else:
        # fallback: random order
        idx = rng.permutation(len(cI))
        release_pool_I = cI[idx]
        release_pool_J = cJ[idx]
        stress = None
    # second-shell pool: ranks ~ ncontacts .. 1200, slack 1e-5..2e-3
    # estimate ncontacts ~460
    start_rank = ncontacts
    # take next 800 pairs
    cand2_I=[]; cand2_J=[]; cand2_score=[]
    for k in range(start_rank, min(len(order), start_rank+1500)):
        i=int(sortedI[k]); j=int(sortedJ[k])
        g=float(sortedG[k])
        slack = m0 - g
        if slack < 5e-6 or slack > 2.5e-3:
            continue
        # bonus for rattler/low incidence
        bonus = 0
        if i in rattlers or j in rattlers:
            bonus = 3
        elif i in lowset or j in lowset:
            bonus = 1
        # prefer smaller slack (closer) + bonus
        # score: slack - bonus*5e-4 (lower better)
        sc = slack - bonus*5e-4
        cand2_I.append(i); cand2_J.append(j); cand2_score.append(sc)
    if len(cand2_I)==0:
        # fallback: take ranks 460-1000 regardless
        for k in range(start_rank, min(len(order), start_rank+800)):
            cand2_I.append(int(sortedI[k])); cand2_J.append(int(sortedJ[k])); cand2_score.append(float(m0-sortedG[k]))
    cand2_I=np.array(cand2_I,dtype=int); cand2_J=np.array(cand2_J,dtype=int); cand2_score=np.array(cand2_score)
    o2=np.argsort(cand2_score)
    cand2_I=cand2_I[o2]; cand2_J=cand2_J[o2]
    # limit pool to top 200
    P2 = min(200, len(cand2_I))
    cand2_I=cand2_I[:P2]; cand2_J=cand2_J[:P2]
    t_target = m0 - 2e-4
    EXPLORE_END = t0 + 148.0
    pool=[]  # list of (score, X)
    ntrials = 30
    # release pool arrays
    RP = len(release_pool_I)
    for trial in range(ntrials):
        if time.time() > EXPLORE_END or time.time()-t0 > DEADLINE-30:
            break
        rng2 = np.random.default_rng(1000+trial)
        nf = int(rng2.integers(1,4))  # 1-3
        nr = int(rng2.integers(3,9))  # 3-8
        # sample forced: bias to top of cand2 (rank-weighted)
        # choose nf distinct
        weights = np.exp(-np.arange(P2)/40.0)
        weights/=weights.sum()
        fidx = rng2.choice(P2, size=nf, replace=False, p=weights)
        fI = cand2_I[fidx]; fJ = cand2_J[fidx]
        # sample released: bias to low stress (front of release pool)
        RPw = min(RP, 400)
        w2 = np.exp(-np.arange(RPw)/60.0); w2/=w2.sum()
        ridx = rng2.choice(RPw, size=min(nr,RPw), replace=False, p=w2)
        rI = release_pool_I[ridx]; rJ = release_pool_J[ridx]
        lam = float(rng2.choice([1.5,2.5,4.0,6.0]))
        sigma = float(rng2.choice([0.008,0.015,0.025]))
        Xs = bestX + sigma*rng2.standard_normal((n,d))
        Xs/=np.linalg.norm(Xs,axis=1,keepdims=True)
        steps = 1500 if trial%3!=0 else 2000
        b0_=50.0; b1_=400.0
        lr0_=0.005; lr1_=0.0005
        # vary lr slightly
        if trial%4==1:
            lr0_=0.007
        Xopt = penalized_adam(Xs, fI, fJ, (rI,rJ), t_target, lam, steps, b0_, b1_, lr0_, lr1_, rng2, t_deadline=EXPLORE_END)
        v_before = max_cos(Xopt)
        # quick LP polish 0.8s
        remain = EXPLORE_END - time.time()
        if remain < 0.4:
            # no time for LP, just evaluate
            if v_before < bestV:
                bestV=v_before; bestX=Xopt.copy()
            continue
        Xp = lp_polish(Xopt, min(0.9, remain-0.05))
        vp = max_cos(Xp)
        if vp < bestV - 1e-12:
            bestV=vp
            bestX=Xp.copy()
            # update reference? keep original contact_set for graph diff? update m0? keep original for target but update best
            # also add to pool
            pool.append((vp, Xp.copy()))
        else:
            # graph difference check vs original record graph
            Gp = Xp @ Xp.T
            maskp = Gp > (vp - 1e-9)
            # count symmetric difference of edge sets (upper tri)
            # build sets only for upper tri
            # fast: compare with contact_set using boolean arrays
            # original contact mask boolean matrix
            # compute edge diff count
            # use triu masks
            Bp = np.triu(maskp,1)
            # original mask triu
            Go = bestX @ bestX.T  # careful bestX may have changed if improved? use initial? use stored contact_set
            # Instead compute diff vs stored contact_set
            # build set for Xp
            pI, pJ = np.where(Bp)
            pset = set(zip(pI.tolist(), pJ.tolist()))
            diff = len(pset.symmetric_difference(contact_set))
            if diff >= 5 and vp < m0 + 5e-4:
                pool.append((vp, Xp.copy()))
            # also keep near-best for diversity even without diff? limit pool size
            if len(pool)>12:
                pool.sort(key=lambda x: x[0])
                pool=pool[:12]
    # re-polish pool members with fresh LP (best few)
    if len(pool)>0:
        pool.sort(key=lambda x: x[0])
        # take top up to 4 distinct
        for k in range(min(4, len(pool))):
            if time.time()-t0 > DEADLINE-12:
                break
            vv, XX = pool[k]
            # fresh LP 2s
            Xq = lp_polish(XX, 2.0)
            vq = max_cos(Xq)
            if vq < bestV:
                bestV=vq; bestX=Xq.copy()
    # final thorough polish
    remain = DEADLINE - (time.time()-t0) - 1.0
    if remain > 1.0:
        bestX = lp_polish(bestX, min(remain, 20.0))
    bestX/=np.linalg.norm(bestX,axis=1,keepdims=True)
    return bestX
