import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_unconstrained(X, iters, beta0, beta1, lr0, lr1, seed_off=0):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    iu_diag = np.diag_indices(n)
    for it in range(1, iters + 1):
        frac = it / float(iters)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        cmax = float(G.max()) if False else None
        # max off-diagonal
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        # avoid overflow: cap beta*(G-cmax)
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1.0 - b1) * Gt
        m2 *= b2; m2 += (1.0 - b2) * Gt * Gt
        mh = m1 / (1.0 - b1p)
        vh = m2 / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _adam_antipodal(Y, iters, beta0, beta1, lr0, lr1):
    Y = _norm(Y.copy())
    n, d = Y.shape
    m1 = np.zeros_like(Y); m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        frac = it / float(iters)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = Y @ Y.T
        np.fill_diagonal(G, 0.0)
        A = np.abs(G)
        np.fill_diagonal(A, -10.0)
        amax = float(A.max())
        Wp = np.exp(beta * (G - amax))
        Wm = np.exp(beta * (-G - amax))
        np.fill_diagonal(Wp, 0.0); np.fill_diagonal(Wm, 0.0)
        s = float(Wp.sum() + Wm.sum())
        if not np.isfinite(s) or s <= 0:
            break
        Wp /= s; Wm /= s
        Gr = (Wp - Wm) @ Y
        proj = (Gr * Y).sum(axis=1, keepdims=True)
        Gt = Gr - proj * Y
        b1p *= b1; b2p *= b2
        m1 *= b1; m1 += (1.0 - b1) * Gt
        m2 *= b2; m2 += (1.0 - b2) * Gt * Gt
        mh = m1 / (1.0 - b1p); vh = m2 / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        Y = Y - step
        Y = _norm(Y)
    return Y

def _antipodal_full(Y):
    Y = _norm(Y)
    return np.vstack([Y, -Y])

def _antipodal_score(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    return float(np.abs(G).max())

def _build_orbit(Generics, q):
    # Generics: (9,6) unit rows, q: (3,) unit
    pts = []
    for r in Generics:
        for s in range(6):
            rs = np.roll(r, s)
            pts.append(rs)
            pts.append(-rs)
    # special size-6: v = [q,q]/sqrt2, its 12 images collapse to 6 unique
    v = np.concatenate([q, q]) / np.sqrt(2.0)
    seen = []
    for s in range(6):
        rs = np.roll(v, s)
        seen.append(rs)
        # -rs is shift-by-3 of rs? include but dedup later
    # dedup: keep unique (should be 6 distinct, but with generic q may be 12?)
    # To guarantee exactly 6, force period-3 structure is broken by roll by 1
    # Actually roll by 1 of [a,b,c,a,b,c] = [c,a,b,c,a,b] which is still period-3
    # So there are only 3 distinct shifts up to period; with sign -> 6.
    # Collect unique:
    uniq = []
    for c in seen:
        dup = False
        for u in uniq:
            if float(np.dot(c, u)) > 1 - 1e-9:
                dup = True; break
        if not dup:
            uniq.append(c)
    # uniq should be 3; add negatives
    full6 = []
    for u in uniq:
        full6.append(u); full6.append(-u)
    # if not 6 (degenerate q), pad by using v shifts anyway to reach 6
    # fixed size-2
    e0 = np.ones(6) / np.sqrt(6.0)
    # assemble
    X = np.array(pts + full6 + [e0, -e0])
    # pts = 9*12=108, full6=6, +2 =116 expected; if uniq!=3 adjust
    if X.shape[0] != 116:
        # fallback: construct exactly 116 by trunc/pad with generics copies
        # build alternative: use 108 + 6 from full orbit of v (12 with dups removed arbitrarily) +2
        # just take first 6 of [roll(v,s), -roll(v,s)]
        alt = []
        for s in range(6):
            alt.append(np.roll(v, s))
        # take 6 of them? need 6 points centrally symmetric: take 3 + negs
        alt = alt[:3]
        full6b = []
        for u in alt:
            full6b.append(u); full6b.append(-u)
        X = np.array(pts + full6b + [e0, -e0])
    return _norm(X)

def _adam_orbit(Generics0, q0, iters, beta0, beta1, lr0, lr1):
    Gmat = _norm(Generics0.copy())
    q = np.asarray(q0, dtype=float).copy()
    q = q / (np.linalg.norm(q) + 1e-15)
    m1g = np.zeros_like(Gmat); m2g = np.zeros_like(Gmat)
    m1q = np.zeros_like(q); m2q = np.zeros_like(q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    ng = Gmat.shape[0]
    for it in range(1, iters + 1):
        frac = it / float(iters)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        X = _build_orbit(Gmat, q)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gfull = W @ X  # (116,6)
        # pull back to generics: first 108 rows correspond to generics in order
        # order: for gi in 0..8, for s in 0..5: [+, -]
        grad_g = np.zeros_like(Gmat)
        idx = 0
        for gi in range(ng):
            acc = np.zeros(6)
            for s in range(6):
                for e in (1.0, -1.0):
                    gg = Gfull[idx]; idx += 1
                    # forward: e*roll(r,s); backward: e*roll(gg,-s)
                    acc += e * np.roll(gg, -s)
            grad_g[gi] = acc
        # pull back to q: next 6 rows are full6 (3 pairs +/-), last 2 fixed
        # full6 construction: uniq (3) + negs; mapping to q:
        # v=[q,q]/sqrt2, u_s = roll(v,s) for s in some set; gradient approx:
        # simpler: aggregate gradient of v-orbit then project
        # Recompute uniq shifts used: we use _build_orbit's uniq; approximate by s=0,1,2
        G6 = Gfull[idx:idx+6]; idx += 6
        # assume G6 = [u0,-u0,u1,-u1,u2,-u2] with u_s=roll(v,s), s=0,1,2
        grad_v = np.zeros(6)
        for k in range(3):
            up = G6[2*k]; um = G6[2*k+1]
            # contribution: up from +roll(v,k), um from -roll(v,k)
            # d v from up: roll(up,-k); from um: -roll(um,-k)
            grad_v += np.roll(up, -k) - np.roll(um, -k)
        # v = [q,q]/sqrt2 => grad_q = (grad_v[:3]+grad_v[3:])/sqrt2
        grad_q = (grad_v[:3] + grad_v[3:]) / np.sqrt(2.0)
        # tangent projections
        projg = (grad_g * Gmat).sum(axis=1, keepdims=True)
        gtg = grad_g - projg * Gmat
        projq = float(np.dot(grad_q, q))
        gtq = grad_q - projq * q
        b1p *= b1; b2p *= b2
        m1g *= b1; m1g += (1.0-b1)*gtg
        m2g *= b2; m2g += (1.0-b2)*gtg*gtg
        m1q *= b1; m1q += (1.0-b1)*gtq
        m2q *= b2; m2q += (1.0-b2)*gtq*gtq
        mhg = m1g/(1.0-b1p); vhg = m2g/(1.0-b2p)
        mhq = m1q/(1.0-b1p); vhq = m2q/(1.0-b2p)
        Gmat = Gmat - lr * mhg / (np.sqrt(vhg)+eps)
        q = q - lr * mhq / (np.sqrt(vhq)+eps)
        Gmat = _norm(Gmat)
        q = q/(np.linalg.norm(q)+1e-15)
    return Gmat, q

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bestX = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0*radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if len(idx) > 1600:
            order = np.argsort(c[idx])[-1600:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c, dtype=bool); sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj,A_ub=A_ub,b_ub=-csel,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best=v; bestX=y.copy(); X=y
            radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
    return bestX

def solve():
    t0=time.time()
    BUDGET=172.0
    rng=np.random.default_rng(7)
    cand_list=[]  # (val, X)
    # ---- Phase A: antipodal 58 reps, many restarts ----
    deadlineA=t0+38.0
    bestA_val=np.inf; bestA_Y=None
    restA=0
    while time.time()<deadlineA:
        restA+=1
        Y0=_norm(rng.standard_normal((58,6)))
        # short schedule beta 50->1500
        Y1=_adam_antipodal(Y0, iters=500, beta0=50, beta1=1200, lr0=0.06, lr1=0.004)
        v=_antipodal_score(Y1)
        if v<bestA_val:
            bestA_val=v; bestA_Y=Y1.copy()
        # quick second refinement on best occasionally
        if restA%6==0 and bestA_Y is not None and time.time()<deadlineA-3:
            Y1b=_adam_antipodal(bestA_Y+0.002*rng.standard_normal((58,6)), iters=300, beta0=200, beta1=1500, lr0=0.02, lr1=0.002)
            vb=_antipodal_score(Y1b)
            if vb<bestA_val:
                bestA_val=vb; bestA_Y=Y1b.copy()
        if time.time()>deadlineA-1:
            break
    if bestA_Y is not None:
        XA=_antipodal_full(bestA_Y)
        cand_list.append((_maxcos(XA), XA))
    # ---- Phase B: orbit construction ----
    deadlineB=t0+78.0
    bestO_val=np.inf; bestO=None
    restO=0
    while time.time()<deadlineB:
        restO+=1
        G0=_norm(rng.standard_normal((9,6)))
        q0=_norm(rng.standard_normal((1,3)))[0]
        G1,q1=_adam_orbit(G0,q0,iters=350,beta0=50,beta1=1000,lr0=0.06,lr1=0.004)
        try:
            XO=_build_orbit(G1,q1)
        except Exception:
            continue
        if XO.shape!=(116,6):
            continue
        v=_maxcos(XO)
        if v<bestO_val:
            bestO_val=v; bestO=(G1.copy(),q1.copy(),XO.copy())
        if time.time()>deadlineB-1:
            break
    if bestO is not None:
        cand_list.append((bestO_val, bestO[2]))
    # ---- Phase C: release + perturb + unconstrained re-optimise ----
    # pick up to 3 released seeds: best antipodal full, best orbit full, plus random de novo unconstrained best-of-few
    seeds=[]
    if bestA_Y is not None:
        seeds.append(_antipodal_full(bestA_Y))
    if bestO is not None:
        seeds.append(bestO[2])
    # add one pure random unconstrained quick search (de novo) to diversify
    # few random restarts with short smooth
    t_rand=min(12.0, max(4.0, (t0+95.0)-time.time()))
    if time.time()+t_rand < t0+BUDGET-40:
        rbest=np.inf; rbestX=None
        rend=time.time()+t_rand
        while time.time()<rend:
            X0=_norm(rng.standard_normal((116,6)))
            X1=_adam_unconstrained(X0,iters=350,beta0=40,beta1=600,lr0=0.05,lr1=0.005)
            v=_maxcos(X1)
            if v<rbest:
                rbest=v; rbestX=X1.copy()
        if rbestX is not None:
            seeds.append(rbestX)
            cand_list.append((rbest, rbestX))
    # release: perturb 1e-3..1e-2 and anneal
    released=[]
    pert_levels=[0.002,0.006,0.012]
    for si,Xs in enumerate(seeds):
        for pi,pl in enumerate(pert_levels):
            if time.time()>t0+140.0:
                break
            if len(released)>7:
                break
            Xn=_norm(Xs+pl*rng.standard_normal(Xs.shape))
            # annealed softmax: two-stage
            Xn=_adam_unconstrained(Xn,iters=500,beta0=30,beta1=900,lr0=0.05,lr1=0.003)
            Xn=_adam_unconstrained(Xn,iters=300,beta0=300,beta1=2000,lr0=0.015,lr1=0.002)
            v=_maxcos(Xn)
            released.append((v,Xn))
        if time.time()>t0+140.0:
            break
    for v,Xn in released:
        cand_list.append((v,Xn))
    # sort and keep top 3 for LP
    cand_list=sorted(cand_list,key=lambda t:t[0])
    top=cand_list[:3]
    # ---- Phase D: LP polish top candidates + final compare ----
    remain=t0+BUDGET-time.time()-4.0
    if remain<6: remain=6
    per=max(2.0, remain/max(len(top),1)-0.5)
    polished=[]
    for v,Xn in top:
        if time.time()>t0+BUDGET-6:
            break
        Xp=_lp_polish(Xn, seconds=min(per, t0+BUDGET-5-(time.time())))
        polished.append((_maxcos(Xp),Xp))
    for v,Xn in polished:
        cand_list.append((v,Xn))
    # final comparison with incumbent files (allowed now)
    def _clean(a):
        try:
            a=np.asarray(a,dtype=float)
        except Exception:
            return None
        if a.shape!=(116,6) or not np.all(np.isfinite(a)):
            return None
        n=np.linalg.norm(a,axis=1,keepdims=True)
        if np.any(n<1e-12):
            return None
        return a/n
    try:
        b=_clean(np.load("best.npy"))
        if b is not None:
            cand_list.append((_maxcos(b),b))
    except Exception:
        pass
    try:
        r=_clean(np.load("record.npy"))
        if r is not None:
            cand_list.append((_maxcos(r),r))
    except Exception:
        pass
    cand_list=sorted(cand_list,key=lambda t:t[0])
    best=cand_list[0][1]
    best=_norm(np.asarray(best,dtype=float))
    return best
