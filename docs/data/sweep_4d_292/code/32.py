import time
import numpy as np
import itertools
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# ---------- basic utils ----------
def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

# ---------- soft relax (full) ----------
def _soft_relax(x, steps=400, beta0=25.0, beta1=350.0, lr0=0.035, lr1=0.0035, mu=0.85):
    n, d = x.shape
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        A = (W @ x) / np.maximum(r, 1e-30)
        Adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - Adot * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        x = _norm(x)
    return x

# ---------- LP polish ----------
def _lp_polish(x, seconds, r0=1e-3):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return xb

# ---------- quaternion groups ----------
def _qmult(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

def _qleft(q):
    w, x, y, z = q
    return np.array([[w,-x,-y,-z],[x,w,-z,y],[y,z,w,-x],[z,-y,x,w]])

def _gen_T24():
    qs = []
    qs.append(np.array([1.,0,0,0])); qs.append(np.array([-1.,0,0,0]))
    for e in [np.array([0.,1,0,0]),np.array([0.,0,1,0]),np.array([0.,0,0,1])]:
        qs.append(e.copy()); qs.append(-e.copy())
    for s in itertools.product([-1.,1.], repeat=4):
        qs.append(np.array(s)/2.0)
    # dedup
    U=[]
    for q in qs:
        if not any(np.linalg.norm(q-u)<1e-9 or np.linalg.norm(q+u)<1e-9 for u in U):
            # keep both signs actually; above dedup with sign would collapse, so just append
            U.append(q)
    # we want all 24 with signs: list has 8+16=24
    return np.array(U)

def _closure(qs_init):
    S=[np.array([1.,0,0,0])]
    # normalize
    init=[q/np.linalg.norm(q) for q in qs_init]
    S=list([np.array([1.,0,0,0])])
    changed=True
    def contains(q):
        for u in S:
            if np.linalg.norm(q-u)<1e-8 or np.linalg.norm(q+u)<1e-8:
                # need exact sign match? for group containing -1, + and - both present, so check exact
                pass
        for u in S:
            if np.linalg.norm(q-u)<1e-8:
                return True
        return False
    for q in init:
        if not contains(q):
            S.append(q)
        if not contains(-q):
            S.append(-q)
    # closure
    it=0
    while True:
        grew=False
        cur=list(S)
        for a in cur:
            for b in cur:
                c=_qmult(a,b)
                c=c/np.linalg.norm(c)
                if not contains(c):
                    S.append(c); grew=True
        it+=1
        if not grew or len(S)>130 or it>10:
            break
    return np.array(S)

def _gen_I120():
    qs=[]
    qs.append(np.array([1.,0,0,0])); qs.append(np.array([-1.,0,0,0]))
    for e in [np.array([0.,1,0,0]),np.array([0.,0,1,0]),np.array([0.,0,0,1])]:
        qs.append(e.copy()); qs.append(-e.copy())
    for s in itertools.product([-1.,1.], repeat=4):
        qs.append(np.array(s)/2.0)
    phi=(1.0+np.sqrt(5.0))/2.0
    inv=1.0/phi
    base=np.array([0.0,1.0,phi,inv])/2.0
    # even permutations of 4 coords
    perms=list(itertools.permutations([0,1,2,3]))
    def perm_parity(p):
        invc=0
        for i in range(4):
            for j in range(i+1,4):
                if p[i]>p[j]: invc+=1
        return invc%2
    even=[p for p in perms if perm_parity(p)==0]
    seen=set()
    for p in even:
        tup=tuple(base[list(p)])
        # all sign combos on the 3 nonzero entries (zero entry sign irrelevant)
        # brute all 8 combos for positions where value !=0
        for signs in itertools.product([-1.,1.], repeat=4):
            v=np.array([tup[0]*signs[0],tup[1]*signs[1],tup[2]*signs[2],tup[3]*signs[3]])
            # canonical: force zero position to be positive to avoid double count
            # just add and dedup later
            qs.append(v)
    # dedup exact
    U=[]
    for q in qs:
        found=False
        for u in U:
            if np.linalg.norm(q-u)<1e-8:
                found=True; break
        if not found:
            U.append(q/np.linalg.norm(q))
    return np.array(U)

def _group_mats(quats):
    return np.array([_qleft(q) for q in quats])

# ---------- symmetric expansion + optimisation ----------
def _expand(seeds_sym, seeds_pad, Gmats):
    # seeds_sym (ns,4), Gmats (G,4,4)
    if seeds_sym.shape[0]>0:
        # X_sym[g*ns + ...]: apply each matrix
        # Gmats @ seeds.T -> (G,4,ns)
        T = np.einsum('gij,sj->gsi', Gmats, seeds_sym)
        Xs = T.reshape(-1, 4)
    else:
        Xs = np.zeros((0,4))
    if seeds_pad.shape[0]>0:
        X = np.vstack([Xs, seeds_pad]) if Xs.shape[0]>0 else seeds_pad.copy()
    else:
        X = Xs
    return _norm(X)

def _sym_optimize(Gmats, n_orb, n_pad, rng, steps_total=900):
    ns = n_orb
    npad = n_pad
    # init seeds
    ss = _norm(rng.standard_normal((max(ns,1), 4))) if ns>0 else np.zeros((0,4))
    sp = _norm(rng.standard_normal((max(npad,1), 4))) if npad>0 else np.zeros((0,4))
    if ns==0: ss=np.zeros((0,4))
    if npad==0: sp=np.zeros((0,4))
    vel_s = np.zeros_like(ss) if ns>0 else ss
    vel_p = np.zeros_like(sp) if npad>0 else sp
    mu=0.85
    Gnum = Gmats.shape[0] if ns>0 else 1
    # continuation 50->200->1000
    betas = np.concatenate([np.linspace(50,200,steps_total//2), np.linspace(200,1000,steps_total-steps_total//2)])
    lr0=0.04; lr1=0.004
    for s in range(steps_total):
        beta=float(betas[s])
        frac=s/max(1,steps_total-1)
        lr=lr0*(lr1/lr0)**frac
        X=_expand(ss,sp,Gmats) if (ns+npad)>0 else None
        n=X.shape[0]
        Gmat=X@X.T
        np.fill_diagonal(Gmat,-2.0)
        gmax=Gmat.max()
        W=np.exp(beta*(Gmat-gmax))
        np.fill_diagonal(W,0.0)
        r=W.sum(axis=1,keepdims=True)
        A=(W@X)/np.maximum(r,1e-30)
        Gt=A-np.sum(A*X,axis=1,keepdims=True)*X
        # pullback
        if ns>0:
            # Gt first ns*Gnum rows correspond to sym part in order g-major? our reshape: (G,ns) -> index g*ns+s
            Gt_sym=Gt[:ns*Gnum].reshape(Gnum,ns,4)
            # grad for seed s = sum_g M_g^T Gt[g,s]
            grad_s=np.einsum('gij,gsj->si',Gmats,Gt_sym)
            # project tangent
            grad_s=grad_s-np.sum(grad_s*ss,axis=1,keepdims=True)*ss
            vel_s=mu*vel_s+(1-mu)*grad_s
            bias=1.0-mu**(s+1)
            ss=ss-lr*(vel_s/max(bias,1e-12))
            ss=_norm(ss)
        if npad>0:
            Gt_pad=Gt[ns*Gnum:]
            grad_p=Gt_pad-np.sum(Gt_pad*sp,axis=1,keepdims=True)*sp
            vel_p=mu*vel_p+(1-mu)*grad_p
            bias=1.0-mu**(s+1)
            sp=sp-lr*(vel_p/max(bias,1e-12))
            sp=_norm(sp)
    X=_expand(ss,sp,Gmats)
    # deduplicate closer than 1e-8
    # greedy keep
    keep=[]
    for i in range(X.shape[0]):
        ok=True
        for j in keep:
            if float(X[i]@X[j])>1.0-5e-17:  # ~1e-8 distance => cos>1-5e-17
                ok=False; break
        if ok:
            keep.append(i)
    Xd=X[np.array(keep)]
    return Xd, _max_cos(Xd) if Xd.shape[0]>1 else 2.0

def solve():
    t0=time.time()
    DEAD=172.0
    rng=np.random.default_rng(20261031)
    # load incumbent
    def _load(p):
        try:
            a=np.load(p)
            a=np.asarray(a,dtype=float)
            if a.shape==(292,4) and np.all(np.isfinite(a)):
                n=np.linalg.norm(a,axis=1,keepdims=True); n[n==0]=1.0
                return a/n
        except Exception:
            pass
        return None
    best=_load("best.npy")
    rec=_load("record.npy")
    if best is None:
        best=_norm(rng.standard_normal((292,4)))
    if rec is not None:
        if _max_cos(rec)<_max_cos(best):
            best=rec.copy()
    best_val=_max_cos(best)
    incumbent=best.copy()

    # ---- symmetry-restricted search (bounded time ~45s) ----
    sym_dead=t0+45.0
    sym_best=None; sym_val=np.inf
    try:
        T24=_gen_T24()
        Gm_T=_group_mats(T24)
        # binary octahedral via closure adding (1,1,0,0)/sqrt2 direction
        q_extra=np.array([1./np.sqrt(2.),1./np.sqrt(2.),0.,0.])
        O48=_closure([q for q in T24]+[q_extra])
        if O48.shape[0]!=48:
            # fallback: use T24 only for second candidate with different pad
            O48=None
            Gm_O=None
        else:
            Gm_O=_group_mats(O48)
        I120=_gen_I120()
        # I120 size may be 120 (allow tolerance)
        if I120.shape[0]>130 or I120.shape[0]<110:
            Gm_I=None
        else:
            Gm_I=_group_mats(I120)
    except Exception:
        Gm_T=None; Gm_O=None; Gm_I=None

    candidates=[]
    # T24: 12*24=288 pad 4
    if Gm_T is not None and time.time()<sym_dead-5:
        try:
            Xd,v=_sym_optimize(Gm_T,12,4,rng,steps_total=700)
            if Xd.shape[0]==292:
                candidates.append((v,Xd))
            elif Xd.shape[0]>0:
                # if dedup collapsed, still try to fix by padding/truncating with random to reach 292 for evaluation
                pass
        except Exception:
            pass
    if Gm_O is not None and time.time()<sym_dead-5:
        try:
            Xd,v=_sym_optimize(Gm_O,6,4,rng,steps_total=700)
            if Xd.shape[0]==292:
                candidates.append((v,Xd))
        except Exception:
            pass
    if Gm_I is not None and time.time()<sym_dead-3:
        try:
            # 2*120=240 pad 52
            Xd,v=_sym_optimize(Gm_I,2,52,rng,steps_total=500)
            if Xd.shape[0]==292:
                candidates.append((v,Xd))
        except Exception:
            pass
    # evaluate symmetric candidates: break symmetry + polish briefly
    for (v,Xd) in candidates:
        if time.time()>sym_dead:
            break
        try:
            # break symmetry tiny tangent perturbation 1e-4
            pert=1e-4*rng.standard_normal(Xd.shape)
            # project tangent
            pert=pert-np.sum(pert*Xd,axis=1,keepdims=True)*Xd
            Y=_norm(Xd+pert)
            # quick soft relax at high beta to settle
            Y=_soft_relax(Y,steps=250,beta0=80.0,beta1=600.0,lr0=0.015,lr1=0.002,mu=0.88)
            Y=_lp_polish(Y,seconds=1.2,r0=8e-4)
            vy=_max_cos(Y)
            if vy<sym_val:
                sym_val=vy; sym_best=Y.copy()
        except Exception:
            continue
    if sym_best is not None and sym_val<best_val-1e-12:
        best=sym_best.copy(); best_val=sym_val

    # ---- main incumbent improvement: shakes + hole fill + anneal ----
    # time left for explore
    EXPLORE_END=t0+150.0
    it=0
    # precompute for hole finding: reuse rng
    while time.time()<EXPLORE_END and time.time()-t0<155:
        it+=1
        u=rng.random()
        n,d=best.shape
        Gc=best@best.T
        np.fill_diagonal(Gc,-2.0)
        mcur=float(Gc.max())
        cnt=(Gc>mcur-1e-7).sum(axis=1)
        if u<0.35:
            sigma=float(rng.choice(np.array([0.008,0.02,0.04,0.08])))
            xt=best+sigma*rng.standard_normal(best.shape)
            xt=_norm(xt)
            xt=_soft_relax(xt,steps=450,beta0=25.0,beta1=380.0,lr0=0.038,lr1=0.003,mu=0.86)
        elif u<0.60:
            # jammed neighbourhood shake
            order=np.argsort(-cnt)
            top=order[:20]
            center=int(top[rng.integers(0,len(top))])
            k=int(rng.integers(5,13))
            dots=best@best[center]
            S=np.argsort(-dots)[:k]
            xt=best.copy()
            xt[S]+=0.35*rng.standard_normal((k,d))
            xt=_norm(xt)
            # local-ish: full relax but starting from shaken
            xt=_soft_relax(xt,steps=450,beta0=28.0,beta1=420.0,lr0=0.038,lr1=0.003,mu=0.86)
        elif u<0.82:
            # kill worst into holes
            ns=int(rng.integers(2,5))
            pm=Gc.max(axis=1)
            src=np.argsort(-pm)[:12]
            sources=rng.choice(src,size=ns,replace=False)
            mask=np.ones(n,dtype=bool); mask[sources]=False
            Xkeep=best[mask]
            # random candidates for holes
            ncand=16000
            Z=_norm(rng.standard_normal((ncand,d)))
            D=Z@Xkeep.T
            worst=D.max(axis=1)
            order=np.argsort(worst)
            holes=[]
            for idx in order:
                z=Z[idx]
                ok=True
                for h in holes:
                    if float(h@z)>0.86:
                        ok=False; break
                if ok:
                    holes.append(z)
                    if len(holes)>=ns: break
            holes=np.array(holes)
            xt=best.copy()
            for s,h in zip(sources,holes):
                xt[int(s)]=h+0.008*rng.standard_normal(d)
            xt=_norm(xt)
            xt=_soft_relax(xt,steps=550,beta0=22.0,beta1=420.0,lr0=0.04,lr1=0.003,mu=0.87)
        else:
            # worst-pair shake
            flat=int(np.argmax(Gc))
            wi,wj=divmod(flat,n)
            tgt=wi if cnt[wi]>=cnt[wj] else wj
            nbrs=np.where(Gc[tgt]>mcur-5e-4)[0]
            if len(nbrs)>12:
                nbrs=nbrs[np.argsort(-Gc[tgt][nbrs])[:12]]
            xt=best.copy()
            xt[tgt]+=0.28*rng.standard_normal(d)
            if len(nbrs)>0:
                xt[nbrs]+=0.07*rng.standard_normal((len(nbrs),d))
            xt+=0.006*rng.standard_normal((n,d))
            xt=_norm(xt)
            xt=_soft_relax(xt,steps=500,beta0=28.0,beta1=380.0,lr0=0.04,lr1=0.0035,mu=0.85)
        # squeeze
        xt=_soft_relax(xt,steps=200,beta0=120.0,beta1=900.0,lr0=0.012,lr1=0.0015,mu=0.88)
        try:
            vt=_max_cos(xt)
        except Exception:
            continue
        if vt<best_val-1e-12:
            remain=EXPLORE_END-(time.time()-t0)
            if remain>1.0:
                xp=_lp_polish(xt,seconds=min(2.0,max(0.6,remain-0.5)),r0=9e-4)
                vp=_max_cos(xp)
                if vp<best_val-1e-12:
                    best=xp; best_val=vp
                else:
                    best=_norm(xt); best_val=vt
            else:
                best=_norm(xt); best_val=vt
        elif vt<best_val+0.0012 and (it%10==0):
            if EXPLORE_END-(time.time()-t0)>2.0:
                xp=_lp_polish(xt,seconds=0.7,r0=8e-4)
                vp=_max_cos(xp)
                if vp<best_val-1e-12:
                    best=xp; best_val=vp
        if time.time()>=EXPLORE_END:
            break

    elapsed=time.time()-t0
    remain=min(28.0,max(0.0,DEAD-elapsed-1.0))
    if remain>1.2:
        best=_lp_polish(best,seconds=remain,r0=7e-4)
    best=_norm(best)
    # safety: return better of best and incumbent and sym
    cands=[best,incumbent]
    if sym_best is not None:
        cands.append(sym_best)
    vals=[_max_cos(c) for c in cands]
    best=cands[int(np.argmin(vals))]
    return _norm(best)
