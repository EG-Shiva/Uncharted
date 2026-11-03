import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_generic(x0, steps, lr, kind, p_or_K0, p_or_K1=None, seed_off=0):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if kind == 'lse':
        la0, la1 = np.log(p_or_K0), np.log(p_or_K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps+1):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        if kind == 'riesz':
            p = p_or_K0
            # normalized Riesz weights
            gmax = float(G.max())
            umax = (1.0+gmax)*0.5
            if umax <= 0:
                umax = 0.5
            logUmax = np.log(umax+1e-300)
            U = (1.0+G)*0.5
            # log, clip diag later
            logU = np.log(np.maximum(U, 1e-300))
            E = (p*0.5-1.0)*(logU-lognp if False else logU-logUmax)
            # need l variable? fix
            E = (p*0.5-1.0)*(logU-logUmax)
            np.fill_diagonal(E, -100.0)
            E = np.clip(E, -50.0, 0.0)
            W = np.exp(E)
            np.fill_diagonal(W, 0.0)
        else:
            K = float(np.exp(la0+(la1-la0)*t/steps))
            gmax = float(G.max())
            W = np.exp(K*(G-gmax))
            np.fill_diagonal(W, 0.0)
        S = float(W.sum()*0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x)/S
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad-dot*x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1-b1)*gt
        v *= b2; v += (1-b2)*(gt*gt)
        mhat = m/(1-b1p); vhat = v/(1-b2p)
        step = lr*mhat/(np.sqrt(vhat)+eps)
        sdot = np.sum(step*x, axis=1, keepdims=True)
        step = step-sdot*x
        x = x-step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t%100==0 or t==steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val; best_x = x.copy()
    return best_x, best_val

def _adam_riesz(x0, steps, p, lr):
    return _adam_generic(x0, steps, lr, 'riesz', p)

def _adam_lse(x0, steps, K0, K1, lr):
    return _adam_generic(x0, steps, lr, 'lse', K0, K1)

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0*radius, 8e-5)
        sel = c > (mval-margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            radius *= 0.5; continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I=I[part]; J=J[part]; csel=csel[part]; pnum=cap
        rows = np.repeat(np.arange(pnum), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((pnum,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((pnum,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(pnum,n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,n*d+1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n*d+1); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y = x+res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y,axis=1,keepdims=True)
        val=_max_cos(y)
        if val<best:
            x,best=y,val
            radius=min(radius*1.4,0.02)
        else:
            radius*=0.5
    return x

def _void_point(x, rng, exclude=-1, trials=5000):
    n,d=x.shape
    Z=rng.normal(size=(trials,d))
    Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    S=Z@x.T
    if exclude>=0:
        S[:,exclude]=-2.0
    cm=S.max(axis=1)
    return Z[int(np.argmin(cm))]

def _dedupe_refill(child, rng, thresh=0.92):
    child=child/np.linalg.norm(child,axis=1,keepdims=True)
    n=child.shape[0]
    for _ in range(8):
        G=child@child.T
        np.fill_diagonal(G,-2.0)
        mv=float(G.max())
        if mv<thresh:
            break
        fi=int(np.argmax(G)); i=fi//n; j=fi%n
        # replace j (higher row max one)
        ri=float(G[i].max()); rj=float(G[j].max())
        k=j if rj>=ri else i
        child[k]=_void_point(child,rng,exclude=k,trials=4000)
    return child

def _fingerprint(x):
    G=x@x.T
    np.fill_diagonal(G,-2.0)
    c=np.sort((G[np.triu_indices(256,1)]))[::-1]
    top=c[:600:3]  # 200 dims
    mc=float(c[0])
    cnt1=int(np.sum(c>mc-1e-4)); cnt2=int(np.sum(c>mc-1e-3))
    return top, mc, (cnt1,cnt2)

def _align_procrustes(A,B):
    # Hungarian on -dot
    C=-(A@B.T)
    r,c=linear_sum_assignment(C)
    # r should be arange; ensure order by r
    perm=np.empty_like(c)
    perm[r]=c
    # Actually we want Bp aligned to A order: A[i] matches B[perm[i]]
    Bp=B[perm]
    M=Bp.T@A
    U,s,Vt=np.linalg.svd(M)
    R=U@Vt
    if np.linalg.det(R)<0:
        U[:,-1]*=-1
        R=U@Vt
    Ball=Bp@R
    return Ball, perm, R

def solve():
    import time as _time
    import numpy as _np
    t0=_time.time()
    DEADLINE=t0+178.0
    CENSUS_END=t0+95.0
    rng=_np.random.default_rng(12345)
    n,d=256,6
    NCODE=12
    parents=[]
    # de novo census
    for ci in range(NCODE):
        if _time.time()>CENSUS_END-5:
            break
        rs=_np.random.default_rng(1000+ci)
        x=rs.normal(size=(n,d)); x/= _np.linalg.norm(x,axis=1,keepdims=True)
        # Riesz anneal p=8,32,128,512 ~ 250 steps each (300 for first few if time)
        for p,lr in [(8,0.015),(32,0.012),(128,0.008),(512,0.006)]:
            steps=300 if ci<4 else 250
            if _time.time()>CENSUS_END-3:
                break
            try:
                x,_=_adam_riesz(x,steps,p,lr)
            except Exception:
                pass
        # sharp LSE
        if _time.time()<CENSUS_END-2:
            try:
                x,_=_adam_lse(x,300,200.0,4000.0,0.008)
            except Exception:
                pass
        val=_max_cos(x)
        top,mc,cnts=_fingerprint(x)
        parents.append({'x':x.copy(),'val':val,'fp':top,'cnts':cnts})
    if len(parents)==0:
        a=_np.load("best.npy"); a=a/_np.linalg.norm(a,axis=1,keepdims=True); return a
    parents.sort(key=lambda e:e['val'])
    # LP polish top 3 distinct
    for k in range(min(3,len(parents))):
        if _time.time()>CENSUS_END:
            break
        try:
            xp=_lp_polish(parents[k]['x'], min(_time.time()+4.0, CENSUS_END+6), r0=2e-3)
            vp=_max_cos(xp)
            if vp<parents[k]['val']:
                parents[k]['x']=xp; parents[k]['val']=vp
                top,mc,cnts=_fingerprint(xp); parents[k]['fp']=top; parents[k]['cnts']=cnts
        except Exception:
            pass
    parents.sort(key=lambda e:e['val'])
    # select genetically different: greedy by fp distance
    sel=[parents[0]]
    for e in parents[1:]:
        dmin=min([float(_np.linalg.norm(e['fp']-s['fp'])) for s in sel])
        if dmin>1e-3 or len(sel)<4:
            # require diversity or fill
            if dmin>5e-4:
                sel.append(e)
        if len(sel)>=4:
            break
    # ensure at least 3
    for e in parents:
        if len(sel)>=4: break
        if all(e['x'] is not s['x'] for s in sel):
            # check not already
            sel.append(e); break
    # crossover among sel
    children=[]
    pair_list=[]
    for i in range(len(sel)):
        for j in range(i+1,len(sel)):
            pair_list.append((i,j))
    # sort pairs by diversity (largest fp distance first)
    def pdist(p):
        return float(_np.linalg.norm(sel[p[0]]['fp']-sel[p[1]]['fp']))
    pair_list.sort(key=pdist, reverse=True)
    for (i,j) in pair_list[:3]:
        if _time.time()>DEADLINE-30:
            break
        A=sel[i]['x']; B=sel[j]['x']
        try:
            Ball,perm,R=_align_procrustes(A,B)
        except Exception:
            continue
        # 128 lowest-row-max points of each
        try:
            ra,_=_row_max(A); rb,_=_row_max(Ball)
        except Exception:
            continue
        ia=_np.argsort(ra)[:128]; ib=_np.argsort(rb)[:128]
        child=_np.vstack([A[ia],Ball[ib]])
        # dedupe + void refill
        try:
            child=_dedupe_refill(child,rng,0.90)
            # extra void refill for worst points of child? replace 2 worst with voids? no, keep structure
        except Exception:
            pass
        # repolish: short Riesz + LSE
        try:
            if _time.time()<DEADLINE-25:
                child,_=_adam_riesz(child,200,128,0.008)
            if _time.time()<DEADLINE-20:
                child,_=_adam_lse(child,300,300.0,5000.0,0.008)
            val=_max_cos(child)
            top,mc,cnts=_fingerprint(child)
            children.append({'x':child,'val':val,'fp':top})
            # second child: complementary halves (worst halves mixed?) to test other basin
            if _time.time()<DEADLINE-25:
                ia2=_np.argsort(ra)[128:]; ib2=_np.argsort(rb)[128:]
                # mix worst of A with best of B? Actually build alternative: 128 random from each
                # use interleaved: take 64 best +64 random from each
                r2=_np.random.default_rng(9000+i*10+j)
                pa=r2.choice(256,128,replace=False); pb=r2.choice(256,128,replace=False)
                # align again? reuse Ball
                child2=_np.vstack([A[pa[:128]//1][:0] if False else A[_np.argsort(ra)[::2][:128]], Ball[_np.argsort(rb)[1::2][:128]]])
                child2=_dedupe_refill(child2,rng,0.90)
                child2,_=_adam_lse(child2,200,300.0,4000.0,0.008)
                v2=_max_cos(child2)
                top2,_,_= _fingerprint(child2)
                children.append({'x':child2,'val':v2,'fp':top2})
        except Exception:
            pass
    # pool all
    pool=parents+children
    # light LP on best children
    pool.sort(key=lambda e:e['val'])
    for k in range(min(3,len(pool))):
        if _time.time()>DEADLINE-8:
            break
        try:
            xp=_lp_polish(pool[k]['x'], _time.time()+3.0, r0=2.5e-3)
            vp=_max_cos(xp)
            if vp<pool[k]['val']:
                pool[k]['x']=xp; pool[k]['val']=vp
        except Exception:
            pass
    pool.sort(key=lambda e:e['val'])
    best_x=pool[0]['x']; best_v=pool[0]['val']
    # fallback to best.npy if we failed (ensure no regression)
    try:
        b=_np.load("best.npy"); b=_np.asarray(b,dtype=float)
        if b.shape==(256,6):
            b=b/_np.linalg.norm(b,axis=1,keepdims=True)
            bv=_max_cos(b)
            if bv<best_v:
                best_x=b; best_v=bv
    except Exception:
        pass
    try:
        r=_np.load("record.npy"); r=_np.asarray(r,dtype=float)
        if r.shape==(256,6):
            r=r/_np.linalg.norm(r,axis=1,keepdims=True)
            rv=_max_cos(r)
            if rv<best_v:
                best_x=r; best_v=rv
    except Exception:
        pass
    best_x=best_x/_np.linalg.norm(best_x,axis=1,keepdims=True)
    return best_x
