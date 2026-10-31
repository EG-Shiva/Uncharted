import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def build_full(Rnorm, tables, Ns):
    # Rnorm: (nrep,4), tables: list of (c1,s1,c2,s2) each (M,)
    N = sum(Ns)
    X = np.empty((N,4))
    off = 0
    for o,(c1,s1,c2,s2) in enumerate(tables):
        r0,r1,r2,r3 = Rnorm[o]
        M = len(c1)
        # first block
        X[off:off+M,0] = c1*r0 - s1*r1
        X[off:off+M,1] = s1*r0 + c1*r1
        X[off:off+M,2] = c2*r2 - s2*r3
        X[off:off+M,3] = s2*r2 + c2*r3
        off += M
    return X

def make_tables(blocks):
    tables=[]; Ns=[]
    for (M,a,b) in blocks:
        k=np.arange(M)
        th1=2*np.pi*a*k/M; th2=2*np.pi*b*k/M
        c1=np.cos(th1); s1=np.sin(th1); c2=np.cos(th2); s2=np.sin(th2)
        tables.append((c1,s1,c2,s2)); Ns.append(M)
    return tables,Ns

def make_sym_obj(blocks, tables, Ns, ksharp, iu):
    nrep=len(blocks); N=sum(Ns)
    def fun(yflat):
        Y=yflat.reshape(nrep,4)
        norms=np.sqrt(np.sum(Y*Y,axis=1,keepdims=True))+1e-16
        R=Y/norms
        X=build_full(R,tables,Ns)
        G=X@X.T
        dots=G[iu]
        m=float(dots.max())
        e=np.exp(ksharp*(dots-m))
        S=float(e.sum())+1e-300
        L=m+np.log(S)/ksharp
        w=e/S
        # symmetric weight matrix times X
        W=np.zeros((N,N)); W[iu]=w
        Ws=W+W.T
        Ggrad=Ws@X  # (N,4)
        Dr=np.zeros((nrep,4))
        off=0
        for o,(c1,s1,c2,s2) in enumerate(tables):
            M=len(c1)
            g=Ggrad[off:off+M]  # Mx4
            g0=g[:,0]; g1=g[:,1]; g2=g[:,2]; g3=g[:,3]
            # Rot(-th)
            Dr[o,0]=np.sum(c1*g0+s1*g1)
            Dr[o,1]=np.sum(-s1*g0+c1*g1)
            Dr[o,2]=np.sum(c2*g2+s2*g3)
            Dr[o,3]=np.sum(-s2*g2+c2*g3)
            off+=M
        # project
        dot=np.sum(Dr*R,axis=1,keepdims=True)
        Gt=(Dr-dot*R)/norms
        return L, Gt.ravel()
    return fun

def sym_optimize(blocks,Y0,ksharp,maxiter):
    tables,Ns=make_tables(blocks)
    N=sum(Ns)
    iu=np.triu_indices(N,1)
    obj=make_sym_obj(blocks,tables,Ns,ksharp,iu)
    y0=Y0.ravel().copy()
    try:
        res=minimize(obj,y0,method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-6})
        Y=res.x.reshape(-1,4)
    except Exception:
        Y=Y0
    Y=Y/np.linalg.norm(Y,axis=1,keepdims=True)
    X=build_full(Y,tables,Ns)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X,Y,float(max_cos(X))

def lbfgs_smooth_all(x0,k,maxiter=80):
    n,d=x0.shape
    iu0=np.triu_indices(n,1)
    y0=x0.ravel().copy()
    def fun(y):
        Y=y.reshape(n,d)
        norms=np.sqrt(np.sum(Y*Y,axis=1,keepdims=True))+1e-16
        Xn=Y/norms
        G=Xn@Xn.T
        dots=G[iu0]
        m=float(dots.max())
        e=np.exp(k*(dots-m))
        S=float(e.sum())+1e-300
        L=m+np.log(S)/k
        w=e/S
        W=np.zeros((n,n)); W[iu0]=w; W=W+W.T
        GX=W@Xn
        dot=np.sum(GX*Xn,axis=1,keepdims=True)
        Gt=(GX-dot*Xn)/norms
        return L,Gt.ravel()
    try:
        res=minimize(fun,y0,method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-7})
        Y=res.x.reshape(n,d)
        Y/=np.linalg.norm(Y,axis=1,keepdims=True)
        return Y
    except Exception:
        return x0

def lp_polish(x, seconds, t_deadline=None):
    x=x/np.linalg.norm(x,axis=1,keepdims=True)
    n,d=x.shape
    best=max_cos(x); best_x=x.copy()
    radius=1e-3
    start=time.time()
    iu=np.triu_indices(n,1)
    while time.time()-start<seconds:
        if t_deadline is not None and time.time()>t_deadline:
            break
        if radius<1e-14:
            break
        Gfull=x@x.T
        c_full=Gfull[iu]
        m=c_full.max()
        sel=c_full>m-max(20*radius,1e-9)
        I=iu[0][sel]; J=iu[1][sel]; c=c_full[sel]
        p=len(I)
        if p==0:
            radius*=0.5; continue
        nv=n*d+1
        rows=np.repeat(np.arange(p),2*d+1)
        cols=np.concatenate([I[:,None]*d+np.arange(d),J[:,None]*d+np.arange(d),np.full((p,1),n*d)],1).ravel()
        vals=np.concatenate([x[J],x[I],-np.ones((p,1))],1).ravel()
        try:
            A_ub=coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq=coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
            obj=np.zeros(nv); obj[-1]=1.0
            res=linprog(obj,A_ub=A_ub,b_ub=-c,A_eq=A_eq,b_eq=np.zeros(n),bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        y=x+res.x[:-1].reshape(n,d)
        y/=np.linalg.norm(y,axis=1,keepdims=True)
        v=max_cos(y)
        if v<best:
            best=v; best_x=y.copy(); x=y; radius=min(radius*1.5,0.05)
        else:
            radius*=0.5
            if radius<1e-7:
                x=best_x.copy(); radius=1e-3
                if time.time()-start>seconds*0.9:
                    break
    return best_x

def solve():
    t0=time.time()
    DEADLINE=t0+168.0
    SYM_END=t0+85.0
    BRK_END=t0+148.0
    rng=np.random.default_rng(7)
    n,d=153,4
    try:
        best0=np.load("best.npy")
    except Exception:
        try: best0=np.load("record.npy")
        except Exception: best0=rng.standard_normal((n,d))
    best0=np.asarray(best0,dtype=float)
    if best0.shape!=(n,d):
        best0=rng.standard_normal((n,d))
    best0/=np.linalg.norm(best0,axis=1,keepdims=True)
    try:
        rec=np.load("record.npy"); rec=np.asarray(rec,dtype=float)
        if rec.shape==(n,d):
            rec/=np.linalg.norm(rec,axis=1,keepdims=True)
            if max_cos(rec)<max_cos(best0):
                best0=rec.copy()
    except Exception: pass
    incumbent=best0.copy(); inc_val=max_cos(incumbent)

    # --- symmetric screening ---
    configs=[]
    for r in range(0,9):
        configs.append(((17,r,1), [(17,r,1)]*9))
    configs.append(((17,1,0), [(17,1,0)]*9))
    for (a,b) in [(1,1),(1,2),(1,3),(1,5),(1,7),(2,3)]:
        configs.append(((51,a,b), [(51,a,b)]*3))
    # mixed 17: each orbit own random (a,b)
    for mi in range(3):
        bl=[]
        for _ in range(9):
            a=int(rng.integers(0,17)); b=int(rng.integers(0,17))
            if a==0 and b==0: b=1
            bl.append((17,a,b))
        configs.append(('mix%d'%mi, bl))

    best_sym=None; best_sym_val=1e9; best_sym_blocks=None
    cands=[]  # (val,X,Y,blocks)
    trial=0
    for _,blocks in configs:
        if time.time()>SYM_END: break
        nrep=len(blocks)
        # 3 inits per config cheap
        for _ in range(3):
            if time.time()>SYM_END: break
            trial+=1
            Y0=rng.standard_normal((nrep,4))
            Y0/=np.linalg.norm(Y0,axis=1,keepdims=True)
            # cheap: k=100, 40 iter
            X,Y,v=sym_optimize(blocks,Y0,ksharp=100.0,maxiter=40)
            cands.append((v,X.copy(),Y.copy(),blocks))
            if v<best_sym_val:
                best_sym_val=v; best_sym=X.copy(); best_sym_blocks=blocks
            # second cheap with different k if very good? skip
    # keep top 5
    cands.sort(key=lambda t: t[0])
    top=cands[:5]
    # full relax top
    for (v,X,Y,blocks) in top:
        if time.time()>SYM_END+12: break
        # continue from Y with higher sharpness
        try:
            X2,Y2,v2=sym_optimize(blocks,Y,ksharp=250.0,maxiter=120)
            if v2<best_sym_val:
                best_sym_val=v2; best_sym=X2.copy(); best_sym_blocks=blocks
            # extra sharpen
            if time.time()<SYM_END+12:
                X3,Y3,v3=sym_optimize(blocks,Y2,ksharp=450.0,maxiter=80)
                if v3<best_sym_val:
                    best_sym_val=v3; best_sym=X3.copy()
        except Exception:
            pass

    # if symmetric search failed, fallback
    if best_sym is None:
        best_sym=incumbent.copy(); best_sym_val=inc_val

    # --- gradual symmetry breaking ---
    # start from best symmetric full config
    cur_sym=best_sym.copy()
    # also consider best among top that may break better: try top[0..2]
    break_best=None; break_val=1e9
    # include incumbent as competitor at end
    # staged amps
    amps=[0.02,0.05,0.09,0.13]
    # try breaking each of top 3 sym configs
    sources=[]
    for i in range(min(3,len(top))):
        sources.append(top[i][1])
    if len(sources)==0:
        sources=[cur_sym]
    # use deterministic rng for breaking
    brng=np.random.default_rng(123)
    for s_idx,src in enumerate(sources):
        if time.time()>BRK_END: break
        Xb=src.copy()
        # chain of increasing amps
        for ai,amp in enumerate(amps):
            if time.time()>BRK_END: break
            Xp=Xb+amp*brng.standard_normal(Xb.shape)
            Xp/=np.linalg.norm(Xp,axis=1,keepdims=True)
            # smooth relax: two stages
            Xp=lbfgs_smooth_all(Xp,k=120.0,maxiter=60)
            if time.time()>BRK_END: break
            Xp=lbfgs_smooth_all(Xp,k=300.0,maxiter=60)
            v=max_cos(Xp)
            if v<break_val:
                break_val=v; break_best=Xp.copy()
            # chain: if improved or randomly, continue from Xp
            # greedy: continue from best of this chain
            # use Xp as next start if better than Xb
            if v<max_cos(Xb)+0.002:
                Xb=Xp.copy()
            else:
                # keep Xb but add smaller noise
                pass
        # extra random restart around break_best with medium amp
        if time.time()<BRK_END-10 and break_best is not None:
            Xp2=break_best+0.03*brng.standard_normal(break_best.shape)
            Xp2/=np.linalg.norm(Xp2,axis=1,keepdims=True)
            Xp2=lbfgs_smooth_all(Xp2,k=250.0,maxiter=80)
            v2=max_cos(Xp2)
            if v2<break_val:
                break_val=v2; break_best=Xp2.copy()

    if break_best is None:
        break_best=cur_sym.copy(); break_val=best_sym_val

    # choose overall to polish: best of break_best and incumbent
    # polish break_best with remaining time, then compare
    remain=max(0.0, DEADLINE-time.time()-2.0)
    # allocate ~ half to LP polish break_best
    lp_time=min(remain, 22.0)
    if lp_time>3 and time.time()<DEADLINE-5:
        break_best=lp_polish(break_best, seconds=lp_time, t_deadline=DEADLINE-2.0)
        break_val=max_cos(break_best)

    # final selection
    candidates=[(inc_val,incumbent),(break_val,break_best),(best_sym_val,best_sym)]
    candidates.sort(key=lambda t: t[0])
    final=candidates[0][1].copy()
    final/=np.linalg.norm(final,axis=1,keepdims=True)
    # tiny shrink guard + NaN check
    final=np.nan_to_num(final,nan=0.0,posinf=0.0,neginf=0.0)
    nr=np.linalg.norm(final,axis=1,keepdims=True)
    nr[nr<1e-12]=1.0
    final/=nr
    return final
