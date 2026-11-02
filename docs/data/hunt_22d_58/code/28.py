import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_quench_full(x, steps, b0, b1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    m2 = np.zeros_like(x)
    b_1 = 0.9; b_2 = 0.999; eps = 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        sW = float(W.sum())
        if not np.isfinite(sW) or sW == 0:
            break
        W /= sW
        Grad = W @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b_1 * m1 + (1 - b_1) * Gt
        m2 = b_2 * m2 + (1 - b_2) * (Gt * Gt)
        c1 = 1.0 - b_1 ** (s + 1)
        c2 = 1.0 - b_2 ** (s + 1)
        m1h = m1 / c1; m2h = m2 / c2
        step = lr * m1h / (np.sqrt(m2h) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 0.05
        scale = np.ones_like(sn)
        scale[over] = 0.05 / (sn[over] + 1e-12)
        step = step * scale
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 60:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-c)
        sel_order = []
        for idx in order:
            if c[idx] > m - window and len(sel_order) < 600:
                sel_order.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_order) >= 600:
                break
        if len(sel_order) == 0:
            break
        sel = np.array(sel_order)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(sel); nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        v = max_cos(y)
        if v < best - 1e-15:
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

# ---- symmetric C29 parametrization ----
def softmax(a):
    e = np.exp(a - np.max(a))
    s = e.sum()
    return e / s

def build_sym(a1, ph1, a2, ph2, K, TAU=2*np.pi/29.0):
    B = len(K)
    p1 = softmax(a1); r1 = np.sqrt(p1)
    p2 = softmax(a2); r2 = np.sqrt(p2)
    N = 29
    X1 = np.zeros((N, 2*B)); X2 = np.zeros((N, 2*B))
    js = np.arange(N)[:, None]  # 29x1
    Kb = K[None, :]  # 1xB
    th1 = ph1[None, :] + js * Kb * TAU  # 29xB
    th2 = ph2[None, :] + js * Kb * TAU
    c1 = np.cos(th1); s1 = np.sin(th1)
    c2 = np.cos(th2); s2 = np.sin(th2)
    for b in range(B):
        X1[:, 2*b] = r1[b] * c1[:, b]
        X1[:, 2*b+1] = r1[b] * s1[:, b]
        X2[:, 2*b] = r2[b] * c2[:, b]
        X2[:, 2*b+1] = r2[b] * s2[:, b]
    return np.vstack([X1, X2])

def sym_optimize(K, rng, steps=1200, b0=40.0, b1=1200.0, lr0=0.06, lr1=0.004):
    B = len(K)
    TAU = 2*np.pi/29.0
    a1 = 0.4*rng.standard_normal(B)
    a2 = 0.4*rng.standard_normal(B)
    ph1 = rng.uniform(0, 2*np.pi, size=B)
    ph2 = rng.uniform(0, 2*np.pi, size=B)
    th = np.concatenate([a1, ph1, a2, ph2])
    m1 = np.zeros_like(th); m2 = np.zeros_like(th)
    b_1=0.9; b_2=0.999; eps=1e-8
    lb0,lb1=np.log(b0),np.log(b1); ll0,ll1=np.log(lr0),np.log(lr1)
    best_val=1e9; best_th=th.copy()
    N=29
    js = np.arange(N)[:,None]; Kb=K[None,:]
    for s in range(steps):
        f=s/max(1,steps-1)
        beta=float(np.exp(lb0+(lb1-lb0)*f))
        lr=float(np.exp(ll0+(ll1-ll0)*f))
        a1=th[0:B]; ph1=th[B:2*B]; a2=th[2*B:3*B]; ph2=th[3*B:4*B]
        p1=softmax(a1); r1=np.sqrt(p1)+1e-12
        p2=softmax(a2); r2=np.sqrt(p2)+1e-12
        th1=ph1[None,:]+js*Kb*TAU
        th2=ph2[None,:]+js*Kb*TAU
        c1=np.cos(th1); s1=np.sin(th1); c2=np.cos(th2); s2=np.sin(th2)
        X1=np.zeros((N,2*B)); X2=np.zeros((N,2*B))
        for b in range(B):
            X1[:,2*b]=r1[b]*c1[:,b]; X1[:,2*b+1]=r1[b]*s1[:,b]
            X2[:,2*b]=r2[b]*c2[:,b]; X2[:,2*b+1]=r2[b]*s2[:,b]
        X=np.vstack([X1,X2])
        G=X@X.T
        np.fill_diagonal(G,-2.0)
        gmax=float(G.max())
        W=np.exp(beta*(G-gmax))
        np.fill_diagonal(W,0.0)
        sw=float(W.sum())
        if not np.isfinite(sw) or sw<=0:
            break
        W/=sw
        Gr=W@X
        G1=Gr[:N,:]; G2=Gr[N:,:]
        dr1=np.zeros(B); dp1=np.zeros(B); dr2=np.zeros(B); dp2=np.zeros(B)
        for b in range(B):
            gx=G1[:,2*b]; gy=G1[:,2*b+1]
            dr1[b]=float(np.sum(gx*c1[:,b]+gy*s1[:,b]))
            dp1[b]=float(np.sum(r1[b]*(-gx*s1[:,b]+gy*c1[:,b])))
            hx=G2[:,2*b]; hy=G2[:,2*b+1]
            dr2[b]=float(np.sum(hx*c2[:,b]+hy*s2[:,b]))
            dp2[b]=float(np.sum(r2[b]*(-hx*s2[:,b]+hy*c2[:,b])))
        u1=dr1*(1.0/(2*r1)); u2=dr2*(1.0/(2*r2))
        da1=p1*(u1-np.dot(u1,p1)); da2=p2*(u2-np.dot(u2,p2))
        grad=np.concatenate([da1,dp1,da2,dp2])
        # clip
        gn=float(np.linalg.norm(grad))
        if gn>5.0:
            grad*=5.0/gn
        m1=b_1*m1+(1-b_1)*grad
        m2=b_2*m2+(1-b_2)*(grad*grad)
        m1h=m1/(1-b_1**(s+1)); m2h=m2/(1-b_2**(s+1))
        step=lr*m1h/(np.sqrt(m2h)+eps)
        th=th-step
        # fix gauge: zero-mean logits
        th[0:B]-=np.mean(th[0:B]); th[2*B:3*B]-=np.mean(th[2*B:3*B])
        if (s%100==0) or s==steps-1:
            a1t=th[0:B]; p1t=th[B:2*B]; a2t=th[2*B:3*B]; p2t=th[3*B:4*B]
            try:
                Xt=build_sym(a1t,p1t,a2t,p2t,K)
                v=max_cos(Xt)
                if v<best_val:
                    best_val=v; best_th=th.copy()
            except Exception:
                pass
    return best_th, best_val

def solve():
    t0=time.time()
    TIME_LIMIT=165.0
    rng=np.random.default_rng(29)
    n,d=58,22
    best=None; best_val=1e9
    for f in ("best.npy","record.npy"):
        try:
            v=np.load(f).astype(float)
            if v.shape!=(58,22): continue
            v=v/np.linalg.norm(v,axis=1,keepdims=True)
            c=max_cos(v)
            if c<best_val: best_val=c; best=v.copy()
        except Exception: continue
    if best is None:
        z=rng.standard_normal((n,d)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        best=z; best_val=max_cos(z)
    incumbent=best.copy(); inc_val=best_val

    # ---- symmetric basin search ----
    sym_deadline=t0+50.0
    # K choices: base + random subsets of 1..14
    K_list=[]
    K_list.append(np.arange(1,12,dtype=float))
    for _ in range(6):
        ch=rng.choice(np.arange(1,15),size=11,replace=False)
        K_list.append(np.sort(ch).astype(float))
    # phase/scale variations via different seeds
    sym_cands=[]
    ki=0
    trial=0
    while time.time()<sym_deadline and trial<24:
        K=K_list[ki%len(K_list)]; ki+=1
        trial+=1
        # vary beta top for sharpness
        b1 = [900.0,1400.0,2000.0][trial%3]
        steps=1100 if time.time()<t0+35 else 700
        try:
            th_bt,v_bt=sym_optimize(K,rng,steps=steps,b0=40.0,b1=b1,lr0=0.06,lr1=0.004)
        except Exception:
            continue
        B=len(K)
        try:
            Xsym=build_sym(th_bt[0:B],th_bt[B:2*B],th_bt[2*B:3*B],th_bt[3*B:4*B],K)
        except Exception:
            continue
        sym_cands.append((v_bt,Xsym,K))
        if time.time()>=sym_deadline: break
    sym_cands.sort(key=lambda t:t[0])
    # ---- release symmetric cands to full coordinates ----
    # quick quench of top few
    for idx in range(min(4,len(sym_cands))):
        if time.time()>t0+TIME_LIMIT-30: break
        _,Xs,_=sym_cands[idx]
        y=Xs.copy()
        try:
            y=adam_quench_full(y,700,60,600,0.004,0.0005)
            y=adam_quench_full(y,250,800,2000,0.001,0.0003)
        except Exception: continue
        v=max_cos(y)
        if v<best_val-1e-12:
            try:
                y2=lp_polish(y,1.2,t0+TIME_LIMIT-5)
                v2=max_cos(y2)
                if v2<v: y,v=y2,v2
            except Exception: pass
            best_val=v; best=y.copy()
    # ---- extra hops mixing symmetric-derived best and incumbent ----
    hop_end=t0+TIME_LIMIT-20.0
    if hop_end>t0+135.0: hop_end=t0+135.0
    it=0
    # choose start pool
    pool=[best.copy()]
    if len(sym_cands)>0:
        # add raw best sym (unquenched) perturbed as alternative basin
        pool.append(sym_cands[0][1].copy())
    while time.time()<hop_end and it<40:
        it+=1
        if time.time()>t0+TIME_LIMIT-25: break
        base=pool[it%len(pool)] if len(pool)>0 else best
        # if best improved, refresh pool entry 0
        if it%7==0:
            base=best
        y=base.copy()
        mode=it%4
        if mode==0:
            y=y+0.004*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
        elif mode==1:
            y=y+0.009*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
        elif mode==2:
            k=2; idx2=rng.choice(n,size=k,replace=False)
            y=y+0.003*rng.standard_normal(y.shape)
            y/=np.linalg.norm(y,axis=1,keepdims=True)
            r=rng.standard_normal((k,d)); r/=np.linalg.norm(r,axis=1,keepdims=True)
            y[idx2]=r
        else:
            # worst-pair kick
            try:
                G=best@best.T; np.fill_diagonal(G,-2.0)
                ij=int(np.argmax(G)); i0=ij//n
                row=G[i0]; top=np.argsort(-row)[:3]
                cand=[int(i0)]+[int(t) for t in top]
                cand=np.array(cand[:4])
                y[cand]=y[cand]+0.03*rng.standard_normal((len(cand),d))
                y/=np.linalg.norm(y,axis=1,keepdims=True)
            except Exception:
                y=y+0.006*rng.standard_normal(y.shape)
                y/=np.linalg.norm(y,axis=1,keepdims=True)
        try:
            if mode in (0,1):
                y=adam_quench_full(y,700,80,600,0.004,0.0006)
                y=adam_quench_full(y,200,800,2000,0.001,0.0003)
            else:
                y=adam_quench_full(y,800,80,700,0.0035,0.0004)
                y=adam_quench_full(y,200,1000,2000,0.0008,0.0002)
        except Exception: continue
        v=max_cos(y)
        if v<best_val-1e-12:
            try:
                y2=lp_polish(y,1.0,t0+TIME_LIMIT-5)
                v2=max_cos(y2)
                if v2<v: y,v=y2,v2
            except Exception: pass
            best_val=v; best=y.copy(); pool[0]=y.copy()
    # ---- final polish ----
    remain=(t0+TIME_LIMIT)-time.time()-1.0
    if remain>4:
        lp_sec=min(remain-1.0,15.0)
        try:
            polished=lp_polish(best,lp_sec,t0+TIME_LIMIT-1.0)
            vp=max_cos(polished)
            if vp<best_val: best=polished; best_val=vp
        except Exception: pass
    # fallback to incumbent if somehow worse (numerics)
    if best_val>inc_val and incumbent is not None:
        # keep the better (lower) -> incumbent
        best=incumbent.copy()
    best=best/np.linalg.norm(best,axis=1,keepdims=True)
    best=np.nan_to_num(best,nan=0.0,posinf=0.0,neginf=0.0)
    nr=np.linalg.norm(best,axis=1,keepdims=True); nr[nr==0]=1.0
    best=best/nr
    return best
