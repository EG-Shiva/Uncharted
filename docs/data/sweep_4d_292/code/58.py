import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normr(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _lp_polish_deadline(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normr(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            # keep tightest only to keep LP tractable
            order = np.argsort(c[sel])[::-1][:2500]
            I, J = I[order], J[order]
            p = len(I)
            csel = c[sel][order]
        else:
            csel = c[sel]
        nv = n*d+1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            break
        if getattr(res,"status",1)!=0:
            radius*=0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n,d)
        except Exception:
            radius*=0.5
            continue
        y = _normr(y)
        try:
            vv = _max_cos(y)
        except Exception:
            radius*=0.5
            continue
        if vv < best:
            x, best, radius = y, vv, min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return x

def _anneal(x, steps, beta0, beta1, lr0, mu):
    n = x.shape[0]
    v = np.zeros_like(x)
    lb = np.log(beta1/beta0) if beta1>beta0 else 0.0
    for t in range(steps):
        frac = t/max(1,steps-1)
        beta = beta0*np.exp(lb*frac)
        lr = lr0*(1.0-0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1,keepdims=True)+1e-300
        A = (W@x)/S
        dot = (A*x).sum(axis=1,keepdims=True)
        Gt = A-dot*x
        v = mu*v+Gt
        vd = (v*x).sum(axis=1,keepdims=True)
        v -= vd*x
        x = x-lr*v
        x = _normr(x)
    return x

def _active_pairs(x, tol=5e-9, cap=1200):
    n = x.shape[0]
    iu = np.triu_indices(n,1)
    c = (x@x.T)[iu]
    m = float(c.max())
    sel = c > m - tol
    if int(sel.sum()) > cap:
        order = np.argsort(c)[::-1][:cap]
        mask = np.zeros_like(sel, dtype=bool)
        mask[order]=True
        sel = mask
    I = iu[0][sel]; J = iu[1][sel]
    return I, J, c[sel], m, iu, c

def _equilibrium_stresses(x, I, J, cact):
    n,d = x.shape
    p = len(I)
    if p==0:
        return np.ones(0)
    # A is (n*d) x p
    A = np.zeros((n*d,p), dtype=np.float64)
    for k in range(p):
        i=int(I[k]); j=int(J[k]); cv=float(cact[k])
        ti = x[j]-cv*x[i]
        tj = x[i]-cv*x[j]
        A[i*d:(i+1)*d, k]=ti
        A[j*d:(j+1)*d, k]=tj
    G = A.T @ A
    # regularize
    tr = float(np.trace(G)/max(1,p))
    if not np.isfinite(tr) or tr<=0:
        tr=1.0
    eps = 1e-7*tr + 1e-12
    Greg = G + eps*np.eye(p)
    ones = np.ones(p)
    try:
        s = np.linalg.solve(Greg, ones)
    except Exception:
        try:
            s = np.linalg.lstsq(Greg, ones, rcond=None)[0]
        except Exception:
            return np.ones(p)
    # normalize mean 1
    mu = float(s.mean())
    if np.isfinite(mu) and abs(mu)>1e-12:
        s = s/mu
    return s

def _hinge_fg_factory(n, T, beta, DI, DJ, w_extra, beta_extra, T_extra):
    dset = None
    if DI is not None and len(DI)>0:
        dset = (np.asarray(DI,dtype=np.int64), np.asarray(DJ,dtype=np.int64))
    def fg(yflat):
        Y = yflat.reshape(n,4)
        nrm = np.linalg.norm(Y,axis=1,keepdims=True)
        nrm = np.maximum(nrm,1e-12)
        X = Y/nrm
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - T
        U = beta*D
        # softplus stable
        sp = np.logaddexp(0.0, U)
        np.fill_diagonal(sp, 0.0)
        sig = 0.5*(1.0+np.tanh(0.5*U))
        np.fill_diagonal(sig, 0.0)
        # objective
        F = np.sum((sp/beta)**2)*0.5
        W = (2.0*sp*sig/beta)
        np.fill_diagonal(W, 0.0)
        if dset is not None:
            di,dj = dset
            # gather values at deactivated pairs
            # compute extra from current G
            for idx in range(len(di)):
                a=int(di[idx]); b=int(dj[idx])
                cab = float(X[a]@X[b])
                ue = beta_extra*(cab - T_extra)
                spe = float(np.logaddexp(0.0, ue))
                sige = 0.5*(1.0+np.tanh(0.5*ue))
                F += w_extra*(spe/beta_extra)**2
                wder = w_extra*2.0*spe*sige/beta_extra
                W[a,b]+=wder
                W[b,a]+=wder
        # symmetrize (U symmetric, W symmetric already except extra double? extra added both)
        # W currently holds full values; gradient = W @ X but double counts? Check: sum_{i<j} -> full/2, W full holds df/dc per ordered pair? Our F used 0.5*sum_full, W= d/dc per full entry? dF/d G_ij (i!=j, full) = 0.5*2*sp*sig/beta = sp*sig/beta. But we set W=2*sp*sig/beta -> factor 2 too big. Fix: halve.
        W = W*0.5
        # extra: we added wder to both (a,b) and (b,a); F counts once, derivative per ordered entry is 0.5*wder_full? We added wder to each ordered entry, so with halving gives correct 0.5*wder each -> sum gives wder. Good.
        gradX = W @ X
        dot = (gradX*X).sum(axis=1,keepdims=True)
        gradY = (gradX - dot*X)/nrm
        return float(F), gradY.ravel()
    return fg

def _hinge_opt(X0, T, DI, DJ, beta, maxiter, w_extra=8.0):
    from scipy.optimize import minimize
    n = X0.shape[0]
    T_extra = T - 1e-5
    beta_extra = beta
    fg = _hinge_fg_factory(n, float(T), float(beta), DI, DJ, float(w_extra), float(beta_extra), float(T_extra))
    y0 = X0.ravel().copy()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'ftol':1e-12, 'gtol':1e-7, 'maxfun':100000})
        y = res.x
    except Exception:
        y = y0
    Y = y.reshape(n,4)
    X = _normr(Y)
    return X

def solve():
    t0=time.time()
    TLIM=170.0
    rng=np.random.default_rng(29229238)
    n,d=292,4
    cands=[]
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn)
            a=np.asarray(a,dtype=np.float64)
            if a.shape==(n,d) and np.all(np.isfinite(a)):
                nr=np.linalg.norm(a,axis=1,keepdims=True)
                if float(nr.min())>1e-9:
                    cands.append(a/nr)
        except Exception:
            pass
    if not cands:
        Xb=rng.standard_normal((n,d)); Xb=_normr(Xb)
    else:
        vals=[_max_cos(c) for c in cands]
        Xb=cands[int(np.argmin(vals))].copy()
    best_val=_max_cos(Xb)
    X_best=Xb.copy()
    # --- stress analysis on incumbent ---
    try:
        I,J,cact,mval,iu_all,call_ = _active_pairs(X_best, tol=5e-9, cap=1200)
        p=len(I)
        if p<400:
            I,J,cact,mval,iu_all,call_ = _active_pairs(X_best, tol=2e-7, cap=1200)
            p=len(I)
    except Exception:
        I,J,cact,p=np.zeros(0,dtype=int),np.zeros(0,dtype=int),np.zeros(0),0
    try:
        s=_equilibrium_stresses(X_best,I,J,cact) if len(I)>0 else np.ones(0)
    except Exception:
        s=np.ones(len(I))
    deact_sets=[]
    try:
        if len(s)>0:
            order=np.argsort(s)
            neg=np.where(s<0)[0]
            # set A: most negative (tensile) up to 45
            if len(neg)>0:
                take=neg[np.argsort(s[neg])][:45]
                deact_sets.append((I[take],J[take]))
            # set B: largest 40
            take=np.argsort(s)[::-1][:40]
            deact_sets.append((I[take],J[take]))
            # set C: smallest 40
            take=np.argsort(s)[:40]
            deact_sets.append((I[take],J[take]))
            # set D: contacts of highest-degree nodes with extreme stress
            degs=np.zeros(n,dtype=int)
            for a,b in zip(I,J):
                degs[int(a)]+=1; degs[int(b)]+=1
            hubs=np.argsort(degs)[::-1][:12]
            hubmask=np.zeros(len(I),dtype=bool)
            for h in hubs:
                hubmask|= (I==h)|(J==h)
            idx=np.where(hubmask)[0]
            if len(idx)>0:
                # pick extreme among hub contacts
                o=np.argsort(np.abs(s[idx]-1.0))[::-1][:40]
                take=idx[o]
                deact_sets.append((I[take],J[take]))
            # set E: random 40 active
            take=rng.choice(len(I), size=min(40,len(I)), replace=False)
            deact_sets.append((I[take],J[take]))
        else:
            take=rng.choice(n*(n-1)//2, size=40)
    except Exception:
        pass
    # fallback random tight pairs if empty
    if not deact_sets:
        try:
            G=X_best@X_best.T; np.fill_diagonal(G,-2.0)
            iu=np.triu_indices(n,1)
            c=(G[iu])
            o=np.argsort(c)[::-1][:200]
            take=rng.choice(o, size=40, replace=False)
            deact_sets.append((iu[0][take], iu[1][take]))
        except Exception:
            deact_sets.append((np.zeros(0,dtype=int),np.zeros(0,dtype=int)))
    # dedupe, limit to 5 sets
    deact_sets=deact_sets[:5]
    Targets=[0.90340,0.90335,0.90325]
    # quick record of trials
    explore_end=t0+TLIM-22.0
    # try hinge trials interleaved
    trial_plan=[]
    for di,(DI,DJ) in enumerate(deact_sets):
        for T in Targets:
            trial_plan.append((di,T))
    # order: prioritize middle T and first sets; shuffle deterministically but keep T=0.90340 first
    # sort by T ascending? want easiest first (0.90340)
    trial_plan.sort(key=lambda t: (t[1], t[0]))
    # limit number by time: estimate 4s per trial -> max ~ (explore_end-now)/4
    for (di,T) in trial_plan:
        if time.time()>explore_end-4.0:
            break
        DI,DJ=deact_sets[di]
        # perturb to leave basin
        sigma=float(rng.choice([0.006,0.012,0.025]))
        N=rng.standard_normal(X_best.shape)*sigma
        N-=(N*X_best).sum(axis=1,keepdims=True)*X_best
        X0=X_best+N
        X0=_normr(X0)
        rem=explore_end-time.time()
        if rem<4.0:
            break
        # two-stage beta
        try:
            X1=_hinge_opt(X0, T, DI, DJ, beta=600.0, maxiter=180, w_extra=8.0)
        except Exception:
            continue
        if time.time()>explore_end-2.0:
            # evaluate anyway
            try:
                v=_max_cos(X1)
                if v<best_val:
                    X_best, best_val=X1.copy(), float(v)
            except Exception:
                pass
            continue
        try:
            X2=_hinge_opt(X1, T, DI, DJ, beta=1500.0, maxiter=120, w_extra=12.0)
        except Exception:
            X2=X1
        try:
            v2=_max_cos(X2)
        except Exception:
            continue
        # anneal briefly to spread load (softmax) starting from hinge result
        try:
            X3=_anneal(X2.copy(), 260, 25.0, 450.0, 0.055, 0.86)
            v3=_max_cos(X3)
        except Exception:
            X3, v3=X2, v2
        # keep best among X2,X3
        if v3<v2:
            Xc, vc=X3, v3
        else:
            Xc, vc=X2, v2
        if vc<best_val:
            # LP only after smooth violation small (as directed): violation = vc - T
            viol=vc-float(T)
            if viol<5e-04:
                dl=min(time.time()+2.2, explore_end)
                if dl-time.time()>0.6:
                    try:
                        Y=_lp_polish_deadline(Xc, dl)
                        vy=_max_cos(Y)
                        if vy<vc:
                            Xc, vc=Y, vy
                    except Exception:
                        pass
            if vc<best_val:
                X_best, best_val=Xc.copy(), float(vc)
                # early exit if we beat record by decent margin? continue to push lower T
                # update stresses? keep original deactivation (topology changed) - recompute for next trials occasionally
                if len(deact_sets)>0 and (best_val<0.90340):
                    pass
        else:
            # near-feasible failure: if within 3e-04 of best, LP polish to get structurally different local optimum
            if vc<best_val+3e-04:
                dl=min(time.time()+1.5, explore_end)
                if dl-time.time()>0.5:
                    try:
                        Y=_lp_polish_deadline(Xc, dl)
                        vy=_max_cos(Y)
                        if vy<best_val:
                            X_best, best_val=Y.copy(), float(vy)
                    except Exception:
                        pass
        # if we achieved feasibility for this T, try to push further immediately with same basin
        if vc<float(T)+1e-12:
            # feasible win: quick LP and continue
            try:
                dl=min(time.time()+1.5, explore_end)
                if dl-time.time()>0.5:
                    Y=_lp_polish_deadline(X_best, dl)
                    vy=_max_cos(Y)
                    if vy<best_val:
                        X_best, best_val=Y.copy(), float(vy)
            except Exception:
                pass
    # final generic basin-hopping if time remains and no win yet (topology-changing kicks)
    while time.time()<explore_end-1.0:
        rem=explore_end-time.time()
        if rem<1.2:
            break
        # subset kick from overloaded vs hole
        X=X_best.copy()
        try:
            G=X@X.T; np.fill_diagonal(G,-2.0)
            rm=G.max(axis=1)
            k=int(rng.choice([6,10,16]))
            idx=np.argsort(rm)[-k:]
            S=rng.standard_normal((1500,4)); S=_normr(S)
            sc=S@X.T
            m2=sc.max(axis=1)
            order=np.argsort(m2)
            for j,ii in enumerate(idx):
                X[ii]=S[order[j%len(order)]]
            X=_normr(X)
            X=_anneal(X, 220 if rem>8 else 120, 20.0, 400.0, 0.06, 0.86)
            v=_max_cos(X)
            if v<best_val:
                dl=min(time.time()+1.8, explore_end)
                if dl-time.time()>0.5:
                    try:
                        Y=_lp_polish_deadline(X, dl)
                        vy=_max_cos(Y)
                        if vy<v:
                            X,v=Y,vy
                    except Exception:
                        pass
                if v<best_val:
                    X_best, best_val=X.copy(), float(v)
            elif v<best_val+2e-04 and rem>3.0:
                dl=min(time.time()+1.2, explore_end)
                if dl-time.time()>0.4:
                    try:
                        Y=_lp_polish_deadline(X, dl)
                        vy=_max_cos(Y)
                        if vy<best_val:
                            X_best, best_val=Y.copy(), float(vy)
                    except Exception:
                        pass
        except Exception:
            break
        # only one or two kicks to save time for final polish
        if time.time()>explore_end-6.0:
            break
    try:
        if time.time()<t0+TLIM-8.0:
            X_best=_lp_polish_deadline(X_best, t0+TLIM-1.5)
    except Exception:
        pass
    X_best=_normr(X_best)
    return X_best
