import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
    n, d13, d14 = 74, 13, 14

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d13):
                cands.append(_normalize(a))
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n, d13))))
    vals = [_max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    IU13 = np.triu_indices(n, 1)
    I0 = IU13[0].astype(np.int64); J0 = IU13[1].astype(np.int64)

    # ---------- 13D polish helpers ----------
    def lse13(X0, tau, maxiter):
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d13)
            r2 = (Y*Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau*np.log(S+1e-300)
            w = e/(S+1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:,None]*X[J0])
            np.add.at(Wx, J0, w[:,None]*X[I0])
            dot = (Wx*X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot*X)/r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter":int(maxiter),"ftol":1e-20,"gtol":1e-7,"maxfun":int(maxiter*8)})
            Y = res.x.reshape(n, d13)
        except Exception:
            return X0
        rn = np.linalg.norm(Y,axis=1,keepdims=True); rn[rn==0]=1.0
        X = Y/rn
        if not np.all(np.isfinite(X)):
            return X0
        return _normalize(X)

    def greedy13(X, steps, lr0, deadline):
        cur = _max_cos(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            if time.time() >= deadline:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(5.0*lr, 2e-6)
            mask = G > (gmax-delta)
            np.fill_diagonal(mask, False)
            Gr = mask.astype(np.float64) @ X
            dd = (Gr*X).sum(axis=1, keepdims=True)
            Gt = Gr - dd*X
            nr = np.linalg.norm(Gt,axis=1,keepdims=True)
            act = (nr[:,0]>1e-14)
            if not np.any(act):
                break
            Step = np.zeros_like(Gt)
            Step[act] = Gt[act]/(nr[act]+1e-12)*lr
            Xn = _normalize(X-Step)
            nv = _max_cos(Xn)
            if nv < cur-1e-15:
                X[:] = Xn; cur = nv
                lr = min(lr*1.08, 8e-5)
            else:
                lr *= 0.5
                if lr < 2e-11:
                    break
        return X, cur

    # ---------- 14D helpers ----------
    IU14 = np.triu_indices(n, 1)
    I14 = IU14[0].astype(np.int64); J14 = IU14[1].astype(np.int64)

    def lse14(Y0, tau, maxiter):
        y0 = Y0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d14)
            r2 = (Y*Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2,1e-24)); r=np.maximum(r,1e-12)
            Yn = Y/r
            G = Yn @ Yn.T
            g = G[I14, J14]
            mmax = float(g.max())
            e = np.exp((g-mmax)/tau)
            S = float(e.sum())
            if not np.isfinite(S) or S<=0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau*np.log(S+1e-300)
            w = e/(S+1e-300)
            Wx = np.zeros_like(Yn)
            np.add.at(Wx, I14, w[:,None]*Yn[J14])
            np.add.at(Wx, J14, w[:,None]*Yn[I14])
            dot=(Wx*Yn).sum(axis=1,keepdims=True)
            gradY=(Wx-dot*Yn)/r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun,y0,jac=True,method="L-BFGS-B",
                           options={"maxiter":int(maxiter),"ftol":1e-20,"gtol":1e-7,"maxfun":int(maxiter*6)})
            Y = res.x.reshape(n,d14)
        except Exception:
            return Y0
        rn=np.linalg.norm(Y,axis=1,keepdims=True); rn[rn==0]=1.0
        Y=Y/rn
        if not np.all(np.isfinite(Y)):
            return Y0
        return Y

    def proj_obj(Yn, W, tau, lam):
        # Yn assumed row-normalized (n,14), W orthonormal (14,13)
        P = Yn @ W
        rn2 = (P*P).sum(axis=1)
        rn = np.sqrt(np.maximum(rn2,1e-24))
        rn = np.maximum(rn,1e-12)
        Xp = P/np.maximum(rn[:,None],1e-12)
        G = Xp @ Xp.T
        g = G[I0,J0]
        mmax = float(g.max())
        e = np.exp((g-mmax)/tau)
        S = float(e.sum())
        F = mmax + tau*np.log(S+1e-300)
        resid = 1.0 - rn2  # >=0 ideally (<=1); can be slightly neg due to numerics
        pen = float(np.mean(resid))
        return F + lam*pen, float(mmax), float(np.max(resid)), (Xp, P, rn, g, e, S)

    def polish_Y_fixedW(Y0n, W, tau, lam, maxiter):
        y0 = Y0n.ravel().copy()  # start already normalized; optimize free then normalize
        def fun(yf):
            Y = yf.reshape(n,d14)
            r2=(Y*Y).sum(axis=1,keepdims=True)
            r=np.sqrt(np.maximum(r2,1e-24)); r=np.maximum(r,1e-12)
            Yn=Y/r
            P=Yn@W
            rn2=(P*P).sum(axis=1)
            rn=np.sqrt(np.maximum(rn2,1e-24)); rn=np.maximum(rn,1e-12)
            Xp=P/rn[:,None]
            G=Xp@Xp.T
            g=G[I0,J0]
            mmax=float(g.max())
            e=np.exp((g-mmax)/tau)
            S=float(e.sum())
            if not np.isfinite(S) or S<=0:
                return float(mmax), np.zeros_like(yf)
            F=mmax+tau*np.log(S+1e-300)
            pen=float(np.mean(1.0-rn2))
            Ftot=F+lam*pen
            w=e/(S+1e-300)
            Gx=np.zeros_like(Xp)
            np.add.at(Gx,I0,w[:,None]*Xp[J0])
            np.add.at(Gx,J0,w[:,None]*Xp[I0])
            # backprop Xp->P
            dot=(Gx*Xp).sum(axis=1,keepdims=True)
            Dp=(Gx-dot*Xp)/rn[:,None]
            # penalty part d/dP
            Dp = Dp - (2.0*lam/n)*P
            # d/dYn
            Gy=Dp@W.T
            dotY=(Gy*Yn).sum(axis=1,keepdims=True)
            gradY=(Gy-dotY*Yn)/r
            return float(Ftot), gradY.ravel()
        try:
            res=minimize(fun,y0,jac=True,method="L-BFGS-B",
                         options={"maxiter":int(maxiter),"ftol":1e-20,"gtol":1e-6,"maxfun":int(maxiter*6)})
            Y=res.x.reshape(n,d14)
        except Exception:
            return Y0n
        rn=np.linalg.norm(Y,axis=1,keepdims=True); rn[rn==0]=1.0
        Y=Y/rn
        if not np.all(np.isfinite(Y)):
            return Y0n
        return Y

    def w_steps(Yn, W, tau, lam, nsteps):
        # Riemannian gradient descent with QR retraction + backtracking
        for _ in range(int(nsteps)):
            if time.time()-t0 > LIM-3:
                break
            P=Yn@W
            rn2=(P*P).sum(axis=1)
            rn=np.sqrt(np.maximum(rn2,1e-24)); rn=np.maximum(rn,1e-12)
            Xp=P/rn[:,None]
            G=Xp@Xp.T
            g=G[I0,J0]
            mmax=float(g.max())
            e=np.exp((g-mmax)/tau)
            S=float(e.sum())
            F=mmax+tau*np.log(S+1e-300)+lam*float(np.mean(1.0-rn2))
            w=e/(S+1e-300)
            Gx=np.zeros_like(Xp)
            np.add.at(Gx,I0,w[:,None]*Xp[J0])
            np.add.at(Gx,J0,w[:,None]*Xp[I0])
            dot=(Gx*Xp).sum(axis=1,keepdims=True)
            Dp=(Gx-dot*Xp)/rn[:,None] - (2.0*lam/n)*P
            Gw=Yn.T@Dp  # 14x13 Euclidean grad
            Sym=(W.T@Gw + (W.T@Gw).T)*0.5
            Rg=Gw - W@Sym
            nr=float(np.linalg.norm(Rg))
            if not np.isfinite(nr) or nr<1e-14:
                break
            Rg/= (nr+1e-12)
            # backtrack
            lr=0.2
            improved=False
            for _bt in range(12):
                Wt = W - lr*Rg*nr*0.2  # scale by grad norm
                # QR retraction
                try:
                    Q,Rm=np.linalg.qr(Wt)
                except Exception:
                    lr*=0.5; continue
                # fix sign for determinism (not essential)
                val,_,_,_2 = proj_obj(Yn,Q,tau,lam)[:4]
                # proj_obj returns tuple; get first
                if val < F - 1e-12:
                    W=Q; improved=True; break
                lr*=0.5
            if not improved:
                break
        return W

    # ---------- lift-and-squeeze trials ----------
    sigmas = [0.01, 0.03, 0.08, 0.03, 0.01]
    seeds  = [11, 22, 33, 44, 55]

    # quick baseline polish of best
    if time.time()-t0 < LIM-10:
        try:
            Xb=lse13(best,3e-4,60)
            vb=_max_cos(Xb)
            if vb<best_val: best_val,best=float(vb),Xb.copy()
        except Exception: pass

    ntrials = 0
    for sigma, seed in zip(sigmas, seeds):
        if time.time()-t0 > LIM-28:
            break
        ntrials += 1
        r2 = np.random.default_rng(seed)
        try:
            z = r2.standard_normal(n)*sigma
            z = np.clip(z, -0.85, 0.85)
            sc = np.sqrt(np.maximum(1.0-z*z, 1e-12))
            Y0 = np.zeros((n,d14))
            Y0[:,:d13]=best*sc[:,None]
            Y0[:,d13]=z
            Y0=_normalize(Y0)
            # small random 14D rotation via skew exp to diversify subspace
            A = r2.standard_normal((d14,d14))*0.25
            Sk = (A-A.T)*0.5
            R = expm(Sk* (0.5+sigma*4.0))
            Y0 = _normalize(Y0 @ R)
            # 14D free optimization: extra freedom lowers max & rearranges
            Y0 = lse14(Y0, 0.01, 50)
            if time.time()-t0 > LIM-25: 
                # still need squeeze; continue
                pass
            Y0 = lse14(Y0, 0.004, 40)
            # initial basis: identity subspace (first 13 coords)
            W = np.zeros((d14,d13)); W[:d13,:d13]=np.eye(d13)
            # squeeze schedule: lam geometric, tau shrinking
            lams = [0.5, 3.0, 15.0, 80.0, 400.0, 2000.0, 12000.0, 60000.0, 250000.0]
            taus = [0.004, 0.003, 0.0022, 0.0015, 0.001, 7e-4, 5e-4, 4e-4, 3e-4]
            for li, lam in enumerate(lams):
                if time.time()-t0 > LIM-22:
                    break
                tau = taus[min(li,len(taus)-1)]
                Y0 = polish_Y_fixedW(Y0, W, tau, lam, 35)
                W = w_steps(Y0, W, tau, lam, 25)
                Y0 = polish_Y_fixedW(Y0, W, tau, lam, 25)
                # early exit if residual tiny and tau small -> continue to next lam
                _,_,maxres,_ = proj_obj(Y0/np.maximum(np.linalg.norm(Y0,axis=1,keepdims=True),1e-12), W, tau, lam)
                if maxres < 1e-10 and li>=5:
                    # keep squeezing with larger lam quickly
                    continue
            # final hard squeeze
            Y0 = polish_Y_fixedW(Y0, W, 2e-4, 1e6, 30)
            W = w_steps(Y0, W, 2e-4, 1e6, 20)
            P = Y0 @ W
            rn = np.linalg.norm(P,axis=1,keepdims=True); rn=np.maximum(rn,1e-12)
            maxres = float(np.max(1.0-(rn[:,0]**2)))
            Xp = _normalize(P/rn if np.all(np.isfinite(P)) else r2.standard_normal((n,d13)))
            # if not squeezed, project anyway
            Xp = _normalize(Xp)
            # 13D polish
            Xp = lse13(Xp, 5e-4, 50)
            Xp = lse13(Xp, 1e-4, 60)
            vq = _max_cos(Xp)
            if vq < best_val + 2e-3:
                Xp, vq = greedy13(Xp, 90, 1.0e-5, min(time.time()+3, t0+LIM-4))
            else:
                vq = _max_cos(Xp)
            if vq < best_val - 1e-13 and maxres < 1e-6:
                best = Xp.copy(); best_val=float(vq)
            elif vq < best_val - 1e-13:
                # accept even if residual slightly larger but projected valid
                if maxres < 1e-4:
                    best = Xp.copy(); best_val=float(vq)
            if ntrials>=4 and time.time()-t0 > LIM-30:
                break
        except Exception:
            continue

    # ---------- final 13D squeeze ----------
    if time.time()-t0 < LIM-1.0:
        try:
            Xf=lse13(best,6e-5,100)
            vf=_max_cos(Xf)
            if vf<best_val: best_val,best=float(vf),Xf.copy()
            else: Xf=best.copy()
            Xf,vf=greedy13(Xf,700,7e-6,t0+LIM-0.5)
            if vf<best_val: best=Xf.copy(); best_val=float(vf)
        except Exception: pass
    best=_normalize(best)
    return best
