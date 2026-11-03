import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _tangent_bases(X):
    n, d = X.shape
    B = np.zeros((n, d, d-1))
    e0 = np.zeros(d); e0[0] = 1.0
    I = np.eye(d)
    for i in range(n):
        xi = X[i]
        s = 1.0 if xi[0] >= 0 else -1.0
        u = xi + s * e0
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            # xi opposite e0, use e1
            u = np.zeros(d); u[1] = 1.0
            nu = 1.0
        u /= nu
        H = I - 2.0 * np.outer(u, u)
        B[i] = H[:, 1:]
    return B

def _contact_sel(X, m, thresh=2e-4):
    n = X.shape[0]
    iu0 = np.triu_indices(n, 1)
    G = X @ X.T
    c = G[iu0]
    mm = float(c.max())
    sel = c > (mm - thresh)
    return iu0[0][sel], iu0[1][sel], c[sel], mm, iu0

def _cluster_S(I, J, c, hub, target=200):
    n = int(max(I.max(), J.max())) + 1
    adj = [[] for _ in range(n)]
    for a, b in zip(I, J):
        adj[a].append(b); adj[b].append(a)
    # BFS order from hub
    seen = np.zeros(n, dtype=bool)
    order = [hub]; seen[hub] = True
    q = [hub]
    qi = 0
    while qi < len(q):
        u = q[qi]; qi += 1
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                order.append(v); q.append(v)
    order = np.array(order, dtype=int)
    # find prefix size giving induced contacts ~ target
    pos = np.zeros(n, dtype=int)
    pos[order] = np.arange(len(order))
    # induced count as function of m: both ends in first m
    # search m
    p = len(I)
    best_m = len(order)
    best_S = None
    # try increasing m
    for m in [15, 25, 40, 60, 90, 130, 180, 249]:
        m = min(m, len(order))
        inset = np.zeros(n, dtype=bool)
        inset[order[:m]] = True
        mask = inset[I] & inset[J]
        cnt = int(mask.sum())
        if cnt >= target - 50 and cnt <= 300:
            best_S = mask; best_m = m; break
        if cnt > 300:
            # trim to tightest 250
            idx = np.where(mask)[0]
            o = np.argsort(-c[idx])[:250]
            nmask = np.zeros(p, dtype=bool)
            nmask[idx[o]] = True
            best_S = nmask; best_m = m; break
        if cnt >= 100:
            best_S = mask; best_m = m
            if cnt >= 150:
                break
    if best_S is None:
        # fallback: top 200 tightest overall near hub 2-hop
        inset = np.zeros(n, dtype=bool)
        inset[order[:60]] = True
        mask = inset[I] | inset[J]
        idx = np.where(mask)[0]
        if len(idx) > 250:
            o = np.argsort(-c[idx])[:200]
            nmask = np.zeros(p, dtype=bool); nmask[idx[o]] = True
            best_S = nmask
        else:
            best_S = mask
    cnt = int(best_S.sum())
    if cnt < 100:
        # pad with tightest overall
        idx_all = np.argsort(-c)
        need = 150 - cnt
        add = []
        for k in idx_all:
            if not best_S[k]:
                best_S[k] = True
                need -= 1
                if need <= 0: break
    elif cnt > 300:
        idx = np.where(best_S)[0]
        o = np.argsort(-c[idx])[250:]
        best_S[idx[o]] = False
    return best_S

def _forced_lp_step(X, I, J, c, S_mask, delta, radius):
    n, d = X.shape
    p = len(I)
    nv = n * d + 1
    ar = np.arange(d)
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate(
        [(I[:, None] * d + ar), (J[:, None] * d + ar),
         np.full((p, 1), n * d)], axis=1).ravel()
    vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    b_ub = -c.copy()
    b_ub[S_mask] -= float(delta)
    rep = np.repeat(np.arange(n), d)
    A_eq = coo_matrix((X.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    bounds = [(-radius, radius)] * (n * d) + [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                  b_eq=np.zeros(n), bounds=bounds, method="highs")
    if res.status != 0:
        return None, None
    dx = res.x[:-1].reshape(n, d)
    tpred = float(res.x[-1])
    Y = X + dx
    Y = _normed(Y)
    return Y, tpred

def _flex_step(X, B, I, J, radius, rng):
    n, d = X.shape
    N5 = n * (d - 1)
    p = len(I)
    # build Jacobian p x N5
    # J[k, I*5:(I+1)*5] = B_I^T x_J
    Jm = np.zeros((p, N5), dtype=float)
    # vectorized loop over p (1300 * small) fine
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        Jm[k, i*5:(i+1)*5] = B[i].T @ X[j]
        Jm[k, j*5:(j+1)*5] = B[j].T @ X[i]
    try:
        _, s, Vt = np.linalg.svd(Jm, full_matrices=False)
        v = Vt[-1]
    except Exception:
        v = rng.standard_normal(N5); v /= (np.linalg.norm(v)+1e-12)
    Dx = np.zeros((n, d))
    for i in range(n):
        Dx[i] = B[i] @ v[i*5:(i+1)*5]
    # scale to trust radius (max row norm)
    mx = np.linalg.norm(Dx, axis=1).max()
    if mx < 1e-12:
        return None, None
    Dx *= (radius / mx)
    # try both signs, pick one opening S more? return positive; caller tries both
    Y1 = _normed(X + Dx)
    Y2 = _normed(X - Dx)
    return (Y1, Y2), s

def _soft_polish(X, tau, maxiter, iu0):
    n, d = X.shape
    I0, J0 = iu0
    def fg(xf):
        Xv = xf.reshape(n, d)
        r = np.sqrt(np.sum(Xv*Xv, axis=1)); r = np.maximum(r, 1e-12)
        Xn = Xv / r[:, None]
        G = Xn @ Xn.T
        g = G[I0, J0]
        gm = np.max(g)
        e = np.exp((g - gm)/tau)
        S = np.sum(e)
        Lsoft = gm + tau*np.log(S)
        w = e / S
        W = np.zeros((n, n)); W[I0, J0] = w; W = W + W.T
        A = W @ Xn
        proj = np.sum(A*Xn, axis=1)
        G0 = (A - proj[:, None]*Xn)/r[:, None]
        diff = r - 1.0
        creg = 0.05
        L = Lsoft + creg*np.sum(diff*diff)
        Gr = G0 + (2.0*creg*diff)[:, None]*(Xv/r[:, None])
        return float(L), Gr.ravel()
    xf = X.ravel().copy()
    try:
        res = minimize(fg, xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'maxfun': int(maxiter*3),
                                'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
        xf = res.x
    except Exception:
        pass
    Xo = xf.reshape(n, d)
    Xo = _normed(Xo)
    return Xo

def _lp_small(X, deadline, radius=1.5e-3):
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    while time.time() < deadline and radius > 1e-13:
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0*radius, 1e-9)
        sel = np.where(c > thresh)[0]
        if sel.size == 0: break
        if sel.size > 3500:
            part = np.argpartition(-c, 3500)[:3500]
            sel = part
        I = iu[0][sel]; J = iu[1][sel]; cs = c[sel]
        p = sel.size
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([(I[:,None]*d+ar),(J[:,None]*d+ar),
                               np.full((p,1),n*d)],axis=1).ravel()
        vals = np.concatenate([X[J],X[I],-np.ones((p,1))],axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(rep,np.arange(n*d))),shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            bounds=[(-radius,radius)]*(n*d)+[(None,None)]
            res = linprog(obj,A_ub=A_ub,b_ub=-cs,A_eq=A_eq,b_eq=np.zeros(n),
                          bounds=bounds,method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        Y = _normed(X + res.x[:-1].reshape(n,d))
        v = _max_cos(Y)
        if v < best:
            X,best=Y,v
            radius=min(radius*1.5,0.02)
        else:
            radius*=0.5
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 249, 6
    cands = []
    for fn in ("best.npy","record.npy"):
        try:
            a=np.load(fn).astype(float)
            if a.shape==(n,d):
                a=_normed(a); cands.append(a)
        except Exception:
            pass
    if not cands:
        a=rng.standard_normal((n,d)); a=_normed(a); cands.append(a)
    vals=[_max_cos(a) for a in cands]
    bi=int(np.argmin(vals))
    bestX=cands[bi].copy(); bestVal=float(vals[bi])
    cur=bestX.copy(); curV=bestVal
    iu0=np.triu_indices(n,1)
    deltas=[1e-4,5e-4,1e-3]
    radii=[0.01,0.02,0.05,0.03]
    it=0
    # break phase until ~95s
    while time.time() < t0+95.0:
        it+=1
        if time.time()>DEAD-80: break
        # contact set on cur
        try:
            I,J,c,mm,_=_contact_sel(cur,mm if 'mm' in locals() else bestVal,thresh=2e-4)
        except Exception:
            I,J,c,mm,_=_contact_sel(cur,bestVal,thresh=2e-4)
        p=len(I)
        if p<50:
            cur=bestX.copy(); curV=bestVal; continue
        # hub ordering by degree
        deg=np.bincount(I,minlength=n)+np.bincount(J,minlength=n)
        hubs=np.argsort(-deg)
        if it%3==0:
            hub=int(rng.integers(0,n))
        else:
            hub=int(hubs[(it-1)%min(8,n)])
        delta=deltas[(it-1)%len(deltas)]
        radius=radii[(it-1)%len(radii)]
        S_mask=_cluster_S(I,J,c,hub,target=200)
        sc=int(S_mask.sum())
        if sc<50 or sc>320:
            # fallback trim/pad already handled; skip if degenerate
            pass
        # tangent bases + flex info (SVD) every other iter
        B=None; flex_pair=None
        if it%2==1:
            try:
                B=_tangent_bases(cur)
                flex_pair,svals=_flex_step(cur,B,I,J,radius,rng)
            except Exception:
                flex_pair=None; B=None
        # forced LP step
        Ystep=None
        try:
            Ystep,_=_forced_lp_step(cur,I,J,c,S_mask,delta,radius)
        except Exception:
            Ystep=None
        cands_step=[]
        if Ystep is not None:
            cands_step.append(Ystep)
        if flex_pair is not None:
            cands_step.extend([flex_pair[0],flex_pair[1]])
        if not cands_step:
            # random tangent jitter fallback
            pert=rng.normal(scale=radius*0.5,size=(n,d))
            pert-=np.sum(pert*cur,axis=1,keepdims=True)*cur
            cands_step.append(_normed(cur+pert))
        # polish each candidate briefly and keep best of this round
        best_round=None; best_round_v=np.inf
        for Y in cands_step:
            if time.time()>t0+100: break
            try:
                Yp=_soft_polish(Y,tau=0.003,maxiter=120,iu0=iu0)
            except Exception:
                Yp=Y
            v=_max_cos(Yp)
            if v<best_round_v:
                best_round_v=v; best_round=Yp
            if v<bestVal-1e-12:
                bestVal=v; bestX=Yp.copy()
        if best_round is None:
            continue
        # move cur even if worsened (limited tolerance) to change basin
        if best_round_v < curV + 0.004:
            cur=best_round; curV=best_round_v
        else:
            # revert to best with small jitter to try new cluster
            jitter=rng.normal(scale=0.005,size=(n,d))
            jitter-=np.sum(jitter*bestX,axis=1,keepdims=True)*bestX
            cur=_normed(bestX+jitter); curV=_max_cos(cur)
            if curV<bestVal:
                bestVal=curV; bestX=cur.copy()
        # opportunistic small LP on best every few iters
        if it%6==0 and time.time()<t0+90:
            try:
                bestX=_lp_small(bestX,t0+92,radius=1.0e-3)
                bestVal=_max_cos(bestX)
                if bestVal<curV-0.003:
                    cur=bestX.copy(); curV=bestVal
            except Exception:
                pass
    # final intensive polish of best
    try:
        for tau,mi in [(0.008,120),(0.004,150),(0.002,200),(0.0012,250)]:
            if time.time()>DEAD-22: break
            bestX=_soft_polish(bestX,tau=tau,maxiter=mi,iu0=iu0)
            v=_max_cos(bestX)
            if v<bestVal: bestVal=v
    except Exception:
        pass
    try:
        if time.time()<DEAD-8:
            bestX=_lp_small(bestX,DEAD-2,radius=1.5e-3)
    except Exception:
        pass
    bestX=_normed(np.nan_to_num(bestX,nan=0.0,posinf=0.0,neginf=0.0))
    nn=np.linalg.norm(bestX,axis=1)
    for i in np.where(nn<1e-12)[0]:
        v=rng.normal(size=d); v/=np.linalg.norm(v); bestX[i]=v
    return bestX
