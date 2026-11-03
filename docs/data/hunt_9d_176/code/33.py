import time
import numpy as np
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def refine_batch(C, Y, alpha=80.0, steps=30, lr=0.08):
    Cb = C.copy()
    for _ in range(steps):
        D = Cb @ Y.T
        m = D.max(axis=1, keepdims=True)
        E = np.exp(alpha * (D - m))
        W = E / (E.sum(axis=1, keepdims=True) + 1e-300)
        Gr = W @ Y
        dot = np.sum(Gr * Cb, axis=1, keepdims=True)
        Gt = Gr - dot * Cb
        Cb = Cb - lr * Gt
        nr = np.linalg.norm(Cb, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Cb /= nr
    return Cb

def insert_one(Ycur, rng, alpha=80.0):
    d = Ycur.shape[1]
    NT = 20000
    trials = rng.standard_normal((NT, d))
    nr = np.linalg.norm(trials, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    trials /= nr
    D = trials @ Ycur.T
    m = D.max(axis=1)
    part = np.argpartition(m, 200)[:200]
    Ds = D[part]
    ms = m[part]
    E0 = np.exp(alpha * (Ds - ms[:, None]))
    S0 = E0.sum(axis=1)
    order = np.lexsort((S0, ms))
    top = part[order[:20]]
    C0 = trials[top]
    Cb = refine_batch(C0, Ycur, alpha=alpha, steps=30, lr=0.08)
    Dr = Cb @ Ycur.T
    mr = Dr.max(axis=1)
    Er = np.exp(alpha * (Dr - mr[:, None]))
    Sr = Er.sum(axis=1)
    best_i = 0
    best_m = mr[0]; best_s = Sr[0]
    for i in range(1, len(mr)):
        if mr[i] < best_m - 1e-12 or (abs(mr[i]-best_m) <= 1e-12 and Sr[i] < best_s):
            best_m = mr[i]; best_s = Sr[i]; best_i = i
    return Cb[best_i]

def cheap_gd(X, steps=300, alpha=70.0, lr0=0.025):
    x = X.copy()
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        rn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        frac = t / max(steps-1, 1)
        lrt = lr0 * (1.0 - 0.5*frac)
        x = x - lrt * (Gt / rn)
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def lp_polish_deadline(x, deadline, radius_init=2e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    itlp = 0
    while time.time() < deadline - 0.15 and radius > 1e-13 and itlp < 80:
        itlp += 1
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0*radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        nr = np.linalg.norm(y, axis=1, keepdims=True); nr[nr==0]=1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline - 0.15:
            break
    return x

def bfs_community(adj, deg, seed, k):
    n = len(deg)
    visited = np.zeros(n, dtype=bool)
    visited[seed]=True
    comm=[seed]
    q=deque([seed])
    while len(comm)<k and q:
        u=q.popleft()
        nbrs=np.where(adj[u])[0]
        if len(nbrs)>1:
            nbrs=nbrs[np.argsort(-deg[nbrs])]
        for v in nbrs:
            if not visited[v]:
                visited[v]=True
                comm.append(int(v))
                q.append(int(v))
                if len(comm)>=k:
                    break
    if len(comm)<k:
        order=np.argsort(-deg)
        for v in order:
            if not visited[v]:
                visited[v]=True
                comm.append(int(v))
                if len(comm)>=k:
                    break
    return np.array(comm[:k], dtype=int)

def solve():
    t0=time.time()
    BUDGET=170.0
    SEARCH=130.0
    search_deadline=t0+SEARCH
    final_deadline=t0+BUDGET
    rng=np.random.default_rng(42)
    try:
        xb=np.load("best.npy")
    except Exception:
        try:
            xb=np.load("record.npy")
        except Exception:
            xb=rng.standard_normal((176,9))
    xb=np.asarray(xb,dtype=float)
    if xb.shape!=(176,9):
        xb=rng.standard_normal((176,9))
    xb=xb/np.linalg.norm(xb,axis=1,keepdims=True)
    best_val=max_cos(xb)
    best_x=xb.copy()
    try:
        xr=np.load("record.npy")
        xr=np.asarray(xr,dtype=float)
        if xr.shape==(176,9):
            xr=xr/np.linalg.norm(xr,axis=1,keepdims=True)
            vr=max_cos(xr)
            if vr<best_val:
                best_val, best_x=vr, xr.copy()
    except Exception:
        pass
    sizes=[12,18,24]
    it=0
    while True:
        if time.time()>=search_deadline:
            break
        G=best_x@best_x.T
        np.fill_diagonal(G,-2.0)
        cur_max=float(np.max(G))
        thr=cur_max-1e-5
        adj=G>thr
        deg=np.sum(adj,axis=1).astype(float)
        order=np.argsort(-deg)
        seed=int(order[it % 176])
        k=sizes[it % 3]
        comm=bfs_community(adj, deg, seed, k)
        mask=np.ones(176,dtype=bool)
        mask[comm]=False
        Yfroz=best_x[mask].copy()
        cdeg=deg[comm]
        ins_order=np.argsort(-cdeg)
        comm_ord=comm[ins_order]
        Ycur=Yfroz.copy()
        new_pts=[]
        ok=True
        for j in comm_ord:
            if time.time()>=search_deadline:
                ok=False
                break
            p=insert_one(Ycur, rng, alpha=80.0)
            new_pts.append(p)
            Ycur=np.vstack([Ycur, p[None,:]])
        if not ok:
            break
        Xnew=np.empty((176,9))
        Xnew[mask]=Yfroz
        for idx_pt, j in enumerate(comm_ord):
            Xnew[int(j)]=new_pts[idx_pt]
        Xnew/=np.linalg.norm(Xnew,axis=1,keepdims=True)
        Xq=cheap_gd(Xnew, steps=300, alpha=70.0, lr0=0.025)
        v=max_cos(Xq)
        if v<best_val-1e-12:
            best_val=v
            best_x=Xq.copy()
        it+=1
        if time.time()>=search_deadline:
            break
    rem=final_deadline-time.time()-0.5
    if rem>0.8:
        best_x=lp_polish_deadline(best_x, time.time()+rem, radius_init=2e-3)
    best_x/=np.linalg.norm(best_x,axis=1,keepdims=True)
    return best_x
