import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1500, radius0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = radius0
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]; csel = c_all[sel]
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None]*d+np.arange(d)).reshape(p,d),
                               (J[:, None]*d+np.arange(d)).reshape(p,d),
                               np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n*d)
            A_eq = coo_matrix((x.ravel(),(erows,ecols)), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn==0]=1.0
        y = y/yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time()-start >= seconds:
            break
    return best_x

def _patch_descend(Yall, patch, steps, T0, T1, lr0, lr1):
    n, d = Yall.shape
    k = len(patch)
    Yp = Yall[patch].copy()
    # map global idx -> row in patch for self masking
    g2r = {g:i for i,g in enumerate(patch)}
    for s in range(steps):
        frac = s/max(steps-1,1)
        T = T0+(T1-T0)*frac
        lr = lr0+(lr1-lr0)*frac
        D = Yp @ Yall.T  # k x n
        for i,g in enumerate(patch):
            D[i,g] = -10.0
        m = D.max(axis=1, keepdims=True)
        E = np.exp((D-m)/T)
        row = E.sum(axis=1, keepdims=True)+1e-20
        Avg = (E @ Yall)/row
        dot = (Avg*Yp).sum(axis=1, keepdims=True)
        Gt = Avg - dot*Yp
        Yp = Yp - lr*Gt
        Yp /= np.linalg.norm(Yp, axis=1, keepdims=True)
        Yall[patch] = Yp
    return Yp

def solve():
    t0 = time.time()
    HARD = 172.0
    n, d = 123, 13
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    best = None; best_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a,float)
            if a.shape==(n,d):
                a = a/np.linalg.norm(a,axis=1,keepdims=True)
                v = _max_cos(a)
                if v < best_val:
                    best_val=v; best=a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n,d)); best/=np.linalg.norm(best,axis=1,keepdims=True)
        best_val=_max_cos(best)
    cur = best.copy(); cur_val = float(best_val)
    # quick initial snap very short
    # main patch cycle
    it = 0
    # pre-determine time for final polish
    while time.time()-t0 < HARD-22:
        # contact graph within 1e-4
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m-1e-4
        deg = (G>thr).sum(axis=1)
        order = np.argsort(-deg)  # descending
        # cycle through top hubs distinctly
        hub = int(order[it % 10])
        # 1-hop neighbourhood
        neigh1 = np.where(G[hub]>thr)[0]
        neigh1 = neigh1[neigh1!=hub]
        # sort neigh1 by closeness
        if len(neigh1)>0:
            o = np.argsort(-G[hub,neigh1])
            neigh1 = neigh1[o]
        patch = [hub]+neigh1.tolist()
        if len(patch)>25:
            patch = patch[:25]
        if len(patch)<15:
            # expand to 2-hop
            inpatch = set(patch)
            # candidates: neighbours of patch
            cand_scores = {}
            for u in list(patch):
                nb = np.where(G[u]>thr)[0]
                for v in nb:
                    if v not in inpatch and v!=u:
                        # score by max dot to patch
                        s = max(G[v,w] for w in patch)
                        if v not in cand_scores or s>cand_scores[v]:
                            cand_scores[v]=s
            # sort by score desc
            cands = sorted(cand_scores.keys(), key=lambda v:-cand_scores[v])
            need = min(20-len(patch), len(cands))
            patch = patch + cands[:need]
            if len(patch)<15:
                # fill with closest non-patch points by dot to hub
                dots = G[hub].copy()
                for v in patch:
                    dots[v]=-2
                extra = np.argsort(-dots)
                for v in extra:
                    if v not in inpatch:
                        patch.append(int(v))
                        inpatch.add(int(v))
                    if len(patch)>=15:
                        break
        patch = np.array(patch, dtype=int)
        k = len(patch)
        frozen_mask = np.ones(n,bool); frozen_mask[patch]=False
        Yf = cur[frozen_mask]  # (n-k) x d
        # greedy deepest holes
        Nc = 3000
        placed = []
        for j in range(k):
            cand = rng.normal(size=(Nc,d))
            cand/=np.linalg.norm(cand,axis=1,keepdims=True)
            s1 = cand @ Yf.T
            m1 = s1.max(axis=1)
            if len(placed)>0:
                P = np.stack(placed,axis=0)
                s2 = cand @ P.T
                m2 = s2.max(axis=1)
                score = np.maximum(m1,m2)
            else:
                score = m1
            bi = int(np.argmin(score))
            placed.append(cand[bi])
        Yp0 = np.stack(placed,axis=0)
        Yall = cur.copy()
        Yall[patch]=Yp0
        # patch-only annealed softmax descent
        # anneal params vary slightly per iteration for diversity
        if it%3==0:
            T0,T1,lr0,lr1,steps=0.02,0.002,0.25,0.03,350
        elif it%3==1:
            T0,T1,lr0,lr1,steps=0.01,0.001,0.15,0.02,350
        else:
            T0,T1,lr0,lr1,steps=0.03,0.003,0.3,0.05,300
        # time guard: patch descend is cheap
        Yp1 = _patch_descend(Yall, patch, steps, T0,T1,lr0,lr1)
        Ytrial = cur.copy(); Ytrial[patch]=Yp1
        Ytrial/=np.linalg.norm(Ytrial,axis=1,keepdims=True)
        # short unfreeze LP polish
        remain = HARD-22-(time.time()-t0)
        lp_t = min(3.0, max(1.0, remain/8.0))
        try:
            Ypol = _lp_polish(Ytrial, lp_t, p_cap=1200, radius0=2e-3)
        except Exception:
            Ypol = Ytrial
        v = _max_cos(Ypol)
        if v < best_val:
            best_val=v; best=Ypol.copy()
        # walk acceptance: improve cur, else occasional sideways within tolerance
        if v < cur_val:
            cur=Ypol.copy(); cur_val=v
        else:
            # allow lateral move if within 2e-3 of best and deterministic coin
            if v < best_val+0.002 and (rng.random()<0.25):
                cur=Ypol.copy(); cur_val=v
            # else keep cur (try next distinct hub from same cur)
        it+=1
        if time.time()-t0 >= HARD-22:
            break
        # safety: if cur drifted too bad, pull back to best
        if cur_val > best_val+0.003:
            cur=best.copy(); cur_val=best_val
    # final polish of best with remaining time
    rem = HARD-(time.time()-t0)
    if rem>2:
        try:
            cand = _lp_polish(best, min(rem-1,18.0), p_cap=1500, radius0=1e-3)
            v=_max_cos(cand)
            if v<best_val:
                best=cand; best_val=v
        except Exception:
            pass
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    return best
