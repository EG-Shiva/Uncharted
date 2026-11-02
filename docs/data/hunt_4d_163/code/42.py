import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    G = x @ x.T
    iu0, iu1 = np.triu_indices(x.shape[0], 1)
    c = G[iu0, iu1]
    m = float(c.max())
    sel = c > m - tol
    return m, iu0[sel], iu1[sel], c, iu0, iu1

def _tangent_bases(x):
    n = x.shape[0]
    T = np.empty((n, 4, 3))
    for i in range(n):
        A = np.eye(4)
        A[:, 0] = x[i]
        Q, _ = np.linalg.qr(A)
        if np.dot(Q[:, 0], x[i]) < 0:
            Q[:, 0] = -Q[:, 0]
        T[i] = Q[:, 1:4]
    return T

def _rigidity(x, CI, CJ, T=None):
    n = x.shape[0]
    if T is None:
        T = _tangent_bases(x)
    p = len(CI)
    N = n * 3
    R = np.zeros((p, N))
    for k in range(p):
        i = int(CI[k]); j = int(CJ[k])
        R[k, i*3:(i+1)*3] = T[i].T @ x[j]
        R[k, j*3:(j+1)*3] = T[j].T @ x[i]
    return R, T

def _lp_jump(x0, RI, RJ, Rc, drop_set, allow, radius):
    n, d = x0.shape
    p = len(RI)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2*d+1)
    cols = np.empty((p, 2*d+1), dtype=int)
    ar = np.arange(d)
    cols[:, :d] = RI[:, None]*d + ar
    cols[:, d:2*d] = RJ[:, None]*d + ar
    cols[:, 2*d] = n*d
    cols = cols.ravel()
    vals = np.empty((p, 2*d+1), dtype=float)
    vals[:, :d] = x0[RJ]
    vals[:, d:2*d] = x0[RI]
    vals[:, 2*d] = -1.0
    vals = vals.ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    b_ub = -Rc.copy()
    if len(drop_set) > 0:
        b_ub[np.asarray(drop_set, dtype=int)] += float(allow)
    A_eq = coo_matrix((x0.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-float(radius), float(radius))]*(n*d)+[(None, None)], method="highs",
                      options={"time_limit": 5.0})
    except Exception:
        return None, None
    if res.status != 0:
        return None, None
    dx = res.x[:-1].reshape(n, d)
    t = float(res.x[-1])
    return dx, t

def _lp_polish(x, deadline, r0=2e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    n, d = x.shape
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0*radius, 1e-9)
            if int(np.count_nonzero(sel)) == 0:
                sel[int(np.argmax(c))] = True
            if int(np.count_nonzero(sel)) > 1500:
                idx = np.argsort(-c)[:1500]
                s2 = np.zeros_like(sel); s2[idx] = True; sel = s2
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I); nv = n*d+1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.empty((p, 2*d+1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None]*d + ar
            cols[:, d:2*d] = J[:, None]*d + ar
            cols[:, 2*d] = n*d
            cols = cols.ravel()
            vals = np.empty((p, 2*d+1), dtype=float)
            vals[:, :d] = x[J]; vals[:, d:2*d] = x[I]; vals[:, 2*d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5; continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.04)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5; continue
    return x, best

def _anneal(x0, steps, T0, T1, lr0, mu, rng, deadline=None):
    x = x0.copy(); n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T; np.fill_diagonal(G, -2.0)
    bv = float(G.max()); bx = x.copy()
    lratio = np.log(T1/T0) if T0>0 and T1>0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127)==0 and time.time()>deadline:
            break
        frac = k/max(1, steps-1)
        T = T0*np.exp(lratio*frac); T = max(T, 1e-12)
        lr = lr0*(1.0-0.55*frac)
        G = x @ x.T; np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G-m)/T); np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if S < 1e-300 or not np.isfinite(S): S = 1e-300
        grad = (E @ x)/S
        grad -= (grad*x).sum(axis=1, keepdims=True)*x
        v = mu*v + (1.0-mu)*grad
        x = x - lr*v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv: bv = m; bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.002, 0.0006), iters=(70, 70)):
    x = x_init/np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1); Np = len(iu0)
    logNp = np.log(Np)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline-0.3: break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y/rn
            g = (X@X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g-m)/_tau); s = float(e.sum())
            if s<=0 or not np.isfinite(s): return float(m), np.zeros_like(yflat)
            L = m + _tau*(np.log(s)-logNp)
            w = e/s
            W = np.zeros((n, n)); W[iu0, iu1]=w; W = W+W.T
            GX = W@X
            GY = (GX-(GX*X).sum(axis=1, keepdims=True)*X)/rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter*5), "ftol":1e-14, "gtol":1e-8})
            y = res.x
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 173.0
    SEARCH_END = t0 + 140.0
    rng = np.random.RandomState(20261102)
    n = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try: bx = np.load("record.npy")
        except Exception: bx = rng.randn(n,4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n,4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(n,4)
    bx = bx/np.maximum(np.linalg.norm(bx,axis=1,keepdims=True),1e-12)
    best_x = bx.copy(); best_val = _max_cos(best_x)
    # quick initial polish of loaded best
    best_x, best_val = _lp_polish(best_x, min(DEADLINE-2, t0+8.0), r0=1e-3)
    div_pool = []
    def contact_set(x):
        G = x@x.T
        iu0, iu1 = np.triu_indices(n,1)
        c = G[iu0,iu1]; m=float(c.max())
        sel = c > m-1e-9
        s=set(zip(iu0[sel].tolist(), iu1[sel].tolist()))
        return m, s
    _, best_set = contact_set(best_x)

    trial = 0
    radii = [0.01, 0.03, 0.06]
    allows = [0.004, 0.008, 0.015]
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE-30: break
        trial += 1
        # choose base: mostly best, sometimes diverse
        if div_pool and (trial % 4 == 0):
            base_x = div_pool[rng.randint(len(div_pool))][1]
        else:
            base_x = best_x
        # build contacts + second tier for base
        try:
            G = base_x@base_x.T
            iu0, iu1 = np.triu_indices(n,1)
            c_all = G[iu0,iu1]
            m0 = float(c_all.max())
            sel_active = c_all > m0-1e-9
            if int(sel_active.sum()) < 50:
                sel_active = c_all > m0-1e-7
            AI0 = iu0[sel_active]; AJ0 = iu1[sel_active]
            p_act = len(AI0)
            # second tier
            sel2 = (c_all > m0-2e-4) & (~sel_active)
            S2I = iu0[sel2]; S2J = iu1[sel2]
            # rigidity on active contacts
            Rmat, Tt = _rigidity(base_x, AI0, AJ0)
            U, Sv, Vt = np.linalg.svd(Rmat, full_matrices=False)
            nvib = Vt.shape[0]
            # pick a floppy mode among 5 smallest
            mi = int(nvib-1 - (rng.randint(min(5,nvib))))
            vmode = Vt[mi]
            stretch = Rmat @ vmode
            Vpts = vmode.reshape(n,3)
            ptnorm = np.linalg.norm(Vpts, axis=1)
            # degree
            deg = np.zeros(n,dtype=int)
            for ii in AI0: deg[int(ii)]+=1
            for jj in AJ0: deg[int(jj)]+=1
            # contact index ranking by |stretch|
            order_s = np.argsort(-np.abs(stretch))
            # build drop set
            K = int(rng.choice([15,20,25,30,40]))
            K = min(K, p_act)
            mode = trial % 3
            drop = None
            if mode == 0:
                # clustered on one point (hub)
                top_pts = np.argsort(-deg)[:6]
                hub = int(top_pts[rng.randint(len(top_pts))])
                cand = np.where((AI0==hub)|(AJ0==hub))[0].tolist()
                # expand to neighbors until K
                if len(cand) < K:
                    nbrs = set()
                    for k in cand:
                        nbrs.add(int(AI0[k])); nbrs.add(int(AJ0[k]))
                    nbrs.discard(hub)
                    extra = [k for k in range(p_act) if k not in cand and (int(AI0[k]) in nbrs or int(AJ0[k]) in nbrs)]
                    # order extra by stretch
                    extra = sorted(extra, key=lambda k: -abs(float(stretch[k])))
                    cand = cand + extra[:max(0,K-len(cand))]
                drop = np.array(cand[:K], dtype=int)
            elif mode == 1:
                # community BFS
                seed = int(rng.randint(n))
                # BFS on contact graph
                adj=[[] for _ in range(n)]
                for k in range(p_act):
                    a=int(AI0[k]); b=int(AJ0[k]); adj[a].append(b); adj[b].append(a)
                seen=set([seed]); front=[seed]
                while len(seen)<11 and front:
                    nxt=[]
                    for u in front:
                        for w in adj[u]:
                            if w not in seen:
                                seen.add(w); nxt.append(w)
                                if len(seen)>=11: break
                        if len(seen)>=11: break
                    front=nxt
                    if not front and len(seen)<11:
                        # add random unvisited
                        cand_u = int(rng.randint(n))
                        if cand_u not in seen: seen.add(cand_u); front=[cand_u]
                sset=set(seen)
                cand=[k for k in range(p_act) if int(AI0[k]) in sset and int(AJ0[k]) in sset]
                if len(cand)<12:
                    # add boundary contacts sorted by stretch
                    bnd=[k for k in range(p_act) if k not in cand and (int(AI0[k]) in sset or int(AJ0[k]) in sset)]
                    bnd=sorted(bnd,key=lambda k:-abs(float(stretch[k])))
                    cand=cand+bnd
                if len(cand)>K:
                    cand=sorted(cand,key=lambda k:-abs(float(stretch[k])))[:K]
                elif len(cand)<K:
                    # top up with top stretch
                    for k in order_s:
                        if int(k) not in cand:
                            cand.append(int(k))
                            if len(cand)>=K: break
                drop=np.array(cand[:K],dtype=int)
            else:
                # along low-singular-vector support
                drop = order_s[:K].copy()
                # optionally bias to high ptnorm points
                if rng.rand()<0.5:
                    order_p = np.argsort(-ptnorm)
                    topP=set(order_p[:12].tolist())
                    cand2=[k for k in range(p_act) if int(AI0[k]) in topP or int(AJ0[k]) in topP]
                    if len(cand2)>=12:
                        cand2=sorted(cand2,key=lambda k:-abs(float(stretch[k])))[:K]
                        drop=np.array(cand2,dtype=int)
            radius = radii[trial % 3]
            # vary radius occasionally
            if rng.rand()<0.2: radius = float(rng.choice(radii))
            allow = allows[trial % 3]
            # retained set = active + second tier
            RI = np.concatenate([AI0, S2I]).astype(int)
            RJ = np.concatenate([AJ0, S2J]).astype(int)
            G2 = base_x@base_x.T
            Rc = G2[RI, RJ].astype(float)
            # map drop (indices into active) to indices into retained (first p_act entries)
            drop_ret = drop  # since first block is active
            dx, tp = _lp_jump(base_x, RI, RJ, Rc, drop_ret, allow, radius)
            if dx is None:
                continue
            y = base_x + dx
            y /= np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
            v_jump = _max_cos(y)
            # repair with anneal + lbfgs + polish, time-boxed
            t_rep = time.time()
            remain = SEARCH_END - t_rep
            if remain < 4: break
            y2, _ = _anneal(y, steps=900, T0=0.006, T1=0.0004, lr0=0.030, mu=0.86, rng=rng, deadline=t_rep+0.5)
            y3, _ = _lbfgs(y2, deadline=t_rep+1.6, taus=(0.002,0.0006), iters=(60,60))
            y4, v4 = _lp_polish(y3, deadline=min(t_rep+3.2, SEARCH_END, DEADLINE-25), r0=2e-3)
            # evaluate change in contacts
            _, set4 = contact_set(y4)
            inter = len(set4 & best_set); union = len(set4 | best_set)
            jacc = inter/max(1,union)
            changed = 1.0-jacc
            if v4 < best_val - 1e-12:
                best_x, best_val = y4, v4
                _, best_set = contact_set(best_x)
                # prune diverse pool worse than best+5e-4? keep
            elif v4 < best_val + 8e-4 and changed > 0.10:
                # keep diverse bifurcation
                div_pool.append((v4, y4.copy()))
                div_pool.sort(key=lambda t: t[0])
                del div_pool[6:]
            elif v4 < best_val + 2e-4 and changed > 0.25 and len(div_pool)<6:
                div_pool.append((v4, y4.copy()))
                div_pool.sort(key=lambda t: t[0])
        except Exception:
            continue
    # final polish best
    try:
        best_x, best_val = _lp_polish(best_x, DEADLINE-1.0, r0=1e-3)
    except Exception:
        pass
    best_x = best_x/np.maximum(np.linalg.norm(best_x,axis=1,keepdims=True),1e-12)
    return best_x
