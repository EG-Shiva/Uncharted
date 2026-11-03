import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _compose(p1, s1, p2, s2):
    p = [p1[p2[j]] for j in range(len(p1))]
    s = [s2[j] * s1[p2[j]] for j in range(len(p1))]
    return tuple(p), tuple(s)

def _closure(gens, d=6):
    seen = set()
    elems = []
    ident = (tuple(range(d)), tuple([1]*d))
    seen.add(ident); elems.append(ident)
    queue = [ident]
    while queue:
        cur = queue.pop()
        for g in gens:
            np_, ns_ = _compose(cur[0], cur[1], g[0], g[1])
            key = (np_, ns_)
            if key not in seen:
                seen.add(key); elems.append(key); queue.append(key)
            np2_, ns2_ = _compose(g[0], g[1], cur[0], cur[1])
            key2 = (np2_, ns2_)
            if key2 not in seen:
                seen.add(key2); elems.append(key2); queue.append(key2)
    return elems

def _build_groups():
    d = 6
    perm_a = tuple([(j-1) % d for j in range(d)])
    sign_a = tuple([1]*d)
    perm_b = tuple([0,5,4,3,2,1])
    sign_b = tuple([1]*d)
    D12 = _closure([(perm_a,sign_a),(perm_b,sign_b)], d=d)
    gens_sign = []
    for c in range(3):
        s = [1]*d; s[c] = -1
        gens_sign.append((tuple(range(d)), tuple(s)))
    S8 = _closure(gens_sign, d=d)
    C6 = _closure([(perm_a,sign_a)], d=d)
    def to_arrays(elems):
        perms = np.array([list(p) for p,s in elems], dtype=int)
        signs = np.array([list(s) for p,s in elems], dtype=float)
        return perms, signs
    p12,s12 = to_arrays(D12)
    p8,s8 = to_arrays(S8)
    p6,s6 = to_arrays(C6)
    return [(p12,s12),(p8,s8),(p6,s6)]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(0)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    groups = _build_groups()
    # (perms,signs,k,nf)
    configs = []
    for (pg, sg) in groups:
        G = pg.shape[0]
        k = (n - 3) // G
        nf = n - k*G
        # ensure exact
        if k*G + nf != n or k <= 0:
            continue
        configs.append((pg, sg, k, nf))
    # configs should be (12,18,3),(8,27,3),(6,36,3)

    def sym_opt(pg, sg, k, nf, seed, betas, maxiters):
        Gg = pg.shape[0]
        rs = np.random.RandomState(seed)
        Yr0 = rs.randn(k, d)
        Yf0 = rs.randn(nf, d)
        # init reps from best: sample k points
        try:
            idx = rs.choice(n, size=k, replace=False)
            Yr0 = x_best[idx] + 0.05*rs.randn(k, d)
        except Exception:
            pass
        y0 = np.concatenate([Yr0.ravel(), Yf0.ravel()]).copy()
        for beta, mi in zip(betas, maxiters):
            if time.time()-t0 > BUDGET-35:
                break
            def fun(y):
                Yr = y[:k*d].reshape(k, d)
                Yf = y[k*d:].reshape(nf, d)
                rn_r = np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
                rn_f = np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
                Xr = Yr / rn_r
                Xf = Yf / rn_f
                N = k*Gg+nf
                Xfull = np.empty((N, d))
                for i in range(k):
                    xi = Xr[i]
                    base = i*Gg
                    for g in range(Gg):
                        Xfull[base+g] = sg[g] * xi[pg[g]]
                Xfull[k*Gg:] = Xf
                Gm = Xfull @ Xfull.T
                # upper pairs
                # use triu indices for N
                iu0N, iu1N = np.triu_indices(N, 1)
                gvec = Gm[iu0N, iu1N]
                m = float(np.max(gvec))
                e = np.exp(beta*(gvec-m))
                S = float(np.sum(e))+1e-300
                loss = m + np.log(S)/beta
                w = e/S
                W = np.zeros((N,N))
                W[iu0N, iu1N] = w
                W = W + W.T
                GX = W @ Xfull
                Drep = GX[:k*Gg]
                Dfree = GX[k*Gg:]
                GR = np.zeros((k, d))
                for g in range(Gg):
                    sgg = sg[g]; pgg = pg[g]
                    for i in range(k):
                        dg = Drep[i*Gg+g]
                        # dg[j] contributes to GR[i, pgg[j]]
                        for j in range(d):
                            GR[i, pgg[j]] += sgg[j]*dg[j]
                dotr = (GR*Xr).sum(axis=1, keepdims=True)
                GYr = (GR - dotr*Xr)/rn_r
                dotf = (Dfree*Xf).sum(axis=1, keepdims=True)
                GYf = (Dfree - dotf*Xf)/rn_f
                grad = np.concatenate([GYr.ravel(), GYf.ravel()])
                return float(loss), grad
            try:
                res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                               options={"maxiter": mi, "ftol":1e-12, "gtol":1e-7, "maxfun":20000})
                y0 = res.x.copy()
            except Exception:
                break
        Yr = y0[:k*d].reshape(k,d); Yf = y0[k*d:].reshape(nf,d)
        Yr = _norm(Yr); Yf = _norm(Yf)
        Xfull = np.empty((k*Gg+nf, d))
        for i in range(k):
            for g in range(Gg):
                Xfull[i*Gg+g] = sg[g]*Yr[i][pg[g]]
        Xfull[k*Gg:] = Yf
        return _norm(Xfull)

    def expand_sym(YrN, YfN, pg, sg, k):
        Gg = pg.shape[0]
        Xfull = np.empty((k*Gg+YfN.shape[0], d))
        for i in range(k):
            for g in range(Gg):
                Xfull[i*Gg+g] = sg[g]*YrN[i][pg[g]]
        Xfull[k*Gg:] = YfN
        return Xfull

    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y/rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta*(g-m))
            S = float(np.sum(e))+1e-300
            loss = m + np.log(S)/beta
            w = e/S
            W = np.zeros((n,n)); W[iu0,iu1]=w; W=W+W.T
            GX = W @ X
            dot = (GX*X).sum(axis=1, keepdims=True)
            GY = (GX-dot*X)/rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter":maxiter,"ftol":1e-12,"gtol":1e-7,"maxfun":20000})
            Y = res.x.reshape(n,d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=10):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps and time.time()-t0 < BUDGET-1.5:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0,iu1]
            m = float(c.max())
            thr = m - max(18.0*radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2-1)[:k2]
                I,J,csel = I[part],J[part],csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1),n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
            except Exception:
                radius*=0.5; continue
            if res.status!=0:
                radius*=0.5; continue
            y = _norm(x+res.x[:-1].reshape(n,d))
            v = _max_cos(y)
            if v < best:
                x,best=y,v; bx=y.copy()
                radius=min(radius*1.5,0.03)
            else:
                radius*=0.5
        return bx,best

    # symmetry phase: try each group with 2 seeds
    tried = []
    for ci,(pg,sg,k,nf) in enumerate(configs):
        for sdx in range(2):
            if time.time()-t0 > BUDGET-55:
                break
            seed = 1000+ci*10+sdx
            Xs = sym_opt(pg, sg, k, nf, seed, betas=[50,150,400], maxiters=[120,120,100])
            vs = _max_cos(Xs)
            # release: free polish
            Xr = run_free_lbfgs(Xs, 180.0, maxiter=70)
            vr = _max_cos(Xr)
            if vr < vs:
                Xs, vs = Xr, vr
            # light LP
            Xs2, vs2 = lp_polish(Xs, t0+BUDGET-30, r_init=8e-4, max_steps=6)
            if vs2 < vs:
                Xs, vs = Xs2, vs2
            tried.append((vs, Xs))
            if vs < best_val:
                best_val = vs
                x_best = Xs.copy()
        if time.time()-t0 > BUDGET-55:
            break

    # also try pure random symmetric inits (no best-seed) for diversity if time
    # final wandering hops from best with topology-breaking kicks
    hop_deadline = t0 + BUDGET - 12.0
    it = 0
    while time.time() < hop_deadline and time.time()-t0 < BUDGET-12:
        it += 1
        if it > 60:
            break
        r = rng.rand()
        cur = x_best.copy()
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        pm = Gc.max(axis=1)
        order = np.argsort(-pm)
        if r < 0.4:
            # cluster kick
            seed = int(order[0] if rng.rand()<0.6 else rng.choice(order[:25]))
            row = Gc[seed]
            nn = np.argsort(-row)[:50]
            k2 = int(rng.randint(10,35))
            idx = np.array(sorted(set(nn[:k2].tolist())), dtype=int)
            pert = rng.randn(len(idx), d)
            pert -= (np.sum(pert*cur[idx],axis=1,keepdims=True))*cur[idx]
            sig = 0.2+0.4*rng.rand()
            cur[idx] = _norm(cur[idx]+sig*pert)
            gj = rng.randn(n,d); gj-=(np.sum(gj*cur,axis=1,keepdims=True))*cur
            cur = _norm(cur+0.004*gj)
        elif r < 0.7:
            k2 = int(rng.randint(15,60))
            idx = rng.choice(n,size=k2,replace=False)
            pert = rng.randn(n,d); pert-=(np.sum(pert*cur,axis=1,keepdims=True))*cur
            sig = np.full((n,1),0.004)
            sig[idx]=0.2+0.3*rng.rand()
            cur = _norm(cur+sig*pert)
        else:
            # antipodal flips
            k2 = int(rng.randint(3,10))
            idx = rng.choice(n,size=k2,replace=False)
            cur[idx]=-cur[idx]
            pert = rng.randn(n,d); pert-=(np.sum(pert*cur,axis=1,keepdims=True))*cur
            cur = _norm(cur+0.015*pert)
        # screen with free lbfgs
        beta = 200.0 if rng.rand()<0.5 else 350.0
        Xn = run_free_lbfgs(cur, beta, maxiter=55)
        vn = _max_cos(Xn)
        if vn < best_val + 0.002:
            Xn2, vn2 = lp_polish(Xn, hop_deadline, r_init=6e-4, max_steps=4)
            if vn2 < vn:
                Xn, vn = Xn2, vn2
        if vn < best_val:
            best_val = vn
            x_best = Xn.copy()

    # final sharpen best
    try:
        Xf = run_free_lbfgs(x_best, 300.0, maxiter=80)
        vf = _max_cos(Xf)
        if vf < best_val:
            x_best, best_val = Xf, vf
        x_best, best_val2 = lp_polish(x_best, t0+BUDGET-1.0, r_init=1e-3, max_steps=10)
        best_val = min(best_val, best_val2)
    except Exception:
        pass
    return _norm(x_best)
