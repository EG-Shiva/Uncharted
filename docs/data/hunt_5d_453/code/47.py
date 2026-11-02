import time
import numpy as np

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    T_COLL = t0 + 138.0
    T_FINAL = t0 + 168.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 453, 5
    try:
        from scipy.optimize import minimize as _min
        has_min = True
    except Exception:
        has_min = False
    try:
        from scipy.linalg import expm as _expm
        has_expm = True
    except Exception:
        has_expm = False

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                nn = np.linalg.norm(a, axis=1, keepdims=True)
                nn[nn == 0] = 1.0
                a = a / nn
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = sorted([(max_cos(c), c) for c in cands], key=lambda t: t[0])
    best_val = float(vals[0][0])
    best_X = vals[0][1].copy()
    X = best_X.copy()
    cur_val = best_val

    iu_pairs = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

    def skew_of(p):
        S = np.zeros((5,5), dtype=np.float64)
        for k,(i,j) in enumerate(iu_pairs):
            v = float(p[k])
            S[i,j] = v
            S[j,i] = -v
        return S

    def rot_of(p, reflect=False):
        S = skew_of(p)
        if has_expm:
            try:
                R = _expm(S)
            except Exception:
                R = np.eye(5) + S
        else:
            R = np.eye(5) + S + 0.5*(S@S)
            # project roughly
            try:
                U,_,Vt = np.linalg.svd(R)
                R = U @ Vt
            except Exception:
                pass
        if reflect:
            D = np.eye(5); D[0,0] = -1.0
            R = D @ R
        return R

    # fast cluster objective: max among pairs touching cluster
    def cluster_max(Xb, cl, Y):
        # Y: k x5 rotated cluster
        # internal
        Gii = Y @ Y.T
        np.fill_diagonal(Gii, -2.0)
        m1 = float(np.max(Gii)) if len(cl)>1 else -2.0
        # cross to outside
        mask = np.ones(n, dtype=bool); mask[cl] = False
        Xo = Xb[mask]  # (n-k)x5
        if Xo.shape[0] > 0:
            Gc = Y @ Xo.T
            m2 = float(np.max(Gc))
        else:
            m2 = -2.0
        return m1 if m1>m2 else m2

    def far_max_of(Xb, cl):
        G = Xb @ Xb.T
        # kill rows/cols of cluster
        G[cl, :] = -2.0
        G[:, cl] = -2.0
        # diag already -2, but need max of remaining
        return float(np.max(G))

    def local_softmax(Xb, active, steps, tau, lr, mu=0.7):
        nonlocal best_val, best_X
        v = np.zeros((len(active), d), dtype=np.float64)
        act = np.asarray(active, dtype=int)
        for it in range(steps):
            if time.time() > T_FINAL and steps>100:
                break
            G = Xb @ Xb.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val = m; best_X = Xb.copy()
            P = np.exp(np.clip((G - m)/tau, -50, 0))
            np.fill_diagonal(P, 0.0)
            Grad = P @ Xb
            Gt = Grad - np.sum(Grad*Xb, axis=1, keepdims=True)*Xb
            Ga = Gt[act]
            v = mu*v + (1.0-mu)*Ga
            Xb[act] = Xb[act] - lr*v
            Xb[act] /= np.linalg.norm(Xb[act], axis=1, keepdims=True)
            if m > best_val + 0.03:
                # revert to best projected onto current? just restore best
                Xb[:,:] = best_X
                v *= 0.5
                # re-extract? best may differ; continue
                # need to keep Xb reference; active still valid
        return Xb

    # ---------- collective rigid-cluster phase ----------
    outer = 0
    # precompute thresholds to get 10-40 sizes: use dot cutoffs
    while time.time() < T_COLL:
        outer += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mcur = float(np.max(G))
        if mcur < cur_val:
            cur_val = mcur
            if mcur < best_val:
                best_val = mcur; best_X = X.copy()
        else:
            # keep X at best if drifted much
            if mcur > best_val + 0.005:
                X = best_X.copy(); cur_val = best_val
                G = X @ X.T; np.fill_diagonal(G, -2.0); mcur = float(np.max(G))
        slack = mcur - G
        # contacts for degree: use slack < 1e-6 (captures jammed set)
        # compute degrees quickly from upper triangle
        iu_r, iu_c = np.triu_indices(n, 1)
        s_all = slack[iu_r, iu_c]
        tight = np.where(s_all < 1e-6)[0]
        deg = np.zeros(n, dtype=int)
        if tight.size>0:
            Ti = iu_r[tight]; Tj = iu_c[tight]
            np.add.at(deg, Ti, 1); np.add.at(deg, Tj, 1)
        hubs = np.argsort(-deg)[:14]
        # if all zero, random hubs
        if np.max(deg) == 0:
            hubs = rng.choice(n, size=8, replace=False)
        # iterate hubs
        improved_outer = False
        for hi, hub in enumerate(hubs):
            if time.time() > T_COLL:
                break
            dots = X @ X[int(hub)]
            # try several radii
            for thr in (0.975, 0.955, 0.930, 0.900, 0.870):
                if time.time() > T_COLL:
                    break
                cl = np.where(dots > thr)[0]
                k = len(cl)
                if k < 9 or k > 42:
                    continue
                # skip if cluster contains a rattler-only? keep hubs overloaded: require hub deg high or k in range
                # evaluate current cluster max and far max
                fmax = far_max_of(X, cl)
                # current Y = X[cl]
                Y0 = X[cl]
                c0 = cluster_max(X, cl, Y0)
                overall0 = c0 if c0>fmax else fmax
                # random search in skew space
                best_p = np.zeros(10, dtype=np.float64)
                best_c = c0
                # scales to try
                for scale in (0.004, 0.010, 0.025):
                    if time.time() > T_COLL:
                        break
                    for _ in range(25):
                        p = rng.standard_normal(10)*scale
                        R = rot_of(p, reflect=False)
                        Y = Y0 @ R.T
                        c = cluster_max(X, cl, Y)
                        if c < best_c:
                            best_c = c; best_p = p.copy()
                    # one reflected trial per scale
                    for _ in range(8):
                        p = rng.standard_normal(10)*scale
                        R = rot_of(p, reflect=True)
                        Y = Y0 @ R.T
                        c = cluster_max(X, cl, Y)
                        if c < best_c:
                            best_c = c; best_p = p.copy()
                            # store reflect flag via extra? we handle by keeping R; simplify: if reflect best, convert to direct fit later
                            # encode by keeping p and a flag
                            pass
                # Nelder-Mead from best_p (proper only)
                if has_min and time.time() < T_COLL - 2.0:
                    def _obj(pp):
                        try:
                            R = rot_of(pp, reflect=False)
                            Y = Y0 @ R.T
                            return cluster_max(X, cl, Y)
                        except Exception:
                            return 2.0
                    try:
                        res = _min(_obj, best_p, method="Nelder-Mead",
                                   options={"maxiter": 70, "maxfev": 220, "xatol": 1e-4, "fatol": 1e-9, "adaptive": True})
                        pf = np.asarray(res.x, dtype=np.float64)
                        cf = _obj(pf)
                        if cf < best_c:
                            best_c = cf; best_p = pf.copy()
                    except Exception:
                        pass
                    # small coordinate refinement
                    try:
                        step = 0.004
                        for _ in range(30):
                            improved = False
                            for a in range(10):
                                for sgn in (1.0,-1.0):
                                    pp = best_p.copy(); pp[a] += sgn*step
                                    c = _obj(pp)
                                    if c < best_c - 1e-12:
                                        best_c = c; best_p = pp.copy(); improved=True
                            if not improved:
                                step *= 0.5
                                if step < 2e-4:
                                    break
                    except Exception:
                        pass
                if best_c < c0 - 1e-9:
                    Ropt = rot_of(best_p, reflect=False)
                    Yopt = Y0 @ Ropt.T
                    overall = best_c if best_c>fmax else fmax
                    if overall < mcur - 1e-10:
                        # accept
                        Xn = X.copy()
                        Xn[cl] = Yopt
                        # counter-rotate / displace neighbours
                        # neighbours: next ring dots in [thr-0.08, thr]
                        nbr = np.where((dots <= thr) & (dots > thr-0.10))[0]
                        # limit to ~30 nearest
                        if len(nbr) > 30:
                            order = np.argsort(-dots[nbr])[:30]
                            nbr = nbr[order]
                        if len(nbr) > 0:
                            # try counter-rotation with -0.4*S
                            try:
                                Sopt = skew_of(best_p)
                                Rc = _expm(-0.4*Sopt) if has_expm else np.eye(5)-0.4*Sopt
                                Yn = Xn[nbr] @ Rc.T
                                Yn /= np.linalg.norm(Yn, axis=1, keepdims=True)
                                Xt = Xn.copy(); Xt[nbr] = Yn
                                vt = max_cos(Xt)
                                vo = max_cos(Xn)
                                if vt < vo:
                                    Xn = Xt
                                else:
                                    # small jitter instead
                                    jn = rng.standard_normal((len(nbr),d))*0.003
                                    for ii,k2 in enumerate(nbr):
                                        vv = jn[ii]-float(np.dot(jn[ii],Xn[k2]))*Xn[k2]
                                        Xn[k2] = Xn[k2]+vv
                                    Xn[nbr] /= np.linalg.norm(Xn[nbr],axis=1,keepdims=True)
                            except Exception:
                                pass
                        vnew = max_cos(Xn)
                        if vnew < mcur:
                            X = Xn; cur_val = vnew; mcur = vnew
                            if vnew < best_val:
                                best_val = vnew; best_X = Xn.copy()
                            improved_outer = True
                            # local softmax on cluster+neighbours
                            act = np.unique(np.concatenate([cl, nbr]) if len(nbr)>0 else cl)
                            # short local polish, tau small
                            tau = 0.002 if (mcur-best_val)<0.001 else 0.004
                            X = local_softmax(X, act, steps=180, tau=tau, lr=3e-4, mu=0.7)
                            G = X @ X.T; np.fill_diagonal(G,-2.0); mcur=float(np.max(G)); cur_val=mcur
                            if mcur < best_val:
                                best_val=mcur; best_X=X.copy()
                            break  # recompute hubs after success
                    # else cluster improvement didn't beat far max: still accept if cluster max improved a lot? skip
                # time guard per hub
            if improved_outer:
                break
        if not improved_outer:
            # no cluster success: shake neighbours slightly and short global softmax to change topology
            if time.time() > T_COLL:
                break
            # diversify: random small global jitter of worst points
            try:
                G = X @ X.T; np.fill_diagonal(G,-2.0)
                flat = int(np.argmax(G)); wi=flat//n; wj=flat%n
                sel = np.array([wi,wj]+list(rng.choice(n, size=8, replace=False)), dtype=int)
                jn = rng.standard_normal((len(sel),d))*0.004
                for ii,k2 in enumerate(sel):
                    vv = jn[ii]-float(np.dot(jn[ii],X[k2]))*X[k2]
                    X[k2]=X[k2]+vv
                X/=np.linalg.norm(X,axis=1,keepdims=True)
                act = sel
                X = local_softmax(X, np.arange(n), steps=120, tau=0.003, lr=3e-4, mu=0.75)
            except Exception:
                pass
            # avoid infinite loop with same basin: perturb hub cap slightly at random
            if outer % 3 == 0:
                try:
                    h = int(hubs[0])
                    dots = X @ X[h]
                    cl = np.where(dots>0.93)[0]
                    if 5 < len(cl) < 60:
                        p = rng.standard_normal(10)*0.008
                        R = rot_of(p)
                        X[cl] = X[cl] @ R.T
                        X/=np.linalg.norm(X,axis=1,keepdims=True)
                except Exception:
                    pass

    # ---------- final global polish ----------
    X = best_X.copy()
    # greedy worst-pair shaving + softmax alternation within remaining budget
    t = time.time()
    # softmax anneal
    try:
        steps = 900
        v = np.zeros_like(X)
        tau0, tau1 = 0.004, 0.0005
        lr0, lr1 = 5e-4, 6e-5
        mu = 0.8
        for it in range(steps):
            if time.time() > T_FINAL:
                break
            frac = it/max(steps-1,1)
            tau = tau0*np.exp(np.log(tau1/tau0)*frac)
            lr = lr0*np.exp(np.log(lr1/lr0)*frac)
            G = X @ X.T; np.fill_diagonal(G,-2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val=m; best_X=X.copy()
            P = np.exp(np.clip((G-m)/tau,-50,0))
            np.fill_diagonal(P,0.0)
            Grad = P @ X
            Gt = Grad - np.sum(Grad*X,axis=1,keepdims=True)*X
            v = mu*v + (1.0-mu)*Gt
            X = X - lr*v
            X/=np.linalg.norm(X,axis=1,keepdims=True)
            if m > best_val+0.02:
                X = best_X.copy(); v*=0.5
    except Exception:
        pass
    # final tiny shave of worst pair
    try:
        for _ in range(120):
            if time.time() > T_END-4.0:
                break
            G = X @ X.T; np.fill_diagonal(G,-2.0)
            m = float(np.max(G))
            if m < best_val:
                best_val=m; best_X=X.copy()
            flat=int(np.argmax(G)); i=flat//n; j=flat%n
            cij=float(X[i]@X[j])
            ti=X[j]-cij*X[i]; ni=float(np.linalg.norm(ti))
            if ni<1e-12:
                break
            ti/=ni
            tj=X[i]-cij*X[j]; nj=float(np.linalg.norm(tj))
            if nj<1e-12:
                break
            tj/=nj
            improved=False
            for sc in (8e-4,4e-4,2e-4):
                vi=X[i]-sc*ti; vi/=np.linalg.norm(vi)
                vj=X[j]-sc*tj; vj/=np.linalg.norm(vj)
                oi=X[i].copy(); oj=X[j].copy()
                X[i]=vi; X[j]=vj
                G2=X@X.T; np.fill_diagonal(G2,-2.0); m2=float(np.max(G2))
                if m2 < m-1e-12:
                    if m2<best_val:
                        best_val=m2; best_X=X.copy()
                    improved=True
                    break
                else:
                    X[i]=oi; X[j]=oj
            if not improved:
                # nudge random point tangentially tiny
                k2=int(rng.integers(0,n))
                rr=rng.standard_normal(d); rr-=float(rr@X[k2])*X[k2]
                rn=float(np.linalg.norm(rr))
                if rn>1e-12:
                    X[k2]=(X[k2]+1e-6*rr/rn)
                    X[k2]/=np.linalg.norm(X[k2])
    except Exception:
        pass

    Xr = best_X.copy()
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
    # strict finite check
    if not np.all(np.isfinite(Xr)):
        Xr = X.copy()/np.linalg.norm(X,axis=1,keepdims=True)
    return Xr
