import numpy as np
import time

def solve():
    N = 26
    t0 = time.time()
    DEADLINE = 126.0
    rng = np.random.RandomState(1)

    try:
        from scipy.optimize import minimize, linprog
        has_scipy = True
    except Exception:
        has_scipy = False
        minimize = None
        linprog = None

    # fallback feasible grid
    k = int(np.ceil(np.sqrt(N)))
    r0f = 0.5 / k * 0.998
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < N:
                pts.append([(2*i+1)*0.5/k, (2*j+1)*0.5/k, r0f])
    pts = np.array(pts, dtype=float)
    best_arr = pts.copy()
    best_sum = float(np.sum(best_arr[:, 2]))
    best_x = best_arr[:, 0].copy()
    best_y = best_arr[:, 1].copy()
    best_r = best_arr[:, 2].copy()

    if not has_scipy:
        return best_arr

    # LP matrix
    pair_i = []
    pair_j = []
    for i in range(N):
        for j in range(i+1, N):
            pair_i.append(i); pair_j.append(j)
    pair_i = np.array(pair_i, dtype=int)
    pair_j = np.array(pair_j, dtype=int)
    npair = len(pair_i)
    nrow = npair + N
    A_ub0 = np.zeros((nrow, N), dtype=float)
    rr0 = np.arange(npair)
    A_ub0[rr0, pair_i] = 1.0
    A_ub0[rr0, pair_j] = 1.0
    for i in range(N):
        A_ub0[npar := npair+i, i] = 1.0
    c_obj = -np.ones(N, dtype=float)
    bounds_lp = [(0.0, None)]*N
    EPS = 1e-7

    def radii_lp(x, y):
        try:
            wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
            if np.min(wall) <= 1e-9:
                return None, -1.0
            Dx = x[:, None]-x[None, :]
            Dy = y[:, None]-y[None, :]
            D = np.sqrt(Dx*Dx+Dy*Dy)
            b_ub = np.empty(nrow, dtype=float)
            b_ub[:npair] = D[pair_i, pair_j] - EPS
            b_ub[npair:] = wall - EPS
            if np.min(b_ub) < -1e-9:
                return None, -1.0
            # clamp tiny negatives
            b_ub = np.maximum(b_ub, 0.0)
            res = linprog(c_obj, A_ub=A_ub0, b_ub=b_ub, bounds=bounds_lp, method='highs')
            if not res.success:
                return None, -1.0
            r = np.array(res.x, dtype=float)
            if np.any(~np.isfinite(r)):
                return None, -1.0
            # allow zeros during search; require non-negative
            r = np.maximum(r, 0.0)
            s = float(np.sum(r))
            return r, s
        except Exception:
            return None, -1.0

    # improve fallback with LP on grid centres
    try:
        rt, st = radii_lp(best_x, best_y)
        if rt is not None and st > best_sum and np.min(rt) > 1e-9:
            best_sum = st; best_r = rt
            best_arr = np.column_stack([best_x, best_y, best_r])
    except Exception:
        pass

    def sig(z):
        # numerically safe sigmoid
        z = np.clip(z, -8, 8)
        return 1.0/(1.0+np.exp(-z))

    def decode_template(p, template):
        R = len(template)
        y_raw = p[0:R]; o_raw = p[R:2*R]; s_raw = p[2*R:3*R]
        y0 = 0.05 + 0.90*sig(y_raw)
        f = sig(o_raw)
        sp = 0.08 + 0.12*sig(s_raw)
        # sort rows by y to keep bottom-to-top order
        order = np.argsort(y0)
        y0s = y0[order]; fs = f[order]; sps = sp[order]
        tpls = [template[r] for r in order]
        xs_list = []; ys_list = []
        for yi, fi, si, nn in zip(y0s, fs, sps, tpls):
            tot = (nn-1)*si
            avail = 1.0 - tot
            left_min = 0.02
            left_max = avail - 0.02
            if left_max < left_min:
                left = avail*0.5
            else:
                left = left_min + fi*(left_max-left_min)
            xs = left + np.arange(nn)*si
            xs_list.append(xs)
            ys_list.append(np.full(nn, yi))
        X = np.concatenate(xs_list)
        Y = np.concatenate(ys_list)
        # safety clip (LP will penalize if outside)
        return X, Y

    templates = [
        (6,5,6,5,4),
        (5,6,5,6,4),
        (6,5,5,5,5),
        (5,5,6,5,5),
        (5,5,5,5,6),
        (6,6,5,5,4),
        (4,6,5,6,5),
        (4,5,6,6,5),
        (4,5,5,4,4,4),
        (5,4,5,4,4,4),
    ]
    # keep only those summing to 26
    templates = [t for t in templates if sum(t)==N]

    # logit helper for init
    def logit(q):
        q = np.clip(q, 1e-3, 1-1e-3)
        return np.log(q/(1-q))

    top_centres = []  # list of (sum, x, y, r)

    # Phase 1: structured row search
    PHASE1_END = t0 + 48.0
    for ti, templ in enumerate(templates):
        R = len(templ)
        dim = 3*R
        # 2 starts per template
        for rep in range(2):
            if time.time() > PHASE1_END or time.time()-t0 > DEADLINE-40:
                break
            # init: evenly spaced y, centred, mid spacing
            try:
                y_targ = [(r+0.5)/R for r in range(R)]
                # map to [0.05,0.95] inverse
                q = [(yy-0.05)/0.90 for yy in y_targ]
                p0 = np.zeros(dim)
                p0[0:R] = logit(np.array(q)) + rng.randn(R)*0.4
                p0[R:2*R] = rng.randn(R)*0.6  # f~0.5
                p0[2*R:3*R] = rng.randn(R)*0.6
                # best eval closure
                best_local = [-1e99]
                def obj(p):
                    try:
                        X, Y = decode_template(p, templ)
                        if np.min(X) < 0.005 or np.max(X) > 0.995 or np.min(Y) < 0.005 or np.max(Y) > 0.995:
                            return 10.0  # penalty
                        _, s = radii_lp(X, Y)
                        if s < 0:
                            return 10.0
                        return -s
                    except Exception:
                        return 10.0
                res = minimize(obj, p0, method='Nelder-Mead',
                               options={'maxiter': 70, 'xatol': 1e-3, 'fatol': 1e-5, 'disp': False})
                p_best = res.x if hasattr(res, 'x') else p0
                Xb, Yb = decode_template(p_best, templ)
                rb, sb = radii_lp(Xb, Yb)
                if rb is not None and sb > 0:
                    top_centres.append((sb, Xb.copy(), Yb.copy(), rb.copy(), templ))
                    if sb > best_sum and np.min(rb) > 1e-7:
                        best_sum = sb; best_x = Xb.copy(); best_y = Yb.copy(); best_r = rb.copy()
                        best_arr = np.column_stack([best_x, best_y, best_r])
            except Exception:
                continue
        if time.time() > PHASE1_END:
            break

    # sort top
    try:
        top_centres.sort(key=lambda t: t[0], reverse=True)
    except Exception:
        pass
    # keep best 3 distinct
    elites = top_centres[:3] if len(top_centres) >= 1 else []

    # Phase 1b: break row ties - perturb elites slightly and hill climb with LP
    # quick centre-only hill climbing on current best + elites
    def hill_climb(budget_sec, sig0=0.015):
        nonlocal best_sum, best_x, best_y, best_r, best_arr
        sig = sig0
        noimp = 0
        t_end = time.time() + budget_sec
        while time.time() < t_end and time.time()-t0 < DEADLINE-22:
            if rng.rand() < 0.75:
                idx = np.array([rng.randint(N)])
            else:
                idx = rng.choice(N, size=2, replace=False)
            s = sig*4.0 if rng.rand() < 0.05 else sig
            xt = best_x.copy(); yt = best_y.copy()
            xt[idx] += rng.randn(len(idx))*s
            yt[idx] += rng.randn(len(idx))*s
            xt = np.clip(xt, 0.004, 0.996); yt = np.clip(yt, 0.004, 0.996)
            # separation guard
            bad = False
            for ii in idx:
                dx = xt[ii]-xt; dy = yt[ii]-yt
                dx[ii]=1.0; dy[ii]=1.0
                if np.min(dx*dx+dy*dy) < (0.012**2):
                    bad=True; break
            if bad:
                noimp+=1
                continue
            try:
                ro, so = radii_lp(xt, yt)
            except Exception:
                noimp+=1; continue
            if ro is not None and so > best_sum + 1e-10:
                # require all positive-ish for best update
                if np.min(ro) > 1e-9:
                    best_sum=so; best_x, best_y, best_r = xt, yt, ro
                    best_arr=np.column_stack([best_x,best_y,best_r])
                    noimp=0
                    sig=min(0.035, sig*1.02)
                else:
                    noimp+=1
            else:
                noimp+=1
            if noimp % 250 == 0 and noimp>0:
                sig*=0.85
                if sig < 0.002:
                    sig=0.006
            if noimp > 4000:
                break

    # if elites exist, seed best from top elite already done; add jitter to break exact rows
    if len(elites) > 0:
        # inject small jitter into best to break ties gradually
        try:
            best_x = np.clip(best_x + rng.randn(N)*0.003, 0.004, 0.996)
            best_y = np.clip(best_y + rng.randn(N)*0.003, 0.004, 0.996)
            rb2, sb2 = radii_lp(best_x, best_y)
            if rb2 is not None and sb2 > 0 and np.min(rb2) > 1e-9:
                best_sum=sb2; best_r=rb2; best_arr=np.column_stack([best_x,best_y,best_r])
        except Exception:
            pass
        # also consider 2nd elite as alternative start if close
        for e in elites[1:]:
            try:
                _, ex, ey, er, _ = e
                # jitter
                exj = np.clip(ex + rng.randn(N)*0.004, 0.004, 0.996)
                eyj = np.clip(ey + rng.randn(N)*0.004, 0.004, 0.996)
                rr, ss = radii_lp(exj, eyj)
                if rr is not None and ss > best_sum and np.min(rr) > 1e-9:
                    best_sum=ss; best_x, best_y, best_r = exj, eyj, rr
                    best_arr=np.column_stack([best_x,best_y,best_r])
            except Exception:
                continue

    # short hill climb before joint
    if time.time()-t0 < DEADLINE-30:
        hill_climb(budget_sec=8.0, sig0=0.012)

    # Phase 2: joint penalized L-BFGS-B polish (wall-free not needed, direct)
    bounds_box = [(0.0,1.0)]*(2*N) + [(0.015,0.25)]*N
    def make_fun(W):
        def fun(zz):
            u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
            om2r = 1.0-2.0*rr
            x = u*om2r+rr
            y = vv*om2r+rr
            Dx = x[:,None]-x[None,:]; Dy = y[:,None]-y[None,:]
            D = np.sqrt(Dx*Dx+Dy*Dy)
            Rsum = rr[:,None]+rr[None,:]
            E = Rsum-D
            E.flat[::N+1]=0.0
            D.flat[::N+1]=1.0
            Ep = np.where(E>0, E, 0.0)
            Pen = 0.25*np.sum(Ep*Ep)
            F = -np.sum(rr)+W*Pen
            InvD = 1.0/D; InvD.flat[::N+1]=0.0
            Wmat = Ep*InvD
            Gx = -np.sum(Wmat*Dx, axis=1)*0.5
            Gy = -np.sum(Wmat*Dy, axis=1)*0.5
            sumEp = np.sum(Ep, axis=1)*0.5
            dFdu = W*Gx*om2r
            dFdv = W*Gy*om2r
            dxdr = 1.0-2.0*u; dydr = 1.0-2.0*vv
            dFdr = -np.ones(N)+W*(sumEp+Gx*dxdr+Gy*dydr)
            g = np.empty_like(zz)
            g[:N]=dFdu; g[N:2*N]=dFdv; g[2*N:]=dFdr
            return F, g
        return fun
    WS = [100.0, 1000.0, 10000.0]
    MITS = [80, 80, 100]
    def run_joint(z0):
        z = z0.copy()
        for si, W in enumerate(WS):
            if time.time()-t0 > DEADLINE-14:
                break
            fun = make_fun(W)
            try:
                res = minimize(fun, z, jac=True, method='L-BFGS-B', bounds=bounds_box,
                               options={'maxiter': MITS[si], 'ftol':1e-9, 'gtol':1e-5, 'maxfun':20000})
                z = res.x
            except Exception:
                break
        return z
    def decode_joint(z):
        u=z[:N]; vv=z[N:2*N]; rr=z[2*N:]
        om2r=1.0-2.0*rr
        x=u*om2r+rr; y=vv*om2r+rr
        return x, y, rr
    def encode_joint(x, y, r):
        rc=np.clip(r, 0.015, 0.24)
        denom=1.0-2.0*rc
        # avoid tiny denom
        denom=np.where(np.abs(denom)<0.2, 0.5, denom)
        u=(x-rc)/denom; vv=(y-rc)/denom
        u=np.clip(u,0,1); vv=np.clip(vv,0,1)
        return np.concatenate([u, vv, rc])

    # joint restarts: alternate between elite-perturbed and random/grid
    gx5=[]; gy5=[]
    for ix in range(5):
        for jy in range(5):
            gx5.append((ix+0.5)/5.0); gy5.append((jy+0.5)/5.0)
    gx5=np.array(gx5); gy5=np.array(gy5)
    n_joint = 0
    while time.time()-t0 < DEADLINE-14 and n_joint < 30:
        mode = n_joint % 5
        try:
            if mode==0:
                z0 = encode_joint(np.clip(best_x+rng.randn(N)*0.01,0.01,0.99),
                                  np.clip(best_y+rng.randn(N)*0.01,0.01,0.99),
                                  np.clip(best_r+rng.randn(N)*0.005,0.015,0.25))
            elif mode==1 and len(elites)>0:
                e = elites[rng.randint(len(elites))]
                _, ex, ey, er, _ = e
                z0 = encode_joint(np.clip(ex+rng.randn(N)*0.02,0.01,0.99),
                                  np.clip(ey+rng.randn(N)*0.02,0.01,0.99),
                                  np.clip(er+rng.randn(N)*0.008,0.015,0.25))
            elif mode==2:
                ux=gx5+rng.randn(25)*0.02; uy=gy5+rng.randn(25)*0.02
                ex=rng.rand()*0.8+0.1; ey=rng.rand()*0.8+0.1
                u0=np.append(ux,ex); v0=np.append(uy,ey)
                p=rng.permutation(N)
                u0=np.clip(u0[p],0,1); v0=np.clip(v0[p],0,1)
                r0=rng.uniform(0.06,0.12,size=N)
                # convert x->u approx: use u0 as u directly
                z0=np.concatenate([u0,v0,r0])
            elif mode==3:
                u0=rng.rand(N); v0=rng.rand(N); r0=np.clip(0.095+rng.randn(N)*0.015,0.03,0.16)
                z0=np.concatenate([u0,v0,r0])
            else:
                u0=rng.rand(N); v0=rng.rand(N); r0=rng.uniform(0.04,0.13,size=N)
                z0=np.concatenate([u0,v0,r0])
            zopt=run_joint(z0)
            xh,yh,rh=decode_joint(zopt)
            xh=np.clip(xh,1e-6,1-1e-6); yh=np.clip(yh,1e-6,1-1e-6)
            ro,so=radii_lp(xh,yh)
            if ro is not None and so>best_sum+1e-12 and np.min(ro)>1e-9:
                best_sum=so; best_x=xh.copy(); best_y=yh.copy(); best_r=ro.copy()
                best_arr=np.column_stack([best_x,best_y,best_r])
        except Exception:
            pass
        n_joint+=1

    # final hill climb with remaining time
    if time.time()-t0 < DEADLINE-4:
        remain = DEADLINE-2.5-(time.time()-t0)
        if remain > 2:
            hill_climb(budget_sec=min(remain, 12.0), sig0=0.008)

    # final strict feasibility via uniform shrink
    try:
        Xb, Yb, Rb = best_x.copy(), best_y.copy(), best_r.copy()
        # ensure inside
        Xb=np.clip(Xb,1e-9,1-1e-9); Yb=np.clip(Yb,1e-9,1-1e-9)
        Rb=np.maximum(Rb,1e-9)
        wall_ratio=np.minimum(np.minimum(Xb,1.0-Xb),np.minimum(Yb,1.0-Yb))/Rb
        Dx=Xb[:,None]-Xb[None,:]; Dy=Yb[:,None]-Yb[None,:]
        D=np.sqrt(Dx*Dx+Dy*Dy)
        S=Rb[:,None]+Rb[None,:]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio=D/S
        np.fill_diagonal(ratio, np.inf)
        a=float(np.min(wall_ratio)); b=float(np.min(ratio))
        alpha=min(1.0,a,b)
        alpha=alpha*(1.0-1e-7)-1e-12
        if not np.isfinite(alpha) or alpha<=0:
            alpha=0.99
        if alpha<1.0:
            Rb=Rb*alpha
        # extra tiny shrink for 1e-12 tolerance
        Rb=Rb*(1.0-1e-9)
        best_arr=np.column_stack([Xb,Yb,Rb])
    except Exception:
        best_arr=np.column_stack([best_x,best_y,best_r])

    return best_arr
