import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 122.0
    rng = np.random.default_rng(0)

    # pair matrix for LP
    I, J = np.triu_indices(n, 1)
    m = len(I)
    A0 = np.zeros((m, n), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cvec = -np.ones(n, dtype=float)
    try:
        from scipy.optimize import minimize, linprog
        has_opt = True
    except Exception:
        has_opt = False

    def lp_eval(xc, yc):
        xc = np.asarray(xc, float); yc = np.asarray(yc, float)
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc)) - 1e-9
        if np.any(wall < -1e-12):
            return -1.0, None
        wall = np.maximum(wall, 0.0)
        # quick reject if centres coincide
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx*dx + dy*dy) - 1e-9
        if np.any(d < -1e-12):
            return -1.0, None
        d = np.maximum(d, 0.0)
        bnds = [(0.0, float(wi)) for wi in wall]
        try:
            res = linprog(cvec, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0) * (1.0 - 1e-7)
        r = np.maximum(r - 1e-12, 0.0)
        return float(np.sum(r)), r

    def uniform_scale(X, Y, R):
        R = np.maximum(np.asarray(R, float), 1e-12)
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            rw = wall / R
            rp = D / S
        np.fill_diagonal(rp, np.inf)
        a = float(np.min(rw)); b = float(np.min(rp))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.finite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha

    # fallback
    k = int(np.ceil(np.sqrt(n)))
    rr0 = 0.5 / k
    fb = []
    for ix in range(k):
        for jy in range(k):
            if len(fb) < n:
                fb.append([(2*ix+1)*rr0, (2*jy+1)*rr0, rr0*0.999])
    fb_arr = np.array(fb, dtype=float)
    best_arr = fb_arr.copy()
    # scale fallback
    try:
        Rs, _ = uniform_scale(best_arr[:,0], best_arr[:,1], best_arr[:,2])
        best_arr[:,2] = Rs
    except Exception:
        pass
    best_sum = float(np.sum(best_arr[:,2]))
    best_xc = best_arr[:,0].copy(); best_yc = best_arr[:,1].copy(); best_r = best_arr[:,2].copy()
    # try LP on fallback grid
    try:
        s0, r0 = lp_eval(best_xc, best_yc)
        if s0 > best_sum:
            best_sum = s0; best_r = r0.copy()
            best_arr = np.column_stack([best_xc, best_yc, best_r])
    except Exception:
        pass

    # joint penalty fun factory
    def make_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn); gr_pen = np.zeros(nn)
            pen = 0.0
            v1 = R - X
            mm = v1 > 0
            if np.any(mm):
                pen += np.sum(v1[mm]**2); gx[mm] += -2.0*v1[mm]; gr_pen[mm] += 2.0*v1[mm]
            v2 = X + R - 1.0
            mm = v2 > 0
            if np.any(mm):
                pen += np.sum(v2[mm]**2); gx[mm] += 2.0*v2[mm]; gr_pen[mm] += 2.0*v2[mm]
            v3 = R - Y
            mm = v3 > 0
            if np.any(mm):
                pen += np.sum(v3[mm]**2); gy[mm] += -2.0*v3[mm]; gr_pen[mm] += 2.0*v3[mm]
            v4 = Y + R - 1.0
            mm = v4 > 0
            if np.any(mm):
                pen += np.sum(v4[mm]**2); gy[mm] += 2.0*v4[mm]; gr_pen[mm] += 2.0*v4[mm]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            np.fill_diagonal(D, 1.0)
            Mask = V > 0
            if np.any(Mask):
                pen += 0.5*np.sum(V[Mask]**2)
                Q = np.zeros((nn, nn))
                Q[Mask] = -2.0*V[Mask]/D[Mask]
                gx += np.sum(Q*Dx, axis=1)
                gy += np.sum(Q*Dy, axis=1)
                G = np.zeros((nn, nn))
                G[Mask] = 2.0*V[Mask]
                gr_pen += np.sum(G, axis=1)
            loss = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:nn] = W*gx
            g[nn:2*nn] = W*gy
            g[2*nn:3*nn] = -1.0 + W*gr_pen
            return loss, g
        return fun
    fun26 = make_fun(n) if has_opt else None
    bounds78 = [(0.0,1.0)]*(2*n) + [(0.02,0.20)]*n

    def joint_polish(X0, Y0, R0, Ws=(1e2,1e3,1e4,1e5), iters=(150,150,150,200)):
        v = np.concatenate([X0, Y0, R0])
        for W, mi in zip(Ws, iters):
            if time.time()-t0 > DEADLINE-6:
                break
            try:
                res = minimize(fun26, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds78, options={'maxiter': mi, 'ftol':1e-9, 'gtol':1e-7})
                v = res.x
            except Exception:
                break
        return np.clip(v[0:n],0,1), np.clip(v[n:2*n],0,1), np.clip(v[2*n:3*n],0.005,0.25)

    def hill_climb(xc, yc, cur_sum, cur_r, budget_s, rng_h):
        t_end = time.time() + budget_s
        sigmas = [0.02, 0.01, 0.005, 0.0025, 0.001]
        for sig in sigmas:
            if time.time() > t_end or time.time()-t0 > DEADLINE-4:
                break
            ntry = 70 if sig > 0.003 else 50
            for tt in range(ntry):
                if time.time() > t_end or time.time()-t0 > DEADLINE-4:
                    break
                xc2 = xc.copy(); yc2 = yc.copy()
                if rng_h.random() < 0.2:
                    xc2 = xc2 + rng_h.normal(0, sig*0.5, n)
                    yc2 = yc2 + rng_h.normal(0, sig*0.5, n)
                else:
                    idx = rng_h.integers(n)
                    xc2[idx] += rng_h.normal(0, sig)
                    yc2[idx] += rng_h.normal(0, sig)
                xc2 = np.clip(xc2, 0.001, 0.999); yc2 = np.clip(yc2, 0.001, 0.999)
                s, r = lp_eval(xc2, yc2)
                if s > cur_sum + 1e-12:
                    xc, yc, cur_sum, cur_r = xc2, yc2, s, r
        return xc, yc, cur_sum, cur_r

    def consider(X, Y, Runi_or_None):
        nonlocal best_sum, best_xc, best_yc, best_r, best_arr
        # uniform scaled
        try:
            if Runi_or_None is not None:
                Rs, _ = uniform_scale(X, Y, Runi_or_None)
                s = float(np.sum(Rs))
                if s > best_sum:
                    best_sum = s; best_xc = X.copy(); best_yc = Y.copy(); best_r = Rs.copy()
                    best_arr = np.column_stack([X, Y, Rs])
        except Exception:
            pass
        try:
            s, r = lp_eval(X, Y)
            if s > best_sum:
                best_sum = s; best_xc = X.copy(); best_yc = Y.copy(); best_r = r.copy()
                best_arr = np.column_stack([X, Y, r])
                return True
        except Exception:
            pass
        return False

    # ---------- template enumeration ----------
    # generate 5-row patterns entries 4..6 sum 26
    patterns = []
    for a in range(4,7):
        for b in range(4,7):
            for cc in range(4,7):
                for d in range(4,7):
                    for e in range(4,7):
                        if a+b+cc+d+e == 26:
                            patterns.append((a,b,cc,d,e))
    # 6-row patterns 4..5 sum 26 (four 4s two 5s permutations)
    pats6 = set()
    import itertools
    for perm in set(itertools.permutations((4,4,4,4,5,5))):
        pats6.add(perm)
    patterns.extend(list(pats6))
    # key patterns first
    key = [(5,5,6,5,5),(6,5,5,5,5),(5,6,5,5,5),(5,5,5,6,5),(5,5,5,5,6)]
    # order patterns: keys first then rest
    ordered = []
    for kk in key:
        if kk in patterns:
            ordered.append(kk)
    for p in patterns:
        if p not in ordered:
            ordered.append(p)

    def centres_for_pattern(pat, stagger, jit, rng_j):
        Rrows = len(pat)
        margin = 0.085
        ys = margin + (1-2*margin)*(np.arange(Rrows)+0.5)/Rrows
        # add small per-row y jitter
        ys = ys + rng_j.normal(0, 0.004, Rrows)
        xs_list=[]; ys_list=[]
        for k, c in enumerate(pat):
            avail = 1-2*margin
            sp = avail / c
            xs = margin + sp*(np.arange(c)+0.5)
            if stagger==1 and (k%2==1):
                xs = xs + sp*0.28
            elif stagger==2 and (k%2==1):
                xs = xs - sp*0.28
            elif stagger==3 and (k%2==0):
                xs = xs + sp*0.14
            # wrap/clip: reflect overflow
            for ii in range(len(xs)):
                if xs[ii] > 1-margin:
                    xs[ii] -= sp
                if xs[ii] < margin:
                    xs[ii] += sp
            xs = xs + rng_j.normal(0, 0.004, c)
            xs_list.append(xs)
            ys_list.append(np.full(c, ys[k]) + rng_j.normal(0, 0.004, c))
        X = np.concatenate(xs_list); Y = np.concatenate(ys_list)
        X = np.clip(X, 0.03, 0.97); Y = np.clip(Y, 0.03, 0.97)
        if jit:
            X = X + rng_j.normal(0, 0.003, n); Y = Y + rng_j.normal(0, 0.003, n)
            X = np.clip(X, 0.02, 0.98); Y = np.clip(Y, 0.02, 0.98)
        return X, Y

    # SLSQP low-dim template optimisation for the 3 main row patterns
    if has_opt and (time.time()-t0 < DEADLINE-60):
        for pat in [(5,5,6,5,5),(6,5,5,5,5),(5,6,5,5,5)]:
            if time.time()-t0 > 22:
                break
            Rrows = len(pat)
            # map circle -> row
            row_of = []
            for k, c in enumerate(pat):
                row_of.extend([k]*c)
            row_of = np.array(row_of, int)
            # initial params: y,o,s,q
            margin0 = 0.09
            y0 = margin0 + (1-2*margin0)*(np.arange(Rrows)+0.5)/Rrows
            o0 = np.full(Rrows, margin0)
            s0 = np.array([(1-2*margin0)/(c-1) if c>1 else 0.2 for c in pat], float)
            q0 = np.full(Rrows, 0.088)
            # adjust 6-row
            for k, c in enumerate(pat):
                if c==6:
                    q0[k]=0.082
            p0 = np.concatenate([y0, o0, s0, q0])
            # bounds
            bnds = [(0.05,0.95)]*Rrows + [(0.02,0.5)]*Rrows + [(0.08,0.30)]*Rrows + [(0.03,0.15)]*Rrows
            def unpack(p):
                y = p[0:Rrows]; o = p[Rrows:2*Rrows]; s = p[2*Rrows:3*Rrows]; q = p[3*Rrows:4*Rrows]
                return y, o, s, q
            def obj(p):
                _,_,_,q = unpack(p)
                tot = 0.0
                for k, c in enumerate(pat):
                    tot += c*q[k]
                return -tot
            # constraints list
            cons = []
            # wall constraints via lambda with closure: use functions returning array? SLSQP needs scalar constraints; build many dicts
            def make_cons():
                cl = []
                # y walls
                for k in range(Rrows):
                    cl.append({'type':'ineq','fun': (lambda p, k=k: unpack(p)[0][k]-unpack(p)[3][k])})
                    cl.append({'type':'ineq','fun': (lambda p, k=k: 1.0-unpack(p)[0][k]-unpack(p)[3][k])})
                    cl.append({'type':'ineq','fun': (lambda p, k=k: unpack(p)[1][k]-unpack(p)[3][k])})
                    # right wall: 1-(o+(c-1)s)-q
                    cl.append({'type':'ineq','fun': (lambda p, k=k: 1.0-(unpack(p)[1][k]+(pat[k]-1)*unpack(p)[2][k])-unpack(p)[3][k])})
                # pairwise: need centre positions
                # build index ranges
                idx_start=[]
                ssum=0
                for c in pat:
                    idx_start.append(ssum); ssum+=c
                # precompute pairs
                for a in range(n):
                    for b in range(a+1, n):
                        def f(p, a=a, b=b):
                            y,o,s,q = unpack(p)
                            # find row and col of a,b
                            # locate
                            ra = row_of[a]; rb = row_of[b]
                            # column within row: need offset index
                            # compute ca = a - start[ra]
                            ca = a - idx_start[ra]; cb = b - idx_start[rb]
                            xa = o[ra]+ca*s[ra]; ya = y[ra]
                            xb = o[rb]+cb*s[rb]; yb = y[rb]
                            return np.sqrt((xa-xb)**2+(ya-yb)**2)-q[ra]-q[rb]
                        cl.append({'type':'ineq','fun': f})
                return cl
            try:
                cl = make_cons()
                res = minimize(obj, p0, method='SLSQP', bounds=bnds, constraints=cl,
                               options={'maxiter': 60, 'ftol':1e-6, 'disp': False})
                if res.success or res.fun < -2.2:
                    y,o,s,q = unpack(res.x)
                    # build centres
                    X=[];Y=[];Rr=[]
                    for k, c in enumerate(pat):
                        for j in range(c):
                            X.append(o[k]+j*s[k]); Y.append(y[k]); Rr.append(q[k])
                    X=np.array(X);Y=np.array(Y);Rr=np.array(Rr)
                    # small jitter then joint polish later via pool
                    consider(np.clip(X,0.02,0.98), np.clip(Y,0.02,0.98), Rr)
                    # store for later full polish: push to template pool via best? also run joint polish now
                    if time.time()-t0 < DEADLINE-50:
                        Xp,Yp,Rp = joint_polish(np.clip(X,0,1), np.clip(Y,0,1), np.clip(Rr,0.02,0.2),
                                               Ws=(2e2,2e3,2e4), iters=(120,120,150))
                        consider(Xp,Yp,Rp)
                        s_lp, r_lp = lp_eval(Xp,Yp)
                        if s_lp > best_sum:
                            best_sum=s_lp; best_xc=Xp.copy(); best_yc=Yp.copy(); best_r=r_lp.copy()
                            best_arr=np.column_stack([Xp,Yp,r_lp])
            except Exception:
                pass

    # screen all templates by LP only (fast)
    scored = []
    if time.time()-t0 < DEADLINE-40:
        for pi, pat in enumerate(ordered):
            if time.time()-t0 > DEADLINE-55:
                break
            for stag in (0,1,2):
                # use fresh rng for reproducibility
                rj = np.random.default_rng(1000+pi*10+stag)
                X, Y = centres_for_pattern(pat, stag, True, rj)
                s, _ = lp_eval(X, Y)
                scored.append((s, pat, stag, X, Y))
                # also transpose
                s2, _ = lp_eval(Y, X)
                scored.append((s2, pat, stag, Y.copy(), X.copy()))
    scored.sort(key=lambda t: -(t[0] if np.finite(t[0]) else -1))
    # take top K
    topK = 12
    pool = []
    for kki in range(min(topK, len(scored))):
        s, pat, stag, X, Y = scored[kki]
        if s < 1.5:
            continue
        pool.append((X, Y))

    # tiered radii assignment helper
    tier_vec = np.array([0.125]+[0.108]*4+[0.092]*16+[0.062]*5)
    # full polish of pool
    for ppi, (X0, Y0) in enumerate(pool):
        if time.time()-t0 > DEADLINE-35:
            break
        # three radius inits: uniform, tier-centre, tier-random
        R_inits = []
        R_inits.append(np.full(n, 0.088))
        # centre-weighted tier: largest near centre
        d2 = (X0-0.5)**2+(Y0-0.5)**2
        order = np.argsort(d2)
        Rt = np.empty(n)
        # largest to most central, small to corners/edges? assign tier sorted descending to central order
        tier_sorted = np.sort(tier_vec)[::-1]
        Rt[order] = tier_sorted
        R_inits.append(Rt)
        Rr = 0.07+0.04*rng.random(n)
        R_inits.append(Rr)
        for R0 in R_inits:
            if time.time()-t0 > DEADLINE-30:
                break
            Xp, Yp, Rp = joint_polish(np.clip(X0+rng.normal(0,0.003,n),0,1),
                                      np.clip(Y0+rng.normal(0,0.003,n),0,1),
                                      np.clip(R0,0.02,0.2))
            improved = consider(Xp, Yp, Rp)
            # hill climb briefly on promising
            s_lp, r_lp = lp_eval(Xp, Yp)
            if s_lp > best_sum - 0.02:  # near best, invest hill climb
                Xh, Yh, sh, rh = hill_climb(Xp, Yp, s_lp if s_lp>0 else -1, r_lp, budget_s=1.2, rng_h=rng)
                if sh > best_sum:
                    best_sum=sh; best_xc=Xh.copy(); best_yc=Yh.copy(); best_r=rh.copy()
                    best_arr=np.column_stack([Xh,Yh,rh])

    # graduated 25 -> 26 homotopy from 5x5 grid
    if time.time()-t0 < DEADLINE-25:
        try:
            gx=[];gy=[]
            for ix in range(5):
                for jy in range(5):
                    gx.append((ix+0.5)/5.0); gy.append((jy+0.5)/5.0)
            gx=np.array(gx); gy=np.array(gy)
            r25 = np.full(25, 0.09)
            # find largest void given equal radii
            best_v=-1; bx=0.5; by=0.5
            for px in np.linspace(0.02,0.98,60):
                for py in np.linspace(0.02,0.98,60):
                    wall = min(px,1-px,py,1-py)
                    dd = np.sqrt((px-gx)**2+(py-gy)**2)-r25
                    cl = min(wall, float(np.min(dd)))
                    if cl>best_v:
                        best_v=cl; bx=px; by=py
            X26 = np.append(gx, bx); Y26 = np.append(gy, by)
            # homotopy caps for new circle
            caps = [0.03, 0.05, 0.07, 0.20]
            R26 = np.full(26, 0.088); R26[25]=0.02
            Xc=X26.copy(); Yc=Y26.copy(); Rc=R26.copy()
            for cap in caps:
                if time.time()-t0 > DEADLINE-20:
                    break
                bnds_h = [(0.0,1.0)]*(2*n)+[(0.02,0.20)]*25+[(0.005,cap)]
                v = np.concatenate([Xc,Yc,Rc])
                for W in (5e2, 5e3, 2e4):
                    try:
                        res = minimize(fun26, v, args=(W,), jac=True, method='L-BFGS-B',
                                       bounds=bnds_h, options={'maxiter':120,'ftol':1e-9,'gtol':1e-7})
                        v=res.x
                    except Exception:
                        break
                Xc=np.clip(v[0:n],0,1); Yc=np.clip(v[n:2*n],0,1); Rc=np.clip(v[2*n:3*n],0.005,0.25)
            consider(Xc,Yc,Rc)
            s_lp,r_lp = lp_eval(Xc,Yc)
            if s_lp>0:
                Xh,Yh,sh,rh = hill_climb(Xc,Yc,s_lp,r_lp,budget_s=2.0,rng_h=rng)
                if sh>best_sum:
                    best_sum=sh; best_xc=Xh.copy(); best_yc=Yh.copy(); best_r=rh.copy()
                    best_arr=np.column_stack([Xh,Yh,rh])
                elif s_lp>best_sum:
                    best_sum=s_lp; best_xc=Xc.copy(); best_yc=Yc.copy(); best_r=r_lp.copy()
                    best_arr=np.column_stack([Xc,Yc,r_lp])
        except Exception:
            pass

    # remaining time: random restarts + hill climb on best
    restart=0
    while time.time()-t0 < DEADLINE-6:
        if restart>60:
            break
        # init
        if restart%4==0 and len(pool)>0:
            bi = restart % len(pool)
            X0 = pool[bi][0]+rng.normal(0,0.008,n)
            Y0 = pool[bi][1]+rng.normal(0,0.008,n)
            X0=np.clip(X0,0.02,0.98); Y0=np.clip(Y0,0.02,0.98)
            R0=0.07+0.04*rng.random(n)
        elif restart%4==1:
            X0=rng.random(n)*0.9+0.05; Y0=rng.random(n)*0.9+0.05
            R0=0.085+0.025*rng.random(n)
        elif restart%4==2:
            # hex-like 6x5 subset
            pts=[]
            for ix in range(6):
                for jy in range(5):
                    xx=(ix+0.5)/6.0; yy=(jy+0.5)/5.0
                    if jy%2==1:
                        xx+=0.5/6.0*0.5
                        if xx>0.95: xx-=1.0/6.0
                    pts.append((xx,yy))
            pts=np.array(pts)
            perm=rng.permutation(len(pts))[:n]
            X0=pts[perm,0]+rng.normal(0,0.005,n)
            Y0=pts[perm,1]+rng.normal(0,0.005,n)
            X0=np.clip(X0,0.02,0.98); Y0=np.clip(Y0,0.02,0.98)
            R0=0.08+0.03*rng.random(n)
        else:
            X0=rng.random(n)*0.9+0.05; Y0=rng.random(n)*0.9+0.05
            R0=np.empty(n); R0[rng.permutation(n)[:1]]=0.125; R0[np.setdiff1d(np.arange(n),np.argmax(R0))[:4]]=0.108
            # fill rest random medium/small
            R0 = 0.075+0.035*rng.random(n)
        Xp,Yp,Rp = joint_polish(X0,Y0,R0, Ws=(1e2,1e3,1e4,5e4), iters=(100,100,120,150))
        consider(Xp,Yp,Rp)
        restart+=1

    # final intensive hill climb on best
    if time.time()-t0 < DEADLINE-2:
        try:
            s_cur, r_cur = lp_eval(best_xc, best_yc)
            if s_cur>0:
                Xh,Yh,sh,rh = hill_climb(best_xc, best_yc, s_cur, r_cur, budget_s=max(1.0, DEADLINE-2-(time.time()-t0)), rng_h=rng)
                if sh>best_sum:
                    best_sum=sh; best_xc=Xh.copy(); best_yc=Yh.copy(); best_r=rh.copy()
                    best_arr=np.column_stack([Xh,Yh,rh])
        except Exception:
            pass

    # final safety shrink
    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rfs,_ = uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rfs])
    except Exception:
        pass
    return best_arr
