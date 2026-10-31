import numpy as np
import time

def solve():
    N = 26
    t0 = time.time()
    BUDGET = 123.0
    rng = np.random.RandomState(0)

    try:
        from scipy.optimize import minimize, linprog
    except Exception:
        k = int(np.ceil(np.sqrt(N)))
        r = 0.5 / k
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) < N:
                    pts.append([(2*i+1)*r, (2*j+1)*r, r*0.999999])
        return np.array(pts)

    # pair indices for LP
    ii_idx, jj_idx = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii_idx.append(i); jj_idx.append(j)
    ii_idx = np.array(ii_idx, dtype=int)
    jj_idx = np.array(jj_idx, dtype=int)
    P = len(ii_idx)
    c_obj = -np.ones(N, dtype=float)
    bounds_lp = [(0.0, None)] * N
    EyeN = np.eye(N, dtype=float)

    def radii_lp(x, y):
        # x,y assumed inside (0,1)
        wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        if np.min(wall) < 1e-9 + 1e-12:
            return None, -1.0
        Dx = x[:, None] - x[None, :]
        Dy = y[:, None] - y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        dpair = D[ii_idx, jj_idx]
        eps = 1e-9
        if np.min(dpair) < eps:  # would need negative bound -> infeasible
            return None, -1.0
        if np.min(wall - eps) <= 0:
            return None, -1.0
        A_ub = np.zeros((P + N, N), dtype=float)
        b_ub = np.zeros(P + N, dtype=float)
        # fill pairs vectorized
        A_ub[np.arange(P), ii_idx] = 1.0
        A_ub[np.arange(P), jj_idx] = 1.0
        b_ub[:P] = dpair - eps
        A_ub[P:, :] = EyeN
        b_ub[P:] = wall - eps
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        except Exception:
            return None, -1.0
        if res.status != 0:
            return None, -1.0
        r = np.asarray(res.x, dtype=float)
        # clip tiny negatives
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return r, s

    def shrink_to_feasible(x, y, r):
        x = np.clip(np.asarray(x, float), 1e-9, 1.0-1e-9)
        y = np.clip(np.asarray(y, float), 1e-9, 1.0-1e-9)
        r = np.maximum(np.asarray(r, float), 1e-9)
        wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        Dx = x[:,None]-x[None,:]
        Dy = y[:,None]-y[None,:]
        D = np.sqrt(Dx*Dx+Dy*Dy)
        np.fill_diagonal(D, 1e100)
        Rsum = r[:,None]+r[None,:]
        with np.errstate(divide='ignore', invalid='ignore'):
            pr = np.min(D / (Rsum+1e-300))
            wr = np.min(wall / (r+1e-300))
        s = min(float(pr), float(wr))
        if not np.isfinite(s) or s <= 0:
            s = 0.5
        s = min(s, 1.0) * (1.0 - 1e-7)
        # extra margin
        s *= (1.0 - 1e-9)
        r2 = r * s
        r2 = np.maximum(r2, 1e-9)
        return x, y, r2

    bounds_box = [(0.0, 1.0)]*(2*N) + [(0.005, 0.30)]*N

    def run_joint(z0, Ws, maxiters):
        z = z0.copy()
        for W in Ws:
            if time.time()-t0 > BUDGET-8:
                break
            def fun(zz, W=W):
                u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
                x = rr*(1.0-2.0*u)+u
                y = rr*(1.0-2.0*vv)+vv
                Dx = x[:,None]-x[None,:]
                Dy = y[:,None]-y[None,:]
                D = np.sqrt(Dx*Dx+Dy*Dy+1e-16)
                np.fill_diagonal(D, 1.0)
                E = rr[:,None]+rr[None,:]-D
                np.fill_diagonal(E, 0.0)
                Ep = np.maximum(E, 0.0)
                Pen = 0.5*np.sum(Ep*Ep)
                F = -np.sum(rr)+W*Pen
                InvD = 1.0/D
                np.fill_diagonal(InvD, 0.0)
                Wmat = Ep*InvD
                Gx = -2.0*np.sum(Wmat*Dx, axis=1)
                Gy = -2.0*np.sum(Wmat*Dy, axis=1)
                sumEp = np.sum(Ep, axis=1)
                om2r = 1.0-2.0*rr
                dFdu = W*Gx*om2r
                dFdv = W*Gy*om2r
                dFdr = -np.ones(N)+W*(2.0*sumEp+Gx*(1.0-2.0*u)+Gy*(1.0-2.0*vv))
                g = np.concatenate([dFdu, dFdv, dFdr])
                return F, g
            try:
                res = minimize(fun, z, method='L-BFGS-B', jac=True, bounds=bounds_box,
                               options={'maxiter': maxiters, 'ftol':1e-9, 'gtol':1e-5, 'maxfun':20000})
                z = res.x
            except Exception:
                break
        return z

    best_sum = -1.0
    best_x = None; best_y = None; best_r = None

    def consider(x, y):
        global_best = False
        nonlocal_best = [False]
        return None

    def try_update(x, y):
        nonlocal best_sum, best_x, best_y, best_r
        # clip centres strictly inside
        x = np.clip(x, 1e-7, 1.0-1e-7)
        y = np.clip(y, 1e-7, 1.0-1e-7)
        r, s = radii_lp(x, y)
        if r is not None and s > best_sum + 1e-12:
            best_sum = float(s)
            best_x = x.copy(); best_y = y.copy(); best_r = r.copy()
            return True
        return False

    # Phase 0: quick random LP screening (diverse centres)
    # grid base
    gx=[]; gy=[]
    for ix in range(6):
        for jy in range(5):
            gx.append((ix+0.5)/6.0); gy.append((jy+0.5)/5.0)
    gx=np.array(gx); gy=np.array(gy)
    n_screen = 250
    for k in range(n_screen):
        if time.time()-t0 > BUDGET-70:
            break
        if k % 3 == 0:
            x = rng.rand(N); y = rng.rand(N)
            x = np.clip(x, 0.05, 0.95); y = np.clip(y, 0.05, 0.95)
        elif k % 3 == 1:
            perm = rng.permutation(len(gx))[:N]
            x = gx[perm]+(rng.rand(N)-0.5)*0.08
            y = gy[perm]+(rng.rand(N)-0.5)*0.08
            x = np.clip(x, 0.02, 0.98); y = np.clip(y, 0.02, 0.98)
        else:
            # jittered 5x5 + extra
            bx=[]; by=[]
            for ix in range(5):
                for jy in range(5):
                    bx.append((ix+0.5)/5.0); by.append((jy+0.5)/5.0)
            bx=np.array(bx); by=np.array(by)
            # pick 25 of them + random one
            perm = rng.permutation(25)[:25]
            x25 = bx[perm]+rng.randn(25)*0.01
            y25 = by[perm]+rng.randn(25)*0.01
            xe = rng.rand(1)*0.9+0.05; ye = rng.rand(1)*0.9+0.05
            x = np.append(x25, xe); y = np.append(y25, ye)
            x = np.clip(x, 0.02, 0.98); y = np.clip(y, 0.02, 0.98)
        try_update(x, y)

    # Phase 1: joint penalty multi-starts
    n_joint = 28
    for restart in range(n_joint):
        if time.time()-t0 > BUDGET-45:
            break
        if restart % 2 == 0:
            u0 = rng.rand(N); v0 = rng.rand(N)
        else:
            perm = rng.permutation(len(gx))[:N]
            u0 = gx[perm]+(rng.rand(N)-0.5)*0.06
            v0 = gy[perm]+(rng.rand(N)-0.5)*0.06
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
        if restart % 3 == 0:
            r0 = 0.07+0.04*rng.rand(N)
        else:
            r0 = np.clip(0.04+0.09*rng.rand(N), 0.015, 0.28)
        z0 = np.concatenate([u0, v0, r0])
        z = run_joint(z0, Ws=(20.0, 200.0, 2000.0), maxiters=220)
        u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
        x = rr*(1.0-2.0*u)+u
        y = rr*(1.0-2.0*vv)+vv
        try_update(x, y)
        # also try extra high-W polish if promising
        if best_sum > 0 and (time.time()-t0 < BUDGET-50):
            # quick extra polish from same start with higher W if near best?
            pass

    # Phase 2: intensive hill climbing on centres with LP
    if best_x is None:
        # fallback: use joint best via shrink
        x = rng.rand(N)*0.9+0.05; y = rng.rand(N)*0.9+0.05
        r = np.full(N, 0.08)
        x, y, r = shrink_to_feasible(x, y, r)
        return np.column_stack((x, y, r))

    # keep small elite pool
    pool_x = [best_x.copy()]; pool_y = [best_y.copy()]; pool_s = [best_sum]
    # add a few perturbed copies of best to pool
    for _ in range(3):
        xp = best_x + rng.randn(N)*0.01
        yp = best_y + rng.randn(N)*0.01
        xp = np.clip(xp, 1e-7, 1-1e-7); yp = np.clip(yp, 1e-7, 1-1e-7)
        r_, s_ = radii_lp(xp, yp)
        if r_ is not None:
            pool_x.append(xp); pool_y.append(yp); pool_s.append(s_)

    sigmas = [0.02, 0.012, 0.007, 0.004, 0.002, 0.001]
    it = 0
    noimp = 0
    while time.time()-t0 < BUDGET-4:
        it += 1
        # choose base: mostly best, sometimes from pool
        if rng.rand() < 0.85 or len(pool_x)==1:
            bx = best_x; by = best_y
        else:
            idx = rng.randint(len(pool_x))
            bx = pool_x[idx]; by = pool_y[idx]
        sig = sigmas[rng.randint(len(sigmas))]
        # occasional large shake
        if rng.rand() < 0.05:
            sig = 0.035
        if rng.rand() < 0.70:
            # single-circle move
            k = rng.randint(N)
            nx = bx.copy(); ny = by.copy()
            nx[k] += rng.randn()*sig
            ny[k] += rng.randn()*sig
            nx[k] = np.clip(nx[k], 1e-7, 1-1e-7)
            ny[k] = np.clip(ny[k], 1e-7, 1-1e-7)
        else:
            nx = bx + rng.randn(N)*sig*0.6
            ny = by + rng.randn(N)*sig*0.6
            nx = np.clip(nx, 1e-7, 1-1e-7); ny = np.clip(ny, 1e-7, 1-1e-7)
        r_, s_ = radii_lp(nx, ny)
        if r_ is not None and s_ > best_sum + 1e-12:
            best_sum = float(s_); best_x = nx.copy(); best_y = ny.copy(); best_r = r_.copy()
            pool_x.append(nx.copy()); pool_y.append(ny.copy()); pool_s.append(s_)
            if len(pool_x) > 8:
                # drop worst
                imin = int(np.argmin(pool_s))
                pool_x.pop(imin); pool_y.pop(imin); pool_s.pop(imin)
            noimp = 0
        else:
            noimp += 1
        # periodic joint refinement from best (escape + fine tune)
        if it % 700 == 0 and (time.time()-t0 < BUDGET-15):
            # convert best to u,v,r and run short joint polish
            # invert mapping: u = (x - r)/(1-2r) approx; need r<0.5
            rc = np.minimum(best_r, 0.30)
            denom = 1.0-2.0*rc
            # avoid div by zero
            denom = np.where(np.abs(denom)<0.1, np.sign(denom)*0.1+1e-9, denom)
            u0 = (best_x - rc)/denom
            v0 = (best_y - rc)/denom
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
            r0 = np.clip(rc, 0.005, 0.30)
            z0 = np.concatenate([u0, v0, r0])
            # small random kick
            z0[:2*N] = np.clip(z0[:2*N]+rng.randn(2*N)*0.01, 0.0, 1.0)
            z = run_joint(z0, Ws=(200.0, 2000.0), maxiters=150)
            u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
            x = rr*(1.0-2.0*u)+u; y = rr*(1.0-2.0*vv)+vv
            r_, s_ = radii_lp(x, y)
            if r_ is not None and s_ > best_sum + 1e-12:
                best_sum=float(s_); best_x=x.copy(); best_y=y.copy(); best_r=r_.copy()

    # final safety shrink
    x, y, r = shrink_to_feasible(best_x, best_y, best_r)
    arr = np.column_stack((x, y, r))
    return arr
