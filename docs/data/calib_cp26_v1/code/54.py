import numpy as np
import time

def solve():
    import numpy as np
    import time
    from scipy.optimize import minimize, linprog
    N = 26
    t0 = time.time()
    DEADLINE = 119.0
    rng = np.random.default_rng(12345)

    # pair incidence for LP
    I, J = np.triu_indices(N, 1)
    m = len(I)
    A0 = np.zeros((m, N), dtype=float)
    A0[np.arange(m), I] = 1.0
    A0[np.arange(m), J] = 1.0
    cvec = -np.ones(N, dtype=float)
    EPS = 1e-9

    def lp_eval(xc, yc):
        xc = np.asarray(xc, float); yc = np.asarray(yc, float)
        wall = np.minimum(np.minimum(xc, 1.0 - xc), np.minimum(yc, 1.0 - yc))
        if np.any(wall < 1e-9 + 1e-12):
            return -1.0, None
        dx = xc[I] - xc[J]
        dy = yc[I] - yc[J]
        d = np.sqrt(dx*dx + dy*dy) - EPS
        # need positive gaps for lower bound feasibility
        # allow zero lower bound instead of 0.005 to keep feasible
        ub = wall - EPS
        if np.any(ub <= 0):
            return -1.0, None
        # quick reject if any pair too close to allow positive radii
        # (still feasible with 0, so don't reject)
        bnds = [(0.0, float(ub[i])) for i in range(N)]
        try:
            res = linprog(cvec, A_ub=A0, b_ub=d, bounds=bnds, method='highs')
        except Exception:
            return -1.0, None
        if res.status != 0:
            return -1.0, None
        r = np.asarray(res.x, dtype=float)
        r = np.maximum(r, 0.0)
        s = float(np.sum(r))
        return s, r

    # fallback
    kgr = int(np.ceil(np.sqrt(N)))
    rgr = 0.5 / kgr
    fb = []
    for ix in range(kgr):
        for jy in range(kgr):
            if len(fb) < N:
                fb.append([(2*ix+1)*rgr, (2*jy+1)*rgr, rgr*0.999])
    fb_arr = np.array(fb, dtype=float)
    best_sum = -1.0
    best_x = fb_arr[:,0].copy(); best_y = fb_arr[:,1].copy(); best_r = fb_arr[:,2].copy()*0.9
    # try lp on grid
    s_fb, r_fb = lp_eval(fb_arr[:,0], fb_arr[:,1])
    if r_fb is not None:
        best_sum = s_fb; best_x = fb_arr[:,0].copy(); best_y = fb_arr[:,1].copy(); best_r = r_fb.copy()
    pool = []
    if best_sum > 0:
        pool.append((best_sum, best_x.copy(), best_y.copy()))

    # ---------- Lubachevsky-Stillinger growth ----------
    def ls_grow(x, y, steps, base_dr, beta, k_rep, mu, kick_sig, alpha_small, seed_jitter):
        # x,y initial centres, common small radius
        r = np.full(N, 0.022 + seed_jitter*0.008, dtype=float)
        vx = np.zeros(N, dtype=float); vy = np.zeros(N, dtype=float)
        # random corner for compression events
        corners = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0],[0.5,0.5]])
        for t in range(steps):
            Dx = x[:, None] - x[None, :]
            Dy = y[:, None] - y[None, :]
            D2 = Dx*Dx + Dy*Dy
            np.fill_diagonal(D2, 1.0)
            D = np.sqrt(D2)
            S = r[:, None] + r[None, :]
            O = S - D
            np.fill_diagonal(O, 0.0)
            np.maximum(O, 0.0, out=O)
            press = np.sum(O, axis=1)
            # wall overlap pressure
            # distances to walls
            dw = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y)) - r
            wov = np.maximum(-dw, 0.0)
            press = press + 2.0*wov
            # repulsion displacements
            # unit vectors
            invD = 1.0 / (D + 1e-12)
            # only overlapping contribute: weight by O
            Wm = O * invD
            # Wm symmetric, sum
            dx_rep = 0.5 * k_rep * (np.sum(Wm * Dx, axis=1))
            dy_rep = 0.5 * k_rep * (np.sum(Wm * Dy, axis=1))
            # wall repulsion analytic
            dx_w = np.zeros(N); dy_w = np.zeros(N)
            # left
            mL = x < r
            dx_w[mL] += 0.9 * (r[mL] - x[mL])
            mR = x > 1.0 - r
            dx_w[mR] -= 0.9 * (x[mR] - (1.0 - r[mR]))
            mB = y < r
            dy_w[mB] += 0.9 * (r[mB] - y[mB])
            mT = y > 1.0 - r
            dy_w[mT] -= 0.9 * (y[mT] - (1.0 - r[mT]))
            # velocity-Verlet / momentum
            vx = mu * vx + dx_rep + dx_w
            vy = mu * vy + dy_rep + dy_w
            # decaying random kicks
            decay = 1.0 - float(t)/float(steps)
            ks = kick_sig * (0.3 + 0.7*decay)
            x = x + vx + rng.normal(0, ks, N)
            y = y + vy + rng.normal(0, ks, N)
            # collective events
            if (t % 350) == 349:
                ev = rng.integers(0, 4)
                if ev == 0:
                    # compression toward random corner/center
                    cc = corners[rng.integers(0, len(corners))]
                    f = 0.965 + 0.02*rng.random()
                    x = cc[0] + (x - cc[0])*f
                    y = cc[1] + (y - cc[1])*f
                elif ev == 1:
                    # radius-role permutation: swap two radii
                    a = rng.integers(0, N); b = rng.integers(0, N)
                    if a != b:
                        r[a], r[b] = r[b], r[a]
                    # also permute positions of two small circles to jump voids
                    if rng.random() < 0.5:
                        a2 = rng.integers(0, N); b2 = rng.integers(0, N)
                        x[a2], x[b2] = x[b2], x[a2]
                        y[a2], y[b2] = y[b2], y[a2]
                        vx[:] = 0; vy[:] = 0
                elif ev == 2:
                    # large kick to subset (tunneling through neighbours)
                    idx = rng.choice(N, size=max(1, N//6), replace=False)
                    x[idx] += rng.normal(0, 0.03, len(idx))
                    y[idx] += rng.normal(0, 0.03, len(idx))
                    vx[idx] = 0; vy[idx] = 0
                else:
                    # temporary growth advantage reset: shrink largest a bit
                    amax = int(np.argmax(r))
                    r[amax] *= 0.97
            # clip to box
            np.clip(x, 0.0, 1.0, out=x)
            np.clip(y, 0.0, 1.0, out=y)
            # growth: inversely related to pressure + small-circle advantage
            rmean = np.mean(r)
            boost = (rmean / (r + 1e-12)) ** alpha_small
            # cap boost
            np.clip(boost, 0.5, 2.0, out=boost)
            grow = (base_dr * boost) / (1.0 + beta * press)
            r = r + grow
            # keep radii bounded
            np.clip(r, 0.005, 0.26, out=r)
            # keep centres inside current radii (soft)
            # project if outside
            # (handled by wall repulsion, but hard clip to feasible)
            lo_x = r; hi_x = 1.0 - r
            # if r>0.5 infeasible, but r<=0.26 so ok
            np.clip(x, lo_x - 0.02, hi_x + 0.02, out=x)
            np.clip(y, lo_x - 0.02, hi_x + 0.02, out=y)  # same bound symmetric
            x = np.minimum(np.maximum(x, 0.001), 0.999)
            y = np.minimum(np.maximum(y, 0.001), 0.999)
        return x, y, r

    # grid points for jittered lattice inits
    gx6 = []; gy6 = []
    for ix in range(6):
        for jy in range(5):
            gx6.append((ix+0.5)/6.0); gy6.append((jy+0.5)/5.0)
    gx6 = np.array(gx6); gy6 = np.array(gy6)

    run = 0
    # LS phase budget ~68s
    while (time.time() - t0) < 68.0 and run < 260:
        typ = run % 5
        if typ == 0:
            x0 = rng.uniform(0.05, 0.95, N); y0 = rng.uniform(0.05, 0.95, N)
        elif typ == 1:
            perm = rng.permutation(len(gx6))[:N]
            x0 = gx6[perm] + rng.normal(0, 0.03, N)
            y0 = gy6[perm] + rng.normal(0, 0.03, N)
            x0 = np.clip(x0, 0.02, 0.98); y0 = np.clip(y0, 0.02, 0.98)
        elif typ == 2:
            # 5x5 + extra
            xs = []; ys = []
            for ix in range(5):
                for jy in range(5):
                    xs.append((ix+0.5)/5.0); ys.append((jy+0.5)/5.0)
            xs.append(0.5); ys.append(0.5)
            x0 = np.array(xs[:N], float) + rng.normal(0, 0.04, N)
            y0 = np.array(ys[:N], float) + rng.normal(0, 0.04, N)
            x0 = np.clip(x0, 0.02, 0.98); y0 = np.clip(y0, 0.02, 0.98)
        elif typ == 3 and len(pool) > 0:
            # elite restart
            bi = rng.integers(0, len(pool))
            _, bx, by = pool[bi]
            sig = 0.03 + 0.05*rng.random()
            x0 = bx + rng.normal(0, sig, N)
            y0 = by + rng.normal(0, sig, N)
            x0 = np.clip(x0, 0.01, 0.99); y0 = np.clip(y0, 0.01, 0.99)
        else:
            # hex-ish
            x0 = rng.uniform(0.03, 0.97, N); y0 = rng.uniform(0.03, 0.97, N)
            # corner compression init: push toward random corner initially
            ccx = rng.choice([0.0, 1.0]); ccy = rng.choice([0.0, 1.0])
            f0 = 0.85 + 0.1*rng.random()
            x0 = ccx + (x0-ccx)*f0; y0 = ccy + (y0-ccy)*f0
            x0 = np.clip(x0, 0.02, 0.98); y0 = np.clip(y0, 0.02, 0.98)
        steps = int(rng.integers(1100, 2200))
        base_dr = float(rng.uniform(5e-5, 1.15e-4))
        beta = float(rng.uniform(20.0, 90.0))
        k_rep = float(rng.uniform(0.65, 1.0))
        mu = float(rng.uniform(0.0, 0.55))
        kick_sig = float(rng.uniform(0.0008, 0.0032))
        alpha_small = float(rng.uniform(0.0, 0.45))
        seed_j = float(rng.random())
        try:
            xf, yf, rf = ls_grow(x0.copy(), y0.copy(), steps, base_dr, beta, k_rep, mu, kick_sig, alpha_small, seed_j)
        except Exception:
            run += 1
            continue
        # evaluate with LP (slightly clipped inside)
        xc = np.clip(xf, 0.02, 0.98); yc = np.clip(yf, 0.02, 0.98)
        s, r_opt = lp_eval(xc, yc)
        if r_opt is not None and s > 0:
            if s > best_sum:
                best_sum = s; best_x = xc.copy(); best_y = yc.copy(); best_r = r_opt.copy()
            # keep diverse pool (top 12)
            pool.append((s, xc.copy(), yc.copy()))
            pool = sorted(pool, key=lambda z: -z[0])[:12]
        run += 1
        if (time.time() - t0) > 68.0:
            break

    # ---------- penalized joint polishing from best LS states ----------
    # quick L-BFGS-B with wall-eliminating reparametrization
    if (time.time() - t0) < 95.0 and len(pool) > 0:
        bounds_box = [(0.0, 1.0)]*(2*N) + [(0.01, 0.30)]*N
        # take top 3
        cand = sorted(pool, key=lambda z: -z[0])[:3]
        for _, cx0, cy0 in cand:
            if (time.time() - t0) > 95.0:
                break
            # init u,v from centres with mid radii
            r_init = np.clip(best_r if best_r is not None else np.full(N, 0.08), 0.02, 0.2)
            # invert: x = r+(1-2r)u => u=(x-r)/(1-2r)
            denom = 1.0 - 2.0*r_init
            denom = np.maximum(denom, 1e-6)
            u0 = (np.clip(cx0,0.05,0.95) - r_init)/denom
            v0 = (np.clip(cy0,0.05,0.95) - r_init)/denom
            u0 = np.clip(u0, 0.0, 1.0); v0 = np.clip(v0, 0.0, 1.0)
            z = np.concatenate([u0, v0, r_init])
            for W in (200.0, 2000.0, 20000.0):
                if (time.time() - t0) > 95.0:
                    break
                def fun(zz, W=W):
                    u = zz[:N]; vv = zz[N:2*N]; rr = zz[2*N:]
                    x = rr*(1.0-2.0*u)+u
                    y = rr*(1.0-2.0*vv)+vv
                    Dx = x[:,None]-x[None,:]; Dy = y[:,None]-y[None,:]
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
                                   options={'maxiter': 180, 'ftol':1e-9, 'gtol':1e-5})
                    z = res.x
                except Exception:
                    break
            u = z[:N]; vv = z[N:2*N]; rr = z[2*N:]
            x = rr*(1.0-2.0*u)+u; y = rr*(1.0-2.0*vv)+vv
            x = np.clip(x, 0.02, 0.98); y = np.clip(y, 0.02, 0.98)
            s, r_opt = lp_eval(x, y)
            if r_opt is not None and s > best_sum:
                best_sum = s; best_x = x.copy(); best_y = y.copy(); best_r = r_opt.copy()
                pool.append((s, x.copy(), y.copy()))
                pool = sorted(pool, key=lambda z: -z[0])[:12]

    # ---------- hill climbing on centres with LP ----------
    if best_x is not None:
        sigmas = [0.02, 0.012, 0.006, 0.003, 0.0015]
        iters_list = [220, 220, 220, 180, 140]
        for sigma, iters in zip(sigmas, iters_list):
            for _ in range(iters):
                if (time.time() - t0) > DEADLINE - 1.5:
                    break
                # perturb single or all? mix: mostly all, sometimes single large jump
                if rng.random() < 0.15:
                    xn = best_x.copy(); yn = best_y.copy()
                    k = rng.integers(0, N)
                    xn[k] += sigma*2.0*rng.standard_normal()
                    yn[k] += sigma*2.0*rng.standard_normal()
                else:
                    xn = best_x + sigma*rng.standard_normal(N)
                    yn = best_y + sigma*rng.standard_normal(N)
                xn = np.clip(xn, 0.02, 0.98); yn = np.clip(yn, 0.02, 0.98)
                s, r_opt = lp_eval(xn, yn)
                if r_opt is not None and s > best_sum + 1e-12:
                    best_sum = s; best_x = xn; best_y = yn; best_r = r_opt
            if (time.time() - t0) > DEADLINE - 1.5:
                break

    out = np.column_stack([best_x, best_y, best_r])
    # strict safety scaling
    x = out[:,0]; y = out[:,1]; r = out[:,2]
    r = np.maximum(r, 1e-9)
    wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    Dx = x[:,None]-x[None,:]; Dy = y[:,None]-y[None,:]
    D = np.sqrt(Dx*Dx+Dy*Dy)
    np.fill_diagonal(D, np.inf)
    S = r[:,None]+r[None,:]
    with np.errstate(divide='ignore', invalid='ignore'):
        fw = float(np.min(wall/r))
        fp = float(np.min(D/S))
    f = min(fw, fp)
    if (not np.isfinite(f)) or f >= 1.0:
        shrink = (1.0-1e-7)
        # ensure margin 1e-9 absolute
        # if already feasible, just tiny shrink
        r = r*shrink
        # re-clip walls that might now be fine (shrinking helps)
        out = np.column_stack([x, y, r])
    else:
        r = r*f*(1.0-1e-9)
        out = np.column_stack([x, y, r])
    # final clip centres inside
    out[:,0] = np.clip(out[:,0], out[:,2], 1.0-out[:,2])
    out[:,1] = np.clip(out[:,1], out[:,2], 1.0-out[:,2])
    # ensure positivity
    out[:,2] = np.maximum(out[:,2], 1e-9)
    return out
