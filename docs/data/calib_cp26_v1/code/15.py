import numpy as np
import time

def solve():
    n = 26
    t0 = time.time()
    DEADLINE = 124.0

    rng = np.random.default_rng(0)

    # fallback
    k = 6
    r0 = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2*i+1)*r0, (2*j+1)*r0, r0*0.99])
    best_arr = np.array(pts[:n], dtype=float)
    def uniform_scale(X, Y, R):
        R = np.maximum(R, 1e-12)
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        wr = wall / R
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        S = R[:, None] + R[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = D / S
        np.fill_diagonal(ratio, np.inf)
        a = float(np.min(wr)); b = float(np.min(ratio))
        alpha = min(1.0, a, b)
        alpha = alpha * (1.0 - 1e-7) - 1e-12
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 0.5
        return R * alpha, alpha
    Xb, Yb, Rb = best_arr[:,0], best_arr[:,1], best_arr[:,2]
    Rb_s, _ = uniform_scale(Xb, Yb, Rb)
    best_arr = np.column_stack([Xb, Yb, Rb_s])
    best_sum = float(np.sum(Rb_s))
    best_X = Xb.copy(); best_Y = Yb.copy(); best_R = Rb_s.copy()

    from scipy.optimize import minimize, linprog

    # LP optimal radii for fixed centres
    def lp_opt(X, Y):
        wall = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        if np.min(wall) <= 1e-9:
            return None, -1.0
        Dx = X[:, None] - X[None, :]
        Dy = Y[:, None] - Y[None, :]
        D = np.sqrt(Dx*Dx + Dy*Dy)
        m = n*(n-1)//2
        A = np.zeros((m, n), dtype=float)
        b = np.zeros(m, dtype=float)
        kk = 0
        for i in range(n):
            for j in range(i+1, n):
                A[kk, i] = 1.0; A[kk, j] = 1.0
                b[kk] = D[i, j]
                kk += 1
        if np.min(b) <= 1e-9:
            return None, -1.0
        c = -np.ones(n)
        bnds = [(0.0, float(w)) for w in wall]
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method='highs')
        except Exception:
            return None, -1.0
        if not res.success:
            return None, -1.0
        Ropt = np.maximum(np.array(res.x, dtype=float), 0.0)
        # shrink for strict feasibility
        Ropt = Ropt * (1.0 - 1e-7) - 1e-13
        Ropt = np.maximum(Ropt, 1e-9)
        # verify quickly (walls & pairs) - re-shrink if needed
        wall2 = np.minimum(np.minimum(X, 1.0-X), np.minimum(Y, 1.0-Y))
        if np.any(Ropt > wall2):
            # scale down violating
            f = float(np.min(wall2 / np.maximum(Ropt,1e-12)))
            Ropt = Ropt * f * (1.0-1e-9)
        # pairwise check
        D2 = D
        S2 = Ropt[:,None]+Ropt[None,:]
        np.fill_diagonal(D2, 1.0)
        viol = S2 - D2
        np.fill_diagonal(viol, -1.0)
        mv = float(np.max(viol))
        if mv > -1e-12:
            # need shrink
            # factor from min ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                rat = D / np.maximum(S2,1e-12)
            np.fill_diagonal(rat, np.inf)
            f2 = float(np.min(rat))
            if f2 < 1.0:
                Ropt = Ropt * f2 * (1.0-1e-9)
        return Ropt, float(np.sum(Ropt))

    # joint penalty fun factory
    def make_fun(nn):
        def fun(v, W):
            X = v[0:nn]; Y = v[nn:2*nn]; R = v[2*nn:3*nn]
            gx = np.zeros(nn); gy = np.zeros(nn)
            pen = 0.0
            gr_pen = np.zeros(nn)
            v1 = R - X
            m1 = v1 > 0
            if np.any(m1):
                pen += np.sum(v1[m1]**2)
                gx[m1] -= 2.0*v1[m1]; gr_pen[m1] += 2.0*v1[m1]
            v2 = X + R - 1.0
            m2 = v2 > 0
            if np.any(m2):
                pen += np.sum(v2[m2]**2)
                gx[m2] += 2.0*v2[m2]; gr_pen[m2] += 2.0*v2[m2]
            v3 = R - Y
            m3 = v3 > 0
            if np.any(m3):
                pen += np.sum(v3[m3]**2)
                gy[m3] -= 2.0*v3[m3]; gr_pen[m3] += 2.0*v3[m3]
            v4 = Y + R - 1.0
            m4 = v4 > 0
            if np.any(m4):
                pen += np.sum(v4[m4]**2)
                gy[m4] += 2.0*v4[m4]; gr_pen[m4] += 2.0*v4[m4]
            Dx = X[:, None] - X[None, :]
            Dy = Y[:, None] - Y[None, :]
            D = np.sqrt(Dx*Dx + Dy*Dy + 1e-20)
            np.fill_diagonal(D, 1.0)
            S = R[:, None] + R[None, :]
            V = S - D
            np.fill_diagonal(V, 0.0)
            Vp = np.maximum(V, 0.0)
            pen += 0.5*np.sum(Vp*Vp)
            # grads pairwise
            Mask = Vp > 0
            if np.any(Mask):
                Inv = np.zeros_like(D)
                Inv[Mask] = 1.0 / D[Mask]
                Wm = 2.0 * Vp * Inv  # 2V/D
                # gx += sum_j Wm_ij * (-Dx_ij)
                gx += np.sum(Wm * (-Dx), axis=1)
                gy += np.sum(Wm * (-Dy), axis=1)
                gr_pen += np.sum(2.0*Vp, axis=1)
            loss = -np.sum(R) + W*pen
            g = np.empty_like(v)
            g[0:nn] = W*gx
            g[nn:2*nn] = W*gy
            g[2*nn:3*nn] = -1.0 + W*gr_pen
            return loss, g
        return fun
    fun = make_fun(n)
    bounds = [(0.0,1.0)]*(2*n) + [(0.02,0.20)]*n
    Ws = [3e2, 3e3, 3e4, 2e5]

    # Phase A: many restarts joint opt + LP
    restart = 0
    # time for phase A ~55s
    PHASE_A_END = t0 + 55.0
    # keep pool of top centres
    pool = []  # list of (sum, X, Y)
    while time.time() - t0 < 55.0 and restart < 60:
        if restart == 0:
            X0 = np.zeros(n); Y0 = np.zeros(n)
            R0 = np.full(n, 0.095)
            idx = 0
            for i in range(5):
                for j in range(5):
                    X0[idx] = (i+0.5)/5.0 + rng.normal(0,0.006)
                    Y0[idx] = (j+0.5)/5.0 + rng.normal(0,0.006)
                    idx += 1
            X0[25] = 0.9 + rng.normal(0,0.01); Y0[25] = 0.1 + rng.normal(0,0.01)
            X0 = np.clip(X0,0.05,0.95); Y0=np.clip(Y0,0.05,0.95)
            R0 = 0.09+0.01*rng.random(n)
        elif restart == 1:
            X0=np.zeros(n); Y0=np.zeros(n)
            pts_l=[]
            for ix in range(6):
                for iy in range(5):
                    if len(pts_l)<n:
                        xx=(ix+0.5)/6.0; yy=(iy+0.5)/5.0
                        if iy%2==1:
                            xx+=0.04
                            if xx>0.95: xx-=0.16
                        pts_l.append((xx,yy))
            arr=np.array(pts_l)
            X0=arr[:,0]+rng.normal(0,0.006,n); Y0=arr[:,1]+rng.normal(0,0.006,n)
            X0=np.clip(X0,0.05,0.95); Y0=np.clip(Y0,0.05,0.95)
            R0=0.085+0.02*rng.random(n)
        else:
            # random or jitter of best
            if restart % 4 == 3 and best_X is not None:
                X0 = best_X + rng.normal(0,0.03,n)
                Y0 = best_Y + rng.normal(0,0.03,n)
                X0=np.clip(X0,0.04,0.96); Y0=np.clip(Y0,0.04,0.96)
                R0 = best_R + rng.normal(0,0.008,n)
                R0=np.clip(R0,0.03,0.18)
            else:
                X0 = rng.random(n)*0.9+0.05
                Y0 = rng.random(n)*0.9+0.05
                R0 = 0.07+0.05*rng.random(n)
        v = np.concatenate([X0,Y0,R0])
        for W in Ws:
            if time.time()-t0 > PHASE_A_END:
                break
            try:
                res = minimize(fun, v, args=(W,), jac=True, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter':220,'ftol':1e-9,'gtol':1e-6})
                v=res.x
            except Exception:
                break
        X=np.clip(v[0:n],0.0,1.0); Y=np.clip(v[n:2*n],0.0,1.0); R=np.clip(v[2*n:3*n],0.01,0.25)
        try:
            Ropt,s = lp_opt(X,Y)
            if Ropt is not None and s>best_sum:
                best_sum=s; best_X=X.copy(); best_Y=Y.copy(); best_R=Ropt.copy()
                best_arr=np.column_stack([X,Y,Ropt])
            if Ropt is not None:
                pool.append((s,X.copy(),Y.copy()))
        except Exception:
            pass
        # also uniform scale candidate (in case LP fails)
        try:
            Rs,_=uniform_scale(X,Y,R)
            su=float(np.sum(Rs))
            if su>best_sum:
                best_sum=su; best_X=X.copy(); best_Y=Y.copy(); best_R=Rs.copy()
                best_arr=np.column_stack([X,Y,Rs])
        except Exception:
            pass
        restart+=1

    # sort pool keep top
    try:
        pool.sort(key=lambda t: -t[0])
        pool=pool[:5]
    except Exception:
        pool=[]

    # Phase B: hill climbing on centres with LP evaluation
    # start from best
    cur_X = best_X.copy(); cur_Y = best_Y.copy(); cur_sum = best_sum
    # if pool has diversity, occasionally restart from pool members
    n_eval=0
    # sigma schedule
    it=0
    # precompute time end
    while time.time()-t0 < DEADLINE-4.0:
        it+=1
        # choose sigma decaying + cyclic
        # cycle through scales
        cyc = it % 10
        if cyc < 6:
            sigma = 0.012 * (0.995**(it//2)) + 0.001
            k_move = 1 if (rng.random()<0.6) else 2
        elif cyc < 8:
            sigma = 0.03
            k_move = 3
        else:
            sigma = 0.005
            k_move = n  # global shake small
            sigma = 0.004
        if sigma < 0.0012:
            sigma = 0.0012
        # propose
        Xn = cur_X.copy(); Yn = cur_Y.copy()
        if k_move == n:
            Xn = Xn + rng.normal(0,sigma,n)
            Yn = Yn + rng.normal(0,sigma,n)
        else:
            idxs = rng.choice(n, size=k_move, replace=False)
            Xn[idxs] += rng.normal(0,sigma,k_move)
            Yn[idxs] += rng.normal(0,sigma,k_move)
        Xn=np.clip(Xn,0.01,0.99); Yn=np.clip(Yn,0.01,0.99)
        Ropt,s = lp_opt(Xn,Yn)
        n_eval+=1
        if Ropt is not None and s > cur_sum + 1e-12:
            cur_X=Xn; cur_Y=Yn; cur_sum=s
            if s>best_sum:
                best_sum=s; best_X=cur_X.copy(); best_Y=cur_Y.copy(); best_R=Ropt.copy()
                best_arr=np.column_stack([cur_X,cur_Y,Ropt])
        # occasional large perturbation restart from best if stuck
        if it % 400 == 0:
            # try jump from best with bigger noise or from pool
            if len(pool)>0 and (rng.random()<0.4):
                p = pool[int(rng.integers(0,len(pool)))]
                cur_X = p[1]+rng.normal(0,0.015,n); cur_Y=p[2]+rng.normal(0,0.015,n)
                cur_X=np.clip(cur_X,0.01,0.99); cur_Y=np.clip(cur_Y,0.01,0.99)
                Ropt2,s2=lp_opt(cur_X,cur_Y)
                if Ropt2 is not None:
                    cur_sum=s2
                    if s2>best_sum:
                        best_sum=s2; best_X=cur_X.copy(); best_Y=cur_Y.copy(); best_R=Ropt2.copy()
                        best_arr=np.column_stack([cur_X,cur_Y,Ropt2])
            else:
                # shake current best strongly then continue from best if worse
                if cur_sum < best_sum - 0.02:
                    cur_X=best_X.copy(); cur_Y=best_Y.copy(); cur_sum=best_sum
        if it % 200 == 0 and time.time()-t0 > DEADLINE-15:
            # intensify: small sigma only
            pass

    # final safety shrink
    try:
        Xf=best_arr[:,0]; Yf=best_arr[:,1]; Rf=best_arr[:,2]
        Rf_s,_=uniform_scale(Xf,Yf,Rf)
        best_arr=np.column_stack([Xf,Yf,Rf_s])
    except Exception:
        pass
    return best_arr
