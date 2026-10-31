import numpy as np
import time as _time

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize, linprog
    n = 26
    t0 = _tm.time()
    HARD_STOP = 115.0
    FINAL_DEADLINE = 123.0
    iu, ju = _np.triu_indices(n, 1)
    m = len(iu)
    A0 = _np.zeros((m, n), dtype=float)
    _ar = _np.arange(m)
    A0[_ar, iu] = 1.0
    A0[_ar, ju] = 1.0
    c_obj = -_np.ones(n, dtype=float)
    # dual constraint matrix fixed: [-A0.T, -I] (n x (m+n))
    A_dual = _np.hstack([-A0.T, -_np.eye(n, dtype=float)])
    b_dual = -_np.ones(n, dtype=float)
    dual_bounds = [(0.0, None)] * (m + n)

    def eval_primal(xc, yc):
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            dvec = _np.maximum(dvec, 1e-12)
            bounds = [(0.0, float(bi)) for bi in b]
            res = linprog(c_obj, A_ub=A0, b_ub=dvec, bounds=bounds, method='highs')
            if res.status != 0 or res.x is None:
                return None, -1.0, None
            r = _np.asarray(res.x, dtype=float).flatten()
            r = _np.maximum(r, 0.0)
            return r, float(_np.sum(r)), res
        except Exception:
            return None, -1.0, None

    def extract_marginals(res):
        # returns pair_m (m,), upper_m (n,) or (None,None)
        try:
            pair_m = None
            upper_m = None
            # ineqlin
            ine = None
            try:
                ine = getattr(res, 'ineqlin', None)
            except Exception:
                ine = None
            if ine is None:
                try:
                    ine = res['ineqlin']
                except Exception:
                    ine = None
            if ine is not None:
                mg = None
                try:
                    mg = getattr(ine, 'marginals', None)
                except Exception:
                    mg = None
                if mg is None:
                    try:
                        mg = ine['marginals']
                    except Exception:
                        mg = None
                if mg is not None:
                    pair_m = _np.asarray(mg, dtype=float).flatten()
            # upper
            up = None
            try:
                up = getattr(res, 'upper', None)
            except Exception:
                up = None
            if up is None:
                try:
                    up = res['upper']
                except Exception:
                    up = None
            # some versions use 'bounds' residual; fallback: try 'upper'
            if up is not None:
                mg2 = None
                try:
                    mg2 = getattr(up, 'marginals', None)
                except Exception:
                    mg2 = None
                if mg2 is None:
                    try:
                        mg2 = up['marginals']
                    except Exception:
                        mg2 = None
                if mg2 is not None:
                    upper_m = _np.asarray(mg2, dtype=float).flatten()
            return pair_m, upper_m
        except Exception:
            return None, None

    def eval_dual(xc, yc):
        try:
            b = _np.minimum(_np.minimum(xc, 1.0 - xc), _np.minimum(yc, 1.0 - yc))
            b = _np.maximum(b, 0.0)
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx * dx + dy * dy)
            dvec = _np.maximum(dvec, 1e-12)
            c_dual = _np.concatenate([dvec, b])
            res = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=dual_bounds, method='highs')
            if res.status != 0 or res.x is None:
                return None, None, -1.0
            y = _np.asarray(res.x, dtype=float).flatten()
            lam = _np.maximum(y[:m], 0.0)
            mu = _np.maximum(y[m:], 0.0)
            S = float(_np.sum(c_dual * y))  # dual objective = primal optimum
            # sanity: S should be >=0 and not huge
            if not _np.isfinite(S) or S < 0 or S > 10:
                return None, None, -1.0
            return lam, mu, S
        except Exception:
            return None, None, -1.0

    def grad_from_duals(xc, yc, lam, mu):
        gx = _np.zeros(n)
        gy = _np.zeros(n)
        # pair part: lam*(ci-cj)/d
        if lam is not None:
            dx = xc[iu] - xc[ju]
            dy = yc[iu] - yc[ju]
            d = _np.sqrt(dx * dx + dy * dy)
            # avoid div0
            nz = d > 1e-12
            ux = _np.zeros(m); uy = _np.zeros(m)
            ux[nz] = dx[nz] / d[nz]
            uy[nz] = dy[nz] / d[nz]
            # only positive duals
            lamc = _np.maximum(lam, 0.0)
            # cap huge duals (degenerate)
            lamc = _np.minimum(lamc, 5.0)
            contribx = lamc * ux
            contributy = lamc * uy
            # accumulate: gx[i]+= , gx[j]-=
            _np.add.at(gx, iu, contribx)
            _np.add.at(gx, ju, -contribx)
            _np.add.at(gy, iu, contributy)
            _np.add.at(gy, ju, -contributy)
        # wall part: only nearest wall direction
        if mu is not None:
            muc = _np.maximum(mu, 0.0)
            muc = _np.minimum(muc, 5.0)
            for i in range(n):
                if muc[i] <= 1e-12:
                    continue
                xl = xc[i]; xr = 1.0 - xc[i]; yb = yc[i]; yt = 1.0 - yc[i]
                # nearest
                vals = (xl, xr, yb, yt)
                k = int(_np.argmin(vals))
                if k == 0:
                    gx[i] += muc[i] * 1.0
                elif k == 1:
                    gx[i] += muc[i] * (-1.0)
                elif k == 2:
                    gy[i] += muc[i] * 1.0
                else:
                    gy[i] += muc[i] * (-1.0)
        return gx, gy

    # deterministic jitter offsets for stabilization (4 x n x 2)
    _jrng = _np.random.RandomState(0)
    JITTER = 1e-5
    joffs = _jrng.randn(4, n, 2) * JITTER

    # ---------- sign-convention check state ----------
    sign_pair = -1.0  # default guess: marginals <=0 for minimize
    sign_wall = -1.0
    use_marginals_ok = False

    def calibrate_signs(xc, yc):
        nonlocal sign_pair, sign_wall, use_marginals_ok
        try:
            r0, S0, res = eval_primal(xc, yc)
            if r0 is None or res is None:
                return
            pm, um = extract_marginals(res)
            if pm is None or um is None:
                return
            # find active pair with |m|>1e-9
            dx = xc[iu] - xc[ju]; dy = yc[iu] - yc[ju]
            dvec = _np.sqrt(dx*dx+dy*dy)
            slack = dvec - (r0[iu] + r0[ju])
            order = _np.argsort(slack)
            found = False
            for kk in order[:40]:
                if abs(pm[kk]) < 1e-9:
                    continue
                i = int(iu[kk]); j = int(ju[kk])
                d = dvec[kk]
                if d < 1e-9:
                    continue
                ux = (xc[i]-xc[j])/d; uy = (yc[i]-yc[j])/d
                eps = 1e-6
                xc2 = xc.copy(); yc2 = yc.copy()
                xc2[i] += ux*eps*0.5; yc2[i] += uy*eps*0.5
                xc2[j] -= ux*eps*0.5; yc2[j] -= uy*eps*0.5
                _, S1, _ = eval_primal(xc2, yc2)
                if S1 < 0:
                    continue
                # actual distance increase
                dx2 = xc2[iu]-xc2[ju]; dy2 = yc2[iu]-yc2[ju]
                dvec2 = _np.sqrt(dx2*dx2+dy2*dy2)
                dd = dvec2[kk]-dvec[kk]
                if abs(dd) < 1e-12:
                    continue
                fd = (S1-S0)/dd
                # fd should be >=0 (~lam)
                if abs(pm[kk]) < 1e-12:
                    continue
                # decide sign so sign*pm ~= fd (fd>=0)
                if pm[kk]*fd < 0:
                    sign_pair = -1.0
                else:
                    sign_pair = 1.0
                # validate magnitude roughly? allow degenerate mismatch
                found = True
                break
            # wall test
            b = _np.minimum(_np.minimum(xc,1.0-xc),_np.minimum(yc,1.0-yc))
            uslack = b - r0  # >=0, small means wall-active
            # upper marginals: r_i <= b_i active when uslack small
            worder = _np.argsort(uslack)
            for ii in worder[:10]:
                if abs(um[ii]) < 1e-9:
                    continue
                # nearest wall direction
                xl=xc[ii]; xr=1.0-xc[ii]; yb=yc[ii]; yt=1.0-yc[ii]
                vals=(xl,xr,yb,yt); k=int(_np.argmin(vals))
                eps=1e-6
                xc2=xc.copy(); yc2=yc.copy()
                if k==0: xc2[ii]+=eps
                elif k==1: xc2[ii]-=eps
                elif k==2: yc2[ii]+=eps
                else: yc2[ii]-=eps
                _,S1,_=eval_primal(xc2,yc2)
                if S1<0: continue
                fd=(S1-S0)/eps
                if um[ii]*fd < 0:
                    sign_wall=-1.0
                else:
                    sign_wall=1.0
                break
            # check nonnegativity after sign
            lam_test = sign_pair*pm
            # at least most should be >= -1e-7
            if _np.mean(lam_test > -1e-7) > 0.9:
                use_marginals_ok = True
            else:
                # try flipped?
                if _np.mean(-lam_test > -1e-7) > 0.9:
                    sign_pair = -sign_pair
                    use_marginals_ok = True
        except Exception:
            pass

    def single_grad(xc, yc):
        # try marginals first, else dual solve; returns (gx,gy,S) or raises
        r, S, res = eval_primal(xc, yc)
        if r is None:
            raise RuntimeError("primal fail")
        if use_marginals_ok and res is not None:
            try:
                pm, um = extract_marginals(res)
                if pm is not None and um is not None and len(pm)==m and len(um)==n:
                    lam = _np.maximum(sign_pair*pm, 0.0)
                    mu = _np.maximum(sign_wall*um, 0.0)
                    # quick FD sanity: if lam huge, cap; if all zero but S large, fallback to dual
                    if _np.sum(lam)+_np.sum(mu) > 1e-9:
                        gx, gy = grad_from_duals(xc, yc, lam, mu)
                        return gx, gy, S
            except Exception:
                pass
        # dual-solve path (exact, nonnegative)
        lam, mu, Sd = eval_dual(xc, yc)
        if lam is None:
            # last resort: use primal S with zero grad? raise
            raise RuntimeError("dual fail")
        gx, gy = grad_from_duals(xc, yc, lam, mu)
        # prefer primal S (more accurate) if close, else dual S
        if abs(Sd - S) > 1e-6:
            S = Sd
        return gx, gy, S

    def stabilized_grad(xc, yc):
        # average over exact + 4 jittered; returns (gxm, gym, S_exact)
        gx0, gy0, S0 = single_grad(xc, yc)
        gxs = [gx0]; gys = [gy0]
        for k in range(4):
            try:
                xck = _np.clip(xc + joffs[k,:,0], 0.02, 0.98)
                yck = _np.clip(yc + joffs[k,:,1], 0.02, 0.98)
                gxk, gyk, _ = single_grad(xck, yck)
                # sanity finite
                if _np.all(_np.isfinite(gxk)) and _np.all(_np.isfinite(gyk)):
                    gxs.append(gxk); gys.append(gyk)
            except Exception:
                continue
        gxm = _np.mean(gxs, axis=0)
        gym = _np.mean(gys, axis=0)
        # clip extreme
        gxm = _np.clip(gxm, -10.0, 10.0)
        gym = _np.clip(gym, -10.0, 10.0)
        return gxm, gym, S0

    # ---------- penalty incumbent phase ----------
    bnds78 = [(0.0,1.0)]*(2*n)+[(0.015,0.24)]*n
    best_xc=None; best_yc=None; best_r=None; best_sum=-1.0
    def try_update_primal(xc, yc):
        nonlocal best_sum, best_xc, best_yc, best_r
        try:
            r,S,_ = eval_primal(_np.clip(xc,0.0,1.0), _np.clip(yc,0.0,1.0))
            if r is not None and S > best_sum:
                best_sum=float(S); best_xc=xc.copy(); best_yc=yc.copy(); best_r=r.copy()
                return True
        except Exception:
            pass
        return False

    rng = _np.random.RandomState(42)
    # quick diverse penalty restarts, time-boxed ~32s
    P_seq = [50.0, 1500.0, 25000.0]
    restart = 0
    while _tm.time()-t0 < 32.0 and restart < 40:
        if restart==0:
            pts=[]
            for i in range(5):
                for j in range(5):
                    pts.append([0.1+0.2*i,0.1+0.2*j])
            pts.append([0.5,0.5])
            xy=_np.array(pts)
            x0=xy[:,0]+rng.randn(n)*0.002; y0=xy[:,1]+rng.randn(n)*0.002
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
            r0=_np.full(n,0.085)
        elif restart==1:
            xs=(_np.arange(6)+0.5)/6.0; ys=(_np.arange(5)+0.5)/5.0
            xx,yy=_np.meshgrid(xs,ys)
            xy=_np.column_stack([xx.ravel(),yy.ravel()])
            perm=rng.permutation(len(xy))[:n]
            x0=xy[perm,0]+rng.randn(n)*0.004; y0=xy[perm,1]+rng.randn(n)*0.004
            x0=_np.clip(x0,0.05,0.95); y0=_np.clip(y0,0.05,0.95)
            r0=rng.uniform(0.06,0.09,size=n)
        else:
            if best_xc is not None and restart%3==0:
                x0=_np.clip(best_xc+rng.randn(n)*0.02,0.02,0.98)
                y0=_np.clip(best_yc+rng.randn(n)*0.02,0.02,0.98)
                r0=_np.clip(_np.maximum(best_r,0.02)+rng.randn(n)*0.005,0.02,0.2)
            else:
                x0=rng.uniform(0.03,0.97,size=n); y0=rng.uniform(0.03,0.97,size=n)
                r0=rng.uniform(0.05,0.10,size=n)
        z=_np.concatenate([x0,y0,r0])
        for P in P_seq:
            if _tm.time()-t0 > 32.0: break
            def fg(zz, P=P):
                x=zz[0:n]; y=zz[n:2*n]; r=zz[2*n:3*n]
                f=-_np.sum(r)
                gx=_np.zeros(n); gy=_np.zeros(n); gr=-_np.ones(n)
                v1=r-x; mm=v1>0
                if _np.any(mm):
                    f+=P*_np.sum(v1[mm]*v1[mm]); co=2.0*P*v1[mm]
                    gx[mm]-=co; gr[mm]+=co
                v2=x+r-1.0; mm=v2>0
                if _np.any(mm):
                    f+=P*_np.sum(v2[mm]*v2[mm]); co=2.0*P*v2[mm]
                    gx[mm]+=co; gr[mm]+=co
                v3=r-y; mm=v3>0
                if _np.any(mm):
                    f+=P*_np.sum(v3[mm]*v3[mm]); co=2.0*P*v3[mm]
                    gy[mm]-=co; gr[mm]+=co
                v4=y+r-1.0; mm=v4>0
                if _np.any(mm):
                    f+=P*_np.sum(v4[mm]*v4[mm]); co=2.0*P*v4[mm]
                    gy[mm]+=co; gr[mm]+=co
                dxm=x[:,None]-x[None,:]; dym=y[:,None]-y[None,:]
                D=_np.sqrt(dxm*dxm+dym*dym)
                Rsum=r[:,None]+r[None,:]
                Smat=Rsum-D; _np.fill_diagonal(Smat,-1.0)
                mask=Smat>0
                if _np.any(mask):
                    f+=P*0.5*_np.sum(Smat[mask]*Smat[mask])
                    W=_np.zeros((n,n)); W[mask]=(2.0*P*Smat[mask])/(D[mask]+1e-12)
                    rs=W.sum(axis=1)
                    gx+=W.dot(x)-rs*x; gy+=W.dot(y)-rs*y
                    Spos=_np.where(mask,Smat,0.0); gr+=2.0*P*Spos.sum(axis=1)
                g=_np.concatenate([gx,gy,gr])
                return f,g
            try:
                res=minimize(fg,z,method='L-BFGS-B',jac=True,bounds=bnds78,options={'maxiter':120,'ftol':1e-10,'gtol':1e-6})
                z=_np.asarray(res.x,dtype=float)
            except Exception:
                break
        try:
            xc=_np.clip(z[0:n],0.0,1.0); yc=_np.clip(z[n:2*n],0.0,1.0)
            try_update_primal(xc,yc)
        except Exception:
            pass
        restart+=1

    if best_xc is None:
        k=int(_np.ceil(_np.sqrt(n))); rr=0.5/k; pts=[]
        for i in range(k):
            for j in range(k):
                if len(pts)<n: pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
        return _np.array(pts,dtype=float)

    # calibrate marginal signs on incumbent
    try:
        calibrate_signs(best_xc, best_yc)
    except Exception:
        pass

    # ---------- centre-only dual trust-region phase ----------
    TR = 0.02
    cur_x = best_xc.copy(); cur_y = best_yc.copy()
    # outer loop
    outer_rng = _np.random.RandomState(1234)
    while _tm.time()-t0 < HARD_STOP:
        # trust box around current best (cap 0.02 per accepted move)
        # use best as centre of trust region
        bx = best_xc.copy(); by = best_yc.copy()
        lo_x = _np.maximum(0.02, bx - TR); hi_x = _np.minimum(0.98, bx + TR)
        lo_y = _np.maximum(0.02, by - TR); hi_y = _np.minimum(0.98, by + TR)
        trust_bounds = [(float(lo_x[i]), float(hi_x[i])) for i in range(n)] + [(float(lo_y[i]), float(hi_y[i])) for i in range(n)]
        # start point: best, occasionally perturbed to escape
        if outer_rng.rand() < 0.25:
            sx = _np.clip(bx + outer_rng.randn(n)*0.008, lo_x, hi_x)
            sy = _np.clip(by + outer_rng.randn(n)*0.008, lo_y, hi_y)
        else:
            sx = bx.copy(); sy = by.copy()
        z0 = _np.concatenate([sx, sy])

        # closure capturing best
        def fg_dual(z):
            # fresh LP verification inside
            xc = z[:n]; yc = z[n:]
            try:
                gxm, gym, S = stabilized_grad(xc, yc)
            except Exception:
                # fallback: fresh primal only, zero grad -> stall gracefully
                try:
                    _, S2, _ = eval_primal(xc, yc)
                    if S2 < 0:
                        return 1e6, _np.zeros(2*n)
                    f = -S2
                    # track best even on failure path
                    if S2 > best_sum:
                        try:
                            r2, _, _ = eval_primal(xc, yc)
                            if r2 is not None:
                                # strictly feasible check via shrink? just track
                                pass
                        except Exception:
                            pass
                    return f, _np.zeros(2*n)
                except Exception:
                    return 1e6, _np.zeros(2*n)
            f = -S
            # track best strictly (raw, shrink at end)
            nonlocal_best = False
            try:
                if S > best_sum:
                    rnew, Snew, _ = eval_primal(xc, yc)
                    if rnew is not None and Snew > best_sum:
                        # update globals
                        globals()['__dummy__'] = 0  # no-op
                        # use closure variables via mutable
                        pass
            except Exception:
                pass
            g = _np.concatenate([-gxm, -gym])
            # replace nonfinite
            g = _np.where(_np.isfinite(g), g, 0.0)
            if not _np.isfinite(f):
                return 1e6, _np.zeros(2*n)
            return f, g

        # we need try_update inside fg_dual with access to best_*; implement via inner func with nonlocal
        # redefine with nonlocal capture correctly:
        def fg2(z):
            xc=z[:n]; yc=z[n:]
            try:
                gxm,gym,S = stabilized_grad(xc,yc)
            except Exception:
                return 1e6, _np.zeros(2*n)
            f=-S
            if _np.isfinite(S) and S > best_sum_holder[0]:
                try:
                    rnew, Snew, _ = eval_primal(xc,yc)
                    if rnew is not None and Snew > best_sum_holder[0]:
                        best_sum_holder[0]=float(Snew)
                        best_sum_holder[1]=xc.copy()
                        best_sum_holder[2]=yc.copy()
                        best_sum_holder[3]=rnew.copy()
                except Exception:
                    pass
            g=_np.concatenate([-gxm,-gym])
            g=_np.where(_np.isfinite(g),g,0.0)
            if not _np.isfinite(f):
                return 1e6, _np.zeros(2*n)
            return f,g

        best_sum_holder = [best_sum, best_xc, best_yc, best_r]
        try:
            res = minimize(fg2, z0, method='L-BFGS-B', jac=True, bounds=trust_bounds,
                           options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-7})
        except Exception:
            # fallback: keep penalty incumbent, small random hill step
            if _tm.time()-t0 > HARD_STOP-2:
                break
            # tiny random step verified by fresh LP
            try:
                nx=_np.clip(best_sum_holder[1]+outer_rng.randn(n)*0.003,0.02,0.98)
                ny=_np.clip(best_sum_holder[2]+outer_rng.randn(n)*0.003,0.02,0.98)
                r,S,_=eval_primal(nx,ny)
                if r is not None and S>best_sum_holder[0]:
                    best_sum_holder[0]=float(S); best_sum_holder[1]=nx; best_sum_holder[2]=ny; best_sum_holder[3]=r
            except Exception:
                pass
            best_sum, best_xc, best_yc, best_r = best_sum_holder[0], best_sum_holder[1], best_sum_holder[2], best_sum_holder[3]
            cur_x, cur_y = best_xc.copy(), best_yc.copy()
            continue
        # sync best
        best_sum, best_xc, best_yc, best_r = best_sum_holder[0], best_sum_holder[1], best_sum_holder[2], best_sum_holder[3]
        # move cur to result if improved else small jitter around best
        try:
            zf = _np.asarray(res.x, dtype=float)
            # verify result centre by fresh LP (already done inside, but ensure)
            rF, SF, _ = eval_primal(zf[:n], zf[n:])
            if rF is not None and SF >= best_sum - 1e-12:
                cur_x = zf[:n].copy(); cur_y = zf[n:].copy()
                if SF > best_sum:
                    best_sum=float(SF); best_xc=cur_x.copy(); best_yc=cur_y.copy(); best_r=rF.copy()
            else:
                cur_x, cur_y = best_xc.copy(), best_yc.copy()
        except Exception:
            cur_x, cur_y = best_xc.copy(), best_yc.copy()
        # occasional LP hill-climb single move to complement gradients (cheap, topology change)
        if _tm.time()-t0 < HARD_STOP-3:
            try:
                for _ in range(8):
                    if _tm.time()-t0 >= HARD_STOP: break
                    idx=int(outer_rng.randint(n))
                    sig=0.004
                    nx=best_xc.copy(); ny=best_yc.copy()
                    nx[idx]+=float(outer_rng.randn())*sig; ny[idx]+=float(outer_rng.randn())*sig
                    nx[idx]=float(_np.clip(nx[idx],0.02,0.98)); ny[idx]=float(_np.clip(ny[idx],0.02,0.98))
                    r,S,_=eval_primal(nx,ny)
                    if r is not None and S>best_sum:
                        best_sum=float(S); best_xc=nx; best_yc=ny; best_r=r
            except Exception:
                pass
        if _tm.time()-t0 >= HARD_STOP:
            break

    # ---------- final strict repair ----------
    try:
        r_opt, _, _ = eval_primal(best_xc, best_yc)
        if r_opt is not None:
            best_r = r_opt
    except Exception:
        pass
    # shrink for strict feasibility
    try:
        shrink = (1.0 - 1e-7)
        r_safe = _np.maximum(best_r, 0.0)*shrink
        # ensure centres respect walls with 1e-9 margin
        xc = best_xc.copy(); yc = best_yc.copy()
        for i in range(n):
            ri = float(r_safe[i])
            if xc[i] < ri+1e-9: xc[i]=ri+1e-9
            if xc[i] > 1.0-ri-1e-9: xc[i]=1.0-ri-1e-9
            if yc[i] < ri+1e-9: yc[i]=ri+1e-9
            if yc[i] > 1.0-ri-1e-9: yc[i]=1.0-ri-1e-9
            xc[i]=float(_np.clip(xc[i],0.02,0.98)); yc[i]=float(_np.clip(yc[i],0.02,0.98))
        # rescale for pair feasibility with eps margin
        eps=1e-9
        b=_np.minimum(_np.minimum(xc,1.0-xc),_np.minimum(yc,1.0-yc))
        t=1.0
        for i in range(n):
            if r_safe[i]>1e-12:
                ti=(b[i]-eps)/r_safe[i]
                if ti<t: t=ti
        dx=xc[iu]-xc[ju]; dy=yc[iu]-yc[ju]
        dvec=_np.sqrt(dx*dx+dy*dy)
        rs=r_safe[iu]+r_safe[ju]
        mask=rs>1e-12
        if _np.any(mask):
            tp=_np.min((dvec[mask]-eps)/rs[mask])
            if tp<t: t=tp
        if not _np.isfinite(t) or t<=0:
            r_safe=_np.full(n,1e-4)
        else:
            if t>1.0: t=1.0
            t=t*(1.0-1e-9)
            r_safe=r_safe*t
        r_safe=_np.maximum(r_safe,1e-9)
        out=_np.column_stack([xc,yc,r_safe])
        # final safety: if anything violates, shrink more
        # quick check
        ok=True
        for i in range(n):
            if not (r_safe[i] <= xc[i]+1e-12 and r_safe[i] <= 1.0-xc[i]+1e-12 and r_safe[i] <= yc[i]+1e-12 and r_safe[i] <= 1.0-yc[i]+1e-12):
                ok=False
        if not ok:
            r_safe=r_safe*0.999
            out=_np.column_stack([xc,yc,r_safe])
        return out
    except Exception:
        # fallback to penalty incumbent with shrink
        try:
            r_safe=_np.maximum(best_r,1e-9)*(1.0-1e-7)*0.9999
            return _np.column_stack([best_xc,best_yc,r_safe])
        except Exception:
            k=int(_np.ceil(_np.sqrt(n))); rr=0.5/k; pts=[]
            for i in range(k):
                for j in range(k):
                    if len(pts)<n: pts.append([(2*i+1)*rr,(2*j+1)*rr,rr*0.999])
            return _np.array(pts,dtype=float)
