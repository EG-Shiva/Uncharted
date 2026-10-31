import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normr(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _smooth_asym(X0, steps, b0, b1, s0, s1, mu=0.85, rng=None):
    X = X0.copy()
    n, d = X.shape
    vel = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ls0, ls1 = np.log(s0), np.log(s1)
    for it in range(steps):
        f = it / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        smax = float(np.exp(ls0 + (ls1 - ls0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        gt = GX - dot * X
        gn = float(np.linalg.norm(gt, axis=1).max())
        if gn > 1e-12:
            gt = gt / gn
        vel = mu * vel + (1.0 - mu) * gt
        vn = float(np.linalg.norm(vel, axis=1).max())
        if vn > 1e-14:
            X = X - (smax / vn) * vel
            X = _normr(X)
    return X

def _q8_mats():
    qs = [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
          (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]
    Ms = []
    for (a1,b1,c1,d1) in qs:
        M = np.array([
            [a1,-b1,-c1,-d1],
            [b1,a1,-d1,c1],
            [c1,d1,a1,-b1],
            [d1,-c1,b1,a1]], dtype=np.float64)
        Ms.append(M)
    return np.stack(Ms)  # (8,4,4)

_Q8 = _q8_mats()
_Q8T = np.transpose(_Q8, (0,2,1))

def _expand_anti(Y):
    return np.concatenate([Y, -Y], axis=0)

def _expand_q8(Y):
    # Y (34,4) -> (272,4)
    ns = Y.shape[0]
    X = np.empty((ns*8,4), dtype=np.float64)
    for e in range(8):
        # X[s*8+e] = M_e @ Y_s
        X[e::8] = ( _Q8[e] @ Y.T ).T if False else Y @ _Q8[e].T
        # careful ordering: want s-major; do loop
    # redo s-major correctly
    for s in range(ns):
        for e in range(8):
            X[s*8+e] = _Q8[e] @ Y[s]
    return X

def _smooth_anti(Y0, steps, b0, b1, s0, s1, mu=0.85):
    Y = _normr(Y0.copy())
    vel = np.zeros_like(Y)
    lb0, lb1 = np.log(b0), np.log(b1)
    ls0, ls1 = np.log(s0), np.log(s1)
    for it in range(steps):
        f = it / max(steps-1,1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        smax = float(np.exp(ls0+(ls1-ls0)*f))
        X = _expand_anti(Y)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta*(G-m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if not np.isfinite(S) or S<=0:
            break
        GX = (E @ X)/S
        g_up = GX[:Y.shape[0]]
        g_lo = GX[Y.shape[0]:]
        grad = g_up - g_lo
        dot = np.sum(grad*Y, axis=1, keepdims=True)
        gt = grad - dot*Y
        gn = float(np.linalg.norm(gt, axis=1).max())
        if gn>1e-12:
            gt = gt/gn
        vel = mu*vel + (1.0-mu)*gt
        vn = float(np.linalg.norm(vel, axis=1).max())
        if vn>1e-14:
            Y = Y - (smax/vn)*vel
            Y = _normr(Y)
    return Y

def _smooth_q8(Y0, steps, b0, b1, s0, s1, mu=0.85):
    Y = _normr(Y0.copy())
    ns = Y.shape[0]
    vel = np.zeros_like(Y)
    lb0, lb1 = np.log(b0), np.log(b1)
    ls0, ls1 = np.log(s0), np.log(s1)
    for it in range(steps):
        f = it / max(steps-1,1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        smax = float(np.exp(ls0+(ls1-ls0)*f))
        X = _expand_q8(Y)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta*(G-m))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum()*0.5)
        if not np.isfinite(S) or S<=0:
            break
        GX = (E @ X)/S  # (272,4)
        grad = np.zeros_like(Y)
        for s in range(ns):
            acc = np.zeros(4)
            for e in range(8):
                acc += _Q8T[e] @ GX[s*8+e]
            grad[s] = acc
        grad = grad / max(S*0+1.0,1.0)  # GX already /S
        dot = np.sum(grad*Y, axis=1, keepdims=True)
        gt = grad - dot*Y
        gn = float(np.linalg.norm(gt, axis=1).max())
        if gn>1e-12:
            gt = gt/gn
        vel = mu*vel + (1.0-mu)*gt
        vn = float(np.linalg.norm(vel, axis=1).max())
        if vn>1e-14:
            Y = Y - (smax/vn)*vel
            Y = _normr(Y)
    return Y

def _lp_polish(X0, seconds, maxpairs=5000):
    X = _normr(X0.copy())
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep_n = np.repeat(np.arange(n), d)
    col_n = np.arange(n*d)
    while time.time()-start < seconds and radius > 1e-14:
        G = X @ X.T
        c_all = G[iu0]
        m = float(c_all.max())
        sel = c_all > m - max(20.0*radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > maxpairs:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], maxpairs-1)[:maxpairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+ar, J[:,None]*d+ar, np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))],1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(rep_n,col_n)), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius*=0.5; continue
        if res.status!=0:
            radius*=0.5; continue
        Y = X + res.x[:-1].reshape(n,d)
        Y = _normr(Y)
        v = _max_cos(Y)
        if v < best:
            X, best, radius = Y, v, min(radius*1.5, 0.05)
        else:
            radius*=0.5
    return X


def solve():
    import time as _t
    t0 = _t.time()
    HARD = 172.0
    LOOP_END = 148.0
    rng = np.random.RandomState(0)
    # load incumbent
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb,dtype=np.float64))
    if xr is not None:
        cands.append(np.asarray(xr,dtype=np.float64))
    if len(cands)==0:
        cands.append(_normr(rng.randn(272,4)))
    best = None; best_val = 2.0
    for c in cands:
        if c.shape != (272,4):
            continue
        c = _normr(c)
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    if best is None:
        best = _normr(rng.randn(272,4)); best_val=_max_cos(best)
    n,d = 272,4

    # ---- Phase A: antipodal symmetric search ----
    deadline_A = t0 + 30.0
    best_anti_X = None; best_anti_v = 2.0
    anti_tried = 0
    # low-discrepancy + random starts
    while _t.time() < deadline_A:
        anti_tried += 1
        if anti_tried % 5 == 0:
            # Hopf-ish init: grid on torus
            u = rng.rand(136); vv = rng.rand(136)
            th = 2*np.pi*u; ph = 2*np.pi*vv
            a = np.cos(th)/np.sqrt(2); b=np.sin(th)/np.sqrt(2)
            cc=np.cos(ph)/np.sqrt(2); dd=np.sin(ph)/np.sqrt(2)
            Y0 = np.stack([a,b,cc,dd],axis=1)
            Y0 += 0.05*rng.randn(136,4); Y0=_normr(Y0)
        else:
            Y0 = _normr(rng.randn(136,4))
        # random schedule
        sched = rng.choice([ (40,300,350,0.05,0.005),(80,400,450,0.04,0.004),(60,250,300,0.06,0.006)])
        b0,b1,steps,s0,s1 = sched
        try:
            Yopt = _smooth_anti(Y0, int(steps), float(b0), float(b1), float(s0), float(s1))
        except Exception:
            continue
        Xopt = _normr(_expand_anti(Yopt))
        v = _max_cos(Xopt)
        if v < best_anti_v:
            best_anti_v = v; best_anti_X = Xopt.copy()
        # early exit if time
        if _t.time() > deadline_A - 0.5:
            break

    # ---- Phase B: Q8 symmetric search ----
    deadline_B = t0 + 55.0
    best_q8_X = None; best_q8_v = 2.0
    best_q8_Y = None
    q_tried = 0
    while _t.time() < deadline_B:
        q_tried += 1
        Y0 = _normr(rng.randn(34,4))
        sched = rng.choice([ (40,300,350,0.05,0.005),(80,500,400,0.04,0.004),(50,200,300,0.06,0.008)])
        b0,b1,steps,s0,s1 = sched
        try:
            Yopt = _smooth_q8(Y0, int(steps), float(b0), float(b1), float(s0), float(s1))
        except Exception:
            continue
        Xopt = _normr(_expand_q8(Yopt))
        v = _max_cos(Xopt)
        if v < best_q8_v:
            best_q8_v = v; best_q8_X = Xopt.copy(); best_q8_Y = Yopt.copy()
        if _t.time() > deadline_B - 0.5:
            break

    # collect symmetric candidates
    sym_list = []
    if best_anti_X is not None:
        sym_list.append(best_anti_X)
    if best_q8_X is not None:
        sym_list.append(best_q8_X)
    # polish symmetric codes briefly to get true jammed symmetric value
    for i in range(len(sym_list)):
        if _t.time() > t0 + 62.0:
            break
        try:
            sym_list[i] = _lp_polish(sym_list[i], 1.2)
        except Exception:
            pass

    # ---- Phase C: graduated symmetry breaking ----
    eps_schedule = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    for Xs in list(sym_list):
        if _t.time() > t0 + 95.0:
            break
        Xcur = Xs.copy()
        # evaluate
        vcur = _max_cos(Xcur)
        # if symmetric much worse, still try breaking but limited effort
        for eps in eps_schedule:
            if _t.time() > t0 + 95.0:
                break
            if eps > 0:
                Xp = Xcur + eps * rng.randn(n,d)
                Xp = _normr(Xp)
            else:
                Xp = Xcur.copy()
            # re-anneal asymmetric, strength depends on eps
            if eps < 5e-4:
                steps, b0,b1,s0,s1 = 300, 80.0, 400.0, 0.03, 0.003
            elif eps < 2e-3:
                steps, b0,b1,s0,s1 = 350, 60.0, 350.0, 0.04, 0.003
            else:
                steps, b0,b1,s0,s1 = 400, 50.0, 300.0, 0.05, 0.004
            try:
                Xo = _smooth_asym(Xp, steps, b0,b1,s0,s1)
            except Exception:
                continue
            vo = _max_cos(Xo)
            # short LP if promising
            if vo < vcur + 0.004 and _t.time() < t0+94.0:
                try:
                    Xo2 = _lp_polish(Xo, 0.8)
                except Exception:
                    Xo2 = Xo
                vo2 = _max_cos(Xo2)
                if vo2 < vo:
                    Xo, vo = Xo2, vo2
            if vo < vcur:
                Xcur, vcur = Xo, vo
            if vo < best_val - 1e-12:
                best, best_val = Xo.copy(), vo
        # end eps loop

    # ---- Phase D: incumbent rescue / asymmetric improvement ----
    anneals = [
        (60,250,350,0.04,0.004),
        (100,400,400,0.03,0.003),
        (40,150,250,0.06,0.006),
        (150,500,300,0.02,0.002),
        (80,300,500,0.05,0.003),
    ]
    while _t.time() - t0 < LOOP_END:
        if _t.time() - t0 > LOOP_END - 2.0:
            break
        mode = rng.rand()
        Xt = best.copy()
        if mode < 0.35:
            sigma = float(rng.choice([0.003,0.006,0.012,0.025,0.05,0.10]))
            Xt = best + sigma*rng.randn(n,d); Xt=_normr(Xt)
        elif mode < 0.55:
            ci = int(rng.randint(n)); center = best[ci]
            dots_c = best @ center
            thr = float(rng.choice([0.90,0.93,0.96,0.98]))
            mask = dots_c > thr
            if int(mask.sum())<2:
                mask=np.zeros(n,dtype=bool); mask[ci]=True
            sc = float(rng.choice([0.01,0.02,0.04,0.08]))
            shift = rng.randn(d); shift/=max(np.linalg.norm(shift),1e-12)
            shift = shift - np.dot(shift,center)*center
            Xt[mask] += sc*shift + (sc*0.4)*rng.randn(int(mask.sum()),d)
            Xt += 0.002*rng.randn(n,d); Xt=_normr(Xt)
        elif mode < 0.75:
            try:
                G0 = best @ best.T; np.fill_diagonal(G0,-2.0)
                rowmax = G0.max(axis=1)
                order = np.argsort(-rowmax)
                k = int(rng.randint(1,4))
                ratt = np.where(rowmax < best_val-0.01)[0]
                cand_idx = list(order[:8])
                if len(ratt)>0:
                    cand_idx += list(rng.choice(ratt, min(len(ratt),4), replace=False))
                idx = np.array(rng.choice(cand_idx, k, replace=False))
                Xt = best + 0.005*rng.randn(n,d)
                # hole centres via random search
                for j in idx:
                    Cb = _normr(rng.randn(6000,4))
                    dots = Cb @ best.T
                    md = dots.max(axis=1)
                    bi = int(np.argmin(md))
                    h = Cb[bi] + 0.01*rng.randn(d)
                    h/=max(np.linalg.norm(h),1e-12)
                    Xt[j]=h
                Xt=_normr(Xt)
            except Exception:
                Xt = best+0.01*rng.randn(n,d); Xt=_normr(Xt)
        else:
            m = int(rng.randint(8,40))
            idx = rng.choice(n,m,replace=False)
            Q,_ = np.linalg.qr(rng.randn(d,d))
            if float(np.linalg.det(Q))<0:
                Q[:,0]=-Q[:,0]
            alpha = float(rng.uniform(0.05,0.4))
            R = np.eye(d)+alpha*(Q-np.eye(d))
            Xt[idx]=Xt[idx]@R.T
            Xt+=0.003*rng.randn(n,d); Xt=_normr(Xt)
        ai=int(rng.randint(len(anneals)))
        b0,b1,steps,s0,s1=anneals[ai]
        steps=int(steps*float(rng.uniform(0.8,1.2))); steps=max(150,min(600,steps))
        try:
            Xo=_smooth_asym(Xt,steps,b0,b1,s0,s1)
        except Exception:
            continue
        v0=_max_cos(Xo)
        if not np.isfinite(v0):
            continue
        if v0>best_val+0.012:
            continue
        if v0<best_val+0.002:
            rem=LOOP_END-(_t.time()-t0)
            if rem<0.3:
                if v0<best_val-1e-12:
                    best,best_val=Xo,v0
                continue
            lp_t=0.9 if v0<best_val else 0.45
            lp_t=min(lp_t,max(0.2,rem-0.15))
            try:
                Xp2=_lp_polish(Xo,lp_t)
            except Exception:
                continue
            v=_max_cos(Xp2)
            if v<best_val-1e-12 and np.isfinite(v):
                best,best_val=Xp2,v
        else:
            if v0<best_val-1e-12:
                best,best_val=Xo,v0
        if _t.time()-t0>=HARD-6:
            break

    remain = HARD-(_t.time()-t0)
    if remain>2.5:
        try:
            best=_lp_polish(best,min(remain-1.0,18.0))
        except Exception:
            pass
    best=_normr(best)
    return best
