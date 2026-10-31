import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, r0=2e-3, cap=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 600:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None]*d+cd, J[:, None]*d+cd, np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _lp_duals(x, radius=1.5e-3, cap=1800):
    n, d = x.shape
    iu0 = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu0]
    m = float(c_all.max())
    thr = m - max(20.0*radius, 1e-9)
    sel = c_all > thr
    if int(np.count_nonzero(sel)) > cap:
        v = c_all[sel]
        kth = np.sort(v)[-cap]
        sel = c_all >= kth
    I = iu0[0][sel]; J = iu0[1][sel]
    c = c_all[sel]
    p = len(I)
    nv = n*d+1
    rows = np.repeat(np.arange(p), 2*d+1)
    cd = np.arange(d)
    cols = np.concatenate([I[:,None]*d+cd, J[:,None]*d+cd, np.full((p,1), n*d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
    except Exception:
        return I, J, np.ones(len(I)), m
    if res.status != 0:
        return I, J, np.ones(len(I)), m
    dual = None
    try:
        if hasattr(res, 'ineqlin') and res.ineqlin is not None:
            mg = None
            if isinstance(res.ineqlin, dict):
                mg = res.ineqlin.get('marginals', None)
            else:
                mg = getattr(res.ineqlin, 'marginals', None)
            if mg is not None:
                dual = np.asarray(mg, dtype=float).ravel()
    except Exception:
        dual = None
    if dual is None or dual.shape[0] != p:
        dual = np.ones(p)
    dual = np.abs(np.asarray(dual, dtype=float).ravel())
    if dual.shape[0] != p or np.max(dual) < 1e-12 or not np.all(np.isfinite(dual)):
        dual = np.ones(p)
    return I, J, dual, m

def _smooth_forced(x, forced_set, m0, delta, iters, T, lr, boost=20.0, allow=0.004):
    n, d = x.shape
    F = np.zeros((n, n), dtype=bool)
    for (a,b) in forced_set:
        F[a,b] = True; F[b,a] = True
    hasF = len(forced_set) > 0
    for k in range(iters):
        frac = k / max(iters-1, 1)
        Tk = T * (0.6 + 0.4*(1.0-frac))
        lrk = lr * (1.0 - 0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        W = np.exp((G - (m0+allow)) / Tk)
        np.fill_diagonal(W, 0.0)
        if hasF:
            Wf = np.exp((G - (m0-delta)) / (Tk*0.7)) * 1.0) * boost
            W = np.where(F, Wf, W)
            np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lrk*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _smooth_plain(x, iters, T0, T1, lr0, lr1):
    for k in range(iters):
        frac = k / max(iters-1, 1)
        T = T0*(T1/T0)**frac
        lr = lr0*(lr1/lr0)**frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G-m)/T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad*x, axis=1, keepdims=True)
        gt = grad - dot*x
        rs = W.sum(axis=1, keepdims=True)
        step = gt/np.maximum(rs,1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step/np.maximum(sn,1.0)
        x = x - lr*step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    rng = np.random.default_rng(163163)
    t0 = time.time()
    def elapsed(): return time.time()-t0
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (163,4):
                c = c/np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(163,4)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        cands.append(z)
    best = None; best_val = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; best = c.copy()
    base = best.copy()
    n = 163
    try:
        I0, J0, dual0, m00 = _lp_duals(base, radius=1.5e-3, cap=1800)
    except Exception:
        iu = np.triu_indices(n,1)
        G = base@base.T
        c_all = G[iu]
        m00 = float(c_all.max())
        I0, J0, dual0 = iu[0], iu[1], np.ones_like(c_all)
    order_dual = np.argsort(-dual0)
    Is = np.asarray(I0)[order_dual]; Js = np.asarray(J0)[order_dual]; ds = np.asarray(dual0)[order_dual]
    G0 = base @ base.T; np.fill_diagonal(G0, -2.0)
    Ccnt = (G0 >= m00-1e-9).sum(axis=1)
    pstress = np.zeros(n)
    for k in range(len(I0)):
        a=int(I0[k]); b=int(J0[k])
        w=float(dual0[k]) if k < len(dual0) else 1.0
        pstress[a]+=w; pstress[b]+=w
    high_pts = np.argsort(-(pstress + 0.5*Ccnt))[:12]
    rand_centers = rng.choice(n, size=12, replace=False)
    centers = list(int(v) for v in high_pts) + list(int(v) for v in rand_centers)
    cur = best.copy(); cur_val = best_val
    deltas = [5e-4, 8e-4, 1.2e-3, 2e-3, 6e-4, 1.5e-3]
    trial = 0
    max_trials = 30
    iu_all = np.triu_indices(n, 1)
    while elapsed() < 128 and trial < max_trials:
        # tight pairs of cur for diversity
        Gc = cur @ cur.T
        cc = Gc[iu_all]
        o = np.argsort(-cc)
        PI = iu_all[0][o]; PJ = iu_all[1][o]
        mode = trial % 3
        delta = deltas[trial % len(deltas)]
        forced = []
        if mode == 0:
            wsize = int(20 + (trial*7) % 21)
            sidx = (trial*13) % max(1, len(Is)-wsize)
            for k in range(sidx, min(sidx+wsize, len(Is))):
                forced.append((int(Is[k]), int(Js[k])))
        elif mode == 1:
            npick = 2 + (trial % 2)
            s0 = (trial*2) % len(high_pts)
            pts = [int(high_pts[(s0+i)%len(high_pts)]) for i in range(npick)]
            ptset = set(pts)
            pool = 700
            inc = []
            for k in range(min(pool, len(PI))):
                a=int(PI[k]); b=int(PJ[k])
                if a in ptset or b in ptset:
                    inc.append((a,b))
            if len(inc) > 40:
                inc = inc[:40]
            if len(inc) < 20:
                need = 24 - len(inc)
                sidx = (trial*17) % max(1, len(Is)-need)
                for k in range(sidx, sidx+need):
                    e=(int(Is[k]),int(Js[k]))
                    if e not in inc:
                        inc.append(e)
            forced = inc[:40]
        else:
            c0 = int(centers[trial % len(centers)])
            dots = cur @ cur[c0]
            ksize = 13 + (trial % 4)
            patch = np.argsort(-dots)[:ksize]
            pset = set(patch.tolist())
            inc = []
            for k in range(min(800, len(PI))):
                a=int(PI[k]); b=int(PJ[k])
                if a in pset and b in pset:
                    inc.append((a,b))
            if len(inc) < 8:
                for ii in range(len(patch)):
                    for jj in range(ii+1, len(patch)):
                        inc.append((int(patch[ii]), int(patch[jj])))
                        if len(inc) >= 30:
                            break
                    if len(inc) >= 30:
                        break
            if len(inc) > 36:
                inc = inc[:36]
            forced = inc
        if len(forced) < 12:
            need = 24 - len(forced)
            sidx = (trial*11) % max(1, len(Is)-need)
            for k in range(sidx, sidx+need):
                forced.append((int(Is[k]), int(Js[k])))
        if len(forced) > 40:
            forced = forced[:40]
        y = cur + rng.normal(scale=0.004, size=cur.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth_forced(y, forced, m0=cur_val, delta=delta, iters=200, T=0.012, lr=0.02, boost=20.0, allow=0.004)
        y = _smooth_plain(y, iters=150, T0=0.008, T1=0.001, lr0=0.015, lr1=0.004)
        tleft = 150 - elapsed()
        if tleft < 2:
            break
        psecs = 3.0 if tleft > 25 else 1.5
        y2 = _lp_polish(y, seconds=psecs, r0=2e-3, cap=1600)
        v = _max_cos(y2)
        if v < best_val:
            best_val = v; best = y2.copy()
        # walker acceptance: allow small uphill to change basin
        if v < cur_val or (v < cur_val + 0.003 and rng.random() < 0.25):
            cur = y2.copy(); cur_val = v
        elif rng.random() < 0.05:
            cur = y2.copy(); cur_val = v
        trial += 1
    # final polish of best
    tleft = 172 - elapsed()
    if tleft > 2:
        best = _lp_polish(best, seconds=min(tleft-1, 10.0), r0=1.5e-3, cap=1800)
        best_val = _max_cos(best)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
