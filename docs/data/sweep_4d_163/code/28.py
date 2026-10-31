import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _pairs_sorted(x):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    order = np.argsort(-c)
    return iu[0][order], iu[1][order], c[order], iu

def _lp_polish(x, seconds, r0=2e-3, cap=2200):
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
            if cnt2 > cap + 800:
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

def _lp_duals(x, radius=1.5e-3, cap=2500):
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
        try:
            dual = np.abs(np.asarray(res.get('ineqlin', {}).get('marginals', np.ones(p)), dtype=float)).ravel()
            if dual.shape[0] != p:
                dual = np.ones(p)
        except Exception:
            dual = np.ones(p)
    dual = np.abs(dual)
    if np.max(dual) < 1e-12:
        dual = np.ones(p)
    return I, J, dual, m

def _smooth_forced(x, forced_set, m0, delta, iters, T, lr, boost=25.0, allow=0.004):
    n, d = x.shape
    F = np.zeros((n, n), dtype=float)
    if len(forced_set) > 0:
        for (a,b) in forced_set:
            F[a,b] = 1.0; F[b,a] = 1.0
    for k in range(iters):
        frac = k / max(iters-1, 1)
        Tk = T * (0.6 + 0.4*(1.0-frac))
        lrk = lr * (1.0 - 0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # weights: others relative to m0+allow (accept increase), forced relative to m0-delta
        W = np.exp((G - (m0+allow)) / Tk)
        np.fill_diagonal(W, 0.0)
        if len(forced_set) > 0:
            # extra weight on forced pairs
            Wf = np.exp((G - (m0-delta)) / (Tk*0.7)) * boost
            W = np.where(F > 0.5, Wf, W)
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
    n, d = x.shape
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

def _edge_set(x, thresh=1e-5):
    G = x @ x.T
    m = float(np.max(G - np.eye(163)*2.0))
    iu = np.triu_indices(163, 1)
    sel = G[iu] >= m - thresh
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s, m

def solve():
    rng = np.random.default_rng(163163)
    t0 = time.time()
    def TIME(): return time.time()-t0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=float)
                if c.shape == (163,4):
                    c = c/np.linalg.norm(c,axis=1,keepdims=True)
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
    # dual analysis at record/best
    base = best.copy()
    I, J, dual, m0 = _lp_duals(base, radius=1.5e-3, cap=2500)
    order_dual = np.argsort(-dual)
    Is = I[order_dual]; Js = J[order_dual]; ds = dual[order_dual]
    # point stress
    n = 163
    pstress = np.zeros(n); pcount = np.zeros(n, dtype=int)
    G0 = base @ base.T; np.fill_diagonal(G0, -2.0)
    rm0 = G0.max(axis=1)
    # contact count at 1e-9
    Ccnt = (G0 >= m0-1e-9).sum(axis=1)
    for k in range(len(I)):
        pstress[I[k]] += dual[k]; pstress[J[k]] += dual[k]
        pcount[I[k]] += 1; pcount[J[k]] += 1
    # tightest pairs sorted (for thresholds)
    PI, PJ, PC, _ = _pairs_sorted(base)
    base_edges, _ = _edge_set(base, thresh=1e-5)
    cur = best.copy(); cur_val = best_val
    cur_edges = base_edges
    deltas = [5e-4, 8e-4, 1.2e-3, 2e-3, 6e-4, 1.5e-3]
    # precompute patch centers: high stress + random
    high_pts = np.argsort(-(pstress + 0.5*Ccnt))[:12]
    rand_centers = rng.choice(n, size=12, replace=False)
    centers = list(high_pts) + list(rand_centers)
    trial = 0
    max_trials = 34
    di = 0
    while TIME() < 128 and trial < max_trials:
        mode = trial % 3
        delta = deltas[trial % len(deltas)]
        forced = []
        if mode == 0:
            # highest-dual edges, sliding window for diversity
            wsize = int(20 + (trial*7) % 21)  # 20..40
            start_idx = (trial*13) % max(1, len(Is)-wsize)
            for k in range(start_idx, min(start_idx+wsize, len(Is))):
                forced.append((int(Is[k]), int(Js[k])))
        elif mode == 1:
            # high-degree / high-stress points: take top 2-3 points, all incident tight edges
            npick = 2 + (trial % 2)  # 2 or 3
            # rotate through high_pts
            s0 = (trial*2) % len(high_pts)
            pts = [int(high_pts[(s0+i)%len(high_pts)]) for i in range(npick)]
            # collect incident pairs among top 700 tightest
            pool = 700
            inc = []
            # use set for fast check
            ptset = set(pts)
            for k in range(min(pool, len(PI))):
                a=int(PI[k]); b=int(PJ[k])
                if a in ptset or b in ptset:
                    inc.append((a,b))
            # also add pairs between stressed points' neighbours via pstress? trim to 20-40
            if len(inc) > 40:
                # keep those with largest PC (tightest)
                inc = inc[:40]
            if len(inc) < 20:
                # pad with top-dual edges
                need = 24 - len(inc)
                sidx = (trial*17) % max(1, len(Is)-need)
                for k in range(sidx, sidx+need):
                    e=(int(Is[k]),int(Js[k]))
                    if e not in inc:
                        inc.append(e)
            forced = inc[:40]
        else:
            # spatially localized patch
            c0 = int(centers[trial % len(centers)])
            dots = base @ base[c0] if trial % 2 == 0 else cur @ cur[c0]
            ksize = 13 + (trial % 4)  # 13..16 points
            patch = np.argsort(-dots)[:ksize]
            pset = set(patch.tolist())
            inc = []
            for k in range(min(800, len(PI))):
                # use cur pairs? recompute quickly for cur every few trials
                a=int(PI[k]); b=int(PJ[k])
                if a in pset and b in pset:
                    inc.append((a,b))
            if len(inc) < 8:
                # fallback: all pairs inside patch regardless of tightness, take closest
                for ii in range(len(patch)):
                    for jj in range(ii+1, len(patch)):
                        inc.append((int(patch[ii]), int(patch[jj])))
                        if len(inc) >= 30:
                            break
                    if len(inc) >= 30:
                        break
            # trim/pad to 20-40
            if len(inc) > 36:
                inc = inc[:36]
            forced = inc
        if len(forced) < 12:
            # pad
            need = 24 - len(forced)
            sidx = (trial*11) % max(1, len(Is)-need)
            for k in range(sidx, sidx+need):
                forced.append((int(Is[k]), int(Js[k])))
        if len(forced) > 40:
            forced = forced[:40]
        y = cur.copy()
        # small global jitter to break symmetry (deterministic via rng)
        y = y + rng.normal(scale=0.004, size=y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        # phase 1: forced reoptimization (accept increase elsewhere)
        y = _smooth_forced(y, forced, m0=cur_val, delta=delta, iters=260, T=0.012, lr=0.010, boost=30.0, allow=0.004)
        # verify forcing achieved? (not required) then remove forcing: plain relax
        y = _smooth_plain(y, iters=180, T0=0.010, T1=0.002, lr0=0.006, lr1=0.001)
        remain = 128 - TIME()
        if remain > 1.0:
            sec = min(0.9, max(0.4, remain/ (max_trials - trial + 6)))
            y = _lp_polish(y, seconds=sec, r0=1.5e-3)
        v = _max_cos(y)
        # topology difference at 1e-5
        try:
            es, _ = _edge_set(y, thresh=1e-5)
            inter = len(es.intersection(base_edges))
            union = len(es.union(base_edges))
            jac = inter/max(union,1)
        except Exception:
            jac = 1.0
        # keep best
        if v < best_val - 1e-12:
            best = y.copy(); best_val = v
            # refresh dual info occasionally around new best
            if trial % 6 == 5 and TIME() < 110:
                try:
                    I2,J2,d2,m2 = _lp_duals(best, radius=1.5e-3, cap=2500)
                    I,J,dual,m0 = I2,J2,d2,m2
                    order_dual = np.argsort(-dual)
                    Is=I[order_dual]; Js=J[order_dual]
                    pstress=np.zeros(n)
                    for k in range(len(I)):
                        pstress[I[k]]+=dual[k]; pstress[J[k]]+=dual[k]
                    high_pts = np.argsort(-pstress)[:12]
                    PI,PJ,PC,_ = _pairs_sorted(best)
                    base_edges,_ = _edge_set(best, thresh=1e-5)
                except Exception:
                    pass
        # hopping acceptance: prefer diverse topologies
        accept = False
        if v < cur_val:
            accept = True
        else:
            # allow uphill if topology is new
            bonus = 0.0
            if jac < 0.75:
                bonus = 0.0015
            elif jac < 0.88:
                bonus = 0.0006
            Tmet = 0.0008
            prob = np.exp(-(v - cur_val - bonus)/max(Tmet,1e-12))
            if rng.random() < prob:
                accept = True
        if accept:
            cur = y.copy(); cur_val = v; cur_edges = es
        # also occasionally reset cur to best if drifting too high
        if cur_val > best_val + 0.004:
            cur = best.copy(); cur_val = best_val
        trial += 1
        di += 1
    # final polish of best
    elapsed = TIME()
    budget = max(6.0, min(38.0, 172.0 - elapsed))
    t1 = time.time()
    try:
        best = _smooth_plain(best, iters=220, T0=0.008, T1=0.0012, lr0=0.004, lr1=0.0007)
    except Exception:
        pass
    left = budget - (time.time()-t1)
    if left > 2:
        best = _lp_polish(best, seconds=left, r0=1.2e-3)
    best = best/np.linalg.norm(best,axis=1,keepdims=True)
    best = np.nan_to_num(best)
    best /= np.linalg.norm(best,axis=1,keepdims=True)
    return best
