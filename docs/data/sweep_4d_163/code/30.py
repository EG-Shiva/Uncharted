import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - thr
    return iu[0][sel], iu[1][sel], c[sel], m

def _lp_polish(x, seconds, t_end, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    start = time.time()
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > (m - window)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.argpartition(c_all, -2200)[-2200:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": 1.5})
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
    return x, best

def _lp_duals(x, radius=1e-3, window=None):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    if window is None:
        window = max(20.0*radius, 2e-4)
    sel = c_all > (m - window)
    cnt = int(np.count_nonzero(sel))
    if cnt > 2500:
        idx = np.argpartition(c_all, -2500)[-2500:]
        sel = np.zeros_like(sel, dtype=bool)
        sel[idx] = True
    I = iu[0][sel]; J = iu[1][sel]
    c = c_all[sel]
    p = len(I)
    nv = n*d+1
    rows = np.repeat(np.arange(p), 2*d+1)
    cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
    A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
    A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
    obj = np.zeros(nv); obj[-1]=1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                      options={"time_limit": 3.0})
    except Exception:
        return I, J, c, m, None, False
    if res.status != 0:
        return I, J, c, m, None, False
    duals = None
    try:
        marg = res.ineqlin.marginals
        duals = np.asarray(marg, dtype=float).ravel()
    except Exception:
        try:
            marg = res['ineqlin']['marginals']
            duals = np.asarray(marg, dtype=float).ravel()
        except Exception:
            try:
                marg = getattr(res, 'marginals', None)
                if marg is not None:
                    duals = np.asarray(marg, dtype=float).ravel()[:p]
                else:
                    duals = None
            except Exception:
                duals = None
    if duals is not None and duals.shape[0] != p:
        try:
            duals = np.asarray(duals).ravel()[:p]
            if duals.shape[0] != p:
                duals = None
        except Exception:
            duals = None
    return I, J, c, m, duals, True

def _forced_reopt(x0, FI, FJ, delta, orig_max, iters=6, r0=0.015, t_end=None):
    n, d = x0.shape
    x = x0.copy()
    target = orig_max - delta
    radius = r0
    iu = np.triu_indices(n, 1)
    fset = set(zip(FI.tolist(), FJ.tolist()))
    for it in range(iters):
        if t_end is not None and time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        # free candidates: near-max excluding forced
        window = max(20.0*radius, 0.012)
        c_all = G[iu]
        sel = c_all > (cur_max - window)
        If = iu[0][sel]; Jf = iu[1][sel]; cf = c_all[sel]
        # remove forced pairs from free set
        if len(FI) > 0:
            # mask via set lookup (p<=~2500, fine)
            mask = np.ones(len(If), dtype=bool)
            for k in range(len(If)):
                a, b = int(If[k]), int(Jf[k])
                if a > b: a, b = b, a
                if (a, b) in fset:
                    mask[k] = False
            If, Jf, cf = If[mask], Jf[mask], cf[mask]
        # cap free to top 2000
        if len(If) > 2000:
            idx = np.argpartition(cf, -2000)[-2000:]
            If, Jf, cf = If[idx], Jf[idx], cf[idx]
        pf = len(If)
        pp = len(FI)
        c_forced_cur = G[FI, FJ] if pp>0 else np.zeros(0)
        # build combined LP: min t s.t. free: <..>-t<=-c ; forced: <..><=target-c
        p = pf + pp
        if p == 0:
            break
        nv = n*d+1
        # rows
        II = np.concatenate([If, FI]) if pp>0 else If
        JJ = np.concatenate([Jf, FJ]) if pp>0 else Jf
        # vals: for free rows last col -1, for forced rows 0
        Xt = np.concatenate([x[JJ[:pf]] if pf>0 else np.zeros((0,d)), x[JJ[pf:]] if pp>0 else np.zeros((0,d))], axis=0) if p>0 else np.zeros((0,d))
        # simpler: build per row
        rows = np.repeat(np.arange(p), 2*d+1)
        cd = np.arange(d)
        cols = np.concatenate([II[:,None]*d+cd, JJ[:,None]*d+cd, np.full((p,1), n*d)], axis=1).ravel()
        top = np.concatenate([x[JJ], x[II]], axis=1) if False else None
        # x[J] for first block = x[other]
        v1 = x[JJ]
        v2 = x[II]
        vlast = np.concatenate([-np.ones((pf,1)), np.zeros((pp,1))], axis=0) if p>0 else np.zeros((0,1))
        vals = np.concatenate([v1, v2, vlast], axis=1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        b_ub = np.concatenate([-cf, target - c_forced_cur]) if p>0 else np.zeros(0)
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs",
                          options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            # infeasible -> relax: shrink delta effect by raising target slightly or shrink radius
            radius *= 0.5
            if radius < 1e-5:
                break
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        Gy = y @ y.T
        np.fill_diagonal(Gy, -2.0)
        forced_new = float(Gy[FI,FJ].max()) if pp>0 else -2.0
        forced_old = float(c_forced_cur.max()) if pp>0 else -2.0
        if forced_new < forced_old - 1e-12:
            x = y
            radius = min(radius*1.25, 0.04)
        else:
            # accept if not worse much? to keep moving
            if forced_new <= forced_old + 1e-9:
                x = y
            radius *= 0.6
        if radius < 1e-6:
            break
    return x

def _edge_set(x, thr=1e-5):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    sel = G[iu] > m - thr
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    return s

def solve():
    t0 = time.time()
    T_END = t0 + 168.0
    rng0 = np.random.RandomState(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng0.randn(163, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (163, 4):
        tmp = rng0.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (163,4) and _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 163, 4
    best_x = xb.copy()
    best_v = _max_cos(best_x)
    # initial short polish
    best_x, best_v = _lp_polish(best_x, min(4.0, max(0.5, T_END-8.0-time.time())), T_END-5.0, r0=2e-3)
    orig_max_for_sets = best_v
    # dual-guided analysis
    I, J, c, m, duals, ok = _lp_duals(best_x, radius=1e-3, window=0.02)
    if (not ok) or (duals is None):
        # fallback: uniform stress proxy = -c (tighter = more stressed) + degree
        w = np.ones_like(c)
        # higher c => higher stress
        order = np.argsort(-c)
        duals = np.zeros_like(c, dtype=float)
        # assign decreasing weights
        duals[order] = 1.0/(1.0+np.arange(len(c)))
    aduals = np.abs(duals)
    # point stress
    pstress = np.zeros(n)
    np.add.at(pstress, I, aduals)
    np.add.at(pstress, J, aduals)
    # degrees at 1e-9
    thr09 = 1e-9
    deg = np.zeros(n, dtype=int)
    sel09 = c > (m - thr09)
    I09, J09 = I[sel09], J[sel09]
    for a,b in zip(I09.tolist(), J09.tolist()):
        deg[a]+=1; deg[b]+=1
    # build clusters (each is (FI,FJ))
    clusters = []
    # A: top-dual edges 30
    o = np.argsort(-aduals)
    k = min(30, len(o))
    clusters.append((I[o[:k]].copy(), J[o[:k]].copy(), "topdual30"))
    # A2: top-dual edges 20
    k2 = min(20, len(o))
    clusters.append((I[o[:k2]].copy(), J[o[:k2]].copy(), "topdual20"))
    # A3: top-dual 40
    k3 = min(40, len(o))
    clusters.append((I[o[:k3]].copy(), J[o[:k3]].copy(), "topdual40"))
    # B: top stress points incident edges
    top_pts = np.argsort(-pstress)[:8]
    sset = set(top_pts.tolist())
    mask = np.array([(a in sset) or (b in sset) for a,b in zip(I.tolist(), J.tolist())])
    idx = np.where(mask)[0]
    if len(idx)>0:
        oo = idx[np.argsort(-aduals[idx])]
        kk = min(36, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "toppts8"))
    top_pts6 = np.argsort(-pstress)[:6]
    sset6 = set(top_pts6.tolist())
    mask6 = np.array([(a in sset6) or (b in sset6) for a,b in zip(I.tolist(), J.tolist())])
    idx6 = np.where(mask6)[0]
    if len(idx6)>0:
        oo = idx6[np.argsort(-aduals[idx6])]
        kk = min(30, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "toppts6"))
    # C: spatial patch around most stressed point (nearest 12)
    p0 = int(np.argmax(pstress))
    Grow = (best_x @ best_x[p0])
    Grow[p0] = -2.0
    nearest = np.argsort(-Grow)[:12]
    S = set(nearest.tolist()+[p0])
    maskS = np.array([(a in S) and (b in S) for a,b in zip(I.tolist(), J.tolist())])
    idxS = np.where(maskS)[0]
    if len(idxS) < 20:
        # add edges from S outward with highest dual
        maskSO = np.array([(a in S) or (b in S) for a,b in zip(I.tolist(), J.tolist())])
        idxSO = np.where(maskSO)[0]
        oo = idxSO[np.argsort(-aduals[idxSO])]
        kk = min(30, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "patch13"))
    else:
        oo = idxS[np.argsort(-aduals[idxS])]
        kk = min(30, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "patch-in"))
    # D: high-degree points
    top_deg = np.argsort(-deg)[:6]
    sd = set(top_deg.tolist())
    maskd = np.array([(a in sd) or (b in sd) for a,b in zip(I.tolist(), J.tolist())])
    idxd = np.where(maskd)[0]
    if len(idxd)>0:
        oo = idxd[np.argsort(-aduals[idxd])]
        kk = min(32, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "highdeg"))
    # E: rattler-adjacent voids? edges incident to lowest-degree patch (second patch for diversity)
    p1 = int(np.argsort(-pstress)[1]) if n>1 else p0
    Grow1 = (best_x @ best_x[p1])
    Grow1[p1] = -2.0
    nearest1 = np.argsort(-Grow1)[:12]
    S1 = set(nearest1.tolist()+[p1])
    maskS1 = np.array([(a in S1) or (b in S1) for a,b in zip(I.tolist(), J.tolist())])
    idxS1 = np.where(maskS1)[0]
    if len(idxS1)>0:
        oo = idxS1[np.argsort(-aduals[idxS1])]
        kk = min(28, len(oo))
        clusters.append((I[oo[:kk]].copy(), J[oo[:kk]].copy(), "patch2"))
    deltas = [5e-4, 8e-4, 1.2e-3, 2e-3]
    base_edge_set = _edge_set(best_x, thr=1e-5)
    # pool of diverse candidates (x, v)
    pool = [(best_x.copy(), best_v)]
    trial = 0
    ci = 0
    # iterate over clusters x deltas until time
    combos = []
    for cl in clusters:
        for dl in deltas:
            combos.append((cl, dl))
    # deterministic shuffle: sort by custom order to interleave deltas
    # prioritize topdual + mid deltas first (as per assignment)
    order_pref = []
    for dl in [8e-4, 1.2e-3, 5e-4, 2e-3]:
        for cl in clusters:
            order_pref.append((cl, dl))
    combos = order_pref
    for (FI0, FJ0, nm), delta in combos:
        if time.time() > T_END - 10.0:
            break
        # choose seed: mostly best, sometimes second pool entry for diversity
        if trial % 5 == 4 and len(pool) > 1:
            seed_x, seed_v = pool[1]
            seed_x = seed_x.copy()
        else:
            seed_x, seed_v = best_x, best_v
        # re-anchor target to seed max
        seed_max = _max_cos(seed_x)
        # copy forced pairs (ensure valid, unique, i<j)
        FI = FI0.copy(); FJ = FJ0.copy()
        # filter pairs still near-max in seed (within 0.01)? keep all but ensure they exist
        # if seed changed, recompute: keep those with cosine > seed_max-0.005
        Gs = seed_x @ seed_x.T
        cfilt = Gs[FI, FJ]
        keep = cfilt > (seed_max - 0.005)
        if int(np.count_nonzero(keep)) >= 12:
            FI, FJ = FI[keep], FJ[keep]
        # cap 20-40
        if len(FI) > 36:
            # keep highest current cosine among them
            cc = Gs[FI, FJ]
            oo = np.argsort(-cc)[:36]
            FI, FJ = FI[oo], FJ[oo]
        if len(FI) < 12:
            continue
        trial += 1
        t_rem = T_END - 6.0 - time.time()
        if t_rem < 3.0:
            break
        x_f = _forced_reopt(seed_x, FI, FJ, delta, seed_max, iters=6, r0=0.015, t_end=T_END-8.0)
        v_f = _max_cos(x_f)
        # if forced blew up too much, skip polish? still polish but with limited budget
        # polish without forcing
        budget = 3.5 if v_f < seed_v + 0.01 else 2.0
        budget = min(budget, T_END-4.0-time.time())
        if budget < 0.6:
            break
        x_p, v_p = _lp_polish(x_f, budget, T_END-2.0, r0=2e-3)
        # diversity check
        es = _edge_set(x_p, thr=1e-5)
        inter = len(es & base_edge_set)
        union = len(es | base_edge_set)
        jac = inter/max(union,1)
        # update pool with diverse near-best
        if v_p < best_v:
            best_v = float(v_p); best_x = x_p.copy()
            base_edge_set = _edge_set(best_x, thr=1e-5)
            pool.insert(0, (best_x.copy(), best_v))
            pool = pool[:4]
        elif jac < 0.85 and v_p < seed_v + 0.003:
            # keep diverse candidate
            pool.append((x_p.copy(), float(v_p)))
            # keep pool small, sort by v
            pool = sorted(pool, key=lambda t: t[1])[:4]
            # also if diverse and close, try extra polish later (handled by seeding)
        ci += 1
    # final polish best
    if time.time() < T_END - 2.0:
        sec = max(0.5, min(8.0, T_END-1.5-time.time()))
        best_x, best_v = _lp_polish(best_x, sec, T_END-0.8, r0=2e-3)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
