import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 300:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9, weights=None):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        if weights is not None:
            W = W * weights
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        scale = np.ones_like(gn)
        scale[big] = 2.0 / np.maximum(gn[big], 1e-12)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _swap_push(x, a, b, c, d, m_ref, delta_open, eps_close, radius, steps=4):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    n, dd = x.shape
    iu0 = np.triu_indices(n, 1)
    pa = (min(a,b), max(a,b))
    pb = (min(c,d), max(c,d))
    target_open = m_ref - delta_open
    target_close = m_ref - eps_close
    for _ in range(steps):
        G = x @ x.T
        cab = float(G[a,b]); ccd = float(G[c,d])
        # already satisfied?
        if cab <= target_open + 2e-4 and ccd >= target_close - 2e-4:
            break
        c_all = G[iu0]
        # active set: top pairs near m_ref, exclude pa/pb
        thresh = m_ref - 0.02
        mask = c_all > thresh
        Im = iu0[0][mask]; Jm = iu0[1][mask]
        cm = c_all[mask]
        keep = ~(((Im==pa[0])&(Jm==pa[1]))|((Im==pb[0])&(Jm==pb[1])))
        Im = Im[keep]; Jm = Jm[keep]; cm = cm[keep]
        if cm.size > 800:
            o = np.argpartition(cm, -800)[-800:]
            Im, Jm, cm = Im[o], Jm[o], cm[o]
        p = cm.size
        nv = n*dd + 1
        # rows: p active + 1 open + 1 close
        nr = p + 2
        rows_list = []; cols_list = []; vals_list = []; bub = []
        # active rows
        if p > 0:
            rr = np.repeat(np.arange(p), 2*dd+1)
            cI = Im[:,None]*dd + np.arange(dd)
            cJ = Jm[:,None]*dd + np.arange(dd)
            cc = np.concatenate([cI, cJ, np.full((p,1), n*dd)], axis=1).ravel()
            vv = np.concatenate([x[Jm], x[Im], -np.ones((p,1))], axis=1).ravel()
            rows_list.append(rr); cols_list.append(cc); vals_list.append(vv)
            bub.append(-cm)
        # open row index p
        ro = np.full(2*dd, p)
        co = np.concatenate([np.array([a*dd+k for k in range(dd)]), np.array([b*dd+k for k in range(dd)])])
        # coeff: x[b] for dx_a, x[a] for dx_b
        vo = np.concatenate([x[b], x[a]])
        rows_list.append(ro); cols_list.append(co); vals_list.append(vo)
        bub.append(target_open - cab)
        # close row index p+1 : -x[d] dx_c - x[c] dx_d <= ccd - target_close
        ro2 = np.full(2*dd, p+1)
        co2 = np.concatenate([np.array([c*dd+k for k in range(dd)]), np.array([d*dd+k for k in range(dd)])])
        vo2 = np.concatenate([-x[d], -x[c]])
        rows_list.append(ro2); cols_list.append(co2); vals_list.append(vo2)
        bub.append(ccd - target_close)
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        bub = np.concatenate(bub)
        A_ub = coo_matrix((vals, (rows, cols)), shape=(nr, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), dd), np.arange(n*dd))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*dd)+[(None,None)], method="highs")
        except Exception:
            return None, False
        if res.status != 0:
            return None, False
        dx = res.x[:-1].reshape(n, dd)
        tval = float(res.x[-1])
        # reject if requires huge uphill
        if tval > m_ref + 0.012:
            # still apply once? reject
            return None, False
        x = _norm(x + dx)
        m_ref_cur = m_ref  # keep ref
    G2 = x @ x.T
    ok = (float(G2[a,b]) <= target_open + 1.5e-3) and (float(G2[c,d]) >= target_close - 1.5e-3)
    return x, ok

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 28.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            if a.shape==(n,d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_v:
                    best_v=v; best_x=a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n,d))); best_v=_max_cos(best_x)
    global_x = best_x.copy(); global_v = best_v
    m_ref0 = global_v

    iu = np.triu_indices(n,1)
    # precompute contacts of global
    def contacts_info(X, tol=1e-7):
        G = X @ X.T
        np.fill_diagonal(G,-2.0)
        m = float(G.max())
        c = G[iu]
        slack = m - c
        deg = np.zeros(n,dtype=int)
        # contacts mask
        cmask = slack < tol
        I = iu[0][cmask]; J = iu[1][cmask]
        for ii,jj in zip(I,J):
            deg[ii]+=1; deg[jj]+=1
        return G, m, c, slack, deg, I, J

    G0,m0,c0,slack0,deg0,I0,J0 = contacts_info(global_x, tol=2e-6)

    trials = 0
    # systematic swap attempts
    # build candidate swaps from low-degree vertices
    # order vertices by degree
    order_v = np.argsort(deg0)
    swap_cands = []
    Gfull = global_x @ global_x.T
    for vi in order_v[:60]:
        vi=int(vi)
        dots = Gfull[vi].copy()
        dots[vi]=-2
        # neighbors (contacts): dots > m0-2e-6
        neigh = np.where(dots > m0-2e-6)[0]
        # second shell: largest dots among non-neighbors, but above m0-0.02
        non = np.where(dots <= m0-2e-6)[0]
        non = non[non!=vi]
        if len(non)==0 or len(neigh)==0: continue
        o = np.argsort(-dots[non])[:4]
        for b in neigh[:6]:
            for kk in o:
                kk=int(non[int(kk)])
                if dots[kk] < m0-0.025: continue
                swap_cands.append((vi,int(b),vi,int(kk)))
    # shuffle deterministically but prioritize low degree
    idx = rng.permutation(len(swap_cands))
    swap_cands = [swap_cands[i] for i in idx]
    # limit
    swap_cands = swap_cands[:120]

    for (a,b,c_,d_) in swap_cands:
        if time.time() > HARD_END - FINAL_RESERVE - 8:
            break
        # skip if a==c etc degenerate (sharing vertex is intended)
        delta_open = float(rng.choice([0.004,0.007,0.010,0.015]))
        eps_close = float(rng.choice([0.0005,0.001,0.002]))
        radius = float(rng.choice([0.05,0.08,0.11]))
        Xp, ok = _swap_push(global_x, a,b,c_,d_, m_ref0, delta_open, eps_close, radius, steps=4)
        if Xp is None:
            continue
        # even if not fully ok, sometimes partial move is useful; only continue if moved
        # short randomized weighted smooth to diversify
        # random weights to break symmetry
        W = np.ones((n,n))
        # boost the closing pair, suppress opening pair implicitly via weights jitter
        jitter = rng.uniform(0.5, 1.5, size=(n,n))
        jitter = (jitter+jitter.T)/2
        np.fill_diagonal(jitter,0)
        Xp = _smooth_opt(Xp, steps=220, tau0=0.010, tau1=0.002, lr0=0.07, lr1=0.006, weights=jitter)
        Xp, vp = _lp_polish(Xp, time.time()+0.9, cap=2000, r0=8e-4)
        trials+=1
        if vp < global_v - 1e-12:
            global_v=vp; global_x=Xp.copy(); m_ref0=vp
            # rebuild candidates around new optimum
            G0,m0,c0,slack0,deg0,I0,J0 = contacts_info(global_x, tol=2e-6)
            Gfull = global_x @ global_x.T
            # early exit if big win? continue to improve
            if vp < 0.8490:
                break
        elif vp < global_v + 0.002:
            # try longer polish on near-miss with different basin: check hash diversity via degree
            pass

    # Phase 2: randomized weighted basin hopping to find floppy moves
    if time.time() < HARD_END - FINAL_RESERVE - 6:
        for it in range(30):
            if time.time() > HARD_END - FINAL_RESERVE - 4:
                break
            base = global_x.copy()
            # tangent noise kick focused on low-degree patch
            G,m,c,slack,deg,Ic,Jc = contacts_info(base, tol=2e-6)
            # pick patch: lowest-degree vertex + its neighbors
            v0 = int(np.argmin(deg + rng.uniform(0,0.5,size=n)))
            patch = set([v0])
            for ii,jj in zip(Ic,Jc):
                if ii==v0: patch.add(int(jj))
                if jj==v0: patch.add(int(ii))
            patch = list(patch)
            sigma = float(rng.choice([0.02,0.04,0.07]))
            noise = np.zeros((n,d))
            noise[patch] = rng.standard_normal(size=(len(patch),d))*sigma
            # tangent project
            noise -= np.sum(noise*base,axis=1,keepdims=True)*base
            y = _norm(base+noise)
            # weighted smooth with strong random anisotropy
            Wj = rng.uniform(0.3,2.0,size=(n,n)); Wj=(Wj+Wj.T)/2; np.fill_diagonal(Wj,0)
            y = _smooth_opt(y, steps=260, tau0=0.012, tau1=0.002, lr0=0.09, lr1=0.006, weights=Wj)
            y, vy = _lp_polish(y, time.time()+0.8, cap=2000, r0=8e-4)
            if vy < global_v:
                global_v=vy; global_x=y.copy(); m_ref0=vy

    try:
        dl = HARD_END - 1.0
        if dl > time.time():
            global_x, global_v = _lp_polish(global_x, dl, cap=3000, r0=1e-3)
    except Exception:
        pass
    return _norm(global_x)
