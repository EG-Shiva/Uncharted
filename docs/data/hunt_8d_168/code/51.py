import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
try:
    from scipy.sparse.linalg import lsmr
    _HAS_LSMR = True
except Exception:
    _HAS_LSMR = False

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d-1), dtype=np.float64)
    I = np.eye(d)
    for i in range(n):
        xi = X[i]
        k = int(np.argmin(np.abs(xi)))
        e = np.zeros(d); e[k] = 1.0
        diff = e - xi
        nd = float(np.sqrt(diff @ diff))
        if nd < 1e-12:
            # xi almost e_k, use any orthogonal basis
            Q, _ = np.linalg.qr(np.eye(d) - np.outer(xi, xi))
            # Q first col ~ xi, take rest; ensure tangent
            # find col most aligned with xi
            dots = np.abs(Q.T @ xi)
            ic = int(np.argmax(dots))
            cols = [c for c in range(d) if c != ic]
            Ti = Q[:, cols]
            # project to be safe
            Ti = Ti - np.outer(xi, (xi @ Ti))
            # orthonormalize
            Ti, _ = np.linalg.qr(Ti)
            T[i] = Ti
            continue
        u = diff / nd
        H = I - 2.0 * np.outer(u, u)
        cols = [c for c in range(d) if c != k]
        T[i] = H[:, cols]
    return T

def _viol_stats(X, c):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    iu = np.triu_indices(n, 1)
    vals = G[iu]
    vp = vals - c
    pos = vp > 0
    S = float((vp[pos]**2).sum()) if np.any(pos) else 0.0
    vmax = float(vp.max()) if vals.size else -1.0
    return G, iu, vals, vp, S, vmax

def _stress_imbalance(X, pairs_I, pairs_J, c_thresh):
    # uniform-weight tangent imbalance norm per point
    n, d = X.shape
    G = X @ X.T
    imb = np.zeros(n)
    deg = np.zeros(n, dtype=int)
    acc = np.zeros_like(X)
    for k in range(len(pairs_I)):
        i = int(pairs_I[k]); j = int(pairs_J[k])
        g = float(G[i, j])
        ti = X[j] - g * X[i]
        tj = X[i] - g * X[j]
        acc[i] += ti
        acc[j] += tj
        deg[i] += 1; deg[j] += 1
    for i in range(n):
        if deg[i] > 0:
            v = acc[i] / deg[i]
            imb[i] = float(np.sqrt(v @ v))
    return imb, deg

def _attempt_target(X0, c, deadline, max_iters=70, cap_viol=2500):
    n, d = X0.shape
    dm1 = d - 1
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    _, _, _, _, S_best, vmax0 = _viol_stats(X, c)
    X_best = X.copy()
    S_best_mem = S_best
    trust = 1e-2
    damp = 2e-2
    zero_streak = None
    if not _HAS_LSMR:
        return X_best, False, S_best_mem
    iu_all = np.triu_indices(n, 1)
    for it in range(max_iters):
        if time.time() > deadline:
            break
        G, iu, vals, vp, S, vmax = _viol_stats(X, c)
        if vmax <= 1e-12:
            return X, True, 0.0
        if S < S_best_mem:
            S_best_mem = S
            X_best = X.copy()
        # violated set
        vmask = vals > c
        # active window for stress: > c-3e-4
        amask = vals > (c - 3e-4)
        I_all = iu[0][vmask]; J_all = iu[1][vmask]
        vp_all = (vals[vmask] - c)
        if I_all.size == 0:
            return X, True, 0.0
        # cap to tightest
        if I_all.size > cap_viol:
            od = np.argsort(vp_all)[-cap_viol:]
            I_all = I_all[od]; J_all = J_all[od]; vp_all = vp_all[od]
        m = I_all.size
        # tangent bases
        T = _tangent_bases(X)
        # build sparse J (m x n*dm1): each row 2*(d-1) nonzeros? actually 2*(d-1) via blocks
        # block values: Ti^T xj and Tj^T xi
        rows = np.repeat(np.arange(m), 2*dm1)
        cols = np.empty(m*2*dm1, dtype=np.int64)
        data = np.empty(m*2*dm1, dtype=np.float64)
        # vectorized per-row using loops (m<=2500, fine)
        for k in range(m):
            i = int(I_all[k]); j = int(J_all[k])
            bi = T[i].T @ X[j]  # dm1
            bj = T[j].T @ X[i]
            base = k*2*dm1
            cols[base:base+dm1] = i*dm1 + np.arange(dm1)
            data[base:base+dm1] = bi
            cols[base+dm1:base+2*dm1] = j*dm1 + np.arange(dm1)
            data[base+dm1:base+2*dm1] = bj
        Nvar = n*dm1
        J = coo_matrix((data, (rows, cols)), shape=(m, Nvar)).tocsr()
        b = -vp_all
        try:
            res = lsmr(J, b, damp=damp, maxiter=min(800, max(200, m//2)))
            y = np.asarray(res[0], dtype=np.float64)
        except Exception:
            trust *= 0.5; damp *= 2.0
            continue
        if y.size != Nvar or not np.all(np.isfinite(y)):
            trust *= 0.5; damp *= 2.0
            continue
        Y = y.reshape(n, dm1)
        DX = np.einsum('nik,nk->ni', T, Y)
        # trust region cap on max per-point norm
        norms = np.sqrt((DX*DX).sum(axis=1))
        mn = float(norms.max()) if norms.size else 0.0
        if mn > trust and mn > 0:
            DX *= (trust / mn)
        Xt = X + DX
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        _, _, _, _, St, vmaxt = _viol_stats(Xt, c)
        if St < S:
            X = Xt
            # progress -> expand trust, reduce damping
            trust = min(trust*1.35, 1e-2)
            damp = max(damp*0.65, 1e-4)
            # stress screening every success near feasibility
            if (it % 8 == 0):
                try:
                    Ia = iu[0][amask]; Ja = iu[1][amask]
                    if Ia.size > 4000:
                        sel = np.argsort(vals[amask])[-4000:]
                        Ia = Ia[sel]; Ja = Ja[sel]
                    imb, deg = _stress_imbalance(X, Ia, Ja, c)
                    # if highly imbalanced hubs persist, nudge slightly opposite imbalance
                    # compute imbalance vectors with uniform weights for top hubs
                    # (small deterministic nudge, keeps feasibility improving)
                    order = np.argsort(imb)[::-1][:6]
                    for idx in order:
                        if deg[idx] >= 4 and imb[idx] > 1e-3:
                            # direction of centroid
                            neigh = []
                            # find neighbors of idx in active set (cheap scan)
                            # use G row
                            row = G[idx] if 'G' in locals() else (X @ X.T)[idx]
                            # top neighbors
                            ni = np.argsort(row)[-deg[idx]:]
                            cen = X[ni].mean(axis=0)
                            cen = cen - float(cen @ X[idx])*X[idx]
                            nn = float(np.sqrt(cen @ cen))
                            if nn > 1e-12:
                                X[idx] = X[idx] - 0.15*trust*(cen/nn)
                    X /= np.linalg.norm(X, axis=1, keepdims=True)
                except Exception:
                    pass
        else:
            trust *= 0.5
            damp *= 1.8
            if trust < 9e-5:
                # no further progress at smallest trust
                # try tiny random tangent kick once
                if it % 15 == 14:
                    trust = 3e-4
                else:
                    # check if S stagnant -> break
                    pass
            if trust < 2e-5:
                break
        if trust < 1e-4 and S > S_best_mem:
            # continue only while feasibility improves (spec)
            # if last 12 iters no improvement, stop
            pass
    # not reached; return best seen for this target
    return X_best, False, S_best_mem

def _balanced_quench(X, steps, b0, b1, lr0, lr1, lam, t_lim):
    n = X.shape[0]
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(b0); lb1 = np.log(b1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps+1):
        if (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t/steps
        beta = np.exp(lb0+(lb1-lb0)*frac)
        lr = np.exp(ll0+(ll1-ll0)*frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-18
        Gs = (W @ X)/S
        Gb = X @ X.T
        np.fill_diagonal(Gb, 0.0)
        np.maximum(Gb, 0.0, out=Gb)
        Gbb = (Gb @ X)*(2.0/n)
        Grad = Gs + lam*Gbb
        dot = (Grad*X).sum(axis=1, keepdims=True)
        Gt = Grad - dot*X
        m *= b1a; m += 0.1*Gt
        v *= b2; v += 0.001*(Gt*Gt)
        bc1 = 1.0-b1a**t; bc2 = 1.0-b2**t
        step = (m/bc1)/(np.sqrt(v/bc2)+eps)
        X = X - lr*step
        X /= np.sqrt((X*X).sum(axis=1, keepdims=True))
    return X

def _lp_polish(x, seconds, r0=1.2e-3):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        mm = float(c.max())
        sel = c > mm - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            idx = np.argsort(c[sel])[-2800:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _find_holes(Y, k, rng, beta=90.0, n_probe=8000):
    d = Y.shape[1]
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 2000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a+CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    holes = []
    for idx in order:
        p = P[idx]
        ok = True
        for q in holes:
            if abs(float(p @ q)) > 0.88:
                ok = False; break
        if ok:
            # refine
            pp = p.copy()/np.linalg.norm(p)
            for it in range(150):
                frac = it/149
                lr = 0.12*(0.004/0.12)**frac
                g = Y @ pp
                gm = g.max()
                w = np.exp(beta*(g-gm))
                s = w.sum()+1e-18
                A = (w @ Y)/s
                At = A - float(A @ pp)*pp
                pp = pp - lr*At
                pp /= np.linalg.norm(pp)
            holes.append(pp)
            if len(holes) >= k:
                break
    while len(holes) < k:
        holes.append(P[order[len(holes)]]/np.linalg.norm(P[order[len(holes)]]))
    return holes

def solve():
    t0 = time.time()
    BUDGET = 174.0
    n, d = 168, 8
    rng = np.random.RandomState(7)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a/np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.randn(n, d); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy(); inc_val = best_val
    # --- target-cosine homotopy ---
    try:
        deltas = [1e-6, 2e-6, 5e-6]
        # use current incumbent max as base for each successive target
        for di, delta in enumerate(deltas):
            if time.time() > t0 + 55.0:
                break
            c = inc_val - delta if di == 0 else inc_val - (delta - deltas[di-1]) - 0.0
            # spec: targets max-1e-6,-2e-6,-5e-6 relative to start; implement relative to current inc
            # recompute as original best_val - delta to be faithful
            c = best_val - delta
            if c >= inc_val:
                continue
            deadline = min(t0+55.0, time.time()+16.0)
            Xt, ok, S = _attempt_target(incumbent, c, deadline, max_iters=60)
            Xt = Xt/np.linalg.norm(Xt, axis=1, keepdims=True)
            v = _max_cos(Xt)
            if v < inc_val and np.all(np.isfinite(Xt)):
                incumbent = Xt.copy(); inc_val = v
                # quick LP squeeze to lock feasibility
                rem = max(1.0, deadline - time.time())
                # tiny polish
                try:
                    Xp = _lp_polish(incumbent, min(3.0, rem*0.4))
                    vp = _max_cos(Xp)
                    if vp < inc_val:
                        incumbent = Xp; inc_val = vp
                except Exception:
                    pass
            # continue only while feasibility improves: if S large (no progress), stop homotopy
            if not ok and v >= inc_val - 1e-12:
                # no improvement for this target; try next harder target? break to explore
                if di >= 1:
                    pass
    except Exception:
        pass
    # --- diverse re-jam explore to leave basin ---
    t_explore = t0 + BUDGET - 22.0
    it = 0
    while time.time() < t_explore:
        if time.time()+3.0 > t0+BUDGET:
            break
        it += 1
        r = rng.rand()
        cur = incumbent.copy()
        if r < 0.5:
            # hole surgery on incumbent
            try:
                X = cur.copy()
                G = X @ X.T; np.fill_diagonal(G, -2.0)
                mcur = G.max()
                imax = G.max(axis=1)
                cnt = (G > mcur-5e-7).sum(axis=1)
                k = 6+int(rng.randint(9))
                ow = np.argsort(imax); oc = np.argsort(cnt)
                victims = set()
                for _ in range(k):
                    u = rng.rand()
                    if u < 0.5:
                        for idx in ow[::-1]:
                            if idx not in victims:
                                victims.add(int(idx)); break
                    elif u < 0.8:
                        for idx in oc[::-1]:
                            if idx not in victims:
                                victims.add(int(idx)); break
                    else:
                        vv = int(rng.randint(n))
                        while vv in victims:
                            vv = int(rng.randint(n))
                        victims.add(vv)
                victims = sorted(victims)
                keep = np.ones(n, bool); keep[victims] = False
                Y = X[keep]
                holes = _find_holes(Y, k, rng, beta=70.0+rng.rand()*60.0)
                for vi, h in zip(victims, holes):
                    X[vi] = h
                X = X + rng.randn(n, d)*0.0035
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _balanced_quench(X, 800+int(rng.randint(400)), 25.0+rng.rand()*25.0, 280.0+rng.rand()*320.0, 0.004+rng.rand()*0.004, 0.0005, 0.03+rng.rand()*0.09, t_explore)
            except Exception:
                continue
        elif r < 0.75:
            try:
                X = rng.randn(n, d); X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _balanced_quench(X, 700, 15.0, 120.0, 0.006, 0.001, 0.08, t_explore)
                if time.time() > t_explore:
                    continue
                X = _balanced_quench(X, 800, 80.0, 450.0, 0.005, 0.0004, 0.04, t_explore)
            except Exception:
                continue
        else:
            try:
                X = cur.copy()
                G = X @ X.T; np.fill_diagonal(G, -2.0)
                imax = G.max(axis=1)
                worst = np.argsort(imax)[-24:]
                X[worst] += rng.randn(len(worst), d)*(0.07+rng.rand()*0.09)
                mask = np.ones(n, bool); mask[worst] = False
                X[mask] += rng.randn(mask.sum(), d)*0.004
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                X = _balanced_quench(X, 850, 30.0, 300.0, 0.005, 0.0005, 0.05, t_explore)
            except Exception:
                continue
        try:
            v = _max_cos(X)
            if v < inc_val:
                incumbent = X.copy(); inc_val = v
        except Exception:
            pass
    # --- final trust-region LP polish ---
    try:
        remain = (t0+BUDGET) - time.time() - 0.5
        if remain > 2.0:
            Xp = _lp_polish(incumbent, min(remain, 20.0))
            vp = _max_cos(Xp)
            if vp < inc_val and np.all(np.isfinite(Xp)):
                incumbent = Xp; inc_val = vp
    except Exception:
        pass
    incumbent = incumbent/np.linalg.norm(incumbent, axis=1, keepdims=True)
    # strict finite check + tiny shrink not needed (normalised)
    return incumbent
