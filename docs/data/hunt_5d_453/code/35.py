import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    SYM_END = t0 + 32.0
    POLISH_END = t0 + 164.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    # ---------- symmetric C151 construction : R5 = R2+R2+R ----------
    # 453 = 3*151, group acts by 2pi/151 on plane1 and 2pi*a/151 on plane2
    # per orbit params: z, t(split), th0, ph0  -> 12 params total
    import math
    K = 151
    ks = np.arange(K)
    ang1 = 2.0 * math.pi * ks / K
    c1b = np.cos(ang1); s1b = np.sin(ang1)
    iu_r, iu_c = np.triu_indices(n, 1)
    twists = [1, 10, 20, 37, 50]
    TWOPI = 2.0 * math.pi

    def build_sym(pvec, c2b, s2b):
        # pvec length 12 -> (3,4)
        P = np.asarray(pvec, dtype=np.float64).reshape(3, 4)
        X = np.empty((453, 5), dtype=np.float64)
        for oi in range(3):
            z = float(P[oi, 0]); tt = float(P[oi, 1])
            th0 = float(P[oi, 2]); ph0 = float(P[oi, 3])
            # clip to feasible
            if z > 0.98: z = 0.98
            if z < -0.98: z = -0.98
            s = math.sqrt(max(0.0, 1.0 - z * z))
            r1 = s * math.cos(tt); r2 = s * math.sin(tt)
            cth0 = math.cos(th0); sth0 = math.sin(th0)
            cph0 = math.cos(ph0); sph0 = math.sin(ph0)
            cth = cth0 * c1b - sth0 * s1b
            sth = sth0 * c1b + cth0 * s1b
            cph = cph0 * c2b - sph0 * s2b
            sph = sph0 * c2b + cph0 * s2b
            base = oi * K
            X[base:base+K, 0] = r1 * cth
            X[base:base+K, 1] = r1 * sth
            X[base:base+K, 2] = r2 * cph
            X[base:base+K, 3] = r2 * sph
            X[base:base+K, 4] = z
        # renormalize (safety)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
        return X

    sym_best_X = None
    sym_best_V = float("inf")
    sym_best_a = None

    try:
        from scipy.optimize import differential_evolution
        bounds_one = [(-0.90, 0.90), (0.05, 1.52), (0.0, TWOPI), (0.0, TWOPI)]
        bounds = bounds_one * 3  # 12
        for ai, a in enumerate(twists):
            if time.time() > SYM_END - 2.0:
                break
            ang2 = 2.0 * math.pi * float(a) * ks / K
            c2b = np.cos(ang2); s2b = np.sin(ang2)
            tau = 0.005
            def obj(pvec):
                try:
                    X = build_sym(pvec, c2b, s2b)
                    G = X @ X.T
                    vals = G[iu_r, iu_c]
                    m = float(np.max(vals))
                    # smooth LogSumExp surrogate
                    diff = (vals - m) / tau
                    # avoid overflow: diff<=0
                    with np.errstate(over='ignore', invalid='ignore'):
                        e = np.exp(diff)
                    lm = float(np.mean(e))
                    if not np.isfinite(lm) or lm <= 0:
                        return m
                    return m + tau * float(np.log(lm))
                except Exception:
                    return 2.0
            # budget per twist
            try:
                res = differential_evolution(
                    obj, bounds, maxiter=22, popsize=7, tol=0.002,
                    seed=0, polish=False, updating='deferred',
                    strategy='best1bin', seed_rng=None if False else 0)
                pv = np.asarray(res.x, dtype=np.float64)
            except Exception:
                # fallback random search
                pv = None
                bv = float("inf")
                for _ in range(400):
                    if time.time() > SYM_END - 1.0:
                        break
                    cand = np.empty(12)
                    for k2 in range(3):
                        cand[k2*4+0] = rng.uniform(-0.9, 0.9)
                        cand[k2*4+1] = rng.uniform(0.05, 1.52)
                        cand[k2*4+2] = rng.uniform(0, TWOPI)
                        cand[k2*4+3] = rng.uniform(0, TWOPI)
                    v = obj(cand)
                    if v < bv:
                        bv = v; pv = cand.copy()
                if pv is None:
                    continue
            try:
                Xs = build_sym(pv, c2b, s2b)
                ve = _max_cos(Xs)
            except Exception:
                continue
            if ve < sym_best_V:
                sym_best_V = ve
                sym_best_X = Xs.copy()
                sym_best_a = a
            if time.time() > SYM_END:
                break
    except Exception:
        pass

    # contact / rattler stats comparison (for basin choice, cheap)
    def rattlers(X, thr=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = (G >= m - thr).astype(np.int32)
        cnt = np.sum(C, axis=1)
        return int(np.sum(cnt == 0)), cnt, m
    try:
        r_inc, _, _ = rattlers(bestX)
    except Exception:
        r_inc = n
    sym_polished = None
    sym_pol_V = float("inf")
    if sym_best_X is not None:
        try:
            r_sym, _, _ = rattlers(sym_best_X)
        except Exception:
            r_sym = n
    else:
        r_sym = n

    # ---------- full-space softmax annealing ----------
    def smooth_run(x, N, tau0, tau1, lr0, lr1, mu, deadline):
        v = np.zeros_like(x)
        try:
            ltau = np.log(tau1 / tau0)
        except Exception:
            ltau = 0.0
        try:
            llr = np.log(lr1 / lr0)
        except Exception:
            llr = 0.0
        for it in range(N):
            if it % 40 == 0 and time.time() > deadline:
                break
            frac = it / max(1, N - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            with np.errstate(over='ignore', invalid='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            # zero out non-finite
            if not np.all(np.isfinite(P)):
                P = np.where(np.isfinite(P), P, 0.0)
            Grad = P @ x
            dot = np.sum(Grad * x, axis=1, keepdims=True)
            Gt = Grad - dot * x
            v = mu * v + (1.0 - mu) * Gt
            x = x - lr * v
            xn = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
            xn[xn == 0] = 1.0
            x = x / xn
        return x

    def find_hole(Xkeep, n_cand=5000):
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xkeep.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        h = C[bi].copy()
        for _ in range(12):
            dots = Xkeep @ h
            ci = int(np.argmax(dots))
            c = float(dots[ci])
            xc = Xkeep[ci]
            tt = xc - c * h
            tn = float(np.linalg.norm(tt))
            if tn < 1e-12:
                break
            tt /= tn
            h = h - 0.03 * tt
            nh = float(np.linalg.norm(h))
            if nh == 0 or not np.isfinite(nh):
                break
            h /= nh
        return h

    # break symmetry and polish symmetric optimum first (genuinely different basin)
    if sym_best_X is not None:
        try:
            y = sym_best_X.copy()
            nz = rng.standard_normal((n, d)) * 0.0015
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            y = y + nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_run(y, 700, 0.004, 0.0006, 0.004, 0.0004, 0.85, POLISH_END)
            vv = _max_cos(y)
            sym_polished = y.copy()
            sym_pol_V = vv
            if vv < bestV:
                bestV = vv
                bestX = y.copy()
        except Exception:
            pass
        # partially symmetrised incumbent: average nearest points onto nearby orbits
        # (cheap approximation: snap a subset to the symmetric template then re-anneal)
        try:
            if time.time() < POLISH_END - 5.0:
                # for each symmetric point find nearest incumbent point and average
                Gc = bestX @ sym_best_X.T  # 453x453
                # greedy assignment: for each sym point pick best unused incumbent
                used = np.zeros(n, dtype=bool)
                avg = bestX.copy()
                for k2 in range(n):
                    row = Gc[:, k2].copy()
                    row[used] = -2.0
                    ii = int(np.argmax(row))
                    used[ii] = True
                # simple blend: halfway between incumbent and its matched sym point
                # build matched sym order
                order = np.empty(n, dtype=int)
                used2 = np.zeros(n, dtype=bool)
                for k2 in range(n):
                    row = Gc[:, k2].copy()
                    row[used2] = -2.0
                    ii = int(np.argmax(row))
                    used2[ii] = True
                    order[ii] = k2
                blend = 0.5 * bestX + 0.5 * sym_best_X[order]
                blend /= np.linalg.norm(blend, axis=1, keepdims=True)
                blend = smooth_run(blend, 300, 0.004, 0.0008, 0.003, 0.0004, 0.85, POLISH_END)
                bv2 = _max_cos(blend)
                if bv2 < bestV:
                    bestV = bv2
                    bestX = blend.copy()
                # keep sym_polished as alternate basin even if worse
                if sym_polished is None:
                    sym_polished = blend.copy()
                    sym_pol_V = bv2
        except Exception:
            pass

    # main basin hopping alternating basins
    it = 0
    taus = [(0.004, 0.0006), (0.007, 0.0008), (0.0025, 0.0004)]
    lrs = [(0.004, 0.0004), (0.006, 0.0005), (0.0025, 0.0003)]
    sigmas = [0.004, 0.007, 0.0025, 0.009]
    use_sym = (sym_polished is not None)
    while time.time() < POLISH_END - 0.8:
        it += 1
        if POLISH_END - time.time() < 0.7:
            break
        ti = it % len(taus)
        tau0, tau1 = taus[ti]
        lr0, lr1 = lrs[ti % len(lrs)]
        sigma = float(sigmas[it % len(sigmas)])
        # choose base basin: every 5th from symmetric basin
        base = None
        try:
            if use_sym and (it % 5 == 0):
                base = sym_polished.copy()
                # keep sym basin alive: shake it
                nz = rng.standard_normal((n, d)) * sigma
                nz -= np.sum(nz * base, axis=1, keepdims=True) * base
                base = base + nz
                base /= np.linalg.norm(base, axis=1, keepdims=True)
                inner_end = min(time.time() + 2.0, POLISH_END)
                base = smooth_run(base, 220, tau0, tau1, lr0, lr1, 0.86, inner_end)
                nv = _max_cos(base)
                # update sym basin if improved
                if nv < sym_pol_V:
                    sym_pol_V = nv
                    sym_polished = base.copy()
                if nv < bestV:
                    bestV = nv
                    bestX = base.copy()
                continue
        except Exception:
            pass
        # otherwise hop from global best
        y = bestX.copy()
        op = it % 3
        try:
            if op == 0:
                nz = rng.standard_normal((n, d)) * (sigma * 0.45)
                nz -= np.sum(nz * y, axis=1, keepdims=True) * y
                y = y + nz
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = smooth_run(y, 220, tau0, tau1, lr0, lr1, 0.87, min(time.time()+2.0, POLISH_END))
            elif op == 1:
                # shake worst neighbourhood
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                wi = flat // n; wj = flat % n
                row = G[wi].copy()
                k = 10
                idx = np.argpartition(row, -k)[-k:]
                sset = set([int(wi), int(wj)])
                for tt2 in idx:
                    sset.add(int(tt2))
                for _ in range(4):
                    sset.add(int(rng.integers(0, n)))
                sidx = np.array(list(sset), dtype=int)
                nz = rng.standard_normal((len(sidx), d)) * sigma
                for a2, idx2 in enumerate(sidx):
                    vv2 = nz[a2] - float(np.dot(nz[a2], y[idx2])) * y[idx2]
                    y[idx2] = y[idx2] + vv2
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = smooth_run(y, 220, tau0, tau1, lr0, lr1, 0.87, min(time.time()+2.0, POLISH_END))
            else:
                # single hole reinsertion of worst point
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                wi = flat // n; wj = flat % n
                rm = np.max(G, axis=1)
                victim = int(wi) if rm[wi] >= rm[wj] else int(wj)
                mask = np.ones(n, dtype=bool)
                mask[victim] = False
                Xe = y[mask]
                h = find_hole(Xe, n_cand=5000)
                y[victim] = h
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = smooth_run(y, 260, tau0, tau1, lr0, lr1, 0.87, min(time.time()+2.2, POLISH_END))
        except Exception:
            continue
        try:
            nv = _max_cos(y)
        except Exception:
            continue
        if nv < bestV:
            bestV = nv
            bestX = y.copy()

    # final micro-shave of worst pair
    try:
        eps = 5.0e-7
        cnt = 0
        while cnt < 2500 and time.time() < T_END - 0.8:
            cnt += 1
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            flat = int(np.argmax(G))
            i = flat // n; j = flat % n
            xi = bestX[i].copy(); xj = bestX[j].copy()
            cij = float(xi @ xj)
            cij = max(-1.0, min(1.0, cij))
            ti_ = xj - cij * xi
            ni = float(np.linalg.norm(ti_))
            if ni < 1e-12:
                break
            ti_ /= ni
            tj_ = xi - cij * xj
            nj = float(np.linalg.norm(tj_))
            if nj < 1e-12:
                break
            tj_ /= nj
            improved = False
            for scale in (eps, eps * 0.5):
                vi = xi - scale * ti_; vi /= (np.linalg.norm(vi) + 1e-30)
                vj = xj - scale * tj_; vj /= (np.linalg.norm(vj) + 1e-30)
                di = bestX @ vi; di[i] = -2.0; di[j] = -2.0
                dj = bestX @ vj; dj[i] = -2.0; dj[j] = -2.0
                mc = m
                # rest max without i,j
                G[i, :] = -2.0; G[:, i] = -2.0; G[j, :] = -2.0; G[:, j] = -2.0
                mr = float(np.max(G))
                mc = mr
                a = float(np.max(di))
                if a > mc: mc = a
                b = float(np.max(dj))
                if b > mc: mc = b
                c2 = float(vi @ vj)
                if c2 > mc: mc = c2
                if mc < m - 1e-15:
                    bestX[i] = vi; bestX[j] = vj
                    bestV = mc
                    improved = True
                    break
            if improved:
                eps *= 0.99
            else:
                eps *= 0.7
                if eps < 5e-14:
                    break
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
