import time
import numpy as np
import itertools

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time as _time
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    t0 = _time.time()
    T_END = t0 + 172.0
    rng = _np.random.default_rng(7)
    _np.random.seed(7)
    n0, d = 249, 6

    # ---------- helpers ----------
    def row_max(X):
        G = X @ X.T
        _np.fill_diagonal(G, -2.0)
        return _np.max(G, axis=1)

    def polish_generic(Xinit, taus, iters, creg=0.06):
        nn = Xinit.shape[0]
        I0, J0 = _np.triu_indices(nn, 1)
        Xf = (Xinit / _np.maximum(_np.linalg.norm(Xinit, axis=1, keepdims=True), 1e-12)).ravel().copy()
        for tau, mi in zip(taus, iters):
            if _time.time() > T_END - 25.0:
                break
            tau = float(tau)
            def fg(xf, tau=tau):
                X = xf.reshape(nn, d)
                r = _np.sqrt(_np.sum(X*X, axis=1))
                r = _np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = _np.max(g)
                e = _np.exp((g-gm)/tau)
                S = _np.sum(e)
                Lsoft = gm + tau*_np.log(S)
                w = e / S
                W = _np.zeros((nn, nn))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = _np.sum(A*Xn, axis=1)
                G0 = (A - proj[:, None]*Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg*_np.sum(diff*diff)
                Gr = G0 + (2.0*creg*diff)[:, None]*Xn
                return float(L), Gr.ravel()
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi*3),
                                         'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(nn, d)
            rn = _np.maximum(_np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm/rn).ravel()
        Xo = Xf.reshape(nn, d)
        Xo /= _np.maximum(_np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def layered_init(nn, split=3, L=10):
        d1 = int(split); d2 = d-d1
        th = _np.linspace(0.12, _np.pi/2-0.12, L)
        X = _np.zeros((nn, d))
        perm = rng.permutation(nn)
        for idx, k in enumerate(perm):
            li = idx % L
            tt = th[li] + 0.04*rng.standard_normal()
            c, s = _np.cos(tt), _np.sin(tt)
            u = rng.standard_normal(size=(d1,)); u/= _np.linalg.norm(u)
            v = rng.standard_normal(size=(d2,)); v/= _np.linalg.norm(v)
            X[k, :d1] = c*u
            X[k, d1:] = s*v
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def random_init(nn):
        X = rng.standard_normal(size=(nn, d))
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def random_rotation():
        A = rng.standard_normal(size=(d, d))
        Q, _ = _np.linalg.qr(A)
        if _np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    # D6 roots 60 x 6
    def d6_roots():
        pts = []
        for i, j in itertools.combinations(range(6), 2):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = _np.zeros(6)
                    v[i] = si; v[j] = sj
                    v /= _np.sqrt(2.0)
                    pts.append(v)
        return _np.stack(pts, axis=0)

    # E8 roots 240 x 8
    def e8_roots():
        pts = []
        for i, j in itertools.combinations(range(8), 2):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = _np.zeros(8)
                    v[i] = si; v[j] = sj
                    pts.append(v)
        for mask in range(256):
            nminus = bin(mask).count("1")
            if nminus % 2 == 1:
                continue
            v = _np.ones(8)*0.5
            for k in range(8):
                if (mask >> k) & 1:
                    v[k] = -0.5
            pts.append(v)
        return _np.stack(pts, axis=0)  # 240x8

    try:
        E8 = e8_roots()
    except Exception:
        E8 = None
    D6 = d6_roots()

    def e8_slice_init(nn):
        # random 6D projection of E8
        A = rng.standard_normal(size=(8, 6))
        Q, _ = _np.linalg.qr(A)  # 8x6 orthonormal
        Y = E8 @ Q  # 240x6
        nr = _np.linalg.norm(Y, axis=1, keepdims=True)
        nr = _np.maximum(nr, 1e-9)
        Y = Y / nr
        # need nn points: E8 gives 240, add random
        if nn <= 240:
            idx = rng.choice(240, size=nn, replace=False)
            X = Y[idx]
        else:
            X = _np.zeros((nn, d))
            X[:240] = Y
            R = rng.standard_normal(size=(nn-240, d))
            R /= _np.linalg.norm(R, axis=1, keepdims=True)
            X[240:] = R
        # random global rotation + small perturb
        Q6 = random_rotation()
        X = X @ Q6.T
        X += 0.01*rng.standard_normal(size=X.shape)
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def d6_init(nn):
        Q6 = random_rotation()
        base = D6 @ Q6.T  # 60
        X = _np.zeros((nn, d))
        X[:60] = base
        R = rng.standard_normal(size=(nn-60, d))
        R /= _np.linalg.norm(R, axis=1, keepdims=True)
        X[60:] = R
        X += 0.015*rng.standard_normal(size=X.shape)
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def pattern_init(nn, a=1.0, b=0.65, c=0.35):
        vals = _np.array([a, b, c], dtype=float)
        allv = []
        for pos in itertools.combinations(range(6), 3):
            for perm in itertools.permutations([0,1,2]):
                # to avoid 6x duplicates, only unique perm assignments; just use all
                for s0 in (1.0,-1.0):
                    for s1 in (1.0,-1.0):
                        for s2 in (1.0,-1.0):
                            v = _np.zeros(6)
                            v[pos[0]] = s0*vals[perm[0]]
                            v[pos[1]] = s1*vals[perm[1]]
                            v[pos[2]] = s2*vals[perm[2]]
                            allv.append(v)
        allv = _np.stack(allv, axis=0)
        allv /= _np.linalg.norm(allv, axis=1, keepdims=True)
        # dedupe roughly by random choice
        idx = rng.choice(allv.shape[0], size=min(nn, allv.shape[0]), replace=False)
        X = allv[idx]
        if nn > X.shape[0]:
            R = rng.standard_normal(size=(nn-X.shape[0], d))
            R /= _np.linalg.norm(R, axis=1, keepdims=True)
            X = _np.vstack([X, R])
        else:
            # subsample exactly nn (already)
            pass
        if X.shape[0] > nn:
            X = X[:nn]
        Q6 = random_rotation()
        X = X @ Q6.T
        X += 0.01*rng.standard_normal(size=X.shape)
        X /= _np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def refine_hole(Xkeep, pt0, tau=0.004):
        # optimize single point on sphere via unconstrained y
        y = pt0.copy()
        nrm0 = max(_np.linalg.norm(y), 1e-12)
        y = y / nrm0
        yf = y.copy()
        def fg(yf):
            r = max(_np.sqrt(_np.sum(yf*yf)), 1e-12)
            u = yf / r
            dots = Xkeep @ u
            gm = _np.max(dots)
            e = _np.exp((dots-gm)/tau)
            S = _np.sum(e)
            L = gm + tau*_np.log(S)
            w = e / S
            # grad w.r.t u
            g_u = Xkeep.T @ w  # 6
            # project to tangent then chain
            g_u_t = g_u - _np.dot(g_u, u)*u
            g_y = g_u_t / r
            return float(L), g_y
        try:
            res = _minimize(lambda yf: fg(yf), yf, jac=True, method='L-BFGS-B',
                            options={'maxiter': 80, 'ftol':1e-12, 'gtol':1e-7})
            yf = res.x
        except Exception:
            pass
        yf = yf / max(_np.linalg.norm(yf), 1e-12)
        return yf

    def largest_hole(Xkeep, pool=15000, do_refine=True):
        C = rng.standard_normal(size=(pool, d))
        C /= _np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xkeep.T
        cm = D.max(axis=1)
        order = _np.argsort(cm)
        best = C[order[0]]
        bestv = float(cm[order[0]])
        if do_refine:
            # refine top-2
            cand_best = best
            cand_bestv = bestv
            for t in [order[0], order[1]]:
                pt = refine_hole(Xkeep, C[t])
                v = float(_np.max(Xkeep @ pt))
                if v < cand_bestv:
                    cand_bestv = v
                    cand_best = pt
            return cand_best
        return best

    def top_set(X, K=1500):
        nn = X.shape[0]
        I0, J0 = _np.triu_indices(nn, 1)
        G = X @ X.T
        g = G[I0, J0]
        idx = _np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(_np.int64)*nn + J0[idx].astype(_np.int64)
        return set(codes.tolist())

    # ---------- load incumbents ----------
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n0, d):
                an = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        cands.append(random_init(n0))
    vals0 = [_max_cos(a) for a in cands]
    bestX = cands[int(_np.argmin(vals0))].copy()
    bestVal = float(_np.min(vals0))

    SHORT_T = [0.015, 0.005, 0.0018]
    SHORT_I = [45, 55, 65]
    MED_T = [0.012, 0.004, 0.0012]
    MED_I = [80, 100, 120]

    results = []  # (val, X)

    # include incumbent polished shortly later, first build seed list
    seeds = []
    seeds.append(("incumb", bestX.copy()))
    seeds.append(("rand1", random_init(n0)))
    seeds.append(("layer3-L10", layered_init(n0, split=3, L=10)))
    seeds.append(("layer2-L12", layered_init(n0, split=2, L=12)))
    seeds.append(("e8-a", e8_slice_init(n0)))
    seeds.append(("e8-b", e8_slice_init(n0)))
    seeds.append(("d6", d6_init(n0)))
    seeds.append(("pat1", pattern_init(n0, a=1.0, b=0.68, c=0.38)))
    seeds.append(("pat2", pattern_init(n0, a=1.0, b=0.52, c=0.27)))

    # race short polishes
    for name, S in seeds:
        if _time.time() > t0 + 62:
            break
        try:
            P = polish_generic(S, SHORT_T, SHORT_I)
            v = _max_cos(P)
        except Exception:
            continue
        results.append((v, P))
        if v < bestVal:
            bestVal = v; bestX = P.copy()

    # ---------- cardinality homotopy: grow from 236 ----------
    try:
        if _time.time() < t0 + 95:
            Ng = 236
            # start from layered 236 for novelty
            G0 = layered_init(Ng, split=3, L=11)
            G0 = polish_generic(G0, MED_T, MED_I)
            cur = G0
            for target in range(Ng+1, n0+1):
                if _time.time() > t0 + 100:
                    break
                pt = largest_hole(cur, pool=15000, do_refine=True)
                cur = _np.vstack([cur, pt])
                # short re-jam
                cur = polish_generic(cur, [0.006, 0.0018], [55, 70])
            if cur.shape[0] == n0:
                v = _max_cos(cur)
                results.append((v, cur.copy()))
                if v < bestVal:
                    bestVal = v; bestX = cur.copy()
    except Exception:
        pass

    # ---------- cardinality homotopy: shrink from 256 ----------
    try:
        if _time.time() < t0 + 120:
            Nb = 257
            B0 = layered_init(Nb, split=3, L=9)
            # mix random for diversity
            B0 += 0.005*rng.standard_normal(size=B0.shape)
            B0 /= _np.linalg.norm(B0, axis=1, keepdims=True)
            B0 = polish_generic(B0, MED_T, MED_I)
            cur = B0
            while cur.shape[0] > n0:
                if _time.time() > t0 + 122:
                    break
                rm = row_max(cur)
                # delete worst point
                del_i = int(_np.argmax(rm))
                mask = _np.ones(cur.shape[0], dtype=bool); mask[del_i]=False
                cur = cur[mask]
                cur = polish_generic(cur, [0.006, 0.0018], [50, 65])
            if cur.shape[0] == n0:
                v = _max_cos(cur)
                results.append((v, cur.copy()))
                if v < bestVal:
                    bestVal = v; bestX = cur.copy()
    except Exception:
        pass

    # ---------- select diverse finalists ----------
    results.sort(key=lambda t: t[0])
    finalists = []
    kept_sets = []
    for v, X in results:
        s = top_set(X, K=1500)
        div = True
        for ks in kept_sets:
            inter = len(s & ks); union = len(s | ks)
            jac = inter/max(union,1)
            if jac > 0.55:  # too similar
                div = False; break
        if div or len(finalists) < 2:
            finalists.append((v, X))
            kept_sets.append(s)
        if len(finalists) >= 3:
            break
    if not finalists:
        finalists = [(bestVal, bestX)]

    # ---------- deep jam finalists ----------
    # order by score
    finalists.sort(key=lambda t: t[0])
    LONG_T = [0.008, 0.003, 0.0012, 0.0005, 0.00022]
    LONG_I = [130, 160, 200, 230, 260]
    for fi, (fv, FX) in enumerate(finalists):
        if _time.time() > T_END - 12:
            break
        # give most time to best
        if fi == 0:
            taus, iters = LONG_T, LONG_I
        else:
            taus, iters = LONG_T[:4], LONG_I[:4]
        try:
            P = polish_generic(FX, taus, iters, creg=0.05)
            v = _max_cos(P)
        except Exception:
            continue
        if v < bestVal:
            bestVal = v; bestX = P.copy()

    # final sharp jam on global best
    if _time.time() < T_END - 6:
        try:
            P = polish_generic(bestX, [0.001, 0.0004, 0.00018], [200, 230, 260], creg=0.03)
            v = _max_cos(P)
            if v < bestVal:
                bestVal = v; bestX = P.copy()
        except Exception:
            pass

    # cheap greedy subgradient squeeze
    try:
        deadline = T_END - 1.0
        cur2 = bestX.copy(); cur2v = bestVal
        step = 2e-3
        while _time.time() < deadline and step > 5e-13:
            G = cur2 @ cur2.T
            _np.fill_diagonal(G, -2.0)
            m = float(_np.max(G))
            window = max(10.0*step, 2e-4)
            mask = G > (m - window)
            F = mask.astype(float) @ cur2
            dot = _np.sum(F*cur2, axis=1)
            Ft = F - dot[:, None]*cur2
            trial = cur2 - step*Ft
            trial /= _np.maximum(_np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            G2 = trial @ trial.T
            _np.fill_diagonal(G2, -2.0)
            v = float(_np.max(G2))
            if v < cur2v:
                cur2, cur2v = trial, v
                step = min(step*1.3, 0.01)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        if cur2v < bestVal:
            bestX, bestVal = cur2, cur2v
    except Exception:
        pass

    bestX /= _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
