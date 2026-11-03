import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    import time
    t0 = time.time()
    BUDGET = 170.0
    def remain():
        return BUDGET - (time.time() - t0)
    n, d = 249, 6
    rng = np.random.default_rng(0)
    I0, J0 = np.triu_indices(n, 1)

    # incumbent fallback
    cands0 = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands0.append(a)
        except Exception:
            pass
    if cands0:
        vals0 = [_max_cos(a) for a in cands0]
        bi = int(np.argmin(vals0))
        incX = cands0[bi].copy()
        incVal = float(vals0[bi])
    else:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        incX = a
        incVal = _max_cos(a)

    globX = incX.copy()
    globVal = incVal

    from scipy.optimize import minimize as _min

    CREG = 0.5
    CSCALE = 0.70

    def power_fg_factory(p):
        p = float(p)
        c = CSCALE
        def fg(xf):
            X = xf.reshape(n, d)
            r = np.sqrt(np.sum(X * X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            # scale and positive part
            Gs = G / c
            np.maximum(Gs, 0.0, out=Gs)
            np.fill_diagonal(Gs, 0.0)
            # powers
            if p == 1:
                Pw = Gs
                Wm = np.ones_like(Gs)
            else:
                # W = (p/c) * Gs^(p-1), Lpart = Gs^p
                # compute safely
                with np.errstate(over='ignore', under='ignore'):
                    Pw = np.power(Gs, p, out=np.empty_like(Gs)) if False else Gs ** p
                    Wm = Gs ** (p - 1.0)
            np.fill_diagonal(Pw, 0.0)
            np.fill_diagonal(Wm, 0.0)
            S = float(np.sum(Pw)) * 0.5
            Wm *= (p / c)
            A = Wm @ Xn
            proj = np.sum(A * Xn, axis=1)
            G0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = S + CREG * float(np.sum(diff * diff))
            Gr = G0 + (2.0 * CREG * diff)[:, None] * Xn
            return float(L), Gr.ravel()
        return fg

    def softmax_fg_factory(tau):
        tau = float(tau)
        def fg(xf):
            X = xf.reshape(n, d)
            r = np.sqrt(np.sum(X * X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0, J0]
            gm = float(np.max(g))
            e = np.exp((g - gm) / tau)
            S = float(np.sum(e))
            Lsoft = gm + tau * np.log(S)
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[I0, J0] = w
            W = W + W.T
            A = W @ Xn
            proj = np.sum(A * Xn, axis=1)
            G0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + CREG * float(np.sum(diff * diff))
            Gr = G0 + (2.0 * CREG * diff)[:, None] * Xn
            return float(L), Gr.ravel()
        return fg

    def run_minimize(Xinit, fg, maxiter, gtol=1e-4):
        xf = Xinit.ravel().copy()
        try:
            res = _min(fg, xf, jac=True, method='L-BFGS-B',
                       options={'maxiter': int(maxiter), 'maxfun': int(maxiter * 8),
                                'ftol': 1e-12, 'gtol': float(gtol), 'maxcor': 5})
            xf = res.x
        except Exception:
            pass
        Xo = xf.reshape(n, d)
        rn = np.linalg.norm(Xo, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xo = Xo / rn
        return Xo

    def rand_rot(key_rng):
        Q, _ = np.linalg.qr(key_rng.standard_normal(size=(6, 6)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        return Q

    # ---------- seed builders ----------
    def greedy_extend(Xinit, k_need, pool,):
        # Xinit (m,6) normalized, pool (C,6) normalized
        C = pool.shape[0]
        if Xinit.shape[0] == 0:
            curmax = np.full(C, -2.0)
        else:
            D = pool @ Xinit.T
            curmax = D.max(axis=1)
        used = np.zeros(C, dtype=bool)
        new_pts = []
        for _ in range(k_need):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = pool[b].copy()
            new_pts.append(pt)
            dd = pool @ pt
            # max update
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        return np.stack(new_pts, axis=0) if new_pts else np.zeros((0, d))

    def D6_set():
        pts = []
        for i in range(6):
            for j in range(i + 1, 6):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        v = np.zeros(6)
                        v[i] = si / np.sqrt(2.0)
                        v[j] = sj / np.sqrt(2.0)
                        pts.append(v)
        return np.stack(pts, axis=0)  # 60x6

    def ternary_pool():
        # all -1,0,1 except 0, normalized
        vals = [-1.0, 0.0, 1.0]
        pts = []
        for a0 in vals:
            for a1 in vals:
                for a2 in vals:
                    for a3 in vals:
                        for a4 in vals:
                            for a5 in vals:
                                if a0 == 0 and a1 == 0 and a2 == 0 and a3 == 0 and a4 == 0 and a5 == 0:
                                    continue
                                v = np.array([a0, a1, a2, a3, a4, a5], dtype=float)
                                v /= np.linalg.norm(v)
                                pts.append(v)
        return np.stack(pts, axis=0)

    seeds = []
    # helper to add with normalization
    def add_seed(X):
        X = np.asarray(X, dtype=float)
        if X.shape != (n, d):
            return
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        if np.all(np.isfinite(X)):
            seeds.append(X)

    # A: random uniform starts (8)
    for s in range(8):
        rs = np.random.default_rng(1000 + s)
        X = rs.standard_normal(size=(n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        add_seed(X)

    # B: low-discrepancy Halton -> normal (3)
    try:
        from scipy.stats.qmc import Halton as _Halton
        from scipy.special import ndtri as _ndtri
        h_ok = True
    except Exception:
        h_ok = False
    if h_ok:
        try:
            for s in range(3):
                eng = _Halton(d=6, scramble=False)
                eng.fast_forward(1 + s * 997)
                u = eng.random(n=n + s * 13)
                # take last n
                u = u[-n:, :]
                u = np.clip(u, 1e-9, 1 - 1e-9)
                z = _ndtri(u)
                z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
                # random rotation to diversify
                rs = np.random.default_rng(2000 + s)
                Q = rand_rot(rs)
                z = z @ Q.T
                add_seed(z)
        except Exception:
            pass

    # C: rotated D6 + greedy extension (4)
    try:
        D6 = D6_set()
        for s in range(4):
            rs = np.random.default_rng(3000 + s)
            Q = rand_rot(rs)
            Dr = D6 @ Q.T
            m0 = Dr.shape[0]
            need = n - m0
            pool = rs.standard_normal(size=(18000, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            ext = greedy_extend(Dr, need, pool)
            X = np.vstack([Dr, ext])
            # shuffle order deterministically
            perm = rs.permutation(n)
            X = X[perm]
            add_seed(X)
    except Exception:
        pass

    # D: cross-polytope + layers rotated + greedy (2)
    try:
        for s in range(2):
            rs = np.random.default_rng(4000 + s)
            Q = rand_rot(rs)
            Cross = np.zeros((12, 6))
            for i in range(6):
                Cross[2 * i, i] = 1.0
                Cross[2 * i + 1, i] = -1.0
            Cr = Cross @ Q.T
            need = n - 12
            pool = rs.standard_normal(size=(18000, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            ext = greedy_extend(Cr, need, pool)
            X = np.vstack([Cr, ext])
            perm = rs.permutation(n)
            X = X[perm]
            add_seed(X)
    except Exception:
        pass

    # E: pure greedy from empty (2)
    try:
        for s in range(2):
            rs = np.random.default_rng(5000 + s)
            pool = rs.standard_normal(size=(22000, d))
            pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
            empty = np.zeros((0, d))
            X = greedy_extend(empty, n, pool)
            add_seed(X)
    except Exception:
        pass

    # F: ternary subset greedy (1)
    try:
        rs = np.random.default_rng(6000)
        Tpool = ternary_pool()  # 728
        # random rotation of ternary pool to break axis alignment? keep one unrotated, one rotated
        # unrotated greedy select 249 from 728
        empty = np.zeros((0, d))
        Xsel = greedy_extend(empty, n, Tpool)
        add_seed(Xsel)
        # rotated version
        Q = rand_rot(rs)
        Tpool2 = (Tpool @ Q.T)
        Xsel2 = greedy_extend(empty, n, Tpool2)
        add_seed(Xsel2)
    except Exception:
        pass

    # limit seeds to fit time: keep at most ~22
    if len(seeds) > 24:
        seeds = seeds[:24]

    # ---------- short screening homotopy ----------
    short_ps = [8, 16, 32, 64]
    short_iters = [35, 35, 35, 35]
    results = []  # (val, X)
    for si, Xs in enumerate(seeds):
        if remain() < 70:
            break
        Xc = Xs.copy()
        for p, mi in zip(short_ps, short_iters):
            if remain() < 55:
                break
            fg = power_fg_factory(p)
            # fewer iters for large seed count
            Xc = run_minimize(Xc, fg, maxiter=mi, gtol=1e-3)
        v = _max_cos(Xc)
        results.append((v, Xc.copy()))
        if v < globVal:
            globVal = v
            globX = Xc.copy()

    # ---------- deduplicate by sorted Gram ----------
    # sort results by val
    results.sort(key=lambda t: t[0])
    # precompute sorted grams
    sgrams = []
    for v, X in results:
        try:
            G = X @ X.T
            g = G[I0, J0]
            gs = np.sort(g)
            sgrams.append(gs)
        except Exception:
            sgrams.append(None)
    kept_idx = []
    # adaptive threshold to get 3
    thresh = 0.01
    for attempt in range(6):
        kept_idx = []
        for i in range(len(results)):
            if sgrams[i] is None:
                continue
            dup = False
            for j in kept_idx:
                # RMS distance of sorted grams
                diff = sgrams[i] - sgrams[j]
                rms = float(np.sqrt(np.mean(diff * diff)))
                if rms < thresh:
                    dup = True
                    break
            if not dup:
                kept_idx.append(i)
            if len(kept_idx) >= 12:
                pass
        # we want at least 3 kept; if too few, lower thresh
        if len(kept_idx) >= 3 or thresh < 1e-4:
            break
        thresh *= 0.5
    # take best 3 distinct (by val order kept is in val order)
    top3 = kept_idx[:3] if len(kept_idx) >= 1 else [0] if results else []
    # if fewer than 3, fill with best remaining
    if len(top3) < 3:
        for i in range(len(results)):
            if i not in top3:
                top3.append(i)
            if len(top3) >= 3:
                break
    # ---------- full anneal best three ----------
    for ti in top3:
        if remain() < 25:
            break
        if ti >= len(results):
            continue
        v0, X0 = results[ti]
        Xc = X0.copy()
        # continue power ladder longer
        for p, mi in [(32, 60), (64, 80), (128, 100)]:
            if remain() < 20:
                break
            fg = power_fg_factory(p)
            Xc = run_minimize(Xc, fg, maxiter=mi, gtol=1e-4)
            v = _max_cos(Xc)
            if v < globVal:
                globVal = v
                globX = Xc.copy()
        # soft-max polish low temps
        for tau, mi in [(0.008, 120), (0.004, 120), (0.002, 120), (0.001, 120)]:
            if remain() < 12:
                break
            fg = softmax_fg_factory(tau)
            Xc = run_minimize(Xc, fg, maxiter=mi, gtol=1e-5)
            v = _max_cos(Xc)
            if v < globVal:
                globVal = v
                globX = Xc.copy()
        # final tiny power touch
        if remain() > 10:
            try:
                fg = power_fg_factory(128)
                Xc2 = run_minimize(Xc, fg, maxiter=40, gtol=1e-5)
                v2 = _max_cos(Xc2)
                if v2 < globVal:
                    globVal = v2
                    globX = Xc2.copy()
            except Exception:
                pass

    # fallback to incumbent if worse
    if globVal > incVal:
        globX = incX.copy()
        globVal = incVal
    # normalize strictly
    globX = globX / np.maximum(np.linalg.norm(globX, axis=1, keepdims=True), 1e-12)
    # shrink margin tiny (no-op, keep unit)
    return globX
