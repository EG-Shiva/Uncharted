import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_perms():
    c = np.array([5, 0, 1, 2, 3, 4], dtype=int)
    r = np.array([5, 4, 3, 2, 1, 0], dtype=int)
    seen = set()
    seen.add(tuple(range(6)))
    stack = [np.arange(6, dtype=int)]
    gens = [c, r]
    # BFS closure under right multiplication
    idx = 0
    while idx < len(stack):
        p = stack[idx]
        idx += 1
        for g in gens:
            q = p[g]  # composition p o g
            t = tuple(q.tolist())
            if t not in seen:
                seen.add(t)
                stack.append(q.copy())
            # also left multiplication to be safe
            q2 = g[p]
            t2 = tuple(q2.tolist())
            if t2 not in seen:
                seen.add(t2)
                stack.append(q2.copy())
    perms = np.stack(stack, axis=0)
    # keep exactly 12, pad/truncate deterministically
    if perms.shape[0] >= 12:
        # sort lexicographically and take 12 that include identity
        # ensure identity first
        lst = [tuple(x.tolist()) for x in perms]
        # deduplicate already done
        # force identity first
        ident = tuple(range(6))
        lst.remove(ident)
        lst = [ident] + sorted(lst)[:11]
        perms = np.array(lst, dtype=int)
    else:
        # fallback: generate dihedral explicitly
        p = []
        for f in [0, 1]:
            for k in range(6):
                pp = np.empty(6, dtype=int)
                for i in range(6):
                    if f == 0:
                        pp[i] = (i - k) % 6
                    else:
                        pp[i] = (-i + k) % 6
                # y[i]=x[pp[i]] convention: need inverse? just collect unique
                t = tuple(pp.tolist())
                if t not in seen:
                    seen.add(t)
                    p.append(pp)
        perms = np.array([list(range(6))] + [x.tolist() for x in p[:11]], dtype=int)
    return perms  # (12,6), y = r[perm]

def solve():
    import time as _tm
    import numpy as _np
    from scipy.optimize import minimize as _minimize
    from scipy.optimize import linprog as _linprog
    from scipy.sparse import coo_matrix as _coo
    t0 = _tm.time()
    BUDGET = 172.0
    _np.random.seed(7)
    n, d = 249, 6
    NREP = 21
    PERMS = _build_perms()
    # ensure 12
    if PERMS.shape[0] != 12:
        # build dihedral directly
        P = []
        for flip in (0, 1):
            for k in range(6):
                pp = _np.empty(6, dtype=int)
                for i in range(6):
                    if flip == 0:
                        pp[i] = (i - k) % 6
                    else:
                        pp[i] = (-i + k) % 6
                P.append(pp)
        PERMS = _np.stack(P, axis=0)[:12]
    P12 = PERMS  # (12,6)

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if _np.all(_np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = _np.random.randn(n, d)
        a /= _np.maximum(_np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(_np.argmin(vals))].copy()
    bestVal = float(_np.min(vals))

    iu0 = _np.triu_indices(n, 1)
    I0, J0 = iu0

    def expand_Rn(Rn):
        X = _np.empty((n, d), dtype=float)
        pos = 0
        for k in range(20):
            rk = Rn[k]
            for g in range(12):
                X[pos] = rk[P12[g]]
                pos += 1
        rk = Rn[20]
        for g in range(9):
            X[pos] = rk[P12[g]]
            pos += 1
        return X

    def sym_fg_factory(tau, creg=0.05):
        tau = float(tau)
        def fg(Rflat):
            R = Rflat.reshape(NREP, d)
            rn = _np.sqrt(_np.sum(R * R, axis=1))
            rn = _np.maximum(rn, 1e-12)
            Rn = R / rn[:, None]
            X = expand_Rn(Rn)
            G = X @ X.T
            g = G[I0, J0]
            gm = _np.max(g)
            e = _np.exp((g - gm) / tau)
            S = _np.sum(e)
            Lsoft = gm + tau * _np.log(S)
            w = e / S
            # W @ X without forming full W
            Wx = _np.zeros((n, d), dtype=float)
            # w indexed by pairs
            # A_i = sum_j w_ij X_j
            # accumulate
            # use scatter via bincount-like loops vectorized:
            # for each pair (I0[k],J0[k]) weight w[k]
            # Wx[I0[k]] += w[k]*X[J0[k]] ; Wx[J0[k]] += w[k]*X[I0[k]]
            Wx[I0] += w[:, None] * X[J0]
            # need accumulation for repeated indices: use np.add.at
            # above fancy indexing overwrites duplicates, so use add.at
            # redo correctly:
            Wx[:] = 0.0
            _np.add.at(Wx, I0, w[:, None] * X[J0])
            _np.add.at(Wx, J0, w[:, None] * X[I0])
            A = Wx  # dL/dX
            # pullback to Rn
            GradRn = _np.zeros((NREP, d), dtype=float)
            pos = 0
            for k in range(20):
                gk = _np.zeros(d, dtype=float)
                for gg in range(12):
                    arow = A[pos]
                    perm = P12[gg]
                    # gk[perm[i]] += arow[i]
                    for i in range(d):
                        gk[perm[i]] += arow[i]
                    pos += 1
                GradRn[k] = gk
            gk = _np.zeros(d, dtype=float)
            for gg in range(9):
                arow = A[pos]
                perm = P12[gg]
                for i in range(d):
                    gk[perm[i]] += arow[i]
                pos += 1
            GradRn[20] = gk
            # chain to raw R + radial reg
            dot = _np.sum(GradRn * Rn, axis=1)
            GradRaw = (GradRn - dot[:, None] * Rn) / rn[:, None]
            diff = rn - 1.0
            L = Lsoft + creg * _np.sum(diff * diff)
            GradRaw = GradRaw + (2.0 * creg * diff)[:, None] * Rn
            return float(L), GradRaw.ravel()
        return fg

    def sym_polish(Rinit, taus, iters):
        Rf = Rinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if _tm.time() - t0 > BUDGET - 95:
                break
            fg = sym_fg_factory(float(tau))
            try:
                res = _minimize(fg, Rf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi * 5),
                                         'ftol': 1e-14, 'gtol': 1e-7})
                Rf = res.x
            except Exception:
                break
        R = Rf.reshape(NREP, d)
        rn = _np.maximum(_np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        Rn = R / rn
        X = expand_Rn(Rn)
        # renormalize (already unit)
        X /= _np.maximum(_np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X, Rn

    def make_full_fg(tau, creg=0.05):
        def fg(xf):
            X = xf.reshape(n, d)
            r = _np.sqrt(_np.sum(X * X, axis=1))
            r = _np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[I0, J0]
            gmax = _np.max(g)
            e = _np.exp((g - gmax) / tau)
            S = _np.sum(e)
            Lsoft = gmax + tau * _np.log(S)
            w = e / S
            Wx = _np.zeros((n, d), dtype=float)
            _np.add.at(Wx, I0, w[:, None] * Xn[J0])
            _np.add.at(Wx, J0, w[:, None] * Xn[I0])
            proj = _np.sum(Wx * Xn, axis=1)
            Grad0 = (Wx - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg * _np.sum(diff * diff)
            Grad = Grad0 + (2.0 * creg * diff)[:, None] * Xn
            return float(L), Grad.ravel()
        return fg

    def full_soft(Xinit, taus, iters):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if _tm.time() - t0 > BUDGET - 12:
                break
            fg = make_full_fg(float(tau))
            try:
                res = _minimize(fg, Xf, jac=True, method='L-BFGS-B',
                                options={'maxiter': int(mi), 'maxfun': int(mi * 4),
                                         'ftol': 1e-14, 'gtol': 1e-8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = _np.maximum(_np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm / rn).ravel()
        Xo = Xf.reshape(n, d)
        Xo /= _np.maximum(_np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def lp_polish(Xinit, seconds, radius_init=2e-3):
        X = Xinit.copy()
        X /= _np.maximum(_np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        best = _max_cos(X)
        bestXl = X.copy()
        radius = float(radius_init)
        start = _tm.time()
        iu = _np.triu_indices(n, 1)
        while _tm.time() - start < seconds and radius > 1e-14:
            if _tm.time() - t0 > BUDGET - 2:
                break
            Gfull = X @ X.T
            c = Gfull[iu]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 2e-7)
            p0 = int(_np.sum(sel))
            if p0 > 5000:
                idx = _np.argpartition(c, -5000)[-5000:]
                sel = _np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p0 = 5000
            if p0 < 10:
                sel = c > m - 1e-6
                p0 = int(_np.sum(sel))
            I = iu[0][sel]; J = iu[1][sel]; p = p0
            csel = c[sel]
            try:
                rows = _np.repeat(_np.arange(p), 2 * d + 1)
                colsA = _np.concatenate([I[:, None] * d + _np.arange(d),
                                         J[:, None] * d + _np.arange(d),
                                         _np.full((p, 1), n * d)], 1).ravel()
                valsA = _np.concatenate([X[J], X[I], -_np.ones((p, 1))], 1).ravel()
                A_ub = _coo((valsA, (rows, colsA)), shape=(p, n * d + 1)).tocsr()
                erows = _np.repeat(_np.arange(n), d)
                ecols = _np.arange(n * d)
                A_eq = _coo((X.ravel(), (erows, ecols)), shape=(n, n * d + 1)).tocsr()
                obj = _np.zeros(n * d + 1); obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(None, None)]
                res = _linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                               b_eq=_np.zeros(n), bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n, d)
            y = X + dx
            y /= _np.maximum(_np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                bestXl = y.copy()
                radius = min(radius * 1.5, 0.015)
            else:
                radius *= 0.5
        return bestXl, best

    # ---- symmetric search ----
    sym_cands = []
    # restart 0,1: random reps
    rng = _np.random.RandomState(11)
    taus_sym = [0.02, 0.01, 0.005, 0.002, 0.0009]
    iters_sym = [60, 60, 70, 70, 80]
    taus_sym2 = [0.01, 0.004, 0.0012]
    iters_sym2 = [50, 60, 70]
    inits = []
    Rr1 = rng.randn(NREP, d); Rr1 /= _np.maximum(_np.linalg.norm(Rr1, axis=1, keepdims=True), 1e-12)
    inits.append(Rr1)
    Rr2 = _np.random.RandomState(23).randn(NREP, d)
    Rr2 /= _np.maximum(_np.linalg.norm(Rr2, axis=1, keepdims=True), 1e-12)
    inits.append(Rr2)
    # restart from incumbent: sample 21 points
    try:
        pick = _np.random.RandomState(31).choice(n, size=NREP, replace=False)
        Rr3 = bestX[pick].copy() + 0.01 * _np.random.RandomState(32).randn(NREP, d)
        Rr3 /= _np.maximum(_np.linalg.norm(Rr3, axis=1, keepdims=True), 1e-12)
        inits.append(Rr3)
    except Exception:
        pass
    # frame-seeded: last rep special? first init with frame-ish reps
    try:
        Rf = _np.random.RandomState(45).randn(NREP, d)
        Rf /= _np.maximum(_np.linalg.norm(Rf, axis=1, keepdims=True), 1e-12)
        # force rep0 to all-ones direction to induce small orbit behavior
        Rf[20] = _np.ones(d) / _np.sqrt(d)
        inits.append(Rf)
    except Exception:
        pass

    for ri, Rinit in enumerate(inits):
        if _tm.time() - t0 > BUDGET - 100:
            break
        if ri < 2:
            Xsym, _ = sym_polish(Rinit, taus_sym, iters_sym)
        else:
            Xsym, _ = sym_polish(Rinit, taus_sym2, iters_sym2)
        v = _max_cos(Xsym)
        sym_cands.append((v, Xsym))
        # keep global best if symmetric already better (unlikely)
        if v < bestVal:
            bestVal = v
            bestX = Xsym.copy()

    sym_cands.sort(key=lambda t: t[0])
    # ---- break symmetry with controlled perturbations + LP jam ----
    sigmas = [0.005, 0.02, 0.05]
    # use top-2 symmetric candidates
    rel_cands = sym_cands[:2] if len(sym_cands) >= 2 else sym_cands
    for (vsym, Xsym) in list(rel_cands):
        for sig in sigmas:
            if _tm.time() - t0 > BUDGET - 30:
                break
            # deterministic perturbation
            pert = _np.random.randn(n, d)
            Xt = Xsym + float(sig) * pert
            Xt /= _np.maximum(_np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            # cheap full soft polish
            Xq = full_soft(Xt, [0.006, 0.0022], [40, 55])
            vq = _max_cos(Xq)
            # LP jam each released configuration
            Xl, vl = lp_polish(Xq, 1.2, radius_init=2e-3)
            if vl < vq:
                Xq, vq = Xl, vl
            else:
                # still try short LP with bigger radius for large sig
                if sig >= 0.02:
                    Xl2, vl2 = lp_polish(Xq, 0.8, radius_init=5e-3)
                    if vl2 < vq:
                        Xq, vq = Xl2, vl2
            if vq < bestVal:
                bestVal = vq
                bestX = Xq.copy()

    # ---- final squeeze of incumbent (retain route if symmetric worse) ----
    if _tm.time() - t0 < BUDGET - 20:
        Xp = full_soft(bestX, [0.005, 0.002, 0.0009], [50, 60, 70])
        vp = _max_cos(Xp)
        if vp < bestVal:
            bestVal = vp
            bestX = Xp.copy()
    if _tm.time() - t0 < BUDGET - 8:
        Xl, vl = lp_polish(bestX, min(6.0, max(1.0, (t0 + BUDGET - 2) - _tm.time())))
        if vl < bestVal:
            bestVal = vl
            bestX = Xl.copy()

    bestX /= _np.maximum(_np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    # shrink by tiny margin guard + finite check
    bestX = _np.where(_np.isfinite(bestX), bestX, 0.0)
    nn = _np.linalg.norm(bestX, axis=1, keepdims=True)
    nn = _np.maximum(nn, 1e-12)
    bestX = bestX / nn
    return bestX
