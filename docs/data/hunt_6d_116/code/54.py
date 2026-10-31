import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_sym(Y0, steps=1500, beta0=30.0, beta1=300.0, lr0=0.01, lr1=0.001):
    Y = _norm(np.array(Y0, dtype=float))
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n2 = Y.shape[0] * 2
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        X = np.vstack([Y, -Y])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        GradX = W @ X
        gY = GradX[:Y.shape[0]] - GradX[Y.shape[0]:]
        dot = np.sum(gY * Y, axis=1, keepdims=True)
        Gt = gY - dot * Y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        step = mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * Y, axis=1, keepdims=True)
        stept = step - d2 * Y
        Y = _norm(Y - lr * stept)
    return Y

def _quench(X0, steps=350, beta0=30.0, beta1=250.0, s0=0.02, s1=0.002):
    X = _norm(np.array(X0, dtype=float))
    vel = np.zeros_like(X)
    mu = 0.7
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        st = s0 + (s1 - s0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        grad = W @ X
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        nrm = np.sqrt(np.sum(gt * gt, axis=1, keepdims=True)) + 1e-15
        direc = gt / nrm
        vel = mu * vel + (1.0 - mu) * direc
        vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-15
        vdir = vel / vn
        X = _norm(X - st * vdir)
    return X

def _lp_polish(X0, deadline):
    X = _norm(np.array(X0, dtype=float))
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(X + res.x[:-1].reshape(n, d))
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx

def _rand_even_signs(rng, m):
    S = np.ones((m, 6))
    R = rng.random((m, 6)) < 0.5
    S[R] = -1.0
    # enforce even parity (product +1): if odd number of -1, flip first coord
    neg = np.sum(S < 0, axis=1)
    odd = (neg % 2) == 1
    S[odd, 0] *= -1.0
    return S

def _apply_orbit_point(g, shift, signs):
    return np.roll(signs * g, shift)

def _make_orbit_Y(rng, g):
    m = 58
    shifts = rng.integers(0, 6, size=m)
    signs = _rand_even_signs(rng, m)
    Y = np.empty((m, 6))
    for i in range(m):
        Y[i] = _apply_orbit_point(g, int(shifts[i]), signs[i])
    return _norm(Y)

def _pair_antipodal_Y(Xb):
    n = Xb.shape[0]
    assert n == 116
    unmatched = set(range(n))
    Y = []
    G = Xb @ Xb.T
    while unmatched:
        i = next(iter(unmatched))
        unmatched.remove(i)
        if not unmatched:
            break
        # most antipodal = minimal dot
        best_j = None; best_v = 1e9
        for j in unmatched:
            v = G[i, j]
            if v < best_v:
                best_v = v; best_j = j
        unmatched.remove(best_j)
        v = Xb[i] - Xb[best_j]
        nv = np.linalg.norm(v)
        if nv < 1e-8:
            v = np.random.default_rng(i * 1009 + best_j).standard_normal(6)
            nv = np.linalg.norm(v)
        Y.append(v / nv)
    Y = np.array(Y)
    if Y.shape[0] > 58:
        Y = Y[:58]
    while Y.shape[0] < 58:
        a = np.random.standard_normal(6); a /= np.linalg.norm(a)
        Y = np.vstack([Y, a])
    return _norm(Y)

def _orbit_forward(Gens, shifts, signs, jidx):
    n = len(jidx)
    X = np.empty((n, 6))
    for p in range(n):
        g = Gens[jidx[p]]
        X[p] = np.roll(signs[p] * g, int(shifts[p]))
    return _norm(X)

def _orbit_adam(Gens0, shifts, signs, jidx, steps=800, beta0=30.0, beta1=250.0, lr0=0.01, lr1=0.001):
    Gens = _norm(np.array(Gens0, dtype=float))
    k = Gens.shape[0]
    m1 = np.zeros_like(Gens); m2 = np.zeros_like(Gens)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = len(jidx)
    for s in range(1, steps + 1):
        frac = (s - 1) / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        X = np.empty((n, 6))
        for p in range(n):
            g = Gens[jidx[p]]
            X[p] = np.roll(signs[p] * g, int(shifts[p]))
        Xn = np.linalg.norm(X, axis=1, keepdims=True); Xn[Xn == 0] = 1.0
        X = X / Xn
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        W /= S
        GradX = W @ X
        gG = np.zeros_like(Gens)
        for p in range(n):
            gp = GradX[p]
            # back through norm (approx tangent) and roll/sign
            # Xp = roll(c*g)/|..| ; use tangent projection
            # pull back: roll^{-1}(gp) * c, then project tangent to g
            back = np.roll(gp, -int(shifts[p])) * signs[p]
            gG[jidx[p]] += back
        # tangent project
        dot = np.sum(gG * Gens, axis=1, keepdims=True)
        Gt = gG - dot * Gens
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s; c2 = 1.0 - b2 ** s
        mh = m1 / c1; vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        d2 = np.sum(step * Gens, axis=1, keepdims=True)
        stept = step - d2 * Gens
        Gens = _norm(Gens - lr * stept)
    # final forward
    Xf = np.empty((n, 6))
    for p in range(n):
        Xf[p] = np.roll(signs[p] * Gens[jidx[p]], int(shifts[p]))
    return _norm(Xf), Gens

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HARD_END = t0 + BUDGET - 1.0
    SYM_END = t0 + 120.0
    rng = np.random.default_rng(20261031)
    def clean(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nr = np.linalg.norm(a, axis=1, keepdims=True); nr[nr == 0] = 1.0
        return a / nr
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])

    sym_candidates = []  # list of (val, Xsym 116x6)
    try:
        # build Y seeds
        seeds_Y = []
        # 1: random
        seeds_Y.append(_norm(rng.standard_normal((58, 6))))
        # 2: antipodally paired incumbent
        try:
            seeds_Y.append(_pair_antipodal_Y(x_best))
        except Exception:
            pass
        # 3,4: orbit of generic generators under 6-cycle + even flips
        for k in range(2):
            gg = rng.standard_normal(6); gg /= np.linalg.norm(gg)
            seeds_Y.append(_make_orbit_Y(rng, gg))
        # 5: subset of best
        idx = rng.choice(116, size=58, replace=False)
        seeds_Y.append(_norm(x_best[idx].copy()))
        for Y0 in seeds_Y:
            if time.time() > SYM_END - 8:
                break
            if time.time() > HARD_END - 30:
                break
            try:
                Yo = _adam_sym(Y0, steps=1500, beta0=30.0, beta1=300.0, lr0=0.01, lr1=0.001)
            except Exception:
                continue
            try:
                Xs = np.vstack([Yo, -Yo])
                Xs = _norm(Xs)
                v = _maxcos(Xs)
            except Exception:
                continue
            if v < 0.605:
                sym_candidates.append((v, Xs.copy(), Yo.copy()))
        # orbit-based variant: only generators are variables
        if time.time() < SYM_END - 15 and time.time() < HARD_END - 35:
            for k, steps_o in [(2, 700), (4, 700)]:
                if time.time() > SYM_END - 10:
                    break
                try:
                    rng2 = np.random.default_rng(1000 + k)
                    Gens0 = _norm(rng2.standard_normal((k, 6)))
                    per = 116 // k
                    rem = 116 - per * k
                    jidx = []
                    for j in range(k):
                        cnt = per + (1 if j < rem else 0)
                        jidx += [j] * cnt
                    jidx = np.array(jidx, dtype=int)
                    shifts = rng2.integers(0, 6, size=116)
                    signs = _rand_even_signs(rng2, 116)
                    Xf, _ = _orbit_adam(Gens0, shifts, signs, jidx, steps=steps_o)
                    v = _maxcos(Xf)
                    if v < 0.605:
                        # try to extract antipodal Y approx for release tracking
                        sym_candidates.append((v, Xf.copy(), None))
                except Exception:
                    continue
    except Exception:
        pass

    # symmetry-breaking release + requench
    released_best = None
    released_val = 1e9
    noise_levels = [1e-3, 3e-3, 1e-2]
    # sort symmetric by val
    sym_candidates.sort(key=lambda t: t[0])
    for (sv, Xs, Yo) in sym_candidates[:4]:
        for nl in noise_levels:
            if time.time() > HARD_END - 25:
                break
            try:
                Xn = Xs.copy()
                pert = rng.standard_normal(Xn.shape)
                pert = pert - np.sum(pert * Xn, axis=1, keepdims=True) * Xn
                pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-15
                pert = pert / pn * nl
                # scale: nl is per-point tangent displacement
                Xp = _norm(Xn + pert)
                Xq = _quench(Xp, steps=350, beta0=30.0, beta1=250.0, s0=0.02, s1=0.002)
                vq = _maxcos(Xq)
                if vq < released_val:
                    released_val = vq
                    released_best = Xq.copy()
                # extra smaller quench from same start with different steps?
            except Exception:
                continue
        if time.time() > HARD_END - 25:
            break
    if released_best is not None and released_val < best_val + 0.01:
        # short LP polish on released if promising and time allows
        try:
            if time.time() < HARD_END - 12:
                rp = _lp_polish(released_best, min(HARD_END - 8, time.time() + 4.0))
                rv = _maxcos(rp)
                if rv < released_val:
                    released_val = rv; released_best = rp
        except Exception:
            pass
        if released_val < best_val:
            best_val = released_val
            x_best = released_best.copy()

    # fallback: if symmetric path failed, ensure we still improve incumbent via quench diversity
    # quick diverse requench of incumbent with tiny noise if time remains and no improvement yet
    if released_best is None and time.time() < HARD_END - 20:
        try:
            for nl in [3e-3, 1e-2]:
                if time.time() > HARD_END - 22:
                    break
                Xp = _norm(x_best + nl * (rng.standard_normal(x_best.shape) - np.sum(rng.standard_normal(x_best.shape) * 0, axis=1, keepdims=True)))
                # proper tangent noise
                Rn = rng.standard_normal(x_best.shape)
                Rn = Rn - np.sum(Rn * x_best, axis=1, keepdims=True) * x_best
                Rn = Rn / (np.linalg.norm(Rn, axis=1, keepdims=True) + 1e-15) * nl
                Xp = _norm(x_best + Rn)
                Xq = _quench(Xp, steps=250, beta0=30.0, beta1=220.0, s0=0.02, s1=0.002)
                vq = _maxcos(Xq)
                if vq < best_val:
                    best_val = vq; x_best = Xq.copy()
                    break
        except Exception:
            pass

    # final LP polish of global best
    remain = HARD_END - time.time() - 0.5
    if remain > 2.5:
        try:
            x_best = _lp_polish(x_best, min(HARD_END - 0.3, time.time() + min(remain, 18.0)))
        except Exception:
            pass
    return _norm(x_best)
