import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

_AMULT = np.array([1., 5., 7., 11., 13., 17.])
_OM = 2.0 * np.pi / 31.0
_OFF = np.outer(np.arange(31), _AMULT) * _OM  # (31,6)
_IU0 = None
_IU1 = None

def _get_iu(n):
    global _IU0, _IU1
    if _IU0 is None or len(_IU0) != n * (n - 1) // 2:
        _IU0, _IU1 = np.triu_indices(n, 1)
    return _IU0, _IU1

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _params_to_X(p):
    # p (36,) -> (93,13)
    p = np.asarray(p, dtype=float)
    X = np.empty((93, 13), dtype=float)
    for o in range(3):
        base = o * 12
        t = p[base:base + 6]
        phi = p[base + 6:base + 12]
        co = np.cos(t)
        si = np.sin(t)
        v = np.empty(7)
        v[0] = co[0]
        prod = si[0]
        for i in range(1, 6):
            v[i] = prod * co[i]
            prod = prod * si[i]
        v[6] = prod
        z = v[0]
        r = v[1:7]
        # radii non-negative by bounds; clip tiny negatives
        r = np.abs(r)
        ang = phi[None, :] + _OFF  # (31,6)
        ca = np.cos(ang) * r[None, :]
        sa = np.sin(ang) * r[None, :]
        blk = np.empty((31, 13))
        blk[:, 0] = z
        for j in range(6):
            blk[:, 1 + 2 * j] = ca[:, j]
            blk[:, 1 + 2 * j + 1] = sa[:, j]
        # normalize rows (should already be unit, fix numerics)
        nr = np.linalg.norm(blk, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        blk /= nr
        X[o * 31:(o + 1) * 31, :] = blk
    return X

def _sym_lse(p, tau):
    X = _params_to_X(p)
    iu0, iu1 = _get_iu(93)
    G = X @ X.T
    c = G[iu0, iu1]
    cmax = float(c.max())
    # stable lse
    e = np.exp((c - cmax) / tau)
    s = float(e.sum())
    return cmax + tau * np.log(s / float(len(c)))

def _sym_maxcos(p):
    X = _params_to_X(p)
    return _maxcos(X)

def _adam_lse_polish(X0, taus, steps_each, lr0, lr1, deadline=1e18):
    n, d = X0.shape
    iu0, iu1 = _get_iu(n)
    X = _norm(X0.copy())
    best = _maxcos(X)
    bestX = X.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = len(taus) * steps_each
    tg = 0
    tt = 0
    for tau in taus:
        m[:] = 0.0
        v[:] = 0.0
        tt = 0
        inv_tau = 1.0 / tau
        for s in range(steps_each):
            if time.time() > deadline:
                break
            frac = tg / max(1, total - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            tg += 1
            G = X @ X.T
            c = G[iu0, iu1]
            cm = float(c.max())
            if cm < best:
                best = cm
                bestX = X.copy()
            # softmax weights, stable
            # shift by max
            z = (c - cm) * inv_tau
            # clip lower to avoid underflow (exp(-50)~2e-22)
            np.clip(z, -50.0, 0.0, out=z)
            e = np.exp(z)
            es = float(e.sum())
            if not np.isfinite(es) or es <= 0:
                continue
            w = e / es
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W[iu1, iu0] = w
            grad = W @ X
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            if not np.all(np.isfinite(gt)):
                continue
            tt += 1
            m *= b1
            m += (1.0 - b1) * gt
            v *= b2
            v += (1.0 - b2) * (gt * gt)
            mh = m / (1.0 - b1 ** tt)
            vh = v / (1.0 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            X = X - step
            X = _norm(X)
    cur = _maxcos(X)
    if cur < best:
        return X, cur
    return bestX, best

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = x @ x.T
        cfull = G[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            idx = np.argpartition(-cfull, 4000)[:4000]
            # rebuild sel from top idx
            sel2 = np.zeros_like(sel)
            sel2[idx] = True
            I, J = iu[0][sel2], iu[1][sel2]
            p = len(I)
            csub = cfull[sel2]
        else:
            csub = cfull[sel]
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _perturb_tangent(X, sigma, rng):
    Z = rng.standard_normal(size=X.shape) * sigma
    Z -= (np.sum(Z * X, axis=1, keepdims=True)) * X
    return _norm(X + Z)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13

    Xbest = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            A = np.load(fn)
            A = np.asarray(A, dtype=float)
            if A.shape != (93, 13):
                continue
            A = _norm(A)
            vv = _maxcos(A)
            if vv < best_val:
                best_val = vv
                Xbest = A.copy()
        except Exception:
            pass
    if Xbest is None:
        Xbest = _norm(rng.standard_normal((n, d)))
        best_val = _maxcos(Xbest)
    Xrec = Xbest.copy()
    try:
        A = np.load("record.npy")
        A = _norm(np.asarray(A, dtype=float))
        if A.shape == (93, 13):
            Xrec = A.copy()
    except Exception:
        pass

    # ---- Phase 1: symmetric C31 x3 search ----
    sym_deadline = t0 + 62.0
    # bounds: t0 in [0,pi], t1..5 in [0,pi/2], phi in [0,2pi]
    bnds = []
    for o in range(3):
        bnds.append((0.0, np.pi))
        for _ in range(5):
            bnds.append((0.0, np.pi / 2))
        for _ in range(6):
            bnds.append((0.0, 2.0 * np.pi))
    sym_cands = []  # (maxcos, params)
    # quick eval of a few deterministic seeds first
    n_restart = 55
    tau_a = 0.02
    tau_b = 0.004
    best_sym_val = 1e9
    best_sym_p = None
    best_sym_X = None
    for r in range(n_restart):
        if time.time() > sym_deadline - 1.0:
            break
        # sample
        p0 = np.empty(36)
        for o in range(3):
            b = o * 12
            p0[b] = rng.uniform(0.05, np.pi - 0.05)
            for k in range(1, 6):
                p0[b + k] = rng.uniform(0.08, np.pi / 2 - 0.05)
            for k in range(6):
                p0[b + 6 + k] = rng.uniform(0, 2 * np.pi)
        try:
            f1 = lambda pp: _sym_lse(pp, tau_a)
            res1 = minimize(f1, p0, method='L-BFGS-B', bounds=bnds,
                            options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-6})
            p1 = res1.x if res1.success or res1.fun < 1e9 else p0
        except Exception:
            p1 = p0
        if time.time() > sym_deadline - 0.5:
            # evaluate without second stage
            try:
                vv = _sym_maxcos(p1)
            except Exception:
                continue
            sym_cands.append((vv, p1.copy()))
            if vv < best_sym_val:
                best_sym_val = vv
                best_sym_p = p1.copy()
            continue
        try:
            f2 = lambda pp: _sym_lse(pp, tau_b)
            res2 = minimize(f2, p1, method='L-BFGS-B', bounds=bnds,
                            options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-7})
            p2 = res2.x if (res2.success or res2.fun < 1e9) else p1
        except Exception:
            p2 = p1
        try:
            vv = _sym_maxcos(p2)
        except Exception:
            continue
        sym_cands.append((vv, p2.copy()))
        if vv < best_sym_val:
            best_sym_val = vv
            best_sym_p = p2.copy()
            try:
                best_sym_X = _params_to_X(p2)
            except Exception:
                best_sym_X = None
    sym_cands = sorted(sym_cands, key=lambda t: t[0])
    # keep top 3 symmetric params
    top_sym_ps = [p for _, p in sym_cands[:3]]

    # expand symmetric to full arrays
    full_starts = []
    for p in top_sym_ps:
        try:
            Xs = _params_to_X(p)
            full_starts.append(Xs)
        except Exception:
            pass
    # if symmetric failed, add random
    if len(full_starts) == 0:
        try:
            full_starts.append(_params_to_X(np.array([np.pi/3]*6 + [0.0]*6 + [np.pi/3]*6 + [1.0]*6 + [np.pi/2]*6 + [2.0]*6)[:36] if False else rng.uniform(0,1,36)))
        except Exception:
            pass

    # break symmetry with tangent perturbations of required sizes
    sym_expanded = list(full_starts)
    for Xs in list(sym_expanded):
        for sig in (1e-4, 1e-3, 1e-2):
            full_starts.append(_perturb_tangent(Xs, sig, rng))
    # also add record/best and perturbations
    full_starts.append(Xbest.copy())
    full_starts.append(Xrec.copy())
    full_starts.append(_perturb_tangent(Xbest, 0.02, rng))
    full_starts.append(_perturb_tangent(Xbest, 0.05, rng))

    # ---- Phase 2: Adam polish each start (short) ----
    # allocate time: leave ~22s for final LP
    adam_deadline = deadline - 24.0
    polished = []
    # order: symmetric first then record (ensure different basin tried)
    # limit number polished depending on time
    max_polish = min(len(full_starts), 9)
    # sort full_starts: put best sym first, then record best last? just iterate
    # evaluate quickly and sort by maxcos to polish promising first
    scored = []
    for Xs in full_starts:
        try:
            scored.append((_maxcos(Xs), Xs))
        except Exception:
            pass
    scored = sorted(scored, key=lambda t: t[0])
    # ensure diversity: keep top sym + best record even if worse
    # take first max_polish
    to_polish = [X for _, X in scored[:max_polish]]
    # make sure Xbest included
    if not any(np.allclose(X, Xbest) for X in to_polish):
        to_polish[-1] = Xbest.copy()

    global_best = best_val
    global_X = Xbest.copy()
    # also consider raw symmetric if already good
    for vv, Xs in scored:
        if vv < global_best:
            global_best = vv
            global_X = Xs.copy()

    taus = [0.01, 0.005, 0.002]
    for idx, Xs in enumerate(to_polish):
        if time.time() > adam_deadline:
            break
        # adapt steps to remaining time
        remain = adam_deadline - time.time()
        left = len(to_polish) - idx
        per = remain / max(1, left)
        # steps_each ~ proportional to per (aim ~ 600 steps per 8s)
        if per < 2.0:
            steps_each = 200
        elif per < 5.0:
            steps_each = 450
        else:
            steps_each = 700
        try:
            Xp, vp = _adam_lse_polish(Xs, taus, steps_each, 3e-3, 3e-4,
                                      deadline=min(deadline - 20, time.time() + per))
        except Exception:
            continue
        polished.append((vp, Xp))
        if vp < global_best and np.all(np.isfinite(Xp)):
            global_best = vp
            global_X = Xp.copy()

    # ---- Phase 3: LP finish on top few ----
    polished = sorted(polished + [(global_best, global_X)], key=lambda t: t[0])
    # dedup keep top 3
    top = polished[:3]
    for i, (vp, Xp) in enumerate(top):
        if time.time() > deadline - 2.0:
            break
        secs = min(7.0, max(2.0, (deadline - 2.0 - time.time()) / (len(top) - i)))
        try:
            Xl = _lp_polish(Xp, secs, deadline=deadline - 1.0)
            vl = _maxcos(Xl)
        except Exception:
            continue
        if vl < global_best:
            global_best = vl
            global_X = Xl.copy()

    global_X = _norm(global_X)
    # strict feasibility: no zero rows, no NaN (already)
    return global_X
