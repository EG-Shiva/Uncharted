import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _adam_quench(X0, steps, a0, a1, lr0, lr1, deadline=None):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        if deadline is not None and (t % 100 == 0) and time.time() > deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = np.max(G)
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = np.sum(E)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (E @ X) / S
        proj = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        X /= nr
    return X

def _active_pairs(X, tol=1e-7, min_p=180, max_p=700):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = float(np.max(G))
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    order = np.argsort(-c)
    # threshold selection
    thr = M - tol
    sel_count = int(np.sum(c > thr))
    if sel_count < min_p:
        sel_count = min(min_p, len(c))
    if sel_count > max_p:
        sel_count = max_p
    take = order[:sel_count]
    I = iu0[0][take]
    J = iu0[1][take]
    return I, J, M, G

def _equilibrium_stress(X, I, J, iters=400):
    # tangent components
    n, d = X.shape
    p = len(I)
    dot = np.sum(X[I] * X[J], axis=1)
    Ti = X[J] - dot[:, None] * X[I]
    Tj = X[I] - dot[:, None] * X[J]
    w = np.full(p, 1.0 / p)
    # mirror descent on min ||F||^2
    eta = 2.0
    for it in range(iters):
        F = np.zeros_like(X)
        np.add.at(F, I, Ti * w[:, None])
        np.add.at(F, J, Tj * w[:, None])
        g = np.sum(F[I] * Ti, axis=1) + np.sum(F[J] * Tj, axis=1)
        g = 2.0 * g
        # normalize gradient scale
        gs = np.max(np.abs(g))
        if gs > 0:
            g = g / (gs + 1e-12)
        w = w * np.exp(-eta * g)
        s = np.sum(w)
        if not np.finite(s) or s <= 0:
            w = np.full(p, 1.0 / p)
            eta *= 0.5
            continue
        w /= s
        eta *= 0.995
        # mix uniform to avoid collapse
        if it % 100 == 99:
            w = 0.95 * w + 0.05 / p
    return w

def _build_Jext(X, I, J):
    n, d = X.shape
    p = len(I)
    N = n * d
    m = p + n
    Jext = np.zeros((m, N), dtype=np.float64)
    # pair rows
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        Jext[k, i * d:(i + 1) * d] = X[j]
        Jext[k, j * d:(j + 1) * d] = X[i]
    for i in range(n):
        Jext[p + i, i * d:(i + 1) * d] = X[i]
    return Jext

def _nullspace_part(Jext, r):
    # r: (N,) tangent-ish random; return nullspace projection
    Jr = Jext @ r
    A = Jext @ Jext.T
    # regularize
    A += np.eye(A.shape[0]) * 1e-8
    try:
        y = np.linalg.solve(A, Jr)
    except Exception:
        y, *_ = np.linalg.lstsq(A, Jr, rcond=None)
    return r - (Jext.T @ y)

def _directed_dx(X, I, J, w, Jext, rng, subset, step, gamma):
    n, d = X.shape
    p = len(I)
    N = n * d
    m = Jext.shape[0]
    b = np.zeros(m)
    # allow subset to worsen (+1), others to reduce (-c)
    if len(subset) > 0:
        b[np.array(subset, dtype=int)] = 1.0
        mask = np.ones(p, dtype=bool)
        mask[np.array(subset, dtype=int)] = False
        b[mask] = -0.5
    else:
        b[:p] = -1.0
    # normal rows target 0 already
    A = Jext @ Jext.T + np.eye(m) * 1e-6
    try:
        y = np.linalg.solve(A, b)
    except Exception:
        y, *_ = np.linalg.lstsq(A, b, rcond=None)
    dx_part = Jext.T @ y
    # random tangent vector
    r = rng.standard_normal(size=N)
    # project to tangent: remove normal components
    R = r.reshape(n, d)
    dotn = np.sum(R * X, axis=1, keepdims=True)
    R = R - dotn * X
    r = R.ravel()
    r_null = _nullspace_part(Jext, r)
    # normalize each
    n1 = np.linalg.norm(dx_part) + 1e-12
    n2 = np.linalg.norm(r_null) + 1e-12
    dx = step * (dx_part / n1 + gamma * r_null / n2)
    # scale so max per-point norm ~ step
    DX = dx.reshape(n, d)
    # remove normal part again for safety
    dn = np.sum(DX * X, axis=1, keepdims=True)
    DX = DX - dn * X
    mx = np.max(np.linalg.norm(DX, axis=1)) + 1e-12
    DX = DX / mx * step
    return DX

def _slsqp_epigraph(X0, I, J, deadline):
    # optimize subset of involved points with epigraph SLSQP, time-boxed
    try:
        n, d = X0.shape
        inv = np.unique(np.concatenate([I, J]))
        if len(inv) > 22:
            # worst by degree
            deg = np.zeros(n, dtype=int)
            for k in range(len(I)):
                deg[I[k]] += 1; deg[J[k]] += 1
            inv = np.argsort(-deg)[:22]
        inv = np.sort(inv)
        k = len(inv)
        mp = {int(v): t for t, v in enumerate(inv)}
        # map pairs with both ends in inv (others fixed as constants)
        Xin = X0[inv]
        x0v = Xin.ravel()
        # precompute fixed dots for pairs with one end outside? include as constraints too
        # build list of (a,b, fixed_vec) where a in inv, b fixed or both in inv
        pairs = []
        for t in range(len(I)):
            i = int(I[t]); j = int(J[t])
            ii = mp.get(i, -1); jj = mp.get(j, -1)
            if ii >= 0 and jj >= 0:
                pairs.append((ii, jj, None, None))
            elif ii >= 0:
                pairs.append((ii, -1, X0[j], None))
            elif jj >= 0:
                pairs.append((-1, jj, None, X0[i]))
        # objective: t (last var) + tiny regularization
        def fun(v):
            return float(v[-1])
        def jac(v):
            g = np.zeros_like(v); g[-1] = 1.0
            return g
        cons = []
        # ineq: dot - t <= 0
        def mk_fun(a, b, fa, fb):
            def cf(v, a=a, b=b, fa=fa, fb=fb):
                t = v[-1]
                Y = v[:-1].reshape(k, d)
                Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
                if a >= 0 and b >= 0:
                    val = float(np.dot(Yn[a], Yn[b]) - t)
                elif a >= 0:
                    val = float(np.dot(Yn[a], fb) - t)
                else:
                    val = float(np.dot(Yn[b], fa) - t)
                return -val  # >=0
            return cf
        for (a, b, fa, fb) in pairs:
            if a == -1 and b == -1:
                continue
            # capture fixed vectors correctly
            if a >= 0 and b == -1:
                fbv = fb
                fav = None
                cons.append({'type': 'ineq', 'fun': mk_fun(a, b, fav, fbv)})
            elif b >= 0 and a == -1:
                fav = fa
                cons.append({'type': 'ineq', 'fun': mk_fun(a, b, fav, None)})
            else:
                cons.append({'type': 'ineq', 'fun': mk_fun(a, b, None, None)})
        # equality norms: ||y_i||^2-1=0
        def eqf(v):
            Y = v[:-1].reshape(k, d)
            return np.sum(Y * Y, axis=1) - 1.0
        def eqj(v):
            Y = v[:-1].reshape(k, d)
            Jm = np.zeros((k, v.size))
            for q in range(k):
                Jm[q, q * d:(q + 1) * d] = 2 * Y[q]
            return Jm
        cons.append({'type': 'eq', 'fun': eqf, 'jac': eqj})
        M0 = _max_cos(X0)
        v0 = np.concatenate([x0v, [M0]])
        # bounds None
        res = minimize(fun, v0, jac=jac, method='SLSQP', constraints=cons,
                       options={'maxiter': 60, 'ftol': 1e-12, 'disp': False, 'eps': 1e-8})
        if not np.all(np.isfinite(res.x)):
            return X0
        Y = res.x[:-1].reshape(k, d)
        Y = _norm_rows(Y)
        Xn = X0.copy()
        Xn[inv] = Y
        Xn = _norm_rows(Xn)
        if time.time() > deadline:
            return Xn
        # accept only if not worse by much (allow sideways to change topology)
        if _max_cos(Xn) < _max_cos(X0) + 5e-4:
            return Xn
        return X0
    except Exception:
        return X0

def _lp_polish(x, deadline):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        thr = m - max(10 * radius, 2e-7)
        sel = cfull > thr
        if np.sum(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:600]
            sel2 = np.zeros_like(sel)
            sel2[idx[order]] = True
            sel = sel2
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = deadline - time.time()
        if tl <= 0.5:
            break
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.5, min(5.0, tl - 0.2))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        x = bx.copy()
    return bx

def solve():
    t_start = time.time()
    HARD = t_start + 150.0
    SEARCH_END = t_start + 128.0
    rng = np.random.default_rng(12345)
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal(size=(n, d)))
        best_score = _max_cos(best)
    cur_best = best.copy()
    cur_score = best_score
    # initial active-set stress analysis on incumbent
    def refresh_stress(X):
        I, J, M, G = _active_pairs(X, tol=1e-7, min_p=180, max_p=700)
        w = _equilibrium_stress(X, I, J, iters=300)
        Jext = _build_Jext(X, I, J)
        order = np.argsort(-w)
        return I, J, M, w, Jext, order
    I, J, M, w, Jext, order = refresh_stress(cur_best)
    trial = 0
    steps_list = [0.004, 0.008, 0.015, 0.03]
    # cycle through stress-selected subsets
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 2:
            break
        # periodically refresh from global best and occasionally from perturbed base
        if trial % 12 == 0:
            I, J, M, w, Jext, order = refresh_stress(cur_best)
        # choose subset of high-stress contacts to deliberately worsen
        r = rng.random()
        topK = 20
        if r < 0.25:
            ksub = int(rng.integers(1, 4))
            subset = order[:ksub].tolist()
        elif r < 0.5:
            ksub = int(rng.integers(3, 8))
            subset = order[:ksub].tolist()
        elif r < 0.75:
            ksub = int(rng.integers(1, 6))
            pool = order[:topK]
            subset = rng.choice(pool, size=min(ksub, len(pool)), replace=False).tolist()
        else:
            # random subset among top 40 + allow empty (pure nullspace move)
            pool = order[:min(40, len(order))]
            ksub = int(rng.integers(0, 6))
            if ksub == 0:
                subset = []
            else:
                subset = rng.choice(pool, size=min(ksub, len(pool)), replace=False).tolist()
        step = steps_list[trial % len(steps_list)] * (0.7 + 0.6 * rng.random())
        gamma = float(rng.choice([0.3, 0.7, 1.5, 3.0]))
        try:
            DX = _directed_dx(cur_best, I, J, w, Jext, rng, subset, step, gamma)
        except Exception:
            continue
        Xp = _norm_rows(cur_best + DX)
        # quick re-optimise: fast annealed Adam (epigraph-like smoothing)
        qdead = min(HARD - 25, time.time() + 4.0)
        try:
            Xq = _adam_quench(Xp, steps=350, a0=25.0, a1=500.0, lr0=0.015, lr1=0.002, deadline=qdead)
        except Exception:
            continue
        s = _max_cos(Xq)
        # occasional epigraph SLSQP using only active/near-active pairs
        if s < cur_score + 2e-4 and (trial % 7 == 0) and time.time() < SEARCH_END - 15:
            try:
                Ia, Ja, _, _ = _active_pairs(Xq, tol=5e-4, min_p=80, max_p=250)
                Xq2 = _slsqp_epigraph(Xq, Ia, Ja, deadline=time.time() + 3.0)
                s2 = _max_cos(Xq2)
                if s2 < s:
                    Xq, s = Xq2, s2
            except Exception:
                pass
        if s < cur_score - 1e-12:
            cur_best = Xq.copy()
            cur_score = s
            # refresh stress immediately on improvement
            try:
                I, J, M, w, Jext, order = refresh_stress(cur_best)
            except Exception:
                pass
            if cur_score < best_score - 1e-12:
                best = cur_best.copy()
                best_score = cur_score
        if time.time() > HARD - 25:
            break
    # keep global best
    if cur_score < best_score:
        best, best_score = cur_best.copy(), cur_score
    # final short squeeze: Adam then SLSQP then LP
    if time.time() < HARD - 4:
        try:
            best = _adam_quench(best, steps=300, a0=80.0, a1=900.0, lr0=0.008, lr1=0.0008, deadline=HARD - 12)
        except Exception:
            pass
    if time.time() < HARD - 10:
        try:
            Ia, Ja, _, _ = _active_pairs(best, tol=5e-4, min_p=80, max_p=250)
            best2 = _slsqp_epigraph(best, Ia, Ja, deadline=HARD - 8)
            # keep if better
            if _max_cos(best2) < _max_cos(best):
                best = best2
        except Exception:
            pass
    if time.time() < HARD - 3:
        try:
            best = _lp_polish(best, deadline=HARD - 0.5)
        except Exception:
            pass
    best = _norm_rows(best)
    return best
