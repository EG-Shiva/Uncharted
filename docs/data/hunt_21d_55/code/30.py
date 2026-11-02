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

def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    Eye = np.eye(d)
    for i in range(n):
        xi = X[i]
        k = int(np.argmax(np.abs(xi)))
        idx = [j for j in range(d) if j != k]
        A = Eye[:, idx] - np.outer(xi, xi[idx])
        Q, _ = np.linalg.qr(A)
        # Q is d x dm for tall case
        if Q.shape != (d, dm):
            Q = Q[:, :dm]
        B[i] = Q
    return B

def _build_Jn(X, B, I, J):
    n, d = X.shape
    dm = d - 1
    Nt = n * dm
    p = len(I)
    Jn = np.zeros((p, Nt))
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        vi = B[i].T @ X[j]
        vj = B[j].T @ X[i]
        Jn[k, i*dm:(i+1)*dm] = vi
        Jn[k, j*dm:(j+1)*dm] = vj
    return Jn

def _proj_simplex(v):
    p = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    cond = u * np.arange(1, p+1) > (css - 1)
    rho = np.nonzero(cond)[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)

def _equilibrium_stress(Jn, iters=500):
    p = Jn.shape[0]
    if p == 0:
        return np.zeros(0)
    Q = Jn @ Jn.T
    L = float(np.max(np.sum(np.abs(Q), axis=1))) + 1e-12
    lr = 0.9 / L
    w = np.full(p, 1.0 / p)
    for _ in range(iters):
        grad = Q @ w
        wn = w - lr * grad
        w = _proj_simplex(wn)
    return w

def _null_components(Jn, K, rng):
    p, Nt = Jn.shape
    if p == 0:
        R = rng.standard_normal((Nt, K))
        R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-12
        return R
    R = rng.standard_normal((Nt, K))
    Q = Jn @ Jn.T + 1e-10 * np.eye(p)
    try:
        Y = np.linalg.solve(Q, Jn @ R)
    except Exception:
        Y = np.linalg.lstsq(Q, Jn @ R, rcond=None)[0]
    Rn = R - Jn.T @ Y
    Rn /= np.linalg.norm(Rn, axis=0, keepdims=True) + 1e-12
    return Rn

def _coeff_to_ambient(v, B):
    n, d, dm = B.shape[0], B.shape[1], B.shape[2]
    D = np.empty((n, d))
    for i in range(n):
        D[i] = B[i] @ v[i*dm:(i+1)*dm]
    return D

def _adam_quench(X0, steps, a0, a1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Mx = np.max(G)
        E = np.exp(alpha * (G - Mx))
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
        X = _norm_rows(X)
    return X

def _lp_polish(x, deadline, radius_init=1e-3, max_it=25):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_it:
        it += 1
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 1200:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1200]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _slsqp_epigraph(X0, deadline, maxiter=12, topk=350):
    # epigraph SLSQP on near-active pairs only
    n, d = X0.shape
    G = X0 @ X0.T
    iu = np.triu_indices(n, 1)
    cfull = G[iu]
    order = np.argsort(-cfull)[:topk]
    I = iu[0][order].astype(int); J = iu[1][order].astype(int)
    m0 = float(cfull.max())
    z0 = np.concatenate([X0.ravel(), [m0]])
    N = n * d
    def fun(z):
        return z[-1]
    def jac_fun(z):
        g = np.zeros_like(z); g[-1] = 1.0
        return g
    def ineq_fun(z):
        Y = z[:N].reshape(n, d)
        return z[-1] - np.sum(Y[I]*Y[J], axis=1)
    def ineq_jac(z):
        Y = z[:N].reshape(n, d)
        p = len(I)
        Jac = np.zeros((p, N+1))
        for k in range(p):
            i = I[k]; j = J[k]
            Jac[k, i*d:(i+1)*d] = -Y[j]
            Jac[k, j*d:(j+1)*d] = -Y[i]
            Jac[k, N] = 1.0
        return Jac
    def eq_fun(z):
        Y = z[:N].reshape(n, d)
        return np.sum(Y*Y, axis=1) - 1.0
    def eq_jac(z):
        Y = z[:N].reshape(n, d)
        Jac = np.zeros((n, N+1))
        for i in range(n):
            Jac[i, i*d:(i+1)*d] = 2*Y[i]
        return Jac
    cons = [{'type':'ineq','fun':ineq_fun,'jac':ineq_jac},
            {'type':'eq','fun':eq_fun,'jac':eq_jac}]
    bounds = [(-1.2,1.2)]*N + [(-1.0,1.0)]
    try:
        res = minimize(fun, z0, jac=jac_fun, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter':maxiter,'ftol':1e-12,'disp':False,'eps':1e-9})
        if not np.all(np.isfinite(res.x)):
            return X0
        Y = _norm_rows(res.x[:N].reshape(n, d))
        if time.time() > deadline:
            return Y
        return Y
    except Exception:
        return X0

def solve():
    t0 = time.time()
    HARD_END = t0 + 150.0
    SEARCH_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 55, 21
    best = None; best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm_rows(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()
    iu = np.triu_indices(n, 1)
    # short initial stabilize not needed (already polished)
    outer = 0
    steps_cycle = [0.06, 0.12, 0.22, 0.09, 0.18]
    beta_cycle = [0.6, 1.2, 2.0, 0.4]
    while time.time() < SEARCH_END and outer < 12:
        outer += 1
        X = x_best.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        tol = 1e-7
        sel = G[iu] > m - tol
        # ensure at least 150 pairs for Jacobian
        if np.count_nonzero(sel) < 150:
            idx = np.argsort(-G[iu])[:300]
            sel = np.zeros_like(sel, dtype=bool); sel[idx] = True
        Iact = iu[0][sel].astype(int); Jact = iu[1][sel].astype(int)
        B = _tangent_bases(X)
        Jn = _build_Jn(X, B, Iact, Jact)
        w = _equilibrium_stress(Jn, iters=400)
        order_w = np.argsort(-w) if w.size else np.array([0])
        top_pool = order_w[:12]
        Rn = _null_components(Jn, 5, rng)
        # 8 directed trials per outer
        for tt in range(8):
            if time.time() > SEARCH_END:
                break
            # select stress subset S
            sz = 1 + (tt % 3)
            # deterministic subset from top pool + randomness
            if len(top_pool) >= sz:
                # rotate choice
                start = (tt*2 + outer) % max(1, len(top_pool)-sz+1)
                # occasionally random sample
                if rng.random() < 0.4:
                    S = rng.choice(top_pool, size=sz, replace=False)
                else:
                    S = top_pool[start:start+sz]
            else:
                S = top_pool
            v_targ = np.sum(Jn[S, :], axis=0) if len(S) else rng.standard_normal(Jn.shape[1])
            nrm = np.linalg.norm(v_targ) + 1e-12
            v_targ /= nrm
            coeff = rng.standard_normal(5)
            coeff /= np.linalg.norm(coeff) + 1e-12
            v_null = Rn @ coeff
            v_null /= np.linalg.norm(v_null) + 1e-12
            beta = beta_cycle[(tt + outer) % len(beta_cycle)]
            alpha = 1.0
            v = alpha * v_null + beta * v_targ
            v /= np.linalg.norm(v) + 1e-12
            step = steps_cycle[(tt*2 + outer) % len(steps_cycle)] * (0.8 + 0.4*rng.random())
            D = _coeff_to_ambient(v, B)
            # scale D has Fro=1 already
            Xt = _norm_rows(X + step * D)
            # annealed Adam quench (epigraph-like smoothing)
            Xtq = _adam_quench(Xt, steps=450, a0=40.0, a1=600.0, lr0=0.015, lr1=0.0015)
            s1 = _max_cos(Xtq)
            # only SLSQP/LP promising ones
            if s1 < best_score + 0.004:
                if time.time() > SEARCH_END - 8:
                    if s1 < best_score:
                        x_best = Xtq; best_score = s1
                    continue
                # epigraph SLSQP trust step on near-active pairs
                try:
                    Xs = _slsqp_epigraph(Xtq, HARD_END, maxiter=10, topk=350)
                    ss = _max_cos(Xs)
                    if ss < s1:
                        Xtq, s1 = Xs, ss
                except Exception:
                    pass
            if s1 < best_score + 0.0015:
                # short LP polish
                dl = min(HARD_END - 2.0, time.time() + 1.2)
                if dl > time.time() + 0.3:
                    Xtq = _lp_polish(Xtq, dl, max_it=18)
                    s1 = _max_cos(Xtq)
            if s1 < best_score and np.all(np.isfinite(Xtq)):
                x_best = Xtq.copy(); best_score = s1
                # early exit if clearly new record by margin? keep searching
                X = x_best.copy()
                break
        # end trials; loop recomputes stress/modes around new best
    # final short LP polish on global best with remaining time
    try:
        if time.time() < HARD_END - 3:
            x_best = _lp_polish(x_best, HARD_END - 1.0, max_it=25)
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    # tiny shrink guard not needed; ensure finite
    return x_best
