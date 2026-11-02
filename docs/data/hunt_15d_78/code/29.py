import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

N_PTS = 78
DIM = 15
N_ORB = 6
ORB_SZ = 13
IU78 = np.triu_indices(N_PTS, 1)

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def gen_c13(S):
    S = np.asarray(S).reshape(N_ORB, DIM)
    Y = np.empty((N_PTS, DIM), dtype=np.float64)
    for o in range(N_ORB):
        b = S[o, :13]
        t = S[o, 13:]
        base = o * ORB_SZ
        for k in range(ORB_SZ):
            r = base + k
            Y[r, :13] = np.roll(b, k)
            Y[r, 13:] = t
    nr = np.linalg.norm(Y, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    return Y / nr, Y, nr

def seed_lse_factory(beta):
    iu0, iu1 = IU78
    def fun(xf):
        S = xf.reshape(N_ORB, DIM)
        Y = np.empty((N_PTS, DIM), dtype=np.float64)
        for o in range(N_ORB):
            b = S[o, :13]
            t = S[o, 13:]
            base = o * ORB_SZ
            for k in range(ORB_SZ):
                r = base + k
                Y[r, :13] = np.roll(b, k)
                Y[r, 13:] = t
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        gv = G[iu0, iu1]
        m = gv.max()
        e = np.exp(beta * (gv - m))
        s = e.sum()
        F = m + np.log(s) / beta
        w = e / s
        W = np.zeros((N_PTS, N_PTS), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        GradX = W @ X
        dot = (GradX * X).sum(axis=1, keepdims=True)
        GradY = (GradX - dot * X) / nr
        GradS = np.zeros((N_ORB, DIM), dtype=np.float64)
        for o in range(N_ORB):
            blk = GradY[o * ORB_SZ:(o + 1) * ORB_SZ, :]
            GradS[o, 13:] = blk[:, 13:].sum(axis=0)
            acc = np.zeros(13, dtype=np.float64)
            for k in range(ORB_SZ):
                acc += np.roll(blk[k, :13], -k)
            GradS[o, :13] = acc
        return float(F), GradS.ravel()
    return fun

def full_lse_factory(beta, n=N_PTS, d=DIM):
    iu0, iu1 = np.triu_indices(n, 1)
    def fun(xf):
        Xraw = xf.reshape(n, d)
        nr = np.linalg.norm(Xraw, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Y = Xraw / nr
        G = Y @ Y.T
        gv = G[iu0, iu1]
        m = gv.max()
        e = np.exp(beta * (gv - m))
        s = e.sum()
        F = m + np.log(s) / beta
        w = e / s
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        GradY = W @ Y
        dot = (GradY * Y).sum(axis=1, keepdims=True)
        GradX = (GradY - dot * Y) / nr
        return float(F), GradX.ravel()
    return fun

def run_lbfgs(fun, x0, maxiter):
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-6})
        return res.x
    except Exception:
        return x0

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(7)
    bestX = None
    bestV = 1e9

    def time_left():
        return LIMIT - (time.time() - t0)

    def consider(X):
        nonlocal bestX, bestV
        X = np.asarray(X, dtype=np.float64)
        if X.shape != (78, 15):
            return
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(np.isfinite(X)):
            return
        v = max_cos(X)
        if v < bestV:
            bestV = v
            bestX = X.copy()

    # ---- Phase A: C13 symmetric screening ----
    NSCREEN = 30
    seeds_list = [rng.standard_normal((N_ORB, DIM)) for _ in range(NSCREEN)]
    screened = []
    fun50s = seed_lse_factory(50.0)
    for S0 in seeds_list:
        if time_left() < 90:
            break
        xf = run_lbfgs(fun50s, S0.ravel().copy(), 60)
        S = xf.reshape(N_ORB, DIM)
        Xg, _, _ = gen_c13(S)
        v = max_cos(Xg)
        screened.append((v, S.copy()))
    screened.sort(key=lambda t: t[0])
    KREF = min(5, len(screened))
    top_seeds = [s for _, s in screened[:KREF]]

    # refine top seeds at beta=200,800
    refined_sym = []
    for S0 in top_seeds:
        if time_left() < 75:
            break
        x = S0.ravel().copy()
        x = run_lbfgs(seed_lse_factory(200.0), x, 150)
        if time_left() < 70:
            Sx = x.reshape(N_ORB, DIM)
            Xg, _, _ = gen_c13(Sx)
            refined_sym.append((max_cos(Xg), Sx.copy()))
            break
        x = run_lbfgs(seed_lse_factory(800.0), x, 150)
        Sx = x.reshape(N_ORB, DIM)
        Xg, _, _ = gen_c13(Sx)
        refined_sym.append((max_cos(Xg), Sx.copy()))
    refined_sym.sort(key=lambda t: t[0])

    candA_sym = None
    if refined_sym:
        candA_sym = refined_sym[0][1]
        Xg, _, _ = gen_c13(candA_sym)
        consider(Xg)

    # ---- Release best symmetric to full 78-pt optimization ----
    if candA_sym is not None and time_left() > 50:
        Xg, _, _ = gen_c13(candA_sym)
        xf = Xg.ravel().copy()
        # small jitter to break exact symmetry
        xf += 1e-4 * rng.standard_normal(xf.shape)
        xf = run_lbfgs(full_lse_factory(200.0), xf, 250)
        if time_left() > 40:
            xf = run_lbfgs(full_lse_factory(800.0), xf, 250)
        Xr = (xf.reshape(78, 15))
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        consider(Xr)
        candA_rel = Xr.copy()
    else:
        candA_rel = None

    # ---- Phase B: signed weight-3 discrete variant ----
    NDISC = 10
    disc_screened = []
    fun50f = full_lse_factory(50.0)
    invsqrt3 = 1.0 / np.sqrt(3.0)
    for _ in range(NDISC):
        if time_left() < 45:
            break
        X0 = np.zeros((78, 15), dtype=np.float64)
        for i in range(78):
            idx = rng.choice(15, size=3, replace=False)
            sgn = rng.choice(np.array([-1.0, 1.0]), size=3)
            X0[i, idx] = sgn * invsqrt3
        xf = run_lbfgs(fun50f, X0.ravel().copy(), 60)
        Xr = xf.reshape(78, 15)
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        disc_screened.append((max_cos(Xr), Xr.copy()))
    disc_screened.sort(key=lambda t: t[0])
    KDISC = min(3, len(disc_screened))
    for k in range(KDISC):
        if time_left() < 30:
            break
        Xr0 = disc_screened[k][1]
        xf = Xr0.ravel().copy()
        xf = run_lbfgs(full_lse_factory(200.0), xf, 200)
        if time_left() > 25:
            xf = run_lbfgs(full_lse_factory(800.0), xf, 200)
        Xr = xf.reshape(78, 15)
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        consider(Xr)

    # ---- Final LP polish on the best found ----
    if bestX is None:
        a = rng.standard_normal((78, 15))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = max_cos(a)
    tl = time_left()
    if tl > 8:
        # polish best with most of remaining time, leave 2s margin
        bestX = lp_polish(bestX, min(tl - 2.0, 25.0))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)

    return bestX
