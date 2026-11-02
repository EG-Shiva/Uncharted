import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

K = 17
J = 9
N = 153
TW = 2 * np.pi / 17.0

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _sym_build(mvec, alpha, theta, phi):
    # alpha,theta,phi shape (9,)
    s = 1.0 / (1.0 + np.exp(-alpha))
    rho = s * (np.pi * 0.5)
    r = np.cos(rho)
    b = np.sin(rho)
    X = np.empty((153, 4))
    ks = np.arange(17)
    for j in range(9):
        u = theta[j] + TW * ks
        v = phi[j] + TW * float(mvec[j]) * ks
        cu = np.cos(u); su = np.sin(u)
        cv = np.cos(v); sv = np.sin(v)
        X[j*17:(j+1)*17, 0] = r[j] * cu
        X[j*17:(j+1)*17, 1] = r[j] * su
        X[j*17:(j+1)*17, 2] = b[j] * cv
        X[j*17:(j+1)*17, 3] = b[j] * sv
    return X

def _sym_opt(mvec, alpha0, theta0, phi0, steps, T0, T1, lr, rng=None):
    alpha = alpha0.copy(); theta = theta0.copy(); phi = phi0.copy()
    m1 = np.zeros((9, 3)); m2 = np.zeros((9, 3))
    b1, b2, eps = 0.9, 0.999, 1e-8
    ks = np.arange(17)
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        s = 1.0 / (1.0 + np.exp(-np.clip(alpha, -20, 20)))
        rho = s * (np.pi * 0.5)
        r = np.cos(rho); b = np.sin(rho)
        # build X + cos/sin caches
        X = np.empty((153, 4))
        CU = np.empty((9, 17)); SU = np.empty((9, 17))
        CV = np.empty((9, 17)); SV = np.empty((9, 17))
        for j in range(9):
            u = theta[j] + TW * ks
            v = phi[j] + TW * float(mvec[j]) * ks
            cu = np.cos(u); su = np.sin(u)
            cv = np.cos(v); sv = np.sin(v)
            CU[j] = cu; SU[j] = su; CV[j] = cv; SV[j] = sv
            X[j*17:(j+1)*17, 0] = r[j]*cu
            X[j*17:(j+1)*17, 1] = r[j]*su
            X[j*17:(j+1)*17, 2] = b[j]*cv
            X[j*17:(j+1)*17, 3] = b[j]*sv
        G = X @ X.T
        np.fill_diagonal(G, -1.0)
        gm = G.max()
        W = np.exp((G - gm) / T)
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z <= 0:
            Z = 1.0
        # grad_X = 2 S X
        GX = (W @ X) * (2.0 / Z)
        # param grads
        gA = np.zeros(9); gT = np.zeros(9); gP = np.zeros(9)
        ds = s * (1.0 - s)
        drho = ds * (np.pi * 0.5)
        for j in range(9):
            gx = GX[j*17:(j+1)*17]  # 17x4
            cu = CU[j]; su = SU[j]; cv = CV[j]; sv = SV[j]
            t1 = gx[:, 0]*cu + gx[:, 1]*su
            t2 = gx[:, 2]*cv + gx[:, 3]*sv
            gA[j] = np.sum(-b[j]*t1 + r[j]*t2) * drho[j]
            gT[j] = r[j] * np.sum(-gx[:, 0]*su + gx[:, 1]*cu)
            gP[j] = b[j] * np.sum(-gx[:, 2]*sv + gx[:, 3]*cv)
        grad = np.stack([gA, gT, gP], axis=1)
        # Adam (descent)
        m1 = b1*m1 + (1-b1)*grad
        m2 = b2*m2 + (1-b2)*(grad*grad)
        c1 = 1 - b1**(step+1); c2 = 1 - b2**(step+1)
        upd = (m1/c1) / (np.sqrt(m2/c2) + eps)
        # scale lr slightly with T
        cur_lr = lr * (0.5 + 0.5*T/T0)
        alpha -= cur_lr * upd[:, 0]
        theta -= cur_lr * upd[:, 1]
        phi -= cur_lr * upd[:, 2]
        alpha = np.clip(alpha, -6, 6)
    return alpha, theta, phi

def _smooth_anneal(x, steps, T0, T1, lrfac):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, deadline, max_solves=100):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, Jj = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               Jj[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[Jj], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    sym_end = t0 + 65.0
    explore_end = t0 + 140.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = vals[bi]

    # ---- symmetric torus search ----
    patterns = []
    patterns.append([1]*9)
    patterns.append([1,2,3,4,5,6,7,8,9])
    patterns.append([1,3,5,7,9,11,13,15,2])
    patterns.append([1,1,2,2,3,3,5,8,13])
    # two random deterministic patterns
    patterns.append(list(rng.integers(1, 17, size=9)))
    patterns.append(list(rng.integers(1, 17, size=9)))
    sym_best = None
    sym_best_val = 1e9
    sym_cands = []
    for pi, mvec in enumerate(patterns):
        if time.time() > sym_end:
            break
        mvec = [int(v) % 17 for v in mvec]
        mvec = [v if v != 0 else 1 for v in mvec]
        for restart in range(3):
            if time.time() > sym_end:
                break
            alpha0 = rng.standard_normal(9) * 0.8
            theta0 = rng.uniform(0, 2*np.pi, size=9)
            phi0 = rng.uniform(0, 2*np.pi, size=9)
            # annealed: coarse then fine
            try:
                a1, t1, p1 = _sym_opt(mvec, alpha0, theta0, phi0,
                                      steps=700, T0=0.025, T1=0.003, lr=0.08)
                a2, t2, p2 = _sym_opt(mvec, a1, t1, p1,
                                      steps=400, T0=0.004, T1=0.0008, lr=0.03)
            except Exception:
                continue
            X = _sym_build(mvec, a2, t2, p2)
            v = _max_cos(X)
            sym_cands.append((v, X.copy(), tuple(mvec), (a2, t2, p2)))
            if v < sym_best_val:
                sym_best_val = v
                sym_best = X.copy()
    # keep top symmetric seeds
    sym_cands.sort(key=lambda z: z[0])
    top_sym = [z[1] for z in sym_cands[:3]]

    # ---- release symmetry: unrestricted anneal of symmetric seeds ----
    for si, sx in enumerate(list(top_sym)):
        if time.time() > explore_end:
            break
        # small symmetry-breaking noise levels
        for nl, seed in [(0.0, 100+si), (0.004, 200+si)]:
            if time.time() > explore_end - 5:
                break
            y = sx.copy()
            if nl > 0:
                rr = np.random.default_rng(seed)
                y = y + nl * rr.standard_normal(y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_anneal(y, steps=500, T0=0.012, T1=0.0018, lrfac=1.2)
            y = _smooth_anneal(y, steps=150, T0=0.003, T1=0.0010, lrfac=1.0)
            v = _max_cos(y)
            if v < best_val - 1e-12:
                # quick LP confirm
                y2, v2 = _lp_polish(y, min(deadline-8, time.time()+2.0), max_solves=20)
                cand, cval = (y2, v2) if v2 < v else (y, v)
                if cval < best_val:
                    best_x, best_val = cand, cval
            elif v < best_val + 0.003 and time.time() < explore_end - 10:
                y2, v2 = _lp_polish(y, time.time()+1.0, max_solves=10)
                if v2 < best_val - 1e-12:
                    best_x, best_val = y2, v2

    # ---- also hill-climb from incumbent with symmetry-broken topology ----
    # use best symmetric as donor: splice halves if symmetric still worse
    # fallback: a few void-reinsert + anneal steps from best_x to ensure no regression
    trial = 0
    while time.time() < explore_end:
        trial += 1
        if time.time() > explore_end:
            break
        base = best_x
        y = base.copy()
        # alternate: perturb worst points or inject symmetric block
        if trial % 4 == 0 and len(top_sym) > 0:
            # splice: replace 17-point block with symmetric orbit block
            donor = top_sym[trial % len(top_sym)]
            rr = np.random.default_rng(9000+trial)
            # random rotation to align roughly? just copy with random O(4)
            A = rr.standard_normal((4, 4))
            Q, _ = np.linalg.qr(A)
            drot = donor @ Q.T
            # replace worst 17 points
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            worst = np.argsort(-per)[:17]
            y[worst] = drot[:17]
            y = y + 0.005 * rr.standard_normal(y.shape)
        elif trial % 2 == 0:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            idx = np.argsort(-per)[:8]
            y[idx] = y[idx] + 0.06 * rng.standard_normal((8, 4))
            y = y + 0.003 * rng.standard_normal(y.shape)
        else:
            y = y + 0.02 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth_anneal(y, steps=350, T0=0.012, T1=0.0018, lrfac=1.2)
        y = _smooth_anneal(y, steps=100, T0=0.003, T1=0.0012, lrfac=1.0)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            if v < best_val - 5e-6:
                y2, v2 = _lp_polish(y, min(deadline-8, time.time()+2.0), max_solves=20)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                elif v < best_val:
                    best_x, best_val = y, v
            else:
                best_x, best_val = y, v
        if trial > 25:
            break

    if time.time() < deadline - 1.0:
        best_x, best_val = _lp_polish(best_x, deadline - 0.5, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
