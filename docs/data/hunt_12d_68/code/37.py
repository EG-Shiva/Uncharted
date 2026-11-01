import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(x, deadline, r0=8e-4, cap_p=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def _full_fg_factory(IU, beta):
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full_lbfgs(x0, beta, maxiter):
    IU = np.triu_indices(68, 1)
    fg = _full_fg_factory(IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Y = r.x.reshape(68, 12)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _quench(x, steps, beta0, beta1, lr0, lr1, mu=0.85, rng=None):
    v = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        Tg = Grad - np.sum(Grad * x, axis=1, keepdims=True) * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        Tg = Tg / rms
        v = mu * v + Tg
        x = x - lr * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _precompute12(D, shift=1):
    d = 12
    K = 12
    idx = np.zeros((K, d), dtype=int)
    coeff = np.zeros((K, d), dtype=float)
    idx[0] = np.arange(d)
    coeff[0] = 1.0
    for k in range(1, K):
        pk = idx[k - 1]; pc = coeff[k - 1]
        for i in range(d):
            j = (i - shift) % d
            idx[k, i] = pk[j]
            coeff[k, i] = D[i] * pc[j]
    return idx, coeff

def _precompute8(D8, shift=1):
    K = 8
    idx = np.zeros((K, 12), dtype=int)
    coeff = np.zeros((K, 12), dtype=float)
    idx[0] = np.arange(12)
    coeff[0] = 1.0
    for k in range(1, K):
        pk = idx[k - 1]; pc = coeff[k - 1]
        for i in range(8):
            j = (i - shift) % 8
            idx[k, i] = pk[j]
            coeff[k, i] = D8[i] * pc[j]
        for i in range(8, 12):
            idx[k, i] = pk[i]
            coeff[k, i] = pc[i]
    return idx, coeff

def _build_X(S, idx12, coeff12, idx8, coeff8):
    X = np.empty((68, 12))
    for b in range(5):
        sb = S[b]
        X[b * 12:(b + 1) * 12] = coeff12 * sb[idx12]
    s5 = S[5]
    X[60:68] = coeff8 * s5[idx8]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _sym_fg_factory(idx12, coeff12, idx8, coeff8, beta, IU):
    flat12 = idx12.ravel()
    flat8 = idx8.ravel()
    def fg(yf):
        Y = yf.reshape(6, 12)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        X = _build_X(S, idx12, coeff12, idx8, coeff8)
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        Gxt = GX - np.sum(GX * X, axis=1, keepdims=True) * X
        GS = np.zeros((6, 12))
        for b in range(5):
            Gb = Gxt[b * 12:(b + 1) * 12]
            GS[b] = np.bincount(flat12, weights=(coeff12.ravel() * Gb.ravel()), minlength=12)
        G8 = Gxt[60:68]
        GS[5] = np.bincount(flat8, weights=(coeff8.ravel() * G8.ravel()), minlength=12)
        GY = (GS - np.sum(GS * S, axis=1, keepdims=True) * S) / nr
        return F, GY.ravel()
    return fg

def _run_sym(y0, idx12, coeff12, idx8, coeff8, IU, betas, maxiters):
    y = y0.copy()
    for beta, mi in zip(betas, maxiters):
        fg = _sym_fg_factory(idx12, coeff12, idx8, coeff8, beta, IU)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': mi * 3})
            y = r.x.copy()
        except Exception:
            break
    return y

def _best_hole(Xr, rng, ncand=30000):
    d = Xr.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ Xr.T).max(axis=1)
    return cand[int(np.argmin(sc))]

def _opt_single(Xr, z0, beta, maxiter=80):
    def fg(z):
        r = max(float(np.linalg.norm(z)), 1e-12)
        y = z / r
        s = Xr @ y
        m = float(s.max())
        e = np.exp(beta * (s - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        gy = Xr.T @ w
        gz = (gy - float(gy @ y) * y) / r
        return F, gz
    try:
        r = minimize(fg, z0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxfun': maxiter * 2})
        z = r.x
        z /= (np.linalg.norm(z) + 1e-300)
        return z
    except Exception:
        z = z0 / (np.linalg.norm(z0) + 1e-300)
        return z

def solve():
    t0 = time.time()
    TL = 172.0
    rng = np.random.default_rng(20261031)
    cands = []
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if not cands:
        x0 = rng.standard_normal((68, 12))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    x_best = cands[0].copy()
    best_val = _max_cos(x_best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            x_best = c.copy()

    IU = np.triu_indices(68, 1)
    # symmetry scan: signed 12-cycle + 8-cycle block
    def Dvec(neg):
        D = np.ones(12)
        for i in neg:
            D[i] = -1.0
        return D
    def D8vec(neg):
        D = np.ones(8)
        for i in neg:
            D[i] = -1.0
        return D
    Dlist = [Dvec([]), Dvec([0, 6]), Dvec([0, 1]), Dvec([0, 3, 6, 9]),
             Dvec([0, 2, 4, 6]), Dvec([0, 1, 6, 7]), Dvec([0, 5, 6, 11]), Dvec([0, 1, 2, 3])]
    D8list = [D8vec([]), D8vec([0, 4])]
    cfgs = []
    for D in Dlist:
        for D8 in D8list:
            i12, c12 = _precompute12(D, shift=1)
            i8, c8 = _precompute8(D8, shift=1)
            cfgs.append((i12, c12, i8, c8))
    # subsample to fit time but diverse
    keep = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15]
    cfgs = [cfgs[i % len(cfgs)] for i in keep]
    sym_cands = []
    sym_deadline = t0 + 45.0
    for ci, (i12, c12, i8, c8) in enumerate(cfgs):
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            y0 = rng.standard_normal(6 * 12)
            yopt = _run_sym(y0, i12, c12, i8, c8, IU, betas=[70.0, 200.0], maxiters=[60, 80])
            Y = yopt.reshape(6, 12)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            X = _build_X(Y, i12, c12, i8, c8)
            v = _max_cos(X)
            sym_cands.append((v, X, (i12, c12, i8, c8), yopt))
            if v < best_val + 0.004:
                yopt2 = _run_sym(yopt, i12, c12, i8, c8, IU, betas=[450.0], maxiters=[70])
                Y2 = yopt2.reshape(6, 12)
                Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
                X2 = _build_X(Y2, i12, c12, i8, c8)
                v2 = _max_cos(X2)
                sym_cands.append((v2, X2, (i12, c12, i8, c8), yopt2))
        if time.time() > sym_deadline:
            break
    sym_cands.sort(key=lambda t: t[0])
    # release top symmetric candidates
    released = []
    for k in range(min(3, len(sym_cands))):
        if time.time() > t0 + 72.0:
            break
        _, Xs, _, _ = sym_cands[k]
        Xr = Xs.copy()
        for beta, mi in [(150.0, 100), (400.0, 120)]:
            if time.time() > t0 + 72.0:
                break
            Xr = _run_full_lbfgs(Xr, beta, mi)
        if time.time() < t0 + 78.0:
            Xr = _lp_polish(Xr, min(time.time() + 2.5, t0 + 78.0))
        vr = _max_cos(Xr)
        released.append((vr, Xr))
        if vr < best_val:
            best_val = vr
            x_best = Xr.copy()
    pool2 = None
    if released:
        released.sort(key=lambda t: t[0])
        if released[0][0] < best_val + 0.01:
            pool2 = released[0][1].copy()

    # basin hopping
    explore_end = t0 + TL - 22.0
    trial = 0
    while time.time() < explore_end:
        trial += 1
        base = x_best
        if pool2 is not None and (trial % 5 == 0):
            base = pool2
        x = base.copy()
        u = rng.random()
        if u < 0.12:
            # rattler / hole reinsertion on worst points
            pm = _per_max(x)
            order = np.argsort(-pm)[:int(rng.integers(1, 4))]
            for idx in order:
                mask = np.ones(68, dtype=bool); mask[idx] = False
                Xr = x[mask]
                hole = _best_hole(Xr, rng, ncand=30000)
                best_y = x[idx].copy()
                best_m = float((Xr @ best_y).max())
                for beta in (150.0, 450.0):
                    y = _opt_single(Xr, hole.copy(), beta, maxiter=80)
                    m = float((Xr @ y).max())
                    if m < best_m:
                        best_m, best_y = m, y
                    y2 = _opt_single(Xr, best_y.copy() + 0.02 * rng.standard_normal(12), beta, maxiter=40)
                    m2 = float((Xr @ y2).max())
                    if m2 < best_m:
                        best_m, best_y = m2, y2
                x[idx] = best_y / np.linalg.norm(best_y)
            x = _run_full_lbfgs(x, float(rng.uniform(250, 450)), 120)
            if time.time() < explore_end:
                x = _lp_polish(x, min(time.time() + 1.8, explore_end + 5.0))
        elif u < 0.60:
            sigma = float(rng.choice([0.004, 0.010, 0.025, 0.06]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            beta0 = float(rng.uniform(40, 120))
            beta1 = float(rng.uniform(300, 700))
            lr0 = float(rng.uniform(0.02, 0.05))
            lr1 = float(rng.uniform(0.001, 0.005))
            steps = int(rng.integers(350, 700))
            mu = float(rng.uniform(0.75, 0.92))
            x = _quench(x, steps, beta0, beta1, lr0, lr1, mu)
            vq = _max_cos(x)
            if vq < best_val + 0.003:
                x = _run_full_lbfgs(x, float(rng.uniform(280, 500)), 130)
                if time.time() < explore_end:
                    x = _lp_polish(x, min(time.time() + 1.5, explore_end + 5.0))
            else:
                continue
        else:
            sigma = float(rng.choice([0.006, 0.02, 0.05]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _run_full_lbfgs(x, float(rng.uniform(200, 450)), 150)
            if time.time() < explore_end:
                x = _lp_polish(x, min(time.time() + 1.5, explore_end + 5.0))
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
        elif pool2 is not None and v < _max_cos(pool2) and rng.random() < 0.3:
            pool2 = x.copy()
        elif pool2 is None and v < best_val + 0.002:
            pool2 = x.copy()
    # final squeeze
    x_best = _lp_polish(x_best, t0 + TL, r0=6e-4)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
