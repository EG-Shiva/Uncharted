import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _lp_polish(X, deadline, radius0=1e-3, cap=2600):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            o = np.argsort(-c[idx])[:cap]
            mk = np.zeros_like(sel, dtype=bool)
            mk[idx[o]] = True
            sel = mk
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _lse_full(X0, betas, iters, deadline):
    from scipy.optimize import minimize
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y = _norm(X0.copy())
    for beta, mi in zip(betas, iters):
        if time.time() > deadline:
            break
        def fun(Yf):
            Ym = Yf.reshape(n, d)
            nr = np.sqrt(np.sum(Ym * Ym, axis=1, keepdims=True)) + 1e-30
            X = Ym / nr
            G = X @ X.T
            c = G[iu0]
            mv = c.max()
            E = np.exp(beta * (c - mv))
            S = E.sum() + 1e-300
            f = float(mv + np.log(S) / beta)
            W = np.zeros((n, n))
            W[iu0] = E
            W = W + W.T
            GX = (W @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            Gt = GX - dot * X
            GY = Gt / nr
            return f, GY.ravel()
        try:
            res = minimize(fun, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-5, 'maxcor': 10})
            Y = _norm(res.x.reshape(n, d))
        except Exception:
            pass
    return _norm(Y)

def solve():
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.default_rng(20231123)
    n, d, m, ns = 138, 8, 23, 6
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = _norm(vv)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c; best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    incumb = best.copy(); incumb_val = best_val

    speed_choices = [(1,3,5,11),(1,2,4,8),(1,2,5,11),(1,4,6,7),(1,3,4,9),(2,3,6,11),(1,5,7,10),(1,2,3,5)]
    tgrid = np.arange(m, dtype=float)
    betas_sched = [50, 150, 350, 600]
    iters_sched = [100, 100, 130, 130]

    from scipy.optimize import minimize

    def build_X(theta, sp, OT):
        th = theta.reshape(ns, 8)
        z = th[:, :4]; phi = th[:, 4:]
        em = np.exp(z - z.max(axis=1, keepdims=True))
        S = em.sum(axis=1, keepdims=True)
        sprob = em / S
        r = np.sqrt(sprob)
        A = phi[:, :, None] + OT[None, :, :]
        Cos = np.cos(A); Sin = np.sin(A)
        Xr = np.zeros((ns, m, d))
        for pp in range(4):
            Xr[:, :, 2*pp] = r[:, pp, None] * Cos[:, pp, :]
            Xr[:, :, 2*pp+1] = r[:, pp, None] * Sin[:, pp, :]
        X = Xr.reshape(ns*m, d)
        return X, r, sprob, phi, Cos, Sin

    def sym_fg(theta, OT, beta):
        th = theta.reshape(ns, 8)
        z = th[:, :4]; phi = th[:, 4:]
        em = np.exp(z - z.max(axis=1, keepdims=True))
        S = em.sum(axis=1, keepdims=True)
        sprob = em / S
        r = np.sqrt(sprob)
        A = phi[:, :, None] + OT[None, :, :]
        Cos = np.cos(A); Sin = np.sin(A)
        Xr = np.zeros((ns, m, d))
        for pp in range(4):
            Xr[:, :, 2*pp] = r[:, pp, None] * Cos[:, pp, :]
            Xr[:, :, 2*pp+1] = r[:, pp, None] * Sin[:, pp, :]
        X = Xr.reshape(ns*m, d)
        N = ns * m
        iu = np.triu_indices(N, 1)
        G = X @ X.T
        c = G[iu]
        mv = c.max()
        E = np.exp(beta * (c - mv))
        Esum = E.sum() + 1e-300
        f = float(mv + np.log(Esum) / beta)
        W = np.zeros((N, N))
        W[iu] = E
        W = W + W.T
        GX = (W @ X) / Esum
        GXr = GX.reshape(ns, m, d)
        G0 = np.stack([GXr[:, :, 0], GXr[:, :, 2], GXr[:, :, 4], GXr[:, :, 6]], axis=2)
        G1 = np.stack([GXr[:, :, 1], GXr[:, :, 3], GXr[:, :, 5], GXr[:, :, 7]], axis=2)
        Ct = np.transpose(Cos, (0, 2, 1))
        St = np.transpose(Sin, (0, 2, 1))
        gr = np.sum(G0 * Ct + G1 * St, axis=1)
        tmp = np.sum(-G0 * St + G1 * Ct, axis=1)
        gp = r * tmp
        dot = np.sum(gr * r, axis=1, keepdims=True)
        gz = 0.5 * (gr * r - sprob * dot)
        grad = np.concatenate([gz, gp], axis=1).ravel()
        return f, grad, X

    sym_best_X = None; sym_best_val = 1e9
    sym_deadline = t0 + 60.0
    for sp in speed_choices:
        if time.time() > sym_deadline:
            break
        om = 2.0 * np.pi * np.array(sp, dtype=float) / float(m)
        OT = om[:, None] * tgrid[None, :]
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            theta = rng.normal(size=(ns*8,)) * 0.9
            for bi, mi in zip(betas_sched, iters_sched):
                if time.time() > sym_deadline:
                    break
                def fg(th, _OT=OT, _b=bi):
                    f, g, _ = sym_fg(th, _OT, _b)
                    return f, g
                try:
                    res = minimize(fg, theta, jac=True, method='L-BFGS-B',
                                   options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-5, 'maxcor': 10})
                    theta = res.x
                except Exception:
                    break
            try:
                Xc, _, _, _, _, _ = build_X(theta, sp, OT)
                Xc = _norm(Xc)
                v = _max_cos(Xc)
            except Exception:
                continue
            if v < sym_best_val:
                sym_best_val = v; sym_best_X = Xc.copy()

    cands = []
    if sym_best_X is not None:
        cands.append((sym_best_X.copy(), True))
        for sig in (0.002, 0.008, 0.02):
            yp = _norm(sym_best_X + sig * rng.normal(size=(n, d)))
            cands.append((yp, False))
    # always include incumbent kicks as fallback
    for sig in (0.003, 0.010):
        yp = _norm(incumb + sig * rng.normal(size=(n, d)))
        cands.append((yp, False))

    for ci, (X0, is_sym) in enumerate(cands):
        if time.time() > deadline - 22.0:
            break
        if is_sym:
            bts = [80, 160, 320, 550]; mits = [150, 150, 250, 250]
        else:
            # lighter for many cands, heavier if promising
            bts = [120, 320, 550]; mits = [120, 200, 200]
        Xj = _lse_full(X0, bts, mits, deadline - 18.0)
        vj = _max_cos(Xj)
        try:
            Xp = _lp_polish(Xj, min(deadline - 12.0, time.time() + 5.0))
            vp = _max_cos(Xp)
            if vp <= vj:
                Xj, vj = Xp, vp
        except Exception:
            pass
        if vj < best_val - 1e-12:
            best_val = vj; best = Xj.copy()

    # extra direct jam on current best if time remains
    if time.time() < deadline - 12.0:
        try:
            X0 = _norm(best + 0.004 * rng.normal(size=(n, d)))
            Xj = _lse_full(X0, [150, 400], [150, 200], deadline - 9.0)
            Xp = _lp_polish(Xj, deadline - 4.0)
            vp = _max_cos(Xp)
            vj = _max_cos(Xj)
            if vp < vj:
                Xj, vj = Xp, vp
            if vj < best_val:
                best_val = vj; best = Xj.copy()
        except Exception:
            pass
    if best is None or _max_cos(best) > incumb_val + 1e-12:
        best = incumb.copy()
    else:
        try:
            if time.time() < deadline - 3.0:
                Xp = _lp_polish(best, deadline - 1.0)
                if _max_cos(Xp) <= _max_cos(best):
                    best = Xp
        except Exception:
            pass
    best = _norm(best)
    return best
