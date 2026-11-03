import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _lp_polish(X, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2600:
            idx = np.where(sel)[0]
            o = np.argsort(-c[idx])[:2600]
            mk = np.zeros_like(sel, dtype=bool)
            mk[idx[o]] = True
            sel = mk
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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

def _build_symmetric(theta, speeds, m=23, nseeds=6):
    d = 8
    N = nseeds * m
    z = theta.reshape(nseeds, 8)[:, :4]
    phi = theta.reshape(nseeds, 8)[:, 4:]
    em = np.exp(z - z.max(axis=1, keepdims=True))
    S = em.sum(axis=1, keepdims=True)
    sprob = em / S
    r = np.sqrt(sprob)
    om = 2.0 * np.pi * np.array(speeds, dtype=float) / float(m)
    X = np.zeros((N, d))
    for k in range(nseeds):
        for t in range(m):
            idx = k * m + t
            for p in range(4):
                a = phi[k, p] + om[p] * t
                X[idx, 2 * p] = r[k, p] * np.cos(a)
                X[idx, 2 * p + 1] = r[k, p] * np.sin(a)
    return X, r, sprob, phi, om

def _sym_fun_grad(theta, speeds, beta, m=23, nseeds=6):
    X, r, sprob, phi, om = _build_symmetric(theta, speeds, m, nseeds)
    N = X.shape[0]
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
    GX = (W @ X) / Esum  # d f / d X
    gr = np.zeros((nseeds, 4))
    gp = np.zeros((nseeds, 4))
    for k in range(nseeds):
        for p in range(4):
            sr = 0.0
            sp = 0.0
            rk = r[k, p]
            for t in range(m):
                idx = k * m + t
                a = phi[k, p] + om[p] * t
                ca = np.cos(a); sa = np.sin(a)
                g0 = GX[idx, 2 * p]; g1 = GX[idx, 2 * p + 1]
                sr += g0 * ca + g1 * sa
                sp += g0 * (-rk * sa) + g1 * (rk * ca)
            gr[k, p] = sr
            gp[k, p] = sp
    gz = np.zeros((nseeds, 4))
    for k in range(nseeds):
        rk = r[k, :]
        s = sprob[k, :]
        dot = float(np.sum(gr[k, :] * rk))
        for q in range(4):
            gz[k, q] = 0.5 * (gr[k, q] * rk[q] - s[q] * dot)
    grad = np.concatenate([gz, gp], axis=1).ravel()
    return f, grad

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
    rng = np.random.default_rng(12345)
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

    # --- symmetric search ---
    from scipy.optimize import minimize
    speed_choices = [(1,3,5,11),(1,2,4,8),(1,2,5,11),(1,4,6,7),(1,3,4,9),(2,3,6,11),(1,5,7,10)]
    sym_best_X = None; sym_best_val = 1e9; sym_best_th = None; sym_best_sp = None
    betas_sched = [50, 150, 350, 600]
    iters_sched = [120, 120, 150, 150]
    trial = 0
    sym_deadline = t0 + 70.0
    for sp in speed_choices:
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            trial += 1
            theta = rng.normal(size=(ns * 8,)) * 0.8
            # ensure radii logits not degenerate
            for bi, mi in zip(betas_sched, iters_sched):
                if time.time() > sym_deadline:
                    break
                def fg(th, _sp=sp, _b=bi):
                    return _sym_fun_grad(th, _sp, _b, m, ns)
                try:
                    res = minimize(fg, theta, jac=True, method='L-BFGS-B',
                                   options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-5, 'maxcor': 10})
                    theta = res.x
                except Exception:
                    break
            try:
                Xc, _, _, _, _ = _build_symmetric(theta, sp, m, ns)
                Xc = _norm(Xc)
                v = _max_cos(Xc)
            except Exception:
                continue
            if v < sym_best_val:
                sym_best_val = v; sym_best_X = Xc.copy(); sym_best_th = theta.copy(); sym_best_sp = sp
        if time.time() > sym_deadline:
            break

    # --- release symmetry ---
    refined = None; refined_val = 1e9
    if sym_best_X is not None:
        cands = []
        cands.append(sym_best_X.copy())
        # small perturbations to break symmetry
        for sig in (0.002, 0.008):
            yp = sym_best_X + sig * rng.normal(size=(n, d))
            cands.append(_norm(yp))
        for ci, X0 in enumerate(cands):
            if time.time() > deadline - 25.0:
                break
            # LSE jam with moderate schedule (lighter for extra cands)
            if ci == 0:
                bts = [80, 160, 320, 550]; mits = [150, 150, 250, 250]
            else:
                bts = [120, 320, 550]; mits = [120, 200, 200]
            Xj = _lse_full(X0, bts, mits, deadline - 20.0)
            vj = _max_cos(Xj)
            # quick LP polish
            try:
                Xp = _lp_polish(Xj, min(deadline - 12.0, time.time() + 4.0))
                vp = _max_cos(Xp)
                if vp <= vj:
                    Xj, vj = Xp, vp
            except Exception:
                pass
            if vj < refined_val:
                refined_val = vj; refined = Xj.copy()
            # keep global incumbent update
            if vj < best_val:
                best_val = vj; best = Xj.copy()
    # if symmetric path failed to beat incumbent, do a short direct polish of incumbent
    if time.time() < deadline - 10.0 and best_val >= incumb_val - 1e-12:
        # try one perturbed incumbent jam to ensure we return at least incumbent quality
        try:
            X0 = incumb + 0.004 * rng.normal(size=(n, d))
            X0 = _norm(X0)
            Xj = _lse_full(X0, [150, 400], [150, 200], deadline - 10.0)
            vj = _max_cos(Xj)
            if vj < best_val:
                best_val = vj; best = Xj.copy()
        except Exception:
            pass
    if best is None:
        best = incumb.copy()
    else:
        # retain only if beats incumbent (allow tiny tolerance); else return incumbent polished
        if _max_cos(best) > incumb_val and refined_val > incumb_val:
            best = incumb.copy()
    try:
        # final light LP if time remains and we have improvement
        if time.time() < deadline - 3.0 and _max_cos(best) < incumb_val + 1e-9:
            best = _lp_polish(best, deadline - 1.0)
    except Exception:
        pass
    best = _norm(best)
    return best
