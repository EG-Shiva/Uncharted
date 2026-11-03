import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm_rows(V):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return V / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

# ---------- full LogSumExp polish ----------
def _full_fun(alpha):
    def fun(v):
        V = v.reshape(99, 13)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr = np.where(nr < 1e-15, 1.0, nr)
        Y = V / nr
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        Sm = E.sum() / 2.0 + 1e-300
        L = mx + np.log(Sm) / alpha
        gY = (E @ Y) / Sm
        dot = (gY * Y).sum(axis=1, keepdims=True)
        gV = (gY - dot * Y) / nr
        return float(L), gV.ravel().astype(float)
    return fun

def _run_full_lse(X0, alphas, iters):
    x = _norm_rows(X0.copy()).ravel().astype(float).copy()
    for a, mi in zip(alphas, iters):
        try:
            r = minimize(_full_fun(a), x, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(mi), 'ftol': 1e-20, 'gtol': 1e-9, 'maxcor': 8})
            x = r.x.copy()
        except Exception:
            break
    X = _norm_rows(x.reshape(99, 13))
    return X

# ---------- reflection symmetric manifold ----------
def _refl_fun_factory(rvec, alpha):
    r = rvec  # (13,) +-1
    def fun(v):
        off = 49 * 13
        Vf = v[:off].reshape(49, 13)
        Vz = v[off:]
        nr = np.linalg.norm(Vf, axis=1, keepdims=True)
        nr = np.where(nr < 1e-15, 1.0, nr)
        Y = Vf / nr
        nz = float(np.linalg.norm(Vz))
        if nz < 1e-15:
            nz = 1.0
        Zs = Vz / nz
        # build full 99x13
        RY = Y * r[None, :]
        Zfull = np.zeros(13)
        # fixed coords are those with r==+1; flipped coords zero
        Zfull[r > 0] = Zs[:np.count_nonzero(r > 0)] if Zs.size >= np.count_nonzero(r > 0) else 0
        # if flip size bookkeeping mismatched, embed safely
        X = np.vstack([Y, RY, Zfull[None, :]])
        # normalize (Y already, RY already, Zfull already unit if correct)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        Sm = E.sum() / 2.0 + 1e-300
        L = mx + np.log(Sm) / alpha
        gX = (E @ X) / Sm
        gY0 = gX[:49]
        gY1 = gX[49:98] * r[None, :]
        H = gY0 + gY1
        dot = (H * Y).sum(axis=1, keepdims=True)
        gVf = (H - dot * Y) / nr
        gZfull = gX[98]
        gZsub = gZfull[r > 0]
        # chain through Zs normalization
        dotz = float(np.dot(gZsub, Zs[:gZsub.size])) if gZsub.size > 0 else 0.0
        if Zs.size == gZsub.size:
            gVz = (gZsub - dotz * Zs) / nz
        else:
            gVz = np.zeros_like(Vz)
            m = min(gZsub.size, Vz.size)
            gVz[:m] = (gZsub[:m] - dotz * Zs[:m]) / nz if m > 0 else 0
        g = np.concatenate([gVf.ravel(), gVz.ravel()]).astype(float)
        return float(L), g
    return fun

def _init_refl_from_best(xb, f, rng):
    r = np.ones(13)
    r[:f] = -1.0
    idx = rng.choice(99, size=49, replace=False)
    Y0 = xb[idx].copy() + 0.01 * rng.standard_normal(size=(49, 13))
    Y0 = _norm_rows(Y0)
    # fixed point: mean of rest projected onto r==+1 subspace
    mask = np.arange(99)
    mask = np.array([i for i in range(99) if i not in set(idx.tolist())])
    if len(mask) > 0:
        m = xb[mask].mean(axis=0)
        m[r < 0] = 0.0
        if np.linalg.norm(m) < 1e-8:
            m = np.zeros(13); m[r > 0] = rng.standard_normal(size=int(np.count_nonzero(r > 0)))
        zfull = m / max(np.linalg.norm(m), 1e-15)
        zsub = zfull[r > 0].copy()
    else:
        zsub = rng.standard_normal(size=int(np.count_nonzero(r > 0)))
        zsub = zsub / max(np.linalg.norm(zsub), 1e-15)
    x0 = np.concatenate([Y0.ravel(), zsub.ravel()]).astype(float)
    return x0, r

def _optim_reflect(xb, f, rng, deadline):
    x0, r = _init_refl_from_best(xb, f, rng)
    x = x0.copy()
    for alpha, mi in [(40.0, 60), (150.0, 80), (500.0, 100)]:
        if time.time() > deadline:
            break
        fun = _refl_fun_factory(r, alpha)
        try:
            res = minimize(fun, x, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-20, 'gtol': 1e-9, 'maxcor': 8})
            x = res.x.copy()
        except Exception:
            break
    # expand
    off = 49 * 13
    Vf = x[:off].reshape(49, 13)
    Vz = x[off:]
    Y = _norm_rows(Vf)
    RY = Y * r[None, :]
    nz = max(float(np.linalg.norm(Vz)), 1e-15)
    Zs = Vz / nz
    Zfull = np.zeros(13)
    Zfull[r > 0] = Zs
    Zfull = Zfull / max(float(np.linalg.norm(Zfull)), 1e-15)
    X = np.vstack([Y, RY, Zfull[None, :]])
    return _norm_rows(X)

# ---------- C11 cyclic manifold ----------
def _build_cs(m, mults):
    th = 2 * np.pi * np.arange(m)[:, None] * np.array(mults, dtype=float)[None, :] / float(m)
    return np.cos(th), np.sin(th)

def _gen_c11(Yf, C, S):
    m = C.shape[0]; nf = Yf.shape[0]; p = C.shape[1]
    Xf = np.zeros((nf * m, 13))
    Yfix = Yf[:, 2 * p:2 * p + 1]  # last coord
    for g in range(m):
        sl = slice(g * nf, (g + 1) * nf)
        for pi in range(p):
            c = C[g, pi]; s = S[g, pi]
            y0 = Yf[:, 2 * pi]; y1 = Yf[:, 2 * pi + 1]
            Xf[sl, 2 * pi] = c * y0 - s * y1
            Xf[sl, 2 * pi + 1] = s * y0 + c * y1
        Xf[sl, 12] = Yf[:, 12]
    return Xf

def _accum_c11(gXf, nf, m, p, C, S):
    gY = np.zeros((nf, 13))
    for g in range(m):
        blk = gXf[g * nf:(g + 1) * nf]
        for pi in range(p):
            c = C[g, pi]; s = S[g, pi]
            a0 = blk[:, 2 * pi]; a1 = blk[:, 2 * pi + 1]
            gY[:, 2 * pi] += c * a0 + s * a1
            gY[:, 2 * pi + 1] += -s * a0 + c * a1
        gY[:, 12] += blk[:, 12]
    return gY

def _rot_back_one_c11(v, g, p, C, S):
    y = np.zeros(13)
    for pi in range(p):
        c = C[g, pi]; s = S[g, pi]
        b0 = v[2 * pi]; b1 = v[2 * pi + 1]
        y[2 * pi] = c * b0 + s * b1
        y[2 * pi + 1] = -s * b0 + c * b1
    y[12] = v[12]
    return y

def _fit_c11(B, mults, rng, iters=8):
    m = 11; p = 6; nf = 9
    C, S = _build_cs(m, mults)
    idx = rng.choice(B.shape[0], size=nf, replace=False)
    Yf = _norm_rows(B[idx].copy())
    for _ in range(iters):
        Xf = _gen_c11(Yf, C, S)
        G = B @ Xf.T
        assign = np.argmax(G, axis=1)
        acc = np.zeros_like(Yf); cnt = np.zeros(nf)
        sums = [np.zeros(13) for _ in range(nf)]
        cc = [0] * nf
        for i in range(B.shape[0]):
            a = int(assign[i]); gg = a // nf; jj = a % nf
            sums[jj] += _rot_back_one_c11(B[i], gg, p, C, S)
            cc[jj] += 1
        for j in range(nf):
            if cc[j] > 0:
                s = sums[j] / cc[j]
                n = np.linalg.norm(s)
                if n > 1e-12:
                    Yf[j] = s / n
    return Yf, C, S

def _optim_c11(xb, mults, rng, deadline):
    m = 11; p = 6; nf = 9
    try:
        Yf0, C, S = _fit_c11(xb, mults, rng, iters=6)
    except Exception:
        C, S = _build_cs(m, mults)
        Yf0 = _norm_rows(rng.standard_normal(size=(nf, 13)))
    Yf0 = Yf0 + 0.005 * rng.standard_normal(size=Yf0.shape)
    Yf0 = _norm_rows(Yf0)
    x = Yf0.ravel().astype(float).copy()

    def make_fun(alpha):
        def fun(v):
            Vf = v.reshape(nf, 13)
            nr = np.linalg.norm(Vf, axis=1, keepdims=True)
            nr = np.where(nr < 1e-15, 1.0, nr)
            Yf = Vf / nr
            X = _gen_c11(Yf, C, S)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            mx = float(G.max())
            E = np.exp(alpha * (G - mx))
            np.fill_diagonal(E, 0.0)
            Sm = E.sum() / 2.0 + 1e-300
            L = mx + np.log(Sm) / alpha
            gX = (E @ X) / Sm
            gY = _accum_c11(gX, nf, m, p, C, S)
            dot = (gY * Yf).sum(axis=1, keepdims=True)
            gV = (gY - dot * Yf) / nr
            return float(L), gV.ravel().astype(float)
        return fun
    for alpha, mi in [(40.0, 60), (150.0, 80), (500.0, 100)]:
        if time.time() > deadline:
            break
        try:
            res = minimize(make_fun(alpha), x, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-20, 'gtol': 1e-9, 'maxcor': 8})
            x = res.x.copy()
        except Exception:
            break
    Vf = x.reshape(nf, 13)
    Yf = _norm_rows(Vf)
    X = _gen_c11(Yf, C, S)
    return _norm_rows(X)

# ---------- short LP polish ----------
def _lp_polish_short(X0, time_budget, deadline):
    X = _norm_rows(X0.copy())
    n, d = X.shape
    best = _max_cos(X)
    t0 = time.time()
    radius = 8e-4
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - t0 < time_budget and time.time() < deadline:
        if radius < 1e-12:
            break
        G = X @ X.T
        c = G[iu0, iu1]
        mcur = float(c.max())
        thr = mcur - max(20 * radius, 3e-6)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        if p == 0:
            break
        if p > 1800:
            order = np.argpartition(c[sel], -1800)[-1800:]
            I = I[order]; J = J[order]; p = len(I)
            bub = -np.array([float(X[I[k]] @ X[J[k]]) for k in range(p)])
        else:
            bub = -c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm_rows(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
        if p < 50:
            break
    return X, best

def solve():
    t_start = time.time()
    deadline = t_start + 168.0
    rng = np.random.RandomState(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        try:
            xb = np.asarray(xb, dtype=float)
            if xb.shape == (99, 13):
                cands.append(_norm_rows(xb))
        except Exception:
            pass
    if xr is not None:
        try:
            xr = np.asarray(xr, dtype=float)
            if xr.shape == (99, 13):
                cands.append(_norm_rows(xr))
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    sym_cands = []
    sym_end = t_start + 88.0
    # reflection patterns: several flip sizes
    for f in [6, 4, 2, 1]:
        if time.time() > sym_end or time.time() > deadline - 75:
            break
        try:
            Xs = _optim_reflect(x_best, f, rng, deadline)
            v = _max_cos(Xs)
            sym_cands.append((v, Xs, f"refl{f}"))
            if v < best_val:
                best_val = v
                x_best = Xs.copy()
        except Exception:
            continue
    # C11 patterns
    for mults in [[1, 2, 3, 4, 5, 6], [1, 2, 4, 7, 8, 10]]:
        if time.time() > sym_end or time.time() > deadline - 70:
            break
        try:
            Xs = _optim_c11(x_best, mults, rng, deadline)
            v = _max_cos(Xs)
            sym_cands.append((v, Xs, f"c11{mults[2]}"))
            if v < best_val:
                best_val = v
                x_best = Xs.copy()
        except Exception:
            continue
    # release symmetry in stages for top candidates
    sym_cands.sort(key=lambda t: t[0])
    top = sym_cands[:2] if len(sym_cands) >= 2 else sym_cands
    # keep unperturbed fallbacks already in best
    for v0, Xs0, _nm in top:
        for sigma in [1e-4, 1e-3]:
            if time.time() > deadline - 30:
                break
            try:
                Xp = _norm_rows(Xs0 + sigma * rng.standard_normal(size=(99, 13)))
                Xp = _run_full_lse(Xp, [120.0, 450.0], [90, 110])
                v = _max_cos(Xp)
                if v < best_val:
                    best_val = v
                    x_best = Xp.copy()
                # short LP on the perturbed-polished point if promising
                if v < best_val + 5e-5 and time.time() < deadline - 35:
                    Xlp, vlp = _lp_polish_short(Xp, 4.0, deadline - 18)
                    # final tiny LSE to smooth
                    if vlp < v:
                        Xlp2 = _run_full_lse(Xlp, [600.0], [60])
                        vlp2 = _max_cos(Xlp2)
                        if vlp2 < best_val:
                            best_val = vlp2
                            x_best = Xlp2.copy()
                        elif vlp < best_val:
                            best_val = vlp
                            x_best = Xlp.copy()
            except Exception:
                continue
    # final polish of global best
    try:
        if time.time() < deadline - 20:
            Xf = _run_full_lse(x_best, [500.0], [80])
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                x_best = Xf.copy()
    except Exception:
        pass
    try:
        if time.time() < deadline - 12:
            Xlp, vlp = _lp_polish_short(x_best, 6.0, deadline - 4)
            if vlp < best_val:
                x_best = Xlp.copy()
                best_val = vlp
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    return x_best
