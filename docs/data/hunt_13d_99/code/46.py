import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _full_anneal(X0, alphas, miters, t_deadline):
    N, d = X0.shape
    U0 = X0.copy().ravel().astype(float)
    Xcur = X0.copy()
    for alpha, mit in zip(alphas, miters):
        if time.time() > t_deadline:
            break
        a = float(alpha)
        def fun(Uflat, _a=a):
            U = Uflat.reshape(N, d)
            nr = np.linalg.norm(U, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            X = U / nr
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp(_a * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5) + 1e-300
            L = m + np.log(S) / _a
            GX = (E @ X) / S
            dot = np.sum(GX * X, axis=1, keepdims=True)
            GU = (GX - dot * X) / nr
            return float(L), GU.ravel().astype(float)
        try:
            res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                           options={'maxiter': int(mit), 'ftol': 1e-20,
                                    'gtol': 1e-7, 'maxfun': int(mit) * 6, 'maxcor': 8})
            U0 = res.x.copy()
        except Exception:
            break
        U = U0.reshape(N, d)
        Xcur = _norm_rows(U)
    return Xcur

def _lp_polish(x, seconds, t_hard):
    x = _norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 6e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and time.time() < t_hard and radius > 1e-14 and it < 18:
        it += 1
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thr = max(25.0 * radius, 5e-7)
        sel = c > m - thr
        if int(np.sum(sel)) < 900:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:900]] = True
            sel = sel | s2
        if int(np.sum(sel)) > 2600:
            o = np.argsort(-c)
            s2 = np.zeros_like(sel, dtype=bool)
            s2[o[:2600]] = True
            sel = s2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        cs = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={'time_limit': 4.0})
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
            x = y
            radius = min(radius * 1.4, 0.008)
        else:
            radius *= 0.5
    return bx, best

# ---------- symmetry machinery ----------
def _rot_apply(V, c, s):
    # V: (...,P,2)
    x0 = V[..., 0]; x1 = V[..., 1]
    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1
    return np.stack([y0, y1], axis=-1)

def _sym_system(m, P):
    F = 13 - 2 * P
    th = 2 * np.pi * np.arange(m) / float(m)
    cg = np.cos(th); sg = np.sin(th)
    return F, cg, sg

def _build_X_sym(Yf, Yx, m, P, F, a, b):
    N = a * m + b
    X = np.zeros((N, 13))
    if a > 0:
        Yp = Yf[:, :2*P].reshape(a, P, 2) if P > 0 else np.zeros((a, 0, 2))
        Yc = Yf[:, 2*P:]  # (a,F)
        _, cg, sg = _sym_system(m, P)
        for g in range(m):
            c = cg[g]; s = sg[g]
            if P > 0:
                Rp = _rot_apply(Yp, c, s).reshape(a, 2*P)
                X[np.arange(a)*m+g, :2*P] = Rp
            if F > 0:
                X[np.arange(a)*m+g, 2*P:] = Yc
    if b > 0:
        base = a * m
        if F == 1:
            X[base:base+b, 2*P:] = 1.0  # pole(s)
        else:
            X[base:base+b, :2*P] = 0.0
            X[base:base+b, 2*P:] = Yx
    return X

def _sym_fun_factory(m, P, F, a, b, cg, sg):
    # returns fun(Uflat) -> (L, grad)
    nfix_vars = 0 if (b == 0 or F <= 1) else b * F
    nfull_vars = a * 13
    def fun(Uflat):
        # split
        Uf = Uflat[:nfull_vars].reshape(a, 13) if a > 0 else np.zeros((0, 13))
        if a > 0:
            nf = np.linalg.norm(Uf, axis=1, keepdims=True); nf[nf == 0] = 1.0
            Yf = Uf / nf
        else:
            Yf = np.zeros((0, 13)); nf = np.ones((1, 1))
        if b > 0 and F > 1:
            Ux = Uflat[nfull_vars:].reshape(b, F)
            nx = np.linalg.norm(Ux, axis=1, keepdims=True); nx[nx == 0] = 1.0
            Yx = Ux / nx
        elif b > 0 and F == 1:
            Yx = np.ones((b, 1))
            nx = np.ones((b, 1))
            Ux = Yx
        else:
            Yx = np.zeros((0, max(F, 1))); nx = np.ones((1, 1))
        X = _build_X_sym(Yf, Yx, m, P, F, a, b)
        N = X.shape[0]
        # caller passes alpha via closure attr
        alpha = fun.alpha
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = mx + np.log(S) / alpha
        GX = (E @ X) / S  # (N,13)
        # accumulate to representatives
        if a > 0:
            gYf = np.zeros_like(Yf)
            Yp_dummy = None
            for g in range(m):
                rows = np.arange(a) * m + g
                V = GX[rows]  # (a,13)
                # rotate plane part by -g
                c = cg[g]; s = -sg[g]
                if P > 0:
                    Vp = V[:, :2*P].reshape(a, P, 2)
                    Rp = _rot_apply(Vp, c, s).reshape(a, 2*P)
                    gYf[:, :2*P] += Rp
                if F > 0:
                    gYf[:, 2*P:] += V[:, 2*P:]
            dotf = np.sum(gYf * Yf, axis=1, keepdims=True)
            gUf = (gYf - dotf * Yf) / nf
        else:
            gUf = np.zeros((0, 13))
        if b > 0 and F > 1:
            base = a * m
            GXf = GX[base:base+b, 2*P:]  # (b,F)
            dotx = np.sum(GXf * Yx, axis=1, keepdims=True)
            gUx = (GXf - dotx * Yx) / nx
        else:
            gUx = np.zeros((0,) if nfix_vars == 0 else (nfix_vars,))
            if nfix_vars > 0:
                gUx = gUx.reshape(-1)
            else:
                gUx = np.zeros((0,))
        grad = np.concatenate([gUf.ravel(), np.atleast_1d(gUx.ravel())]) if (nfull_vars + nfix_vars) > 0 else np.zeros_like(Uflat)
        return float(L), grad.astype(float)
    return fun

def _symmetrised_start(B, m, P, F, a, b, cg, sg, rng):
    N = B.shape[0]
    # choose fixed seeds: smallest plane norm
    perm = np.arange(N)
    if b > 0 and F > 1:
        pn = np.linalg.norm(B[:, :2*P], axis=1) if P > 0 else np.zeros(N)
        idx_fix = np.argsort(pn)[:b]
        mask = np.ones(N, dtype=bool); mask[idx_fix] = False
        Brem = B[~mask]
        Yx0 = B[idx_fix, 2*P:]
        Yx0 = _norm_rows(Yx0 + 1e-9 * rng.standard_normal(size=Yx0.shape))
    elif b > 0 and F == 1:
        pn = np.linalg.norm(B[:, :2*P], axis=1) if P > 0 else np.zeros(N)
        idx_fix = np.argsort(pn)[:b]
        mask = np.ones(N, dtype=bool); mask[idx_fix] = False
        Brem = B[~mask]
        Yx0 = np.ones((b, 1))
    else:
        Brem = B.copy()
        Yx0 = np.zeros((0, max(F, 1)))
        # random permute
    # random partition of Brem into a groups of size m
    rp = rng.permutation(Brem.shape[0])
    Brem = Brem[rp]
    Yf0 = np.zeros((a, 13))
    for o in range(a):
        grp = Brem[o*m:(o+1)*m]
        if grp.shape[0] < m or grp.shape[0] == 0:
            Yf0[o] = _norm_rows(rng.standard_normal(size=(1, 13)))[0]
            continue
        p0 = grp[0]
        p0p = p0[:2*P].reshape(P, 2) if P > 0 else np.zeros((0, 2))
        p0c = p0[2*P:]
        acc = np.zeros(13)
        for p in grp:
            pp = p[:2*P].reshape(P, 2) if P > 0 else np.zeros((0, 2))
            pc = p[2*P:]
            bestd = -1e100; bestg = 0
            # brute g
            for g in range(m):
                c = cg[g]; s = sg[g]
                # R_{-g} p dot p0  <=> (R_{-g}pp)·p0p + pc·p0c ; R_{-g} uses -s
                if P > 0:
                    # rotate p by -g
                    x0 = pp[:, 0]; x1 = pp[:, 1]
                    y0 = c * x0 + s * x1
                    y1 = -s * x0 + c * x1
                    d = float(np.sum(y0 * p0p[:, 0] + y1 * p0p[:, 1]))
                else:
                    d = 0.0
                d += float(np.dot(pc, p0c))
                if d > bestd:
                    bestd = d; bestg = g
            c = cg[bestg]; s = sg[bestg]
            if P > 0:
                x0 = pp[:, 0]; x1 = pp[:, 1]
                y0 = c * x0 + s * x1
                y1 = -s * x0 + c * x1
                acc[:2*P] += np.concatenate([y0[:, None], y1[:, None]], axis=1).reshape(-1) if P > 0 else 0
            acc[2*P:] += pc
        nrm = float(np.linalg.norm(acc))
        if nrm < 1e-8 or not np.isfinite(nrm):
            acc = rng.standard_normal(size=13)
            nrm = float(np.linalg.norm(acc))
        Yf0[o] = acc / max(nrm, 1e-12)
    Yf0 = _norm_rows(Yf0 + 1e-12 * rng.standard_normal(size=Yf0.shape)) if a > 0 else Yf0
    return Yf0, Yx0

def solve():
    t0 = time.time()
    T_HARD = t0 + 172.0
    T_SYM_END = t0 + 88.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
    xb = _norm_rows(xb)
    best_x = xb.copy()
    best_val = _max_cos(best_x)

    groups = [
        (11, 6, 9, 0),
        (9, 6, 11, 0),
        (3, 6, 33, 0),
        (2, 6, 49, 1),
        (16, 5, 6, 3),
        (12, 5, 8, 3),
    ]
    sym_bests = []  # (val, X)
    sym_bests.append((best_val, best_x.copy()))

    for (m, P, a, b) in groups:
        if time.time() > T_SYM_END - 2.0:
            break
        g_dead = min(time.time() + 14.0, T_SYM_END)
        F, cg, sg = _sym_system(m, P)
        assert a * m + b == 99, "bad decomp"
        fun = _sym_fun_factory(m, P, F, a, b, cg, sg)
        nfull = a * 13
        nfix = 0 if (b == 0 or F <= 1) else b * F
        # starts
        starts = []
        try:
            Yf0, Yx0 = _symmetrised_start(best_x, m, P, F, a, b, cg, sg, rng)
            starts.append((Yf0, Yx0, "sym-best"))
        except Exception:
            pass
        try:
            # second symmetrised variant from perturbed best
            Bp = _norm_rows(best_x + rng.standard_normal(size=(99, 13)) * 0.02)
            Yf1, Yx1 = _symmetrised_start(Bp, m, P, F, a, b, cg, sg, rng)
            starts.append((Yf1, Yx1, "sym-pert"))
        except Exception:
            pass
        try:
            Yfr = _norm_rows(rng.standard_normal(size=(a, 13))) if a > 0 else np.zeros((0, 13))
            if b > 0 and F > 1:
                Yxr = _norm_rows(rng.standard_normal(size=(b, F)))
            elif b > 0 and F == 1:
                Yxr = np.ones((b, 1))
            else:
                Yxr = np.zeros((0, 1))
            starts.append((Yfr, Yxr, "rand"))
        except Exception:
            pass
        alphas = [20.0, 70.0, 180.0]
        miters = [45, 45, 40]
        for (Yf0, Yx0, tag) in starts:
            if time.time() > g_dead - 1.0:
                break
            U0 = np.concatenate([Yf0.ravel(), (Yx0.ravel() if nfix > 0 else np.zeros(0))]).astype(float)
            if U0.size == 0:
                continue
            for ai, (al, mi) in enumerate(zip(alphas, miters)):
                if time.time() > g_dead - 0.5:
                    break
                fun.alpha = float(al)
                try:
                    res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                                   options={'maxiter': int(mi), 'ftol': 1e-20, 'gtol': 1e-6,
                                            'maxfun': int(mi) * 6, 'maxcor': 8})
                    U0 = res.x.copy()
                except Exception:
                    break
            # evaluate
            try:
                Uf = U0[:nfull].reshape(a, 13) if a > 0 else np.zeros((0, 13))
                Yf = _norm_rows(Uf) if a > 0 else Uf
                if b > 0 and F > 1:
                    Yx = _norm_rows(U0[nfull:].reshape(b, F))
                elif b > 0 and F == 1:
                    Yx = np.ones((b, 1))
                else:
                    Yx = np.zeros((0, 1))
                X = _build_X_sym(Yf, Yx, m, P, F, a, b)
                X = _norm_rows(X)
                v = _max_cos(X)
                sym_bests.append((v, X.copy()))
                if v < best_val - 1e-12:
                    best_val = v
                    best_x = X.copy()
            except Exception:
                continue
        # end starts for group
    # order sym candidates
    sym_bests.sort(key=lambda t: t[0])
    # stage: release symmetry in stages - brief annealed full optimisation
    # first allow fixed coords (if any): just full optimise top candidates
    T_REL_END = min(T_HARD - 28.0, t0 + 140.0)
    cand_pool = []
    seen = set()
    for v, X in sym_bests[:4]:
        cand_pool.append(X.copy())
        if len(cand_pool) >= 3:
            break
    # add incumbent
    # release: full anneal on each candidate, time-boxed
    for ci, Xc in enumerate(list(cand_pool)):
        if time.time() > T_REL_END:
            break
        # tiny de-symmetrisation shake to break exact symmetry
        amp = 1e-4 if ci == 0 else 3e-3
        X0 = _norm_rows(Xc + rng.standard_normal(size=(99, 13)) * amp)
        # short anneal
        X1 = _full_anneal(X0, [30.0, 120.0, 350.0], [60, 60, 50], min(T_REL_END, T_HARD - 28.0))
        v1 = _max_cos(X1)
        sym_bests.append((v1, X1.copy()))
        if v1 < best_val:
            best_val = v1
            best_x = X1.copy()
    # re-sort, keep best
    sym_bests.sort(key=lambda t: t[0])
    if sym_bests[0][0] < best_val:
        best_val = sym_bests[0][0]
        best_x = sym_bests[0][1].copy()
    # final wide-active LP polish on best
    try:
        if time.time() < T_HARD - 8.0:
            bx2, bv2 = _lp_polish(best_x, seconds=min(20.0, T_HARD - time.time() - 4.0), t_hard=T_HARD - 3.0)
            if bv2 < best_val:
                best_val = bv2
                best_x = bx2.copy()
    except Exception:
        pass
    # final light full touch if time remains
    try:
        if time.time() < T_HARD - 6.0:
            Xf = _full_anneal(best_x, [80.0, 300.0], [40, 40], T_HARD - 4.0)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best_x = Xf.copy()
                try:
                    bx3, bv3 = _lp_polish(best_x, seconds=min(6.0, T_HARD - time.time() - 1.5), t_hard=T_HARD - 1.0)
                    if bv3 < best_val:
                        best_val = bv3
                        best_x = bx3.copy()
                except Exception:
                    pass
    except Exception:
        pass
    best_x = _norm_rows(np.asarray(best_x, dtype=float))
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    best_x = _norm_rows(best_x)
    return best_x
