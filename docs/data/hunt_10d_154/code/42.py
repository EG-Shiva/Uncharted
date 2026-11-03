import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    dd = d - 1
    T = np.empty((n, d, dd), dtype=float)
    e = np.zeros(d, dtype=float)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        w = x.copy()
        w[k] -= 1.0
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            # fallback: random orthogonal
            q, _ = np.linalg.qr(np.random.randn(d, d))
            # make first col align with x
            T[i] = q[:, 1:]
            continue
        w /= nw
        H = np.eye(d) - 2.0 * np.outer(w, w)
        # k-th column of H is x (up to sign), others span tangent
        cols = [c for c in range(d) if c != k]
        T[i] = H[:, cols]
        # re-orthogonalize against x numerically
        # T should already be orthogonal; clean:
        # project out x component
        corr = T[i].T @ x
        T[i] -= np.outer(x, corr)
        # renormalize via QR
        Q, _ = np.linalg.qr(T[i])
        T[i] = Q
    return T

def _smooth_refine(X, steps, a0, a1, lr0, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0, la1 = np.log(a0), np.log(a1)
    for t in range(1, steps + 1):
        if (t % 200 == 0) and (time.time() > deadline):
            break
        frac = t / max(steps, 1)
        alpha = np.exp(la0 + (la1 - la0) * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S == 0:
            break
        Grad = (2.0 / S) * (E @ X)
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))[:, None]
    return X

def _lp_polish(X, seconds, deadline=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (X @ X.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap active set for speed
        if p > 4000:
            # keep tightest 4000
            idx = np.argpartition(-c[sel], 4000)[:4000]
            I, J = I[idx], J[idx]
            p = 4000
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel] if p != 4000 else - (X @ X.T)[iu][sel][idx] if False else None,
                          b_ub=None, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        # NOTE: above b_ub handling: recompute correctly
        # fallback recompute if needed (cheap)
        # we already built A_ub with -c[sel] mismatch when truncated; fix by rebuilding b
        # To avoid bug, recompute b vector:
        try:
            if p == 4000:
                # need b for truncated set
                Gall = (X @ X.T)[iu]
                # reselect already done; need c values for I,J directly
                bvec = -np.sum(X[I] * X[J], axis=1)
            else:
                bvec = -c[sel]
            # if first linprog used wrong b, rerun once with correct b when truncated
            if p == 4000:
                res = linprog(obj, A_ub=A_ub, b_ub=bvec, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            else:
                # first call had b_ub=None bug, rerun correctly
                res = linprog(obj, A_ub=A_ub, b_ub=bvec, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            X, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _lp_short(X, seconds, deadline):
    # thin wrapper ensuring deadline
    try:
        return _lp_polish(X, seconds, deadline=deadline)
    except Exception:
        return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    Xbest = best.copy()
    # cache iu
    trial = 0
    # precompute nothing else
    while time.time() < HOP_END and time.time() < DEAD - 25:
        trial += 1
        # ---- build rigidity soft modes from incumbent ----
        try:
            Gc = Xbest @ Xbest.T
            np.fill_diagonal(Gc, -2.0)
            mcur = float(np.max(Gc))
            # realistic threshold
            for delta in (1e-5, 5e-5, 1e-4, 5e-4):
                iu_t = np.triu_indices(n, 1)
                c_all = Gc[iu_t]
                sel = c_all > mcur - delta
                p0 = int(np.sum(sel))
                if p0 >= 150:
                    break
            I_all = iu_t[0][sel]
            J_all = iu_t[1][sel]
            c_sel = c_all[sel]
            # cap to tightest 2500
            if len(I_all) > 2500:
                kk = np.argpartition(-c_sel, 2500)[:2500]
                I_all = I_all[kk]; J_all = J_all[kk]
            p = len(I_all)
            use_soft = p >= 50
            T = None
            Vt = None
            Svals = None
            if use_soft:
                T = _tangent_bases(Xbest)
                dd = d - 1
                Nv = n * dd
                R = np.zeros((p, Nv), dtype=float)
                # fill rows
                for k in range(p):
                    ii = int(I_all[k]); jj = int(J_all[k])
                    # T[ii].T @ Xbest[jj]
                    R[k, ii*dd:(ii+1)*dd] = T[ii].T @ Xbest[jj]
                    R[k, jj*dd:(jj+1)*dd] = T[jj].T @ Xbest[ii]
                # economy SVD
                try:
                    _, Svals, Vtt = np.linalg.svd(R, full_matrices=False)
                    Vt = Vtt  # p x Nv
                    del R
                except Exception:
                    use_soft = False
                    Vt = None
            if use_soft and Vt is not None:
                dd = d - 1
                Nv = n * dd
                # softest rows
                order = np.argsort(Svals)  # ascending
                Ksoft = min(12, p)
                soft_rows = Vt[order[:Ksoft]]  # Ksoft x Nv
                # null-projected random modes
                Knull = 14
                Z = rng.standard_normal((Knull, Nv))
                V = Vt.T  # Nv x p
                # project out row space: Z - (Z V) Vt
                Coef = Z @ V  # Knull x p
                Znull = Z - Coef @ Vt
                # normalize
                zn = np.linalg.norm(Znull, axis=1, keepdims=True) + 1e-12
                Znull = Znull / zn
                # also normalize soft rows (already unit)
                Modes = np.concatenate([soft_rows, Znull], axis=0)  # K x Nv
                K = Modes.shape[0]
                # random combination
                coeffs = rng.standard_normal(K)
                # occasionally emphasize null part
                avec = coeffs @ Modes  # Nv
                Amat = avec.reshape(n, dd)
                # map to ambient
                Dfield = np.einsum('idk,ik->id', T, Amat)
                # scale to rho radians
                r norms
                norms = np.linalg.norm(Dfield, axis=1)
                rms = float(np.sqrt(np.mean(norms**2))) + 1e-12
                rho = float(rng.uniform(0.03, 0.25))
                # vary rho occasionally larger for diversity
                if trial % 11 == 0:
                    rho = float(rng.uniform(0.15, 0.30))
                Dfield = Dfield * (rho / rms)
                # sign flip to reduce current highest pairs
                try:
                    # top 30 pairs
                    iu2 = np.triu_indices(n, 1)
                    Gall = Gc[iu2]
                    topk = min(40, len(Gall))
                    tidx = np.argpartition(-Gall, topk)[:topk]
                    It = iu2[0][tidx]; Jt = iu2[1][tidx]
                    dvals = np.sum(Dfield[It] * Xbest[Jt], axis=1) + np.sum(Xbest[It] * Dfield[Jt], axis=1)
                    if float(np.mean(dvals)) > 0:
                        Dfield = -Dfield
                    # 50% random sign flip for diversity on some trials
                    if trial % 4 == 0 and rng.random() < 0.5:
                        Dfield = -Dfield
                except Exception:
                    pass
                # geodesic step
                dn = np.linalg.norm(Dfield, axis=1, keepdims=True) + 1e-16
                Dir = Dfield / dn
                ang = np.linalg.norm(Dfield, axis=1, keepdims=True)  # rho-scaled
                Xtry = np.cos(ang) * Xbest + np.sin(ang) * Dir
                Xtry = _norm(Xtry)
            else:
                # fallback random shake
                sigma = 0.08
                Xtry = Xbest + sigma * rng.standard_normal((n, d))
                Xtry = _norm(Xtry)
        except Exception:
            Xtry = Xbest + 0.08 * rng.standard_normal((n, d))
            Xtry = _norm(Xtry)

        steps = int(rng.integers(300, 601))
        a0, a1, lr0 = 25.0, 550.0, 0.015
        # occasional hotter schedule
        if trial % 7 == 0:
            a0, a1 = 18.0, 450.0
        Xref = _smooth_refine(Xtry, steps, a0, a1, lr0, deadline=min(HOP_END, DEAD-22))
        val = _max_cos(Xref)
        # filter: only LP if promising
        promising = (val < best_val + 0.006)
        if promising:
            sec = 1.2
            if val < best_val - 1e-4:
                sec = 1.6
            # ensure time
            if time.time() + sec + 2 > DEAD:
                sec = max(0.4, DEAD - time.time() - 3)
            Xpol = _lp_short(Xref, sec, deadline=DEAD-3)
            vpol = _max_cos(Xpol)
            if vpol < best_val - 1e-12:
                Xbest, best_val = Xpol, vpol
            elif val < best_val - 1e-12:
                Xbest, best_val = Xref, val
        else:
            # if big improvement even without LP, accept (should be rare)
            if val < best_val - 5e-4:
                # quick LP verify with tiny budget
                if time.time() + 1.0 < DEAD - 5:
                    Xpol = _lp_short(Xref, 0.8, deadline=DEAD-3)
                    vpol = _max_cos(Xpol)
                    if vpol < best_val:
                        Xbest, best_val = Xpol, vpol
                    elif val < best_val:
                        Xbest, best_val = Xref, val
        # occasional fallback: if stuck many trials, do a plain small shake + refine without SVD (diversity)
        if trial % 13 == 0 and time.time() < HOP_END - 5:
            try:
                Xs = Xbest + 0.05 * rng.standard_normal((n, d))
                Xs = _norm(Xs)
                Xs = _smooth_refine(Xs, 300, 25.0, 500.0, 0.015, deadline=min(HOP_END, DEAD-22))
                vs = _max_cos(Xs)
                if vs < best_val + 0.004:
                    Xp2 = _lp_short(Xs, 0.8, deadline=DEAD-3)
                    vp2 = _max_cos(Xp2)
                    if vp2 < best_val:
                        Xbest, best_val = Xp2, vp2
            except Exception:
                pass
        if time.time() > HOP_END:
            break

    # final polish with remaining budget
    remaining = (t0 + 172.0) - time.time() - 1.0
    if remaining > 2.0:
        try:
            Xbest = _lp_short(Xbest, min(remaining, 8.0), deadline=t0+173.0)
        except Exception:
            pass
    Xbest = _norm(Xbest)
    return Xbest
