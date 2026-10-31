import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline, max_iter=60):
    X = _norm(X.copy())
    n, d = X.shape
    try:
        best = _maxcos(X)
    except Exception:
        return X, 9.0
    bx = X.copy()
    radius = 2.0e-3
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_iter:
        it += 1
        G = X @ X.T
        c = G[iu0]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu0[0][sel]; J = iu0[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = X + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d - 1))
    I6 = np.eye(d)
    for i in range(n):
        A = I6.copy()
        A[:, 0] = X[i]
        Q, R = np.linalg.qr(A)
        # ensure first column aligns with X[i]
        if float(np.dot(Q[:, 0], X[i])) < 0:
            Q[:, 0] = -Q[:, 0]
            # flip another column to keep det positive (irrelevant); keep orthonormal
        T[i] = Q[:, 1:]
    return T

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(0)
    n, d = 116, 6
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nn = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nn < 1e-12):
            return None
        return a / nn
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    cur_x = best_x.copy()
    cur_v = best_v
    # quick initial LP squeeze
    if HARD_END - time.time() > 15:
        try:
            bx, bv = _lp_polish(best_x, min(time.time() + 2.5, HARD_END - 8.0))
            if bv < best_v and np.isfinite(bv):
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
        except Exception:
            pass
    iu0 = np.triu_indices(n, 1)
    Nvar = n * (d - 1)  # 580
    alphas = [1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4]
    outer = 0
    # active-set nullspace exploitation
    while time.time() < HARD_END - 12.0:
        outer += 1
        if outer > 60:
            break
        X = cur_x.copy()
        Mnow = _maxcos(X)
        Gfull = X @ X.T
        cfull = Gfull[iu0]
        slack = Mnow - cfull
        # collect pairs with slack < 3e-7
        order = np.argsort(slack)
        tight_thresh = 3.0e-7
        mask = slack < tight_thresh
        cnt = int(np.sum(mask))
        if cnt < 300:
            # expand to include plateau
            tight_thresh = 1.0e-6
            mask = slack < tight_thresh
            cnt = int(np.sum(mask))
        if cnt < 250:
            # take 400 tightest
            take = min(400, len(order))
            sel_idx = order[:take]
        elif cnt > 1100:
            sel_idx = order[:1100]
        else:
            sel_idx = np.where(mask)[0]
        # if too few, widen further
        if sel_idx.size < 100:
            sel_idx = order[:min(600, len(order))]
        I = iu0[0][sel_idx]; J = iu0[1][sel_idx]
        p = len(I)
        # tangent bases
        try:
            T = _tangent_bases(X)  # n,6,5
        except Exception:
            break
        # rigidity matrix p x 580
        try:
            Rmat = np.empty((p, Nvar), dtype=float)
            # precompute projections: T[i].T @ X[j]
            # loop
            for k in range(p):
                i = int(I[k]); j = int(J[k])
                # 5-vectors
                vi = T[i].T @ X[j]  # (5,)
                vj = T[j].T @ X[i]
                Rmat[k, i * 5:(i + 1) * 5] = vi
                Rmat[k, j * 5:(j + 1) * 5] = vj
        except Exception:
            break
        # nullspace via Gram eig
        try:
            C = Rmat.T @ Rmat  # 580x580
            # symmetrize
            C = 0.5 * (C + C.T)
            w, V = np.linalg.eigh(C)  # ascending, V columns
        except Exception:
            break
        # determine K
        # w are eigenvalues = s^2
        try:
            nsmall = int(np.sum(w < 1e-10))
        except Exception:
            nsmall = 0
        K = max(nsmall, 40)
        K = min(K, 80)
        # also include a few low-but-nonzero modes: already in smallest K
        Vnull = V[:, :K]  # 580 x K
        # random search in nullspace
        ndir = 260
        if time.time() > HARD_END - 40:
            ndir = 140
        best_trial_v = Mnow
        best_trial_X = None
        # generate coeffs
        try:
            Cmat = rng.standard_normal((ndir, K))
        except Exception:
            break
        # normalize each direction in 580-space to unit norm
        # Vnull is orthonormal, so norm(u)=norm(c); normalize c
        cn = np.linalg.norm(Cmat, axis=1, keepdims=True)
        cn[cn == 0] = 1.0
        Cmat = Cmat / cn
        # loop dirs
        for di in range(ndir):
            if time.time() > HARD_END - 12.0:
                break
            cvec = Cmat[di]
            u = Vnull @ cvec  # 580
            Umat = u.reshape(n, d - 1)
            # DX = T @ Umat per point
            # einsum: (n,6,5),(n,5)->(n,6)
            try:
                DX = np.einsum('nkl,nl->nk', T, Umat)
            except Exception:
                continue
            # quick first-order check: Ru = Rmat @ u
            # if huge first-order increase, still need exact? skip only if enormous
            # compute Ru max
            try:
                Ru = Rmat @ u
                rumax = float(np.max(Ru))
            except Exception:
                rumax = 0.0
            # if rumax * 1e-2 > 5e-4, this dir is stiff, skip largest alpha
            # decide alphas to try
            if rumax > 0.05:
                alist = alphas[2:]
            elif rumax > 0.01:
                alist = alphas[1:]
            else:
                alist = alphas
            for alpha in alist:
                Y = X + alpha * DX
                Y = _norm(Y)
                v = _maxcos(Y)
                if v < best_trial_v:
                    best_trial_v = v
                    best_trial_X = Y.copy()
                    if v < Mnow - 2e-7:
                        break
            if best_trial_v < Mnow - 2e-7:
                break
            # also try opposite sign for a few dirs (random already covers, but explicit)
            if di % 4 == 0:
                for alpha in alist[:2]:
                    Y = X - alpha * DX
                    Y = _norm(Y)
                    v = _maxcos(Y)
                    if v < best_trial_v:
                        best_trial_v = v
                        best_trial_X = Y.copy()
        # accept?
        improved = False
        if best_trial_X is not None and best_trial_v < Mnow - 1e-12:
            # short LP polish on trial
            dl = min(time.time() + 1.2, HARD_END - 10.0)
            try:
                if dl > time.time() + 0.15:
                    Xp, vp = _lp_polish(best_trial_X, dl, max_iter=50)
                else:
                    Xp, vp = best_trial_X, best_trial_v
            except Exception:
                Xp, vp = best_trial_X, best_trial_v
            if vp <= best_trial_v + 1e-12:
                # accept as new cur even if not better than global best (basin move)
                cur_x, cur_v = Xp.copy(), float(vp)
                if vp < best_v - 5e-13:
                    best_x, best_v = Xp.copy(), float(vp)
                    cur_x, cur_v = Xp.copy(), float(vp)
                improved = True
            else:
                # still accept raw trial if it improves cur
                if best_trial_v < cur_v:
                    cur_x, cur_v = best_trial_X.copy(), float(best_trial_v)
                    if best_trial_v < best_v:
                        best_x, best_v = best_trial_X.copy(), float(best_trial_v)
                    improved = True
        if not improved:
            # no nullspace gain: small random walk inside nullspace + LP to change basin
            # take a random null step of size 3e-3 and polish, keep wandering via Metropolis
            try:
                cvec = rng.standard_normal(K)
                cvec /= (np.linalg.norm(cvec) + 1e-15)
                u = Vnull @ cvec
                Umat = u.reshape(n, d - 1)
                DX = np.einsum('nkl,nl->nk', T, Umat)
                alpha = 3.0e-3 if (outer % 2 == 0) else 1.0e-3
                Y = _norm(X + alpha * DX)
                dl = min(time.time() + 0.9, HARD_END - 10.0)
                if dl > time.time() + 0.15:
                    Yp, vp = _lp_polish(Y, dl, max_iter=40)
                else:
                    Yp, vp = Y, _maxcos(Y)
                # Metropolis on cur with slowly shrinking temp
                frac = (time.time() - t0) / max(HARD_END - 12.0 - t0, 1.0)
                frac = min(max(frac, 0.0), 1.0)
                Temp = 2e-6 * (1.0 - frac) + 2e-7
                if vp < cur_v or rng.random() < np.exp(-(vp - cur_v) / max(Temp, 1e-15)):
                    if vp < cur_v + 2e-4:
                        cur_x, cur_v = Yp.copy(), float(vp)
                        if vp < best_v - 5e-13:
                            best_x, best_v = Yp.copy(), float(vp)
                else:
                    # occasionally reset to best
                    if outer % 8 == 0 and cur_v > best_v + 1e-5:
                        cur_x, cur_v = best_x.copy(), best_v
                # also try opposite sign next time via cur update
            except Exception:
                pass
            # if many failures, perturb cur slightly with isotropic noise to refresh active set
            if outer % 6 == 0:
                try:
                    Yn = _norm(cur_x + rng.normal(scale=2e-4, size=(n, d)))
                    dl2 = min(time.time() + 0.7, HARD_END - 10.0)
                    if dl2 > time.time() + 0.15:
                        Yp2, vp2 = _lp_polish(Yn, dl2, max_iter=30)
                        if vp2 < best_v - 5e-13:
                            best_x, best_v = Yp2.copy(), float(vp2)
                            cur_x, cur_v = Yp2.copy(), float(vp2)
                except Exception:
                    pass
        # loop continues with updated active set
    # final squeeze
    rem = HARD_END - time.time()
    if rem > 4.0:
        try:
            fb = min(rem - 1.0, 22.0)
            Xf, vf = _lp_polish(best_x, time.time() + fb, max_iter=200)
            if vf <= best_v and np.isfinite(vf):
                best_x, best_v = Xf, vf
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
