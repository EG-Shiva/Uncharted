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
    return float(G.max())

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    T = np.empty((n, d, k))
    e0 = np.zeros(d); e0[0] = 1.0
    for i in range(n):
        x = X[i]
        v = e0 - x
        nv = float(np.linalg.norm(v))
        nv2 = float(np.linalg.norm(e0 + x))
        if nv < 1e-12 or nv2 < 1e-12:
            Ti = np.zeros((d, k))
            for j in range(k):
                Ti[j + 1, j] = 1.0
            # if x ~ -e0 same basis is still orthogonal to x
            T[i] = Ti
        else:
            u = v / nv
            H = np.eye(d) - 2.0 * np.outer(u, u)
            T[i] = H[:, 1:]
    return T

def _rigidity(X, T, pairs):
    n, d = X.shape
    k = d - 1
    p = pairs.shape[0]
    nv = n * k
    A = np.empty((p, nv))
    for l in range(p):
        i = int(pairs[l, 0]); j = int(pairs[l, 1])
        A[l, i*k:(i+1)*k] = T[i].T @ X[j]
        A[l, j*k:(j+1)*k] = T[j].T @ X[i]
    return A

def _lp_polish(X, deadline, radius0=2.0e-3, max_pairs=1300):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > max_pairs:
            o = np.argsort(c[idx])[-max_pairs:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); X = y
            radius = min(radius*1.5, 0.05)
        else:
            if v <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    rng = np.random.default_rng(11606)
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nn = np.linalg.norm(a, axis=1)
        if np.any(nn < 1e-12):
            return None
        return a / nn[:, None]
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
    # initial tiny LP squeeze to establish baseline
    try:
        if HARD - time.time() > 25:
            bx, bv = _lp_polish(best_x, min(time.time()+2.0, HARD-15))
            if bv < best_v:
                best_x, best_v = bx, bv
                cur_x, cur_v = bx.copy(), bv
    except Exception:
        pass
    n, d = 116, 6
    k = d - 1
    steps = [1e-2, 3e-3, 1e-3, 3e-4]
    iu0 = np.triu_indices(n, 1)
    outer = 0
    while time.time() < HARD - 12:
        outer += 1
        X = cur_x.copy()
        m = _maxcos(X)
        # active set: slack < 3e-7
        try:
            G = X @ X.T
            cfull = G[iu0]
            slack = m - cfull
            thrA = 3e-7
            aidx = np.where(slack < thrA)[0]
            if aidx.size < 50:
                # ensure enough constraints: widen
                o = np.argsort(slack)[:400]
                aidx = o
            if aidx.size > 1600:
                o = np.argsort(slack[aidx])[:1600]
                aidx = aidx[o]
            I = iu0[0][aidx]; J = iu0[1][aidx]
            pairs = np.stack([I, J], axis=1)
            T = _tangent_bases(X)
            A = _rigidity(X, T, pairs)
            # SVD
            try:
                _, S, Vt = np.linalg.svd(A, full_matrices=True)
            except Exception:
                # fallback: jitter
                cur_x = _norm(X + 1e-4*rng.standard_normal(X.shape))
                continue
            smax = float(S.max()) if S.size else 0.0
            tol = max(1e-10, smax*1e-11)
            r = int(np.sum(S > tol))
            nv = n*k
            null_dim = nv - r
            if null_dim <= 0:
                # rigid: take small random tangent step to change topology
                Ndir = 200
                null_dim = 0
                use_null = False
            else:
                use_null = True
                Nbasis = Vt[r:, :].T  # nv x null_dim
        except Exception:
            use_null = False
            Ndir = 200
        # search nullspace directions
        best_step_v = m
        best_step_x = None
        if use_null:
            # adapt number of directions to time left
            trem = HARD - time.time()
            if trem < 20:
                Ndir = 250
            elif outer < 3:
                Ndir = 800
            else:
                Ndir = 600
            # pre-generate random coeffs
            try:
                for _ in range(Ndir):
                    if time.time() > HARD - 11:
                        break
                    coeff = rng.standard_normal(null_dim)
                    nn = float(np.linalg.norm(coeff))
                    if nn < 1e-15:
                        continue
                    coeff /= nn
                    u = Nbasis @ coeff  # nv
                    U = u.reshape(n, k)
                    DX = np.einsum('nik,nk->ni', T, U)
                    mxn = float(np.max(np.linalg.norm(DX, axis=1)))
                    if mxn < 1e-15:
                        continue
                    DX = DX / mxn
                    # try steps both signs
                    for t in steps:
                        for sgn in (1.0, -1.0):
                            Y = X + sgn*t*DX
                            Y = _norm(Y)
                            # exact max
                            G2 = Y @ Y.T
                            np.fill_diagonal(G2, -2.0)
                            v = float(G2.max())
                            if v < best_step_v:
                                best_step_v = v
                                best_step_x = Y.copy()
                                if best_step_v < best_v - 1e-12:
                                    pass
                        # early exit if big win
                    # also cheap: if time critical break
                # end dir loop
            except Exception:
                pass
        else:
            # rigid fallback: random tangent drift
            try:
                T = _tangent_bases(X)
                for _ in range(Ndir):
                    if time.time() > HARD - 11:
                        break
                    U = rng.standard_normal((n, k))
                    DX = np.einsum('nik,nk->ni', T, U)
                    mxn = float(np.max(np.linalg.norm(DX, axis=1)))
                    if mxn < 1e-15:
                        continue
                    DX /= mxn
                    for t in steps:
                        Y = _norm(X + t*DX)
                        v = _maxcos(Y)
                        if v < best_step_v:
                            best_step_v = v
                            best_step_x = Y.copy()
            except Exception:
                pass
        if best_step_x is None:
            # no direction: small isotropic jitter along valley
            try:
                T = _tangent_bases(X)
                U = rng.standard_normal((n, k))
                DX = np.einsum('nik,nk->ni', T, U)
                mxn = float(np.max(np.linalg.norm(DX, axis=1)))
                DX /= max(mxn, 1e-15)
                best_step_x = _norm(X + 3e-4*DX)
                best_step_v = _maxcos(best_step_x)
            except Exception:
                break
        # decision: accept improving step + LP polish; else drift along best valley direction
        improved = best_step_v < m - 1e-12
        # if not improved, still take the least-bad step to move basin (valley walk)
        # but only if it does not blow up too much
        take_x = best_step_x
        take_v = best_step_v
        # short LP polish on taken point
        polished_x, polished_v = take_x, take_v
        try:
            if time.time() < HARD - 10:
                polished_x, polished_v = _lp_polish(take_x, min(time.time()+0.9, HARD-9))
        except Exception:
            pass
        if polished_v < best_v - 1e-15:
            best_x, best_v = polished_x.copy(), float(polished_v)
            cur_x, cur_v = polished_x.copy(), float(polished_v)
        elif polished_v < cur_v - 1e-12:
            cur_x, cur_v = polished_x.copy(), float(polished_v)
        elif not improved:
            # valley walk: accept small degradation to explore (bounded)
            # accept if within 2e-4 of current to keep wandering floppy valley
            if take_v < cur_v + 2e-4 and time.time() < HARD - 15:
                # polish may have reduced it; use polished if closer
                if polished_v < take_v + 1e-9:
                    cur_x, cur_v = polished_x.copy(), float(polished_v)
                else:
                    cur_x, cur_v = take_x.copy(), float(take_v)
                # occasionally restart from best to intensify
                if rng.random() < 0.25:
                    cur_x, cur_v = best_x.copy(), best_v
            else:
                # restart from best with tiny null perturbation
                cur_x, cur_v = best_x.copy(), best_v
        else:
            # improved over cur but not best? update cur
            cur_x, cur_v = polished_x.copy(), float(polished_v)
            if polished_v < best_v:
                best_x, best_v = polished_x.copy(), float(polished_v)
        # periodic intensify from best
        if outer % 8 == 0 and time.time() < HARD - 12:
            try:
                bx2, bv2 = _lp_polish(best_x, min(time.time()+1.0, HARD-9))
                if bv2 < best_v:
                    best_x, best_v = bx2, bv2
                    cur_x, cur_v = bx2.copy(), bv2
            except Exception:
                pass
    # final polish
    try:
        remain = HARD - time.time() - 0.5
        if remain > 1.5:
            bx, bv = _lp_polish(best_x, time.time()+min(remain, 10.0))
            if bv < best_v:
                best_x, best_v = bx, bv
    except Exception:
        pass
    return _norm(best_x)
