import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X, R0):
    # X (n,6) unit, R0 (6,5) fixed random; return B (n,6,5) orthonormal, B_i^T x_i=0
    n = X.shape[0]
    B = np.empty((n, 6, 5), dtype=float)
    for i in range(n):
        xi = X[i]
        Rp = R0 - np.outer(xi, (xi @ R0))
        Q, _ = np.linalg.qr(Rp)
        B[i] = Q
    return B

def solve():
    t0 = _time.time()
    BUDGET = 172.0
    n, d, k = 249, 6, 5
    Nvar = n * k + 1
    rng = np.random.default_rng(12345)
    R0 = rng.standard_normal(size=(6, 5))

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [_max_cos(a) for a in cands]
    bestX = cands[int(np.argmin(vals))].copy()
    bestVal = float(np.min(vals))
    globX = bestX.copy()
    globVal = bestVal

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    iu0 = np.triu_indices(n, 1)
    Iall, Jall = iu0

    def lp_jam(Xinit, radius_init, time_limit, slack=2e-3, max_iters=200):
        nonlocal globX, globVal
        from scipy.optimize import linprog as _lp
        from scipy.sparse import coo_matrix as _coo
        X = Xinit.copy()
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        cur = _max_cos(X)
        # update global if better
        if cur < globVal:
            globVal = cur
            globX = X.copy()
        radius = float(radius_init)
        it = 0
        obj = np.zeros(Nvar, dtype=float)
        obj[-1] = 1.0
        deadline = _time.time() + float(time_limit)
        while it < max_iters and _time.time() < deadline:
            if _time.time() - t0 > BUDGET - 8:
                break
            if radius < 1e-13:
                break
            it += 1
            Gf = X @ X.T
            cfull = Gf[Iall, Jall]
            m = float(np.max(cfull))
            sel = cfull > (m - slack)
            p = int(np.sum(sel))
            if p < 20:
                sel = cfull > (m - 1e-6)
                p = int(np.sum(sel))
            if p > 6000:
                # keep tightest 6000
                idx = np.argpartition(cfull, -6000)[-6000:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p = 6000
            I = Iall[sel]; J = Jall[sel]; csel = cfull[sel]
            # tangent bases at current X
            B = _tangent_bases(X, R0)  # (n,6,5)
            BI = B[I]  # (p,6,5)
            BJ = B[J]
            XJ = X[J]  # (p,6)
            XI = X[I]
            coeffI = np.einsum('pkm,pk->pm', BI, XJ)  # (p,5)
            coeffJ = np.einsum('pkm,pk->pm', BJ, XI)
            # sparse A_ub: p rows, Nvar cols, 11 nnz per row
            rows = np.repeat(np.arange(p), 2 * k + 1)
            colI = (I[:, None] * k + np.arange(k)).ravel()
            colJ = (J[:, None] * k + np.arange(k)).ravel()
            colT = np.full(p, Nvar - 1)
            cols = np.concatenate([colI.reshape(p, k), colJ.reshape(p, k), colT[:, None]], axis=1).ravel()
            data = np.concatenate([coeffI, coeffJ, -np.ones((p, 1))], axis=1).ravel()
            try:
                A_ub = _coo((data, (rows, cols)), shape=(p, Nvar)).tocsr()
                b_ub = -csel
                bounds = [(-radius, radius)] * (n * k) + [(None, None)]
                res = _lp(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            Avec = res.x[:-1]  # (n*5,)
            Amat = Avec.reshape(n, k)
            # displacement
            DX = np.einsum('nkm,nm->nk', B, Amat)  # (n,6)
            Y = X + DX
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Y)
            if v < cur - 1e-15:  # exact decrease
                X = Y
                cur = v
                if v < globVal:
                    globVal = v
                    globX = Y.copy()
                radius = min(radius * 1.5, 0.015)
                # shrink slack as we tighten? keep fixed per spec
            else:
                radius *= 0.5
        return X, cur

    def contacts_info(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        mask = G > (m - tol)
        deg = np.sum(mask, axis=1)
        ppm = G.max(axis=1)
        return deg, ppm, m, G

    def largest_holes(X, need, nsample=22000, exclude=None):
        R = rng.standard_normal(size=(nsample, d))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        S = R @ X.T
        if exclude is not None and len(exclude) > 0:
            # ignore columns of excluded points when scoring holes
            keep = np.ones(n, dtype=bool)
            keep[np.asarray(exclude, dtype=int)] = False
            if np.any(keep):
                S2 = S[:, keep]
                m = S2.max(axis=1)
            else:
                m = S.max(axis=1)
        else:
            m = S.max(axis=1)
        idx = np.argpartition(m, need)[:need]
        idx = idx[np.argsort(m[idx])]
        return R[idx]

    # Phase 1: exact jam of incumbent
    Xj, vj = lp_jam(bestX, radius_init=2e-3, time_limit=28.0, slack=2e-3, max_iters=120)
    if vj < bestVal:
        bestX, bestVal = Xj.copy(), vj

    # outer rattler / multiplier-free moves
    trial = 0
    while _time.time() - t0 < BUDGET - 22 and trial < 60:
        trial += 1
        if _time.time() - t0 > BUDGET - 23:
            break
        # use globally best jammed config as base
        base = globX.copy()
        deg, ppm, m0, G0 = contacts_info(base, tol=1e-9)
        # if all jammed (deg>=5 everywhere) relax tol to find weakest
        ratt = np.where(deg == 0)[0]
        if len(ratt) == 0:
            # multiplier-free proxy: lowest degree points
            order = np.argsort(deg + 0.5 * (ppm - ppm.min()) * 1000.0)
            # move 1-3 weakest
            nr = 1 if (trial % 3 != 0) else (2 if (trial % 3 == 1) else 3)
            # occasionally move hub neighbour cluster
            if trial % 7 == 0:
                hub = int(np.argmax(deg))
                # neighbours of hub at 5e-4
                adj = G0[hub] > (m0 - 5e-4)
                cand = np.where(adj)[0]
                if len(cand) > 0:
                    ratt = np.array([hub] + cand[:nr].tolist())[: nr + 1]
                else:
                    ratt = order[:nr]
            else:
                ratt = order[: max(1, nr)]
        # number to move: all rattlers if few, else subset
        if len(ratt) > 4:
            # pick worst 4 by ppm
            sc = ppm[ratt]
            ratt = ratt[np.argsort(-sc)[:4]]
        nmove = len(ratt)
        holes = largest_holes(base, nmove + 2, nsample=22000, exclude=ratt)
        Xt = base.copy()
        for s, wi in enumerate(ratt):
            h = holes[s % len(holes)] + 0.002 * rng.standard_normal(size=(d,))
            h /= (np.linalg.norm(h) + 1e-12)
            Xt[int(wi)] = h
        # tiny global tangent jitter to break symmetry
        Xt += 1.5e-4 * rng.standard_normal(size=(n, d))
        Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        # re-jam briefly
        Xq, vq = lp_jam(Xt, radius_init=1.2e-3, time_limit=3.2, slack=2e-3, max_iters=14)
        # lp_jam already updates glob on improvement; continue
        # also try second hole set if no improvement and time allows
        if vq > globVal and (trial % 4 == 0) and (_time.time() - t0 < BUDGET - 30):
            # alternative: pair-split of worst pair
            Gq = base @ base.T
            np.fill_diagonal(Gq, -2.0)
            flat = int(np.argmax(Gq))
            ii, jj = flat // n, flat % n
            Xt2 = base.copy()
            dvec = Xt2[ii] - Xt2[jj]
            dvec /= max(np.linalg.norm(dvec), 1e-12)
            amp = 0.004 + 0.002 * rng.random()
            Xt2[ii] += amp * dvec + 0.001 * rng.standard_normal(size=(d,))
            Xt2[jj] -= amp * dvec + 0.001 * rng.standard_normal(size=(d,))
            Xt2 /= np.maximum(np.linalg.norm(Xt2, axis=1, keepdims=True), 1e-12)
            Xq2, vq2 = lp_jam(Xt2, radius_init=1.0e-3, time_limit=2.5, slack=2e-3, max_iters=10)

    # final intensive jam of global best
    remain = BUDGET - (_time.time() - t0) - 4.0
    if remain > 3:
        Xf, vf = lp_jam(globX, radius_init=8e-4, time_limit=min(remain, 18.0), slack=2e-3, max_iters=120)
    # fallback safety
    out = globX.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    # shrink tiny margin not needed; ensure finite
    out[~np.isfinite(out)] = 1.0 / np.sqrt(6.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
