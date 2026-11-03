import numpy as np
import time as _time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X, R0):
    n = X.shape[0]
    B = np.empty((n, 6, 5), dtype=float)
    for i in range(n):
        xi = X[i]
        Rp = R0 - np.outer(xi, (xi @ R0))
        Q, _ = np.linalg.qr(Rp)
        # Q is 6x5 for 6x5 input in reduced mode; guard fallback
        if Q.shape != (6, 5):
            # complete mode gave 6x6, take 5 cols orthogonal to xi
            # project again
            Q = Q[:, :5]
            # ensure orthogonality to xi
            Q = Q - np.outer(xi, (xi @ Q))
            Q, _ = np.linalg.qr(Q)
        B[i] = Q
    return B

def solve():
    t0 = _time.time()
    BUDGET = 172.0
    n, d, k = 249, 6, 5
    Nvar = n * k + 1
    rng = np.random.default_rng(12345)
    R0 = rng.standard_normal(size=(6, 5))
    from scipy.optimize import linprog as _lp
    from scipy.sparse import coo_matrix as _coo

    Iall, Jall = np.triu_indices(n, 1)

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
    bi = int(np.argmin(vals))
    globX = cands[bi].copy()
    globVal = float(vals[bi])

    obj = np.zeros(Nvar, dtype=float)
    obj[-1] = 1.0

    def lp_jam(Xinit, radius_init, time_limit, slack=2e-3, max_iters=100, pmax=3500, rmax=0.015):
        nonlocal globX, globVal
        X = Xinit.copy()
        Xn = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(Xn, 1e-12)
        cur = _max_cos(X)
        if cur < globVal:
            globVal = cur
            globX = X.copy()
        radius = float(radius_init)
        deadline = _time.time() + float(time_limit)
        it = 0
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
                # take tightest 400
                idx = np.argpartition(cfull, -400)[-400:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p = 400
            if p > pmax:
                idx = np.argpartition(cfull, -pmax)[-pmax:]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
                p = pmax
            I = Iall[sel]; J = Jall[sel]; csel = cfull[sel]
            B = _tangent_bases(X, R0)
            BI = B[I]
            BJ = B[J]
            XJ = X[J]
            XI = X[I]
            coeffI = np.einsum('pkm,pk->pm', BI, XJ)
            coeffJ = np.einsum('pkm,pk->pm', BJ, XI)
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
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            Avec = res.x[:-1]
            # check finite
            if not np.all(np.isfinite(Avec)):
                radius *= 0.5
                continue
            Amat = Avec.reshape(n, k)
            DX = np.einsum('nkm,nm->nk', B, Amat)
            Y = X + DX
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(Y)
            if v < cur - 1e-15:
                X = Y
                cur = v
                if v < globVal:
                    globVal = v
                    globX = Y.copy()
                radius = min(radius * 1.5, rmax)
            else:
                radius *= 0.5
        return X, cur

    def holes_sample(X, need, exclude=None, nsample=30000):
        R = rng.standard_normal(size=(nsample, d))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
        S = R @ X.T
        if exclude is not None and len(exclude) > 0:
            keep = np.ones(n, dtype=bool)
            keep[np.asarray(exclude, dtype=int)] = False
            if np.any(keep):
                m = np.max(S[:, keep], axis=1)
            else:
                m = np.max(S, axis=1)
        else:
            m = np.max(S, axis=1)
        idx = np.argpartition(m, need)[:need]
        idx = idx[np.argsort(m[idx])]
        return R[idx]

    def greedy_fill(Xbase, del_idx, pool_size=40000):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[np.asarray(del_idx, dtype=int)] = False
        Xkeep = Xbase[keep_mask]
        Pool = rng.standard_normal(size=(pool_size, d))
        Pool /= np.maximum(np.linalg.norm(Pool, axis=1, keepdims=True), 1e-12)
        D = Pool @ Xkeep.T
        curmax = D.max(axis=1)
        C = Pool.shape[0]
        used = np.zeros(C, dtype=bool)
        new_pts = []
        kk = len(del_idx)
        for _ in range(kk):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = Pool[b].copy()
            new_pts.append(pt)
            dd = Pool @ pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
        new_pts = np.stack(new_pts, axis=0)
        return new_pts

    # Phase 1: jam incumbent
    try:
        Xj, vj = lp_jam(globX.copy(), radius_init=2e-3, time_limit=22.0, slack=2e-3, max_iters=110)
    except Exception:
        pass

    # outer basin moves
    trial = 0
    while _time.time() - t0 < BUDGET - 24 and trial < 80:
        trial += 1
        if _time.time() - t0 > BUDGET - 25:
            break
        base = globX.copy()
        # contact degrees at looser tol to find structure
        try:
            G0 = base @ base.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(np.max(G0))
        except Exception:
            continue
        # adaptive tol to get ~1200 contacts
        try:
            gflat = G0[Iall, Jall]
            thr = np.sort(gflat)[-1300] if len(gflat) >= 1300 else m0 - 1e-6
            tol_loose = max(m0 - thr, 1e-7)
            tol_loose = min(max(tol_loose, 1e-7), 5e-4)
        except Exception:
            tol_loose = 2e-4
        deg = np.sum(G0 > (m0 - tol_loose), axis=1)
        ppm = G0.max(axis=1)
        mode = trial % 5
        Xt = base.copy()
        try:
            if mode == 0:
                # rattler / weakest to holes
                order = np.argsort(deg + 2000.0 * np.maximum(ppm - (m0 - tol_loose), 0))
                # number to move 1-4
                nm = 1 + (trial % 4 == 0) + (trial % 7 == 0) + (trial % 11 == 0)
                nm = min(nm, 4)
                ratt = order[:nm]
                hs = holes_sample(base, nm, exclude=ratt, nsample=30000)
                for s, wi in enumerate(ratt):
                    h = hs[s % len(hs)] + 0.0015 * rng.standard_normal(size=(d,))
                    h /= (np.linalg.norm(h) + 1e-12)
                    Xt[int(wi)] = h
                Xt += 1.0e-4 * rng.standard_normal(size=(n, d))
            elif mode == 1:
                # spherical cap deletion
                u = rng.standard_normal(size=(d,))
                u /= (np.linalg.norm(u) + 1e-12)
                proj = base @ u
                kdel = 5 + (trial % 6)  # 5-10
                del_idx = np.argsort(proj)[-kdel:]
                newp = greedy_fill(base, del_idx, pool_size=35000)
                for s, wi in enumerate(del_idx):
                    Xt[int(wi)] = newp[s]
                Xt += 8e-5 * rng.standard_normal(size=(n, d))
            elif mode == 2:
                # hub BFS cluster
                cnt = np.sum(G0 > (m0 - 2e-4), axis=1)
                hub = int(np.argmax(cnt * 1000.0 + ppm))
                adj_mask = G0 > (m0 - 5e-4)
                # BFS
                visited = np.zeros(n, dtype=bool)
                visited[hub] = True
                frontier = [hub]
                while len(frontier) > 0 and int(np.sum(visited)) < 12:
                    cur2 = frontier.pop(0)
                    nbrs = np.where(adj_mask[cur2])[0]
                    for nb in nbrs:
                        if not visited[int(nb)]:
                            visited[int(nb)] = True
                            frontier.append(int(nb))
                            if int(np.sum(visited)) >= 10:
                                break
                del_idx = np.where(visited)[0]
                if len(del_idx) < 3:
                    del_idx = np.argsort(ppm)[-6:]
                if len(del_idx) > 10:
                    # keep worst 8 by ppm
                    sc = ppm[del_idx]
                    del_idx = del_idx[np.argsort(-sc)[:8]]
                newp = greedy_fill(base, del_idx, pool_size=35000)
                for s, wi in enumerate(del_idx):
                    Xt[int(wi)] = newp[s]
                Xt += 8e-5 * rng.standard_normal(size=(n, d))
            elif mode == 3:
                # worst-pair split
                flat = int(np.argmax(G0))
                ii, jj = flat // n, flat % n
                dvec = Xt[ii] - Xt[jj]
                dvec /= max(np.linalg.norm(dvec), 1e-12)
                amp = 0.003 + 0.003 * rng.random()
                Xt[ii] = Xt[ii] + amp * dvec + 0.0008 * rng.standard_normal(size=(d,))
                Xt[jj] = Xt[jj] - amp * dvec + 0.0008 * rng.standard_normal(size=(d,))
                # also move one weakest point to hole
                w = int(np.argmin(ppm - deg * 1e-6))
                hs = holes_sample(base, 1, exclude=[w], nsample=20000)
                if rng.random() < 0.5:
                    Xt[w] = hs[0] + 0.001 * rng.standard_normal(size=(d,))
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
            else:
                # global shake
                amp = 3e-4 if (trial % 2 == 0) else 8e-4
                Xt = base + amp * rng.standard_normal(size=(n, d))
            Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
        except Exception:
            continue
        # short re-jam
        try:
            sl = 1.5e-3 if (trial % 2 == 0) else 2e-3
            Xq, vq = lp_jam(Xt, radius_init=1.2e-3, time_limit=2.6, slack=sl, max_iters=13, pmax=3000)
        except Exception:
            continue

    # final intensive jam
    remain = BUDGET - (_time.time() - t0) - 3.0
    if remain > 4:
        try:
            Xf, vf = lp_jam(globX.copy(), radius_init=8e-4, time_limit=min(remain * 0.6, 14.0), slack=1.5e-3, max_iters=120, pmax=3500, rmax=0.008)
        except Exception:
            pass
        remain2 = BUDGET - (_time.time() - t0) - 2.0
        if remain2 > 3:
            try:
                Xf2, vf2 = lp_jam(globX.copy(), radius_init=3e-4, time_limit=min(remain2, 10.0), slack=8e-4, max_iters=120, pmax=3000, rmax=0.004)
            except Exception:
                pass

    out = globX.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    out[~np.isfinite(out)] = 1.0 / np.sqrt(6.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
