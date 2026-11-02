import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(20261102)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n0, d):
                cands.append(_norm(a))
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n0, d))))
    vals = [_mc(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    base0 = best.copy()

    # ---------- generic polishers ----------
    def lse_polish(X0, tau, maxiter):
        n = X0.shape[0]
        IU0 = np.triu_indices(n, 1)
        I0 = IU0[0].astype(np.int64); J0 = IU0[1].astype(np.int64)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(n, d)
            r = np.sqrt(np.maximum((Y*Y).sum(axis=1, keepdims=True), 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            dot = (Wx * X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot * X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-7,
                                    "maxfun": int(maxiter*8)})
            Y = res.x.reshape(n, d)
        except Exception:
            return X0
        rn = np.linalg.norm(Y, axis=1, keepdims=True); rn[rn == 0] = 1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0
        return _norm(X)

    def greedy(X, steps, lr0):
        cur = _mc(X)
        lr = float(lr0)
        for _ in range(int(steps)):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0*lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            norms = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = norms[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (norms[act] + 1e-12) * lr
            Xn = _norm(X - step)
            nv = _mc(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-11:
                    break
        return X, cur

    def coarse_polish(X):
        # time-boxed to ~0.209-0.215 range
        X = lse_polish(X, 0.006, 45)
        X, _ = greedy(X, 30, 0.002)
        return X

    def full_polish(X):
        X = lse_polish(X, 0.004, 55)
        X, v = greedy(X, 70, 0.0015)
        X = lse_polish(X, 0.001, 55)
        X, v = greedy(X, 70, 0.0008)
        X = lse_polish(X, 0.0004, 30)
        X, v = greedy(X, 50, 0.0005)
        return X, _mc(X)

    def per_point(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return G, mx

    def degrees(X, tol=1e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        cnt = (G >= gmax - tol).sum(axis=1)
        mx = G.max(axis=1)
        return cnt, mx, gmax

    def fingerprint(X, K=300):
        G = X @ X.T
        iu = np.triu_indices(X.shape[0], 1)
        g = G[iu]
        gs = np.sort(g)[::-1]
        if len(gs) >= K:
            gs = gs[:K]
        cnt, mx, _ = degrees(X, tol=1e-4)
        deg_sorted = np.sort(cnt)
        mx_sorted = np.sort(mx)[::-1][:40]
        return gs, deg_sorted, mx_sorted

    def fp_dist(f1, f2):
        return float(np.max(np.abs(f1[0]-f2[0])) + 0.001*np.sum(np.abs(f1[1].astype(float)-f2[1].astype(float))))

    # ---------- build archive ----------
    parents = []
    parents.append(base0.copy())
    # tangent kicks
    kicks = [0.03, 0.03, 0.08, 0.08, 0.15, 0.15, 0.30, 0.30]
    for ks in kicks:
        if time.time()-t0 > 55:
            break
        Z = rng.standard_normal((n0, d))
        # tangent projection
        proj = (Z*base0).sum(axis=1, keepdims=True)*base0
        Zt = Z - proj
        Xk = _norm(base0 + float(ks)*Zt)
        Xk = coarse_polish(Xk)
        parents.append(Xk.copy())
        v = _mc(Xk)
        if v < best_val:
            best, best_val = Xk.copy(), float(v)
    # 70-homotopy (undershoot): drop 4 worst, coarse 70-polish, regrow holes
    try:
        if time.time()-t0 < 70:
            G0, mx0 = per_point(base0)
            drop = np.argsort(mx0)[-4:]
            mask = np.ones(n0, bool); mask[drop] = False
            X70 = base0[mask].copy()
            X70 = coarse_polish(X70)
            # regrow 4 largest holes
            for _ in range(4):
                M = 8000
                C = _norm(rng.standard_normal((M, d)))
                S = C @ X70.T
                sc = S.max(axis=1)
                j = int(np.argmin(sc))
                X70 = np.vstack([X70, C[j:j+1]])
            X70 = coarse_polish(X70)
            if X70.shape[0] == n0:
                parents.append(X70.copy())
                v = _mc(X70)
                if v < best_val:
                    best, best_val = X70.copy(), float(v)
    except Exception:
        pass
    # 78-homotopy (overshoot): add 4, polish 78, prune 4 worst
    try:
        if time.time()-t0 < 85:
            X78 = np.vstack([base0, _norm(rng.standard_normal((4, d)))])
            X78 = coarse_polish(X78)
            G78 = X78 @ X78.T
            np.fill_diagonal(G78, -2.0)
            mx78 = G78.max(axis=1)
            drop = np.argsort(mx78)[-4:]
            mask = np.ones(78, bool); mask[drop] = False
            Xh = _norm(X78[mask].copy())
            Xh = coarse_polish(Xh)
            parents.append(Xh.copy())
            v = _mc(Xh)
            if v < best_val:
                best, best_val = Xh.copy(), float(v)
    except Exception:
        pass
    # one random start, coarsely polished
    try:
        if time.time()-t0 < 95:
            Xr = _norm(rng.standard_normal((n0, d)))
            Xr = lse_polish(Xr, 0.01, 50)
            Xr = lse_polish(Xr, 0.005, 50)
            Xr, _ = greedy(Xr, 30, 0.002)
            parents.append(Xr.copy())
    except Exception:
        pass

    # fingerprint dedup (rotation/permutation invariant)
    fps = [fingerprint(p) for p in parents]
    uniq = []
    uniq_fps = []
    for p, f in zip(parents, fps):
        dup = False
        for g in uniq_fps:
            try:
                if fp_dist(f, g) < 5e-5:
                    dup = True
                    break
            except Exception:
                pass
        if not dup:
            uniq.append(p); uniq_fps.append(f)
    parents = uniq

    def align_parents(Xa, Xb, iters=2):
        # Hungarian on Gram + SVD rotation, iterated
        Xbp = Xb.copy()
        R = np.eye(d)
        perm = np.arange(n0)
        try:
            for _ in range(iters+1):
                Gc = Xa @ Xbp.T
                _, col = linear_sum_assignment(-Gc)
                Xbp = Xbp[col]
                perm = perm[col] if 'perm' in locals() else col
                M = Xbp.T @ Xa
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                R2 = U @ Vt
                Xbp = Xbp @ R2
                R = R @ R2
            return Xbp, perm, R
        except Exception:
            return Xb, np.arange(n0), np.eye(d)

    def repair_dup(X):
        # replace near-duplicate rows by largest-hole resampling
        for _ in range(6):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(G.max())
            if m < 0.90:
                break
            k = int(np.argmax(G))
            i = k // n0; j = k % n0
            # replace j (higher per-point max)
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            mi = float(G2[i].max()); mj = float(G2[j].max())
            rpl = j if mj >= mi else i
            M = 6000
            C = _norm(rng.standard_normal((M, d)))
            keep = np.ones(n0, bool); keep[rpl] = False
            S = C @ X[keep].T
            sc = S.max(axis=1)
            b = int(np.argmin(sc))
            X[rpl] = C[b]
        return _norm(X)

    def halfmix_child(Xa, Xb_al):
        _, mxa = per_point(Xa)
        _, mxb = per_point(Xb_al)
        ia = np.argsort(mxa)[:37]
        ib = np.argsort(mxb)[:37]
        # child1: best37 of A + complement of B
        c1 = Xb_al.copy()
        c1[ia] = Xa[ia]
        # child2: best37 of B + complement of A
        c2 = Xa.copy()
        # need complement mapping: positions not in ia get B, for c2 positions in ib get B
        c2[ib] = Xb_al[ib]
        return (_norm(repair_dup(c1)), _norm(repair_dup(c2)))

    def hubswap_child(Xa, Xb_raw, R):
        # hubs of A replaced by matched low-degree points of B
        cntA, mxA, _ = degrees(Xa, tol=5e-4)
        cntB, mxB, _ = degrees(Xb_raw, tol=5e-4)
        nh = 8
        hubs = np.argsort(-mxA - 0.01*cntA)[:nh]
        lows = np.argsort(cntB + 0.0*mxB)[:16]
        Xb_rot = Xb_raw @ R
        # sub-assignment hubs x lows
        try:
            S = Xa[hubs] @ Xb_rot[lows].T
            _, col = linear_sum_assignment(-S)
            ch = Xa.copy()
            for ii, cc in enumerate(col[:nh]):
                ch[hubs[ii]] = Xb_rot[lows[cc]]
            return _norm(repair_dup(ch))
        except Exception:
            return None

    # order parent pairs by dissimilarity
    P = len(parents)
    pair_list = []
    for a in range(P):
        for b in range(a+1, P):
            try:
                dd = fp_dist(uniq_fps[a], uniq_fps[b])
            except Exception:
                dd = 1.0
            pair_list.append((dd, a, b))
    pair_list.sort(reverse=True)

    seen = list(uniq_fps)
    children_done = 0
    for dd, a, b in pair_list:
        if time.time()-t0 > LIM-22:
            break
        if dd < 1e-6:
            continue
        Xa = parents[a]; Xb = parents[b]
        Xb_al, perm, R = align_parents(Xa, Xb, iters=2)
        Xa_al, _, R2 = align_parents(Xb, Xa, iters=1)
        cand_children = []
        try:
            c1, c2 = halfmix_child(Xa, Xb_al)
            cand_children.append(c1); cand_children.append(c2)
        except Exception:
            pass
        try:
            h1 = hubswap_child(Xa, Xb, R)
            if h1 is not None:
                cand_children.append(h1)
            h2 = hubswap_child(Xb, Xa, R2)
            if h2 is not None:
                cand_children.append(h2)
        except Exception:
            pass
        for ch in cand_children:
            if time.time()-t0 > LIM-22:
                break
            try:
                f = fingerprint(ch)
            except Exception:
                continue
            is_new = True
            for g in seen:
                try:
                    if fp_dist(f, g) < 5e-5:
                        is_new = False
                        break
                except Exception:
                    pass
            if not is_new:
                continue
            seen.append(f)
            chp, v = full_polish(ch.copy())
            children_done += 1
            if v < best_val:
                best, best_val = chp.copy(), float(v)
                # keep archive fresh
                parents.append(chp.copy())
            if children_done >= 8:
                break
        if children_done >= 8:
            break

    # final intensive polish of global best
    if time.time()-t0 < LIM-6:
        try:
            Xf = best.copy()
            Xf = lse_polish(Xf, 0.002, 60)
            Xf, vf = greedy(Xf, 120, 0.001)
            if vf < best_val:
                best, best_val = Xf.copy(), float(vf)
            Xf = lse_polish(best.copy(), 0.0006, 40)
            Xf, vf = greedy(Xf, 80, 0.0005)
            if vf < best_val:
                best, best_val = Xf.copy(), float(vf)
        except Exception:
            pass
    best = _norm(best)
    return best
