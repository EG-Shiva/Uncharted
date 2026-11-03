import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t_start = time.time()
    BUDGET = 175.0
    RESERVE_FINAL = 20.0
    rng = np.random.default_rng(0)

    n, d = 162, 8
    TDIM = 7
    NCOORD = n * TDIM  # 1134

    # ---------- helpers ----------
    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    def all_G(X):
        G = X @ X.T
        return G

    # seeds
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None and a.shape == (162, 8):
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            cands.append(a)
    if not cands:
        a = rng.standard_normal(size=(162, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(c) for c in cands]
    X_best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    X_cur = X_best.copy()
    cur_val = best_val

    iu0 = np.triu_indices(n, 1)
    iuI0 = iu0[0].copy()
    iuJ0 = iu0[1].copy()

    def tangent_bases(X):
        # 7x8 orthonormal tangent basis per point by SVD
        # Ti is 8x7 with Ti.T @ Ti = I, Ti.T @ x = 0
        T = np.empty((n, 8, 7), dtype=float)
        B = np.empty((n, 7, 8), dtype=float)
        for i in range(n):
            xi = X[i]
            # SVD of 1x8 matrix xi[None,:]
            _, _, Vh = np.linalg.svd(xi[None, :], full_matrices=True)
            Bi = Vh[1:, :]  # 7x8
            B[i] = Bi
            T[i] = Bi.T
        return T, B

    def build_A(X, T, I, J):
        p = len(I)
        A = np.zeros((p, NCOORD), dtype=float)
        # row k: T_i^T x_j at block i, T_j^T x_i at block j
        # loop is fine for p~1000
        for k in range(p):
            i = int(I[k]); j = int(J[k])
            A[k, i*7:(i+1)*7] = T[i].T @ X[j]
            A[k, j*7:(j+1)*7] = T[j].T @ X[i]
        return A

    def smooth_correct(X, iters=180, beta=300.0, lr=0.015, restrict=1e-3):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        for it in range(iters):
            if time.time() - t_start > BUDGET - RESERVE_FINAL - 2.0:
                break
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            # restrict to pairs within `restrict` of max
            # mask far pairs by -inf in exponent
            D = G - m
            # zero out far pairs
            mask = D > -restrict
            # exp, avoid overflow (D<=0)
            E = np.exp(beta * D, out=np.zeros_like(D), where=mask)
            # E already 0 on diag? diag -10-m -> far, so 0
            s = E.sum()
            if not np.isfinite(s) or s <= 0:
                break
            # gradient of weighted sum: (E/s) @ X roughly
            # use unnormalized to keep scale, but normalize step
            P = E / s
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            mx = np.linalg.norm(Gt, axis=1).max()
            if mx < 1e-14 or not np.isfinite(mx):
                break
            cur_lr = lr * (1.0 - 0.5 * it / max(1, iters))
            # increase beta halfway
            if it == iters // 2:
                beta *= 1.4
            step = Gt / mx * cur_lr
            X = X - step
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def lp_polish(X, seconds, r0=1e-3, rmax=0.05):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        best = max_cos(X)
        radius = r0
        t0 = time.time()
        while time.time() - t0 < seconds and radius > 1e-14:
            if time.time() - t_start > BUDGET - 1.5:
                break
            Gfull = X @ X.T
            cfull = Gfull[iuI0, iuJ0]
            m = float(cfull.max())
            if m < best:
                best = m
            sel = cfull > m - max(20.0 * radius, 1e-9)
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                idx = np.argpartition(-cfull, 2500)[:2500]
                mask = np.zeros_like(sel)
                mask[idx] = True
                sel = mask
                cnt = 2500
            I = iuI0[sel]; J = iuJ0[sel]
            c = cfull[sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([(I[:, None]*d + np.arange(d)),
                                   (J[:, None]*d + np.arange(d)),
                                   np.full((p, 1), n*d)], axis=1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            dx = res.x[:-1].reshape(n, d)
            Y = X + dx
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            v = max_cos(Y)
            if v < best - 1e-15:
                X = Y
                best = v
                radius = min(radius*1.5, rmax)
            else:
                radius *= 0.5
        return X

    # ---------- manifold crawl ----------
    STEPS = np.array([0.002, -0.002, 0.01, -0.01, 0.03, -0.03, 0.08, -0.08], dtype=float)
    SLACK = 1e-6
    SING_TOL = 1e-8

    crawl_iters = 0
    # run crawl until reserve time
    while time.time() - t_start < BUDGET - RESERVE_FINAL - 3.0:
        crawl_iters += 1
        # build contacts for X_cur with slack < 1e-6
        Gc = X_cur @ X_cur.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(Gc.max())
        # pairs
        # use full triu for selection
        cfull = Gc[iuI0, iuJ0]
        sel = cfull > mcur - SLACK
        I = iuI0[sel]; J = iuJ0[sel]
        p = len(I)
        if p < 10 or p > 5000:
            # fallback: take top 1100
            k = min(1100, len(cfull))
            idx = np.argpartition(-cfull, k-1)[:k]
            I = iuI0[idx]; J = iuJ0[idx]
            p = k
        # tangent bases by SVD
        try:
            T, _ = tangent_bases(X_cur)
        except Exception:
            break
        try:
            A = build_A(X_cur, T, I, J)
        except Exception:
            break
        # dense SVD
        try:
            U, S, Vh = np.linalg.svd(A, full_matrices=False)
        except Exception:
            break
        # pool of low singular vectors
        small_idx = np.where(S < SING_TOL)[0]
        if len(small_idx) >= 5:
            pool_idx = small_idx
        else:
            # lowest 25 available
            kpool = min(25, len(S))
            pool_idx = np.argsort(S)[-kpool:] if len(S) >= kpool else np.arange(len(S))
            # actually argsort ascending, take smallest
            pool_idx = np.argsort(S)[:kpool] if False else np.argsort(S)[len(S)-kpool:]
            # S sorted descending by svd, so smallest are at end
            pool_idx = np.arange(len(S)-kpool, len(S))
        Vpool = Vh[pool_idx, :]  # kpool x N
        kpool = Vpool.shape[0]

        ndir = 4
        best_cand_this = None
        best_cand_val = np.inf
        # pre-generate complement helper: Vh for projection
        # for each direction
        for di in range(ndir):
            if time.time() - t_start > BUDGET - RESERVE_FINAL - 3.0:
                break
            coeff = rng.standard_normal(size=kpool)
            # random combination of low singular vectors
            w = coeff @ Vpool  # N
            # add exact-null complement component
            z = rng.standard_normal(size=NCOORD)
            # project out row space: z -= Vh.T @ (Vh @ z)
            Vz = Vh @ z
            z = z - Vh.T @ Vz
            nz = np.linalg.norm(z)
            nw = np.linalg.norm(w)
            if nz > 1e-12 and nw > 1e-12:
                w = 0.7 * (w / nw) + 0.3 * (z / nz)
            elif nz > 1e-12:
                w = z / nz
            else:
                w = w / (nw + 1e-12)
            # w is null to first order, unit norm
            # reshape to per-point
            W2 = w.reshape(n, 7)  # a_i
            # dx per point: T[i] @ a_i
            # compute DX (n,8)
            DX = np.einsum('nij,nj->ni', T, W2)
            # test step sizes, first raw filter
            # compute raw stepped values quickly
            raw_vals = []
            Xstep_list = []
            for s in STEPS:
                Y = X_cur + s * DX
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                v = max_cos(Y)
                raw_vals.append(v)
                Xstep_list.append(Y)
            # pick 2 most promising steps with raw < cur+0.008 to polish
            order = np.argsort(raw_vals)
            polished = 0
            for oi in order:
                if polished >= 2:
                    break
                rv = raw_vals[oi]
                if rv > cur_val + 0.008 and rv > best_val + 0.008:
                    continue
                Y0 = Xstep_list[oi]
                # corrective smooth restricted to 1e-3
                Yp = smooth_correct(Y0, iters=160, beta=280.0, lr=0.014, restrict=1e-3)
                vp = max_cos(Yp)
                # track best of this iteration
                if vp < best_cand_val:
                    best_cand_val = vp
                    best_cand_this = Yp.copy()
                # global best?
                if vp < best_val - 1e-12:
                    X_best = Yp.copy()
                    best_val = vp
                polished += 1
            # if time low break
        if best_cand_this is not None:
            # move crawl point: allow sideways/uphill within tolerance
            # if improvement, definitely move; else move if within 1.5e-3 of cur
            if best_cand_val < cur_val + 0.0015:
                X_cur = best_cand_this.copy()
                cur_val = float(best_cand_val)
                # if we found new global best, quick LP lock (short)
                if cur_val <= best_val + 1e-12 and (time.time() - t_start) < BUDGET - RESERVE_FINAL - 10:
                    # short LP 1.2s to squeeze
                    X_lp = lp_polish(X_cur, 1.2, r0=0.002)
                    v_lp = max_cos(X_lp)
                    if v_lp < best_cand_val:
                        X_cur = X_lp
                        cur_val = v_lp
                    if v_lp < best_val:
                        X_best = X_lp.copy()
                        best_val = v_lp
            else:
                # stuck: randomize slightly along null with bigger step to escape?
                # take a random null step without polish as new cur if too many stalls
                if crawl_iters % 6 == 0:
                    # force move to least-bad
                    X_cur = best_cand_this.copy()
                    cur_val = float(best_cand_val)
        else:
            # no candidate: perturb cur slightly
            X_cur = X_cur + 0.005 * rng.standard_normal(size=X_cur.shape)
            X_cur /= np.linalg.norm(X_cur, axis=1, keepdims=True)
            cur_val = max_cos(X_cur)
            if cur_val < best_val:
                X_best = X_cur.copy()
                best_val = cur_val
        # safety: if cur drifted too bad, reset to best
        if cur_val > best_val + 0.004:
            X_cur = X_best.copy()
            cur_val = best_val

    # final full LP polish on best
    remain = BUDGET - (time.time() - t_start) - 1.0
    if remain > 4:
        # polish best, and also cur if competitive
        X_best = lp_polish(X_best, remain * 0.75)
        best_val = max_cos(X_best)
        # if cur is close, polish it too with leftover
        leftover = BUDGET - (time.time() - t_start) - 1.0
        if leftover > 3:
            X_alt = lp_polish(X_cur, leftover * 0.5)
            v_alt = max_cos(X_alt)
            if v_alt < best_val:
                X_best = X_alt
                best_val = v_alt
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(X_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    X_best = X_best / nr
    return X_best
