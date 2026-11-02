import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases_subset(Xf):
    nf = Xf.shape[0]
    d = Xf.shape[1]
    B = np.empty((nf, d, 4), dtype=np.float64)
    for i in range(nf):
        x = Xf[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    HARD = t0 + 160.0
    rng0 = np.random.default_rng(0)
    n, d = 453, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc * cc, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng0.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    incumbent_V = float(bestV)
    iu_r, iu_c = np.triu_indices(n, 1)

    def restricted_polish(X, free_idx, radius_init, max_solves):
        # X: (n,5), free_idx: array of global indices
        free_idx = np.asarray(sorted(set(int(v) for v in free_idx)), dtype=int)
        nf = int(free_idx.size)
        if nf == 0:
            return X
        pos_of = np.full(n, -1, dtype=int)
        pos_of[free_idx] = np.arange(nf, dtype=int)
        Nvar = nf * 4
        c_obj = np.zeros(Nvar + 1, dtype=np.float64)
        c_obj[-1] = 1.0
        radius = float(radius_init)
        X = X.copy()
        is_free = np.zeros(n, dtype=bool)
        is_free[free_idx] = True
        for _ in range(max_solves):
            if time.time() > HARD - 2.0 or time.time() > T_END - 6.0:
                break
            if radius < 1e-12:
                break
            G = X @ X.T
            # global max (upper triangle)
            # use iu for exactness
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            slack_all = m - c_all
            involve = is_free[iu_r] | is_free[iu_c]
            idx_inv = np.where(involve)[0]
            if idx_inv.size == 0:
                break
            sl = slack_all[idx_inv]
            # tight: slack<2e-5, cap 1500 tightest
            tight_mask = sl < 2e-5
            if np.count_nonzero(tight_mask) == 0:
                take = min(800, idx_inv.size)
                part = np.argpartition(sl, take - 1)[:take]
                tight_idx = idx_inv[part]
            else:
                cand = idx_inv[tight_mask]
                if cand.size > 1500:
                    s2 = slack_all[cand]
                    part = np.argpartition(s2, 1500)[:1500]
                    tight_idx = cand[part]
                else:
                    tight_idx = cand
            I = iu_r[tight_idx]
            J = iu_c[tight_idx]
            p = int(I.size)
            if p == 0:
                break
            b_ub = slack_all[tight_idx].astype(np.float64)
            B = _tangent_bases_subset(X[free_idx])
            # coefficients
            # map to positions
            pI = pos_of[I]
            pJ = pos_of[J]
            both = (pI >= 0) & (pJ >= 0)
            onlyI = (pI >= 0) & (pJ < 0)
            onlyJ = (pI < 0) & (pJ >= 0)
            # build sparse rows
            # count nnz: both->9, single->5
            # build via python lists but vectorized per group
            rows_list = []
            cols_list = []
            vals_list = []
            # precompute aI for rows with pI>=0, aJ for pJ>=0
            # aI = B[pI] . X[J]
            # do in blocks
            brow = np.arange(p, dtype=np.int64)
            # both
            if np.any(both):
                ib = np.where(both)[0]
                BI = B[pI[ib]]  # (kb,5,4)
                XJ = X[J[ib]]   # (kb,5)
                XI = X[I[ib]]
                BJ = B[pJ[ib]]
                aI = np.einsum('bdk,bd->bk', BI, XJ)
                aJ = np.einsum('bdk,bd->bk', BJ, XI)
                for k in range(4):
                    rows_list.append(ib)
                    cols_list.append(pI[ib] * 4 + k)
                    vals_list.append(aI[:, k])
                    rows_list.append(ib)
                    cols_list.append(pJ[ib] * 4 + k)
                    vals_list.append(aJ[:, k])
            if np.any(onlyI):
                ib = np.where(onlyI)[0]
                BI = B[pI[ib]]
                XJ = X[J[ib]]
                aI = np.einsum('bdk,bd->bk', BI, XJ)
                for k in range(4):
                    rows_list.append(ib)
                    cols_list.append(pI[ib] * 4 + k)
                    vals_list.append(aI[:, k])
            if np.any(onlyJ):
                ib = np.where(onlyJ)[0]
                BJ = B[pJ[ib]]
                XI = X[I[ib]]
                aJ = np.einsum('bdk,bd->bk', BJ, XI)
                for k in range(4):
                    rows_list.append(ib)
                    cols_list.append(pJ[ib] * 4 + k)
                    vals_list.append(aJ[:, k])
            # t column
            rows_list.append(brow)
            cols_list.append(np.full(p, Nvar, dtype=np.int64))
            vals_list.append(np.full(p, -1.0, dtype=np.float64))
            rows_f = np.concatenate(rows_list).astype(np.int64)
            cols_f = np.concatenate(cols_list).astype(np.int64)
            vals_f = np.concatenate(vals_list).astype(np.float64)
            try:
                A_ub = coo_matrix((vals_f, (rows_f, cols_f)), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            rem = min(HARD - 2.0 - time.time(), T_END - 6.0 - time.time(), 4.5)
            if rem < 0.6:
                break
            bounds = [(-radius, radius)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                              method="highs", options={"time_limit": max(0.6, rem)})
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0 or res.x is None:
                radius *= 0.5
                continue
            try:
                U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(nf, 4)
            except Exception:
                radius *= 0.5
                continue
            if not np.all(np.isfinite(U)):
                radius *= 0.5
                continue
            DX = np.einsum('ndk,nk->nd', B, U)
            Xt = X.copy()
            Xt[free_idx] = X[free_idx] + DX
            nn = np.sqrt(np.sum(Xt[free_idx] * Xt[free_idx], axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            Xt[free_idx] = Xt[free_idx] / nn
            mt = _max_cos(Xt)
            if np.isfinite(mt) and mt < m - 1e-13:
                X = Xt
                # update free-adjacency for next round
                radius = min(radius * 1.4, 8e-5)
            else:
                radius *= 0.5
        return X

    # main cluster attempts
    n_attempts = 7
    for attempt in range(n_attempts):
        if time.time() > HARD - 12.0:
            break
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            row_max = np.max(G, axis=1)
            mcur = float(np.max(G))
            # contact degrees at 1e-9
            iu_c_all = G[iu_r, iu_c]
            contact = (mcur - iu_c_all) < 1e-9
            deg = np.zeros(n, dtype=int)
            np.add.at(deg, iu_r[contact], 1)
            np.add.at(deg, iu_c[contact], 1)
        except Exception:
            break
        hubs = np.where(deg >= 12)[0]
        if hubs.size == 0:
            hubs = np.argsort(-deg)[:8]
        # pick hub cyclically
        hub = int(hubs[attempt % hubs.size])
        # contact neighbours of hub
        # find pairs involving hub that are contacts
        # use row of G
        grow = G[hub].copy()
        # contact threshold: slack<1e-9 from mcur
        neigh = np.where((mcur - grow) < 1e-9)[0]
        neigh = neigh[neigh != hub]
        rattlers = np.where(deg == 0)[0]
        target = 28 + (attempt * 7) % 13  # 28..40
        cluster = [hub] + [int(v) for v in neigh]
        # add nearby rattler(s): closest rattler to hub (largest dot)
        if rattlers.size > 0:
            dots_r = bestX[rattlers] @ bestX[hub]
            order_r = np.argsort(-dots_r)
            for k in range(min(2, rattlers.size)):
                v = int(rattlers[order_r[k]])
                if v not in cluster:
                    cluster.append(v)
        # fill by closest dots to hub
        if len(cluster) < target:
            dots_all = bestX @ bestX[hub]
            order = np.argsort(-dots_all)
            for v in order:
                v = int(v)
                if v not in cluster:
                    cluster.append(v)
                if len(cluster) >= target:
                    break
        cluster = np.asarray(cluster[:target], dtype=int)
        k = int(cluster.size)
        if k < 20:
            continue
        in_cl = np.zeros(n, dtype=bool)
        in_cl[cluster] = True
        keep_mask = ~in_cl
        X_rem0 = bestX[keep_mask]  # (n-k,5)
        rng = np.random.default_rng(1234 + attempt * 977)
        # greedy reinsertion
        Xref_list = [X_rem0[i].copy() for i in range(X_rem0.shape[0])]
        # stack as array for fast eval: maintain array Xref (grows)
        Xref = X_rem0.copy()
        new_pts = []
        feasible = True
        for si in range(k):
            if time.time() > HARD - 10.0:
                feasible = False
                break
            # 20000 candidates
            C = rng.standard_normal((20000, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
            D = C @ Xref.T
            md = np.max(D, axis=1)
            bi = int(np.argmin(md))
            h = C[bi].copy()
            # local tangent descent refinement (~40 iters)
            for it in range(40):
                dots = Xref @ h
                ci = int(np.argmax(dots))
                cc_ = float(dots[ci])
                xc = Xref[ci]
                tt = xc - cc_ * h
                tn = float(np.linalg.norm(tt))
                if tn < 1e-12:
                    break
                tt /= tn
                step = 0.02 * (0.97 ** it)
                h = h - step * tt
                nh = float(np.linalg.norm(h))
                if nh == 0 or not np.isfinite(nh):
                    break
                h /= nh
            new_pts.append(h)
            Xref = np.vstack([Xref, h.reshape(1, d)])
        if not feasible:
            continue
        # build full candidate
        Xnew = bestX.copy()
        # assign new points to cluster indices in random order to diversify
        perm = rng.permutation(k)
        new_arr = np.asarray(new_pts, dtype=np.float64)
        for ii, ci_ in enumerate(cluster):
            Xnew[int(ci_)] = new_arr[int(perm[ii])]
        # quick exact eval; skip hopeless (>best+0.004)
        vnew = _max_cos(Xnew)
        if not np.isfinite(vnew) or vnew > bestV + 0.004:
            # still try polish only if close? skip to save time
            continue
        # one-ring neighbourhood: points with slack <5e-5 to any cluster point
        try:
            G2 = Xnew @ Xnew.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(np.max(G2))
            # for each j not in cluster, min slack to cluster
            # G2[j, cluster]
            sub = G2[:, cluster]  # n x k
            # slack = m2 - sub
            min_slack = np.min(m2 - sub, axis=1)
            ring = np.where((min_slack < 5e-5) & (~in_cl))[0]
            free_idx = np.concatenate([cluster, ring])
            # cap free size
            if free_idx.size > 130:
                # keep cluster + closest ring
                rs = ring[np.argsort(min_slack[ring])[: max(0, 130 - k)]]
                free_idx = np.concatenate([cluster, rs])
        except Exception:
            free_idx = cluster
        Xpol = restricted_polish(Xnew, free_idx, 8e-5, 4)
        vpol = _max_cos(Xpol)
        if np.isfinite(vpol) and vpol < bestV - 1e-12 and vpol < incumbent_V - 1e-12 + 1e-12:
            # accept only if beats incumbent sequence (strict improvement over bestV)
            # also require beating published-lab best tracked in bestV
            bestX = Xpol.copy()
            bestV = float(vpol)
        elif np.isfinite(vpol) and vpol < bestV - 1e-12:
            # still accept intra-search improvement (helps chain)
            bestX = Xpol.copy()
            bestV = float(vpol)
        # hard stop if time low
        if time.time() > HARD - 8.0:
            break

    # final safety: ensure finite, normalized, fallback is best found
    try:
        nn = np.sqrt(np.sum(bestX * bestX, axis=1, keepdims=True))
        nn[nn == 0] = 1.0
        bestX = bestX / nn
    except Exception:
        pass
    return np.asarray(bestX, dtype=np.float64)
