import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
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

def _tangent_bases_subset(X, idx):
    m = len(idx)
    B = np.empty((m, 5, 4), dtype=np.float64)
    for a, i in enumerate(idx):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((5, 4), dtype=np.float64)
        c = 0
        for j in range(5):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[a] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    HOLE_CUTOFF = t0 + 45.0
    LP_CUTOFF = t0 + 150.0
    rng = np.random.default_rng(0)
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
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    X = bestX.copy()
    cur_best_V = float(bestV)
    cur_best_X = bestX.copy()

    # contact degrees at slack 1e-7
    try:
        G0 = X @ X.T
        iu_r0 = np.triu_indices(n, 1)[0]
        iu_c0 = np.triu_indices(n, 1)[1]
        # reuse triu
        iu_r, iu_c = iu_r0, iu_c0
        c_all0 = G0[iu_r, iu_c]
        m0 = float(np.max(c_all0))
        deg = np.zeros(n, dtype=int)
        cmask = (m0 - c_all0) < 1e-7
        np.add.at(deg, iu_r[cmask], 1)
        np.add.at(deg, iu_c[cmask], 1)
        Gtmp = G0.copy()
        np.fill_diagonal(Gtmp, -2.0)
        row_max = np.max(Gtmp, axis=1)
    except Exception:
        iu_r, iu_c = np.triu_indices(n, 1)
        deg = np.zeros(n, dtype=int)
        row_max = np.zeros(n)
        m0 = cur_best_V

    low_idx = np.where(deg <= 2)[0].tolist()
    # ensure at least rattlers + few lowest
    if len(low_idx) < 6:
        order = np.argsort(deg + row_max * 1e-6)
        low_idx = order[:max(6, len(low_idx))].tolist()
    # order by degree then row_max descending (worst holes last? do rattlers first)
    low_idx = sorted(low_idx, key=lambda v: (int(deg[v]), -float(row_max[v])))
    # cap to 12 to save time
    low_idx = low_idx[:12]

    # ---- hole recruitment: sample 50000, keep 50, batched LSE refine ----
    taus = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    steps_per = 30
    for vv in list(low_idx):
        if time.time() > HOLE_CUTOFF or time.time() > T_END - 100.0:
            break
        try:
            mask = np.ones(n, dtype=bool)
            mask[vv] = False
            Xf = X[mask]  # 452x5
            NS = 50000
            S = rng.standard_normal((NS, d))
            S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
            D0 = S @ Xf.T
            md0 = np.max(D0, axis=1)
            # 50 largest holes = smallest md
            if 50 < NS:
                part = np.argpartition(md0, 50)[:50]
            else:
                part = np.arange(NS)
            C = S[part].copy()  # 50x5
            # batched tangent L-BFGS-like gradient descent on LSE
            for tau in taus:
                step = 1.5 * tau
                for _ in range(steps_per):
                    D = C @ Xf.T  # 50x452
                    mr = np.max(D, axis=1, keepdims=True)
                    E = np.exp((D - mr) / tau)
                    Ssum = np.sum(E, axis=1, keepdims=True) + 1e-300
                    W = E / Ssum
                    Grad = W @ Xf  # 50x5
                    dot = np.sum(Grad * C, axis=1, keepdims=True)
                    Gt = Grad - dot * C
                    # guard
                    if not np.all(np.isfinite(Gt)):
                        break
                    C = C - step * Gt
                    nn = np.linalg.norm(C, axis=1, keepdims=True)
                    nn[nn == 0] = 1.0
                    C = C / nn
                if time.time() > HOLE_CUTOFF:
                    break
            # score refined candidates by exact max to frozen
            Df = C @ Xf.T
            mdf = np.max(Df, axis=1)
            bi = int(np.argmin(mdf))
            h = C[bi] / (np.linalg.norm(C[bi]) + 1e-30)
            # count near-contacts at global max scale
            # accept if hole depth is competitive
            hole_depth = float(mdf[bi])
            # global max elsewhere (excluding vv)
            Ge = Xf @ Xf.T
            np.fill_diagonal(Ge, -2.0)
            m_else = float(np.max(Ge))
            # allow tiny increase up to 2e-6 to enable re-bracing
            if hole_depth > m_else + 2e-6 and hole_depth > cur_best_V + 2e-6:
                # try next best that fits? find first fitting
                order_c = np.argsort(mdf)
                placed = False
                for bi2 in order_c:
                    if float(mdf[bi2]) <= m_else + 2e-6:
                        h = C[int(bi2)] / (np.linalg.norm(C[int(bi2)]) + 1e-30)
                        placed = True
                        hole_depth = float(mdf[bi2])
                        break
                if not placed:
                    continue
            old = X[vv].copy()
            X[vv] = h
            Gn = X @ X.T
            np.fill_diagonal(Gn, -2.0)
            vn = float(np.max(Gn))
            if np.isfinite(vn) and vn <= cur_best_V + 2e-6:
                # keep; update best if improved
                if vn < cur_best_V - 1e-13:
                    cur_best_V = vn
                    cur_best_X = X.copy()
            else:
                X[vv] = old
        except Exception:
            continue

    # sync best
    if cur_best_V < bestV:
        bestV = cur_best_V
        bestX = cur_best_X.copy()
    else:
        # X holds recruited holes (maybe same max); keep X for LP stage
        # but ensure we track best
        pass
    # exact best of X
    try:
        vx = _max_cos(X)
        if vx < bestV:
            bestV = vx
            bestX = X.copy()
            cur_best_V = vx
            cur_best_X = X.copy()
    except Exception:
        pass

    # ---- restricted HiGHS tangent LP freeing moved + 30 neighbours ----
    # moved = low_idx that were changed (or all low_idx)
    moved = list(low_idx)
    # precompute neighbour sets from current X
    def lp_restricted(Xcur, free_list, radius, max_solves):
        nonlocal cur_best_X, cur_best_V
        free = np.array(sorted(set(int(v) for v in free_list)), dtype=int)
        if free.size == 0:
            return Xcur
        pos = np.full(n, -1, dtype=int)
        pos[free] = np.arange(free.size)
        Nvar = int(free.size * 4)
        c_obj = np.zeros(Nvar + 1, dtype=np.float64)
        c_obj[-1] = 1.0
        Xw = Xcur.copy()
        rad = float(radius)
        for _ in range(max_solves):
            if time.time() > LP_CUTOFF or time.time() > T_END - 12.0:
                break
            if rad < 1e-12:
                break
            G = Xw @ Xw.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            slack_all = m - c_all
            # tight pairs involving free set
            # first select slack<2e-5
            tight0 = np.where(slack_all < 2e-5)[0]
            if tight0.size == 0:
                tight0 = np.argsort(slack_all)[:500]
            # filter to involving free
            If0 = iu_r[tight0]; Jf0 = iu_c[tight0]
            inv = np.zeros(n, dtype=bool)
            inv[free] = True
            keep = inv[If0] | inv[Jf0]
            tight = tight0[keep]
            if tight.size == 0:
                # take tightest involving free
                # find 500 tightest overall then filter expanded
                cand = np.argsort(slack_all)[:3000]
                Ic = iu_r[cand]; Jc = iu_c[cand]
                keep2 = inv[Ic] | inv[Jc]
                tight = cand[keep2][:1000]
                if tight.size == 0:
                    break
            if tight.size > 2500:
                part = np.argpartition(slack_all[tight], 2500)[:2500]
                tight = tight[part]
            I = iu_r[tight]; J = iu_c[tight]
            p = int(I.size)
            b_ub = slack_all[tight].astype(np.float64)
            Bfull = _tangent_bases(Xw)
            aI = np.einsum('pdk,pd->pk', Bfull[I], Xw[J])
            aJ = np.einsum('pdk,pd->pk', Bfull[J], Xw[I])
            mI = inv[I]; mJ = inv[J]
            # count nnz
            # build coo arrays
            # t column always
            nI = int(np.sum(mI)); nJ = int(np.sum(mJ))
            tot = nI * 4 + nJ * 4 + p
            rows = np.empty(tot, dtype=np.int64)
            cols = np.empty(tot, dtype=np.int64)
            vals = np.empty(tot, dtype=np.float64)
            ptr = 0
            # t column
            rows[ptr:ptr+p] = np.arange(p, dtype=np.int64)
            cols[ptr:ptr+p] = Nvar
            vals[ptr:ptr+p] = -1.0
            ptr += p
            if nI > 0:
                rI = np.where(mI)[0]
                # row indices repeated
                rr = np.repeat(rI, 4)
                # col indices
                cc = (pos[I[rI]][:, None] * 4 + np.arange(4)[None, :]).ravel()
                vv_ = aI[rI].ravel()
                rows[ptr:ptr+rr.size] = rr
                cols[ptr:ptr+rr.size] = cc
                vals[ptr:ptr+rr.size] = vv_
                ptr += rr.size
            if nJ > 0:
                rJ = np.where(mJ)[0]
                rr = np.repeat(rJ, 4)
                cc = (pos[J[rJ]][:, None] * 4 + np.arange(4)[None, :]).ravel()
                vv_ = aJ[rJ].ravel()
                rows[ptr:ptr+rr.size] = rr
                cols[ptr:ptr+rr.size] = cc
                vals[ptr:ptr+rr.size] = vv_
                ptr += rr.size
            try:
                A_ub = coo_matrix((vals[:ptr], (rows[:ptr], cols[:ptr])), shape=(p, Nvar + 1)).tocsr()
            except Exception:
                rad *= 0.5
                continue
            rem = min(LP_CUTOFF - time.time(), T_END - 12.0 - time.time(), 9.0)
            if rem < 0.5:
                break
            bounds = [(-rad, rad)] * Nvar + [(None, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
            except Exception:
                rad *= 0.5
                continue
            if res.status != 0 or res.x is None:
                rad *= 0.5
                continue
            try:
                U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(free.size, 4)
            except Exception:
                rad *= 0.5
                continue
            if not np.all(np.isfinite(U)):
                rad *= 0.5
                continue
            Bf = Bfull[free]  # F x5x4
            DXf = np.einsum('fdk,fk->fd', Bf, U)
            Xt = Xw.copy()
            Xt[free] = Xw[free] + DXf
            nn = np.linalg.norm(Xt[free], axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            Xt[free] = Xt[free] / nn
            Gt = Xt @ Xt.T
            np.fill_diagonal(Gt, -2.0)
            mt = float(np.max(Gt))
            if np.isfinite(mt) and mt < m - 1e-13:
                Xw = Xt
                if mt < cur_best_V - 1e-13:
                    cur_best_V = mt
                    cur_best_X = Xt.copy()
                rad = min(rad * 1.5, 5e-4)
            else:
                rad *= 0.5
        return Xw

    # cycle through rattlers/low-degree points repeatedly
    # compute nearest neighbours for each moved point
    try:
        Gnn = X @ X.T
        np.fill_diagonal(Gnn, -2.0)
    except Exception:
        Gnn = None
    for cyc in range(3):
        if time.time() > LP_CUTOFF or time.time() > T_END - 12.0:
            break
        for vv in list(moved):
            if time.time() > LP_CUTOFF or time.time() > T_END - 12.0:
                break
            try:
                # 30 nearest neighbours by cosine
                row = (X[vv] @ X.T)
                row[vv] = -2.0
                nb = np.argsort(-row)[:30]
                F = [vv] + [int(x) for x in nb]
                # also include other moved points within close range? keep small
                Xnew = lp_restricted(X, F, 1e-4, 3)
                # accept only if exact max improves (or ties keep for topology?)
                vn = _max_cos(Xnew)
                vo = _max_cos(X)
                if vn <= vo + 1e-12 and vn <= cur_best_V + 1e-9:
                    X = Xnew
                    if vn < cur_best_V:
                        cur_best_V = vn
                        cur_best_X = X.copy()
                else:
                    # if LP worsened, keep old X; but if Xnew improves best, take it
                    if vn < cur_best_V - 1e-13:
                        X = Xnew
                        cur_best_V = vn
                        cur_best_X = X.copy()
            except Exception:
                continue

    # final union polish freeing all moved + neighbours union (bounded size)
    try:
        if time.time() < LP_CUTOFF - 2.0 and time.time() < T_END - 10.0:
            union = set()
            for vv in moved:
                row = (X[vv] @ X.T)
                row[vv] = -2.0
                nb = np.argsort(-row)[:30]
                union.add(int(vv))
                for u in nb:
                    union.add(int(u))
            # cap size to 200 for speed
            union = sorted(union)[:200]
            X = lp_restricted(X, union, 8e-5, 4)
            vn = _max_cos(X)
            if vn < cur_best_V:
                cur_best_V = vn
                cur_best_X = X.copy()
    except Exception:
        pass

    # choose best
    if cur_best_V < bestV and np.all(np.isfinite(cur_best_X)):
        bestX = cur_best_X.copy()
        bestV = cur_best_V
    # tiny shrink guard + normalize
    try:
        bestX = np.asarray(bestX, dtype=np.float64)
        nn = np.linalg.norm(bestX, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        bestX = bestX / nn
        bestX = np.clip(bestX, -1.0, 1.0)
        # renormalize after clip
        nn2 = np.linalg.norm(bestX, axis=1, keepdims=True)
        nn2[nn2 == 0] = 1.0
        bestX = bestX / nn2
    except Exception:
        pass
    return bestX
