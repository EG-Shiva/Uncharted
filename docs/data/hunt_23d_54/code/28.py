import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _smooth(x, steps, a0, a1, lr0, lr1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 > 0 else a1
        lr = lr0 * (lr1 / lr0) ** frac if steps > 1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True)
        Gn = Gt / (deg + 1.0)
        v = mu * v + Gn
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    x_best = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 700:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vy = _max_cos(y)
        if vy < best:
            x, best = y, vy
            x_best = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x_best

def _golay_basis Poly_exp(exp_list):
    g = np.zeros(23, dtype=int)
    g[exp_list] = 1
    B = np.zeros((12, 23), dtype=int)
    for i in range(12):
        B[i, :] = np.roll(g, i)
    return B

def _golay_pool():
    # two candidate generator polynomials, pick one with min weight 7
    exps1 = [0, 1, 5, 6, 7, 9, 11]
    exps2 = [0, 2, 4, 5, 6, 10, 11]
    bestB = None
    for exps in (exps1, exps2):
        B = _golay_basis_poly_exp(exps) if False else None
        pass
    return None

def _build_pool(rng, d=23):
    pools = []
    # --- binary Golay (23,12,7) cyclic ---
    def make_B(exps):
        g = np.zeros(23, dtype=int)
        g[exps] = 1
        B = np.zeros((12, 23), dtype=int)
        for i in range(12):
            B[i, :] = np.roll(g, i)
        return B
    B1 = make_B([0, 1, 5, 6, 7, 9, 11])
    B2 = make_B([0, 2, 4, 5, 6, 10, 11])
    # choose B with min nonzero weight 7
    chosen = None
    for B in (B1, B2):
        masks = np.zeros((4096, 12), dtype=int)
        for m in range(4096):
            mm = m
            for i in range(12):
                masks[m, i] = mm & 1
                mm >>= 1
                if mm == 0 and i >= 0:
                    pass
        # recompute properly fast
        break
    # vectorized masks
    mvals = np.arange(4096)[:, None]
    bits = ((mvals >> np.arange(12)[None, :]) & 1).astype(int)  # 4096x12
    best_min = -1
    bestC = None
    for B in (B1, B2):
        Cbin = (bits @ B) % 2  # 4096x23
        w = Cbin.sum(axis=1)
        wmin = int(w[w > 0].min()) if np.any(w > 0) else 99
        Cbip = (1 - 2 * Cbin).astype(float) / np.sqrt(23)
        if bestC is None or (wmin == 7):
            bestC = Cbip
            best_min = wmin
            if wmin == 7:
                break
    pools.append(bestC)
    pools.append(-bestC)
    # --- Sylvester Hadamard 32 -> delete 9 coords ---
    H = np.array([[1.0]])
    for _ in range(5):
        H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]]))
    Hs = H[:, :d] / np.sqrt(d)
    pools.append(Hs)
    pools.append(-Hs)
    # --- coordinate e_i ---
    E = np.eye(d)
    pools.append(E)
    pools.append(-E)
    # --- e_i +- e_j normalized (all) ---
    lst = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = s1
                    v[j] = s2
                    v /= np.sqrt(2.0)
                    lst.append(v)
    pools.append(np.array(lst))
    # --- sparse supports 4-8 ---
    for k in (4, 5, 6, 7, 8):
        m = 3000
        M = np.zeros((m, d))
        for t in range(m):
            supp = rng.choice(d, size=k, replace=False)
            sgn = rng.choice(np.array([-1.0, 1.0]), size=k)
            M[t, supp] = sgn / np.sqrt(k)
        pools.append(M)
        pools.append(-M)
    # --- random bipolar + random sphere ---
    m2 = 12000
    Rb = rng.randint(0, 2, size=(m2, d)).astype(float)
    Rb = (1 - 2 * Rb) / np.sqrt(d)
    pools.append(Rb)
    m3 = 8000
    Rg = rng.randn(m3, d)
    Rg /= np.linalg.norm(Rg, axis=1, keepdims=True)
    pools.append(Rg)
    C = np.vstack(pools)
    # normalize (safety)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    # deduplicate exact (fast hash via unique)
    # use rounding to 10 decimals to catch negation duplicates? keep both signs so no rounding merge of near values
    try:
        _, uidx = np.unique((C * 1e8).astype(np.int64), axis=0, return_index=True)
        C = C[np.sort(uidx)]
    except Exception:
        pass
    # cap size to ~60000 for speed
    if C.shape[0] > 60000:
        sel = rng.choice(C.shape[0], size=60000, replace=False)
        C = C[sel]
    return C

def solve():
    t0 = time.time()
    BUDGET = 176.0
    rng = np.random.RandomState(7)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d):
            vr = _max_cos(xr)
            if vr < best_val:
                best_val = vr
                x_best = xr.copy()
    except Exception:
        pass

    # ---- discrete pool ----
    try:
        C = _build_pool(np.random.RandomState(0), d=d)
    except Exception:
        C = None
    x_disc = None
    disc_val = 1e9
    if C is not None:
        try:
            P = C.shape[0]
            # greedy best-hole insertion
            sel = []
            scores = np.full(P, -2.0)
            first = 0
            sel.append(first)
            dots = C @ C[first]
            scores = np.maximum(scores, dots)
            scores[first] = np.inf
            for _ in range(1, n):
                idx = int(np.argmin(scores))
                sel.append(idx)
                dots = C @ C[idx]
                # update only non-selected
                scores = np.maximum(scores, dots)
                scores[idx] = np.inf
                for s in sel:
                    scores[s] = np.inf
            S = C[np.array(sel)].copy()
            # one-for-one swaps minimizing max then counts
            t_disc_end = t0 + 75.0
            # helper to compute global max
            def gmax(M):
                G = M @ M.T
                np.fill_diagonal(G, -2.0)
                return float(G.max()), G
            cur_max, G = gmax(S)
            # swap loop
            for it in range(60):
                if time.time() > t_disc_end:
                    break
                # base maxima
                base_max = np.empty(n)
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    Gm = G[np.ix_(mask, mask)]
                    np.fill_diagonal(Gm, -2.0)
                    base_max[i] = float(Gm.max())
                M = C @ S.T  # P x n
                arg1 = np.argmax(M, axis=1)
                max1 = M[np.arange(P), arg1]
                M2 = M.copy()
                M2[np.arange(P), arg1] = -2.0
                max2 = M2.max(axis=1)
                is_sel = np.zeros(P, dtype=bool)
                is_sel[np.array(sel)] = True
                best_i = -1
                best_c = -1
                best_new = cur_max
                # find best max-improving swap
                for i in range(n):
                    h = np.where(arg1 != i, max1, max2)
                    h = h.copy()
                    h[is_sel] = np.inf
                    # exclude candidates duplicating selected? already inf
                    c = int(np.argmin(h))
                    hv = float(h[c])
                    ng = base_max[i] if base_max[i] > hv else hv
                    if ng < best_new - 1e-12:
                        best_new = ng
                        best_i = i
                        best_c = c
                if best_i >= 0:
                    sel[best_i] = best_c
                    S[best_i] = C[best_c]
                    cur_max, G = gmax(S)
                    continue
                # no max improvement: try count reduction at threshold cur_max
                eps = 1e-9
                # current contacts
                iu = np.triu_indices(n, 1)
                cur_cnt = int(np.sum(G[iu] > cur_max - 1e-7))
                # base counts
                improved = False
                for i in range(n):
                    if time.time() > t_disc_end:
                        break
                    hmax = np.where(arg1 != i, max1, max2)
                    # candidates keeping max <= cur_max+1e-12
                    feas = (~is_sel) & (hmax <= cur_max + 1e-12) & (base_max[i] <= cur_max + 1e-12)
                    if not np.any(feas):
                        continue
                    # hole counts
                    # count of j!=i with M[:,j] > cur_max-1e-7
                    thr = cur_max - 1e-7
                    hole_cnt = np.sum(M[:, [k for k in range(n) if k != i]] > thr, axis=1)
                    # base count for i
                    mask = np.ones(n, dtype=bool); mask[i] = False
                    Gb = S[mask] @ S[mask].T
                    np.fill_diagonal(Gb, -2.0)
                    base_cnt = int(np.sum(Gb[np.triu_indices(n - 1, 1)] > cur_max - 1e-7))
                    # current hole count for existing point
                    cur_hole = int(np.sum(M[sel[i], [k for k in range(n) if k != i]] > thr)) if False else 0
                    # total current = cur_cnt ; candidate total = base_cnt + hole_cnt
                    tot = base_cnt + hole_cnt
                    tot[~feas] = 10 ** 9
                    # also mask selected
                    c = int(np.argmin(tot))
                    if tot[c] < cur_cnt - 1e-9:
                        sel[i] = c
                        S[i] = C[c]
                        cur_max, G = gmax(S)
                        improved = True
                        break
                if improved:
                    continue
                break
            # occasional two-for-two on worst pair
            try:
                if time.time() < t_disc_end - 2.0:
                    cur_max, G = gmax(S)
                    np.fill_diagonal(G, -2.0)
                    # worst indices by max contact
                    rowmax = G.max(axis=1)
                    order = np.argsort(-rowmax)
                    i, j = int(order[0]), int(order[1])
                    M = C @ S.T
                    keep = np.ones(n, dtype=bool); keep[i] = False; keep[j] = False
                    R = S[keep]  # (n-2) x d
                    hole = (C @ R.T).max(axis=1)  # P
                    baseG = R @ R.T
                    np.fill_diagonal(baseG, -2.0)
                    baseM = float(baseG.max()) if R.shape[0] > 1 else -2.0
                    K = 400
                    if P > K:
                        o = np.argsort(hole)[:K]
                    else:
                        o = np.arange(P)
                    A = C[o]
                    hA = hole[o]
                    Gram = A @ A.T
                    np.fill_diagonal(Gram, -2.0)
                    best_pair = None
                    best_v = cur_max
                    # brute KxK search vectorized in chunks
                    for a in range(K):
                        # vector over b
                        hv = np.maximum(hA[a], hA)
                        gv = np.maximum(hv, Gram[a, :])
                        gv = np.maximum(gv, baseM)
                        # skip pairs already selected duplicates? rough
                        b = int(np.argmin(gv))
                        if gv[b] < best_v - 1e-12:
                            best_v = float(gv[b])
                            best_pair = (int(o[a]), int(o[b]))
                    if best_pair is not None and best_v < cur_max - 1e-12:
                        # ensure distinct and not already selected (allow)
                        a_c, b_c = best_pair
                        if a_c != b_c:
                            sel[i] = a_c
                            sel[j] = b_c
                            S[i] = C[a_c]; S[j] = C[b_c]
                            cur_max, G = gmax(S)
            except Exception:
                pass
            x_disc = S.copy()
            disc_val = _max_cos(x_disc)
        except Exception:
            x_disc = None

    # choose starting point for continuous polish: better of discrete and incumbent
    x_start = x_best.copy()
    if x_disc is not None:
        try:
            # quick smooth on discrete to evaluate potential
            xd = _smooth(x_disc.copy(), 250, 80, 1200, 0.03, 0.003)
            vd = _max_cos(xd)
            # keep discrete polished version for final choice
            if vd < best_val or disc_val < best_val + 0.02:
                # prefer discrete as start to explore new basin
                x_start = xd
                if vd < best_val:
                    best_val = vd
                    x_best = xd.copy()
            else:
                # still try discrete if time allows? keep incumbent
                pass
            # save discrete smoothed as candidate
            x_disc_sm = xd
        except Exception:
            x_disc_sm = x_disc
    else:
        x_disc_sm = None

    # ---- continuous basin exploration from x_start + final LP ----
    # short annealed smoothing
    try:
        if time.time() - t0 < BUDGET - 30.0:
            x_start = _smooth(x_start.copy(), 600, 60, 1500, 0.025, 0.0025)
            v = _max_cos(x_start)
            if v < best_val:
                best_val = v
                x_best = x_start.copy()
    except Exception:
        pass
    # a couple of quick kicks around discrete/incumbent to try different basin
    try:
        rng2 = np.random.RandomState(1)
        t_cont_end = t0 + BUDGET - 25.0
        it = 0
        while time.time() < t_cont_end and it < 6:
            it += 1
            x = x_best.copy()
            x += rng2.randn(n, d) * (0.01 if it % 2 == 0 else 0.03)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _smooth(x, 500, 70, 1300, 0.028, 0.0025)
            v = _max_cos(x)
            if v < best_val:
                best_val = v
                x_best = x.copy()
    except Exception:
        pass

    # final trust-region LP polish on best
    t_end = t0 + BUDGET
    remain = t_end - time.time() - 0.5
    if remain > 3.0:
        try:
            x_best = _lp_polish(x_best, min(remain, 24.0), t_end - 0.3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best[~np.isfinite(x_best)] = 0.0
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    bad = (nr.squeeze() == 0)
    if np.any(bad):
        x_best[bad] = rng.randn(np.count_nonzero(bad), d)
        x_best[bad] /= np.linalg.norm(x_best[bad], axis=1, keepdims=True)
    return x_best
