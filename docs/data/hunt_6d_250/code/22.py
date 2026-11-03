import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20.0 * radius, 5e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            thr = np.sort(iu_c[sel])[-4000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _build_pool():
    d = 6
    vecs = []
    # all nonzero integer vectors coords in -2..2 with small support (nnz<=4)
    for tup in itertools.product((-2, -1, 0, 1, 2), repeat=d):
        nnz = 0
        for v in tup:
            if v != 0:
                nnz += 1
        if nnz == 0 or nnz > 4:
            continue
        vecs.append(tup)
    V = np.array(vecs, dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    # dedup directions (exact)
    seen = {}
    uniq = []
    for v in V:
        key = tuple(np.round(v, 8))
        if key not in seen:
            seen[key] = True
            uniq.append(v)
    P = np.array(uniq, dtype=np.float64)
    # extra algebraic patterns: signed permutations of given patterns
    extra = []
    pats = [(np.array([1, 1, 1, 1, 0, 0], float), 2.0),
            (np.array([2, 1, 1, 0, 0, 0], float), np.sqrt(6.0)),
            (np.array([1, 1, 1, 0, 0, 0], float), np.sqrt(3.0))]
    import itertools as it
    for base, nr in pats:
        # all choices of support positions
        nz = int(np.count_nonzero(base))
        for pos in it.combinations(range(d), nz):
            # all sign combos for nonzero entries
            for signs in it.product((-1.0, 1.0), repeat=nz):
                # all permutations of base nonzero values (to cover (2,1,1) orderings)
                vals = base[base != 0]
                # unique permutations
                seenp = set()
                for perm in it.permutations(vals):
                    if perm in seenp:
                        continue
                    seenp.add(perm)
                    v = np.zeros(d)
                    for k, pp in enumerate(perm):
                        v[pos[k]] = pp * signs[k]
                    v = v / np.linalg.norm(v)
                    extra.append(v)
    if extra:
        E = np.array(extra)
        for v in E:
            key = tuple(np.round(v, 8))
            if key not in seen:
                seen[key] = True
                uniq.append(v)
        P = np.array(uniq, dtype=np.float64)
    # line pool: dedup up to sign (canonical first nonzero positive)
    lseen = {}
    lines = []
    for v in P:
        w = v.copy()
        for c in w:
            if abs(c) > 1e-12:
                if c < 0:
                    w = -w
                break
        key = tuple(np.round(w, 8))
        if key not in lseen:
            lseen[key] = True
            lines.append(w)
    PL = np.array(lines, dtype=np.float64)
    return P, PL

def _greedy_points(P, k, rng, start=None):
    M = P.shape[0]
    if start is None:
        start = int(rng.integers(M))
    sel = [int(start)]
    mask = np.zeros(M, dtype=bool)
    mask[sel[0]] = True
    for _ in range(1, k):
        S = P[sel]
        D = P @ S.T
        rm = D.max(axis=1)
        rm[mask] = np.inf
        m = float(np.min(rm))
        cand = np.where(rm <= m + 1e-12)[0]
        pick = int(cand[0]) if len(cand) == 1 else int(rng.choice(cand))
        sel.append(pick)
        mask[pick] = True
    return P[np.array(sel)].copy(), sel

def _greedy_lines(PL, k, rng, start=None):
    M = PL.shape[0]
    if start is None:
        start = int(rng.integers(M))
    sel = [int(start)]
    mask = np.zeros(M, dtype=bool)
    mask[sel[0]] = True
    for _ in range(1, k):
        S = PL[sel]
        D = np.abs(PL @ S.T)
        rm = D.max(axis=1)
        rm[mask] = np.inf
        m = float(np.min(rm))
        cand = np.where(rm <= m + 1e-12)[0]
        pick = int(cand[0]) if len(cand) == 1 else int(rng.choice(cand))
        sel.append(pick)
        mask[pick] = True
    L = PL[np.array(sel)]
    X = np.vstack([L, -L]).copy()
    return X, sel

def _swap_points(X, P, deadline, eps=1e-5):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = int(np.count_nonzero(G > m - eps) // 2)
    rowmax = G.max(axis=1)
    order = np.argsort(-rowmax)
    for idx_pos, i in enumerate(list(order)):
        if time.time() > deadline:
            break
        i = int(i)
        mask = np.ones(n, dtype=bool); mask[i] = False
        Xr = X[mask]
        Gs = Xr @ Xr.T
        np.fill_diagonal(Gs, -2.0)
        max_wo = float(np.max(Gs))
        D = P @ Xr.T
        pmax = D.max(axis=1)
        ach = np.maximum(max_wo, pmax)
        bi = int(np.argmin(ach))
        bv = float(ach[bi])
        if bv < m - 1e-12:
            X[i] = P[bi]
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G)); cnt = int(np.count_nonzero(G > m - eps) // 2)
        elif bv <= m + 1e-12:
            K = 15
            part = np.argpartition(ach, K)[:K]
            cnt_wo = int(np.count_nonzero(Gs > m - eps) // 2)
            for p in part:
                p = int(p)
                if ach[p] > m + 1e-12:
                    continue
                cpool = int(np.count_nonzero(D[p] > m - eps))
                if cpool + cnt_wo < cnt:
                    Xt = X.copy(); Xt[i] = P[p]
                    Gt = Xt @ Xt.T; np.fill_diagonal(Gt, -2.0)
                    mt = float(np.max(Gt)); ct = int(np.count_nonzero(Gt > mt - eps) // 2)
                    if mt < m - 1e-12 or (abs(mt - m) <= 1e-12 and ct < cnt):
                        X[i] = P[p]; m = mt; cnt = ct
                        G = Gt
                        break
    return X

def _swap_lines(L, PL, deadline, eps=1e-5):
    k = L.shape[0]
    def lmax(Lm):
        G = np.abs(Lm @ Lm.T)
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))
    m = lmax(L)
    order = np.argsort([np.max(np.abs(L[i] @ L.T)) for i in range(k)])[::-1]
    for i in list(order):
        if time.time() > deadline:
            break
        i = int(i)
        mask = np.ones(k, dtype=bool); mask[i] = False
        Lr = L[mask]
        Gr = np.abs(Lr @ Lr.T); np.fill_diagonal(Gr, -2.0)
        max_wo = float(np.max(Gr)) if Gr.size else -2.0
        D = np.abs(PL @ Lr.T)
        pmax = D.max(axis=1) if D.size else np.zeros(len(PL))
        ach = np.maximum(max_wo, pmax)
        bi = int(np.argmin(ach))
        if float(ach[bi]) < m - 1e-12:
            L[i] = PL[bi]
            m = lmax(L)
    return L

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    # load fallback benchmark
    best0 = None
    best0_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                v = _max_cos(z)
                if v < best0_val:
                    best0_val = v; best0 = z.copy()
        except Exception:
            continue
    if best0 is None:
        best0 = rng.standard_normal((n, d))
        best0 /= np.linalg.norm(best0, axis=1, keepdims=True)
        best0_val = _max_cos(best0)
    # build structured pools
    try:
        P, PL = _build_pool()
    except Exception:
        P, PL = None, None
    struct_cands = []
    disc_deadline = t_start + min(60.0, 42.0)
    if P is not None and rem() > 20:
        # several greedy constructions within ~60s (here capped ~42s)
        try:
            X1, _ = _greedy_points(P, n, rng, start=int(rng.integers(len(P))))
            X1 = _swap_points(X1, P, min(disc_deadline, t_start + 38.0))
            struct_cands.append(X1 / np.linalg.norm(X1, axis=1, keepdims=True))
        except Exception:
            pass
        if time.time() < disc_deadline - 6 and rem() > 20:
            try:
                X2, _ = _greedy_points(P, n, rng, start=int(rng.integers(len(P))))
                X2 = _swap_points(X2, P, disc_deadline - 2)
                struct_cands.append(X2 / np.linalg.norm(X2, axis=1, keepdims=True))
            except Exception:
                pass
        # antipodal variant: 125 lines duplicated
        if time.time() < disc_deadline and rem() > 15:
            try:
                Xa, _ = _greedy_lines(PL, n // 2, rng)
                # discrete swap on lines
                L = Xa[:n // 2].copy()
                L = _swap_lines(L, PL, disc_deadline)
                Xa = np.vstack([L, -L])
                Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
                struct_cands.append(Xa)
            except Exception:
                pass
    # rank structured by discrete max, keep best two
    scored = []
    for X in struct_cands:
        try:
            scored.append((_max_cos(X), X))
        except Exception:
            pass
    scored.sort(key=lambda t: t[0])
    top_struct = [X for _, X in scored[:2]]
    # annealed LogSumExp polish of best two structured
    polished = []
    for X in top_struct:
        if rem() < 22:
            break
        cur = X.copy()
        try:
            for beta, it in [(2500.0, 70), (10000.0, 70)]:
                if rem() < 14:
                    break
                cur = _smooth_opt(cur, beta, it)
            v = _max_cos(cur)
            polished.append((v, cur))
        except Exception:
            continue
    # also polish fallback benchmark (to have fair comparison and ensure progress)
    best_pol = best0.copy()
    best_pol_val = best0_val
    if rem() > 30:
        try:
            cur = best0.copy()
            for beta, it in [(4000.0, 90), (15000.0, 90)]:
                if rem() < 18:
                    break
                cur = _smooth_opt(cur, beta, it)
            v = _max_cos(cur)
            if v < best_pol_val:
                best_pol, best_pol_val = cur.copy(), v
            # include structured polished in comparison pool
            for v2, X2 in polished:
                if v2 < best_pol_val:
                    best_pol, best_pol_val = X2.copy(), v2
            # extra high-beta tie on current leader
            if rem() > 14:
                try:
                    lead = best_pol.copy()
                    lead2 = _smooth_opt(lead, 30000.0, 60)
                    if _max_cos(lead2) < best_pol_val:
                        best_pol, best_pol_val = lead2, _max_cos(lead2)
                except Exception:
                    pass
            # final trust-region LP squeeze on leader
            if rem() > 8:
                try:
                    sq = _lp_polish(best_pol, min(6.0, rem() - 3.0), time.time())
                    vs = _max_cos(sq)
                    if vs < best_pol_val:
                        best_pol, best_pol_val = sq, vs
                except Exception:
                    pass
        except Exception:
            pass
    else:
        for v2, X2 in polished:
            if v2 < best_pol_val:
                best_pol, best_pol_val = X2.copy(), v2
    # final choice: whichever lower between structured polished and loaded best
    # (best_pol already is min over both, but verify against raw best0 too)
    cand_pool = [(best_pol_val, best_pol), (best0_val, best0)]
    for v2, X2 in polished:
        cand_pool.append((v2, X2))
    cand_pool.sort(key=lambda t: t[0])
    final = cand_pool[0][1].copy()
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    # strict feasibility: no zero rows / NaN (already normalised)
    return final
