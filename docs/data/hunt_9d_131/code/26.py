import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth(X, steps, T0, T1, lr0, lr1, mom=0.85):
    n = X.shape[0]
    V = np.zeros_like(X)
    for k in range(steps):
        f = k / max(1, steps - 1)
        T = T0 + (T1 - T0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp((G - m) / max(T, 1e-12))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr = Gr - dot * X
        s = float(np.sum(W)) / n + 1e-12
        Gr = Gr / s
        V = mom * V + (1.0 - mom) * Gr
        X = X - lr * V
        X = _norm(X)
    return X

def _lp(X, deadline):
    X = _norm(X)
    n, d = X.shape
    best = _mc(X)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gf = X @ X.T
        ca = Gf[iu0]
        m = float(ca.max())
        th = m - max(20 * radius, 1e-9)
        sel = ca > th
        if int(sel.sum()) == 0:
            break
        if int(sel.sum()) > 4500:
            idx = np.where(sel)[0]
            order = np.argsort(-ca[idx])[:4500]
            mk = np.zeros_like(sel)
            mk[idx[order]] = True
            sel = mk
        I = iu0[0][sel]; J = iu0[1][sel]; c = ca[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _mc(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if deadline - time.time() < 0.3:
            break
    return X

def _degrees(X, mv):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    C = G >= (mv - 1e-7)
    return C.sum(axis=1), G

def _edge_set(X, mv, tol=1e-7):
    G = X @ X.T
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    sel = G[iu] >= (mv - tol)
    return set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))

def _build_pool(rng, n_extra=8000):
    N = 3 ** 9
    idx = np.arange(N, dtype=np.int64)
    dig = np.empty((N, 9), dtype=np.float64)
    tmp = idx.copy()
    for k in range(9):
        dig[:, k] = (tmp % 3) - 1.0
        tmp //= 3
    # zero vector is at idx where all digits 0 -> base3 all 1
    zero_idx = int((N - 1) // 2)
    mask = np.ones(N, dtype=bool)
    mask[zero_idx] = False
    dig = dig[mask]
    nr = np.linalg.norm(dig, axis=1, keepdims=True)
    pool = dig / np.maximum(nr, 1e-12)
    if n_extra > 0:
        extra = rng.integers(-2, 3, size=(n_extra, 9)).astype(np.float64)
        nz = np.linalg.norm(extra, axis=1)
        extra = extra[nz > 1e-9]
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pool = np.vstack([pool, extra])
    return pool

def _greedy_select(pool, rng, k=131, shortlist=8):
    P = pool.shape[0]
    start = int(rng.integers(P))
    sel = [start]
    in_sel = np.zeros(P, dtype=bool)
    in_sel[start] = True
    best = pool @ pool[start]
    best[start] = np.inf
    # ensure coordinate + D9 types are representable; greedy naturally uses them
    for _ in range(1, k):
        # shortlist smallest
        # argpartition for speed
        part = np.argpartition(best, min(shortlist, P - 1))[:shortlist]
        # among part pick minimal with random tie handling: choose uniformly among part sorted? pick random
        # weight toward smaller: pick random among part but bias: evaluate values
        vals = best[part]
        m = float(vals.min())
        # candidates within 1e-12 of m among part, else whole part random
        tight = part[vals <= m + 1e-12]
        if len(tight) > 1:
            pick = int(rng.choice(tight))
        else:
            # random choice among shortlist to diversify (still greedy-ish)
            # pick uniformly among shortlist with prob, else best
            # use 50% best, 50% random in shortlist
            if rng.random() < 0.6:
                pick = int(tight[0])
            else:
                pick = int(rng.choice(part))
        sel.append(pick)
        in_sel[pick] = True
        d = pool @ pool[pick]
        # update best
        # keep inf for selected
        best = np.maximum(best, d)
        best[pick] = np.inf
        # also keep previous selected inf (maximum may have overwritten? no, inf stays)
        # but d for old selected could be inf? no
        for s in sel[:-1]:
            best[s] = np.inf
    S = pool[np.array(sel)]
    return S, np.array(sel, dtype=np.int64)

def _swap_improve(pool, sel_idx, max_swaps=30):
    k = len(sel_idx)
    S = pool[sel_idx].copy()
    # selected set for exclusion (by index in pool; duplicates in extra part may have same direction but different index - allow)
    for _ in range(max_swaps):
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        # max without each i
        # compute quickly
        D = pool @ S.T  # P x k
        # row excl max: for each pool row, need max over j!=i for each i -> need top2 per row
        # get top2
        # argmax
        a1 = np.argmax(D, axis=1)
        m1 = D[np.arange(D.shape[0]), a1]
        D2 = D.copy()
        D2[np.arange(D.shape[0]), a1] = -2.0
        a2 = np.argmax(D2, axis=1)
        m2 = D2[np.arange(D.shape[0]), a2]
        # for each i, excl_max[p] = m1 if a1!=i else m2
        # find best swap
        # max_without_i
        best_gain = 0
        best_pair = None
        # precompute max_without_i for all i
        max_wo = np.empty(k)
        for i in range(k):
            Gm = G.copy()
            Gm[i, :] = -2.0
            Gm[:, i] = -2.0
            max_wo[i] = float(Gm.max())
        # for each i, best pool p minimizing excl
        # vectorized per i would be P*k loop; k=131, P~27k => 3.5M per swap iter, times 131 i => 460M too big if naive
        # instead iterate i (131) each O(P)
        for i in range(k):
            excl = np.where(a1 != i, m1, m2)
            # exclude pool indices already selected (avoid duplicate pick)
            # set excl for selected to inf so not chosen (except allow replacing with itself? skip)
            # quick mask via isin? use boolean array of size P
            # build once outside? do here with set lookup via sel_idx
            # to avoid O(P*k) set, use a boolean selected mask updated each outer iter
            # we compute mask once per outer iter below
            new_max_vec = np.maximum(max_wo[i], excl)
            # mask selected
            # (done outside, apply here)
            # find min over allowed
            # we pass mask in
            # placeholder, will handle with mask array
            pass
        # implement with mask properly (second loop to avoid code dup)
        P = pool.shape[0]
        is_sel = np.zeros(P, dtype=bool)
        # pool may contain duplicate directions but different indices; only exact indices excluded
        is_sel[sel_idx] = True
        best_new = cur_max
        best_pair = None
        for i in range(k):
            excl = np.where(a1 != i, m1, m2)
            excl_masked = excl.copy()
            excl_masked[is_sel] = np.inf
            # allow the current i's own pool index to be considered? it's masked, but keeping it would give same max; skip
            p = int(np.argmin(excl_masked))
            nm = float(max(max_wo[i], excl_masked[p]))
            if nm < best_new - 1e-12:
                best_new = nm
                best_pair = (i, p)
        if best_pair is None:
            # try count-reducing swap: reduce number of near-max pairs
            # count pairs at cur_max-1e-9
            cnt = int(((G >= cur_max - 1e-9).sum()) // 2)
            improved = False
            # try random sample of swaps that keep max same but reduce count
            # evaluate a few candidates
            for _try in range(200):
                i = int(np.random.randint(k)) if False else int(sel_idx[0] * 0)  # placeholder
                break
            break
        i, p = best_pair
        sel_idx[i] = p
        S[i] = pool[p]
        # update is_sel implicitly next iter
    return S, sel_idx

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    n, d = 131, 9
    rng = np.random.default_rng(12345)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm(a)
                s = _mc(a)
                if s < best_v:
                    best_v = s; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _mc(best_x)
    x_best = best_x.copy(); s_best = best_v
    try:
        base_edges = _edge_set(x_best, s_best)
    except Exception:
        base_edges = set()

    # build pool
    pool = _build_pool(rng, n_extra=8000)

    seeds_kept = []  # list of (X, val, deg_zero_count, overlap)
    # discrete phase: 40-60 s total
    DISC_END = min(t0 + 58.0, DEADLINE - 110.0)
    if DISC_END < t0 + 20:
        DISC_END = t0 + 40.0
    n_greedy = 4
    for g in range(n_greedy):
        if time.time() > DISC_END:
            break
        r2 = np.random.default_rng(1000 + g)
        Ssel, sidx = _greedy_select(pool, r2, k=n, shortlist=8)
        try:
            Ssel2, sidx2 = _swap_improve(pool, sidx.copy(), max_swaps=25)
        except Exception:
            Ssel2 = Ssel
        # smooth repulsion short
        try:
            Y = _norm(Ssel2 + 1e-6 * r2.standard_normal(Ssel2.shape))
            Y = _smooth(Y, 260, 0.02, 0.002, 0.012, 0.0015)
            v = _mc(Y)
            # LP short squeeze (1.2s)
            Y2 = _lp(Y, min(time.time() + 1.5, DISC_END + 2.0))
            v2 = _mc(Y2)
            if v2 < v:
                Y, v = Y2, v2
        except Exception:
            continue
        # degrees + overlap
        try:
            deg, _ = _degrees(Y, v)
            nz = int((deg == 0).sum())
            es = _edge_set(Y, v)
            ov = 0.0
            if len(base_edges) > 0 and len(es) > 0:
                inter = len(base_edges.intersection(es))
                ov = inter / max(len(base_edges), len(es))
            else:
                ov = 0.0
            seeds_kept.append((Y.copy(), v, nz, ov))
        except Exception:
            seeds_kept.append((Y.copy(), v, 999, 1.0))
        # retain if no rattlers or diverse even if worse (per brief)
        # update global best
        if v < s_best and np.all(np.isfinite(Y)):
            x_best, s_best = Y.copy(), v
            try:
                base_edges = _edge_set(x_best, s_best)
            except Exception:
                pass
        # rotated version of same discrete seed
        if time.time() > DISC_END:
            break
        try:
            Q, _ = np.linalg.qr(r2.standard_normal((d, d)))
            Yr = Ssel2 @ Q.T
            Yr = _norm(Yr)
            Yr = _smooth(Yr, 220, 0.018, 0.002, 0.011, 0.0015)
            vr = _mc(Yr)
            Yr2 = _lp(Yr, min(time.time() + 1.2, DISC_END + 2.0))
            vr2 = _mc(Yr2)
            if vr2 < vr:
                Yr, vr = Yr2, vr2
            deg2, _ = _degrees(Yr, vr)
            nz2 = int((deg2 == 0).sum())
            es2 = _edge_set(Yr, vr)
            ov2 = len(base_edges.intersection(es2)) / max(len(base_edges), len(es2)) if len(base_edges) > 0 and len(es2) > 0 else 0.0
            seeds_kept.append((Yr.copy(), vr, nz2, ov2))
            if vr < s_best and np.all(np.isfinite(Yr)):
                x_best, s_best = Yr.copy(), vr
                base_edges = _edge_set(x_best, s_best)
        except Exception:
            continue

    # remaining time: topology-changing hops from global best (short, to ensure record beaten/ matched)
    HOP_END = DEADLINE - 6.0
    # degrees-based probs
    def probs_deg(X, mv):
        try:
            deg, _ = _degrees(X, mv)
        except Exception:
            deg = np.zeros(X.shape[0])
        w = (deg.astype(float) + 1.0) ** 1.5
        return w / w.sum(), deg
    try:
        probs, deg = probs_deg(x_best, s_best)
    except Exception:
        probs = np.full(n, 1.0 / n); deg = np.zeros(n)
    trial = 0
    rng2 = np.random.default_rng(7)
    while time.time() < HOP_END:
        trial += 1
        if trial % 10 == 1:
            probs, deg = probs_deg(x_best, s_best)
        mode = trial % 6
        Y = None
        try:
            if mode == 0:
                Y = _norm(x_best + 0.018 * rng2.standard_normal((n, d)))
                Y = _smooth(Y, 200, 0.015, 0.002, 0.011, 0.0015)
            elif mode == 1:
                k = 2 if (trial % 2 == 0) else 3
                idx = rng2.choice(n, k, replace=False, p=probs)
                Y = x_best.copy()
                Y[idx] += 0.14 * rng2.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 240, 0.015, 0.002, 0.011, 0.0012)
            elif mode == 2:
                # inject diverse discrete seed material occasionally
                if len(seeds_kept) > 0 and (trial % 7 == 0):
                    cand = seeds_kept[rng2.integers(len(seeds_kept))][0]
                    # crossover: take half points from cand (aligned arbitrarily) + half from best with jitter
                    Y = x_best.copy()
                    m = 15
                    idx = rng2.choice(n, m, replace=False)
                    cidx = rng2.choice(n, m, replace=False)
                    Y[idx] = cand[cidx] + 0.01 * rng2.standard_normal((m, d))
                    Y = _norm(Y)
                    Y = _smooth(Y, 220, 0.014, 0.002, 0.010, 0.0012)
                else:
                    idx = rng2.choice(n, 1, replace=False, p=probs)
                    Y = x_best.copy()
                    Y[idx] += 0.10 * rng2.standard_normal((1, d))
                    Y = _norm(Y)
                    Y = _smooth(Y, 220, 0.014, 0.002, 0.010, 0.0012)
            elif mode == 3:
                # rattler / hub hole reinsertion via random hole search
                order = np.argsort(-deg)
                k = 1 if (trial % 3) else 2
                rem = order[:k]
                keep = np.array([i for i in range(n) if i not in set(rem.tolist())])
                Xfix = x_best[keep]
                ns = 8000
                Z = _norm(rng2.standard_normal((ns, d)))
                worst = (Z @ Xfix.T).max(axis=1)
                holes = []
                cur = worst.copy()
                for _hh in range(k):
                    ii = int(np.argmin(cur))
                    z = Z[ii].copy()
                    for _ in range(80):
                        dots = Xfix @ z
                        m = float(dots.max())
                        W = np.exp((dots - m) / 0.008)
                        W = W / (W.sum() + 1e-12)
                        gg = W @ Xfix
                        gg = gg - float(np.dot(gg, z)) * z
                        z = z - 0.02 * gg
                        z = z / max(np.linalg.norm(z), 1e-12)
                    holes.append(z)
                    dp = Z @ holes[-1]
                    cur = np.maximum(cur, dp * 0.999)
                    cur[ii] = 1e9
                Y = x_best.copy()
                Y[rem] = np.array(holes)
                Y = _smooth(Y, 200, 0.011, 0.0018, 0.009, 0.001)
            elif mode == 4:
                Y = _norm(x_best + 0.010 * rng2.standard_normal((n, d)))
                idx = rng2.choice(n, 1, replace=False, p=probs)
                Y[idx] += 0.12 * rng2.standard_normal((1, d))
                Y = _norm(Y)
                Y = _smooth(Y, 220, 0.014, 0.002, 0.010, 0.0012)
            else:
                k = 10
                idx = rng2.choice(n, k, replace=False)
                Y = x_best.copy()
                Y[idx] += 0.05 * rng2.standard_normal((k, d))
                Y = _norm(Y)
                Y = _smooth(Y, 200, 0.013, 0.002, 0.010, 0.0012)
            v = _mc(Y)
            if v < s_best:
                Y2 = _lp(Y, min(time.time() + 1.5, HOP_END + 1.0))
                v2 = _mc(Y2)
                if v2 < s_best:
                    x_best, s_best = Y2, v2
                    probs, deg = probs_deg(x_best, s_best)
            elif v < s_best + 0.002 and (trial % 15 == 0):
                Y2 = _lp(Y, time.time() + 0.8)
                v2 = _mc(Y2)
                if v2 < s_best:
                    x_best, s_best = Y2, v2
                    probs, deg = probs_deg(x_best, s_best)
        except Exception:
            continue
        if HOP_END - time.time() < 0.8:
            break
    try:
        x_best = _lp(x_best, DEADLINE - 0.5)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
