import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, p_cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(20 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x


def _build_pool():
    patterns = [
        [1, 1, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
        [2, 1, 1, 1],
        [3, 1, 1],
        [3, 2, 1],
    ]
    D = 13
    allv = []
    for pat in patterns:
        k = len(pat)
        perms = list(set(itertools.permutations(pat)))
        norm = float(np.sqrt(sum(float(v) * float(v) for v in pat)))
        for pos in itertools.combinations(range(D), k):
            for perm in perms:
                # all sign combos
                for mask in range(1 << k):
                    v = [0.0] * D
                    for j in range(k):
                        s = 1.0 if ((mask >> j) & 1) == 0 else -1.0
                        v[pos[j]] = perm[j] * s / norm
                    allv.append(v)
    C = np.asarray(allv, dtype=np.float64)
    # deduplicate
    # round to 12 decimals for stable unique
    Cr = np.round(C, 12)
    _, idx = np.unique(Cr, axis=0, return_index=True)
    idx.sort()
    C = C[idx]
    # remove any zero rows (none)
    nrm = np.linalg.norm(C, axis=1)
    C = C[nrm > 1e-12]
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return C


def _try_12_swap(sel, C, t, neigh, rng, max_tries=40):
    # sel: list of ints, try to grow by 1-for-2 swaps
    sel = list(sel)
    sel_set = set(sel)
    for _ in range(max_tries):
        if len(sel) >= 123:
            break
        # pick random v in sel
        v = sel[int(rng.integers(0, len(sel)))]
        nbr_v = neigh[v]
        cand = []
        # collect vertices only blocked by v
        for u in nbr_v:
            uu = int(u)
            if uu in sel_set:
                continue
            cu = C[uu]
            blocked_by_other = False
            for w in sel:
                if w == v:
                    continue
                if float(cu.dot(C[w])) > t:
                    blocked_by_other = True
                    break
            if not blocked_by_other:
                cand.append(uu)
        if len(cand) < 2:
            continue
        # shuffle cand to randomize pair search
        order = rng.permutation(len(cand))
        found = None
        L = len(cand)
        # limit pairwise checks
        for ii in range(min(L, 30)):
            a = cand[int(order[ii])]
            ca = C[a]
            for jj in range(ii + 1, min(L, 30)):
                b = cand[int(order[jj])]
                if float(ca.dot(C[b])) <= t:
                    found = (a, b)
                    break
            if found is not None:
                break
        if found is not None:
            a, b = found
            sel = [x for x in sel if x != v] + [a, b]
            sel_set = set(sel)
    return sel


def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    rng = np.random.default_rng(0)
    n, d = 123, 13
    # load best / record
    best_arr = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_arr = a.copy()
        except Exception:
            continue
    if best_arr is None:
        best_arr = rng.normal(size=(n, d))
        best_arr /= np.linalg.norm(best_arr, axis=1, keepdims=True)
        best_val = _max_cos(best_arr)

    # --- discrete pool ---
    try:
        C = _build_pool()
    except Exception:
        C = None
    found_X = None
    found_val = 1e9
    if C is not None and C.shape[0] > 10:
        M0 = C.shape[0]
        MAXM = 6000
        if M0 > MAXM:
            # stratified deterministic subsample
            pick = rng.choice(M0, size=MAXM, replace=False)
            pick.sort()
            C = C[pick]
        M = C.shape[0]
        Cf = C.astype(np.float32)
        thresholds = [0.270, 0.268, 0.266, 0.264, 0.262, 0.260]
        # time budget: leave at least 35s for final polish
        search_deadline = t0 + TIME_LIMIT - 40.0
        best_size_overall = 0
        for t in thresholds:
            if time.time() > search_deadline:
                break
            # build neighbor lists for this t (chunked)
            chunk = 1200
            neigh = [None] * M
            try:
                for s in range(0, M, chunk):
                    e = min(M, s + chunk)
                    Dmat = Cf[s:e] @ Cf.T  # (chunk x M) float32
                    # threshold
                    for r in range(e - s):
                        row = Dmat[r]
                        idx = np.where(row > t)[0].astype(np.int32)
                        # ensure self included
                        neigh[s + r] = idx
                    if time.time() > search_deadline:
                        break
                    del Dmat
            except Exception:
                break
            if time.time() > search_deadline:
                break
            # many fast randomized greedy independent-set searches
            # estimate attempts from remaining time
            attempts = 0
            max_attempts_this_t = 600
            while attempts < max_attempts_this_t and time.time() < search_deadline:
                attempts += 1
                order = rng.permutation(M)
                blocked = np.zeros(M, dtype=bool)
                sel = []
                # manual loop
                for vv in order:
                    vvi = int(vv)
                    if not blocked[vvi]:
                        sel.append(vvi)
                        blocked[neigh[vvi]] = True
                        if len(sel) >= 123:
                            break
                if len(sel) > best_size_overall:
                    best_size_overall = len(sel)
                # augment close sets with 1-for-2 swaps
                if len(sel) >= 108 and len(sel) < 123:
                    # only spend swap time if promising
                    try:
                        sel2 = _try_12_swap(sel, C, t, neigh, rng, max_tries=25)
                        if len(sel2) > len(sel):
                            sel = sel2
                            if len(sel) > best_size_overall:
                                best_size_overall = len(sel)
                    except Exception:
                        pass
                if len(sel) >= 123:
                    sel = sel[:123]
                    X = C[np.asarray(sel, dtype=int)].copy()
                    # de-duplicate jitter to avoid exact ties issues? keep as is
                    v = _max_cos(X)
                    if v < found_val:
                        found_val = v
                        found_X = X.copy()
                    break
            if found_X is not None:
                break
            # if graph too dense and best size tiny, continue to next (sparser? actually lower t denser)
            # thresholds go denser, so if 0.27 already tiny, lower will be worse -> stop early
            if t == thresholds[0] and best_size_overall < 40:
                # still try one more lower? break to save time for polish
                pass

    # if discrete search found a 123-set, LP-polish it with at most 30s
    if found_X is not None:
        polish_sec = min(30.0, max(2.0, TIME_LIMIT - (time.time() - t0) - 3.0))
        try:
            P = _lp_polish(found_X, polish_sec)
            pv = _max_cos(P)
            if pv < best_val:
                best_val = pv
                best_arr = P.copy()
            elif found_val < best_val:
                best_arr = found_X.copy()
                best_val = found_val
        except Exception:
            if found_val < best_val:
                best_arr = found_X.copy()
                best_val = found_val
        # also final squeeze if time remains
        rem = TIME_LIMIT - (time.time() - t0) - 1.0
        if rem > 2.0:
            try:
                P2 = _lp_polish(best_arr, min(rem, 10.0))
                pv2 = _max_cos(P2)
                if pv2 < best_val:
                    best_arr = P2
                    best_val = pv2
            except Exception:
                pass
        best_arr = best_arr / np.linalg.norm(best_arr, axis=1, keepdims=True)
        return best_arr

    # no 123-set: return polished record/best (must beat record / not time out)
    rem = TIME_LIMIT - (time.time() - t0) - 1.0
    if rem > 1.0:
        try:
            P = _lp_polish(best_arr, min(rem, 25.0))
            pv = _max_cos(P)
            if pv < best_val and np.all(np.isfinite(P)):
                best_arr = P
                best_val = pv
        except Exception:
            pass
    best_arr = best_arr / np.linalg.norm(best_arr, axis=1, keepdims=True)
    return best_arr
