import time
import numpy as np
from scipy.optimize import minimize

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _count_close(X, tol=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    return int(np.sum(G > m - tol) // 2), m

def _polish_subset(X0, free_idx, alphas, iters):
    N, d = X0.shape
    f = len(free_idx)
    if f == 0:
        return X0.copy()
    free_idx = np.asarray(free_idx, dtype=int)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[free_idx] = False
    Xfrozen = X0[frozen_mask].copy()
    # map free position to full row: need inverse
    # build full index array
    U0 = X0[free_idx].copy()  # already unit, use as U
    v0 = U0.ravel().astype(float)

    # precompute row order for assembly
    # for fast assembly, keep permutation
    full_free_pos = free_idx  # rows in full matrix

    for alpha, maxiter in zip(alphas, iters):
        def fun(vf):
            U = vf.reshape(f, d)
            nr = np.sqrt(np.sum(U * U, axis=1, keepdims=True))
            nr = np.maximum(nr, 1e-12)
            Xf = U / nr
            X = np.empty((N, d), dtype=float)
            X[frozen_mask] = Xfrozen
            X[full_free_pos] = Xf
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            m = float(np.max(G))
            E = np.exp(alpha * (G - m))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E) * 0.5) + 1e-300
            L = m + np.log(S) / alpha
            GX = (E @ X) / S
            Gf = GX[full_free_pos]
            dot = np.sum(Gf * Xf, axis=1, keepdims=True)
            GU = (Gf - dot * Xf) / nr
            return float(L), GU.ravel().astype(float)
        try:
            r = minimize(fun, v0, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(maxiter), 'ftol': 1e-20,
                                  'gtol': 1e-7, 'maxfun': int(maxiter) * 6, 'maxcor': 8})
            v0 = r.x
        except Exception:
            break
    U = v0.reshape(f, d)
    nr = np.linalg.norm(U, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xf = U / nr
    X = X0.copy()
    X[free_idx] = Xf
    return _norm_rows(X)

def solve():
    t0 = time.time()
    HARD = 172.0
    CYCLE_BUDGET = 150.0
    rng = np.random.default_rng(1234567)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xcur = _norm_rows(cands[bi].copy())
    cur_cnt, cur_m = _count_close(Xcur, 1e-6)
    best_X = Xcur.copy()
    best_m = cur_m
    best_cnt = cur_cnt

    NCAND = 50000
    MAX_CYCLES = 20

    for cyc in range(MAX_CYCLES):
        if time.time() - t0 > CYCLE_BUDGET:
            break
        # identify argmax pair
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        ai = flat // 99
        aj = flat % 99
        # row maxima for preferential sampling
        rm = np.max(G, axis=1)
        mcur = float(np.max(G))
        # plateau set within 1e-5
        plateau = np.where(rm >= mcur - 1e-5)[0]
        if len(plateau) < 3:
            # widen to top 20 rows
            plateau = np.argsort(-rm)[:20]
        # number to remove: 3..7 cycling
        k_total = 3 + (cyc % 5)  # 3,4,5,6,7
        # one endpoint of argmax pair: alternate
        endpoint = ai if (cyc % 2 == 0) else aj
        remove_set = set([int(endpoint)])
        # sample 2..6 additional preferentially from plateau (excluding already)
        need = k_total - 1
        # build weights: closer to max => higher weight
        pool = [p for p in plateau.tolist() if p not in remove_set]
        if len(pool) < need:
            # fill with random others
            others = [i for i in range(99) if i not in remove_set and i not in pool]
            # random pick
            extra = rng.choice(others, size=min(need - len(pool), len(others)), replace=False).tolist() if others else []
            pool = pool + extra
        # weighted sampling without replacement: weight = 1/(mcur - rm + 1e-9)
        pool_arr = np.array(pool, dtype=int)
        w = 1.0 / np.maximum(mcur - rm[pool_arr] + 2e-7, 2e-7)
        w = w / np.sum(w)
        # sample need indices
        if len(pool_arr) >= need:
            # use choice with p
            chosen = rng.choice(pool_arr, size=need, replace=False, p=w).tolist()
        else:
            chosen = pool_arr.tolist()
        for c in chosen:
            remove_set.add(int(c))
        # if still short (duplicates), fill randomly
        while len(remove_set) < k_total:
            r_ = int(rng.integers(0, 99))
            remove_set.add(r_)
        remove_idx = sorted(list(remove_set))
        nk = len(remove_idx)
        keep_mask = np.ones(99, dtype=bool)
        keep_mask[remove_idx] = False
        Xfrozen = Xcur[keep_mask]  # (99-nk, 13)

        # generate random candidates
        Cand = _norm_rows(rng.standard_normal(size=(NCAND, 13)))
        # greedy maximin insertion
        # scores = max dot with frozen
        Df = Cand @ Xfrozen.T  # (NCAND, Nf)
        cur_min = np.max(Df, axis=1)  # lower is better (more separated)
        inserted = []
        # to avoid re-picking same candidate, mark used
        used = np.zeros(NCAND, dtype=bool)
        for s in range(nk):
            # argmin among unused
            # set used to large
            if np.any(used):
                tmp = cur_min.copy()
                tmp[used] = 2.0
                pick = int(np.argmin(tmp))
            else:
                pick = int(np.argmin(cur_min))
            used[pick] = True
            xnew = Cand[pick]
            inserted.append(xnew.copy())
            # update cur_min with separation from this new point
            dnew = Cand @ xnew  # (NCAND,)
            # max dot so far
            np.maximum(cur_min, dnew, out=cur_min)
            cur_min[pick] = 2.0  # don't pick again
        Xnew = Xcur.copy()
        Xnew[remove_idx] = np.stack(inserted, axis=0)
        Xnew = _norm_rows(Xnew)

        # determine neighbours within slack 5e-6 to inserted (pairs close to max)
        G2 = Xnew @ Xnew.T
        np.fill_diagonal(G2, -2.0)
        m2 = float(np.max(G2))
        thresh = m2 - 5e-6
        free_set = set([int(i) for i in remove_idx])
        # neighbours: any j with G2[i,j] > thresh for i in remove_idx
        for i in remove_idx:
            row = G2[i]
            nbrs = np.where(row > thresh)[0]
            for j in nbrs.tolist():
                free_set.add(int(j))
        # cap free set size to keep optimization cheap: if > 30, keep closest ones
        if len(free_set) > 30:
            # score each candidate free point by max dot to inserted set
            free_arr = np.array(sorted(free_set), dtype=int)
            # keep all inserted + top neighbours by proximity to inserted
            # compute max over inserted rows
            ins_arr = np.array(remove_idx, dtype=int)
            # for each free point, proximity = max G2[f, i] over i in inserted
            prox = np.max(G2[np.ix_(free_arr, ins_arr)], axis=1)
            order = np.argsort(-prox)  # largest first
            # always keep inserted
            keep_free = set([int(i) for i in remove_idx])
            for o in order:
                if len(keep_free) >= 30:
                    break
                keep_free.add(int(free_arr[o]))
            free_set = keep_free
        free_list = sorted(list(free_set))

        # optimize only inserted + neighbours with annealed LSE 100,500,2000
        # adapt iters to remaining time and free size
        remain = CYCLE_BUDGET - (time.time() - t0)
        if remain < 8:
            break
        # iters scaled: smaller for large free
        if len(free_list) <= 12:
            iters_sub = [60, 60, 80]
        elif len(free_list) <= 20:
            iters_sub = [40, 40, 60]
        else:
            iters_sub = [30, 30, 40]
        # if very late, shrink
        if remain < 30:
            iters_sub = [20, 20, 30]
        try:
            Xopt = _polish_subset(Xnew, free_list, [100.0, 500.0, 2000.0], iters_sub)
        except Exception:
            Xopt = Xnew
        # brief global polish: short full LSE at alpha 2000
        if time.time() - t0 < CYCLE_BUDGET - 5:
            try:
                # only if subset improved or time allows: 15-25 iters full
                full_iter = 25 if remain > 40 else 12
                Xopt = _polish_subset(Xopt, list(range(99)), [2000.0], [full_iter])
            except Exception:
                pass
        Xopt = _norm_rows(Xopt)
        new_cnt, new_m = _count_close(Xopt, 1e-6)

        # update global best
        if new_m < best_m - 1e-15:
            best_m = new_m
            best_X = Xopt.copy()
            best_cnt = new_cnt
        # acceptance for chain: improve score, or materially decrease count
        accept = False
        if new_m < cur_m - 1e-12:
            accept = True
        elif new_m <= cur_m + 5e-7 and new_cnt <= cur_cnt - 20:
            accept = True
        elif new_m <= cur_m + 1e-9 and new_cnt < cur_cnt:
            # tiny tolerance with any count improvement
            accept = True
        if accept:
            Xcur = Xopt.copy()
            cur_m = new_m
            cur_cnt = new_cnt
        else:
            # revert: keep Xcur, but if new is new global best we already saved it
            # occasional random shake to avoid stagnation? stay frozen per spec (revert)
            pass
        # early stop if time low
        if time.time() - t0 > HARD - 12:
            break

    # final safety: return best
    best_X = _norm_rows(best_X)
    # shrink by tiny margin not needed; ensure no NaN/zero
    return best_X
