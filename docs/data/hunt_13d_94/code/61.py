import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_slacks(X, iu0, iu1):
    G = X @ X.T
    vals = G[iu0, iu1]
    m = float(vals.max())
    slacks = m - vals
    return m, vals, slacks

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d-1))
    E = np.eye(d)
    for i in range(n):
        M = np.column_stack([X[i], E])  # 13x14
        Q, _ = np.linalg.qr(M)
        T[i] = Q[:, 1:]
    return T

def solve():
    T0 = time.time()
    BUDGET = 165.0
    END = T0 + BUDGET
    # load incumbent
    rec = None; bst = None
    for fn, tag in [("record.npy","r"), ("best.npy","b")]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-30
            if a.shape == (94,13) and np.all(np.isfinite(a)):
                if tag=="r": rec = a.copy()
                else: bst = a.copy()
        except Exception:
            pass
    rng = np.random.default_rng(0)
    if rec is None and bst is None:
        r = rng.standard_normal(size=(94,13))
        r /= np.linalg.norm(r, axis=1, keepdims=True)+1e-30
        rec = r.copy(); bst = r.copy()
    if rec is None: rec = bst.copy()
    if bst is None: bst = rec.copy()
    n, d = 94, 13
    iu0, iu1 = np.triu_indices(n, 1)
    def maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    rv = maxcos(rec); bv = maxcos(bst)
    if bv < rv:
        Xcur = bst.copy(); cur_m = bv
    else:
        Xcur = rec.copy(); cur_m = rv
    Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)+1e-30
    cur_m = maxcos(Xcur)
    best_X = Xcur.copy(); best_m = cur_m
    # tie count at 1e-10
    G0 = Xcur @ Xcur.T
    v0 = G0[iu0, iu1]
    tie_best = int(np.sum((cur_m - v0) < 1e-10))
    tie_cur = tie_best

    thresholds = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    th_idx = 0  # start wide 1e-6
    radius = 3e-4
    NV = n*(d-1)+1
    TDT = d-1

    it = 0
    fails = 0
    while time.time() < END - 3.0:
        it += 1
        if it > 800:
            break
        thresh = thresholds[th_idx]
        # compute slacks for current X
        m, vals, slacks = _max_cos_slacks(Xcur, iu0, iu1)
        cur_m = m
        # active set: slack < thresh, capped to tightest 2000, at least tightest 200
        order = np.argsort(slacks)  # ascending = tightest first
        # count under thresh
        n_under = int(np.searchsorted(slacks[order], thresh, side='right'))
        # also enforce 3e-7 cap rule initially: but unify
        # cap at 2000, floor at 200 (or at least 50)
        if n_under < 200:
            n_act = min(200, len(order))
            # also if thresh is tiny, still take 200 to have direction
        elif n_under > 2000:
            n_act = 2000
        else:
            n_act = n_under
        # the active threshold actually used (for diagnostics)
        sel_idx = order[:n_act]
        # if smallest slack is large (should not), continue
        I = iu0[sel_idx]; J = iu1[sel_idx]
        c_act = vals[sel_idx]
        p = n_act
        if p < 5:
            # nothing to constrain, jitter slightly to escape?
            radius = min(radius*1.5, 3e-4)
            if th_idx > 0:
                th_idx -= 1  # widen
            continue
        # tangent bases
        try:
            Tb = _tangent_bases(Xcur)
        except Exception:
            break
        # build sparse A_ub: p x NV, coeffs
        # coeff_i = Tb[i].T @ Xcur[J_k]  (12), coeff_j = Tb[j].T @ Xcur[I_k]
        # vectorize
        try:
            Ci = np.einsum('kdr,kr->kd', Tb[I], Xcur[J])  # p x 12
            Cj = np.einsum('kdr,kr->kd', Tb[J], Xcur[I])
        except Exception:
            radius *= 0.5
            continue
        # coo arrays
        nnz_per_row = 2*TDT+1
        rows = np.repeat(np.arange(p), nnz_per_row)
        cols = np.empty(p*nnz_per_row, dtype=np.int64)
        valsA = np.empty(p*nnz_per_row, dtype=float)
        # fill: use loops over d-1 blocks via broadcasting (p small so loop ok)
        # build with python loop for clarity but vectorized assignment
        # i-block cols
        base_i = I*TDT
        base_j = J*TDT
        # create col indices
        # for k in range(p): cols[k*25 : ...]
        # vectorized:
        off = np.arange(TDT)
        cols_i = (base_i[:,None] + off[None,:]).ravel()
        cols_j = (base_j[:,None] + off[None,:]).ravel()
        # interleave into cols array: [Ci(12), Cj(12), t]
        # rows already repeated; need to place
        cols = np.empty(p*nnz_per_row, dtype=np.int64)
        valsA = np.empty(p*nnz_per_row, dtype=float)
        cols[0::nnz_per_row] = np.nan  # placeholder not used
        # simpler: build via lists with numpy stacking
        # redo cleanly:
        row_idx = np.repeat(np.arange(p), nnz_per_row)
        col_idx = np.concatenate([
            (base_i[:,None]+off).ravel() if False else np.empty(0)
        ]) if False else None
        # fallback explicit construction using tile
        col_blocks_i = base_i[:,None] + off[None,:]  # p x 12
        col_blocks_j = base_j[:,None] + off[None,:]
        col_idx_arr = np.empty((p, nnz_per_row), dtype=np.int64)
        col_idx_arr[:, :TDT] = col_blocks_i
        col_idx_arr[:, TDT:2*TDT] = col_blocks_j
        col_idx_arr[:, 2*TDT] = NV-1
        col_idx_flat = col_idx_arr.ravel()
        val_arr = np.empty((p, nnz_per_row), dtype=float)
        val_arr[:, :TDT] = Ci
        val_arr[:, TDT:2*TDT] = Cj
        val_arr[:, 2*TDT] = -1.0
        val_flat = val_arr.ravel()
        b_ub = -c_act  # constraint: Ci.yi + Cj.yj - t <= -c
        try:
            A_ub = coo_matrix((val_flat, (row_idx, col_idx_flat)), shape=(p, NV)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(NV); obj[-1] = 1.0
        bounds = [(-radius, radius)]*(NV-1) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            fails += 1
            if radius < 1e-12:
                radius = 1e-4
                if th_idx > 0: th_idx -= 1
            continue
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            radius *= 0.5
            fails += 1
            if radius < 1e-13:
                # tighten or reset
                if th_idx < len(thresholds)-1:
                    th_idx += 1
                    radius = 1e-4
                else:
                    radius = 3e-4
                    th_idx = 0
            continue
        y = res.x[:NV-1].reshape(n, TDT)
        t_pred = float(res.x[-1])
        # if predicted no improvement, shrink / tighten
        if not np.isfinite(t_pred) or t_pred >= m - 1e-12:
            # no descent direction at this radius/thresh
            radius *= 0.5
            if radius < 2e-7:
                if th_idx < len(thresholds)-1:
                    # tightening may reveal structure; check new ties
                    old_count = int(np.sum(slacks < thresholds[min(th_idx+1, len(thresholds)-1)]))
                    # move to tighter
                    th_idx += 1
                    new_m, new_vals, new_sl = _max_cos_slacks(Xcur, iu0, iu1)
                    new_count = int(np.sum(new_sl < thresholds[th_idx]))
                    # restart radius if many ties (spec: >=50 new)
                    if new_count >= old_count + 50 or True:
                        radius = 1e-4
                    else:
                        radius = 3e-4
                else:
                    radius = 3e-4
                    # cycle wide again to keep moving
                    th_idx = max(0, th_idx-1)
            continue
        # build dx
        dx = np.einsum('ndr,nd->nr', Tb, y)  # n x 13 tangent
        # backtracking over alpha
        accepted = False
        best_alpha_val = None
        best_alpha_X = None
        best_alpha_tie = None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            Y = Xcur + alpha*dx
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[~np.isfinite(nr)] = 1.0
            nr[nr==0] = 1.0
            Y /= nr
            if not np.all(np.isfinite(Y)):
                continue
            mv = maxcos(Y)
            # update global best ever regardless of tie rule
            if mv < best_m - 1e-15:
                best_m = mv; best_X = Y.copy()
                # update tie_best to this config's tie count
                Gt = Y @ Y.T
                vv = Gt[iu0, iu1]
                tie_best = int(np.sum((mv - vv) < 1e-10))
            if mv < m - 1e-12:
                # tie guard: count slack<1e-10
                Gt2 = Y @ Y.T
                vv2 = Gt2[iu0, iu1]
                tie_new = int(np.sum((mv - vv2) < 1e-10))
                # accept if tie not reduced
                if tie_new >= tie_cur:
                    best_alpha_val = mv; best_alpha_X = Y; best_alpha_tie = tie_new
                    accepted = True
                    break
                else:
                    # remember first improving but tie-reducing as fallback
                    if best_alpha_val is None:
                        best_alpha_val = mv; best_alpha_X = Y; best_alpha_tie = tie_new
                        # do not accept yet, try smaller alpha which may preserve ties
                        continue
        if accepted:
            # accept with tie preservation
            Xcur = best_alpha_X; cur_m = best_alpha_val
            old_tie = tie_cur; tie_cur = best_alpha_tie
            fails = 0
            # tighten threshold geometrically
            if th_idx < len(thresholds)-1:
                prev_thr = thresholds[th_idx]
                th_idx += 1
                # count new near-ties at new threshold
                _, vv_n, sl_n = _max_cos_slacks(Xcur, iu0, iu1)
                n_new = int(np.sum(sl_n < thresholds[th_idx]))
                _, _, sl_o = _max_cos_slacks(best_alpha_X, iu0, iu1)  # same
                # heuristic: if many ties emerged, restart radius at 1e-4
                # estimate new ties vs old: compare count <1e-7 before/after?
                # use simple: if number of pairs with slack<1e-7 increased by >=50
                cnt7 = int(np.sum(sl_n < 1e-7))
                # compare to count before step? approximate with p? use 600 baseline
                # restart if tight set is large
                if n_new >= 50 or cnt7 >= 200:
                    radius = 1e-4
                else:
                    radius = min(radius*1.3, 3e-4)
            else:
                radius = min(radius*1.3, 3e-4)
            # also successful step: keep radius moderate
            radius = min(max(radius, 5e-5), 3e-4)
            continue
        else:
            # no tie-preserving improvement; if there was a tie-reducing improvement, we already saved global best, shrink
            radius *= 0.5
            fails += 1
            if radius < 1e-12:
                radius = 1e-4
                # loosen slightly to find new basin direction
                if th_idx > 0 and fails > 3:
                    th_idx -= 1
                    fails = 0
            continue
    # final safety: return best ever
    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)+1e-30
    return best_X
