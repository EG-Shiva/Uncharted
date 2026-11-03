import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

RECORD = 0.461182770249

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lp_polish_deadline(x, deadline, r_init=0.005, max_pairs=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        if radius > 0.01:
            k = max_pairs
            if len(c_all) > k:
                part = np.argpartition(c_all, -k)[-k:]
                sel_mask = np.zeros(len(c_all), dtype=bool)
                sel_mask[part] = True
                sel_mask = sel_mask & (c_all > thresh)
                if int(sel_mask.sum()) < 200:
                    sel_mask = np.zeros(len(c_all), dtype=bool)
                    sel_mask[part] = True
                sel = sel_mask
            else:
                sel = c_all > thresh
        else:
            sel = c_all > thresh
            if int(sel.sum()) > max_pairs:
                part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
                sel = np.zeros(len(c_all), dtype=bool)
                sel[part] = True
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2*d] = J[:, None] * d + np.arange(d)
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def build_pool_reps():
    reps = []
    # 144 vectors (±1,±1,0^7)/sqrt2 -> 72 reps
    s2 = 1.0 / np.sqrt(2.0)
    for i, j in itertools.combinations(range(9), 2):
        for a, b in [(1.0, 1.0), (1.0, -1.0)]:
            v = np.zeros(9)
            v[i] = a * s2
            v[j] = b * s2
            reps.append(v)
    # 18 vectors ±e_i -> 9 reps
    for i in range(9):
        v = np.zeros(9); v[i] = 1.0
        reps.append(v)
    # 512 sign vectors (±1^9)/3 -> 256 reps with first coord +1
    for mask in range(512):
        bits = np.array([(1.0 if (mask >> k) & 1 else -1.0) for k in range(9)], dtype=float)
        # k=0 corresponds to coord 0? enforce coord0=+1
        if bits[0] < 0:
            continue
        reps.append(bits / 3.0)
    # permutations/signs of (±1,±1,±1,±1,0^5)/2 -> 1008 reps
    for combo in itertools.combinations(range(9), 4):
        for mask in range(16):
            signs = np.array([(1.0 if (mask >> k) & 1 else -1.0) for k in range(4)], dtype=float)
            if signs[0] < 0:
                continue
            v = np.zeros(9)
            for k, pos in enumerate(combo):
                v[pos] = signs[k] * 0.5
            reps.append(v)
    P = np.array(reps, dtype=float)
    # normalize (should already be unit) and deduplicate
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    # dedup rounding
    seen = {}
    keep = []
    for idx, v in enumerate(P):
        key = tuple(np.round(v, 10))
        if key not in seen:
            seen[key] = True
            keep.append(idx)
    P = P[keep]
    return P

def greedy_select(P, need=88):
    N = P.shape[0]
    Gabs = np.abs(P @ P.T)
    # exp table
    selected = [0]
    sel_mask = np.zeros(N, dtype=bool)
    sel_mask[0] = True
    old_max = 0.0
    old_sum = 0.0
    # precompute exp Gabs on the fly per step
    for step in range(1, need):
        unsel = np.where(~sel_mask)[0]
        sel_idx = np.array(selected, dtype=int)
        # U x S matrix
        M = Gabs[np.ix_(unsel, sel_idx)]
        max_c = M.max(axis=1)
        sum_e = np.exp(50.0 * M).sum(axis=1)
        new_max = np.maximum(old_max, max_c)
        new_sum = old_sum + sum_e
        obj = new_max + 1e-3 * new_sum
        pick_pos = int(np.argmin(obj))
        pick = int(unsel[pick_pos])
        # update old
        old_max = float(new_max[pick_pos])
        old_sum = float(new_sum[pick_pos])
        selected.append(pick)
        sel_mask[pick] = True
    return np.array(selected, dtype=int), Gabs

def rep_metrics(sel, Gabs):
    # sel: array of indices into pool
    S = Gabs[np.ix_(sel, sel)]
    iu = np.triu_indices(len(sel), 1)
    vals = S[iu]
    M = float(vals.max())
    C = int(np.sum(vals > M - 1e-4))
    # cap exp to avoid inf: vals<=1 so exp(50) ~5e21 fits
    Sm = float(np.exp(50.0 * vals).sum())
    return M, C, Sm, vals

def pair_swap_search(sel_init, Gabs, deadline, max_sweeps=20):
    sel = sel_init.copy()
    N = Gabs.shape[0]
    curM, curC, curS, _ = rep_metrics(sel, Gabs)
    sel_set = set(sel.tolist())
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        # find worst-involved indices
        Smat = Gabs[np.ix_(sel, sel)]
        iu = np.triu_indices(len(sel), 1)
        vals = Smat[iu]
        M = float(vals.max())
        # involved positions in sel array
        mask_worst = vals > M - 1e-4
        wi = iu[0][mask_worst]
        wj = iu[1][mask_worst]
        involved = np.unique(np.concatenate([wi, wj]))
        # order involved by participation count (descending) deterministically
        # count
        cnt = {}
        for w in np.concatenate([wi, wj]):
            cnt[int(w)] = cnt.get(int(w), 0) + 1
        involved = sorted(involved.tolist(), key=lambda w: (-cnt.get(int(w), 0), int(w)))
        # cap to 30 to save time
        if len(involved) > 30:
            involved = involved[:30]
        sel_mask = np.zeros(N, dtype=bool)
        sel_mask[sel] = True
        unsel = np.where(~sel_mask)[0]
        if len(unsel) == 0:
            break
        best_tuple = (curM, curC, curS)
        best_swap = None  # (pos_in_sel, new_pool_idx, newM,newC,newS)
        # precompute ExpGabs column for unsel? compute per w
        for pos in involved:
            fixed = np.delete(sel, pos)
            # fixed metrics
            Sf = Gabs[np.ix_(fixed, fixed)]
            iuf = np.triu_indices(len(fixed), 1)
            vf = Sf[iuf] if len(fixed) > 1 else np.array([0.0])
            Mf = float(vf.max()) if vf.size > 0 else 0.0
            Sf_sum = float(np.exp(50.0 * vf).sum()) if vf.size > 0 else 0.0
            vsorted = np.sort(vf) if vf.size > 0 else np.array([0.0])
            # connections of all unsel to fixed: U x F
            A = Gabs[np.ix_(unsel, fixed)]
            max_conn = A.max(axis=1)
            sum_conn = np.exp(50.0 * A).sum(axis=1)
            newM_arr = np.maximum(Mf, max_conn)
            newS_arr = Sf_sum + sum_conn
            # need count: for each u, count = #{vf > newM-1e-4} + #{A[u] > newM-1e-4}
            # use binary search for fixed part
            # vectorize: for each u threshold
            # To avoid per-u python loop over 1257, loop in python but with numpy ops for A part vectorized + searchsorted vectorized
            thr = newM_arr - 1e-4
            # counts fixed via searchsorted
            idx = np.searchsorted(vsorted, thr)
            cnt_fixed = len(vsorted) - idx
            # counts conn: (A > thr[:,None]).sum(axis=1)
            cnt_conn = (A > thr[:, None]).sum(axis=1)
            cnt_tot = cnt_fixed + cnt_conn
            # find best u lexicographically
            # first minimize newM, then cnt, then newS
            # get order by (newM, cnt, newS)
            order = np.lexsort((newS_arr, cnt_tot, newM_arr))
            b = int(order[0])
            tup = (float(newM_arr[b]), int(cnt_tot[b]), float(newS_arr[b]))
            if tup < best_tuple:
                best_tuple = tup
                best_swap = (int(pos), int(unsel[b]), tup)
        if best_swap is not None and best_tuple < (curM, curC, curS):
            pos, new_idx, tup = best_swap
            sel[pos] = new_idx
            curM, curC, curS = tup
        else:
            break
    return sel

def solve():
    t0 = time.time()
    HARD_END = t0 + 176.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    # load fallback best
    Xfallback = None
    vfallback = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    v = max_cos(a)
                    if v < vfallback:
                        vfallback = v
                        Xfallback = a.copy()
        except Exception:
            continue
    # ---- dictionary construction ----
    try:
        P = build_pool_reps()
    except Exception:
        P = None
    Xdict = None
    vdict = np.inf
    if P is not None:
        try:
            # greedy
            sel_greedy, Gabs = greedy_select(P, need=88)
            # pair-swap with limited time
            swap_deadline = min(t0 + 30.0, HARD_END - 80.0)
            if time.time() < swap_deadline:
                try:
                    sel = pair_swap_search(sel_greedy, Gabs, swap_deadline, max_sweeps=20)
                except Exception:
                    sel = sel_greedy
            else:
                sel = sel_greedy
            Pr_sel = P[sel]
            Xd = np.vstack([Pr_sel, -Pr_sel])
            Xd = Xd / np.linalg.norm(Xd, axis=1, keepdims=True)
            # polish at most 60s
            polish_deadline = min(time.time() + 60.0, HARD_END - 25.0)
            if time.time() < polish_deadline:
                Xd = lp_polish_deadline(Xd, polish_deadline, r_init=0.005, max_pairs=2500)
            vdict = max_cos(Xd)
            Xdict = Xd
            # symmetry-broken versions if within 0.01 of record
            if vdict < RECORD + 0.01:
                bestX = Xdict
                bestV = vdict
                # try up to 2 perturbations while time allows
                for trial in range(2):
                    if time.time() > HARD_END - 22.0:
                        break
                    try:
                        noise = rng.standard_normal(bestX.shape)
                        # tangent projection
                        proj = (noise * bestX).sum(axis=1, keepdims=True)
                        noise = noise - proj * bestX
                        Xt = bestX + 0.005 * noise
                        Xt = Xt / np.linalg.norm(Xt, axis=1, keepdims=True)
                        dl = min(time.time() + 12.0, HARD_END - 20.0)
                        if dl <= time.time() + 0.3:
                            break
                        Xt = lp_polish_deadline(Xt, dl, r_init=0.004, max_pairs=2500)
                        vt = max_cos(Xt)
                        if vt < bestV:
                            bestV = vt
                            bestX = Xt
                    except Exception:
                        continue
                Xdict = bestX
                vdict = bestV
        except Exception:
            pass
    # ---- choose best between dict and fallback ----
    bestX = None
    bestV = np.inf
    if Xdict is not None and np.all(np.isfinite(Xdict)):
        try:
            vv = max_cos(Xdict / np.linalg.norm(Xdict, axis=1, keepdims=True))
            if vv < bestV:
                bestV = vv
                bestX = Xdict / np.linalg.norm(Xdict, axis=1, keepdims=True)
        except Exception:
            pass
    if Xfallback is not None:
        if vfallback < bestV:
            bestV = vfallback
            bestX = Xfallback
    if bestX is None:
        # random fallback
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    # final short polish of overall best if time remains
    try:
        if time.time() < HARD_END - 2.0:
            dl = HARD_END - 1.0
            # only polish if we have >3s
            if dl - time.time() > 3.0:
                # cap final polish to 15s to stay safe
                dl2 = min(dl, time.time() + 15.0)
                bestX = lp_polish_deadline(bestX, dl2, r_init=0.0012, max_pairs=2500)
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
