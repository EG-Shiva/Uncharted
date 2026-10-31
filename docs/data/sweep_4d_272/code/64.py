import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    r = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (r[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _score_tuple(x):
    n = x.shape[0]
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    iu = np.triu_indices(n, 1)
    c = g[iu]
    cnt = int((c > m - 1e-6).sum())
    if c.size >= 100:
        part = np.partition(c, c.size - 100)
        s100 = float(part[c.size - 100:].sum())
    else:
        s100 = float(c.sum())
    return (m, cnt, s100)

def _get_duals(res, p):
    try:
        for key in ("ineqlin", "inequality", "ineq"):
            obj = None
            try:
                if hasattr(res, key):
                    obj = getattr(res, key)
                elif isinstance(res, dict) and key in res:
                    obj = res[key]
                elif hasattr(res, "get"):
                    try:
                        obj = res.get(key, None)
                    except Exception:
                        obj = None
            except Exception:
                obj = None
            if obj is None:
                continue
            m = None
            if isinstance(obj, dict):
                for k in ("marginals", "dual", "duals", "pi"):
                    if k in obj:
                        m = obj[k]
                        break
            else:
                for k in ("marginals", "dual", "duals", "pi"):
                    if hasattr(obj, k):
                        try:
                            m = getattr(obj, k)
                            break
                        except Exception:
                            continue
                if m is None:
                    try:
                        m = np.asarray(obj, dtype=float).ravel()
                    except Exception:
                        m = None
            if m is None:
                continue
            m = np.asarray(m, dtype=float).ravel()
            if m.size == p and np.all(np.isfinite(m)):
                return m
    except Exception:
        pass
    return None

def _build_solve(x, radius, I, J, c_sel, deltas, t0deadline=None):
    n, d = x.shape
    p = len(I)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    b_ub = -c_sel - deltas
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None
    return res

def _lp_polish(x, seconds, deadline, exclude_set=None, exclude_iters=2):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    it = 0
    start = time.time()
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        if exclude_set is not None and it < exclude_iters:
            # mask out excluded pairs
            Ii = iu[0][sel]; Ji = iu[1][sel]
            keep = np.ones(len(Ii), dtype=bool)
            for k in range(len(Ii)):
                a = int(Ii[k]); b = int(Ji[k])
                if a > b: a, b = b, a
                if (a, b) in exclude_set:
                    keep[k] = False
            if keep.sum() == 0:
                it += 1
                radius *= 0.7
                continue
            I = Ii[keep]; J = Ji[keep]
            cs = c[sel][keep]
        else:
            I = iu[0][sel]; J = iu[1][sel]
            cs = c[sel]
        p = len(I)
        if p == 0:
            break
        res = _build_solve(x, radius, I, J, cs, np.zeros(p))
        if res is None or res.status != 0:
            radius *= 0.5
            it += 1
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        it += 1
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal((272, 4))
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (272, 4):
        rng0 = np.random.default_rng(0)
        xb = _norm(rng0.standard_normal((272, 4)))
    n, d = 272, 4
    iu = np.triu_indices(n, 1)
    best_x = xb.copy()
    best_tup = _score_tuple(best_x)
    best_val = best_tup[0]
    # brief initial polish to reach local optimum
    if HARD_END - time.time() > 25:
        best_x, bv = _lp_polish(best_x, seconds=8.0, deadline=HARD_END - 5)
        best_tup = _score_tuple(best_x)
        best_val = best_tup[0]
    cur_x = best_x.copy()
    cur_tup = best_tup
    deltas_list = [1e-4, 3e-4, 1e-3, 3e-3]
    alphas = [1.0, 2.0, 4.0, 8.0, 16.0]
    rnd = 0
    max_rounds = 12
    for rnd in range(max_rounds):
        if time.time() > HARD_END - 22:
            break
        # 1) final trust-region LP at cur to get duals
        radius0 = 1e-3
        G = cur_x @ cur_x.T
        cfull = G[iu]
        mcur = float(cfull.max())
        selm = cfull > mcur - max(20.0 * radius0, 1e-9)
        I0 = iu[0][selm]; J0 = iu[1][selm]
        c0 = cfull[selm]
        p0 = len(I0)
        if p0 == 0:
            break
        res0 = _build_solve(cur_x, radius0, I0, J0, c0, np.zeros(p0))
        if res0 is None or res0.status != 0:
            # retry smaller radius
            radius0 = 5e-4
            selm = cfull > mcur - max(20.0 * radius0, 1e-9)
            I0 = iu[0][selm]; J0 = iu[1][selm]
            c0 = cfull[selm]
            res0 = _build_solve(cur_x, radius0, I0, J0, c0, np.zeros(len(I0)))
            if res0 is None or res0.status != 0:
                # fallback: plain polish and continue
                cur_x, _ = _lp_polish(cur_x, seconds=4.0, deadline=HARD_END - 5)
                cur_tup = _score_tuple(cur_x)
                continue
        duals = _get_duals(res0, len(I0))
        if duals is None:
            stress = c0.copy()  # tighter = higher stress
        else:
            stress = np.abs(np.asarray(duals, dtype=float).ravel())
            # if all zero, fallback
            if np.max(stress) <= 0 or not np.all(np.isfinite(stress)):
                stress = c0.copy()
        order = np.argsort(-stress)
        # target K cycles 24,28,32,36
        K = [24, 28, 32, 36][rnd % 4]
        # diversity: limit per-point occurrences
        per_lim = 5
        chosen_idx = []
        cntp = np.zeros(n, dtype=int)
        for oi in order:
            if len(chosen_idx) >= K:
                break
            a = int(I0[oi]); b = int(J0[oi])
            if cntp[a] >= per_lim or cntp[b] >= per_lim:
                continue
            chosen_idx.append(int(oi))
            cntp[a] += 1; cntp[b] += 1
        # if not enough, relax limit
        if len(chosen_idx) < 20:
            chosen_idx = []
            cntp[:] = 0
            per_lim = 8
            for oi in order:
                if len(chosen_idx) >= K:
                    break
                a = int(I0[oi]); b = int(J0[oi])
                if cntp[a] >= per_lim or cntp[b] >= per_lim:
                    continue
                chosen_idx.append(int(oi))
                cntp[a] += 1; cntp[b] += 1
        if len(chosen_idx) < 10:
            chosen_idx = [int(o) for o in order[:K]]
        chosen_set = set()
        is_forced = np.zeros(len(I0), dtype=bool)
        for ci in chosen_idx:
            a = int(I0[ci]); b = int(J0[ci])
            if a > b: a, b = b, a
            chosen_set.add((a, b))
            is_forced[ci] = True
        # also need hub-diversity check: ensure not all share one hub
        # (already enforced by per_lim); if violated, skip round
        # 2) for each delta, modified LP with larger radius + linesearch
        radius_mod = 3e-3
        # recompute selection under radius_mod threshold for stability
        selm2 = cfull > mcur - max(20.0 * radius_mod, 1e-9)
        I2 = iu[0][selm2]; J2 = iu[1][selm2]
        c2 = cfull[selm2]
        # map chosen_set onto I2
        is_forced2_base = np.zeros(len(I2), dtype=bool)
        pair_to_pos = {}
        for k in range(len(I2)):
            a = int(I2[k]); b = int(J2[k])
            if a > b: a, b = b, a
            pair_to_pos[(a, b)] = k
        for pr in chosen_set:
            if pr in pair_to_pos:
                is_forced2_base[pair_to_pos[pr]] = True
        # if too few overlap (threshold changed), keep at least those overlapping
        best_round_tup = None
        best_round_x = None
        for di, delta in enumerate(deltas_list):
            if time.time() > HARD_END - 18:
                break
            deltas = np.where(is_forced2_base, delta, 0.0)
            resm = _build_solve(cur_x, radius_mod, I2, J2, c2, deltas)
            if resm is None or resm.status != 0:
                continue
            dx = resm.x[:-1].reshape(n, d)
            if not np.all(np.isfinite(dx)):
                continue
            dn = np.linalg.norm(dx)
            if dn < 1e-14 or not np.isfinite(dn):
                continue
            # line search multiples
            cand_best = None
            cand_best_t = None
            for al in alphas:
                y = _norm(cur_x + al * dx)
                tt = _score_tuple(y)
                if cand_best_t is None or (tt[0], tt[1], tt[2]) < (cand_best_t[0], cand_best_t[1], cand_best_t[2]):
                    cand_best_t = tt
                    cand_best = y
            if cand_best is None:
                continue
            # re-polish with selected contacts initially excluded
            time_left = HARD_END - time.time()
            if time_left < 8:
                yp, vp = cand_best, cand_best_t[0]
            else:
                # budget per delta polish
                per = min(6.0, max(2.5, (time_left - 10) / (len(deltas_list) - di)))
                yp, vp = _lp_polish(cand_best, seconds=per, deadline=HARD_END - 3,
                                    exclude_set=chosen_set, exclude_iters=3)
            tp = _score_tuple(yp)
            if best_round_tup is None or (tp[0], tp[1], tp[2]) < (best_round_tup[0], best_round_tup[1], best_round_tup[2]):
                best_round_tup = tp
                best_round_x = yp.copy()
        if best_round_x is None or best_round_tup is None:
            # no move; try to diversify by small deterministic shake of a hub? do tiny polish
            continue
        # lexicographic acceptance for cur (allow equal max with better secondary)
        # tolerance on max: 2e-12
        cur_m, cur_c, cur_s = cur_tup
        br_m, br_c, br_s = best_round_tup
        improved_cur = False
        if br_m < cur_m - 1e-12:
            improved_cur = True
        elif abs(br_m - cur_m) <= 1e-12 and (br_c < cur_c or (br_c == cur_c and br_s < cur_s - 1e-12)):
            improved_cur = True
        # also accept tiny worsening within 5e-13 if secondary improves a lot? retain non-worsening: require no worsen
        # we keep exploring even without improvement by moving to best round if it is not worse by >1e-9? To change basin.
        # Decide: if not improved, but round is within 2e-9 and secondary better, still move to force topology walk.
        move = improved_cur
        if not move:
            if br_m <= cur_m + 1e-9 and (br_c < cur_c - 5 or (br_c <= cur_c and br_s < cur_s)):
                move = True
        if move:
            cur_x = best_round_x.copy()
            cur_tup = best_round_tup
        # update global best on max strictly
        if best_round_tup[0] < best_val - 1e-13:
            best_val = float(best_round_tup[0])
            best_x = best_round_x.copy()
            best_tup = best_round_tup
            cur_x = best_round_x.copy()
            cur_tup = best_round_tup
        elif best_round_tup[0] <= best_val + 1e-12 and best_round_tup[1] < best_tup[1]:
            # keep best secondary for tracking but don't overwrite best_val
            pass
        # early stop if clearly new record? continue to squeeze
    # final polish of best with remaining time
    if time.time() < HARD_END - 4:
        sec = max(2.0, min(12.0, HARD_END - time.time() - 1.0))
        try:
            xp, vp = _lp_polish(best_x, seconds=sec, deadline=HARD_END - 0.5)
            tp = _score_tuple(xp)
            if tp[0] < best_val:
                best_x = xp
                best_val = tp[0]
        except Exception:
            pass
    # also consider cur if it is better max (numerical)
    try:
        if cur_tup[0] < best_val:
            best_x = cur_x.copy()
    except Exception:
        pass
    out = _norm(best_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    return out
