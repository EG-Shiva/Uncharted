import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _slacks(X):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    m = float(c.max())
    slack = m - c
    return m, slack, iu0, iu1, c

def _tie_stats(slack, thr=1e-10):
    cnt = int(np.count_nonzero(slack < thr))
    return cnt

def _contacts_per_point(n, iu0, iu1, slack, thr=1e-10):
    cp = np.zeros(n, dtype=int)
    m = np.where(slack < thr)[0]
    if len(m):
        a = iu0[m]; b = iu1[m]
        np.add.at(cp, a, 1)
        np.add.at(cp, b, 1)
    return cp, int(cp.min()) if n else 0

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    T = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        xi = X[i]
        kc = int(np.argmax(np.abs(xi)))
        den = float(xi[kc])
        if abs(den) < 1e-12:
            den = 1e-12 if den >= 0 else -1e-12
        M = np.zeros((d, k), dtype=np.float64)
        col = 0
        for j in range(d):
            if j == kc:
                continue
            M[j, col] = 1.0
            M[kc, col] = -float(xi[j]) / den
            col += 1
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _renorm(X):
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    nr[~np.isfinite(nr)] = 1.0
    nr[nr < 1e-12] = 1.0
    return X / nr

def _lp_step(X, active_I, active_J, active_c, radius, time_left):
    n, d = X.shape
    k = d - 1
    Nv = n * k
    q = len(active_I)
    if q == 0:
        return None
    T = _tangent_bases(X)
    Xi = X[active_I]
    Xj = X[active_J]
    Ti = T[active_I]
    Tj = T[active_J]
    Ai = np.einsum('qdk,qd->qk', Ti, Xj)
    Aj = np.einsum('qdk,qd->qk', Tj, Xi)
    # sparse A_ub shape (q, Nv+1)
    rows = np.repeat(np.arange(q), 2 * k + 1)
    col_blocks_i = (active_I[:, None] * k + np.arange(k)).ravel()
    col_blocks_j = (active_J[:, None] * k + np.arange(k)).ravel()
    cols_i = np.repeat(col_blocks_i.reshape(q, k), 1, axis=0) if False else None
    # build coo arrays
    rpt = np.repeat(np.arange(q), k)
    c_i = (active_I.repeat(k) * k + np.tile(np.arange(k), q))
    c_j = (active_J.repeat(k) * k + np.tile(np.arange(k), q))
    r_all = np.concatenate([rpt, rpt, np.arange(q)])
    c_all = np.concatenate([c_i, c_j, np.full(q, Nv)])
    v_all = np.concatenate([Ai.ravel(), Aj.ravel(), -np.ones(q)])
    try:
        A_ub = csr_matrix((v_all, (r_all, c_all)), shape=(q, Nv + 1))
    except Exception:
        return None
    b_ub = -active_c
    c_obj = np.zeros(Nv + 1)
    c_obj[-1] = 1.0
    bounds = [(-radius, radius)] * Nv + [(None, None)]
    try:
        tl = max(1.0, min(5.0, time_left))
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'time_limit': tl})
    except Exception:
        return None
    if res.status != 0 or res.x is None:
        return None
    x = np.asarray(res.x, dtype=float)
    if not np.all(np.isfinite(x)):
        return None
    vflat = x[:-1]
    vflat = np.clip(vflat, -radius, radius)
    # reconstruct dX
    V = vflat.reshape(n, k)
    dX = np.einsum('ndk,nk->nd', T, V)
    return dX

def _continuation(X0, deadline, rng, start_radius=3e-4, label=""):
    n, d = X0.shape
    curX = _renorm(X0.copy())
    cur_m, cur_slack, iu0, iu1, cur_c = _slacks(curX)
    cur_tie = _tie_stats(cur_slack, 1e-10)
    _, cur_min_c = _contacts_per_point(n, iu0, iu1, cur_slack, 1e-10)
    bestX = curX.copy(); best_val = cur_m; best_tie = cur_tie
    # thresholds geometric tightening
    thresholds = [1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9, 3e-10, 1e-10]
    th_idx = 1  # start at 3e-7 per brief
    radius = start_radius
    fails = 0
    it = 0
    alphas = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    while time.time() < deadline - 0.5:
        it += 1
        if radius < 1e-12:
            radius = 1e-4
            if th_idx < len(thresholds) - 1:
                th_idx += 1
            else:
                break
        th = thresholds[th_idx]
        m, slack, _, _, cfull = _slacks(curX)
        # active set slack < th capped at tightest 2000
        # ensure at least 300 for stability
        order_all = np.argpartition(slack, min(2000, len(slack)) - 1)[:min(2000, len(slack))]
        # filter by th within those? Actually take tightest 2000 then filter? Brief: slack<3e-7 capped at 2000.
        # So: mask slack<th, if >2000 keep tightest 2000.
        mask_idx = np.where(slack < th)[0]
        if len(mask_idx) > 2000:
            part = np.argpartition(slack, 2000 - 1)[:2000]
            # ensure max included
            order = part[np.argsort(slack[part])]
        elif len(mask_idx) < 300:
            # take 300 tightest to keep LP informed
            kk = min(300, len(slack))
            part = np.argpartition(slack, kk - 1)[:kk]
            order = part[np.argsort(slack[part])]
        else:
            order = mask_idx[np.argsort(slack[mask_idx])]
        q = len(order)
        if q == 0:
            radius *= 0.5
            continue
        I = iu0[order]; J = iu1[order]; cact = cfull[order]
        tl = deadline - time.time()
        if tl < 1.0:
            break
        dX = _lp_step(curX, I, J, cact, radius, tl)
        if dX is None:
            radius *= 0.5
            fails += 1
            if fails >= 8 and th_idx > 0:
                # loosen to get different direction
                th_idx = max(0, th_idx - 1)
                fails = 0
                radius = min(3e-4, radius * 2 + 1e-5)
            continue
        # backtracking with exact renormalisation
        acc_X = None; acc_m = None; acc_tie = None; acc_min = None
        # first pass: require tie non-decreasing
        for a in alphas:
            Y = _renorm(curX + a * dX)
            if not np.all(np.isfinite(Y)):
                continue
            vm, vslack, _, _, _ = _slacks(Y)
            vtie = _tie_stats(vslack, 1e-10)
            if vm < m - 1e-13 and vtie >= cur_tie:
                acc_X = Y; acc_m = vm; acc_tie = vtie
                break
        if acc_X is None:
            # second pass: allow tie decrease only if global best improves
            for a in alphas:
                Y = _renorm(curX + a * dX)
                if not np.all(np.isfinite(Y)):
                    continue
                vm, vslack, _, _, _ = _slacks(Y)
                vtie = _tie_stats(vslack, 1e-10)
                if vm < m - 1e-13 and vm < best_val - 1e-13:
                    acc_X = Y; acc_m = vm; acc_tie = vtie
                    break
        if acc_X is not None:
            prev_tie = cur_tie
            prev_th = th
            curX = acc_X
            cur_m = acc_m
            cur_tie = acc_tie
            _, cur_min_c = _contacts_per_point(n, iu0, iu1, _slacks(curX)[1], 1e-10)
            if cur_m < best_val - 1e-15:
                best_val = cur_m; bestX = curX.copy(); best_tie = cur_tie
            # tighten threshold geometrically
            if th_idx < len(thresholds) - 1:
                th_idx += 1
                # check restart condition: tightening creates >=50 new near-ties?
                # estimate new near-ties under new threshold vs old count
                _, nslack, _, _, _ = _slacks(curX)
                new_th = thresholds[th_idx]
                # count pairs with slack < 1e-7 plateau? use tie at 1e-9?
                # brief: restart radius at 1e-4 whenever tightening creates >=50 new near-ties
                # interpret as tie (1e-10) growth >=50 since prev
                if cur_tie - prev_tie >= 50:
                    radius = 1e-4
                else:
                    # also check active-set growth under old th
                    cnt_old = int(np.count_nonzero(nslack < prev_th))
                    cnt_new_scope = int(np.count_nonzero(nslack < 3e-7))
                    if cnt_new_scope - cnt_old >= 50:
                        radius = 1e-4
                    else:
                        radius = min(3e-4, radius * 1.2)
            else:
                radius = min(3e-4, radius * 1.1)
            fails = 0
        else:
            radius *= 0.5
            fails += 1
            # if many fails, loosen threshold to widen set
            if fails >= 6:
                if th_idx > 0:
                    th_idx = max(0, th_idx - 1)
                radius = max(radius, 5e-6)
                # small random tangent kick to avoid deadlock? keep deterministic
                # do not kick here; just reset fails
                fails = 0
        # early exit if radius tiny and at tightest threshold
        if radius < 2e-7 and th_idx >= len(thresholds) - 1:
            # try restart radius once
            radius = 1e-4
            # if no progress for a while break to allow outer perturb
            if it > 400:
                break
        if it > 800:
            break
    return bestX, best_val, best_tie

def solve():
    T0 = time.time()
    DEADLINE = T0 + 165.0
    HARD = T0 + 176.0
    rng = np.random.default_rng(0)
    rec = None; bst = None
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                if fn == "best.npy":
                    bst = a.copy()
                else:
                    rec = a.copy()
        except Exception:
            pass
    if rec is None and bst is None:
        r = rng.standard_normal(size=(94, 13))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        rec = r.copy(); bst = r.copy()
    if rec is None:
        rec = bst.copy()
    if bst is None:
        bst = rec.copy()
    rv = _max_cos(bst * 0 + bst) if False else _max_cos(bst)
    try:
        rv2 = _max_cos(rec)
    except Exception:
        rv2 = 1e9
    bv = _max_cos(bst)
    if bv <= rv2:
        incumbent = bst.copy(); inc_val = bv
    else:
        incumbent = rec.copy(); inc_val = rv2
    global_best = incumbent.copy(); global_val = inc_val
    try:
        _, gslack, _, _, _ = _slacks(global_best)
        global_tie = _tie_stats(gslack, 1e-10)
    except Exception:
        global_tie = 0
    # Phase 1: continuation from incumbent with radius 3e-4
    try:
        if time.time() < DEADLINE - 5:
            P, pv, pt = _continuation(incumbent, min(DEADLINE - 2, T0 + 95.0), rng, start_radius=3e-4)
            if np.all(np.isfinite(P)) and pv < global_val - 1e-15:
                global_best = _renorm(P); global_val = pv; global_tie = pt
            elif pv < global_val and np.all(np.isfinite(P)):
                # keep best ever evaluated even if tiny
                pass
            # keep best ever: if P better than incumbent keep
            # also track incumbent for perturbations: use best
    except Exception:
        pass
    # Phase 2: perturbed restarts to change topology while preserving scale
    # small tangent perturbations at 1e-7-scale relevant magnitudes
    sigmas = [2e-4, 5e-4, 1e-3]
    for si, sg in enumerate(sigmas):
        if time.time() > DEADLINE - 8:
            break
        try:
            N = rng.standard_normal(size=global_best.shape)
            dot = (N * global_best).sum(axis=1, keepdims=True)
            Nt = N - dot * global_best
            nn = np.linalg.norm(Nt, axis=1, keepdims=True)
            nn[nn < 1e-12] = 1.0
            Nt = Nt / nn
            Y0 = _renorm(global_best + sg * Nt)
            # short continuation
            per_dead = min(DEADLINE - 1, time.time() + (DEADLINE - time.time()) / (len(sigmas) - si + 1) + 5)
            P2, pv2, pt2 = _continuation(Y0, per_dead, rng, start_radius=1e-4)
            if np.all(np.isfinite(P2)) and pv2 < global_val - 1e-15:
                # enforce tie guard: accept only if tie not worse than global_tie - small tol,
                # unless global improvement is clear
                if pt2 >= global_tie or pv2 < global_val - 5e-9:
                    global_best = _renorm(P2); global_val = pv2; global_tie = pt2
                else:
                    # still keep if strictly better max (hard deadline rule: keep best ever)
                    if pv2 < global_val:
                        global_best = _renorm(P2); global_val = pv2; global_tie = pt2
        except Exception:
            continue
        if time.time() > HARD - 4:
            break
    # fallback
    try:
        gv = _max_cos(global_best)
        if not np.isfinite(gv) or gv > inc_val + 1e-12:
            return incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    except Exception:
        return incumbent
    global_best = _renorm(global_best)
    # final safety: no NaN, unit rows
    global_best[~np.isfinite(global_best)] = 0.0
    # if any zero row, replace with incumbent row
    nr = np.linalg.norm(global_best, axis=1)
    bad = ~(np.isfinite(nr)) | (nr < 1e-12)
    if np.any(bad):
        global_best[bad] = incumbent[bad]
        global_best = _renorm(global_best)
    return global_best
