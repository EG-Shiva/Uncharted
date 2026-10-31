import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    # fix any bad row
    nr = np.linalg.norm(x, axis=1)
    bad = ~(nr > 1e-9)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_full(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _sub_anneal(x_full, mov, taus, iters_list, t_limit):
    # optimize only mov indices, frozen fixed, minimize smooth max over mov-involved pairs
    n, d = x_full.shape
    mov = np.asarray(mov, dtype=int)
    k = len(mov)
    if k == 0:
        return x_full.copy()
    is_mov = np.zeros(n, dtype=bool); is_mov[mov] = True
    froz = np.where(~is_mov)[0]
    Xf = x_full[froz].copy() if len(froz) > 0 else np.zeros((0, d))
    Xm0 = x_full[mov].copy()
    y = _norm(Xm0).ravel().copy()
    mov_map = {g: l for l, g in enumerate(mov)}
    t0 = time.time()
    for tau, mit in zip(taus, iters_list):
        if time.time() - t0 > t_limit:
            break
        def fun(yflat, _tau=tau):
            Y = yflat.reshape(k, d)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            Xm = Y / norms
            # build all relevant dots
            # M-M
            Gmm = Xm @ Xm.T
            # M-F
            Gmf = Xm @ Xf.T if len(froz) > 0 else np.zeros((k, 0))
            # max over off-diag M-M (i<j) and all M-F
            # for stability compute Gmax
            m1 = -2.0
            if k > 1:
                # max off diag
                tmp = Gmm.copy(); np.fill_diagonal(tmp, -2.0)
                m1 = float(tmp.max())
            m2 = float(Gmf.max()) if Gmf.size > 0 else -2.0
            Gmax = m1 if m1 > m2 else m2
            # weights
            # M-M weights (sym, zero diag)
            Wmm = np.zeros((k, k))
            if k > 1:
                E = np.exp((Gmm - Gmax) / _tau)
                np.fill_diagonal(E, 0.0)
                # keep only upper? for loss sum over i<j once
                # sum over unordered once:
                S = float(E.sum()) * 0.5
                if len(froz) > 0:
                    Ef = np.exp((Gmf - Gmax) / _tau)
                    Sf = float(Ef.sum())
                else:
                    Ef = None; Sf = 0.0
                Stot = S + Sf
                if Stot <= 0:
                    Stot = 1e-300
                loss = Gmax + _tau * np.log(Stot)
                # normalized weights: w = E/Stot etc (unordered weight)
                # For M-M unordered, weight for pair = E_ij/Stot
                Wmm = E / Stot
                # gradient: Grad_i = sum_j Wmm_ij * Xm_j  (for i!=j) + sum_f Wmf_if * Xf_f
                # Wmf:
                if Ef is not None:
                    Wmf = Ef / Stot
                else:
                    Wmf = np.zeros((k, 0))
            else:
                Ef = np.exp((Gmf - Gmax) / _tau) if Gmf.size > 0 else np.zeros((k, 0))
                Stot = float(Ef.sum()) if Ef.size > 0 else 1e-300
                if Stot <= 0:
                    Stot = 1e-300
                loss = Gmax + _tau * np.log(max(Stot, 1e-300))
                Wmf = Ef / Stot if Ef.size > 0 else np.zeros((k, 0))
            GradX = Wmm @ Xm
            if Wmf.size > 0:
                GradX = GradX + Wmf @ Xf
            dot = (GradX * Xm).sum(axis=1, keepdims=True)
            Gt = GradX - dot * Xm
            GradY = Gt / norms
            return float(loss), GradY.ravel()
        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mit), "ftol": 1e-14, "gtol": 1e-7, "maxfun": int(mit * 8)})
            y = res.x
        except Exception:
            break
    Xm = y.reshape(k, d)
    Xm = _norm(Xm)
    out = x_full.copy()
    out[mov] = Xm
    out = _norm(out)
    return out

def _mov_max(x_full, mov):
    n = x_full.shape[0]
    is_mov = np.zeros(n, dtype=bool); is_mov[np.asarray(mov)] = True
    G = x_full @ x_full.T
    # mask pairs where at least one is mov and i!=j
    m = -2.0
    for i in mov:
        row = G[i].copy()
        row[i] = -2.0
        v = float(row.max())
        if v > m:
            m = v
    return float(m)

def _equidistant_candidates(near_pts):
    # near_pts: (m,4) frozen neighbours
    cands = []
    m = near_pts.shape[0]
    if m < 4:
        return cands
    # limit combos: take up to 12 nearest, sample combos
    idxs = list(range(min(m, 12)))
    combos = list(itertools.combinations(idxs, 4))
    # cap combos to 250
    if len(combos) > 250:
        step = len(combos) / 250.0
        combos = [combos[int(i * step)] for i in range(250)]
    for c in combos:
        A = near_pts[list(c), :]
        try:
            # solve A x = 1
            ones = np.ones(4)
            v = np.linalg.solve(A, ones)
        except Exception:
            continue
        nv = np.linalg.norm(v)
        if not np.isfinite(nv) or nv < 1e-9:
            continue
        x = v / nv
        # dots should be positive ~ 1/nv ; if negative, flip?
        # keep both x and -x? evaluate later; keep x if dots positive
        cands.append(x)
        # also antipode is far from these 4 but keep as diversity? skip
    return cands

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((272, 4))
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (272, 4):
        xb = _norm(rng.standard_normal((272, 4)))
    record = _max_cos(xb)
    best_x = xb.copy()
    best_val = record
    n, d = 272, 4
    # contact analysis with strict threshold
    G0 = best_x @ best_x.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    thr_strict = 1e-9
    counts = (G0 >= m0 - thr_strict).sum(axis=1)
    rattlers = np.where(counts == 0)[0].tolist()
    low = np.where(counts <= 4)[0].tolist()
    # build movable set: low + neighbours with cosine > 0.75
    neigh_thresh = 0.75
    Gfull = best_x @ best_x.T
    mov_set = set(low)
    for i in low:
        row = Gfull[i]
        nbrs = np.where(row > neigh_thresh)[0].tolist()
        for j in nbrs:
            if j != i:
                mov_set.add(int(j))
    # ensure rattlers in
    for r in rattlers:
        mov_set.add(int(r))
    mov = sorted(mov_set)
    # size control: if too small, add <=5 contacts; if too large, trim to 55
    if len(mov) < 20:
        add = np.where(counts <= 5)[0].tolist()
        for a in add:
            mov_set.add(int(a))
        mov = sorted(mov_set)
    if len(mov) > 60:
        # keep rattlers + lowest counts + worst max first
        pw = Gfull.max(axis=1)
        order = sorted(mov, key=lambda i: (counts[i], -pw[i]))
        mov = sorted(order[:60])
        mov_set = set(mov)
    # --- candidate seeding for rattlers: dense random + equidistant ---
    Xcur = best_x.copy()
    # frozen for scoring
    for r in rattlers:
        if time.time() > HARD_END - 100:
            break
        # others fixed currently
        others_mask = np.ones(n, dtype=bool); others_mask[r] = False
        others = Xcur[others_mask]
        # dense random samples
        Nc = 20000
        cand = rng.standard_normal((Nc, 4)); cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        s = cand @ others.T
        sc = s.max(axis=1)
        bi = int(np.argmin(sc))
        best_c = cand[bi]; best_s = float(sc[bi])
        # equidistant from nearby frozen
        dots = (Xcur[r] @ others.T) if False else None
        # nearest frozen = largest dots to current rattler? use largest Gfull[r, others]
        row = (Xcur @ Xcur.T)[r]
        # get indices of largest dots excluding r
        nn_idx = np.argsort(-row)[:12]
        # filter out other mov? prefer frozen neighbours
        near_pts = Xcur[nn_idx]
        eqs = _equidistant_candidates(near_pts)
        for e in eqs:
            v = float((e @ others.T).max())
            if v < best_s:
                best_s = v; best_c = e
        # local Voronoi-ish: also try small perturbations around best_c
        # replace rattler if improves its worst dot
        cur_s = float((Xcur[r] @ others.T).max())
        if best_s < cur_s - 1e-6:
            Xcur[r] = best_c / np.linalg.norm(best_c)
    # evaluate after seeding
    Xcur = _norm(Xcur)
    vcur = _max_cos(Xcur)
    # if seeding created violation above record, revert rattlers that violate? keep only if global not worse than record+1e-9
    # Actually seeding should lower rattler dots, so global stays = frozen-frozen = record; allow tiny tol
    if vcur > record + 5e-4:
        Xcur = best_x.copy(); vcur = record
    # --- frozen-core feasibility loop over targets ---
    targets = [0.8975, 0.8970, 0.8965]
    cur_mov = list(mov)
    cur_X = Xcur.copy()
    cur_thresh = neigh_thresh
    for ti, tgt in enumerate(targets):
        if time.time() > HARD_END - 95:
            break
        # sub-anneal with decreasing tau
        taus = [0.01, 0.005, 0.0025]
        iters = [120, 120, 150]
        # time budget per target ~12s
        budget = min(14.0, max(6.0, (HARD_END - 95 - time.time()) / (len(targets) - ti)))
        # ensure mov includes neighbours at cur_thresh (recompute from cur_X? use original)
        attempt_X = _sub_anneal(cur_X, cur_mov, taus, iters, budget)
        mm = _mov_max(attempt_X, cur_mov)
        glob = _max_cos(attempt_X)
        # near-contact check for rattlers: count dots > tgt-0.003
        ok_contacts = True
        G = attempt_X @ attempt_X.T
        for r in rattlers:
            # map global r: if r not in cur_mov, it is frozen? but rattlers are in mov
            cnt_near = int(((G[r] > tgt - 0.003).sum()) - 0)  # includes self? self=1, so subtract
            # G[r,r]=1 > threshold, so subtract 1
            cnt_near = cnt_near - 1
            if cnt_near < 3:
                ok_contacts = False
        # acceptance: no pair above record (allow 2e-12) and mov max <= tgt+2e-4
        if glob <= record + 2e-9 and mm <= tgt + 2e-4:
            # accept even if contacts not yet 3? prefer accept to continue hardening
            cur_X = attempt_X.copy()
            # if rattlers lack contacts, do extra local densification: try to pull rattler toward 3 nearest to create contacts?
            # just record
            if mm < best_val:
                pass
        else:
            # infeasible: progressively unfreeze larger neighbourhood
            if cur_thresh > 0.60:
                cur_thresh -= 0.05
                # expand mov
                new_set = set(cur_mov)
                for i in list(cur_mov):
                    row = (cur_X @ cur_X.T)[i]
                    for j in np.where(row > cur_thresh)[0].tolist():
                        new_set.add(int(j))
                # cap at 70
                if len(new_set) > 70:
                    # trim by contact count recomputed
                    Gt = cur_X @ cur_X.T; np.fill_diagonal(Gt, -2.0)
                    mt = float(Gt.max()); ct = (Gt >= mt - 1e-9).sum(axis=1)
                    pwt = Gt.max(axis=1)
                    order = sorted(new_set, key=lambda i: (ct[i], -pwt[i]))
                    new_set = set(order[:70])
                cur_mov = sorted(new_set)
                # retry same target once with expanded set if time allows
                if time.time() < HARD_END - 100 and len(cur_mov) <= 75:
                    attempt_X2 = _sub_anneal(cur_X, cur_mov, taus, iters, budget)
                    mm2 = _mov_max(attempt_X2, cur_mov); glob2 = _max_cos(attempt_X2)
                    if glob2 <= record + 2e-9 and mm2 <= tgt + 2e-4:
                        cur_X = attempt_X2.copy()
                    # else keep cur_X unchanged, move to next easier? break expanding
                else:
                    pass
            else:
                pass
    # after surgery, cur_X is balanced candidate (mov max below record hopefully)
    surg_X = _norm(cur_X)
    surg_val = _max_cos(surg_X)
    # --- global anneal from surgery + from best to allow hubs to move ---
    # short global smooth-max anneal
    def _global_anneal(x_init, taus_g, iters_g, budget):
        y = _norm(x_init).ravel().copy()
        nn, dd = x_init.shape
        tstart = time.time()
        for tau, mit in zip(taus_g, iters_g):
            if time.time() - tstart > budget or time.time() > HARD_END - 45:
                break
            def fun(yflat, _tau=tau):
                Y = yflat.reshape(nn, dd)
                norms = np.linalg.norm(Y, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                X = Y / norms
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                Gm = float(G.max())
                E = np.exp((G - Gm) / _tau)
                np.fill_diagonal(E, 0.0)
                Es = float(E.sum()) * 0.5
                if Es <= 0:
                    Es = 1e-300
                loss = Gm + _tau * np.log(max(Es, 1e-300))
                W = E / max(Es * 2.0, 1e-300) * 2.0
                # careful: Es is sum/2? we did *0.5, so W = E/(2*Es_half?) Let's just use E/Es_half/2?
                # Simpler: total unordered sum S = Es_half*? Actually Es above = sum(E)/2.
                # normalized unordered weight = E / (2*Es)?? No: E sum = 2*Es, so E/(2*Es) sums to 1 over ordered pairs (both directions).
                # Gradient needs ordered: Grad = (E/(2*Es)) @ X *2? Let's derive: loss grad = sum_{i<j} w_ij*(...), w=E_ij/Es_total_unordered.
                # With Word = E/(2*Es_unord?) Wait Es_unord = sum(E)/2, so E_ij/Es_unord = 2*E_ij/sum(E).
                # Ordered weight = E_ij/sum(E) = (E_ij/Es_unord)/2. Gradient Grad_i = sum_j (E_ij/Es_unord) X_j.
                # So Grad = (E/Es_unord) @ X with zero diag? Since E/Es_unord sums to 2 per? Check: use Wu = E/Es (Es=sum/2) => Wu sum =2*Npairs-weighted.
                # Then Grad = Wu @ X gives factor 1? For unordered weight w=E/Es, Grad_i=sum w_ij X_j correct.
                W2 = E / max(Es, 1e-300)
                np.fill_diagonal(W2, 0.0)
                GradX = W2 @ X
                dot = (GradX * X).sum(axis=1, keepdims=True)
                Gt = GradX - dot * X
                GradY = Gt / norms
                return float(loss), GradY.ravel()
            try:
                res = minimize(fun, y, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mit), "ftol": 1e-14, "gtol": 1e-7, "maxfun": int(mit * 6)})
                y = res.x
            except Exception:
                break
        return _norm(y.reshape(nn, dd))
    cands = [(best_val, best_x.copy()), (surg_val, surg_X.copy())]
    # global anneal from surgery
    if time.time() < HARD_END - 60:
        budg = max(10.0, min(30.0, (HARD_END - 45 - time.time()) / 2.0))
        try:
            g1 = _global_anneal(surg_X, [0.008, 0.004, 0.002], [80, 100, 120], budg)
            cands.append((_max_cos(g1), g1))
        except Exception:
            pass
    if time.time() < HARD_END - 50:
        budg2 = max(8.0, min(20.0, HARD_END - 45 - time.time()))
        # small kick from best to diversify
        xk = _norm(best_x + 0.01 * rng.standard_normal(best_x.shape))
        try:
            g2 = _global_anneal(xk, [0.006, 0.003], [80, 100], budg2)
            cands.append((_max_cos(g2), g2))
        except Exception:
            pass
    cands.sort(key=lambda kv: kv[0])
    # final LP polish top candidates
    polished_x = cands[0][1]; polished_v = cands[0][0]
    for v, xc in cands[:3]:
        if time.time() > HARD_END - 6:
            break
        deadline = min(HARD_END - 1, time.time() + max(10.0, (HARD_END - 1 - time.time()) / 2.0))
        try:
            xp, vp = _lp_polish_full(xc, deadline)
        except Exception:
            continue
        if vp < polished_v:
            polished_v = vp; polished_x = xp.copy()
        if vp < best_val:
            best_val = vp; best_x = xp.copy()
    out = _norm(polished_x if 'polished_x' in locals() else best_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = _norm(out)
    return out
