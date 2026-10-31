import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.88):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row = np.maximum(row, 1e-12)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        scale = np.minimum(1.0, 2.0 / np.maximum(gn, 1e-12))
        grad = grad * np.where(gn > 2.0, scale, 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, radius_init=1e-3, cap=2500, maxit=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _degrees_sums(x, tol=1e-7, near_gap=0.002):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    adj = G > (m - tol)
    deg = np.sum(adj, axis=1).astype(int)
    thr = m - near_gap
    mask = G > thr
    # sum of near-max cosines per row (include only >thr)
    ssum = np.sum(np.where(mask, G, 0.0), axis=1)
    rowmax = np.max(G, axis=1)
    return deg, ssum, rowmax, m, G

def _graph_hash_deg(deg, m):
    ds = tuple(sorted(deg.tolist()))
    return (ds, round(float(m), 9))

def _coverage_frac(X, rng, nprobe=12000):
    n, d = X.shape
    P = rng.standard_normal(size=(nprobe, d))
    P = _norm(P)
    G = P @ X.T  # nprobe x n
    assign = np.argmax(G, axis=1)
    cnt = np.bincount(assign, minlength=n).astype(float)
    cnt /= float(nprobe)
    return cnt

def _largest_hole(X, rng, n_cand=25000, batch=8000):
    d = X.shape[1]
    best_c = None
    best_v = 1e9
    rem = n_cand
    while rem > 0:
        b = min(batch, rem); rem -= b
        C = rng.standard_normal(size=(b, d))
        C = _norm(C)
        G = C @ X.T
        vals = np.max(G, axis=1)
        i = int(np.argmin(vals))
        if float(vals[i]) < best_v:
            best_v = float(vals[i])
            best_c = C[i].copy()
    return best_c, best_v

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 32.0
    EXPLORE_END = HARD_END - FINAL_RESERVE
    n0, d = 153, 4
    rng_master = np.random.default_rng(7)
    base_x = None; base_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n0, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < base_v:
                    base_v = v; base_x = a.copy()
        except Exception:
            pass
    if base_x is None:
        base_x = _norm(rng_master.standard_normal(size=(n0, d)))
        base_v = _max_cos(base_x)
    record_v = float(base_v)
    best_x = base_x.copy(); best_v = float(base_v)
    # quick record hash
    deg0, _, _, m0, _ = _degrees_sums(base_x)
    record_hash = _graph_hash_deg(deg0, m0)

    # ---------- overshoot: insert K points ----------
    K = 14  # 153 -> 167 (within 165-170)
    Nover = n0 + K
    rng_ins = np.random.default_rng(12345)
    Xo = base_x.copy()
    for k in range(K):
        if _time.time() > EXPLORE_END - 60:
            break
        hc, hv = _largest_hole(Xo, rng_ins, n_cand=25000)
        Xo = np.vstack([Xo, hc.reshape(1, d)])
        Xo = _norm(Xo)
    # annealed softmax + short LP polish of oversized code
    if _time.time() < EXPLORE_END - 20:
        Xo = _smooth_opt(Xo, 1100, 0.03, 0.0025, 0.07, 0.003)
    dl = min(_time.time() + 3.0, EXPLORE_END - 10)
    if dl > _time.time() + 0.3:
        Xo, vo = _lp_polish(Xo, dl, radius_init=1e-3, cap=2500)
    else:
        vo = _max_cos(Xo)
    Nover = Xo.shape[0]
    is_ins0 = np.zeros(Nover, dtype=bool)
    if Nover > n0:
        is_ins0[Nover-K:] = True  # last K are inserted (approx, since we appended)
    # beam init: single member
    # each beam entry: dict with X, flag, del_orig (bool whether an original deleted so far)
    beam = [{'X': Xo.copy(), 'flag': is_ins0.copy(), 'del_orig': False}]
    seen_hashes = set()
    seen_hashes.add(record_hash)
    BEAM_W = 5
    rng_beam = np.random.default_rng(999)
    ndel_total = Nover - n0
    # ---------- prune in stages ----------
    for stage in range(ndel_total):
        if _time.time() > EXPLORE_END - 8:
            break
        children = []
        for bi, mem in enumerate(beam):
            X = mem['X']; flag = mem['flag']; del_orig = mem['del_orig']
            ncur = X.shape[0]
            deg, ssum, rowmax, mcur, Gcur = _degrees_sums(X, tol=1e-7)
            cov = _coverage_frac(X, rng_beam, nprobe=10000)
            # criteria
            # c1: highest incident contact count (tie -> larger ssum)
            # use lexsort: primary deg, secondary ssum
            order_deg = np.lexsort((ssum, deg))
            c1 = int(order_deg[-1])
            c2 = int(np.argmax(ssum))
            # avoid identical: if c2==c1 pick second best ssum not equal c1
            if c2 == c1:
                idx2 = np.argsort(ssum)
                # pick largest not c1
                for cand in idx2[::-1]:
                    if int(cand) != c1:
                        c2 = int(cand); break
            c3 = int(np.argmin(cov))
            if c3 in (c1, c2):
                idx3 = np.argsort(cov)
                for cand in idx3:
                    if int(cand) not in (c1, c2):
                        c3 = int(cand); break
            c4 = int(rng_beam.integers(0, ncur))
            # ensure distinct, add extra: second-highest deg as 5th for breadth
            cands = []
            for c in (c1, c2, c3, c4):
                if c not in cands:
                    cands.append(c)
            # add one more: highest rowmax (most constrained point) if distinct
            c5 = int(np.argmax(rowmax))
            if c5 not in cands and len(cands) < 5:
                cands.append(c5)
            for c in cands:
                Y = np.delete(X, c, axis=0)
                Y = _norm(Y)
                new_flag = np.delete(flag, c)
                was_orig = not bool(flag[c])
                new_del_orig = bool(del_orig or was_orig)
                vraw = _max_cos(Y)
                children.append({
                    'Y': Y, 'flag': new_flag, 'del_orig': new_del_orig,
                    'vraw': float(vraw), 'cov_del': float(cov[c]),
                    'deg_del': int(deg[c]), 'ssum_del': float(ssum[c]),
                    'parent': bi, 'crit': c,
                })
        # rank children by vraw, then cov, then -deg
        children.sort(key=lambda e: (e['vraw'], e['cov_del'], -e['deg_del']))
        # select next beam with hash dedup
        next_beam = []
        used_hash = set()
        # first pass: greedy in sorted order, skip duplicate hash
        for e in children:
            Y = e['Y']
            degY, _, _, mY, _ = _degrees_sums(Y, tol=1e-7)
            h = _graph_hash_deg(degY, mY)
            e['_hash'] = h
            if h in used_hash or h in seen_hashes:
                # allow if it is strictly best raw? skip to enforce diversity
                # but to avoid empty beam, allow after first pass
                continue
            used_hash.add(h)
            next_beam.append(e)
            if len(next_beam) >= BEAM_W:
                break
        # if not enough, second pass allowing seen hashes but still distinct within beam
        if len(next_beam) < BEAM_W:
            for e in children:
                h = e.get('_hash')
                if h is None:
                    degY, _, _, mY, _ = _degrees_sums(e['Y'], tol=1e-7)
                    h = _graph_hash_deg(degY, mY); e['_hash'] = h
                if h in used_hash:
                    continue
                used_hash.add(h)
                next_beam.append(e)
                if len(next_beam) >= BEAM_W:
                    break
        # fallback: fill with top children regardless
        if len(next_beam) < BEAM_W:
            for e in children:
                if e in next_beam:
                    continue
                next_beam.append(e)
                if len(next_beam) >= BEAM_W:
                    break
        # enforce at least one branch with del_orig=True (deletes original record point)
        has_orig_del = any(e['del_orig'] for e in next_beam)
        if not has_orig_del:
            # find best child with del_orig True
            best_orig = None
            for e in children:
                if e['del_orig']:
                    best_orig = e; break
            if best_orig is not None:
                # replace worst in next_beam
                next_beam[-1] = best_orig
        # convert to beam format
        beam = []
        for e in next_beam:
            beam.append({'X': e['Y'], 'flag': e['flag'], 'del_orig': e['del_orig']})
            seen_hashes.add(e['_hash'])
        # short re-optimisation after every 3 deletions (and at 2 remaining)
        steps_done = stage + 1
        cur_n = beam[0]['X'].shape[0]
        do_polish = (steps_done % 3 == 0) or (cur_n <= 155)
        if do_polish and _time.time() < EXPLORE_END - 5:
            for mem in beam:
                if _time.time() > EXPLORE_END - 4:
                    break
                Xc = mem['X']
                # short softmax
                Xc = _smooth_opt(Xc, 220, 0.015, 0.003, 0.03, 0.004)
                dl2 = _time.time() + 0.45
                Xc, vc = _lp_polish(Xc, dl2, radius_init=6e-4, cap=2000)
                mem['X'] = Xc
    # ---------- forced original-deletion branch (guarantee) ----------
    # Build one extra branch that explicitly deletes an original high-degree point first
    extra_branches = []
    try:
        if _time.time() < EXPLORE_END - 4:
            degO, ssumO, _, _, _ = _degrees_sums(Xo, tol=1e-7)
            # highest-deg original index in Xo (among non-inserted)
            orig_idx = np.where(~is_ins0)[0]
            # pick max deg among originals
            pick = orig_idx[int(np.argmax(degO[orig_idx]))]
            Xf = np.delete(Xo, pick, axis=0)
            flagf = np.delete(is_ins0, pick)
            # need to delete (K-1) more to reach 153: delete inserted with highest deg greedily
            while Xf.shape[0] > n0:
                d2, s2, _, _, _ = _degrees_sums(Xf, tol=1e-7)
                # prefer deleting inserted points with high degree
                ins_where = np.where(flagf)[0]
                if len(ins_where) > 0:
                    # highest deg among inserted
                    sub = d2[ins_where]
                    sub2 = s2[ins_where]
                    ord2 = np.lexsort((sub2, sub))
                    cdel = int(ins_where[ord2[-1]])
                else:
                    cdel = int(np.argmax(d2))
                Xf = np.delete(Xf, cdel, axis=0)
                flagf = np.delete(flagf, cdel)
            Xf = _smooth_opt(Xf, 300, 0.015, 0.003, 0.03, 0.003)
            dl2 = _time.time() + 0.8
            Xf, vf = _lp_polish(Xf, dl2, radius_init=7e-4, cap=2000)
            extra_branches.append((float(vf), Xf))
    except Exception:
        pass

    # collect beam survivors at 153
    survivors = []
    for mem in beam:
        Xc = mem['X']
        if Xc.shape[0] != n0:
            continue
        survivors.append(Xc)
    for vf, Xf in extra_branches:
        if Xf.shape[0] == n0:
            survivors.append(Xf)
    # dedup survivors by hash, keep up to 6
    uniq = []
    seen2 = set()
    for Xc in survivors:
        try:
            degc, _, _, mc, _ = _degrees_sums(Xc, tol=1e-7)
            h = _graph_hash_deg(degc, mc)
        except Exception:
            h = None
        if h is not None and h in seen2:
            continue
        seen2.add(h)
        uniq.append(Xc)
        if len(uniq) >= 6:
            break
    if len(uniq) == 0 and len(survivors) > 0:
        uniq = survivors[:6]
    # ---------- fully LP-polish every survivor ----------
    polished = []
    for Xc in uniq:
        if _time.time() > HARD_END - 6:
            # no time, keep raw
            polished.append((float(_max_cos(Xc)), Xc))
            continue
        # a bit of anneal then full polish
        try:
            Xc = _smooth_opt(Xc, 350, 0.012, 0.002, 0.025, 0.002)
        except Exception:
            pass
        budget = 4.0
        dl = min(_time.time() + budget, HARD_END - 3.0)
        try:
            Xp, vp = _lp_polish(Xc, dl, radius_init=8e-4, cap=2500)
        except Exception:
            Xp, vp = Xc, _max_cos(Xc)
        polished.append((float(vp), Xp.copy()))
        if float(vp) < best_v:
            best_v = float(vp); best_x = Xp.copy()
    # also consider extra_branches already polished
    for v, Xc in polished:
        if v < best_v:
            best_v = v; best_x = Xc.copy()
    # return best if beats incumbent, else incumbent (which is best_x anyway)
    # ensure strictly valid: normalise
    best_x = _norm(best_x)
    # tiny shrink guard not needed (we return as is)
    # if nothing beat record, return best found (incumbent or better)
    return np.asarray(best_x, dtype=float)
