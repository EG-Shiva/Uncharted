import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _deep_hole(Y, rng, n_cand=200000):
    # Y: (m,d) frozen+already placed. return unit vec minimizing max dot
    d = Y.shape[1]
    best_cand = None
    best_val = 1e9
    # batch to bound memory but total >=200k
    batch = 50000
    done = 0
    while done < n_cand:
        b = min(batch, n_cand - done)
        cand = rng.standard_normal((b, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Y.T
        md = np.max(dots, axis=1)
        j = int(np.argmin(md))
        if md[j] < best_val:
            best_val = float(md[j])
            best_cand = cand[j].copy()
        done += b
    return best_cand

def _refine_point(y0, Y, steps, beta, lr0, t_deadline=None):
    # minimize soft-max of dots Y@y subject to |y|=1
    y = y0.copy() / np.linalg.norm(y0)
    for s in range(steps):
        if t_deadline is not None and (s % 50 == 0) and time.time() > t_deadline:
            break
        f = s / max(steps - 1, 1)
        lr = lr0 * (1.0 - 0.6 * f)
        dots = Y @ y  # (m,)
        m = np.max(dots)
        # softmax weights
        e = np.exp(beta * (dots - m))
        w = e / np.sum(e)
        grad = w @ Y  # (d,)
        # tangential projection
        gt = grad - np.dot(grad, y) * y
        n = np.linalg.norm(gt)
        if n < 1e-12:
            break
        gt /= n
        y = y - lr * gt
        n2 = np.linalg.norm(y)
        if n2 == 0 or not np.isfinite(n2):
            break
        y /= n2
    return y

def _pick_cluster(X, k, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # degree with looser threshold for connectivity
    thr_conn = m - 1e-3
    thr_tight = m - 1e-7
    cnt = np.sum(G > thr_conn, axis=1)
    # seed among high-degree points
    order = np.argsort(-cnt)
    top = order[:15]
    seed = int(top[rng.integers(0, len(top))])
    cluster = [seed]
    in_c = np.zeros(n, dtype=bool)
    in_c[seed] = True
    # grow to size k keeping connectivity to cluster
    while len(cluster) < k:
        # candidates that contact current cluster
        # max dot to cluster
        mc = np.max(G[:, cluster], axis=1)
        # exclude already in
        mc[in_c] = -2.0
        # prefer tight contacts
        # take top 12 candidates
        cand_order = np.argsort(-mc)
        # filter to those above thr_conn, else take best anyway
        valid = [c for c in cand_order if mc[c] > thr_conn]
        if len(valid) == 0:
            valid = list(cand_order[:12])
        # random among top few to diversify
        pool = valid[:min(12, len(valid))]
        nxt = int(pool[rng.integers(0, len(pool))])
        cluster.append(nxt)
        in_c[nxt] = True
    return sorted(cluster)

def _lp_squeeze(X, deadline, radius_init=5e-4):
    # very short trust-region LP, respects deadline
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 12:
        it += 1
        if time.time() > deadline - 0.05:
            break
        G = X @ X.T
        iu = iu0
        cfull = G[iu]
        m = float(np.max(cfull))
        sel_thr = m - max(20 * radius, 1e-9)
        sel = cfull > sel_thr
        cntp = int(np.count_nonzero(sel))
        if cntp == 0:
            break
        if cntp > 700:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:700]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": max(0.2, deadline - time.time() - 0.05)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    rng = np.random.default_rng(20261101)
    t0 = time.time()
    HARD = t0 + 150.0
    SEARCH_END = t0 + 138.0
    n, d = 55, 21
    # load incumbent
    xb = None
    bscore = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm_rows(a)
                s = _max_cos(a)
                if s < bscore:
                    bscore = s
                    xb = a.copy()
        except Exception:
            pass
    if xb is None:
        xb = _norm_rows(rng.standard_normal((n, d)))
        bscore = _max_cos(xb)
    x_best = xb.copy()
    best = bscore
    cur = xb.copy()
    cur_score = best

    T0, T1 = 0.003, 0.00015
    it = 0
    while time.time() < SEARCH_END:
        it += 1
        if time.time() > HARD - 8.0:
            break
        elapsed = time.time() - t0
        frac = elapsed / 150.0
        Temp = T0 * (1.0 - frac) + T1 * frac
        # cluster size 4-8
        k = int(rng.integers(4, 9))
        try:
            cluster = _pick_cluster(cur, k, rng)
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cluster] = False
        frozen = cur[keep_mask]  # (n-k,d)
        # --- reinsert jointly ---
        # order random
        # new_pts in cluster order
        new_pts = np.zeros((k, d))
        ok = True
        for pi in range(k):
            if time.time() > HARD - 8.0:
                ok = False
                break
            if pi == 0:
                Y = frozen
            else:
                Y = np.vstack([frozen, new_pts[:pi]])
            try:
                h = _deep_hole(Y, rng, n_cand=200000)
            except Exception:
                ok = False
                break
            # local optimise on sphere vs Y (soft-min)
            beta = 150.0 + 100.0 * rng.random()
            h = _refine_point(h, Y, steps=350, beta=beta, lr0=0.04, t_deadline=HARD - 6.0)
            # tiny jitter
            h = h + 1e-4 * rng.standard_normal(d)
            h /= np.linalg.norm(h)
            new_pts[pi] = h
        if not ok:
            continue
        # cycling through cluster several times (joint refinement)
        n_cycles = 3
        for cy in range(n_cycles):
            if time.time() > HARD - 8.0:
                break
            perm = rng.permutation(k)
            beta_c = 200.0 + 150.0 * rng.random()
            for pi in perm:
                if time.time() > HARD - 7.0:
                    break
                # others = frozen + all new except pi
                others = np.vstack([frozen, np.delete(new_pts, pi, axis=0)])
                y0 = new_pts[pi]
                y1 = _refine_point(y0, others, steps=180, beta=beta_c, lr0=0.02, t_deadline=HARD - 6.0)
                new_pts[int(pi)] = y1
        x_try = cur.copy()
        for idx, pi in enumerate(cluster):
            x_try[pi] = new_pts[idx]
        x_try = _norm_rows(x_try)
        v_try = _max_cos(x_try)
        # only 0.5-1s LP squeeze after repair
        lp_budget = 0.5 + 0.5 * rng.random()
        deadline = min(time.time() + lp_budget, HARD - 5.0, SEARCH_END + 4.0)
        if deadline - time.time() > 0.25:
            # only squeeze if promising-ish to save time? spec says after each repair,
            # but skip hopeless to avoid timeouts
            if v_try < cur_score + 0.02:
                try:
                    x_try2, v_try2 = _lp_squeeze(x_try, deadline, radius_init=5e-4)
                except Exception:
                    v_try2 = v_try
                    x_try2 = x_try
                x_try, v_try = x_try2, v_try2
        # SA acceptance on true max, walk from accepted (not reset)
        delta = v_try - cur_score
        accept = False
        if delta < -1e-12:
            accept = True
        else:
            if Temp > 0:
                try:
                    p = float(np.exp(-delta / Temp)) if delta > 0 else 1.0
                except Exception:
                    p = 0.0
                if rng.random() < p:
                    accept = True
        if accept:
            cur = x_try.copy()
            cur_score = v_try
            if v_try < best - 1e-12:
                best = v_try
                x_best = x_try.copy()
        # time check inside loop
        if time.time() > HARD - 5.0:
            break
    # final tiny squeeze of best if time left (respect 150s cap)
    if time.time() < HARD - 2.0:
        try:
            dl = min(HARD - 0.5, time.time() + min(6.0, HARD - time.time() - 0.5))
            if dl - time.time() > 0.5:
                x_best, best = _lp_squeeze(x_best, dl, radius_init=4e-4)
        except Exception:
            pass
    x_best = _norm_rows(x_best)
    return x_best
