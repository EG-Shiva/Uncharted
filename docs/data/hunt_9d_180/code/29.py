import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def per_info(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    # sum of near-max cosines for tie-break
    mask = G > m - 1e-7
    # exclude diagonal (-2)
    s = np.sum(np.where(mask, G, 0.0), axis=1)
    per_max = np.max(G, axis=1)
    return G, m, cnt, s, per_max

def lp_polish(x, budget, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = 3e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    # cache iu per n (n changes) - just use iu0
    it = 0
    while True:
        if time.time() >= deadline - 0.5:
            break
        if time.time() - start >= budget:
            break
        if radius < 1e-12:
            break
        it += 1
        G = x @ x.T
        # upper tri values
        c = G[iu0]
        m = float(np.max(c))
        thr = max(5.0 * radius, 5e-8)
        # select near-max
        sel = c > m - thr
        if np.count_nonzero(sel) == 0:
            break
        # cap to tightest 2500 to keep LP fast
        if np.count_nonzero(sel) > 2500:
            # get indices of top 2500
            idx = np.where(sel)[0]
            vals = c[idx]
            part = np.argpartition(-vals, 2500)[:2500]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
        I = iu0[0][sel]; J = iu0[1][sel]
        csub = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.4, 0.02)
        elif v < max_cos(x) - 1e-15:
            x = y
            radius *= 0.7
        else:
            # no progress along linear model: shrink
            # keep best x as current if we wandered? stay at x
            radius *= 0.5
            # if stuck, restart x from best
            if radius < 1e-6:
                x = bx.copy()
        # early stop if time low
        if it > 60:
            break
    return bx / np.linalg.norm(bx, axis=1, keepdims=True), best

def find_void(others, rng, n_try=20000, n_ref=400):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_ref):
        frac = k / max(1, n_ref - 1)
        sig = 0.05 * (0.0005 / 0.05) ** frac
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12 or not np.isfinite(nrm):
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    delete_end = t0 + 100.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        try:
            if c is None:
                continue
            c = np.asarray(c, dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
                    cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    # initial quick check: ensure best is polished copy? no, keep as is
    n, d = 180, 9
    cyc = 0
    # precompute nothing
    while time.time() < delete_end and time.time() < deadline - 50:
        cyc += 1
        mode = cyc % 3  # 1,2,0 cycle
        # time left for this cycle: aim ~9s
        if deadline - time.time() < 12:
            break
        try:
            _, m_cur, cnt_cur, s_cur, per_cur = per_info(x_best)
        except Exception:
            break
        if mode == 1:
            # MODE A: delete single most overloaded
            order = np.argsort(-(cnt_cur * 1000.0 + (s_cur - s_cur.min()) * 10.0 + per_cur))
            # vary choice among top to avoid repeating same deletion
            topk = min(8, len(order))
            pick_pos = ((cyc // 3) % topk)
            # also random jitter: occasionally pick random top
            if rng.rand() < 0.3:
                pick_pos = rng.randint(topk)
            del_idx = int(order[pick_pos])
            mask = np.ones(n, dtype=bool); mask[del_idx] = False
            x179 = x_best[mask].copy()
            # tiny shake to break symmetry (only neighbours get larger kick)
            # compute neighbours of deleted point
            try:
                dp = x_best[del_idx] @ x179.T
                neigh = np.argsort(-dp)[:20]
                x179[neigh] += 0.002 * rng.randn(len(neigh), d)
                x179 /= np.linalg.norm(x179, axis=1, keepdims=True)
            except Exception:
                pass
            x179p, m179 = lp_polish(x179, budget=5.0, deadline=deadline)
            # insert void
            y, _ = find_void(x179p, rng, n_try=18000, n_ref=350)
            x180 = np.vstack([x179p, y.reshape(1, -1)])
            x180p, m180 = lp_polish(x180, budget=3.5, deadline=deadline)
            if m180 < b_best - 1e-12:
                x_best = x180p.copy(); b_best = m180
            # else discard
        elif mode == 2:
            # MODE B: delete rattler + overloaded -> 178 then add 2
            # rattler = smallest count (tie smallest per_max)
            order_low = np.argsort(cnt_cur * 100.0 + per_cur)
            order_high = np.argsort(-(cnt_cur * 1000.0 + per_cur))
            # pick rattler among lowest 5 randomly/deterministically
            r_idx = int(order_low[(cyc // 3) % min(5, len(order_low))])
            # pick high among top 8 avoiding r_idx
            h_idx = None
            for cand_idx in order_high:
                cand_idx = int(cand_idx)
                if cand_idx != r_idx:
                    # vary
                    h_idx = cand_idx
                    break
            # if same cycle offset, shift high choice
            shift = ((cyc // 3) % 6)
            cand_list = [int(v) for v in order_high if int(v) != r_idx][:8]
            h_idx = cand_list[shift % len(cand_list)]
            mask = np.ones(n, dtype=bool); mask[r_idx] = False; mask[h_idx] = False
            x178 = x_best[mask].copy()
            x178 += 0.0025 * rng.randn(*x178.shape)
            x178 /= np.linalg.norm(x178, axis=1, keepdims=True)
            x178p, m178 = lp_polish(x178, budget=4.5, deadline=deadline)
            # two sequential voids
            y1, _ = find_void(x178p, rng, n_try=15000, n_ref=300)
            tmp = np.vstack([x178p, y1.reshape(1, -1)])
            y2, _ = find_void(tmp, rng, n_try=15000, n_ref=300)
            x180 = np.vstack([tmp, y2.reshape(1, -1)])
            x180p, m180 = lp_polish(x180, budget=3.5, deadline=deadline)
            if m180 < b_best - 1e-12:
                x_best = x180p.copy(); b_best = m180
        else:
            # MODE C: grow to 182 then prune 2 worst
            y1, _ = find_void(x_best, rng, n_try=15000, n_ref=300)
            tmp = np.vstack([x_best, y1.reshape(1, -1)])
            y2, _ = find_void(tmp, rng, n_try=15000, n_ref=300)
            x182 = np.vstack([tmp, y2.reshape(1, -1)])
            # light polish of 182
            x182p, m182 = lp_polish(x182, budget=4.0, deadline=deadline)
            # delete two worst: largest per-point max, tie by count
            try:
                _, m182v, cnt182, _, per182 = per_info(x182p)
            except Exception:
                continue
            worst = np.argsort(-per182)[:4]
            # among worst 4 pick 2 with largest count
            sub = worst[np.argsort(-cnt182[worst])[:2]]
            mask = np.ones(182, dtype=bool); mask[sub] = False
            x180 = x182p[mask].copy()
            x180p, m180 = lp_polish(x180, budget=3.0, deadline=deadline)
            if m180 < b_best - 1e-12:
                x_best = x180p.copy(); b_best = m180
        # loop
    # final polish reserve <=40s
    rem = deadline - time.time() - 1.0
    if rem > 3:
        try:
            budget = min(rem, 38.0)
            x_best, b_best = lp_polish(x_best, budget=budget, deadline=deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # tiny shrink guard not needed (we output unit vectors)
    return x_best
