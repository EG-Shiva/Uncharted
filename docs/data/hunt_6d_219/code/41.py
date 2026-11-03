import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    BUDGET = 170.0
    HOP_END = t0 + 122.0
    rng = np.random.RandomState(20261104)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    # map pair (i,j) i<j -> index k
    # build lookup via dict for active only on demand; for cap mask use vectorised search
    # precompute pair index matrix
    idxmat = np.full((n, n), -1, dtype=int)
    idxmat[iu0, iu1] = np.arange(Np)
    idxmat[iu1, iu0] = np.arange(Np)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = _max_cos(x_best)

    def contact_set(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        I, J = np.where(np.triu(G >= m - tol, 1))
        s = set(zip(I.tolist(), J.tolist()))
        # normalize to i<j
        return s, m, G

    def degrees_from_set(s):
        deg = np.zeros(n, dtype=int)
        for a, b in s:
            deg[a] += 1
            deg[b] += 1
        return deg

    def jaccard(s1, s2):
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        if union == 0:
            return 1.0
        return inter / union

    def run_lbfgs_cap(x0, beta, maxiter, cap_idx, cap_val, K):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum()) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            pen = 0.0
            if cap_idx is not None and len(cap_idx) > 0:
                gc = g[cap_idx]
                viol = gc - cap_val
                pos = viol > 0
                if np.any(pos):
                    vv = viol[pos]
                    pen = K * float(np.sum(vv * vv))
                    wp = np.zeros_like(g)
                    wp[cap_idx[pos]] = 2.0 * K * vv
                    Wp = np.zeros((n, n))
                    Wp[iu0, iu1] = wp
                    Wp = Wp + Wp.T
                    W = W + Wp
            loss = loss + pen
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def run_free(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum()) + 1e-300
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=9e-4, max_steps=7):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        last_dual = None
        last_I = None; last_J = None
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            if time.time() - t0 > BUDGET - 2.0:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3400:
                kk = 3400
                part = np.argpartition(-csel, kk - 1)[:kk]
                I, J, csel = I[part], J[part], csel[part]
                p = kk
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            # capture duals
            try:
                marg = None
                if hasattr(res, "ineqlin") and res.ineqlin is not None:
                    marg = getattr(res.ineqlin, "marginals", None)
                    if marg is None:
                        marg = getattr(res.ineqlin, "marginal", None)
                if marg is not None:
                    last_dual = np.asarray(marg, dtype=float)
                    last_I = I.copy(); last_J = J.copy()
            except Exception:
                pass
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best, (last_dual, last_I, last_J)

    def dual_to_full(dual_tuple):
        dual, I, J = dual_tuple
        wfull = np.zeros(Np)
        if dual is None or I is None:
            return wfull, False
        try:
            # duals for A_ub x <= b with min c^Tx are <=0; stress = -marginal
            s = -np.asarray(dual, dtype=float)
            # clip negatives from tolerance
            s[s < 0] = 0
            for k in range(len(I)):
                a = int(I[k]); b = int(J[k])
                idx = int(idxmat[a, b])
                if idx >= 0:
                    wfull[idx] = s[k]
            return wfull, True
        except Exception:
            return np.zeros(Np), False

    # initial polish to get duals + graph
    dl = time.time() + 8.0
    x_best, best_val, dtup = lp_polish(x_best, dl, r_init=9e-4, max_steps=8)
    w_best, have_dual = dual_to_full(dtup)
    gs_best, mb, _ = contact_set(x_best, tol=1e-9)
    deg_best = degrees_from_set(gs_best)

    # pool: list of dicts
    pool = [{"X": x_best.copy(), "score": best_val, "graph": set(gs_best), "w": w_best.copy(), "deg": deg_best.copy()}]
    cur_X = x_best.copy(); cur_val = best_val

    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 42:
        trial += 1
        # select base: mostly best, sometimes diverse
        rsel = rng.rand()
        if rsel < 0.55:
            base = pool[0]
        elif len(pool) > 1 and rsel < 0.85:
            # prefer diverse: sample by rank
            pi = int(rng.randint(0, len(pool)))
            base = pool[pi]
        else:
            # current
            # build temp base from cur_X
            gs_cur, m_cur, _ = contact_set(cur_X, tol=1e-9)
            base = {"X": cur_X, "score": cur_val, "graph": gs_cur, "w": np.zeros(Np), "deg": degrees_from_set(gs_cur)}
        Xb = base["X"]; sb = base["score"]; gb = base["graph"]; wb = base["w"]; degb = base["deg"]
        # recompute m
        Gtmp = Xb @ Xb.T
        np.fill_diagonal(Gtmp, -2.0)
        mcur = float(Gtmp.max())
        ppm = Gtmp.max(axis=1)
        # identify hubs and rattlers
        order_deg = np.argsort(-degb)
        order_pm = np.argsort(-ppm)
        rattlers = [int(i) for i in range(n) if degb[i] == 0]
        hubs = order_deg[:4].tolist()
        # choose capped edge set (6-15 edges)
        # gather active edge list
        active_list = list(gb)
        # map active edge -> weight
        # build weight lookup
        def edge_w(e):
            a, b = e
            idx = int(idxmat[a, b])
            return float(wb[idx]) if idx >= 0 else 0.0
        mode = rng.rand()
        capped_pairs = []
        if mode < 0.42:
            # hub mode: pick one hub among top3
            cand_hubs = [h for h in hubs if degb[h] >= 10]
            if len(cand_hubs) == 0:
                cand_hubs = hubs[:2]
            h = int(cand_hubs[int(rng.randint(0, len(cand_hubs)))])
            # all incident contacts
            inc = [e for e in active_list if e[0] == h or e[1] == h]
            # if too many, keep all (14-15); if too few, add near-contacts of hub (next tightest pairs involving h)
            if len(inc) < 6:
                # add closest pairs involving h by Gtmp
                row = Gtmp[h].copy()
                row[h] = -2
                nn = np.argsort(-row)[:12]
                inc_set = set(inc)
                for u in nn:
                    a, b = (h, int(u)) if h < int(u) else (int(u), h)
                    if (a, b) not in inc_set:
                        inc.append((a, b))
                        inc_set.add((a, b))
                    if len(inc) >= 10:
                        break
            # subsample to 6-15 if larger
            if len(inc) > 15:
                # keep weakest by dual weight first (to keep strong? we want to cap all hub edges, but limit 15)
                inc_sorted = sorted(inc, key=edge_w)[:15]
                capped_pairs = inc_sorted
            else:
                capped_pairs = inc
        elif mode < 0.75:
            # weak cluster: smallest weights
            if np.sum(wb > 0) > 10:
                # sort active by weight
                scored = sorted(active_list, key=edge_w)
                # start from weakest
                start = scored[int(rng.randint(0, min(5, len(scored)-1)))]
                # BFS expand sharing vertices, prefer weak
                chosen = set([start])
                # order remaining by weight
                for e in scored:
                    if len(chosen) >= int(rng.randint(8, 15)):
                        break
                    # share vertex with chosen?
                    vs = set()
                    for a, b in chosen:
                        vs.add(a); vs.add(b)
                    if e[0] in vs or e[1] in vs:
                        chosen.add(e)
                # if still small, just take weakest k
                if len(chosen) < 6:
                    k = int(rng.randint(6, 13))
                    chosen = set(scored[:k])
                capped_pairs = list(chosen)
            else:
                # random local cluster: pick random point + neighbours
                c0 = int(rng.randint(0, n))
                row = Gtmp[c0].copy()
                row[c0] = -2
                nn = np.argsort(-row)[:8]
                pts = [c0] + nn[:5].tolist()
                spts = set(pts)
                cand = [e for e in active_list if e[0] in spts or e[1] in spts]
                if len(cand) < 6:
                    # take tightest pairs among pts neighbourhood
                    cand = []
                    for ii in range(len(pts)):
                        for jj in range(ii+1, len(pts)):
                            a, b = pts[ii], pts[jj]
                            if a > b: a, b = b, a
                            cand.append((a, b))
                # trim to 6-15
                if len(cand) > 15:
                    cand = cand[:15]
                capped_pairs = cand
        else:
            # mixed: hub + weak
            cand_hubs = hubs[:3]
            h = int(cand_hubs[int(rng.randint(0, len(cand_hubs)))])
            inc = [e for e in active_list if e[0] == h or e[1] == h]
            # add weak edges elsewhere
            if np.sum(wb > 0) > 10:
                scored = sorted(active_list, key=edge_w)
                add = [e for e in scored[:20] if e not in set(inc)][:6]
                capped_pairs = (inc[:10] + add)[:15]
            else:
                capped_pairs = inc[:15] if len(inc) > 6 else inc
            if len(capped_pairs) < 6:
                # pad with random tight pairs
                flat_order = np.argsort(-Gtmp[iu0, iu1])[:30]
                for k in flat_order:
                    a, b = int(iu0[k]), int(iu1[k])
                    if (a, b) not in set(capped_pairs):
                        capped_pairs.append((a, b))
                    if len(capped_pairs) >= 10:
                        break
        if len(capped_pairs) == 0:
            continue
        # cap value max - delta
        delta = float(rng.uniform(0.001, 0.003))
        cap_val = mcur - delta
        cap_idx = np.array([int(idxmat[a, b]) for a, b in capped_pairs if int(idxmat[a, b]) >= 0], dtype=int)
        # initial kick: tangential noise, stronger on involved vertices
        involved = set()
        for a, b in capped_pairs:
            involved.add(a); involved.add(b)
        if len(rattlers) > 0:
            for r in rattlers:
                involved.add(int(r))
        sigma = float(rng.uniform(0.012, 0.030))
        Xinit = Xb.copy()
        Nz = rng.randn(n, d)
        Nz -= (Nz * Xinit).sum(axis=1, keepdims=True) * Xinit
        # scale per row
        scale = np.ones(n) * sigma * 0.6
        for v in involved:
            scale[v] = sigma * 1.6
        # rattler extra kick
        for r in rattlers:
            scale[int(r)] = sigma * 2.2
        Xinit = Xinit + Nz * scale[:, None]
        Xinit = _norm(Xinit)
        beta_high = float(rng.choice([1500.0, 2200.0, 3000.0]))
        Kpen = float(rng.choice([3000.0, 8000.0, 15000.0]))
        # capped re-jam
        X1 = run_lbfgs_cap(Xinit, beta_high, 140, cap_idx, cap_val, Kpen)
        # check caps respected? if not, continue anyway to free phase (mechanism may still differ)
        # remove caps, free re-jam moderate beta
        beta_free = float(rng.choice([400.0, 650.0, 900.0]))
        X2 = run_free(X1, beta_free, 110)
        # quick eval
        v2 = _max_cos(X2)
        # if far worse than best+0.01 skip LP to save time
        if v2 > best_val + 0.012:
            # still maybe keep as diverse? skip
            cur_X, cur_val = X2, v2
            continue
        # LP polish short
        deadline = time.time() + 2.2
        # tighter radius for capped escapees
        X3, v3, dtup3 = lp_polish(X2, deadline, r_init=7e-4, max_steps=5)
        gs3, m3, G3 = contact_set(X3, tol=1e-9)
        deg3 = degrees_from_set(gs3)
        # metrics: hub drop, rattler gain, jaccard
        # former hub h? track first hub
        # compute jaccard vs best graph
        jac_best = jaccard(gs3, pool[0]["graph"])
        dist_best = 1.0 - jac_best
        # hub drop: max degree among former hubs?
        # rattler gain: degree of former rattlers
        ratt_gain = 0
        if len(rattlers) > 0:
            ratt_gain = int(np.max([deg3[int(r)] for r in rattlers]))
        hub_now = int(np.max(degb[hubs[:2]])) if len(hubs) >= 2 else 0
        # evaluate acceptance
        improved = v3 < best_val - 1e-12
        if improved:
            best_val = float(v3)
            x_best = X3.copy()
            w3, _ = dual_to_full(dtup3)
            pool.insert(0, {"X": X3.copy(), "score": float(v3), "graph": set(gs3), "w": w3.copy(), "deg": deg3.copy()})
            # trim pool to 8
            pool = sorted(pool, key=lambda e: e["score"])[:8]
            cur_X, cur_val = X3.copy(), float(v3)
        else:
            cur_X, cur_val = X3.copy(), float(v3)
            # diverse pool admission: within tolerance and graph-distant, reward hub-drop/rattler-gain
            tol = 0.004
            # reward: if rattler gained >=3 or hub dropped, enlarge tol
            bonus = False
            # check hub drop: former hub degree vs new degree at same index
            # use first hub targeted (if hub mode) else max degree drop
            maxdeg_new = int(deg3.max())
            maxdeg_old = int(degb.max())
            if ratt_gain >= 3:
                bonus = True
            # former hub index degree
            # find h variable if defined
            try:
                if 'h' in dir() or True:
                    pass
            except Exception:
                pass
            if maxdeg_new < 11 and maxdeg_old >= 13:
                bonus = True
            if bonus:
                tol = 0.007
            if v3 < pool[0]["score"] + tol:
                # check distance to all pool members
                mindist = min([1.0 - jaccard(gs3, e["graph"]) for e in pool]) if len(pool) > 0 else 1.0
                # require novelty unless very good
                need = 0.02 if not bonus else 0.008
                if v3 < pool[0]["score"] + 0.0015:
                    need = 0.015
                if mindist > need or (bonus and mindist > 0.005):
                    w3, _ = dual_to_full(dtup3)
                    pool.append({"X": X3.copy(), "score": float(v3), "graph": set(gs3), "w": w3.copy(), "deg": deg3.copy()})
                    pool = sorted(pool, key=lambda e: e["score"])[:8]
        # small time check
        if time.time() - t0 > BUDGET - 42:
            break

    # final intensive polish of top candidates
    pool = sorted(pool, key=lambda e: e["score"])[:4]
    # re-evaluate best
    x_best = pool[0]["X"].copy()
    best_val = float(pool[0]["score"])
    # spend remaining time polishing best with alternating free + LP
    end_final = t0 + BUDGET - 1.0
    # try a couple of free betas then LP
    cur = x_best.copy()
    # short free runs
    for beta in [600.0, 1500.0, 3000.0]:
        if time.time() > end_final - 12.0:
            break
        cur = run_free(cur, beta, 120)
        v = _max_cos(cur)
        if v < best_val:
            best_val = v
            x_best = cur.copy()
    # final LP long
    if time.time() < end_final - 2.0:
        x_best, best_val, _ = lp_polish(x_best if _max_cos(cur) >= best_val else cur, end_final, r_init=1.0e-3, max_steps=14)
    else:
        x_best = _norm(x_best)
        best_val = _max_cos(x_best)
    # also check other pool members polished quickly
    for e in pool[1:]:
        if time.time() > end_final - 1.0:
            break
        Xc = e["X"]
        vc = _max_cos(Xc)
        if vc < best_val:
            best_val = vc
            x_best = Xc.copy()
    x_best = _norm(np.asarray(x_best, dtype=float))
    # guard NaN
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    # renormalize, fix zero rows
    nr = np.linalg.norm(x_best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-9:
            v = rng.randn(d)
            v /= np.linalg.norm(v)
            x_best[i] = v
        else:
            x_best[i] /= nr[i]
    return x_best
