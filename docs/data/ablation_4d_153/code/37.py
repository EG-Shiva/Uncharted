import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

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

def _lp_polish(x, deadline, radius_init=1e-3, cap=2500):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
        if time.time() >= deadline:
            break
    return x, best

def _contact_degrees(x, tol=1e-7):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    adj = G > (m - tol)
    deg = np.sum(adj, axis=1).astype(int)
    return deg, m

def _graph_hash(x, tol=1e-7):
    deg, m = _contact_degrees(x, tol)
    ds = tuple(sorted(deg.tolist()))
    return (ds, round(m, 9))

def _largest_hole_insert(X, rng, n_cand=50000, batch=10000):
    n, d = X.shape
    best_c = None
    best_v = 1e9
    remaining = n_cand
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b
        C = rng.standard_normal(size=(b, d))
        C = _norm(C)
        # max cosine to X: C @ X.T max over axis 1
        # chunk further to limit memory: b x n
        G = C @ X.T
        vals = np.max(G, axis=1)
        i = int(np.argmin(vals))
        if float(vals[i]) < best_v:
            best_v = float(vals[i])
            best_c = C[i].copy()
    return best_c, best_v

def _joint_polish(X, smooth_steps, tau0, tau1, lr0, lr1, lp_budget_sec):
    X = _smooth_opt(X, smooth_steps, tau0, tau1, lr0, lr1)
    dl = time.time() + lp_budget_sec
    Xp, vp = _lp_polish(X, dl)
    return Xp, vp

def solve():
    import time as _time
    t0 = _time.time()
    HARD_END = t0 + 175.0
    FINAL_RESERVE = 30.0
    EXPLORE_END = HARD_END - FINAL_RESERVE
    n0, d = 153, 4
    rng_master = np.random.default_rng(0)
    # load base
    base_x = None
    base_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n0, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < base_v:
                    base_v = v
                    base_x = a.copy()
        except Exception:
            pass
    if base_x is None:
        base_x = _norm(rng_master.standard_normal(size=(n0, d)))
        base_v = _max_cos(base_x)
    record_hash = _graph_hash(base_x)
    record_v = base_v
    best_x = base_x.copy()
    best_v = base_v
    seen_hashes = set()
    seen_hashes.add(record_hash)
    endpoints = []
    def time_left():
        return EXPLORE_END - _time.time()

    # ---------- overshoot-and-prune trajectories ----------
    # do 2 trajectories with different hole-search seeds
    for traj in range(2):
        if time_left() < 25.0:
            break
        rng = np.random.default_rng(100 + traj)
        X = base_x.copy()
        inserted_idx = []
        # build N=154,155,156
        ok = True
        for N in (154, 155, 156):
            if _time.time() > EXPLORE_END - 20.0:
                ok = False; break
            hc, hv = _largest_hole_insert(X, rng, n_cand=50000)
            X = np.vstack([X, hc.reshape(1, d)])
            X = _norm(X)
            # joint soft-max + LP
            X, v = _joint_polish(X, 800, 0.03, 0.003, 0.07, 0.003, 1.4)
            if _time.time() > EXPLORE_END - 15.0:
                pass
        if not ok:
            continue
        # prune back to 153 greedily with trial deletion + short LP
        cur = X.copy()
        cur_n = cur.shape[0]
        # track which indices were inserted (last 3)
        # after each deletion indices shift; use flag array
        is_inserted = np.zeros(cur_n, dtype=bool)
        is_inserted[-3:] = True
        while cur_n > 153:
            if _time.time() > EXPLORE_END - 8.0:
                break
            deg, mcur = _contact_degrees(cur, tol=1e-7)
            # candidate set: all inserted + top high-degree points
            cand = set(np.where(is_inserted)[0].tolist())
            order = np.argsort(-deg)
            # add top 6 high-degree (non-duplicate)
            added = 0
            for o in order:
                if int(o) not in cand:
                    cand.add(int(o))
                    added += 1
                    if added >= 6:
                        break
            cand = sorted(cand)
            # limit to 9 candidates max
            if len(cand) > 9:
                # keep all inserted + top among rest by degree
                ins = sorted(np.where(is_inserted)[0].tolist())
                rest = [c for c in cand if c not in ins]
                # sort rest by degree desc
                rest = sorted(rest, key=lambda c: -deg[c])[:max(0, 9 - len(ins))]
                cand = sorted(ins + rest)
            best_cand_v = 1e9
            best_cand_x = None
            best_cand_del = None
            for cdel in cand:
                if _time.time() > EXPLORE_END - 5.0:
                    break
                Y = np.delete(cur, cdel, axis=0)
                Y = _norm(Y)
                dl = _time.time() + 0.65
                Yp, vp = _lp_polish(Y, dl, radius_init=8e-4, cap=2000)
                # diversity: penalize exact record basin? just record value
                if vp < best_cand_v:
                    best_cand_v = vp
                    best_cand_x = Yp
                    best_cand_del = cdel
            if best_cand_x is None:
                break
            # update inserted flags
            new_flag = np.delete(is_inserted, best_cand_del)
            is_inserted = new_flag
            cur = best_cand_x
            cur_n = cur.shape[0]
            # brief smooth to relax after deletion
            if time_left() > 8.0:
                cur = _smooth_opt(cur, 300, 0.015, 0.003, 0.03, 0.003)
                dl2 = _time.time() + 0.4
                cur, _ = _lp_polish(cur, dl2, radius_init=5e-4, cap=2000)
        if cur.shape[0] == 153:
            # medium polish
            dl = _time.time() + 1.5
            cur, vcur = _lp_polish(cur, dl)
            # extra smooth+polish for diversity if same basin as record
            h = _graph_hash(cur)
            # if hash equals record hash and value not better, kick it
            if h == record_hash and vcur >= record_v - 1e-9 and time_left() > 10.0:
                # coordinated kick: perturb high-contact cluster
                deg2, _ = _contact_degrees(cur, tol=1e-6)
                hi = np.argsort(-deg2)[:12]
                nz = rng_master.standard_normal(size=cur.shape) * 0.02
                nz[hi] *= 3.0
                cur2 = _norm(cur + nz)
                cur2 = _smooth_opt(cur2, 600, 0.025, 0.003, 0.06, 0.003)
                dl3 = _time.time() + 1.0
                cur2, v2 = _lp_polish(cur2, dl3)
                if v2 < vcur:
                    cur, vcur = cur2, v2
                else:
                    # keep kicked version as separate endpoint if diverse
                    endpoints.append((v2, cur2))
            endpoints.append((float(_max_cos(cur)), cur.copy()))
            if float(_max_cos(cur)) < best_v:
                best_v = float(_max_cos(cur)); best_x = cur.copy()
            seen_hashes.add(h)

    # ---------- shrink-and-regrow trajectories ----------
    for traj in range(2):
        if time_left() < 25.0:
            break
        rng = np.random.default_rng(1000 + traj * 77)
        # identify rattler: point with smallest row-max
        G = base_x @ base_x.T
        np.fill_diagonal(G, -2.0)
        rowmax = np.max(G, axis=1)
        rattler = int(np.argmin(rowmax))
        # delete rattler + 10 random points
        pool = [i for i in range(n0) if i != rattler]
        chosen = rng.choice(pool, size=10, replace=False).tolist()
        dels = sorted([rattler] + chosen)
        X = np.delete(base_x, dels, axis=0)  # 142
        X = _norm(X)
        # fully re-optimise at 142
        X = _smooth_opt(X, 2200, 0.04, 0.0025, 0.08, 0.0025)
        dl = _time.time() + 2.5
        if dl > EXPLORE_END - 12.0:
            dl = EXPLORE_END - 12.0
        if dl > _time.time() + 0.3:
            X, _ = _lp_polish(X, dl)
        # greedily regrow to 153
        for k in range(11):
            if _time.time() > EXPLORE_END - 6.0:
                break
            hc, hv = _largest_hole_insert(X, rng, n_cand=30000)
            X = np.vstack([X, hc.reshape(1, d)])
            X = _norm(X)
            # brief joint polish after each addition
            X = _smooth_opt(X, 350, 0.02, 0.003, 0.05, 0.003)
            dl2 = _time.time() + 0.55
            X, _ = _lp_polish(X, dl2, radius_init=7e-4, cap=2000)
        if X.shape[0] == 153:
            dl = _time.time() + 1.5
            if dl < EXPLORE_END:
                X, vx = _lp_polish(X, dl)
            else:
                vx = _max_cos(X)
            h = _graph_hash(X)
            if h not in seen_hashes or vx < best_v:
                endpoints.append((float(vx), X.copy()))
            else:
                endpoints.append((float(vx), X.copy()))
            if float(vx) < best_v:
                best_v = float(vx); best_x = X.copy()
            seen_hashes.add(h)

    # pick best endpoint
    for v, X in endpoints:
        if v < best_v:
            best_v = v; best_x = X.copy()
    # fallback to base if nothing better and time remains: try one extra diverse kick
    # reserve final full polish
    try:
        best_x, best_v = _lp_polish(best_x, HARD_END - 1.0, radius_init=1e-3, cap=2500)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
