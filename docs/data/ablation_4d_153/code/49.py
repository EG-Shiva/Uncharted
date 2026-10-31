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

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.9):
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
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _guided_opt(x0, a, b, c, d, steps, tau, lr, boost, mom=0.9):
    x = x0.copy()
    n, dd = x.shape
    vel = np.zeros_like(x)
    for t in range(steps):
        G = x @ x.T
        # m over all except break pair
        G[a, b] = -np.inf
        G[b, a] = -np.inf
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        # restore for weights (need original dots)
        # recompute dot for break quickly: dot_ab = x[a]@x[b]
        # build W from G (with -inf at break/diag)
        W = np.exp((G - m) / tau)
        # break already ~0, diag ~0
        # boost form pair
        W[c, d] *= boost
        W[d, c] *= boost
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2500, r0=1e-3):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 300:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
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
    return x, best

def _hole_insert(X, rng, n_cand=20000, batch=10000):
    n, d = X.shape
    best_c = None
    best_s = np.inf
    nb = max(1, n_cand // batch)
    for _ in range(nb):
        C = rng.standard_normal(size=(batch, d))
        C = _norm(C)
        S = C @ X.T
        sm = S.max(axis=1)
        i = int(np.argmin(sm))
        if float(sm[i]) < best_s:
            best_s = float(sm[i])
            best_c = C[i].copy()
    return np.vstack([X, best_c.reshape(1, -1)])

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = HARD_END - 26.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    best_x = None
    best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = _norm(np.asarray(a, dtype=float))
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v
                        best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

    attempt = 0
    # cache slack ordering of global best, refresh after improvement or every 10 tries
    Gc = global_best_x @ global_best_x.T
    cc = Gc[iu]
    mc = float(cc.max())
    slack = mc - cc
    order = np.argsort(slack)
    # number of contacts
    ncon = int(np.sum(slack < 1e-7))
    if ncon < 100:
        ncon = 450
    deg = None
    try:
        B = (Gc > (mc - 1e-6))
        np.fill_diagonal(B, False)
        deg = np.sum(B, axis=1)
    except Exception:
        deg = np.zeros(n)

    def refresh_cache():
        nonlocal Gc, cc, mc, slack, order, ncon, deg
        Gc = global_best_x @ global_best_x.T
        cc = Gc[iu]
        mc = float(cc.max())
        slack = mc - cc
        order = np.argsort(slack)
        ncon = int(np.sum(slack < 1e-7))
        if ncon < 50:
            ncon = 450
        try:
            Bb = (Gc > (mc - 1e-6))
            np.fill_diagonal(Bb, False)
            deg = np.sum(Bb, axis=1)
        except Exception:
            deg = np.zeros(n)

    while time.time() < EXPLORE_END:
        attempt += 1
        if attempt % 10 == 1:
            refresh_cache()
        mode = attempt % 5  # 0,1,3 swap; 2 reinsert; 4 patch
        try:
            if mode in (0, 1, 3):
                # ---- contact-swap guided ----
                # pick break among contacts
                con_order = order[:max(ncon, 100)]
                bi = int(rng.integers(0, len(con_order)))
                bidx = int(con_order[bi])
                a = int(Iall[bidx]); b = int(Jall[bidx])
                # near pool: next ~400
                lo = ncon
                hi = min(len(order), ncon + 500)
                near_pool = order[lo:hi]
                # sharing candidates
                # limit search to first 300 of near_pool for speed
                cand_pool = near_pool[:300] if len(near_pool) > 300 else near_pool
                Ic = Iall[cand_pool]; Jc = Jall[cand_pool]
                share_mask = (Ic == a) | (Jc == a) | (Ic == b) | (Jc == b)
                if np.any(share_mask) and rng.random() < 0.75:
                    s_idx = np.where(share_mask)[0]
                    # pick among 5 smallest slack sharing (they are already sorted)
                    take = s_idx[:min(5, len(s_idx))]
                    pick = int(rng.choice(take)) if len(take) > 1 else int(take[0])
                    fidx = int(cand_pool[pick])
                else:
                    # random among top 60 imminent
                    top = near_pool[:min(60, len(near_pool))]
                    fidx = int(rng.choice(top))
                c_ = int(Iall[fidx]); d_ = int(Jall[fidx])
                if len({a, b, c_, d_}) < 3 or (min(a, b) == min(c_, d_) and max(a, b) == max(c_, d_)):
                    continue
                tau = float(rng.uniform(0.0025, 0.006))
                lr = float(rng.uniform(0.04, 0.085))
                boost = float(rng.uniform(5.0, 12.0))
                steps = int(rng.integers(220, 360))
                Y = _guided_opt(global_best_x, a, b, c_, d_, steps=steps, tau=tau, lr=lr, boost=boost)
                # short unbiased settle
                Y = _smooth_opt(Y, steps=120, tau0=0.008, tau1=0.0015, lr0=0.05, lr1=0.005)
                v_sm = _max_cos(Y)
                if v_sm > global_best_v + 8e-4:
                    continue
                Yp, vp = _lp_polish(Y, time.time() + 0.8, cap=2000, r0=8e-4)
                if vp < global_best_v - 1e-12:
                    global_best_x = Yp.copy()
                    global_best_v = vp
                    refresh_cache()
            elif mode == 2:
                # ---- excise + regrow one vertex ----
                if deg is not None and np.max(deg) > 0:
                    # prefer high-degree vertices
                    o = np.argsort(-deg)
                    topv = o[:min(20, n)]
                    v = int(rng.choice(topv))
                else:
                    v = int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool); mask[v] = False
                Y = global_best_x[mask].copy()  # 152
                # relax hole
                Y = _smooth_opt(Y, steps=220, tau0=0.015, tau1=0.002, lr0=0.09, lr1=0.006)
                Y = _hole_insert(Y, rng, n_cand=20000, batch=10000)
                Y = _smooth_opt(Y, steps=180, tau0=0.010, tau1=0.0015, lr0=0.07, lr1=0.005)
                v_sm = _max_cos(Y)
                if v_sm > global_best_v + 8e-4:
                    continue
                Yp, vp = _lp_polish(Y, time.time() + 0.8, cap=2000, r0=8e-4)
                if vp < global_best_v - 1e-12:
                    global_best_x = Yp.copy()
                    global_best_v = vp
                    refresh_cache()
            else:
                # ---- coherent patch shear ----
                center = int(rng.integers(0, n))
                dots = global_best_x @ global_best_x[center]
                k = int(rng.integers(10, 16))
                idx = np.argsort(-dots)[:k + 1]
                ang = np.arccos(np.clip(dots[idx], -1.0, 1.0))
                sig_ang = float(rng.uniform(0.25, 0.45))
                w = np.exp(-(ang ** 2) / (2 * sig_ang ** 2))
                # random tangent direction at center
                ru = rng.standard_normal(size=(d,))
                ru -= float(ru @ global_best_x[center]) * global_best_x[center]
                nru = np.linalg.norm(ru)
                if nru < 1e-9:
                    continue
                ru /= nru
                amp = float(rng.uniform(0.08, 0.18))
                Y = global_best_x.copy()
                for ii, wi in zip(idx, w):
                    ui = ru - float(ru @ Y[ii]) * Y[ii]
                    Y[ii] = Y[ii] + amp * float(wi) * ui
                Y = _norm(Y)
                Y = _smooth_opt(Y, steps=220, tau0=0.012, tau1=0.0018, lr0=0.08, lr1=0.005)
                v_sm = _max_cos(Y)
                if v_sm > global_best_v + 8e-4:
                    continue
                Yp, vp = _lp_polish(Y, time.time() + 0.8, cap=2000, r0=8e-4)
                if vp < global_best_v - 1e-12:
                    global_best_x = Yp.copy()
                    global_best_v = vp
                    refresh_cache()
        except Exception:
            continue
        # safety: if time nearly up break
        if time.time() > EXPLORE_END:
            break

    # ---- final anneal + full polish ----
    try:
        if time.time() < HARD_END - 2.0:
            global_best_x = _smooth_opt(global_best_x, steps=400, tau0=0.008, tau1=0.001, lr0=0.06, lr1=0.004)
        global_best_x, global_best_v = _lp_polish(global_best_x, HARD_END - 1.0, cap=3000, r0=1e-3)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
