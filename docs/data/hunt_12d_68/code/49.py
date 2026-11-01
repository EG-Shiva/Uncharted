import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

N = 68
D = 12

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _make_full_fg(beta):
    IU = np.triu_indices(N, 1)
    def fg(yf):
        Y = yf.reshape(N, D)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((N, N))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full(x0, beta, maxiter):
    fg = _make_full_fg(float(beta))
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': int(maxiter), 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': int(maxiter)*2})
        Y = r.x.reshape(N, D)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _make_part_fg(F, k, beta, IUk):
    m = F.shape[0]
    def fg(yf):
        Y = yf.reshape(k, D)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        if k > 1:
            g_int = (X @ X.T)[IUk]
        else:
            g_int = np.empty(0)
        g_cross = (X @ F.T).ravel()
        if g_int.size > 0:
            mx = float(max(float(g_int.max()), float(g_cross.max())))
            e_int = np.exp(beta * (g_int - mx))
            e_cr = np.exp(beta * (g_cross - mx))
            S = float(e_int.sum() + e_cr.sum()) + 1e-300
            Fv = mx + np.log(S) / beta
            w_int = e_int / S
            w_cr = e_cr / S
            P = np.zeros((k, k))
            P[IUk] = w_int
            P = P + P.T
            GX = P @ X + w_cr.reshape(k, m) @ F
        else:
            mx = float(g_cross.max())
            e_cr = np.exp(beta * (g_cross - mx))
            S = float(e_cr.sum()) + 1e-300
            Fv = mx + np.log(S) / beta
            w_cr = e_cr / S
            GX = w_cr.reshape(k, m) @ F
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return Fv, GY.ravel()
    return fg

def _run_part(Y0, F, betas_iters):
    k = Y0.shape[0]
    if k == 0:
        return Y0
    IUk = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    y = (Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)).ravel().copy()
    for beta, it in betas_iters:
        fg = _make_part_fg(F, k, float(beta), IUk)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(it), 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': int(it)*2})
            y = r.x
        except Exception:
            break
    Y = y.reshape(k, D)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _lp_polish(x, deadline, r0=6e-4, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.3 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            od = np.argsort(-csel)[:cap]
            I, J, csel = I[od], J[od], csel[od]
            p = cap
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d), J[:, None]*d + np.arange(d), np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        ds = res.x[:-1].reshape(n, d)
        y = x + ds
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + ds * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    n, d = N, D
    # load incumbent
    inc = None
    inc_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if v < inc_val:
                inc_val = v
                inc = a.copy()
        except Exception:
            continue
    rng = np.random.default_rng(987654321)
    if inc is None:
        inc = rng.standard_normal((n, d))
        inc /= np.linalg.norm(inc, axis=1, keepdims=True)
        inc_val = _max_cos(inc)
    x_best = inc.copy()
    best_val = float(inc_val)
    cur = inc.copy()
    cur_val = float(inc_val)

    # 200k candidate pool (Gaussian -> uniform on sphere)
    rng_pool = np.random.default_rng(1234567)
    NCAND = 200000
    pool = rng_pool.standard_normal((NCAND, d))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)

    # geographically distinct seeds via farthest-point sampling on current best
    def fps_order(X, k):
        G = X @ X.T
        # start from max per-max (most jammed)
        pm = _per_max(X)
        order = [int(np.argmax(pm))]
        best_sim = G[order[0]].copy()
        for _ in range(1, k):
            best_sim[order] = 2.0
            nxt = int(np.argmin(best_sim))
            order.append(nxt)
            best_sim = np.minimum(best_sim, G[nxt])
        return order

    try:
        seeds = fps_order(x_best, 10)
    except Exception:
        seeds = list(range(10))

    sizes = [18, 14, 22, 16, 20, 15, 21, 17, 19, 14]
    part_chain = [(100.0, 120), (300.0, 120), (900.0, 120)]

    si = 0
    rnd = 0
    while time.time() < t0 + TLIM - 12:
        seed = int(seeds[si % len(seeds)])
        kdel = int(sizes[(si + rnd) % len(sizes)])
        kdel = max(14, min(22, kdel))
        si += 1
        if si % len(seeds) == 0:
            rnd += 1
            # refresh order from current
            try:
                seeds = fps_order(cur, 10)
            except Exception:
                pass
        # cluster = seed + nearest neighbours in cosine
        try:
            dots = cur[seed] @ cur.T
            order_nn = np.argsort(-dots)
            # ensure seed first
            del_idx = order_nn[:kdel]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            F = cur[keep_mask].copy()  # (m,d) frozen, m=46..54
            m = F.shape[0]
            # greedy refill from pool: minimise max dot vs frozen + already inserted
            core_max = (pool @ F.T).max(axis=1)  # (NCAND,)
            cur_max = core_max.copy()
            inserted = np.empty((kdel, d))
            for t in range(kdel):
                j = int(np.argmin(cur_max))
                v = pool[j].copy()
                inserted[t] = v
                # update cur_max with dots to v
                dv = pool @ v  # (NCAND,)
                # avoid re-picking same: dv[j]=1 so cur_max[j]=1
                np.maximum(cur_max, dv, out=cur_max)
            # staged partial L-BFGS of inserted vs frozen core
            Yopt = _run_part(inserted, F, part_chain)
            # assemble full config
            x_new = np.empty((n, d))
            x_new[keep_mask] = F
            x_new[~keep_mask] = Yopt
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            # short full anneal (unfrozen)
            if time.time() > t0 + TLIM - 25:
                # no time for anneal, just evaluate
                pass
            else:
                x_new = _run_full(x_new, 250.0, 60)
                if time.time() < t0 + TLIM - 18:
                    x_new = _run_full(x_new, 900.0, 60)
            # LP polish with short budget
            lp_dead = min(time.time() + 4.0, t0 + TLIM - 8)
            if lp_dead > time.time() + 0.8:
                x_new = _lp_polish(x_new, lp_dead, r0=6e-4, cap=2000)
            v_new = _max_cos(x_new)
            if v_new < best_val:
                best_val = float(v_new)
                x_best = x_new.copy()
                cur = x_new.copy()
                cur_val = float(v_new)
            else:
                # allow drift to new basin if not too bad (topology change)
                if v_new < cur_val + 0.015:
                    cur = x_new.copy()
                    cur_val = float(v_new)
                    # occasional restart from best to intensify
                    if rng.random() < 0.25:
                        cur = x_best.copy()
                        cur_val = best_val
                else:
                    # too bad, revert to best with small kick for next try
                    cur = x_best.copy()
                    cur_val = best_val
        except Exception:
            continue
        # early exit if time low
        if time.time() > t0 + TLIM - 10:
            break

    # final polish of best with remaining time
    try:
        if time.time() < t0 + TLIM - 3:
            # quick LSE touch
            x_best = _run_full(x_best, 1200.0, 80)
        if time.time() < t0 + TLIM - 2:
            x_best = _lp_polish(x_best, t0 + TLIM - 0.5, r0=4e-4, cap=2200)
            v = _max_cos(x_best)
            if v < best_val:
                best_val = v
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
