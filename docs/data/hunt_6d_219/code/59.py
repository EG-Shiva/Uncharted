import time
import numpy as np
from scipy.optimize import minimize

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 125.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(12345)

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
    x_best = _norm(x_best)
    best_val = _max_cos(x_best)

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 10000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def refine_void(v, Others, iters=80):
        v = v / max(1e-12, float(np.linalg.norm(v)))
        best_v = v.copy()
        try:
            best_m = float(np.max(Others @ v))
        except Exception:
            return v
        lr = 0.08
        cur = v.copy()
        for _ in range(iters):
            dots = Others @ cur
            m = float(np.max(dots))
            tau = 0.004
            w = np.exp((dots - m) / tau)
            w[dots < m - 0.02] = 0.0
            s = float(np.sum(w))
            if not np.isfinite(s) or s <= 0:
                break
            w = w / s
            g = w @ Others
            gt = g - float(g @ cur) * cur
            nm = float(np.linalg.norm(gt))
            if not np.isfinite(nm) or nm < 1e-14:
                break
            step = lr / max(nm, 1e-12) * 0.3
            nxt = cur - step * gt
            nl = float(np.linalg.norm(nxt))
            if not np.isfinite(nl) or nl < 1e-12:
                break
            nxt = nxt / nl
            mv = float(np.max(Others @ nxt))
            if mv < m + 1e-4:
                cur = nxt
                if mv < best_m:
                    best_m = mv
                    best_v = nxt.copy()
            else:
                lr *= 0.7
                if lr < 1e-4:
                    break
        return best_v

    def greedy_fill(Xkeep, k_new, rs):
        cur = Xkeep.copy()
        outs = []
        for _ in range(k_new):
            S = 4000
            C = rs.randn(S, d)
            C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            M = C @ cur.T
            mx = M.max(axis=1)
            idx = np.argsort(mx)[:8]
            bv = 1e9
            bc = C[idx[0]]
            for b in idx:
                cc = refine_void(C[b], cur, iters=60)
                vv = float(np.max(cur @ cc))
                if vv < bv:
                    bv = vv
                    bc = cc
            outs.append(bc)
            cur = np.vstack([cur, bc.reshape(1, d)])
        return np.array(outs)

    def row_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        m = float(rm.max())
        deg = (G > m - 1e-3).sum(axis=1)
        return rm, deg, m, G

    x_cur = x_best.copy()
    cur_val = best_val
    it = 0
    while time.time() < HOP_END and (time.time() - t0) < BUDGET - 30:
        it += 1
        if it > 120:
            break
        r = rng.rand()
        use_best_base = rng.rand() < 0.35
        base = x_best.copy() if use_best_base else x_cur.copy()
        try:
            rm, deg, mglob, Gfull = row_stats(base)
        except Exception:
            continue
        order_max = np.argsort(-rm)
        try:
            if r < 0.55:
                k_kill = int(rng.randint(2, 7))
                cand = np.unique(np.concatenate([order_max[:20], np.argsort(-deg)[:12]]))
                scores = rm[cand] + 0.002 * deg[cand].astype(float)
                pick = np.argsort(-scores)[:k_kill]
                victims = cand[pick].tolist()
                ratt = int(np.argmin(rm))
                if ratt in victims and len(victims) > 1:
                    victims = [vv for vv in victims if vv != ratt]
                    # refill to k_kill
                    for c in cand:
                        if int(c) not in victims and int(c) != ratt:
                            victims.append(int(c))
                            if len(victims) >= k_kill:
                                break
                    victims = victims[:k_kill]
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[np.array(victims, dtype=int)] = False
                Xkeep = base[keep_mask]
                rs = np.random.RandomState(10000 + it * 97)
                newpts = greedy_fill(Xkeep, len(victims), rs)
                Xn0 = base.copy()
                Xn0[np.array(victims, dtype=int)] = newpts
                beta = 250.0 if rng.rand() < 0.5 else 450.0
                Xn = run_lbfgs(Xn0, beta, maxiter=45)
                vn = _max_cos(Xn)
                if vn < best_val:
                    best_val = vn
                    x_best = Xn.copy()
                    x_cur = Xn.copy()
                    cur_val = vn
                elif vn < cur_val + 0.0015 and rng.rand() < 0.45:
                    x_cur = Xn.copy()
                    cur_val = vn
                elif vn < best_val + 0.001 and rng.rand() < 0.12:
                    x_cur = Xn.copy()
                    cur_val = vn
            elif r < 0.82:
                seed = int(order_max[0] if rng.rand() < 0.6 else rng.choice(order_max[:20]))
                row = Gfull[seed].copy()
                nn = np.argsort(-row)[:60]
                k2 = int(rng.randint(12, 35))
                idx = np.unique(nn[:k2])
                cur = base.copy()
                pert = rng.randn(len(idx), d)
                pert -= (np.sum(pert * cur[idx], axis=1, keepdims=True)) * cur[idx]
                sig = 0.25 + 0.45 * rng.rand()
                cur[idx] = _norm(cur[idx] + sig * pert)
                gj = rng.randn(n, d)
                gj -= (np.sum(gj * cur, axis=1, keepdims=True)) * cur
                cur = _norm(cur + 0.004 * gj)
                beta = 200.0 if rng.rand() < 0.5 else 350.0
                Xn = run_lbfgs(cur, beta, maxiter=45)
                vn = _max_cos(Xn)
                if vn < best_val:
                    best_val = vn
                    x_best = Xn.copy()
                    x_cur = Xn.copy()
                    cur_val = vn
                elif vn < cur_val and rng.rand() < 0.6:
                    x_cur = Xn.copy()
                    cur_val = vn
            else:
                ratt = int(np.argmin(rm))
                dots = base @ base[ratt]
                nb = np.argsort(-dots)[:int(rng.randint(6, 14))]
                # exclude ratt itself if included
                nb = nb[nb != ratt][:10]
                cur = base.copy()
                mask = np.ones(n, dtype=bool)
                mask[ratt] = False
                Xrest = cur[mask]
                rs = np.random.RandomState(20000 + it * 131)
                C = rs.randn(2500, d)
                C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
                mx = (C @ Xrest.T).max(axis=1)
                bc0 = C[int(np.argmin(mx))]
                bc = refine_void(bc0, Xrest, iters=60)
                cur[ratt] = bc
                if len(nb) > 0:
                    pert = rng.randn(len(nb), d)
                    pert -= (np.sum(pert * cur[nb], axis=1, keepdims=True)) * cur[nb]
                    cur[nb] = _norm(cur[nb] + (0.15 + 0.25 * rng.rand()) * pert)
                Xn = run_lbfgs(cur, 300.0, maxiter=45)
                vn = _max_cos(Xn)
                if vn < best_val:
                    best_val = vn
                    x_best = Xn.copy()
                    x_cur = Xn.copy()
                    cur_val = vn
                elif vn < cur_val + 0.001 and rng.rand() < 0.4:
                    x_cur = Xn.copy()
                    cur_val = vn
        except Exception:
            continue

    # final sharpen best
    try:
        if time.time() - t0 < BUDGET - 12:
            x_best = run_lbfgs(x_best, 250.0, maxiter=60)
            best_val = _max_cos(x_best)
        if time.time() - t0 < BUDGET - 8:
            x_best = run_lbfgs(x_best, 600.0, maxiter=60)
            best_val = _max_cos(x_best)
        if time.time() - t0 < BUDGET - 4:
            x_best = run_lbfgs(x_best, 1000.0, maxiter=40)
            best_val = _max_cos(x_best)
    except Exception:
        pass
    x_best = _norm(x_best)
    # safety: replace any bad rows
    try:
        nr = np.linalg.norm(x_best, axis=1)
        bad = ~(np.isfinite(nr)) | (nr < 1e-9)
        if np.any(bad):
            rr = np.random.RandomState(0).randn(int(np.sum(bad)), d)
            x_best[bad] = _norm(rr)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
