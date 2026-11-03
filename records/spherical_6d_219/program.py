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

def _per_max_deg(X, m=None, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pm = G.max(axis=1)
    if m is None:
        m = float(pm.max())
    deg = np.sum(G > m - tol, axis=1)
    return pm, deg, float(m), G

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.RandomState(20261103)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

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

    def run_lbfgs(x0, beta, maxiter):
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

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=10):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
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
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3200:
                k2 = 3200
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
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
        return bx, best

    def void_for(others, rng_, n_cand=4000):
        C = _norm(rng_.randn(n_cand, d))
        D = C @ others.T
        worst = D.max(axis=1)
        bi = int(np.argmin(worst))
        return C[bi]

    def refine_void(v, others, iters=60):
        v = v / max(1e-12, np.linalg.norm(v))
        lr = 0.08
        best = float(np.max(others @ v))
        bv = v.copy()
        for _ in range(iters):
            dots = others @ v
            m = float(np.max(dots))
            tau = 0.003
            w = np.exp((dots - m) / tau)
            w[dots < m - 0.015] = 0.0
            g = w @ others
            gt = g - float(g @ v) * v
            nm = float(np.linalg.norm(gt))
            if nm < 1e-14 or not np.isfinite(nm):
                break
            vn = v - (lr / nm) * gt * 0.3
            vn = vn / max(1e-12, np.linalg.norm(vn))
            mv = float(np.max(others @ vn))
            if mv < m:
                v = vn
                if mv < best:
                    best = mv
                    bv = vn.copy()
            else:
                lr *= 0.7
                if lr < 1e-4:
                    break
        return bv

    # quick LP squeeze of incumbent if time allows
    try:
        if time.time() - t0 < BUDGET - 100:
            xb2, vb2 = lp_polish(x_best, min(t0 + 12.0, time.time() + 5.0), r_init=8e-4, max_steps=8)
            if vb2 < best_val:
                best_val = vb2
                x_best = xb2.copy()
    except Exception:
        pass

    hop = 0
    # main basin-hopping loop
    while time.time() - t0 < BUDGET - 12.0:
        hop += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 13:
            break
        try:
            pm, deg, mcur, _ = _per_max_deg(x_best)
        except Exception:
            pm, deg, mcur = None, None, best_val
        base = x_best.copy()
        cur = base.copy()
        # choose block size: mostly 4-12, occasionally large 20-38
        r = rng.rand()
        if r < 0.15:
            k = int(rng.randint(18, 36))
        elif r < 0.45:
            k = int(rng.randint(8, 16))
        else:
            k = int(rng.randint(3, 9))
        # select victims: rattlers + worst per-max + high-degree hubs + random
        victims = set()
        if deg is not None:
            ratt = list(np.where(deg == 0)[0])
            for idx in ratt[:2]:
                victims.add(int(idx))
            order_worst = np.argsort(-pm)
            order_hub = np.argsort(-deg)
            nw = min(len(order_worst), k)
            # take some worst
            for j in range(min(3, k)):
                victims.add(int(order_worst[(hop + j * 3) % n]))
            # take some hubs
            for j in range(min(2, k)):
                victims.add(int(order_hub[(hop * 2 + j * 5) % n]))
        while len(victims) < k:
            victims.add(int(rng.randint(0, n)))
        victims = list(victims)[:k]
        mode = rng.rand()
        if mode < 0.55:
            # transplant into maximin voids sequentially
            # remove first, then insert one by one to spread
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            others = cur[keep_mask].copy()
            new_pts = []
            for _ in victims:
                v = void_for(others, rng, n_cand=4000)
                v = refine_void(v, others, iters=50)
                new_pts.append(v)
                others = np.vstack([others, v.reshape(1, d)])
            for vi, vp in zip(victims, new_pts):
                cur[vi] = vp
            cur = _norm(cur)
        elif mode < 0.8:
            # large Gaussian tangent kick on block
            amp = 0.35 + 0.4 * rng.rand()
            for vi in victims:
                g = rng.randn(d)
                g -= float(g @ cur[vi]) * cur[vi]
                gn = float(np.linalg.norm(g))
                if gn < 1e-12:
                    continue
                g /= gn
                ang = amp * (0.5 + rng.rand())
                cur[vi] = cur[vi] * np.cos(ang) + g * np.sin(ang)
            cur = _norm(cur)
        else:
            # full random re-init of block
            for vi in victims:
                cur[vi] = _norm(rng.randn(1, d))[0]
            cur = _norm(cur)
        # occasional global jitter to help topology
        if rng.rand() < 0.25:
            gj = rng.randn(n, d) * 0.015
            gj -= (gj * cur).sum(1, keepdims=True) * cur
            cur = _norm(cur + gj)
        # two-stage soft-max re-jam
        if remain > 40:
            cur = run_lbfgs(cur, 130.0, 55)
            if time.time() - t0 > BUDGET - 12:
                v = _max_cos(cur)
                if v < best_val:
                    best_val = v
                    x_best = cur.copy()
                break
            cur = run_lbfgs(cur, 420.0, 55)
        else:
            cur = run_lbfgs(cur, 150.0, 45)
            cur = run_lbfgs(cur, 500.0, 45)
        v = _max_cos(cur)
        # selective LP polish only for promising candidates
        if v < best_val + 0.0015 and time.time() - t0 < BUDGET - 14:
            try:
                cur2, v2 = lp_polish(cur, time.time() + 2.5, r_init=8e-4, max_steps=7)
                if v2 < v:
                    cur, v = cur2, v2
            except Exception:
                pass
        if v < best_val:
            best_val = v
            x_best = cur.copy()
        # else discard (greedy); occasionally keep near-best as base for next hop? stay greedy on best
    # final squeeze
    try:
        if time.time() - t0 < BUDGET - 3:
            x_best, best_val2 = lp_polish(x_best, t0 + BUDGET - 1.0, r_init=9e-4, max_steps=12)
            best_val = min(best_val, best_val2)
            # one last lbfgs at high beta
            if time.time() - t0 < BUDGET - 4:
                cand = run_lbfgs(x_best, 600.0, 60)
                vc = _max_cos(cand)
                if vc < best_val:
                    x_best = cand
                    best_val = vc
                    x_best, best_val = lp_polish(x_best, t0 + BUDGET - 0.5, r_init=6e-4, max_steps=6)
    except Exception:
        pass
    return _norm(x_best)
