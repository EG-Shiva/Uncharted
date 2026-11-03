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
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(7)

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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def anneal(x0, betas, iters):
        X = _norm(x0)
        for b, mi in zip(betas, iters):
            if time.time() - t0 > BUDGET - 12:
                break
            X = run_lbfgs(X, b, mi)
        return X

    def lp_polish(x, deadline, r_init=1.2e-3, max_steps=10):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5 and steps < max_steps:
            steps += 1
            if radius < 1e-14:
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
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
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
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    def rowmax_deg(X, tol=1e-9):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        m = float(rm.max())
        deg = np.sum(G > m - tol, axis=1)
        return rm, deg, m, G

    def best_void(Y, rng_loc, n_cand=8000):
        # Y: (m,d) fixed set, find point minimizing max dot
        C = rng_loc.randn(n_cand, d)
        C = _norm(C)
        dots = C @ Y.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        p = C[bi:bi+1].copy()
        # local refine: tangential GD on softmax
        beta = 250.0
        step = 0.05
        for _ in range(60):
            dv = (p @ Y.T).ravel()
            mm = float(dv.max())
            w = np.exp(beta * (dv - mm))
            w = w / (w.sum() + 1e-300)
            g = (w @ Y).reshape(1, d)
            gt = g - (g @ p.T) * p
            gn = float(np.linalg.norm(gt))
            if gn < 1e-12:
                break
            p = _norm(p - (step / (1.0 + gn)) * gt * min(1.0, 2.0 / max(1.0, gn * 5)))
            step *= 0.985
        return _norm(p).ravel()

    cur = x_best.copy()
    cur_val = best_val

    # quick initial LP squeeze on incumbent (few steps)
    try:
        if time.time() - t0 < BUDGET - 60:
            Xp, vp = lp_polish(x_best, deadline=time.time() + 4.0, max_steps=8)
            if vp < best_val:
                x_best, best_val = Xp, vp
                cur, cur_val = Xp.copy(), vp
    except Exception:
        pass

    hop = 0
    # basin hopping until ~ BUDGET-18
    while time.time() - t0 < BUDGET - 18 and hop < 200:
        hop += 1
        rm, deg, m, G = rowmax_deg(cur, tol=1e-9)
        ratt = np.where(deg == 0)[0]
        order_worst = np.argsort(-rm)
        order_deg = np.argsort(-deg)
        mode = hop % 5
        victims = []
        if mode == 0:
            # two worst row-max
            victims = [int(order_worst[0]), int(order_worst[1])]
        elif mode == 1:
            # rattler + worst
            if len(ratt) > 0:
                victims = [int(ratt[0]), int(order_worst[0])]
            else:
                victims = [int(order_worst[0]), int(order_worst[1])]
            if hop % 10 == 1 and len(victims) == 2:
                victims.append(int(order_deg[rng.randint(0, 10)]))
        elif mode == 2:
            # hubs: high degree
            k = 2 + (hop % 2)
            victims = [int(order_deg[i]) for i in range(k)]
            # ensure distinct + include a random mid point to break symmetry
            victims[0] = int(order_worst[0])
        elif mode == 3:
            # random 12-point shake, no kill
            victims = []
        else:
            # single worst + random
            victims = [int(order_worst[0]), int(rng.randint(0, n))]
        victims = sorted(set([v for v in victims if 0 <= v < n]))
        # build candidate
        Xc = cur.copy()
        rng_loc = np.random.RandomState(1000 + hop * 131)
        if len(victims) > 0:
            keep_idx = [i for i in range(n) if i not in set(victims)]
            Ykeep = Xc[keep_idx]
            # sequential void fill
            new_pts = []
            Ygrow = Ykeep.copy()
            for _ in victims:
                p = best_void(Ygrow, rng_loc, n_cand=6000)
                new_pts.append(p)
                Ygrow = np.vstack([Ygrow, p.reshape(1, d)])
            for v, p in zip(victims, new_pts):
                Xc[v] = p
        else:
            pass
        # add tangential noise to diversify
        if mode == 3:
            idx = rng_loc.choice(n, size=14, replace=False)
            sig = 0.12 if (hop % 2 == 0) else 0.05
            Nz = rng_loc.randn(len(idx), d)
            Nz -= (Nz * Xc[idx]).sum(1, keepdims=True) * Xc[idx]
            nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
            Xc[idx] = _norm(Xc[idx] + sig * Nz / nn)
        else:
            # small jitter on victims neighbourhood: jitter top-close points slightly
            sig = 0.01
            Nz = rng_loc.randn(n, d)
            Nz -= (Nz * Xc).sum(1, keepdims=True) * Xc
            nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
            # only jitter non-victims a tiny bit to allow topology change
            Xc = _norm(Xc + sig * Nz / nn * 0.3)
            # restore victims exactly (they are voids)
            # already jittered; renormalize
            Xc = _norm(Xc)

        # annealed re-jam: low beta to allow moves, then high beta
        if mode == 3:
            Xa = anneal(Xc, [90.0, 300.0], [50, 45])
        else:
            Xa = anneal(Xc, [110.0, 350.0], [55, 50])
        va = _max_cos(Xa)
        # opportunistic extra sharpen
        if va < best_val + 0.002 and time.time() - t0 < BUDGET - 20:
            Xb = run_lbfgs(Xa, 600.0, 35)
            vb = _max_cos(Xb)
            if vb < va:
                Xa, va = Xb, vb
        # LP if promising
        if va < best_val + 0.0015 and time.time() - t0 < BUDGET - 18:
            try:
                Xp, vp = lp_polish(Xa, deadline=time.time() + 2.5, max_steps=6)
                if vp < va:
                    Xa, va = Xp, vp
            except Exception:
                pass
        if va < best_val - 1e-12:
            x_best = Xa.copy()
            best_val = va
            cur, cur_val = Xa.copy(), va
        elif va < cur_val + 0.004:
            # exploratory accept (esp. if rattler integrated)
            _, deg2, _, _ = rowmax_deg(Xa)
            nrat2 = int(np.sum(deg2 == 0))
            nratC = int(np.sum(deg == 0))
            # prefer balanced (fewer rattlers)
            if nrat2 <= nratC or rng.rand() < 0.35:
                cur, cur_val = Xa.copy(), va
        else:
            # revert to best with small shake to avoid stagnation
            if hop % 7 == 0:
                Nz = rng.randn(n, d)
                Nz -= (Nz * x_best).sum(1, keepdims=True) * x_best
                nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
                cur = _norm(x_best + 0.02 * Nz / nn)
                cur_val = _max_cos(cur)
        # early exit if big win
        if best_val < 0.690:
            break

    # final squeeze on global best
    try:
        if time.time() - t0 < BUDGET - 10:
            Xf = run_lbfgs(x_best, 350.0, 60)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = Xf, vf
            Xf2 = run_lbfgs(x_best, 700.0, 40)
            vf2 = _max_cos(Xf2)
            if vf2 < best_val:
                x_best, best_val = Xf2, vf2
        Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 0.8, max_steps=25)
        if vp < best_val:
            x_best, best_val = Xp, vp
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
