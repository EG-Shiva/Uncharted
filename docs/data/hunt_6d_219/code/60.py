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
    HOP_END = t0 + 140.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(20261103)

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
            W = np.zeros((n, n), dtype=float)
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

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=30):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            if time.time() - t0 > BUDGET - 0.5:
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
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return bx, best

    def greedy_void_fill(x_init, victims, n_cand=12000):
        # victims: list of indices to reinsert one-by-one
        fixed_mask = np.ones(n, dtype=bool)
        fixed_mask[np.array(victims, dtype=int)] = False
        order = list(victims)
        rng.shuffle(order)
        for w in order:
            cand = _norm(rng.randn(n_cand, d))
            # dots to all points except w (use current positions)
            D = cand @ x_init.T
            D[:, w] = -2.0
            # for victims not yet placed, ignore them (set -2)
            # we want max over fixed + already placed; simplest: mask columns
            # set columns of future victims to -2 as well except already done
            # build mask: keep fixed + placed
            # placed = victims already processed before w in this loop? approximate: keep all except future
            # To avoid complexity, just max over all except w and future victims
            # future victims are those in order after w
            # find position
            try:
                idx_w = order.index(int(w))
                future = set(order[idx_w + 1:])
                if len(future) > 0:
                    D[:, list(future)] = -2.0
            except Exception:
                pass
            worst = D.max(axis=1)
            bi = int(np.argmin(worst))
            x_init[w] = cand[bi]
        return x_init

    # light tighten of incumbent
    try:
        Xb = run_lbfgs(x_best, 300.0, 40)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val = vb
            x_best = Xb.copy()
    except Exception:
        pass

    cur_X = x_best.copy()
    cur_val = best_val
    temp0 = 0.0025
    it = 0
    while time.time() < HOP_END:
        it += 1
        if time.time() - t0 > BUDGET - 32:
            break
        elapsed = time.time() - t0
        frac = elapsed / max(1e-9, (HOP_END - t0))
        temp = temp0 * max(0.12, 1.0 - frac)
        do_cold = (it % 6 == 0)  # every 6th is cold restart
        try:
            if do_cold:
                x_init = _norm(rng.randn(n, d))
                X1 = run_lbfgs(x_init, 80.0, 35)
                v1 = _max_cos(X1)
                if v1 > best_val + 0.02 and rng.rand() < 0.5:
                    newX, newV = X1, v1
                else:
                    X2 = run_lbfgs(X1, 260.0, 45)
                    newV = _max_cos(X2)
                    newX = X2
                    if newV < best_val + 0.004 and time.time() < HOP_END:
                        X3 = run_lbfgs(newX, 650.0, 30)
                        v3 = _max_cos(X3)
                        if v3 < newV:
                            newX, newV = X3, v3
            else:
                # choose base
                rsel = rng.rand()
                if rsel < 0.70:
                    base_X = x_best
                else:
                    base_X = cur_X
                Gcur = base_X @ base_X.T
                np.fill_diagonal(Gcur, -2.0)
                rowmax = Gcur.max(axis=1)
                mcur = float(rowmax.max())
                # contact loads with loose threshold to find hubs
                thr_hub = mcur - 1.0e-3
                C = (Gcur > thr_hub).sum(axis=1)
                # strict rattlers
                thr_strict = mcur - 1.0e-7
                Cs = (Gcur > thr_strict).sum(axis=1)
                ratt = list(map(int, np.where(Cs == 0)[0]))
                order_worst = list(map(int, np.argsort(-rowmax)))
                order_hub = list(map(int, np.argsort(-C)))
                k = int(rng.randint(6, 15))
                victims = []
                for u in ratt:
                    if u not in victims:
                        victims.append(u)
                # add worst rowmax
                for u in order_worst:
                    if len(victims) >= k + len(ratt):
                        break
                    if u not in victims:
                        victims.append(u)
                    if len(victims) >= k + 2 and rng.rand() < 0.5:
                        break
                # add hubs
                nh = int(rng.randint(2, 6))
                added = 0
                for u in order_hub:
                    if added >= nh:
                        break
                    if u not in victims:
                        victims.append(u)
                        added += 1
                # cap size
                victims = victims[:18]
                x_init = base_X.copy()
                # small tangential shake of background to absorb
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig_bg = 0.005 + 0.015 * rng.rand()
                x_init = _norm(x_init + sig_bg * pert)
                # greedy void fill victims
                nc = 12000 if len(victims) > 8 else 18000
                x_init = greedy_void_fill(x_init, victims, n_cand=nc)
                # extra collective kick on crowded neighbourhood of worst point
                if rng.rand() < 0.6:
                    w0 = int(order_worst[0])
                    nn = list(map(int, np.argsort(-Gcur[w0])[:12]))
                    sub = np.array(sorted(set(nn + [w0])), dtype=int)
                    p2 = rng.randn(len(sub), d)
                    p2 -= (np.sum(p2 * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                    x_init[sub] = _norm(x_init[sub] + (0.03 + 0.06 * rng.rand()) * p2)
                # two-stage polish
                X1 = run_lbfgs(x_init, 150.0, 35)
                v1 = _max_cos(X1)
                if v1 > best_val + 0.015 and rng.rand() < 0.7:
                    newX, newV = X1, v1
                else:
                    X2 = run_lbfgs(X1, 500.0, 40)
                    newV = _max_cos(X2)
                    newX = X2
                    if newV < best_val + 0.002 and time.time() < HOP_END - 5:
                        X3 = run_lbfgs(newX, 900.0, 22)
                        v3 = _max_cos(X3)
                        if v3 < newV:
                            newX, newV = X3, v3
            # Metropolis for wanderer
            if newV < cur_val or rng.rand() < np.exp(-(newV - cur_val) / max(1e-9, temp)):
                cur_X, cur_val = newX.copy(), float(newV)
            if newV < best_val - 1e-12:
                best_val = float(newV)
                x_best = newX.copy()
                cur_X, cur_val = newX.copy(), float(newV)
        except Exception:
            continue

    # final deep polish
    try:
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 300.0, 55)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 700.0, 60)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 1400.0, 35)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
    except Exception:
        pass
    try:
        deadline = t0 + BUDGET - 0.4
        Xp, vp = lp_polish(x_best, deadline, r_init=9.0e-4, max_steps=30)
        if vp < best_val:
            best_val, x_best = vp, Xp.copy()
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
