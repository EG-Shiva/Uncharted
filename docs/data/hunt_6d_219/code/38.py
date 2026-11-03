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
    BUDGET = 168.0
    HOP_END = t0 + 128.0
    FINAL_START = t0 + 140.0
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

    def lp_polish(x, deadline, r_init=1.2e-3, max_steps=40):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            if time.time() - t0 > BUDGET - 1.0:
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

    # quick baseline tighten
    try:
        Xb = run_lbfgs(x_best, 300.0, 60)
        vb = _max_cos(Xb)
        if vb < best_val:
            best_val = vb
            x_best = Xb.copy()
    except Exception:
        pass

    cur_X = x_best.copy()
    cur_val = best_val
    pool = [(best_val, x_best.copy())]

    trial = 0
    temp0 = 0.0020
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 40:
        trial += 1
        elapsed = time.time() - t0
        frac = elapsed / max(1e-9, (HOP_END - t0))
        temp = temp0 * max(0.1, 1.0 - frac)
        # base selection: mostly best, sometimes wander
        rsel = rng.rand()
        if rsel < 0.65:
            base_X = x_best
        elif rsel < 0.85:
            base_X = cur_X
        else:
            pi = int(rng.randint(0, len(pool)))
            base_X = pool[pi][1]
        Gcur = base_X @ base_X.T
        np.fill_diagonal(Gcur, -2.0)
        rowmax = Gcur.max(axis=1)
        order = np.argsort(-rowmax)
        x_init = base_X.copy()
        mode = int(rng.randint(0, 6))
        try:
            if mode == 0:
                # destroy worst cluster: full random reinit
                k = int(rng.randint(6, 16))
                victims = list(map(int, order[:k]))
                # add neighbors of worst
                w0 = int(order[0])
                nn = list(map(int, np.argsort(-Gcur[w0])[:4]))
                for u in nn:
                    if u not in victims:
                        victims.append(u)
                victims = victims[:k+4]
                for w in victims:
                    x_init[w] = rng.randn(d)
                x_init[victims] = _norm(x_init[victims])
                # small shake of rest to absorb
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.008 * pert)
            elif mode == 1:
                # void reinsert with maximin search
                nw = int(rng.randint(2, 6))
                victims = list(map(int, order[:nw]))
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.01 * pert)
                for w in victims:
                    candv = _norm(rng.randn(6000, d))
                    dots = candv @ x_init.T
                    dots[:, w] = -2.0
                    worst = dots.max(axis=1)
                    bi = int(np.argmin(worst))
                    x_init[w] = candv[bi]
            elif mode == 2:
                # large collective block kick
                k = int(rng.randint(20, 55))
                idx = rng.choice(n, size=k, replace=False)
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig = 0.14 + 0.20 * rng.rand()
                x_init[idx] = x_init[idx] + sig * pert[idx]
                x_init[idx] = _norm(x_init[idx])
            elif mode == 3:
                # global shake medium-large
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig = 0.05 + 0.09 * rng.rand()
                x_init = _norm(x_init + sig * pert)
            elif mode == 4:
                # random rotation jump of subset + repulsion bias
                k = int(rng.randint(10, 30))
                idx = rng.choice(n, size=k, replace=False)
                # random orthogonal-ish via QR
                A = rng.randn(d, d)
                Q, _ = np.linalg.qr(A)
                if rng.rand() < 0.5:
                    # small rotation interpolation
                    alpha = 0.25 + 0.35 * rng.rand()
                    R = (1 - alpha) * np.eye(d) + alpha * Q
                    # orthonormalize
                    R, _ = np.linalg.qr(R)
                else:
                    R = Q
                    alpha = 0.4
                x_init[idx] = x_init[idx] @ R.T
                pert = rng.randn(len(idx), d)
                pert -= (np.sum(pert * x_init[idx], axis=1, keepdims=True)) * x_init[idx]
                x_init[idx] = _norm(x_init[idx] + (0.03 + 0.05 * rng.rand()) * pert)
            else:
                # worst-pair split + crowd explosion
                i = int(order[0])
                j = int(np.argmax(Gcur[i]))
                for (a, b) in [(i, j), (j, i)]:
                    xa = x_init[a]; xb = x_init[b]
                    t = xb - float(xb @ xa) * xa
                    nm = float(np.linalg.norm(t))
                    if nm > 1e-12:
                        t /= nm
                        x_init[a] = xa - (0.10 + 0.10 * rng.rand()) * t
                x_init[[i, j]] = _norm(x_init[[i, j]])
                k = int(rng.randint(15, 35))
                crowd = list(map(int, order[:k]))
                sub = np.array(sorted(set(crowd + [i, j])), dtype=int)
                pert = rng.randn(len(sub), d)
                pert -= (np.sum(pert * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                x_init[sub] = _norm(x_init[sub] + (0.08 + 0.10 * rng.rand()) * pert)
        except Exception:
            x_init = base_X.copy()

        # two-stage polish
        try:
            X1 = run_lbfgs(x_init, 140.0, 35)
            v1 = _max_cos(X1)
            # early gate: if far worse, skip deep stage with prob
            if v1 > best_val + 0.015 and rng.rand() < 0.7:
                newX, newV = X1, v1
            else:
                X2 = run_lbfgs(X1, 400.0, 45)
                newV = _max_cos(X2)
                newX = X2
                if newV < best_val + 0.002 and time.time() < HOP_END:
                    X3 = run_lbfgs(newX, 800.0, 25)
                    v3 = _max_cos(X3)
                    if v3 < newV:
                        newX, newV = X3, v3
        except Exception:
            continue

        # Metropolis update of wanderer
        if newV < cur_val or rng.rand() < np.exp(-(newV - cur_val) / max(1e-9, temp)):
            cur_X, cur_val = newX.copy(), float(newV)
        # elitist update
        if newV < best_val - 1e-12:
            best_val = float(newV)
            x_best = newX.copy()
            pool.append((best_val, x_best.copy()))
            if len(pool) > 6:
                pool = sorted(pool, key=lambda z: z[0])[:6]
            cur_X, cur_val = newX.copy(), float(newV)
        elif newV < best_val + 0.004:
            # keep diverse near-best
            if len(pool) < 6:
                pool.append((float(newV), newX.copy()))
            else:
                # replace worst if distinct
                pool = sorted(pool, key=lambda z: z[0])
                if newV < pool[-1][0]:
                    # check distance
                    dmin = min(float(np.linalg.norm(newX - p[1])) for p in pool)
                    if dmin > 0.5 or rng.rand() < 0.2:
                        pool[-1] = (float(newV), newX.copy())

    # final deep polish on best
    try:
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 250.0, 60)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 600.0, 70)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
        if time.time() - t0 < BUDGET - 2:
            Xf = run_lbfgs(x_best, 1200.0, 40)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, x_best = vf, Xf.copy()
    except Exception:
        pass
    try:
        deadline = t0 + BUDGET - 0.5
        Xp, vp = lp_polish(x_best, deadline, r_init=1.0e-3, max_steps=30)
        if vp < best_val:
            best_val, x_best = vp, Xp.copy()
    except Exception:
        pass

    x_best = _norm(x_best)
    return x_best
