import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _ppm(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _fast_relax(X, iters, lr0=0.02, margin0=0.004):
    X = _norm(X)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    for it in range(iters):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tau = margin * 0.2 + 1e-12
        W = np.exp((G - m) / tau)
        W[G < m - margin * 5.0] = 0.0
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        gn = float(np.max(np.linalg.norm(Gt, axis=1)))
        if not np.isfinite(gn) or gn < 1e-14:
            margin *= 0.5
            if margin < 1e-7:
                break
            continue
        Xn = cur - (lr / gn) * Gt
        Xn = _norm(Xn)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 0:
                margin *= 0.85
        else:
            lr *= 0.6
            if lr < 1e-6:
                margin *= 0.7
                lr = lr0 * 0.3
                if margin < 5e-7:
                    break
    return bestX, best

def solve():
    t0 = time.time()
    BUDGET = 170.0
    HOP_END = t0 + 120.0
    np.random.seed(0)
    rng = np.random.RandomState(7)
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

    def run_full(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = g.max()
            e = np.exp(beta * (g - m))
            S = e.sum()
            loss = m + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y = _norm(Y)
        return Y

    def run_sub(Xbase_fixed, idx, beta, maxiter):
        idx = np.asarray(idx, dtype=int)
        k = len(idx)
        if k == 0:
            return Xbase_fixed.copy()
        is_in = np.zeros(n, dtype=bool)
        is_in[idx] = True
        # involved pairs I,J
        # build via mask on triu
        mask = is_in[iu0] | is_in[iu1]
        I = iu0[mask].copy()
        J = iu1[mask].copy()
        # map global row -> sub position for gradient
        pos = np.full(n, -1, dtype=int)
        pos[idx] = np.arange(k)
        Xrest = Xbase_fixed.copy()
        y0 = Xbase_fixed[idx].ravel().copy()
        def fun(y):
            Y = y.reshape(k, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            Xs = Y / rn
            Xf = Xrest.copy()
            Xf[idx] = Xs
            # gather involved dots
            g = np.sum(Xf[I] * Xf[J], axis=1)
            m = g.max()
            e = np.exp(beta * (g - m))
            S = e.sum()
            loss = m + np.log(S) / beta
            w = e / S
            GX = np.zeros((n, d), dtype=float)
            # GX[I] += w*Xf[J], GX[J] += w*Xf[I]
            # vectorised with add.at
            np.add.at(GX, I, w[:, None] * Xf[J])
            np.add.at(GX, J, w[:, None] * Xf[I])
            Gs = GX[idx]  # k,d
            dot = (Gs * Xs).sum(axis=1, keepdims=True)
            GY = (Gs - dot * Xs) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(k, d)
        Y = _norm(Y)
        Xo = Xbase_fixed.copy()
        Xo[idx] = Y
        return _norm(Xo)

    def lp_polish(x, deadline):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = 1.2e-3
        while time.time() < deadline and time.time() - t0 < BUDGET - 2.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(20 * radius, 1e-9)
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
                radius = min(radius * 1.5, 0.03)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    # initial tighten
    try:
        if time.time() - t0 < BUDGET - 100:
            Xi, vi = _fast_relax(x_best, iters=600, lr0=0.02, margin0=0.003)
            if vi < best_val:
                x_best, best_val = Xi, vi
    except Exception:
        pass

    trial = 0
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 48:
        trial += 1
        mode = trial % 4
        ppm = _ppm(x_best)
        order = np.argsort(-ppm)
        Gcur = x_best @ x_best.T
        np.fill_diagonal(Gcur, -2.0)
        x_init = x_best.copy()
        sub_idx = None
        if mode == 0:
            # worst cluster subset kick + subset anneal
            k = int(rng.randint(18, 34))
            seed = order[:max(6, k // 2)]
            # expand to neighbours of worst points
            cand = list(seed)
            # add worst neighbours
            for s in seed[:6]:
                j = int(np.argmax(Gcur[s]))
                if j not in cand:
                    cand.append(j)
                # second neighbour
                row = Gcur[s].copy()
                row[j] = -2
                j2 = int(np.argmax(row))
                if j2 not in cand and len(cand) < k + 6:
                    cand.append(j2)
            # pad random if needed, trim if too many
            while len(cand) < k:
                cand.append(int(rng.randint(0, n)))
            sub_idx = np.array(sorted(set(cand))[:max(k, 18)], dtype=int)
            pert = rng.randn(len(sub_idx), d)
            pert -= (np.sum(pert * x_init[sub_idx], axis=1, keepdims=True)) * x_init[sub_idx]
            sig = 0.08 + 0.08 * rng.rand()
            x_init[sub_idx] = x_init[sub_idx] + sig * pert
            x_init[sub_idx] = _norm(x_init[sub_idx])
        elif mode == 1:
            # void reinsertion of 1-2 worst + neighbourhood subset
            nw = 1 if rng.rand() < 0.6 else 2
            victims = order[:nw]
            # light global shake
            pert = rng.randn(n, d)
            pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
            x_init = x_init + 0.005 * pert
            x_init = _norm(x_init)
            for w in victims:
                cand = _norm(rng.randn(2500, d))
                dots = cand @ x_init.T
                dots[:, w] = -2.0
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                x_init[w] = cand[bi]
            x_init = _norm(x_init)
            # subset = victims + their closest neighbours
            sub_idx = list(victims)
            for w in victims:
                nn = np.argsort(-Gcur[w])[:8]
                for u in nn:
                    if u not in sub_idx:
                        sub_idx.append(int(u))
            sub_idx = np.array(sub_idx, dtype=int)
        elif mode == 2:
            # kick worst pair + crowded neighbours hard (global relax after)
            i = int(order[0])
            j = int(np.argmax(Gcur[i]))
            crowd = np.argsort(-ppm)[:14]
            sub_idx = np.array(sorted(set([i, j] + list(crowd))), dtype=int)
            pert = rng.randn(len(sub_idx), d)
            pert -= (np.sum(pert * x_init[sub_idx], axis=1, keepdims=True)) * x_init[sub_idx]
            x_init[sub_idx] = x_init[sub_idx] + (0.10 + 0.06 * rng.rand()) * pert
            x_init[sub_idx] = _norm(x_init[sub_idx])
        else:
            # broad medium kick on random 25-50 + small global
            k = int(rng.randint(25, 50))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.randn(n, d)
            pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
            sig = np.full((n, 1), 0.004)
            sig[idx] = 0.05 + 0.04 * rng.rand()
            x_init = x_init + sig * pert
            x_init = _norm(x_init)
            sub_idx = idx if rng.rand() < 0.5 else None

        # local optimisation: subset anneal if available else fast relax
        try:
            if sub_idx is not None and len(sub_idx) < 70:
                # two-stage subset LBFGS with increasing beta
                b1 = 200.0 if (trial % 2 == 0) else 130.0
                Xo = run_sub(x_init, sub_idx, b1, maxiter=60)
                # expand subset slightly for second stage
                vsub = _max_cos(Xo)
                # quick global soft relax to settle
                Xr, vr = _fast_relax(Xo, iters=350, lr0=0.015, margin0=0.003)
                # choose better as starting for full sharpen
                Xc = Xr if vr < vsub else Xo
                # full sharpen short
                betaF = 320.0 if (trial % 3 == 0) else 220.0
                Xo2 = run_full(Xc, betaF, maxiter=50)
                v = _max_cos(Xo2)
                Xo = Xo2
            else:
                Xr, vr = _fast_relax(x_init, iters=600, lr0=0.02, margin0=0.004)
                if vr > best_val + 0.008 and rng.rand() < 0.8:
                    continue
                Xo = run_full(Xr, 180.0, maxiter=60)
                v = _max_cos(Xo)
                # second beta occasionally
                if v < best_val + 0.002:
                    Xo2 = run_full(Xo, 420.0, maxiter=60)
                    v2 = _max_cos(Xo2)
                    if v2 < v:
                        Xo, v = Xo2, v2
        except Exception:
            continue
        try:
            v = _max_cos(Xo)
        except Exception:
            continue
        if v < best_val:
            x_best = Xo
            best_val = v
            # short LP lock
            try:
                Xp, vp = lp_polish(x_best, deadline=time.time() + 2.0)
                if vp < best_val:
                    x_best, best_val = Xp, vp
            except Exception:
                pass
        if time.time() - t0 > BUDGET - 48:
            break

    # escalation on global best
    for beta, mi in [(150.0, 100), (320.0, 100), (600.0, 100)]:
        if time.time() - t0 > BUDGET - 36:
            break
        try:
            Xo = run_full(x_best, beta, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val:
                x_best, best_val = Xo, v
        except Exception:
            pass

    if time.time() < t0 + BUDGET - 5:
        try:
            Xp, vp = lp_polish(x_best, deadline=t0 + BUDGET - 2.0)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass

    x_best = _norm(x_best)
    return x_best
