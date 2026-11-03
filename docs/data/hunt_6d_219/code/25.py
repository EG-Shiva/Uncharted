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
    BUDGET = 168.0
    HOP_END = t0 + 122.0
    np.random.seed(0)
    rng = np.random.RandomState(12345)
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
            Y = _norm(Y)
            return Y
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.2e-3):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5:
            if radius < 1e-14:
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
            if p > 3800:
                k2 = 3800
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

    # pool for diversity
    pool = [(best_val, x_best.copy())]
    cur_X = x_best.copy()
    cur_val = best_val

    # quick initial tighten if time allows
    try:
        if time.time() - t0 < BUDGET - 110:
            Xi, vi = _fast_relax(x_best, iters=400, lr0=0.02, margin0=0.003)
            Xf = run_lbfgs(Xi, 250.0, maxiter=60)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                x_best = Xf.copy()
                pool = [(best_val, x_best.copy())]
                cur_X, cur_val = x_best.copy(), best_val
            elif vi < best_val:
                best_val = vi
                x_best = Xi.copy()
                pool = [(best_val, x_best.copy())]
                cur_X, cur_val = x_best.copy(), best_val
    except Exception:
        pass

    trial = 0
    temp0 = 0.0025
    while time.time() < HOP_END and time.time() - t0 < BUDGET - 45:
        trial += 1
        elapsed = time.time() - t0
        temp = temp0 * max(0.08, 1.0 - elapsed / (HOP_END - t0))
        # choose base: mostly best, sometimes wanderer
        if trial % 4 == 0 or rng.rand() < 0.62:
            base_X = x_best
            base_val = best_val
        else:
            if len(pool) > 1 and rng.rand() < 0.5:
                pi = int(rng.randint(0, len(pool)))
                base_X = pool[pi][1]
                base_val = pool[pi][0]
            else:
                base_X = cur_X
                base_val = cur_val
        Gcur = base_X @ base_X.T
        np.fill_diagonal(Gcur, -2.0)
        ppm = Gcur.max(axis=1)
        order = np.argsort(-ppm)
        mcur = float(Gcur.max())
        x_init = base_X.copy()
        mode = int(rng.randint(0, 5))
        try:
            if mode == 0:
                # worst block kick
                k = int(rng.randint(15, 32))
                seeds = order[:max(5, k // 3)]
                cand = list(map(int, seeds))
                for s in list(seeds[:6]):
                    nn = np.argsort(-Gcur[s])[:4]
                    for u in nn:
                        if int(u) not in cand:
                            cand.append(int(u))
                        if len(cand) >= k + 4:
                            break
                while len(cand) < k:
                    cand.append(int(rng.randint(0, n)))
                sub = np.array(sorted(set(cand))[:k], dtype=int)
                pert = rng.randn(len(sub), d)
                pert -= (np.sum(pert * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                sig = 0.07 + 0.09 * rng.rand()
                x_init[sub] = x_init[sub] + sig * pert
                x_init[sub] = _norm(x_init[sub])
            elif mode == 1:
                # void reinsertion of 1-2 worst
                nw = 1 if rng.rand() < 0.6 else 2
                victims = list(map(int, order[:nw]))
                # small global shake first
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.004 * pert)
                for w in victims:
                    candv = _norm(rng.randn(4000, d))
                    dots = candv @ x_init.T
                    dots[:, w] = -2.0
                    worst = dots.max(axis=1)
                    bi = int(np.argmin(worst))
                    x_init[w] = candv[bi]
                x_init = _norm(x_init)
                # neighbourhood jitter of reinserted
                sub = []
                for w in victims:
                    nn = np.argsort(-Gcur[w])[:7]
                    sub.extend(map(int, nn))
                    sub.append(w)
                sub = np.array(sorted(set(sub)), dtype=int)
                pert = rng.randn(len(sub), d)
                pert -= (np.sum(pert * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                x_init[sub] = _norm(x_init[sub] + 0.02 * pert)
            elif mode == 2:
                # collective cluster shift
                s = int(order[int(rng.randint(0, 8))])
                k = int(rng.randint(12, 26))
                nn = list(map(int, np.argsort(-Gcur[s])[:k]))
                if s not in nn:
                    nn.append(s)
                sub = np.array(nn, dtype=int)
                common = rng.randn(d)
                common = common / max(1e-12, np.linalg.norm(common))
                sig_c = 0.06 + 0.07 * rng.rand()
                sig_i = 0.015 + 0.015 * rng.rand()
                pert_i = rng.randn(len(sub), d)
                pert_i -= (np.sum(pert_i * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                # project common to each tangent
                for ii, gidx in enumerate(sub):
                    xi = x_init[gidx]
                    tc = common - float(common @ xi) * xi
                    nm = float(np.linalg.norm(tc))
                    if nm > 1e-12:
                        tc /= nm
                    else:
                        tc = np.zeros(d)
                    x_init[gidx] = xi + sig_c * tc + sig_i * pert_i[ii]
                x_init[sub] = _norm(x_init[sub])
            elif mode == 3:
                # worst-pair break + crowd noise
                i = int(order[0])
                j = int(np.argmax(Gcur[i]))
                # push i,j apart
                for (a, b) in [(i, j), (j, i)]:
                    xa = x_init[a]; xb = x_init[b]
                    t = xb - float(xb @ xa) * xa
                    nm = float(np.linalg.norm(t))
                    if nm > 1e-12:
                        t /= nm
                        x_init[a] = xa - (0.07 + 0.05 * rng.rand()) * t
                x_init[[i, j]] = _norm(x_init[[i, j]])
                crowd = list(map(int, np.argsort(-ppm)[:12]))
                if i not in crowd:
                    crowd.append(i)
                if j not in crowd:
                    crowd.append(j)
                sub = np.array(sorted(set(crowd)), dtype=int)
                pert = rng.randn(len(sub), d)
                pert -= (np.sum(pert * x_init[sub], axis=1, keepdims=True)) * x_init[sub]
                x_init[sub] = _norm(x_init[sub] + (0.03 + 0.03 * rng.rand()) * pert)
            else:
                # broad random block medium kick
                k = int(rng.randint(28, 55))
                idx = rng.choice(n, size=k, replace=False)
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig = np.full((n, 1), 0.004)
                sig[idx] = 0.05 + 0.04 * rng.rand()
                x_init = _norm(x_init + sig * pert)
        except Exception:
            x_init = base_X.copy()

        # refine
        try:
            # light relax then two LBFGS
            Xr, vr = _fast_relax(x_init, iters=260, lr0=0.018, margin0=0.0032)
            b1 = 150.0 if (trial % 2 == 0) else 110.0
            Xb = run_lbfgs(Xr, b1, maxiter=55)
            vb = _max_cos(Xb)
            # pick better of Xr/Xb for second stage
            Xmid = Xb if vb < vr else Xr
            b2 = 380.0 if (trial % 3 == 0) else 280.0
            # reduce work if far from best
            vmid = min(vb, vr)
            if vmid > best_val + 0.009 and rng.rand() < 0.75:
                # skip expensive second stage, treat as failed
                Xo, v = Xmid, vmid
            else:
                Xo = run_lbfgs(Xmid, b2, maxiter=55)
                v = _max_cos(Xo)
        except Exception:
            continue

        # update pool / wanderer
        if v < best_val:
            best_val = v
            x_best = Xo.copy()
            pool.append((v, Xo.copy()))
            pool = sorted(pool, key=lambda z: z[0])[:5]
            cur_X, cur_val = Xo.copy(), v
        elif v < cur_val + temp:
            cur_X, cur_val = Xo.copy(), v
            # keep pool diverse
            if v < pool[-1][0] + 0.001 and len(pool) < 5:
                pool.append((v, Xo.copy()))
                pool = sorted(pool, key=lambda z: z[0])[:5]
            elif len(pool) > 1 and v < pool[-1][0]:
                pool[-1] = (v, Xo.copy())
                pool = sorted(pool, key=lambda z: z[0])
        # occasional re-anchor wanderer to best if drifted too far
        if cur_val > best_val + 0.006:
            cur_X, cur_val = x_best.copy(), best_val

    # final escalation + LP polish on top candidates
    pool = sorted(pool, key=lambda z: z[0])[:3]
    # extra LBFGS sharpen
    new_pool = []
    for (v, Xc) in pool:
        if time.time() - t0 > BUDGET - 30:
            new_pool.append((v, Xc))
            continue
        try:
            Xs = run_lbfgs(Xc, 500.0, maxiter=80)
            vs = _max_cos(Xs)
            if vs < v:
                new_pool.append((vs, Xs))
            else:
                new_pool.append((v, Xc))
        except Exception:
            new_pool.append((v, Xc))
    pool = sorted(new_pool, key=lambda z: z[0])
    if len(pool) > 0:
        best_val, x_best = pool[0][0], pool[0][1].copy()
    # LP polish best with remaining time
    try:
        deadline = t0 + BUDGET - 1.0
        bx, bv = lp_polish(x_best, deadline)
        if bv < best_val:
            best_val, x_best = bv, bx
        # polish second if time
        if len(pool) > 1 and time.time() < t0 + BUDGET - 12:
            b2x, b2v = lp_polish(pool[1][1], deadline)
            if b2v < best_val:
                best_val, x_best = b2v, b2x
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
