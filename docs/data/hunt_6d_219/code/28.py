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

def _ppm_row(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pm = np.max(G, axis=1)
    return pm, G

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
            margin *= 0.6
            if margin < 5e-7:
                break
            continue
        Xn = cur - (lr / gn) * Gt
        Xn = _norm(Xn)
        G2 = Xn @ Xn.T
        np.fill_diagonal(G2, -2.0)
        mn = float(np.max(G2))
        if mn < m + 1e-12:
            cur = Xn
            if mn < best:
                best = mn
                bestX = Xn.copy()
            if it % 200 == 199:
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
    HOP_END = t0 + 125.0
    rng = np.random.RandomState(0)
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
        return _norm(Y)

    def lp_polish(x, deadline):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = 1.2e-3
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.5:
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

    # initial sharpen of loaded best
    try:
        Xo = run_lbfgs(x_best, 250.0, maxiter=60)
        vo = _max_cos(Xo)
        if vo < best_val:
            x_best, best_val = Xo, vo
    except Exception:
        pass

    cur = x_best.copy()
    cur_val = best_val
    T0 = 0.0018

    trial = 0
    while time.time() < HOP_END:
        trial += 1
        if time.time() - t0 > BUDGET - 45:
            break
        prog = (time.time() - t0) / max(1e-9, (HOP_END - t0))
        T = T0 * (1.0 - 0.75 * prog) + 1e-9
        # reset wanderer if drifted too far
        if cur_val > best_val + 0.006 and rng.rand() < 0.7:
            cur = x_best.copy()
            cur_val = best_val

        pm, Gcur = _ppm_row(cur)
        order = np.argsort(-pm)
        x_init = cur.copy()
        r = rng.rand()
        try:
            if r < 0.22:
                # hard cluster kick around worst point
                seed = int(order[0] if rng.rand() < 0.7 else rng.choice(order[:20]))
                row = Gcur[seed]
                nn = np.argsort(-row)[:55]
                k = int(rng.randint(12, 42))
                # take top-k among neighbours + random
                idx = list(nn[:k])
                # pad
                while len(idx) < k:
                    idx.append(int(rng.randint(0, n)))
                idx = np.array(sorted(set(idx)), dtype=int)
                pert = rng.randn(len(idx), d)
                pert -= (np.sum(pert * x_init[idx], axis=1, keepdims=True)) * x_init[idx]
                sig = 0.25 + 0.45 * rng.rand()
                x_init[idx] = x_init[idx] + sig * pert
                x_init[idx] = _norm(x_init[idx])
                # small global jitter
                gj = rng.randn(n, d); gj -= (np.sum(gj * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.004 * gj)
            elif r < 0.45:
                # mass void reinsertion: kill 3-10 worst/crowded, refill greedily
                nw = int(rng.randint(3, 9))
                # victims: mix of worst ppm and high degree
                Gtmp = cur @ cur.T
                np.fill_diagonal(Gtmp, -2.0)
                deg = np.sum(Gtmp > cur_val - 2e-7, axis=1)
                score = pm + 0.002 * (deg / max(1, deg.max()))
                victims = np.argsort(-score)[:nw + 4]
                # random subset
                victims = np.array(rng.choice(victims, size=nw, replace=False), dtype=int)
                keep = np.ones(n, dtype=bool); keep[victims] = False
                Xkeep = x_init[keep]
                # sequential farthest insertion
                for vi, w in enumerate(victims):
                    cand = _norm(rng.randn(2500, d))
                    dots = cand @ Xkeep.T
                    worst = dots.max(axis=1)
                    # diversity: add noise to pick different voids
                    worst = worst + 0.002 * rng.randn(len(worst))
                    bi = int(np.argmin(worst))
                    nv = cand[bi]
                    # local refine of this single point vs keep+previous
                    others = Xkeep if vi == 0 else np.vstack([Xkeep, x_init[victims[:vi]]])
                    # few steps of tangential descent
                    vv = nv.copy()
                    lr = 0.1
                    for _ in range(60):
                        dd = others @ vv
                        m_ = float(np.max(dd))
                        ww = np.exp((dd - m_) / 0.002)
                        ww[dd < m_ - 0.012] = 0.0
                        g_ = ww @ others
                        gt = g_ - np.dot(g_, vv) * vv
                        nn_ = float(np.linalg.norm(gt))
                        if nn_ < 1e-12:
                            break
                        vn = vv - (lr / nn_) * gt
                        vn /= np.linalg.norm(vn)
                        mn_ = float(np.max(others @ vn))
                        if mn_ < m_:
                            vv = vn
                        else:
                            lr *= 0.7
                            if lr < 1e-4:
                                break
                    x_init[w] = vv
                    Xkeep = np.vstack([Xkeep, vv.reshape(1, d)]) if False else Xkeep
                    # update Xkeep for next insertion to avoid collisions among new
                    # append to others via Xkeep extension
                    # trick: keep expanding Xkeep
                    Xkeep = np.vstack([Xkeep, vv.reshape(1, d)])
                    # remove last duplicate handling: we double counted; fix by keeping base
                    # actually rebuild: simpler keep list
                # light shake of neighbours of victims
                for w in victims:
                    drow = (x_init @ x_init[w])
                    drow[w] = -2
                    nb = np.argsort(-drow)[:6]
                    for u in nb:
                        if u in victims:
                            continue
                        pp = rng.randn(d)
                        pp -= np.dot(pp, x_init[u]) * x_init[u]
                        x_init[u] = _norm((x_init[u] + 0.03 * pp).reshape(1, -1))[0]
                x_init = _norm(x_init)
            elif r < 0.62:
                # broad medium-large kick on random subset
                k = int(rng.randint(20, 70))
                idx = rng.choice(n, size=k, replace=False)
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig = np.full((n, 1), 0.003 + 0.003 * rng.rand())
                s_big = 0.15 + 0.35 * rng.rand()
                sig[idx] = s_big
                x_init = _norm(x_init + sig * pert)
            elif r < 0.74:
                # antipodal flip of few points + kick (topology break)
                k = int(rng.randint(4, 14))
                idx = rng.choice(n, size=k, replace=False)
                x_init[idx] = -x_init[idx]
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.02 * pert)
                # extra kick on flipped
                p2 = rng.randn(k, d)
                p2 -= (np.sum(p2 * x_init[idx], axis=1, keepdims=True)) * x_init[idx]
                x_init[idx] = _norm(x_init[idx] + 0.2 * p2)
            elif r < 0.88:
                # rigid rotation of a cluster (preserve internal, change cross)
                seed = int(rng.choice(order[:30]))
                row = Gcur[seed]
                nn = np.argsort(-row)[:int(rng.randint(10, 30))]
                # random orthogonal in 6D
                A = rng.randn(d, d)
                Q, _ = np.linalg.qr(A)
                # rotate cluster points slightly interpolated: X <- norm( (1-a)X + a XQ )
                a = 0.4 + 0.6 * rng.rand()
                rot = x_init[nn] @ Q
                x_init[nn] = _norm((1 - a) * x_init[nn] + a * rot)
                # plus noise
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                x_init = _norm(x_init + 0.005 * pert)
            else:
                # global large shake (escape)
                pert = rng.randn(n, d)
                pert -= (np.sum(pert * x_init, axis=1, keepdims=True)) * x_init
                sig = 0.06 + 0.10 * rng.rand()
                x_init = _norm(x_init + sig * pert)

            # local optimisation: relax then LBFGS
            Xr, vr = _fast_relax(x_init, iters=380, lr0=0.02, margin0=0.004)
            # early reject if catastrophically bad to save LBFGS time
            if vr > cur_val + 0.02 and rng.rand() < 0.9:
                # still allow occasional
                v = vr
                Xo = Xr
            else:
                beta = 180.0 if (trial % 2 == 0) else 320.0
                Xo = run_lbfgs(Xr, beta, maxiter=55)
                v = _max_cos(Xo)
                # extra short relax to settle ties
                if v < cur_val + 0.005:
                    Xo2, v2 = _fast_relax(Xo, iters=150, lr0=0.008, margin0=0.001)
                    if v2 < v:
                        Xo, v = Xo2, v2
        except Exception:
            continue

        if v < best_val - 1e-12:
            x_best, best_val = Xo.copy(), float(v)
            cur, cur_val = Xo.copy(), float(v)
        elif v < cur_val or rng.rand() < np.exp(-(v - cur_val) / max(T, 1e-9)):
            # accept sideways/uphill to wander to new basin
            cur, cur_val = Xo.copy(), float(v)
        # occasional pull back to best region with big kick next time
        if trial % 25 == 0 and rng.rand() < 0.4:
            cur = x_best.copy()
            cur_val = best_val

    # final sharpening of best
    try:
        for beta, mi in [(250.0, 80), (500.0, 80), (1000.0, 60)]:
            if time.time() - t0 > BUDGET - 25:
                break
            Xo = run_lbfgs(x_best, beta, maxiter=mi)
            vo = _max_cos(Xo)
            if vo < best_val:
                x_best, best_val = Xo, vo
        # narrow relax
        if time.time() - t0 < BUDGET - 20:
            Xo, vo = _fast_relax(x_best, iters=400, lr0=0.008, margin0=0.001)
            if vo < best_val:
                x_best, best_val = Xo, vo
        # LP polish with remaining time
        if time.time() - t0 < BUDGET - 5:
            try:
                Xlp, vlp = lp_polish(x_best, t0 + BUDGET - 1.0)
                if vlp < best_val:
                    x_best, best_val = Xlp, vlp
            except Exception:
                pass
    except Exception:
        pass

    return _norm(np.asarray(x_best, dtype=float))
