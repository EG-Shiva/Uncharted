import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _fast_relax(X, iters, lr0=0.02, margin0=0.004):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    best = _max_cos(X)
    bestX = X.copy()
    cur = X.copy()
    lr = lr0
    margin = margin0
    n = cur.shape[0]
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
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        # evaluate max quickly
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
    HOP_END = t0 + 118.0
    np.random.seed(0)
    rng = np.random.RandomState(7)

    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    # load
    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.randn(n, d)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
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
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def lp_polish(x, seconds, deadline):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = 1.2e-3
        while time.time() - t0 < BUDGET - 3 and time.time() < deadline:
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
                k = 3500
                part = np.argpartition(-csel, k-1)[:k]
                I, J, csel = I[part], J[part], csel[part]
                p = k
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p, n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1]=1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                if v < best_val - 1e-12:
                    pass
                radius = min(radius*1.5, 0.03)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    # quick initial soften to ensure starting point is locally tight
    try:
        if time.time()-t0 < BUDGET-100:
            Xi, vi = _fast_relax(x_best, iters=800, lr0=0.02, margin0=0.003)
            if vi < best_val:
                x_best, best_val = Xi, vi
    except Exception:
        pass

    trial = 0
    # basin hopping
    while time.time() < HOP_END and time.time()-t0 < BUDGET-50:
        trial += 1
        mode = trial % 5
        ppm = _per_point_max(x_best)
        order = np.argsort(-ppm)
        x_init = x_best.copy()
        if mode == 0:
            # global small shake
            sigma = 0.012
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + sigma*pert
        elif mode == 1:
            # kick worst 12 hard
            k = 12
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            sig = np.full((n,1), 0.004)
            sig[idx] = 0.14
            # randomize sigma a bit
            sig[idx] *= (0.7 + 0.6*rng.rand(k,1))
            x_init = x_init + sig*pert
        elif mode == 2:
            # re-insert 1-2 worst at void + small shake
            nw = 1 if (rng.rand() < 0.6) else 2
            victims = order[:nw]
            # small shake others
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + 0.006*pert
            for w in victims:
                cand = rng.randn(600, d)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ x_init.T
                # exclude self column: set victim col to -inf
                dots[:, w] = -2.0
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                x_init[w] = cand[bi]
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
        elif mode == 3:
            # random subset medium kick
            k = int(rng.randint(20, 45))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            sig = np.full((n,1), 0.005)
            sig[idx] = 0.06 + 0.04*rng.rand()
            x_init = x_init + sig*pert
        else:
            # antipodal / orthogonal jump of worst few
            k = int(rng.randint(3, 8))
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + 0.02*pert
            # push worst points to orthogonal complement of their worst neighbor
            G = x_init @ x_init.T
            np.fill_diagonal(G, -2)
            for w in idx:
                j = int(np.argmax(G[w]))
                # remove component along x_j
                v = x_init[w] - (x_init[w]@x_init[j])*x_init[j]
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    v /= nv
                    alpha = 0.3 + 0.4*rng.rand()
                    x_init[w] = (1-alpha)*x_init[w] + alpha*v
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)

        # cheap relax first
        try:
            Xr, vr = _fast_relax(x_init, iters=700, lr0=0.02, margin0=0.004)
        except Exception:
            continue
        # filter: only refine if within striking distance
        if vr > best_val + 0.008 and rng.rand() < 0.8:
            continue
        # sharpen with L-BFGS (beta schedule)
        try:
            beta1 = 140.0 if (trial % 2 == 0) else 220.0
            Xo = run_lbfgs(Xr, beta1, maxiter=80)
            # second sharpen occasionally
            if vr < best_val + 0.002 or (trial % 4 == 0):
                beta2 = 420.0 if (trial % 3 == 0) else 320.0
                Xo2 = run_lbfgs(Xo, beta2, maxiter=80)
                # keep better of the two
                v1 = _max_cos(Xo); v2 = _max_cos(Xo2)
                if v2 < v1:
                    Xo, v1 = Xo2, v2
                v = v1
            else:
                v = _max_cos(Xo)
        except Exception:
            continue
        if v < best_val:
            x_best = Xo
            best_val = v
            # quick LP squeeze to lock new basin (short)
            try:
                Xp, vp = lp_polish(x_best, seconds=2.0, deadline=time.time()+2.5)
                if vp < best_val:
                    x_best, best_val = Xp, vp
            except Exception:
                pass
        if time.time()-t0 > BUDGET-50:
            break

    # escalation on global best
    for beta, mi in [(150.0,120),(320.0,120),(600.0,120)]:
        if time.time()-t0 > BUDGET-38:
            break
        try:
            Xo = run_lbfgs(x_best, beta, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val:
                x_best, best_val = Xo, v
        except Exception:
            pass

    # final LP polish with remaining time
    remain_deadline = t0 + BUDGET - 2.0
    if time.time() < remain_deadline - 4:
        try:
            Xp, vp = lp_polish(x_best, seconds=30.0, deadline=remain_deadline)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
