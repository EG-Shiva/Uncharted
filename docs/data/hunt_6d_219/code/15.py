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

def _fast_relax(X, iters, lr0=0.025, margin0=0.005):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
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
        # soft-max weights
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
    HOP_END = t0 + 128.0
    np.random.seed(0)
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
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum())
            loss = m + np.log(S + 1e-300) / beta
            w = e / (S + 1e-300)
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

    def lp_polish(x, deadline, r_init=1.0e-3):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline - 0.5 and time.time() - t0 < BUDGET - 2.0:
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
            if p > 4000:
                k = 4000
                part = np.argpartition(-csel, k-1)[:k]
                I, J, csel = I[part], J[part], csel[part]
                p = k
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
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
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
            if p < 10 and radius < 1e-6:
                break
        return bx, best

    # light initial tighten
    try:
        if time.time()-t0 < BUDGET-100:
            Xi, vi = _fast_relax(x_best, iters=600, lr0=0.02, margin0=0.003)
            if vi < best_val:
                x_best, best_val = Xi, vi
    except Exception:
        pass

    x_cur = x_best.copy()
    cur_val = best_val

    def void_reinsert(x_init, victims, n_cand=2500):
        for w in victims:
            cand = rng.randn(n_cand, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ x_init.T
            dots[:, w] = -2.0
            worst = dots.max(axis=1)
            bi = int(np.argmin(worst))
            # second stage local refine around best cand
            best_c = cand[bi]
            # small local search
            for _ in range(2):
                pert = rng.randn(300, d)*0.08
                # project to tangent at best_c
                pert -= (pert @ best_c)[:,None]*best_c[None,:]
                c2 = best_c[None,:] + pert
                c2 /= np.linalg.norm(c2, axis=1, keepdims=True)
                d2 = c2 @ x_init.T
                d2[:, w] = -2.0
                w2 = d2.max(axis=1)
                bi2 = int(np.argmin(w2))
                if w2[bi2] < worst[bi]:
                    best_c = c2[bi2]
                    worst[bi] = w2[bi2]
            x_init[w] = best_c
        return x_init

    trial = 0
    while time.time() < HOP_END and time.time()-t0 < BUDGET-45:
        trial += 1
        # choose base: mostly global best, sometimes wandering cur
        if trial % 4 == 3 and cur_val < best_val + 0.004:
            base = x_cur
            base_val = cur_val
        else:
            base = x_best
            base_val = best_val
        ppm = _per_point_max(base)
        order = np.argsort(-ppm)
        x_init = base.copy()
        mode = trial % 6
        if mode == 0:
            sigma = 0.010 + 0.005*rng.rand()
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + sigma*pert
        elif mode == 1:
            k = int(rng.randint(8, 16))
            idx = order[:k]
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            sig = np.full((n,1), 0.004)
            sig[idx] = 0.12*(0.7+0.6*rng.rand(k,1))
            x_init = x_init + sig*pert
        elif mode == 2:
            nw = 1 if rng.rand()<0.5 else (2 if rng.rand()<0.8 else 3)
            victims = order[:nw]
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + 0.006*pert
            x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
            x_init = void_reinsert(x_init, victims, n_cand=2000)
        elif mode == 3:
            # collective cluster move: worst + its nearest neighbors move together
            w = int(order[0])
            G = x_init @ x_init.T
            G[w,w] = -2
            nbrs = np.argsort(-G[w])[:int(rng.randint(8,14))]
            # common random direction in tangent
            common = rng.randn(d)
            # make orthogonal-ish to cluster mean
            # apply common shift + individual jitter
            for idx in nbrs:
                t = rng.randn(d)
                t -= (t @ x_init[idx])*x_init[idx]
                tn = np.linalg.norm(t)+1e-12
                t /= tn
                step = 0.05 + 0.05*rng.rand()
                # mix common and random
                mix = 0.5*common + 0.5*rng.randn(d)
                mix -= (mix @ x_init[idx])*x_init[idx]
                mn = np.linalg.norm(mix)+1e-12
                mix /= mn
                x_init[idx] = x_init[idx] + step*mix + 0.01*t
            # also jitter rest lightly
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + 0.003*pert
        elif mode == 4:
            k = int(rng.randint(25, 50))
            idx = rng.choice(n, size=k, replace=False)
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            sig = np.full((n,1), 0.005)
            sig[idx] = 0.05 + 0.05*rng.rand()
            x_init = x_init + sig*pert
        else:
            # orthogonal jump of worst few
            k = int(rng.randint(3, 7))
            idx = order[:k]
            G = x_init @ x_init.T
            np.fill_diagonal(G, -2)
            pert = rng.randn(n, d)
            pert -= (pert*x_init).sum(1, keepdims=True)*x_init
            x_init = x_init + 0.015*pert
            for w in idx:
                j = int(np.argmax(G[w]))
                v = x_init[w] - (x_init[w]@x_init[j])*x_init[j]
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    v /= nv
                    alpha = 0.3 + 0.4*rng.rand()
                    x_init[w] = (1-alpha)*x_init[w] + alpha*v
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)

        try:
            Xr, vr = _fast_relax(x_init, iters=600, lr0=0.025, margin0=0.0045)
        except Exception:
            continue
        # wandering acceptance
        if vr < cur_val - 1e-9 or (vr < cur_val + 0.002 and rng.rand() < 0.15):
            # update wandering state (keep normalized)
            x_cur, cur_val = Xr, vr
        if vr > best_val + 0.009 and rng.rand() < 0.85:
            continue
        try:
            beta1 = [130.0, 190.0, 260.0][trial % 3]
            Xo = run_lbfgs(Xr, beta1, maxiter=70)
            v = _max_cos(Xo)
            if v < best_val + 0.0025 or (trial % 3 == 0):
                beta2 = [350.0, 500.0, 800.0][(trial//3) % 3]
                Xo2 = run_lbfgs(Xo, beta2, maxiter=70)
                v2 = _max_cos(Xo2)
                if v2 < v:
                    Xo, v = Xo2, v2
        except Exception:
            continue
        if v < cur_val:
            x_cur, cur_val = Xo.copy(), v
        if v < best_val - 1e-12:
            x_best, best_val = Xo.copy(), v
            # reset wandering to new best occasionally
            if rng.rand() < 0.7:
                x_cur, cur_val = Xo.copy(), v
        if time.time()-t0 > BUDGET-45:
            break

    # escalation on global best
    for beta, mi in [(160.0,100),(320.0,100),(650.0,100),(1200.0,80)]:
        if time.time()-t0 > BUDGET-30:
            break
        try:
            Xo = run_lbfgs(x_best, beta, maxiter=mi)
            v = _max_cos(Xo)
            if v < best_val:
                x_best, best_val = Xo, v
        except Exception:
            pass

    # final LP polish with remaining time
    deadline = t0 + BUDGET - 1.5
    if time.time() < deadline - 3:
        try:
            Xp, vp = lp_polish(x_best, deadline, r_init=9e-4)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
