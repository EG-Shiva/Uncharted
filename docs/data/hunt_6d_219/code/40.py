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
    FINAL_RESERVE = 22.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    P = len(iu0)
    rng = np.random.RandomState(0)

    # load incumbent
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
    global_best = x_best.copy()
    global_val = best_val

    # tempering setup
    R = 7
    betas = np.geomspace(60.0, 3000.0, R)  # 0=hot, R-1=cold
    beta_max = betas.max()
    # replicas
    Xr = np.zeros((R, n, d))
    # coldest = incumbent
    Xr[-1] = global_best.copy()
    for k in range(R - 1):
        # hotter => larger perturbation
        frac = 1.0 - k / float(R - 1)  # 1 for hottest
        sigma = 0.03 + 0.16 * frac
        if k == 0:
            Xr[k] = _norm(rng.randn(n, d))
        else:
            pert = rng.randn(n, d)
            proj = (pert * global_best).sum(axis=1, keepdims=True) * global_best
            pert = pert - proj
            Xp = global_best + sigma * pert
            Xr[k] = _norm(Xp)
    # Adam states
    M = np.zeros((R, n, d))
    V = np.zeros((R, n, d))
    lr_cold = 0.0025
    lr_k = lr_cold * np.sqrt(beta_max / betas)  # (R,)
    noise_std = np.sqrt(2.0 * lr_k / betas)  # (R,)
    b1, b2, eps = 0.9, 0.999, 1e-8
    adam_t = 0

    def quench_lbfgs(x_init, beta, maxiter):
        y0 = x_init.ravel().copy()
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
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 10000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x_init)

    def lp_polish(x, deadline, max_steps=14):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = 1.0e-3
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
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
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                if v < global_val - 1e-12:
                    pass
                radius = min(radius * 1.5, 0.02)
                bx = x.copy()
            else:
                radius *= 0.5
        return bx

    last_quench = time.time()
    quench_count = 0
    block_steps = 40

    # main tempering loop
    while time.time() - t0 < BUDGET - FINAL_RESERVE:
        # Langevin block
        for _ in range(block_steps):
            if time.time() - t0 > BUDGET - FINAL_RESERVE:
                break
            adam_t += 1
            # grams for all replicas
            G = np.matmul(Xr, np.transpose(Xr, (0, 2, 1)))  # R,n,n
            gmat = G[:, iu0, iu1]  # R,P
            mvec = gmat.max(axis=1)  # R
            # weights
            # exp(beta_k*(g-m))
            B = betas[:, None]  # R,1
            Emat = np.exp(B * (gmat - mvec[:, None]))  # R,P
            Svec = Emat.sum(axis=1) + 1e-300  # R
            Wmat = Emat / Svec[:, None]  # R,P
            # build W (R,n,n)
            W = np.zeros((R, n, n))
            W[:, iu0, iu1] = Wmat
            W[:, iu1, iu0] = Wmat
            GX = np.matmul(W, Xr)  # R,n,d
            dot = (GX * Xr).sum(axis=2, keepdims=True)
            Gt = GX - dot * Xr
            # Adam update
            M = b1 * M + (1 - b1) * Gt
            V = b2 * V + (1 - b2) * (Gt * Gt)
            mhat = M / (1 - b1 ** adam_t)
            vhat = V / (1 - b2 ** adam_t)
            D = mhat / (np.sqrt(vhat) + eps)
            # project D to tangent
            dproj = (D * Xr).sum(axis=2, keepdims=True) * Xr
            Dt = D - dproj
            # noise
            Z = rng.randn(R, n, d)
            zproj = (Z * Xr).sum(axis=2, keepdims=True) * Xr
            Zt = Z - zproj
            step = lr_k[:, None, None] * Dt - noise_std[:, None, None] * Zt
            # SGD-style: X - lr*D + noise
            # note Dt already direction, step = lr*Dt - noise*Zt? we did lr*Dt - noise*Z
            # Actually we want X <- X - lr*Dt + noise*Zt
            Xn = Xr - 2.0 * lr_k[:, None, None] * Dt * 0.5  # placeholder
            # recompute correctly:
            Xn = Xr - lr_k[:, None, None] * Dt + noise_std[:, None, None] * Zt
            # retract
            nr = np.linalg.norm(Xn, axis=2, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Xr = Xn / nr
            # safety: replace non-finite
            bad = ~np.isfinite(Xr).all(axis=2).all(axis=1) if False else None
        # replica swaps (adjacent, random order)
        try:
            G2 = np.matmul(Xr, np.transpose(Xr, (0, 2, 1)))
            g2 = G2[:, iu0, iu1]
            m2 = g2.max(axis=1)
            # cross LSE E[i,j]
            # E[i,j] = m_j + log(sum exp(beta_i*(g_j - m_j)))/beta_i
            Ecross = np.zeros((R, R))
            for j in range(R):
                gj = g2[j]
                mj = m2[j]
                diff = gj - mj  # P
                for i in range(R):
                    bi = betas[i]
                    s = np.exp(bi * diff).sum() + 1e-300
                    Ecross[i, j] = mj + np.log(s) / bi
            order = np.arange(R - 1)
            rng.shuffle(order)
            for k in order:
                a, b = k, k + 1
                # log acceptance for swapping configs a<->b
                logA = betas[a] * (Ecross[a, a] - Ecross[a, b]) + betas[b] * (Ecross[b, b] - Ecross[b, a])
                if logA >= 0 or rng.rand() < np.exp(logA):
                    tmp = Xr[a].copy()
                    Xr[a] = Xr[b]
                    Xr[b] = tmp
                    # keep M,V with temperature (do not swap)
        except Exception:
            pass

        # periodic quench of coldest copy (leave replicas unquenched)
        now = time.time()
        if now - last_quench > 2.5 or (now - t0) < 1.0:
            last_quench = now
            quench_count += 1
            # pick replica to quench: mostly coldest, every 4th a middle one
            if quench_count % 4 == 0:
                # random middle/hot to probe new basin
                qidx = int(rng.randint(1, R - 1))
                # bias to colder middle
                if rng.rand() < 0.5:
                    qidx = R - 2
            else:
                qidx = R - 1
            try:
                xq = Xr[qidx].copy()
                # short quench: two-stage sharpening
                xh = quench_lbfgs(xq, 350.0, maxiter=60)
                xh = quench_lbfgs(xh, 700.0, maxiter=60)
                v = _max_cos(xh)
                if v < global_val:
                    global_val = v
                    global_best = xh.copy()
            except Exception:
                pass
            # also update cold replica reference? no, leave unquenched per spec

    # final polish of global best with remaining time
    deadline = t0 + BUDGET
    try:
        # quick L-BFGS sharpen then LP
        if time.time() < deadline - 5:
            try:
                xh = quench_lbfgs(global_best, 500.0, maxiter=120)
                v = _max_cos(xh)
                if v < global_val:
                    global_val = v
                    global_best = xh
            except Exception:
                pass
        if time.time() < deadline - 2:
            try:
                xp = lp_polish(global_best, deadline - 1.0, max_steps=14)
                v = _max_cos(xp)
                if v <= global_val:
                    global_best = xp
                    global_val = v
            except Exception:
                pass
    except Exception:
        pass

    global_best = _norm(global_best)
    return global_best
