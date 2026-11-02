import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= budget:
            break
    return x, best

def _adam(x0, steps, beta0, beta1, lr0, lr1, Iu, Ju):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _lbfgs_opt(x0, beta, maxiter, Iu, Ju):
    n, d = x0.shape
    y0 = x0.ravel().copy()
    def fun(yflat):
        Y = yflat.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        X = Y / nr
        G = X @ X.T
        g = G[Iu, Ju]
        gm = float(g.max())
        e = np.exp(beta * (g - gm))
        Z = float(e.sum())
        val = gm + np.log(Z) / beta
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        GX = W @ X
        dot = (GX * X).sum(axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return val, GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-22, "gtol": 1e-14, "maxfun": maxiter * 3})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(20261102)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    Iu, Ju = np.triu_indices(n, 1)
    it = 0
    # quick L-BFGS lift of incumbent at moderate beta then LP snap if time allows
    # (skip heavy init to save time for hopping)
    while time.time() - t0 < 138.0:
        it += 1
        if time.time() - t0 > HARD - 32:
            break
        G = x_best @ x_best.T
        gall = G[Iu, Ju]
        mx = float(gall.max())
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, Iu[tight], 1)
        np.add.at(cnt, Ju[tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        # find current worst pair for surgery
        k = int(np.argmax(gall))
        wi, wj = int(Iu[k]), int(Ju[k])
        mode = it % 8
        y = x_best.copy()
        if mode == 0:
            y += rng.normal(0, 0.012, size=y.shape)
        elif mode == 1:
            y += rng.normal(0, 0.035, size=y.shape)
            if len(ratt) > 0:
                for idx in ratt:
                    if rng.random() < 0.7:
                        y[idx] = rng.normal(size=4)
                    else:
                        y[idx] += rng.normal(0, 0.4, size=4)
        elif mode == 2:
            y += rng.normal(0, 0.08, size=y.shape)
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = rng.normal(size=(2, 4))
        elif mode == 3:
            # max-pair surgery: push worst pair apart tangentially + jitter hubs
            y += rng.normal(0, 0.02, size=y.shape)
            vi = y[wi] / np.linalg.norm(y[wi])
            vj = y[wj] / np.linalg.norm(y[wj])
            # move i away from j along tangent
            ti = vi - (vi @ vj) * vj
            tj = vj - (vj @ vi) * vi
            tn = np.linalg.norm(ti) + 1e-12
            ti /= tn; tj /= (np.linalg.norm(tj) + 1e-12)
            y[wi] += 0.06 * ti + rng.normal(0, 0.01, size=4)
            y[wj] += 0.06 * tj + rng.normal(0, 0.01, size=4)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.02, size=(len(hubs), d))
        elif mode == 4:
            y += rng.normal(0, 0.15, size=y.shape)
            rep = rng.choice(n, size=3, replace=False)
            y[rep] = rng.normal(size=(3, 4))
        elif mode == 5:
            y += rng.normal(0, 0.025, size=y.shape)
            # random orthogonal twist on subset to change topology
            sub = rng.choice(n, size=12, replace=False)
            A = rng.normal(size=(4, 4))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            m = y[sub].mean(axis=0, keepdims=True)
            y[sub] = (y[sub] - m) @ Q.T + m
            if len(ratt) > 0:
                y[rng.choice(ratt)] = rng.normal(size=4)
        elif mode == 6:
            y += rng.normal(0, 0.05, size=y.shape)
            rep = rng.choice(n, size=1, replace=False)
            y[rep] = rng.normal(size=(1, 4))
            # also relocate worst-pair midpoint neighbor: pick point closest to midpoint
            mid = (y[wi] + y[wj])
            mid /= (np.linalg.norm(mid) + 1e-12)
            dmid = y @ mid
            dmid[wi] = -2; dmid[wj] = -2
            q = int(np.argmax(dmid))
            y[q] += rng.normal(0, 0.2, size=4)
        else:
            y += rng.normal(0, 0.04, size=y.shape)
            # antipodal flip of random point (different basin on projective space)
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = -y[rep] + rng.normal(0, 0.05, size=(2, 4))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if mode in (2, 4):
            beta0, beta1 = 30.0, 380.0
            steps = 900
        elif mode == 3:
            beta0, beta1 = 60.0, 450.0
            steps = 1000
        else:
            beta0, beta1 = 45.0, 300.0
            steps = 850
        y2 = _adam(y, steps, beta0, beta1, 0.007, 0.0004, Iu, Ju)
        v2 = _max_cos(y2)
        thresh = best_val + (0.005 if mode in (2, 4) else 0.002)
        if v2 < thresh:
            # L-BFGS escalating beta refinement
            y3 = _lbfgs_opt(y2, 350.0, 120, Iu, Ju)
            v3 = _max_cos(y3)
            if v3 < v2:
                y2, v2 = y3, v3
                y4 = _lbfgs_opt(y2, 700.0, 120, Iu, Ju)
                v4 = _max_cos(y4)
                if v4 < v2:
                    y2, v2 = y4, v4
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 1.0:
                if v2 < best_val:
                    x_best, best_val = y2, v2
                break
            if v2 < best_val + 0.0008:
                b = min(3.0, max(1.0, remain / 10.0))
                # only spend LP on clearly promising
                if v2 < best_val + 0.0002:
                    yp, vp = _lp_polish(y2, b)
                else:
                    yp, vp = y2, v2
                if vp < best_val - 1e-12:
                    x_best, best_val = yp, vp
                elif vp < best_val + 1e-12 and rng.random() < 0.03:
                    x_best = yp
            elif rng.random() < 0.005:
                # rare random restart drift
                pass
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 2.0:
        remain = min(remain, 28.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
