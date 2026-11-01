import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        cuc = G[iu]
        m = float(cuc.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cuc > thresh
        if int(sel.sum()) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(cuc[idx])[-3500:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        c = cuc[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_polish(X, beta=150.0, maxiter=80):
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    Y0 = (X / np.linalg.norm(X, axis=1, keepdims=True)).ravel().copy()
    lam = 1.0
    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        Xn = Y / norms
        G = Xn @ Xn.T
        vals = G[iu]
        m = np.max(vals)
        e = np.exp(beta * (vals - m))
        S = np.sum(e)
        s = m + np.log(S) / beta
        diff = (norms.ravel() - 1.0)
        Epen = lam * np.sum(diff * diff)
        E = s + Epen
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu] = w
        W = W + W.T
        grad_X = W @ Xn
        dot = np.sum(grad_X * Xn, axis=1, keepdims=True)
        grad_Y = (grad_X - dot * Xn) / norms + (2.0 * lam * diff[:, None] * Xn)
        return E, grad_Y.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": maxiter * 3})
        Yopt = res.x.reshape(n, d)
    except Exception:
        return X
    yn = np.linalg.norm(Yopt, axis=1, keepdims=True)
    yn[yn == 0] = 1.0
    return Yopt / yn

def _best_hole(R, rng, n_sample=25000, batch=5000):
    n, d = R.shape
    best_v = None
    best_s = 1e9
    nb = n_sample // batch
    for _ in range(nb):
        C = rng.normal(size=(batch, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = R @ C.T
        scores = D.max(axis=0)
        k = int(np.argmin(scores))
        if float(scores[k]) < best_s:
            best_s = float(scores[k])
            best_v = C[k].copy()
    return best_v, best_s

def _refine_hole(v0, R, beta=250.0, iters=350, lr0=0.08):
    v = v0 / np.linalg.norm(v0)
    best = v.copy()
    best_s = float((R @ v).max())
    lr = lr0
    for _ in range(iters):
        dots = R @ v
        m = float(dots.max())
        e = np.exp(beta * (dots - m))
        w = e / e.sum()
        g = w @ R
        gt = g - float(g @ v) * v
        v = v - lr * gt
        nv = np.linalg.norm(v)
        if nv == 0:
            v = best.copy()
            continue
        v = v / nv
        sc = float((R @ v).max())
        if sc < best_s:
            best_s = sc
            best = v.copy()
        lr *= 0.997
    return best, best_s

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    betas = [90.0, 150.0, 280.0, 500.0]
    trial = 0
    while True:
        rem = deadline - time.time()
        if rem < 8.0:
            break
        pt = 2.2 if rem > 35 else max(1.1, rem - 6.5)
        t_trial = time.time()
        # contact counts of best
        Gb = best @ best.T
        np.fill_diagonal(Gb, -2.0)
        m0 = float(Gb.max())
        cnt = (Gb >= m0 - 1e-9).sum(axis=1)
        order_desc = np.argsort(-cnt)
        order_asc = np.argsort(cnt)
        mode = trial % 7
        beta = betas[trial % len(betas)]
        y = None
        try:
            if mode == 0:
                # remove worst, relax rest, reinsert
                victim = int(order_desc[0])
                R = np.delete(best, victim, axis=0)
                R = _smooth_polish(R, beta=120.0, maxiter=60)
                if deadline - time.time() < 3:
                    R1 = R
                else:
                    R1 = _lp_polish(R, 0.55)
                hv, _ = _best_hole(R1, rng, n_sample=25000, batch=5000)
                hv, _ = _refine_hole(hv, R1, beta=250.0, iters=300, lr0=0.08)
                y = np.vstack([R1, hv[None, :]])
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                left = pt - (time.time() - t_trial) - 0.05
                if left > 0.35:
                    y = _smooth_polish(y, beta=beta, maxiter=50)
                    left2 = pt - (time.time() - t_trial) - 0.02
                    if left2 > 0.3:
                        y = _lp_polish(y, left2)
            elif mode == 1:
                sigma = 0.022
                y = best + sigma * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth_polish(y, beta=beta, maxiter=80)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.3:
                    y = _lp_polish(y, left)
            elif mode == 2:
                k_rand = rng.choice(n, 9, replace=False)
                subset = np.unique(np.concatenate([order_desc[:3], k_rand]))
                y = best.copy()
                y[subset] += 0.085 * rng.normal(size=(len(subset), d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth_polish(y, beta=beta, maxiter=70)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.3:
                    y = _lp_polish(y, left)
            elif mode == 3:
                # fast random victim reinsert, no rest relax
                victim = int(rng.integers(n))
                R = np.delete(best, victim, axis=0)
                hv, _ = _best_hole(R, rng, n_sample=30000, batch=5000)
                hv, _ = _refine_hole(hv, R, beta=300.0, iters=400, lr0=0.1)
                y = np.vstack([R, hv[None, :]])
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.3:
                    y = _lp_polish(y, left)
            elif mode == 4:
                # coordinate sweep on worst few points
                y = best.copy()
                for victim in order_desc[:4]:
                    victim = int(victim)
                    R = np.delete(y, victim, axis=0)
                    cur_v = y[victim]
                    cur_s = float((R @ cur_v).max())
                    hv, hs = _best_hole(R, rng, n_sample=12000, batch=4000)
                    hv, hs = _refine_hole(hv, R, beta=300.0, iters=250, lr0=0.08)
                    if hs < cur_s - 1e-7:
                        # replace
                        y[victim] = hv
                    # keep others fixed sequentially
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.3:
                    y = _lp_polish(y, left)
            elif mode == 5:
                subset = rng.choice(n, 22, replace=False)
                y = best.copy()
                y[subset] += 0.05 * rng.normal(size=(len(subset), d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _smooth_polish(y, beta=beta, maxiter=90)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.3:
                    y = _lp_polish(y, left)
            else:
                subset = order_desc[:10]
                y = best.copy()
                y[subset] += 0.03 * rng.normal(size=(len(subset), d))
                # also nudge rattlers if any
                ratt = np.where(cnt == 0)[0]
                if len(ratt) > 0:
                    y[ratt] += 0.06 * rng.normal(size=(len(ratt), d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                left = pt - (time.time() - t_trial) - 0.02
                if left > 0.25:
                    y = _lp_polish(y, left)
        except Exception:
            trial += 1
            continue
        if y is not None:
            try:
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                v = _max_cos(y)
                if v < best_val:
                    best = y
                    best_val = v
            except Exception:
                pass
        trial += 1

    rem = deadline - time.time()
    if rem > 1.3:
        try:
            best = _lp_polish(best, rem - 0.7)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
