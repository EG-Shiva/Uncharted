import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def make_smooth_fun(n, d, beta, iu0, iu1):
    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        X = Y / norms
        G = X @ X.T
        v = G[iu0, iu1]
        M = float(v.max())
        e = np.exp(beta * (v - M))
        S = float(e.sum())
        loss = M + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n), dtype=np.float64)
        W[iu0, iu1] = w
        W = W + W.T
        grad_X = W @ X
        dot = (grad_X * X).sum(axis=1, keepdims=True)
        grad_Y = (grad_X - dot * X) / norms
        return loss, grad_Y.ravel()
    return fun

def lbfgs_stage(X0, beta, iu0, iu1, maxiter):
    n, d = X0.shape
    fun = make_smooth_fun(n, d, beta, iu0, iu1)
    x0 = X0.ravel().copy()
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 20,
                                'ftol': 1e-14, 'gtol': 1e-9})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = X0.copy()
    X = _normed(Y)
    return X

def greedy_refine(X, steps, lr0, t0, budget):
    cur = X.copy()
    best = X.copy()
    best_val = _max_cos(best)
    lr = float(lr0)
    n = cur.shape[0]
    for s in range(steps):
        if time.time() - t0 > budget:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best = cur.copy()
        tol = max(0.0015, 8.0 * lr)
        mask = G > (m - tol)
        # binary symmetric mask, diag False
        grad = mask.astype(np.float64) @ cur
        dot = (grad * cur).sum(axis=1, keepdims=True)
        grad_t = grad - dot * cur
        deg = mask.sum(axis=1, keepdims=True).astype(np.float64)
        deg = np.maximum(deg, 1.0)
        step = grad_t / deg
        # clip step norm to avoid jumps
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        # scale down huge steps (rattlers with 0 deg have 0 step)
        cur = cur - lr * step
        cur = _normed(cur)
        # schedule
        if (s + 1) % 400 == 0:
            # check progress: if no improvement recently, decay
            v = _max_cos(cur)
            if v < best_val - 1e-12:
                best_val = v
                best = cur.copy()
            else:
                # revert partially toward best and shrink lr
                lr *= 0.6
                # pull cur toward best a bit to stabilize
                cur = _normed(0.7 * cur + 0.3 * best)
            if lr < 1e-6:
                lr = 1e-6
    v = _max_cos(cur)
    if v < best_val:
        best_val = v
        best = cur.copy()
    return best, best_val

def polish_chain(X, betas, iters, iu0, iu1, t0, budget, do_greedy=True):
    cur = X.copy()
    for b, it in zip(betas, iters):
        if time.time() - t0 > budget:
            break
        cur = lbfgs_stage(cur, b, iu0, iu1, it)
    if do_greedy and (time.time() - t0 < budget):
        # short greedy to squeeze max
        cur, _ = greedy_refine(cur, steps=1200, lr0=8e-4, t0=t0, budget=budget)
    return _normed(cur)

def solve():
    import time as _tm
    t0 = _tm.time()
    budget = 165.0  # leave margin for 180s limit; loop checks vs t0
    # deadline relative: use t0+budget
    n, d = 59, 22
    rng = np.random.default_rng(12345)
    iu0, iu1 = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1)
                if np.all(nr > 1e-9):
                    cands.append(a / nr[:, None])
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 = _normed(X0)
    else:
        vals = [_max_cos(c) for c in cands]
        X0 = cands[int(np.argmin(vals))].copy()

    best_X = _normed(X0)
    best_val = _max_cos(best_X)

    # initial deep polish of record basin
    try:
        cur = polish_chain(best_X, betas=[80, 300, 900], iters=[150, 150, 150],
                           iu0=iu0, iu1=iu1, t0=t0, budget=t0 + budget - 25)
        v = _max_cos(cur)
        if v < best_val and np.all(np.isfinite(cur)):
            best_val = v
            best_X = cur.copy()
    except Exception:
        pass

    # basin hopping
    sigmas = [0.004, 0.012, 0.03, 0.008, 0.02, 0.05, 0.006, 0.015]
    it = 0
    while _tm.time() - t0 < budget - 8:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        # build perturbation from best
        Xt = best_X.copy()
        # identify rattlers: points whose max contact is well below best
        try:
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # rattler threshold
            is_rattler = rowmax < (best_val - 0.004)
        except Exception:
            is_rattler = np.zeros(n, dtype=bool)

        noise = rng.standard_normal((n, d))
        # stronger kick for rattlers
        scale = np.full((n, 1), sigma)
        scale[is_rattler, 0] = max(sigma * 4.0, 0.05)
        Xt = Xt + scale * noise
        # occasionally replace 1-2 worst/rattler points by fresh random
        if it % 3 == 0:
            k = 1 + int(rng.integers(0, 2))
            if np.any(is_rattler):
                idx = np.where(is_rattler)[0]
                # choose up to k of them
                if len(idx) > k:
                    ch = rng.choice(idx, size=k, replace=False)
                else:
                    ch = idx
                for c in ch:
                    Xt[c] = rng.standard_normal(d)
            else:
                # kick points with smallest rowmax (least constrained)
                order = np.argsort(rowmax)
                for c in order[:k]:
                    if rng.random() < 0.5:
                        Xt[c] = Xt[c] + 0.15 * rng.standard_normal(d)
        # occasionally big global shake
        if it % 11 == 0:
            Xt = best_X + 0.08 * rng.standard_normal((n, d))
            Xt = _normed(Xt)
        else:
            Xt = _normed(Xt)

        # quick polish: choose depth by time left
        time_left = (t0 + budget - 8) - _tm.time()
        if time_left < 3:
            break
        try:
            if it % 4 == 0:
                cur = polish_chain(Xt, betas=[250, 1000], iters=[90, 90],
                                   iu0=iu0, iu1=iu1, t0=t0, budget=t0 + budget - 5)
            else:
                cur = polish_chain(Xt, betas=[400, 1400], iters=[60, 60],
                                   iu0=iu0, iu1=iu1, t0=t0, budget=t0 + budget - 5)
            v = _max_cos(cur)
            if v < best_val - 1e-12:
                best_val = v
                best_X = cur.copy()
        except Exception:
            continue
        # opportunistic extra squeeze if close
        if _tm.time() - t0 > budget - 8:
            break

    # final squeeze on best with high sharpness + greedy
    try:
        if _tm.time() - t0 < budget:
            cur = lbfgs_stage(best_X, 2200, iu0, iu1, 120)
            v = _max_cos(cur)
            if v < best_val:
                best_val = v
                best_X = cur.copy()
        if _tm.time() - t0 < budget + 10:
            best_X, best_val = greedy_refine(best_X, steps=2500, lr0=5e-4,
                                             t0=t0, budget=t0 + 172)
    except Exception:
        pass

    best_X = _normed(best_X)
    # safety: no NaN / zero rows
    best_X = np.nan_to_num(best_X, nan=1.0, posinf=1.0, neginf=-1.0)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best_X = best_X / nr
    return best_X
