import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def exp_relax(x, iters, k0, k1, lr0, lr1, mu, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        x = normalize(x)
    return x

def riesz_relax(x, iters, s, lr0, lr1, mu, rng, thresh=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        D2 = 2.0 - 2.0 * G
        D2 = np.maximum(D2, 1e-6)
        W = np.power(D2, -s / 2.0 - 1.0)
        if thresh is not None:
            W[G < thresh] = 0.0
        np.fill_diagonal(W, 0.0)
        # avoid overflow
        # scale per row
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        # clip extreme
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        # normalize step direction to unit-ish to tame Riesz magnitudes
        gt = gt / np.maximum(gn, 1.0)
        v = mu * v + gt
        vc = v / (1.0 - mu ** (t + 1)) if mu > 0 else v
        x = x - lr * vc
        x = normalize(x)
    return x

def lbfgs_lse(x0, k, maxiter, tlim=None):
    n, d = x0.shape
    # work in normalized space with Y variable, normalize inside
    y0 = x0.ravel().copy()
    iu0 = np.triu_indices(n, 1)
    def fun_grad(y):
        Y = y.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = Y / nr
        G = X @ X.T
        c = G[iu0]
        moff = float(c.max())
        e = np.exp(k * (c - moff))
        s = float(e.sum())
        f = moff + np.log(s) / k
        # weights
        w = e / s  # (P,)
        # build dense symmetric weight matrix
        W = np.zeros((n, n))
        W[iu0] = w
        W = W + W.T
        GX = W @ X  # sum_j w_ij X_j
        # gradient w.r.t Y
        dot = np.sum(GX * X, axis=1, keepdims=True)
        Gt = (GX - dot * X) / nr
        return f, Gt.ravel()
    try:
        res = minimize(fun_grad, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': maxiter*2})
        Y = res.x.reshape(n, d)
        return normalize(Y)
    except Exception:
        return x0

def lp_polish(x, seconds):
    x = normalize(x)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = normalize(y)
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        best = rng.standard_normal((n, d))
    best = normalize(best)
    try:
        bv = max_cos(best)
    except Exception:
        bv = 1.0
    # also try record
    try:
        rec = normalize(np.asarray(np.load("record.npy"), dtype=float))
        rv = max_cos(rec)
        if rv < bv and rec.shape == (n, d):
            best = rec.copy()
            bv = rv
    except Exception:
        pass

    incumbent = best.copy()
    inc_val = bv
    cur = best.copy()
    cur_val = bv
    cands = [(bv, best.copy())]

    SEARCH_END = t0 + 112
    trial = 0
    # temperature schedule
    while time.time() < SEARCH_END:
        trial += 1
        progress = (time.time() - t0) / max(1e-9, (SEARCH_END - t0))
        Temp = 0.004 * (1.0 - 0.8 * progress)  # ~0.004 -> 0.0008
        # choose base: mostly current (basin hopping), sometimes incumbent/best
        r = rng.random()
        if r < 0.70:
            base = cur
        elif r < 0.88:
            base = incumbent
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.28:
            sigma = float(rng.choice(np.array([0.01, 0.03, 0.07, 0.15, 0.28])))
            x = x + sigma * rng.standard_normal(x.shape)
            x = normalize(x)
        elif mode < 0.62:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(3, 11))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            for j in idx:
                nc = 250
                cand = rng.standard_normal((nc, d))
                cand = normalize(cand)
                dots = cand @ Xkeep.T
                md = dots.max(axis=1)
                order = np.argsort(md)
                # pick among top few for diversity
                if rng.random() < 0.75:
                    pick = int(order[0])
                else:
                    pick = int(order[int(rng.integers(1, min(6, nc)))])
                x[j] = cand[pick]
                Xkeep = np.vstack([Xkeep, x[j:j+1]])
        elif mode < 0.82:
            cdir = rng.standard_normal(d)
            cdir /= np.linalg.norm(cdir)
            dots = x @ cdir
            thr = float(rng.uniform(0.60, 0.93))
            mask = dots > thr
            if int(mask.sum()) == 0:
                mask[int(rng.integers(n))] = True
            mcnt = int(mask.sum())
            sigma = float(rng.uniform(0.10, 0.38))
            # with prob randomize part of cap completely
            if rng.random() < 0.35:
                nrand = min(mcnt, int(rng.integers(1, 4)))
                midx = np.where(mask)[0]
                rng.shuffle(midx)
                for j in midx[:nrand]:
                    v = rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                    # bias toward cap center with some spread
                    v = v * 0.6 + cdir * 0.8
                    v /= np.linalg.norm(v)
                    x[j] = v
                rest = midx[nrand:]
                if len(rest) > 0:
                    x[rest] += sigma * rng.standard_normal((len(rest), d))
            else:
                x[mask] += sigma * rng.standard_normal((mcnt, d))
            x = normalize(x)
        else:
            # contact-break: push apart worst pairs
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            top = np.argsort(-c)[:25]
            sel = top[int(rng.integers(len(top)))]
            i, j = int(iu[0][sel]), int(iu[1][sel])
            alpha = float(rng.uniform(0.08, 0.30))
            x[i] = normalize((x[i] - alpha * x[j]).reshape(1, -1))[0]
            x[j] = normalize((x[j] - alpha * x[i]).reshape(1, -1))[0]
            # add small noise to neighbours
            neigh = np.argsort(-G[i])[:5]
            x[neigh] += 0.02 * rng.standard_normal((len(neigh), d))
            x = normalize(x)

        if time.time() > SEARCH_END - 2:
            break
        # choose relax method
        u = rng.random()
        try:
            if u < 0.55:
                k0 = float(rng.choice(np.array([5.0, 10.0, 15.0, 20.0])))
                k1 = float(rng.choice(np.array([150.0, 250.0, 400.0, 600.0])))
                lr0 = float(rng.choice(np.array([0.02, 0.03, 0.05])))
                lr1 = float(rng.choice(np.array([0.0008, 0.002, 0.004])))
                mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
                iters = int(rng.choice(np.array([350, 550, 850, 1100])))
                xo = exp_relax(x, iters, k0, k1, lr0, lr1, mu, rng)
            elif u < 0.78:
                s = float(rng.choice(np.array([6.0, 10.0, 16.0, 24.0])))
                lr0 = float(rng.choice(np.array([0.03, 0.05, 0.08])))
                lr1 = float(rng.choice(np.array([0.001, 0.003])))
                mu = float(rng.choice(np.array([0.8, 0.9])))
                iters = int(rng.choice(np.array([350, 600, 900])))
                th = float(rng.choice(np.array([-1.0, 0.0, 0.3])))
                xo = riesz_relax(x, iters, s, lr0, lr1, mu, rng, thresh=th)
            else:
                # L-BFGS smooth max
                kk = float(rng.choice(np.array([80.0, 150.0, 250.0])))
                mi = int(rng.choice(np.array([80, 140, 200])))
                # short pre-relax to stabilize
                xpre = exp_relax(x, 150, 10.0, 80.0, 0.03, 0.005, 0.85, rng)
                xo = lbfgs_lse(xpre, kk, mi)
        except Exception:
            continue
        try:
            v = max_cos(xo)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        # Metropolis accept into cur
        if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-9)):
            cur = xo.copy()
            cur_val = v
        # occasional restart of cur to incumbent if stuck
        if trial % 30 == 0 and cur_val > inc_val + 0.01:
            cur = incumbent.copy()
            cur_val = inc_val
        # scratch restart trial
        if trial % 22 == 0:
            xr = rng.standard_normal((n, d))
            xr = normalize(xr)
            try:
                xo2 = exp_relax(xr, 500, 12.0, 250.0, 0.04, 0.002, 0.85, rng)
                v2 = max_cos(xo2)
                if v2 < inc_val - 1e-12:
                    inc_val = v2
                    incumbent = xo2.copy()
                    cands.append((v2, xo2.copy()))
                    cands = sorted(cands, key=lambda z: z[0])[:5]
                if v2 < cur_val and rng.random() < 0.5:
                    cur = xo2.copy()
                    cur_val = v2
            except Exception:
                pass

    cands = sorted(cands, key=lambda z: z[0])
    # ensure best included
    if bv < cands[0][0] - 1e-15:
        cands = [(bv, best.copy())] + cands
        cands = cands[:5]
    time_left = 172 - (time.time() - t0)
    if time_left < 12:
        time_left = 12
    # polish top 3
    polished_best = None
    polished_val = 1e9
    num_p = min(3, len(cands))
    for i in range(num_p):
        v0, xc = cands[i]
        budget = time_left / num_p
        if i == 0:
            budget *= 1.2
        budget = min(budget, 45)
        # short high-k exp sharpen before LP for best diversity
        try:
            if time.time() - t0 < 168:
                xc2 = exp_relax(xc.copy(), 250, 80.0, 600.0, 0.008, 0.0008, 0.9, rng)
            else:
                xc2 = xc
        except Exception:
            xc2 = xc
        try:
            xp = lp_polish(xc2, budget)
        except Exception:
            xp = xc2
        try:
            vp = max_cos(xp)
        except Exception:
            continue
        if vp < polished_val:
            polished_val = vp
            polished_best = xp.copy()
        if time.time() - t0 > 173:
            break
    if polished_best is None:
        polished_best = incumbent
    polished_best = normalize(polished_best)
    # return better of polished and original best
    try:
        pv = max_cos(polished_best)
    except Exception:
        return normalize(best)
    if bv < pv:
        return normalize(best)
    return normalize(polished_best)
