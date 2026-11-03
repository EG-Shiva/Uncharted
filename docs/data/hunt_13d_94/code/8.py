import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    worst = G.max(axis=1)
    cnt_tight = (G > m - 1e-9).sum(axis=1)
    cnt_loose = (G > m - 1e-4).sum(axis=1)
    return m, worst, cnt_tight, cnt_loose, G

def _adam_refine(x, steps, k0, k1, lr0, lr1, b1=0.9, b2=0.999):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    eps = 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac if k0 != k1 else k0
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        W = np.exp(k * (G - mval))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
        if not np.all(np.isfinite(x)):
            break
    return x

def _mom_refine(x, steps, k0, k1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = np.max(G - np.eye(n) * 2.0)
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Grad = E @ x / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _reinsert_randomized(others, rng, n_cand=8000, steps=120, pick_top=24, kk=50.0):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    idx = np.argpartition(scores, pick_top)[:pick_top]
    # random among top to diversify, bias to best
    if rng.random() < 0.6:
        c = cand[idx[0]].copy()
    else:
        c = cand[rng.choice(idx)].copy()
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(kk * (dots - m))
        w /= w.sum()
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.15 * (1.0 - 0.5 * it / steps)
        c = c - step * gt
        nc = np.linalg.norm(c)
        if nc == 0 or not np.isfinite(nc):
            break
        c /= nc
    return c

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    T0 = time.time()
    BUDGET = 170.0
    HOP_END = T0 + 138.0
    rng = np.random.default_rng(12345)
    # load
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    x_best = best.copy()
    cur_x = best.copy(); cur_val = best_val
    n, d = x_best.shape
    it = 0
    sigmas = [0.003, 0.008, 0.02, 0.045, 0.09, 0.16]
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - T0
        # temperature anneal
        T = 0.0022 * max(0.05, 1.0 - elapsed / 140.0)
        T = max(T, 0.00015)
        m, worst, cntT, cntL, G = _per_point(cur_x)
        y = cur_x.copy()
        mode = it % 7
        if mode == 0:
            # worst-edge separation avalanche: push top edges apart
            iu = np.triu_indices(n, 1)
            gv = G[iu]
            order = np.argsort(-gv)[:6]
            for k in order[:3]:
                a = iu[0][k]; b = iu[1][k]
                diff = y[a] - y[b]
                # tangential component
                diff = diff - (diff @ y[a]) * y[a]
                dn = np.linalg.norm(diff)
                if dn > 1e-12:
                    diff /= dn
                    y[a] = y[a] + (0.02 + 0.02 * rng.random()) * diff
                diff2 = y[b] - y[a]
                # recompute roughly opposite
                y[b] = y[b] - (0.02 + 0.02 * rng.random()) * diff if dn > 1e-12 else y[b] + 0.02 * rng.standard_normal(d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.5:
                y = _adam_refine(y, steps=700, k0=18, k1=350, lr0=0.02, lr1=0.003)
            else:
                y = _mom_refine(y, steps=700, k0=22, k1=320, lr0=0.25, mu=0.87)
        elif mode == 1:
            # jammed core avalanche
            jam = np.argsort(-cntL)[:10]
            loose = np.argsort(cntL)[:10]
            y[jam] += 0.22 * rng.standard_normal(size=(len(jam), d))
            y[loose] += 0.06 * rng.standard_normal(size=(len(loose), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_refine(y, steps=800, k0=15, k1=300, lr0=0.025, lr1=0.004)
        elif mode == 2:
            # randomized hole re-insertion of 1-2 worst points
            ow = np.argsort(-worst)
            nkill = 1 if rng.random() < 0.6 else 2
            victims = []
            victims.append(int(ow[0]))
            if nkill == 2:
                # second: worst or loose rattler-like
                if rng.random() < 0.5:
                    victims.append(int(ow[1]))
                else:
                    victims.append(int(np.argsort(cntL)[0]))
            victims = list(dict.fromkeys(victims))
            for idx in victims:
                others = np.delete(y, idx, axis=0)
                y[idx] = _reinsert_randomized(others, rng, n_cand=8000, steps=120, pick_top=24)
            y += 0.004 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _mom_refine(y, steps=650, k0=30, k1=380, lr0=0.2, mu=0.88)
        elif mode == 3:
            # global shake new basin
            sigma = sigmas[rng.integers(0, len(sigmas))]
            base = x_best if rng.random() < 0.4 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_refine(y, steps=750, k0=20, k1=400, lr0=0.02, lr1=0.003)
        elif mode == 4:
            # neighbourhood rotation: pick cluster of nearest neighbours
            c0 = int(rng.integers(0, n))
            dists = (y @ y[c0])
            nn = np.argsort(-dists)[1:9]
            # random tangent rotation: add correlated + orthogonal noise
            shift = rng.standard_normal(d) * 0.12
            shift = shift - (shift @ y[c0]) * y[c0]
            for idx in np.concatenate([[c0], nn]):
                y[idx] = y[idx] + shift * (0.5 + 0.5 * rng.random()) + 0.05 * rng.standard_normal(d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _mom_refine(y, steps=600, k0=25, k1=300, lr0=0.22, mu=0.86)
        elif mode == 5:
            # small exploitation around best
            kpick = int(rng.integers(5, 12))
            idx = rng.choice(n, size=kpick, replace=False)
            y = x_best.copy()
            y[idx] += 0.05 * rng.standard_normal(size=(kpick, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_refine(y, steps=550, k0=40, k1=500, lr0=0.012, lr1=0.002)
        else:
            # occasional fresh random restart with long relax
            if it % 28 == 0:
                y = rng.standard_normal(size=(n, d))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _adam_refine(y, steps=1500, k0=12, k1=250, lr0=0.03, lr1=0.005)
                y = _mom_refine(y, steps=500, k0=30, k1=400, lr0=0.18, mu=0.9)
            else:
                sigma = 0.03 + 0.05 * rng.random()
                y = cur_x + sigma * rng.standard_normal(size=y.shape)
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                y = _mom_refine(y, steps=600, k0=25, k1=330, lr0=0.22, mu=0.87)
        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy(); best_val = new_val
            cur_x = y.copy(); cur_val = new_val
        else:
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy(); cur_val = new_val
        # early exit if time low
        if time.time() - T0 > BUDGET - 35:
            break

    # final tightening + LP polish with remaining time
    remain = BUDGET - (time.time() - T0) - 1.5
    if remain > 6:
        try:
            cand = _adam_refine(x_best.copy(), steps=400, k0=80, k1=700, lr0=0.008, lr1=0.002)
            v = _max_cos(cand)
            if v < best_val:
                x_best = cand; best_val = v
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=max(5.0, remain - 1.0), t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol; best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x_best) > _max_cos(x0) + 1e-12:
            # keep better of the two (must not regress)
            if _max_cos(x0) < _max_cos(x_best):
                x_best = x0
    except Exception:
        pass
    return x_best
