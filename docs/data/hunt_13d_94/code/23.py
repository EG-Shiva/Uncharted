import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_softmax(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
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

def _adam_riesz(x, steps, p0, p1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        p = p0 + (p1 - p0) * frac
        lr = lr0 * (lr1 / lr0) ** frac if lr0 != lr1 else lr0
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        # cap to avoid overflow when points coincide
        np.clip(G, -1.0, 0.999999, out=G)
        np.fill_diagonal(G, -10.0)
        D2 = 2.0 - 2.0 * G
        D2[D2 < 1e-9] = 1e-9
        W = D2 ** (-p / 2.0 - 1.0)
        np.fill_diagonal(W, 0.0)
        # normalize to keep step stable
        sc = np.median(W[W > 0]) if np.any(W > 0) else 1.0
        if sc > 0 and np.isfinite(sc):
            W = W / sc
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

def _best_hole(others, rng, n_cand=15000, pick_top=16, steps=100, kk=60.0):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    idx = np.argpartition(scores, pick_top)[:pick_top]
    # bias to best but randomize
    if rng.random() < 0.55:
        c = cand[idx[0]].copy()
    else:
        c = cand[rng.choice(idx)].copy()
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(kk * (dots - m))
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.2 * (1.0 - 0.6 * it / steps)
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
    HOP_END = T0 + 136.0
    rng = np.random.default_rng(7)
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
    sigmas = [0.003, 0.008, 0.02, 0.05, 0.10, 0.17]
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - T0
        T = 0.0025 * max(0.05, 1.0 - elapsed / 140.0)
        T = max(T, 0.00012)
        G = cur_x @ cur_x.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        worst = G.max(axis=1)
        cnt = (G > mx - 1e-4).sum(axis=1)
        y = cur_x.copy()
        mode = it % 6
        if mode == 0:
            # worst-edge split avalanche
            iu = np.triu_indices(n, 1)
            gv = G[iu]
            order = np.argsort(-gv)[:8]
            for kk in order[:4]:
                a = int(iu[0][kk]); b = int(iu[1][kk])
                diff = y[a] - y[b]
                diff = diff - (diff @ y[a]) * y[a]
                dn = np.linalg.norm(diff)
                if dn > 1e-12:
                    y[a] = y[a] + (0.02 + 0.025 * rng.random()) * diff / dn
                diff2 = y[b] - y[a]
                diff2 = diff2 - (diff2 @ y[b]) * y[b]
                dn2 = np.linalg.norm(diff2)
                if dn2 > 1e-12:
                    y[b] = y[b] + (0.02 + 0.025 * rng.random()) * diff2 / dn2
                else:
                    y[b] = y[b] + 0.02 * rng.standard_normal(d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.5:
                y = _adam_softmax(y, steps=380, k0=20, k1=380, lr0=0.02, lr1=0.003)
            else:
                y = _adam_riesz(y, steps=300, p0=6, p1=14, lr0=0.015, lr1=0.003)
        elif mode == 1:
            # hemisphere shear: collective slip to rewire contact graph
            ax = rng.standard_normal(d); ax /= np.linalg.norm(ax)
            proj = y @ ax
            med = np.median(proj)
            mask = proj > med
            if mask.sum() == 0 or mask.sum() == n:
                mask = proj > 0
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            amp = 0.10 + 0.15 * rng.random()
            for i in np.where(mask)[0]:
                ut = u - (u @ y[i]) * y[i]
                y[i] = y[i] + amp * (0.6 + 0.4 * rng.random()) * ut
            for i in np.where(~mask)[0]:
                ut = u - (u @ y[i]) * y[i]
                y[i] = y[i] - amp * (0.6 + 0.4 * rng.random()) * ut * 0.7
            y += 0.01 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=420, k0=15, k1=320, lr0=0.022, lr1=0.003)
        elif mode == 2:
            # multi-point hole re-insertion (1-3 worst)
            ow = np.argsort(-worst)
            nkill = 1 if rng.random() < 0.45 else (2 if rng.random() < 0.7 else 3)
            victims = [int(ow[0])]
            if nkill >= 2:
                victims.append(int(ow[1]) if rng.random() < 0.6 else int(np.argsort(cnt)[0]))
            if nkill >= 3:
                victims.append(int(ow[2]))
            victims = list(dict.fromkeys(victims))
            # delete sequentially, reinsert one by one
            tmp = y.copy()
            for idx in victims:
                others = np.delete(tmp, idx, axis=0)
                tmp[idx] = _best_hole(others, rng, n_cand=12000, pick_top=16, steps=90, kk=55.0)
            y = tmp
            y += 0.003 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=420, k0=30, k1=420, lr0=0.018, lr1=0.0025)
        elif mode == 3:
            # global shake from best or cur
            sigma = sigmas[rng.integers(0, len(sigmas))]
            base = x_best if rng.random() < 0.45 else cur_x
            y = base + sigma * rng.standard_normal(size=base.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.5:
                y = _adam_softmax(y, steps=400, k0=18, k1=380, lr0=0.02, lr1=0.003)
            else:
                y = _adam_riesz(y, steps=320, p0=8, p1=16, lr0=0.012, lr1=0.002)
                y = _adam_softmax(y, steps=200, k0=60, k1=450, lr0=0.01, lr1=0.002)
        elif mode == 4:
            # cluster excision: pick center + nearest neighbours, large noise there
            c0 = int(rng.integers(0, n))
            dots = y @ y[c0]
            nn = np.argsort(-dots)[0:9]
            y[nn] += (0.15 + 0.15 * rng.random()) * rng.standard_normal(size=(len(nn), d))
            # small noise elsewhere
            y += 0.005 * rng.standard_normal(size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _adam_softmax(y, steps=450, k0=12, k1=300, lr0=0.025, lr1=0.0035)
        else:
            # Riesz balancing exploitation around best
            kpick = int(rng.integers(6, 14))
            idx = rng.choice(n, size=kpick, replace=False)
            y = x_best.copy()
            y[idx] += 0.06 * rng.standard_normal(size=(kpick, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            if rng.random() < 0.6:
                y = _adam_softmax(y, steps=350, k0=40, k1=520, lr0=0.012, lr1=0.002)
            else:
                y = _adam_riesz(y, steps=280, p0=10, p1=18, lr0=0.010, lr1=0.002)
                y = _adam_softmax(y, steps=180, k0=80, k1=500, lr0=0.008, lr1=0.002)
            if it % 30 == 0:
                # occasional fresh Riesz start
                yr = rng.standard_normal(size=(n, d))
                yr /= np.linalg.norm(yr, axis=1, keepdims=True)
                yr = _adam_riesz(yr, steps=500, p0=4, p1=12, lr0=0.02, lr1=0.004)
                yr = _adam_softmax(yr, steps=300, k0=20, k1=350, lr0=0.018, lr1=0.003)
                vy = _max_cos(yr)
                if vy < best_val:
                    x_best = yr.copy(); best_val = vy
                if vy < cur_val or rng.random() < np.exp((cur_val - vy) / max(T, 1e-4)):
                    cur_x = yr.copy(); cur_val = vy
        new_val = _max_cos(y)
        if new_val < best_val - 1e-15:
            x_best = y.copy(); best_val = new_val
            cur_x = y.copy(); cur_val = new_val
            # intensify immediately with short LP-free tighten
            if time.time() - T0 < BUDGET - 30:
                try:
                    yt = _adam_softmax(x_best.copy(), steps=250, k0=100, k1=650, lr0=0.008, lr1=0.0015)
                    vt = _max_cos(yt)
                    if vt < best_val:
                        x_best = yt; best_val = vt; cur_x = yt.copy(); cur_val = vt
                except Exception:
                    pass
        else:
            if new_val < cur_val or rng.random() < np.exp((cur_val - new_val) / T):
                cur_x = y.copy(); cur_val = new_val
        if time.time() - T0 > BUDGET - 35:
            break
    remain = BUDGET - (time.time() - T0) - 1.0
    if remain > 6:
        try:
            cand = _adam_softmax(x_best.copy(), steps=450, k0=80, k1=750, lr0=0.007, lr1=0.0015)
            v = _max_cos(cand)
            if v < best_val:
                x_best = cand; best_val = v
        except Exception:
            pass
        try:
            x_pol = _lp_polish(x_best, seconds=max(5.0, remain - 1.5), t0=time.time())
            pv = _max_cos(x_pol)
            if pv < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol; best_val = pv
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x0) + 1e-12 < _max_cos(x_best):
            x_best = x0
    except Exception:
        pass
    return x_best
