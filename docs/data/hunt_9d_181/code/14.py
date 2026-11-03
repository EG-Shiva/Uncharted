import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, init_radius=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start + 0.25 > seconds and radius < 5e-4:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = max(20 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            kth = np.partition(c_all, -2500)[-2500]
            sel = c_all >= kth
            idx = np.where(sel)[0]
            if len(idx) > 2800:
                order = np.argsort(-c_all[idx])[:2800]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def smooth_anneal(x, iters=200, p_start=25, p_end=70, lr=0.04):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        if m <= 1e-12:
            break
        Gp = np.where(G > 0, G, 0.0)
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nrm
        x = x - step * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    cand = []
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        tmp = rng.normal(size=(181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cand.append(tmp)
    scores = [max_cos(a) for a in cand]
    bi = int(np.argmin(scores))
    xb = cand[bi].copy()
    bb = scores[bi]
    rem = BUDGET - (time.time() - t0)
    if rem > 10:
        xb = lp_polish(xb, seconds=min(3.0, rem - 6.0), init_radius=2e-3)
        bb = max_cos(xb)
    xc = xb.copy()
    bc = bb
    n, d = xb.shape
    T = 4e-05
    it = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 3.0:
            break
        it += 1
        y = xc.copy()
        mode = it % 6
        if mode == 0:
            y = y + rng.normal(size=y.shape) * 0.008
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.008
            sm = False
        elif mode == 1:
            y = y + rng.normal(size=y.shape) * 0.03
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            sm = True
        elif mode == 2:
            k = 2 + (it % 2)
            idx = rng.choice(n, size=k, replace=False)
            R = rng.normal(size=(k, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            y[idx] = R
            init_r = 0.03
            sm = True
        elif mode == 3:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            order = np.argsort(contacts)
            for r in order[:3]:
                nt = 4000
                R = rng.normal(size=(nt, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                S = R @ y.T
                S[:, r] = -10.0
                sc = S.max(axis=1)
                y[r] = R[int(np.argmin(sc))]
            y = y + rng.normal(size=y.shape) * 0.005
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            sm = True
        elif mode == 4:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            np.fill_diagonal(G, 1.0)
            contacts = np.sum(G > m - 1e-7, axis=1) - 1
            top = int(np.argmax(contacts))
            sims = G[top].copy()
            sims[top] = -10
            nn = np.argsort(-sims)[:10]
            grp = np.concatenate([[top], nn])
            y[grp] = y[grp] + rng.normal(size=(len(grp), d)) * 0.12
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025
            sm = True
        else:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            K = 50
            o = np.argsort(-c)[:K]
            pick = o[rng.integers(0, K)]
            i0 = int(iu[0][pick]); j0 = int(iu[1][pick])
            diff = y[i0] - y[j0]
            diff /= (np.linalg.norm(diff) + 1e-12)
            y[i0] = y[i0] + diff * 0.05 + rng.normal(size=d) * 0.02
            y[j0] = y[j0] - diff * 0.05 + rng.normal(size=d) * 0.02
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.015
            sm = True
        if sm:
            y = smooth_anneal(y, iters=200, p_start=25, p_end=70, lr=0.04)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 2.5:
            break
        budg = min(2.6, rem2 - 1.0)
        if budg < 0.8:
            budg = max(0.6, rem2 - 0.5)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-14:
            xb = y2.copy()
            bb = v
            xc = y2.copy()
            bc = v
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 8:
                xb2 = lp_polish(xb, seconds=1.2, init_radius=1e-3)
                vv = max_cos(xb2)
                if vv < bb:
                    xb = xb2
                    bb = vv
                    xc = xb2.copy()
                    bc = vv
        else:
            if v < bc or rng.random() < np.exp((bc - v) / T):
                xc = y2.copy()
                bc = v
        T *= 0.9995
        if T < 8e-06:
            T = 8e-06
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = lp_polish(xb, seconds=min(remf - 0.5, 4.0), init_radius=1e-3)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf
                bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
