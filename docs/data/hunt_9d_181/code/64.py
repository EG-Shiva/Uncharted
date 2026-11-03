import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, init_radius=1e-3, pair_cap=1500, it_cap=25):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < it_cap:
        it += 1
        if time.time() - start + 0.15 > seconds and radius < 5e-4:
            break
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > pair_cap:
            kth = np.partition(c_all, -pair_cap)[-pair_cap]
            sel = c_all >= kth
            # trim to pair_cap tightest if still over
            idx = np.where(sel)[0]
            if len(idx) > pair_cap + 200:
                order = np.argsort(-c_all[idx])[:pair_cap]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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


def smooth_exp(x, steps=150, k0=50.0, k1=220.0, lr0=0.02, lr1=0.003):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ x
        dot = np.einsum('ij,ij->i', Gr, x)[:, None]
        Gt = Gr - dot * x
        nrm = np.linalg.norm(Gt, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        x = x - (lr / (nrm + 1e-15)) * Gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def block_rotate(x, rng, theta, frac_lo=0.2, frac_hi=0.55):
    n, d = x.shape
    a0 = rng.normal(size=d)
    a0 /= np.linalg.norm(a0) + 1e-15
    proj = x @ a0
    q = rng.uniform(frac_lo, frac_hi)
    thr = np.quantile(proj, q)
    if rng.random() < 0.5:
        S = np.where(proj > thr)[0]
    else:
        S = np.where(proj <= thr)[0]
    if len(S) < 10 or len(S) > n - 10:
        return x
    # random plane
    u = rng.normal(size=d)
    u /= np.linalg.norm(u) + 1e-15
    v = rng.normal(size=d)
    v = v - (v @ u) * u
    v /= np.linalg.norm(v) + 1e-15
    au = x[S] @ u
    bv = x[S] @ v
    rest = x[S] - np.outer(au, u) - np.outer(bv, v)
    c, s = np.cos(theta), np.sin(theta)
    au2 = c * au - s * bv
    bv2 = s * au + c * bv
    y = x.copy()
    y[S] = rest + np.outer(au2, u) + np.outer(bv2, v)
    y[S] /= np.linalg.norm(y[S], axis=1, keepdims=True)
    return y


def contact_info(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    np.fill_diagonal(G, 1.0)
    cnt = np.sum(G > m - tol, axis=1) - 1
    return m, cnt, G


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
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
    if rem > 12:
        xb = lp_polish(xb, seconds=min(3.0, rem - 8.0), init_radius=2e-3)
        bb = max_cos(xb)
    xc = xb.copy()
    bc = bb
    n, d = xb.shape
    T = 3e-05
    it = 0
    while True:
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        it += 1
        mode = it % 6
        y = xc.copy()
        if mode == 1:
            th = float(rng.uniform(0.05, 0.12))
            y = block_rotate(y, rng, th)
            y = y + rng.normal(size=y.shape) * 0.003
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.02
        elif mode == 2:
            th = float(rng.uniform(0.12, 0.28))
            y = block_rotate(y, rng, th, 0.3, 0.7)
            y = y + rng.normal(size=y.shape) * 0.005
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.03
        elif mode == 3:
            # hub dispersal: shake top hubs + neighbours
            try:
                m, cnt, G = contact_info(y)
                order = np.argsort(-cnt)
                hubs = order[:5]
                # neighbours of worst hub
                sims = (y @ y[hubs[0]])
                nn = np.argsort(-sims)[1:11]
                grp = np.unique(np.concatenate([hubs, nn]))
                y[grp] = y[grp] + rng.normal(size=(len(grp), d)) * 0.12
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                y = y + rng.normal(size=y.shape) * 0.06
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.025
        elif mode == 4:
            # low-contact hole filling (rattlers + 1-4 contact points)
            try:
                m, cnt, G = contact_info(y)
                order = np.argsort(cnt)
                nr = 3 if rng.random() < 0.7 else 4
                targets = order[:nr]
                for r in targets:
                    nt = 6000
                    R = rng.normal(size=(nt, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    S = R @ y.T
                    S[:, r] = -10.0
                    sc = S.max(axis=1)
                    y[r] = R[int(np.argmin(sc))]
                y = y + rng.normal(size=y.shape) * 0.004
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                pass
            init_r = 0.03
        elif mode == 5:
            # avalanche: separate several tight pairs at once
            try:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(n, 1)
                c = G[iu]
                K = 40
                o = np.argsort(-c)[:K]
                picks = o[rng.choice(len(o), size=min(8, len(o)), replace=False)]
                for pk in picks:
                    i0 = int(iu[0][pk]); j0 = int(iu[1][pk])
                    diff = y[i0] - y[j0]
                    diff /= (np.linalg.norm(diff) + 1e-12)
                    y[i0] = y[i0] + diff * 0.04 + rng.normal(size=d) * 0.015
                    y[j0] = y[j0] - diff * 0.04 + rng.normal(size=d) * 0.015
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                y = y + rng.normal(size=y.shape) * 0.05
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.02
        else:
            y = y + rng.normal(size=y.shape) * 0.035
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            init_r = 0.02
        # cheap collective smoothing
        y = smooth_exp(y, steps=140, k0=50.0, k1=220.0, lr0=0.02, lr1=0.003)
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 3.0:
            break
        budg = min(1.6, rem2 - 1.2)
        if budg < 0.7:
            budg = max(0.5, rem2 - 0.8)
        y2 = lp_polish(y, seconds=float(budg), init_radius=init_r)
        v = max_cos(y2)
        if v < bb - 1e-15:
            xb = y2.copy()
            bb = v
            xc = y2.copy()
            bc = v
        else:
            if v < bc or rng.random() < np.exp((bc - v) / max(T, 1e-12)):
                xc = y2.copy()
                bc = v
        T *= 0.9993
        if T < 7e-06:
            T = 7e-06
        # restart current from best occasionally to avoid drift
        if it % 25 == 0:
            xc = xb.copy()
            bc = bb
    remf = BUDGET - (time.time() - t0)
    if remf > 2.0:
        try:
            yf = lp_polish(xb, seconds=min(remf - 0.6, 5.0), init_radius=8e-4, pair_cap=2000, it_cap=30)
            vf = max_cos(yf)
            if vf < bb:
                xb = yf
                bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
