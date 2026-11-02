import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _smooth_polish(x0, steps, tau0=0.015, tau1=0.0015, lr0=0.03, lr1=0.003, mu=0.7):
    n, d = x0.shape
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    vel = np.zeros_like(x)
    best_x = x.copy()
    # initial max
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_x = x.copy()
        diff = (G - m) / tau
        # only significant weights
        W = np.zeros_like(G)
        mask = diff > -12.0
        # G has -inf on diagonal -> diff -inf -> mask False -> stays 0
        W[mask] = np.exp(diff[mask])
        # F = weighted mean of neighbours
        rowsum = W.sum(axis=1, keepdims=True) + 1e-12
        F = (W @ x) / rowsum
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        # clip tangential force to avoid blowup
        fn = np.linalg.norm(Ft, axis=1, keepdims=True)
        # optional clip at 1.0
        np.clip(fn, 0, 1.0, out=fn)
        vel = mu * vel - lr * Ft
        x = x + vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    # final check
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        return x, mf
    return best_x, best_m


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cols.reshape(p, 2 * d + 1)[:, :d] = I[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols.reshape(p, 2 * d + 1)[:, 2 * d] = n * d
        vals.reshape(p, 2 * d + 1)[:, :d] = x[J]
        vals.reshape(p, 2 * d + 1)[:, d:2 * d] = x[I]
        vals.reshape(p, 2 * d + 1)[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
    return x


def _greedy_replace(Y, victims, rng, ncand=3000):
    n, d = Y.shape
    for k in victims:
        cand = rng.standard_normal(size=(ncand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Y.T
        dots[:, k] = -2.0
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        Y[k] = cand[bi]
    return Y


def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    bestX = None
    bestV = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < bestV:
                    bestV = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        bestX = a
        bestV = _max_cos(a)

    n, d = 323, 7
    NITER = 120
    for it in range(NITER):
        if time.time() - t0 > 125:
            break
        Y = bestX.copy()
        # analyse contacts
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        worst_pp = G.max(axis=1)
        contacts = np.sum(G > bestV - 1e-6, axis=1)
        hub_order = np.argsort(-contacts)
        rattlers = np.where(contacts == 0)[0]
        mode = it % 3
        if mode == 0:
            k = [4, 7, 10, 3][it % 4]
        elif mode == 1:
            k = [6, 3, 12, 5][it % 4]
        else:
            k = 9
        victims = []
        # include a rattler cyclically
        if len(rattlers) > 0 and (it % 2 == 0):
            victims.append(int(rattlers[it % len(rattlers)]))
        # include hubs
        nhub = min(2, k)
        # pick hubs deterministically via rng
        perm = rng.permutation(hub_order[:12])[:nhub]
        for h in perm:
            victims.append(int(h))
        if mode == 2:
            # cluster kick: neighbours of a hub/center
            center = int(hub_order[it % n]) if it % 2 == 0 else int(rng.integers(n))
            dots_c = bestX[center] @ Y.T
            # nearest (largest dot) excluding center
            nn = np.argsort(-dots_c)
            cnt = 0
            for u in nn:
                if int(u) == center:
                    continue
                if int(u) not in victims:
                    victims.append(int(u))
                    cnt += 1
                if len(victims) >= k:
                    break
        # fill rest randomly biased to worst
        while len(victims) < k:
            # 50% uniform, 50% worst-biased
            if rng.random() < 0.5:
                c = int(rng.integers(n))
            else:
                # sample 8 candidates, pick worst among them
                cand_idx = rng.integers(0, n, size=8)
                c = int(cand_idx[np.argmax(worst_pp[cand_idx])])
            if c not in victims:
                victims.append(c)
        victims = victims[:k]

        if mode == 1:
            # jitter all slightly + greedy replace victims
            sig = 0.015 if (it % 4) < 2 else 0.03
            Y = Y + sig * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            _greedy_replace(Y, victims, rng, ncand=2500)
        else:
            _greedy_replace(Y, victims, rng, ncand=3000)
            # small jitter on victims' neighbours to break symmetry
            if it % 5 == 0:
                Y += 0.005 * rng.standard_normal(size=Y.shape)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        steps = 600 + (it % 3) * 250
        Yp, vp = _smooth_polish(Y, steps=steps)
        if vp < bestV - 1e-12:
            bestX = Yp / np.linalg.norm(Yp, axis=1, keepdims=True)
            bestV = float(vp)

    # final snap with LP using remaining time
    remain = 172.0 - (time.time() - t0)
    if remain > 8:
        lp_time = min(remain - 2.0, 40.0)
        bestX = _lp_polish(bestX, lp_time)
        bestV = _max_cos(bestX)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
