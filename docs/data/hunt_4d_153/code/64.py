import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _counts(x, tol=1e-7):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    g = (x @ x.T)[iu]
    m = g.max()
    tight = g > m - tol
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, iu[0][tight], 1)
    np.add.at(cnt, iu[1][tight], 1)
    return cnt, float(m)

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
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu, mask=None):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
    if mask is None:
        mv = None
    else:
        mv = np.zeros(n, dtype=bool)
        mv[np.asarray(mask)] = True
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        if mv is not None:
            gt[~mv] = 0.0
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        if mv is not None:
            step[~mv] = 0.0
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _best_hole(x, idx, rng, nsamp=12000):
    n = x.shape[0]
    others = np.delete(x, idx, axis=0)
    S = rng.normal(size=(nsamp, 4))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    D = S @ others.T
    scores = D.max(axis=1)
    b = int(np.argmin(scores))
    return S[b]

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(7)
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
    iu = np.triu_indices(n, 1)
    elites = [(best_val, x_best.copy())]
    it = 0
    while time.time() - t0 < 140.0:
        it += 1
        if time.time() - t0 > HARD - 30:
            break
        # pick base: mostly best, sometimes 2nd elite for splice diversity
        if len(elites) > 1 and (it % 9 == 8) and rng.random() < 0.5:
            base = elites[1][1]
        else:
            base = x_best
        cnt, mx = _counts(base)
        ratt = np.where(cnt <= 2)[0]
        low = np.where(cnt <= 3)[0]
        hubs = np.where(cnt >= 7)[0]
        order = np.argsort(cnt)
        mode = it % 10
        y = base.copy()
        if mode == 0:
            # rattler / low hole fill + local then global
            targets = list(ratt[:4])
            if len(targets) == 0:
                targets = list(order[:2])
            if len(low) > len(targets):
                # add one more low
                for c in order[:6]:
                    if c not in targets:
                        targets.append(c)
                        break
            targets = targets[:3]
            for idx in targets:
                y[int(idx)] = _best_hole(y, int(idx), rng, nsamp=15000)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 250, 30.0, 150.0, 0.01, 0.001, iu, mask=targets)
            y = _smooth_opt(y, 650, 40.0, 260.0, 0.006, 0.0004, iu)
        elif mode == 1:
            # break one hub by full reinsertion at hole
            if len(hubs) > 0:
                h = int(rng.choice(hubs))
            else:
                h = int(order[-1])
            y[h] = _best_hole(y, h, rng, nsamp=15000)
            # jitter its old neighbours slightly
            G = base @ base.T
            G[h, h] = -2
            nbrs = np.argsort(-G[h])[:6]
            y[nbrs] += rng.normal(0, 0.02, size=(len(nbrs), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 800, 35.0, 300.0, 0.006, 0.0004, iu)
        elif mode == 2:
            # random cap re-optimization with frozen background
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(8, 21))
            mask = np.argsort(-proj)[:k]
            y[mask] += rng.normal(0, 0.05, size=(k, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 300, 25.0, 150.0, 0.012, 0.001, iu, mask=mask)
            y = _smooth_opt(y, 600, 40.0, 280.0, 0.005, 0.0004, iu)
        elif mode == 3:
            # balanced small shake: hubs move more
            sig = np.full(n, 0.015)
            sig[hubs] = 0.035
            if len(ratt) > 0:
                sig[ratt] = 0.05
            y += rng.normal(0, 1.0, size=y.shape) * sig[:, None]
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 750, 40.0, 280.0, 0.006, 0.0004, iu)
        elif mode == 4:
            # medium shake + 2 random reinsertions
            y += rng.normal(0, 0.06, size=y.shape)
            rep = rng.choice(n, size=2, replace=False)
            y[rep] = rng.normal(size=(2, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 850, 25.0, 320.0, 0.007, 0.0004, iu)
        elif mode == 5:
            # smash 3 hubs hard
            vic = list(hubs[:0])
            if len(hubs) >= 3:
                vic = list(rng.choice(hubs, size=3, replace=False))
            else:
                vic = list(order[-3:])
            for h in vic:
                if rng.random() < 0.5:
                    y[int(h)] = rng.normal(size=4)
                else:
                    y[int(h)] += rng.normal(0, 0.3, size=4)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 900, 22.0, 350.0, 0.007, 0.0004, iu)
        elif mode == 6:
            # boost all low-contact points
            if len(low) == 0:
                low = order[:3]
            y[low] += rng.normal(0, 0.08, size=(len(low), d))
            # one hole-guided reinsert for the worst
            w = int(order[0])
            if cnt[w] <= 3 and rng.random() < 0.7:
                y[w] = _best_hole(y, w, rng, nsamp=12000)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 750, 35.0, 280.0, 0.006, 0.0004, iu)
        elif mode == 7:
            # rigid rotation of a cap (topology change preserving local structure)
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(10, 25))
            mask = np.argsort(-proj)[:k]
            A = rng.normal(size=(4, 4))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            # small rotation: interpolate toward Q
            alpha = float(rng.uniform(0.15, 0.5))
            R = np.eye(4) * (1 - alpha) + Q * alpha
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            y[mask] = y[mask] @ R.T
            y[mask] += rng.normal(0, 0.02, size=(len(mask), d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 800, 30.0, 300.0, 0.006, 0.0004, iu)
        elif mode == 8:
            # splice two elites across random plane
            if len(elites) > 1:
                A2 = elites[0][1]; B2 = elites[1][1]
            else:
                A2 = x_best; B2 = x_best + rng.normal(0, 0.05, size=x_best.shape)
                B2 /= np.linalg.norm(B2, axis=1, keepdims=True)
            u = rng.normal(size=4); u /= np.linalg.norm(u)
            pa = A2 @ u; pb = B2 @ u
            # take roughly half from each to keep 153
            ia = np.argsort(-pa); ib = np.argsort(pb)
            k = 153 // 2
            sel = np.concatenate([ia[:k], ib[:153 - k]])
            # build by nearest matching: simply stack chosen rows
            y = np.concatenate([A2[ia[:k]], B2[ib[:153 - k]]], axis=0)
            y += rng.normal(0, 0.02, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 900, 25.0, 300.0, 0.007, 0.0004, iu)
        else:
            # large global kick
            y += rng.normal(0, 0.11, size=y.shape)
            rep = rng.choice(n, size=1, replace=False)
            y[rep] = rng.normal(size=(1, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, 850, 22.0, 330.0, 0.007, 0.0004, iu)
        v2 = _max_cos(y)
        thresh = best_val + (0.005 if mode in (4, 5, 8, 9) else 0.002)
        if v2 < thresh:
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 1.0:
                break
            b = min(2.6, max(1.0, remain / 10.0))
            # only polish the most promising more thoroughly
            if v2 > best_val + 0.001:
                b = min(b, 1.2)
            yp, vp = _lp_polish(y, b)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
                elites.append((vp, yp.copy()))
                elites = sorted(elites, key=lambda t: t[0])[:3]
            elif vp < best_val + 0.0008 and len(elites) < 3:
                elites.append((vp, yp.copy()))
                elites = sorted(elites, key=lambda t: t[0])[:3]
            elif v2 < best_val - 1e-12:
                # smooth alone improved but LP failed to improve; keep smooth
                if v2 < best_val:
                    x_best, best_val = y, v2
                    elites.append((v2, y.copy()))
                    elites = sorted(elites, key=lambda t: t[0])[:3]
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 26.0)
        x_best, best_val = _lp_polish(x_best, remain)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
