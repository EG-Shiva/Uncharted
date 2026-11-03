import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Grad = (W @ x) / rs
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        # limit step
        tn = np.sqrt((Tan * Tan).sum(axis=1, keepdims=True)) + 1e-12
        # clip extreme
        scale = np.minimum(1.0, 1.0 / (tn * 20.0 + 1e-12))
        # actually simple: x - lr*Tan
        x = x - lr * Tan
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1
        ).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)
        ).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(
                obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                b_eq=np.zeros(n),
                bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _e8_roots():
    r = []
    d = 8
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    r.append(v)
    # half-integer set, even number of minus signs (128 vectors)
    for mask in range(256):
        v = np.ones(d) * 0.5
        nm = 0
        for k in range(d):
            if (mask >> k) & 1:
                v[k] = -0.5
                nm += 1
        if nm % 2 == 0:
            r.append(v)
    R = np.stack(r, axis=0)  # 240 x 8
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R


def _find_voids(Xkeep, k, rng, trials=4000, refine=150):
    nkeep, d = Xkeep.shape
    cand = rng.normal(size=(trials, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ Xkeep.T
    score = S.max(axis=1)
    order = np.argsort(score)
    picked = []
    for idx in order:
        v = cand[idx]
        ok = True
        for u in picked:
            if float(v @ u) > 0.85:
                ok = False
                break
        if ok:
            picked.append(v.copy())
        if len(picked) >= k:
            break
    if len(picked) < k:
        for idx in order:
            if len(picked) >= k:
                break
            picked.append(cand[idx].copy())
    P = np.stack(picked, axis=0)
    # refine each void against keep-set (and each other progressively)
    for a in range(k):
        v = P[a].copy()
        for s in range(refine):
            # softmax weights to keepers + already refined voids
            Gk = Xkeep @ v  # (nkeep,)
            m = Gk.max()
            beta = 40.0 + 160.0 * (s / max(refine - 1, 1))
            w = np.exp(beta * (Gk - m))
            g = (w[:, None] * Xkeep).sum(axis=0) / (w.sum() + 1e-12)
            # repulsion from other new points
            if k > 1:
                Go = P @ v
                Go[a] = -2.0
                mo = Go.max()
                wo = np.exp(beta * (Go - mo))
                wo[a] = 0.0
                ssum = wo.sum()
                if ssum > 1e-12:
                    g = g + 0.7 * ((wo[:, None] * P).sum(axis=0) / ssum)
            dot = float(g @ v)
            tan = g - dot * v
            lr = 0.08 * (1.0 - s / refine) + 0.005
            v = v - lr * tan
            v = v / np.linalg.norm(v)
        P[a] = v
    return P


def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    try:
        E8 = _e8_roots()
    except Exception:
        E8 = None

    it = 0
    # hopping loop, reserve ~8s for final polish
    while time.time() < deadline - 8.0 and it < 300:
        it += 1
        now = time.time()
        budget = min(deadline - 8.0, now + 2.5)
        mode = it % 10
        try:
            if mode == 0 and E8 is not None:
                # E8-subset fresh basin
                perm = rng.permutation(len(E8))[:n]
                y = E8[perm] + 0.02 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=900, beta0=8.0, beta1=280.0, lr0=0.08, lr1=0.005)
                if _max_cos(y) < best_val + 0.01 and time.time() < budget:
                    y = _lp_polish(y, min(budget, time.time() + 1.0))
            elif mode == 1:
                # pure random fresh basin
                y = rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=1000, beta0=8.0, beta1=300.0, lr0=0.08, lr1=0.005)
                if _max_cos(y) < best_val + 0.01 and time.time() < budget:
                    y = _lp_polish(y, min(budget, time.time() + 1.0))
            elif mode in (2, 3, 6):
                # kill-and-rebirth of worst points into voids
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                # near-contact count and worst max per point
                cnt = (G > cmax - 0.015).sum(axis=1)
                wmax = G.max(axis=1)
                score = wmax + 0.002 * cnt
                k = 6 if mode == 2 else (4 if mode == 3 else 9)
                kill = np.argsort(-score)[:k]
                mask = np.ones(n, dtype=bool)
                mask[kill] = False
                Xkeep = cur[mask]
                P = _find_voids(Xkeep, k, rng, trials=4000, refine=120)
                y = cur.copy()
                y[kill] = P
                # small jitter elsewhere
                y = y + 0.008 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=650, beta0=15.0, beta1=260.0, lr0=0.06, lr1=0.004)
                if _max_cos(y) < best_val + 0.008 and time.time() < budget:
                    y = _lp_polish(y, min(budget, time.time() + 1.2))
            elif mode in (4, 8):
                # targeted shake of overloaded cluster
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.01).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:24]
                y = cur.copy()
                sigma_big = 0.30 if mode == 4 else 0.15
                y[pool] += sigma_big * rng.normal(size=(len(pool), d))
                y += 0.008 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal(y, steps=700, beta0=12.0, beta1=260.0, lr0=0.06, lr1=0.004)
                if _max_cos(y) < best_val + 0.008 and time.time() < budget:
                    y = _lp_polish(y, min(budget, time.time() + 1.2))
            else:
                # small Gaussian basin hop around best/cur
                base = best if (it % 2 == 0) else cur
                sigma = 0.03 if (it % 3 == 0) else 0.06
                y = base + sigma * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                # occasionally re-insert one random point
                if it % 5 == 0:
                    idx = rng.choice(n, size=3, replace=False)
                    rp = rng.normal(size=(3, d))
                    rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                    y[idx] = rp
                y = _anneal(y, steps=600, beta0=18.0, beta1=280.0, lr0=0.05, lr1=0.004)
                if _max_cos(y) < best_val + 0.008 and time.time() < budget:
                    y = _lp_polish(y, min(budget, time.time() + 1.0))
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        else:
            # Metropolis wander for cur to change basin
            if v < cur_val or rng.random() < 0.04:
                cur = y.copy()
                cur_val = v
            # periodically reset cur to best
            if it % 15 == 0:
                cur = best.copy()
                cur_val = best_val

    try:
        best = _lp_polish(best, deadline - 0.5)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
