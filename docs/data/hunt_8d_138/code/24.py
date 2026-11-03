import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_mom(x, steps, beta0, beta1, lr0, lr1, mom=0.88, noise=0.0, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # stability clip
        A = beta * (G - m)
        np.clip(A, -50.0, 0.0, out=A)
        E = np.exp(A)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-12
        Grad = (E @ x) / rs
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Tan = Grad - dot * x
        if noise > 0.0 and rng is not None:
            amp = noise * (1.0 - f)
            N = rng.normal(size=(n, d))
            N = N - (N * x).sum(axis=1, keepdims=True) * x
            Tan = Tan + amp * N
        v = mom * v + Tan
        x = x - lr * v
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
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _e8_roots():
    d = 8
    r = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = si
                    v[j] = sj
                    r.append(v)
    for mask in range(256):
        v = np.ones(d) * 0.5
        nm = 0
        for k in range(d):
            if (mask >> k) & 1:
                v[k] = -0.5
                nm += 1
        if nm % 2 == 0:
            r.append(v)
    R = np.stack(r, axis=0)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R

def _find_voids(Xkeep, k, rng, trials=20000, refine=80):
    nkeep, d = Xkeep.shape
    cand = rng.normal(size=(trials, d))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    S = cand @ Xkeep.T
    smax = S.max(axis=1)
    order = np.argsort(smax)
    picked = []
    for idx in order:
        v = cand[idx]
        ok = True
        for u in picked:
            if float(v @ u) > 0.80:
                ok = False
                break
        if ok:
            picked.append(v.copy())
        if len(picked) >= k:
            break
    while len(picked) < k:
        picked.append(cand[order[len(picked) % len(order)]].copy())
    P = np.stack(picked, axis=0)
    for a in range(k):
        v = P[a].copy()
        others = np.delete(P, a, axis=0) if k > 1 else None
        for s in range(refine):
            beta = 40.0 + 160.0 * (s / max(refine - 1, 1))
            Gk = Xkeep @ v
            m = Gk.max()
            w = np.exp(np.clip(beta * (Gk - m), -50, 0))
            g = (w[:, None] * Xkeep).sum(axis=0) / (w.sum() + 1e-12)
            if others is not None and len(others) > 0:
                Go = others @ v
                mo = Go.max()
                wo = np.exp(np.clip(beta * (Go - mo), -50, 0))
                if wo.sum() > 1e-12:
                    g = g + 0.6 * ((wo[:, None] * others).sum(axis=0) / wo.sum())
            g = g - float(g @ v) * v
            lr = 0.07 * (1.0 - s / refine) + 0.005
            v = v - lr * g
            v = v / (np.linalg.norm(v) + 1e-30)
        P[a] = v
    return P

def _rupture_top_edge(y, rng, strength=0.25):
    n = y.shape[0]
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    ij = np.unravel_index(int(np.argmax(G)), G.shape)
    a, b = int(ij[0]), int(ij[1])
    dvec = y[a] - y[b]
    dvec = dvec - 0.5 * float(dvec @ (y[a] + y[b])) * (y[a] + y[b]) / 2.0
    # tangentialize
    dvec = dvec - float(dvec @ y[a]) * y[a]
    dn = np.linalg.norm(dvec) + 1e-12
    dvec = dvec / dn
    # push apart
    y[a] = y[a] + strength * dvec
    y[b] = y[b] - strength * dvec
    # jitter neighbours of a,b
    order_a = np.argsort(-G[a])[:12]
    order_b = np.argsort(-G[b])[:12]
    for idx in list(order_a) + list(order_b):
        y[idx] = y[idx] + 0.03 * rng.normal(size=(y.shape[1],))
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 170.0
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
    try:
        E8 = _e8_roots()
    except Exception:
        E8 = None
    cur = best.copy()
    cur_val = best_val
    it = 0
    while time.time() < deadline - 10.0 and it < 1000:
        it += 1
        mode = it % 8
        y = None
        try:
            if mode == 0:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.015).sum(axis=1)
                wmax = G.max(axis=1)
                score = wmax + 0.0025 * cnt
                k = int(rng.choice([4, 6, 8, 10]))
                kill = np.argsort(-score)[:k]
                mask = np.ones(n, dtype=bool)
                mask[kill] = False
                Xkeep = cur[mask]
                P = _find_voids(Xkeep, k, rng, trials=20000, refine=70)
                y = cur.copy()
                y[kill] = P
                y = y + 0.006 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_mom(y, steps=650, beta0=12.0, beta1=320.0, lr0=0.018, lr1=0.002, mom=0.88, noise=0.004, rng=rng)
            elif mode == 1:
                y = cur.copy()
                y = _rupture_top_edge(y, rng, strength=0.20 + 0.15 * rng.random())
                y = y + 0.006 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_mom(y, steps=600, beta0=15.0, beta1=340.0, lr0=0.016, lr1=0.002, mom=0.90, noise=0.002, rng=rng)
            elif mode == 2:
                # rattler mobilization: move all loose points to voids
                G = best @ best.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                loose = np.where(per < best_val - 0.002)[0]
                if len(loose) < 4:
                    loose = np.argsort(per)[-8:]
                    # actually take least constrained
                    loose = np.argsort(per)[:0]
                    # fallback: take rattlers by threshold
                    loose = np.where(per < G.max() - 0.005)[0]
                if len(loose) > 12:
                    # pick 12 worst-loose randomly
                    loose = rng.choice(loose, size=12, replace=False)
                k = len(loose)
                if k == 0:
                    y = best + 0.05 * rng.normal(size=(n, d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    y = _anneal_mom(y, steps=550, beta0=18.0, beta1=320.0, lr0=0.014, lr1=0.002, mom=0.88, rng=rng)
                else:
                    mask = np.ones(n, dtype=bool)
                    mask[loose] = False
                    Xkeep = best[mask]
                    P = _find_voids(Xkeep, k, rng, trials=20000, refine=70)
                    y = best.copy()
                    y[loose] = P
                    y = y + 0.005 * rng.normal(size=(n, d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    y = _anneal_mom(y, steps=600, beta0=14.0, beta1=340.0, lr0=0.016, lr1=0.002, mom=0.89, noise=0.002, rng=rng)
            elif mode == 3:
                sigma = float(rng.choice([0.03, 0.05, 0.08, 0.12]))
                base = best if (it % 2 == 0) else cur
                y = base + sigma * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_mom(y, steps=600, beta0=16.0, beta1=360.0, lr0=0.014, lr1=0.0015, mom=0.90, noise=0.001, rng=rng)
            elif mode == 4:
                if E8 is not None and (it % 3 == 0):
                    perm = rng.permutation(len(E8))[:n]
                    y = E8[perm] + 0.03 * rng.normal(size=(n, d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    y = _anneal_mom(y, steps=800, beta0=8.0, beta1=320.0, lr0=0.018, lr1=0.002, mom=0.88, noise=0.003, rng=rng)
                else:
                    y = rng.normal(size=(n, d))
                    y = y / np.linalg.norm(y, axis=1, keepdims=True)
                    y = _anneal_mom(y, steps=800, beta0=8.0, beta1=320.0, lr0=0.018, lr1=0.002, mom=0.88, noise=0.003, rng=rng)
            elif mode == 5:
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                cmax = G.max()
                cnt = (G > cmax - 0.01).sum(axis=1)
                order = np.argsort(-cnt)
                pool = order[:24]
                y = cur.copy()
                y[pool] += (0.25 if (it % 2 == 0) else 0.12) * rng.normal(size=(len(pool), d))
                y += 0.006 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_mom(y, steps=600, beta0=12.0, beta1=320.0, lr0=0.016, lr1=0.002, mom=0.89, noise=0.002, rng=rng)
            elif mode == 6:
                # second rupture + rebirth single worst
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                victim = int(np.argmax(per + 0.001 * rng.normal(size=(n,))))
                Xrem = np.delete(cur, victim, axis=0)
                P = _find_voids(Xrem, 1, rng, trials=25000, refine=100)
                y = cur.copy()
                y[victim] = P[0]
                y = y + 0.004 * rng.normal(size=(n, d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
                y = _anneal_mom(y, steps=550, beta0=18.0, beta1=360.0, lr0=0.014, lr1=0.0015, mom=0.90, rng=rng)
            else:
                base = best
                y = base + 0.02 * rng.normal(size=(n, d))
                # reinsert 2 random
                idx = rng.choice(n, size=2, replace=False)
                rp = rng.normal(size=(2, d))
                rp = rp / np.linalg.norm(rp, axis=1, keepdims=True)
                y[idx] = rp
                y = _anneal_mom(y, steps=550, beta0=14.0, beta1=340.0, lr0=0.015, lr1=0.002, mom=0.89, rng=rng)
        except Exception:
            continue
        try:
            v0 = _max_cos(y)
        except Exception:
            continue
        # selective LP
        try:
            if v0 < best_val + 0.008 and time.time() < deadline - 10.0:
                y = _lp_polish(y, min(deadline - 10.0, time.time() + 0.9))
                v0 = _max_cos(y)
        except Exception:
            pass
        if v0 < best_val - 1e-12:
            best_val = v0
            best = y.copy()
            cur = y.copy()
            cur_val = v0
        else:
            if v0 < cur_val or rng.random() < 0.05:
                cur = y.copy()
                cur_val = v0
            if it % 20 == 0:
                cur = best.copy()
                cur_val = best_val
    try:
        # final intensive: alternate anneal + LP
        if time.time() < deadline - 1.0:
            best = _anneal_mom(best, steps=400, beta0=80.0, beta1=600.0, lr0=0.008, lr1=0.001, mom=0.90, rng=rng)
        best = _lp_polish(best, deadline - 0.4)
        # one more squeeze at ultra-high beta
        if time.time() < deadline - 2.0:
            best = _anneal_mom(best, steps=250, beta0=200.0, beta1=800.0, lr0=0.005, lr1=0.0008, mom=0.90, rng=rng)
            best = _lp_polish(best, deadline - 0.3)
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
