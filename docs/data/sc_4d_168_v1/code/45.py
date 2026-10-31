import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * rad, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best = y, v
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x, best

def _fib(n):
    pts = np.zeros((n, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * (i + 0.5) / max(1, n)
        r = np.sqrt(max(0.0, 1 - y * y))
        th = ga * i
        pts[i] = [r * np.cos(th), y, r * np.sin(th)]
    return pts

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 136.0
    rng = np.random.RandomState(20261031)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _nm(xr)
        xb = _nm(xb)
        if _mc(xr) < _mc(xb):
            xb = xr.copy()
    except Exception:
        xb = _nm(xb)
    if xb.shape != (168, 4):
        xb = _nm(rng.randn(168, 4))
    xb = _nm(xb)
    n, d = 168, 4
    best_val = _mc(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val

    def refine(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88, balance=False):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        scale = np.ones(n)
        for s in range(steps):
            if time.time() > SEARCH_END:
                break
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            if balance and (s % 100 == 0) and s > 0:
                G0 = x @ x.T
                np.fill_diagonal(G0, -2.0)
                m0 = float(np.max(G0))
                deg = np.sum(G0 > m0 - 0.005, axis=1)
                mean = np.mean(deg) + 1e-9
                scale = 1.0 + 0.8 * (deg - mean) / mean
                scale = np.clip(scale, 0.5, 2.5)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            if balance:
                gt = gt * scale[:, None]
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _nm(x)
        return x

    def refine_single(s0, Xref, steps=100, beta0=40.0, beta1=250.0, lr0=0.09, lr1=0.006):
        s = s0 / (np.linalg.norm(s0) + 1e-30)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for it in range(steps):
            f = it / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = Xref @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ Xref
            gt = g - np.dot(g, s) * s
            gn = np.linalg.norm(gt) + 1e-12
            s = s - lr * gt / gn
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4); ns = np.linalg.norm(s)
            s = s / ns
        return s

    def best_void(Xref, nsamp=8000):
        C = rng.randn(nsamp, 4)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Xref.T
        m = D.max(axis=1)
        return C[int(np.argmin(m))]

    def layered_init():
        nl = int(rng.choice([6, 7, 8, 9]))
        if nl % 2 == 1:
            half = nl // 2
            hh = np.sort(rng.uniform(0.05, 0.92, size=half))
            h = np.concatenate([-hh[::-1], [0.0], hh])
        else:
            half = nl // 2
            hh = np.sort(rng.uniform(0.05, 0.92, size=half))
            h = np.concatenate([-hh[::-1], hh])
        w = (1 - h ** 2) + 0.2 * rng.rand(nl)
        w /= w.sum()
        cnt = np.floor(w * n).astype(int)
        while cnt.sum() < n:
            cnt[rng.randint(nl)] += 1
        while cnt.sum() > n:
            cnt[int(np.argmax(cnt))] -= 1
        X = []
        for hi, mi in zip(h, cnt):
            r = np.sqrt(max(0.0, 1 - hi * hi))
            if mi <= 0:
                continue
            if mi == 1:
                v = rng.randn(3); v /= np.linalg.norm(v)
                X.append(np.array([hi, r * v[0], r * v[1], r * v[2]]))
            else:
                pts = _fib(mi)
                Q, _ = np.linalg.qr(rng.randn(3, 3))
                pts = pts @ Q.T
                for p in pts:
                    X.append(np.array([hi, r * p[0], r * p[1], r * p[2]]))
        X = np.array(X)
        Q4, _ = np.linalg.qr(rng.randn(4, 4))
        X = X @ Q4.T
        X = X + 0.008 * rng.randn(*X.shape)
        return _nm(X)

    def grid_init():
        triples = [(4, 6, 7), (3, 7, 8), (2, 12, 7), (3, 8, 7), (4, 7, 6), (2, 14, 6)]
        na, nb, nc = triples[rng.randint(len(triples))]
        avals = (np.arange(na) + 0.5) * (np.pi / 2.0) / na
        # jitter a slightly
        avals = avals + 0.03 * rng.randn(na)
        avals = np.clip(avals, 0.03, np.pi / 2.0 - 0.03)
        X = []
        for ai in avals:
            ca, sa = np.cos(ai), np.sin(ai)
            for j in range(nb):
                bj = 2 * np.pi * j / nb + 0.05 * rng.randn()
                for k in range(nc):
                    ck = 2 * np.pi * k / nc + 0.05 * rng.randn()
                    X.append([ca * np.cos(bj), ca * np.sin(bj), sa * np.cos(ck), sa * np.sin(ck)])
        X = np.array(X)
        if X.shape[0] != n:
            # fallback: random fill/trim
            if X.shape[0] > n:
                idx = rng.choice(X.shape[0], n, replace=False)
                X = X[idx]
            else:
                extra = rng.randn(n - X.shape[0], 4)
                extra = _nm(extra)
                X = np.vstack([X, extra])
        Q4, _ = np.linalg.qr(rng.randn(4, 4))
        X = X @ Q4.T
        X = X + 0.008 * rng.randn(*X.shape)
        return _nm(X)

    # Phase A: diverse structured pool
    for _ in range(14):
        if time.time() > SEARCH_END - 20:
            break
        if rng.rand() < 0.5:
            x0 = layered_init()
        else:
            x0 = grid_init()
        x0 = refine(x0, 350, 20.0, 450.0, 0.018, 0.0009, balance=True)
        v0 = _mc(x0)
        if v0 < best_val:
            x_best, best_val = x0.copy(), v0
            x_cur, cur_val = x0.copy(), v0
        elif v0 < cur_val + 0.008 and rng.rand() < 0.3:
            x_cur, cur_val = x0.copy(), v0
        # quick LP if surprisingly good
        if v0 < best_val + 0.0005 and time.time() < SEARCH_END - 5:
            xs, vs = _lp_polish_deadline(x0, min(time.time() + 0.8, SEARCH_END))
            if vs < best_val:
                x_best, best_val = xs.copy(), vs
                x_cur, cur_val = xs.copy(), vs

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.0:
            break
        base = x_best if (trial % 5 != 0 or cur_val > best_val + 0.004) else x_cur
        # use cur occasionally
        if trial % 7 == 0 and cur_val < best_val + 0.004:
            base = x_cur
        typ = trial % 12
        x = base.copy()
        if typ == 0:
            x = _nm(x + 0.012 * rng.randn(n, d))
            x = refine(x, 450, 25.0, 500.0, 0.014, 0.0007)
        elif typ == 1:
            x = _nm(x + 0.05 * rng.randn(n, d))
            x = refine(x, 550, 30.0, 700.0, 0.016, 0.0007, balance=True)
        elif typ == 2:
            x = _nm(x + 0.15 * rng.randn(n, d))
            x = refine(x, 600, 20.0, 500.0, 0.018, 0.0008)
        elif typ in (3, 4):
            # cap ruin: delete geographic cap of 10-22 pts
            c0 = rng.randn(4); c0 /= np.linalg.norm(c0)
            dots = x @ c0
            k = int(rng.choice([10, 12, 14, 16, 18, 22]))
            victims = np.argsort(-dots)[:k]
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xk = x[keep]
            newp = []
            for si in range(k):
                Xr = Xk if len(newp) == 0 else np.vstack([Xk, np.array(newp)])
                cnd = best_void(Xr, nsamp=8000)
                xn = refine_single(cnd, Xr, steps=90, beta0=45.0, beta1=260.0)
                newp.append(xn)
            x[victims] = np.array(newp)
            x = _nm(x + 0.005 * rng.randn(n, d))
            x = refine(x, 550, 28.0, 650.0, 0.015, 0.0007, balance=(typ == 4))
        elif typ == 5:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > m - 0.004, axis=1)
            k = int(rng.choice([3, 4, 5, 6]))
            victims = np.argsort(-deg, kind="stable")[:k].copy()
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xk = x[keep]
            newp = []
            for si in range(k):
                Xr = Xk if len(newp) == 0 else np.vstack([Xk, np.array(newp)])
                cnd = best_void(Xr, nsamp=9000)
                xn = refine_single(cnd, Xr, steps=100, beta0=45.0, beta1=280.0)
                newp.append(xn)
            x[victims] = np.array(newp)
            x = _nm(x + 0.006 * rng.randn(n, d))
            x = refine(x, 500, 30.0, 700.0, 0.015, 0.0007, balance=True)
        elif typ == 6:
            x = layered_init()
            x = refine(x, 500, 22.0, 550.0, 0.017, 0.0008, balance=True)
        elif typ == 7:
            x = grid_init()
            x = refine(x, 500, 22.0, 550.0, 0.017, 0.0008, balance=True)
        elif typ == 8:
            c = rng.randint(n)
            G = x @ x.T
            nbh = np.argsort(-G[c])[:int(rng.choice([5, 7, 9]))]
            x[nbh] = x[nbh] + 0.16 * rng.randn(len(nbh), d)
            x = _nm(x)
            x = refine(x, 500, 25.0, 600.0, 0.016, 0.0007)
        elif typ == 9:
            x = _nm(x + 0.02 * rng.randn(n, d))
            x = refine(x, 450, 80.0, 1500.0, 0.010, 0.0005, mu=0.85)
        elif typ == 10:
            k = int(rng.choice([1, 2, 3]))
            victims = rng.choice(n, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            Xk = x[keep]
            newp = []
            for si in range(k):
                Xr = Xk if len(newp) == 0 else np.vstack([Xk, np.array(newp)])
                cnd = best_void(Xr, nsamp=10000)
                xn = refine_single(cnd, Xr, steps=110, beta0=35.0, beta1=300.0)
                newp.append(xn)
            x[victims] = np.array(newp)
            x = _nm(x + 0.004 * rng.randn(n, d))
            x = refine(x, 500, 30.0, 700.0, 0.014, 0.0006)
        else:
            x = _nm(x + 0.28 * rng.randn(n, d))
            x = refine(x, 650, 15.0, 400.0, 0.020, 0.0009)
        v = _mc(x)
        if v < cur_val:
            x_cur, cur_val = x.copy(), v
        elif v < cur_val + 0.003 and rng.rand() < 0.12:
            x_cur, cur_val = x.copy(), v
        if trial % 25 == 0 and cur_val > best_val + 0.005:
            x_cur, cur_val = x_best.copy(), best_val
        if v < best_val + 0.0008 and time.time() < SEARCH_END - 3.0:
            if v < best_val or (v < best_val + 0.00015 and (trial % 3 == 0)):
                xs, vs = _lp_polish_deadline(x, min(time.time() + 1.3, SEARCH_END))
                if vs < best_val:
                    x_best, best_val = xs.copy(), vs
                    x_cur, cur_val = xs.copy(), vs
                elif vs < cur_val:
                    x_cur, cur_val = xs.copy(), vs
        else:
            if v < best_val:
                x_best, best_val = x.copy(), v
                x_cur, cur_val = x.copy(), v
    try:
        x_best, best_val = _lp_polish_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    return _nm(x_best)
