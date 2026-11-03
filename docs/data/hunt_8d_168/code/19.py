import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel() if False else np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    n, d = 168, 8
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.normal(size=(n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_x = min(cands, key=_max_cos)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    best_val = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_val = best_val

    def quench(x_init, steps, b0, b1, lr0, lr1, mu, noise=0.0):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0, lb1 = np.log(b0), np.log(b1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for s in range(steps):
            frac = s / max(steps - 1, 1)
            beta = np.exp(lb0 + (lb1 - lb0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # softmax weights, row-normalised (stable)
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            S = W.sum(axis=1, keepdims=True) + 1e-12
            A = (W @ x) / S
            dot = np.sum(A * x, axis=1, keepdims=True)
            Gt = A - dot * x
            if noise > 0:
                # decaying Langevin tangent noise
                amp = noise * (1.0 - frac)
                Z = rng.normal(size=x.shape)
                Zd = np.sum(Z * x, axis=1, keepdims=True)
                Zt = Z - Zd * x
                Gt = Gt + amp * Zt
            vel = mu * vel + Gt
            x = x - lr * vel
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
        return x

    def refine_hole(Y, p, iters=200, beta=120.0, lr0=0.12, lr1=0.004):
        p = p / np.linalg.norm(p)
        for it in range(iters):
            frac = it / max(iters - 1, 1)
            lr = lr0 * (lr1 / lr0) ** frac
            g = Y @ p
            m = g.max()
            w = np.exp(beta * (g - m))
            s = w.sum() + 1e-12
            A = (w @ Y) / s
            At = A - float(A @ p) * p
            p = p - lr * At
            nn = np.linalg.norm(p)
            if nn == 0:
                break
            p /= nn
        return p

    def find_holes(Y, k, n_probe=16000):
        P = rng.normal(size=(n_probe, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        best_s = np.empty(n_probe)
        CH = 4000
        for a in range(0, n_probe, CH):
            b = min(a + CH, n_probe)
            D = P[a:b] @ Y.T
            best_s[a:b] = D.max(axis=1)
        order = np.argsort(best_s)
        sel = []
        for idx in order:
            if len(sel) >= k:
                break
            p = P[idx]
            ok = True
            for q in sel:
                if abs(float(p @ q)) > 0.90:
                    ok = False
                    break
            if ok:
                sel.append(p)
        if len(sel) < k:
            for idx in order:
                if len(sel) >= k:
                    break
                p = P[idx]
                dup = False
                for q in sel:
                    if np.allclose(p, q):
                        dup = True
                        break
                if not dup:
                    sel.append(p)
        holes = [refine_hole(Y, p, iters=180, beta=float(rng.uniform(60, 160))) for p in sel[:k]]
        return np.array(holes)

    def contact_counts(X, thr=0.003):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        return (G > m - thr).sum(axis=1), m

    # short diverse warm starts
    for _ in range(1):
        if time.time() - t0 > BUDGET - 60:
            break
        x0 = rng.normal(size=(n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        xq = quench(x0, 1500, 35.0, 160.0, 0.07, 0.005, 0.85)
        v = _max_cos(xq)
        if v < best_val:
            best_val, best_x = v, xq
            cur_val, cur_x = v, xq.copy()

    it = 0
    while time.time() - t0 < BUDGET - 36:
        it += 1
        # annealing temperature for sideways moves
        frac_time = (time.time() - t0) / (BUDGET - 36)
        temp = 3e-5 * (1.0 - frac_time) + 2e-7
        r = rng.random()
        base = cur_x if rng.random() < 0.65 else best_x
        cnt, mval = contact_counts(base, thr=0.003)
        # strict rattlers
        Gtmp = base @ base.T
        np.fill_diagonal(Gtmp, -2.0)
        ratt = np.where((Gtmp > mval - 1e-4).sum(axis=1) == 0)[0]
        if r < 0.45:
            # rattler-to-hole rewiring (topology change)
            k = int(rng.integers(4, 12))
            if len(ratt) >= k:
                # prefer rattlers + low-contact points
                sel = rng.choice(ratt, size=k, replace=False)
            else:
                order = np.argsort(cnt)
                pool = order[:max(30, 2 * k)]
                sel = rng.choice(pool, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[sel] = False
            Y = base[keep]
            try:
                holes = find_holes(Y, k)
            except Exception:
                holes = rng.normal(size=(k, d))
                holes /= np.linalg.norm(holes, axis=1, keepdims=True)
            x0 = base.copy()
            x0[sel] = holes
            x0 += rng.normal(size=(n, d)) * 0.003
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1400, 2600))
            b0 = float(rng.uniform(30, 60)); b1 = float(rng.uniform(140, 320))
            lr0 = float(rng.uniform(0.04, 0.085)); lr1 = float(rng.uniform(0.002, 0.007))
            mu = float(rng.uniform(0.75, 0.90))
            nz = float(rng.uniform(0.0, 0.02))
            xq = quench(x0, steps, b0, b1, lr0, lr1, mu, noise=nz)
        elif r < 0.72:
            # jammed-hub release: heavy kick to top contacts
            k = int(rng.integers(3, 9))
            order = np.argsort(-cnt)
            pool = order[:30]
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            x0 = base.copy()
            s_big = float(rng.uniform(0.20, 0.60))
            x0[sel] += rng.normal(size=(len(sel), d)) * s_big
            # also relocate 1-2 rattlers to holes occasionally
            if len(ratt) > 0 and rng.random() < 0.5:
                kk = min(len(ratt), int(rng.integers(1, 3)))
                rs = rng.choice(ratt, size=kk, replace=False)
                keep2 = np.ones(n, dtype=bool); keep2[rs] = False
                try:
                    holes = find_holes(base[keep2], kk)
                    x0[rs] = holes
                except Exception:
                    pass
            x0 += rng.normal(size=(n, d)) * 0.006
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1400, 2600))
            b0 = float(rng.uniform(30, 65)); b1 = float(rng.uniform(140, 350))
            lr0 = float(rng.uniform(0.035, 0.075)); lr1 = float(rng.uniform(0.002, 0.007))
            mu = float(rng.uniform(0.75, 0.90))
            xq = quench(x0, steps, b0, b1, lr0, lr1, mu, noise=0.005)
        else:
            # global / subset jitter
            if rng.random() < 0.5:
                sigma = 10.0 ** float(rng.uniform(-2.3, -0.9))
                x0 = base + rng.normal(size=(n, d)) * sigma
            else:
                k = int(rng.integers(5, 16))
                sel = rng.choice(n, size=k, replace=False)
                x0 = base.copy()
                x0[sel] = rng.normal(size=(k, d)) / np.linalg.norm(rng.normal(size=(k, d)), axis=1, keepdims=True)
                x0 += rng.normal(size=(n, d)) * 0.004
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1200, 2400))
            b0 = float(rng.uniform(30, 60)); b1 = float(rng.uniform(120, 280))
            lr0 = float(rng.uniform(0.03, 0.08)); lr1 = float(rng.uniform(0.002, 0.008))
            mu = float(rng.uniform(0.70, 0.90))
            xq = quench(x0, steps, b0, b1, lr0, lr1, mu)
        v = _max_cos(xq)
        if v < best_val - 1e-12:
            best_val, best_x = v, xq
            cur_val, cur_x = v, xq.copy()
            # extra fine quench on improvement
            if time.time() - t0 < BUDGET - 55:
                xq2 = quench(best_x, 900, 110.0, 400.0, 0.016, 0.002, 0.82)
                v2 = _max_cos(xq2)
                if v2 < best_val:
                    best_val, best_x = v2, xq2
                    cur_val, cur_x = v2, xq2.copy()
        elif v < cur_val + temp and rng.random() < 0.35:
            cur_val, cur_x = v, xq.copy()
        if time.time() - t0 > BUDGET - 36:
            break

    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 5:
        best_x = _lp_polish(best_x, min(remain, 34.0))
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
