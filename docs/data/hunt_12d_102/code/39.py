import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def soft_quench(x_init, steps, T0, T1, lr0, mu):
    x = x_init.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        frac = s / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if S == 0 or not np.isfinite(S):
            break
        # global-normalized, scaled by n for O(1) magnitude
        Grad = (W @ x) * (n / S)
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # clip extreme
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        # avoid explosion: scale down if huge
        # (cheap, vectorized)
        scale = np.maximum(gn / 3.0, 1.0)
        Gt = Gt / scale
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def hard_refine(x_init, steps, lr0, tol=0.003):
    x = x_init.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.8
    for s in range(1, steps + 1):
        frac = s / steps
        lr = lr0 * (1.0 - 0.7 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(G.max())
        mask = G > (M - tol)
        # upper weight by slack to focus hardest
        # uniform weight is fine and fast
        W = mask.astype(float)
        np.fill_diagonal(W, 0.0)
        rs = W.sum()
        if rs == 0:
            break
        Grad = (W @ x) * (n / rs)
        # row-reweight to help overloaded points a bit more
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def best_holes(X_rem, rng, n_samp=6000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    return S, worst


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    FINAL_LP = 22.0
    rng = np.random.default_rng(0)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                cands.append((max_cos(a), a.copy()))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12

    it = 0
    # hopping phase
    while time.time() < T_END - FINAL_LP and it < 5000:
        it += 1
        cyc = it % 10
        x_try = x_best.copy()
        if cyc == 0:
            # cold random restart (new basin with holes/rattlers)
            x_try = rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            steps, T0, T1, lr, mu = 500, 0.03, 0.004, 0.006, 0.85
            x_try = soft_quench(x_try, steps, T0, T1, lr, mu)
            x_try = hard_refine(x_try, 250, 0.004, tol=0.004)
        elif cyc == 1:
            # heavy kick to worst points
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            rowmax = G0.max(axis=1)
            order = np.argsort(-rowmax)
            k = 4
            idx = order[:k]
            x_try[idx] += 0.35 * rng.standard_normal((k, d))
            x_try += 0.015 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 450, 0.022, 0.003, 0.006, 0.85)
            x_try = hard_refine(x_try, 200, 0.0035, tol=0.003)
        elif cyc == 2:
            # cluster shake: random center + nearest
            ci = int(rng.integers(n))
            dots = x_best @ x_best[ci]
            dots[ci] = -2.0
            nbr = np.argsort(-dots)[:6]
            cl = np.append(nbr, ci)
            x_try[cl] += 0.28 * rng.standard_normal((len(cl), d))
            x_try += 0.010 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 450, 0.020, 0.003, 0.006, 0.85)
            x_try = hard_refine(x_try, 200, 0.0035, tol=0.003)
        elif cyc == 3 or cyc == 4:
            # greedy hole re-insertion of worst 2-5
            G0 = x_best @ x_best.T
            np.fill_diagonal(G0, -2.0)
            mval = float(G0.max())
            cnt = (G0 > mval - 0.004).sum(axis=1)
            order = np.argsort(-cnt - 0.5 * (G0.max(axis=1) - mval) * 1000)
            nk = int(rng.integers(2, 6))
            worst = order[:nk]
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = x_best[mask]
            S, sc = best_holes(X_rem, rng, n_samp=6000)
            # pick nk distinct good holes greedily
            X_cur = X_rem.copy()
            new_pts = []
            used = np.zeros(len(S), dtype=bool)
            for _ in range(nk):
                # recompute scores against current set (cheap update)
                dots = S @ X_cur.T
                w = dots.max(axis=1)
                w[used] = 1e9
                bi = int(np.argmin(w))
                used[bi] = True
                q = S[bi].copy()
                # small local tune of hole: few gradient steps away from X_cur
                for _ in range(60):
                    dd = X_cur @ q
                    mm = float(dd.max())
                    ww = np.exp((dd - mm) / 0.004)
                    gw = (ww[:, None] * X_cur).sum(axis=0)
                    gw = gw - float(gw @ q) * q
                    q = q - 0.02 * gw
                    q /= np.linalg.norm(q)
                new_pts.append(q)
                X_cur = np.vstack([X_cur, q[None, :]])
            x_try[mask] = X_rem
            x_try[worst] = np.array(new_pts)
            x_try += 0.006 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 400, 0.016, 0.0028, 0.005, 0.85)
            x_try = hard_refine(x_try, 220, 0.0035, tol=0.003)
        elif cyc == 5:
            # partial randomization: keep 90, randomize 12
            idx = rng.choice(n, size=12, replace=False)
            x_try[idx] = rng.standard_normal((12, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 500, 0.028, 0.0035, 0.007, 0.85)
            x_try = hard_refine(x_try, 250, 0.004, tol=0.004)
        elif cyc == 6:
            # global medium shake
            x_try = x_best + 0.10 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 400, 0.018, 0.003, 0.006, 0.85)
            x_try = hard_refine(x_try, 200, 0.0035, tol=0.003)
        elif cyc == 7:
            # antipodal-mix: flip half with noise (different topology)
            flip = rng.choice(n, size=n // 2, replace=False)
            x_try[flip] = -x_try[flip] + 0.05 * rng.standard_normal((len(flip), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 450, 0.025, 0.003, 0.006, 0.85)
            x_try = hard_refine(x_try, 200, 0.0035, tol=0.003)
        else:
            # light jitter + hot quench
            x_try = x_best + 0.04 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = soft_quench(x_try, 350, 0.015, 0.003, 0.005, 0.85)
            x_try = hard_refine(x_try, 180, 0.003, tol=0.003)
        try:
            v = max_cos(x_try)
        except Exception:
            continue
        if v < best and np.all(np.isfinite(x_try)):
            best = v
            x_best = x_try.copy()
            # early exit if clearly record-breaking by margin
            if best < 0.2665:
                break
        if time.time() >= T_END - FINAL_LP:
            break

    # final snap with trust-region LP
    remain = T_END - time.time()
    if remain > 3.0:
        lp_time = min(FINAL_LP, remain - 0.5)
        try:
            x_best = lp_polish(x_best, lp_time, t_end=T_END - 0.3)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
