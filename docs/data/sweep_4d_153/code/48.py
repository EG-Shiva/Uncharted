import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth(X, steps, b0, b1, lr0, lr1, mu=0.9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    V = np.zeros_like(X)
    eye2 = 2.0 * np.eye(n)
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        beta = b0 + (b1 - b0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        m = float(np.max(G - eye2))
        A = beta * (G - m)
        np.clip(A, -12.0, 0.0, out=A)
        np.fill_diagonal(A, -30.0)
        np.exp(A, out=A)
        Grad = A @ X
        d = np.sum(Grad * X, axis=1, keepdims=True)
        T = Grad - d * X
        s = float(np.mean(np.sqrt(np.sum(T * T, axis=1)))) + 1e-12
        T /= s
        V *= mu
        V += (1.0 - mu) * T
        X -= lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def _align_mix(X, Y, rng, mode):
    n, d = X.shape
    try:
        from scipy.optimize import linear_sum_assignment
        G = X @ Y.T
        # maximize sum -> minimize -G
        ri, ci = linear_sum_assignment(-G)
        perm = np.empty(n, dtype=int)
        perm[ri] = ci
        Ym = Y[perm]
    except Exception:
        Ym = Y.copy()
    try:
        M = Ym.T @ X
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        Yr = Ym @ R
    except Exception:
        Yr = Ym
    child = np.empty_like(X)
    if mode == 0:
        mask = rng.random(n) < 0.5
        child[mask] = X[mask]
        child[~mask] = Yr[~mask]
    elif mode == 1:
        # cap mixing: take cap from X, rest from Yr
        c = rng.standard_normal(d); c /= np.linalg.norm(c)
        dx = X @ c
        dy = Yr @ c
        # choose threshold to keep roughly half from each, random quantile
        q = float(rng.uniform(-0.5, 0.5))
        mask = dx > q
        # balance count: if too skewed, randomize a bit
        child[mask] = X[mask]
        child[~mask] = Yr[~mask]
        # avoid empty side
        if np.sum(mask) == 0 or np.sum(mask) == n:
            m2 = rng.random(n) < 0.5
            child[m2] = X[m2]
            child[~m2] = Yr[~m2]
    else:
        # slerp midpoint + jitter (basin bridge)
        w = float(rng.uniform(0.3, 0.7))
        child = w * X + (1.0 - w) * Yr
        child += 0.005 * rng.standard_normal((n, d))
    # tiny jitter to break exact symmetry / degeneracy
    child += 0.002 * rng.standard_normal((n, d))
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child

def solve():
    t0 = time.time()
    HARD = 176.0
    n, d = 153, 4
    rng = np.random.default_rng(15304)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_val:
                        best_val = c
                        best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # population: (val, matrix)
    pop = [(best_val, best.copy())]
    # build a few diverse smoothed members quickly
    for s in range(4):
        if time.time() - t0 > 40.0:
            break
        try:
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            # varied repulsion schedule per seed
            X = _smooth(X, 900, 8.0 + 4.0 * s, 180.0 + 30.0 * s, 0.025, 0.002)
            v = _max_cos(X)
            pop.append((v, X.copy()))
            if v < best_val - 1e-12:
                best_val = v
                best = X.copy()
        except Exception:
            continue
    # sort pop by value
    try:
        pop = sorted(pop, key=lambda t: t[0])[:6]
    except Exception:
        pass
    trial = 0
    # main crossover + mutation loop
    while time.time() - t0 < 132.0 and trial < 80:
        trial += 1
        elapsed = time.time() - t0
        remain = HARD - elapsed - 18.0
        if remain < 2.0:
            break
        # pick parents
        try:
            if len(pop) >= 2 and trial % 3 != 0:
                # bias to best + random other
                a = pop[0][1]
                bi = int(rng.integers(1, len(pop))) if len(pop) > 1 else 0
                b = pop[bi][1]
            elif len(pop) >= 2:
                i1 = int(rng.integers(0, len(pop)))
                i2 = int(rng.integers(0, len(pop)))
                if i2 == i1:
                    i2 = (i2 + 1) % len(pop)
                a, b = pop[i1][1], pop[i2][1]
            else:
                a = best
                Xtmp = best + 0.05 * rng.standard_normal((n, d))
                Xtmp /= np.linalg.norm(Xtmp, axis=1, keepdims=True)
                b = Xtmp
        except Exception:
            a, b = best, best
        mode = trial % 3  # 0 uniform, 1 cap, 2 midpoint
        try:
            child = _align_mix(a, b, rng, mode)
        except Exception:
            continue
        # rattler / hub surgery: randomize 1-2 worst points occasionally
        try:
            if rng.random() < 0.5:
                Gc = child @ child.T
                np.fill_diagonal(Gc, -2.0)
                mx = Gc.max()
                cnt = np.sum(Gc > mx - 1e-7, axis=1)
                # local worst max per point
                per = Gc.max(axis=1)
                order = np.argsort(-per)
                k = 1 + (trial % 2)
                for idx in order[:k]:
                    if rng.random() < 0.7:
                        child[idx] = rng.standard_normal(d)
                # also fully randomize any exact rattler of best occasionally
                Gb = best @ best.T
                np.fill_diagonal(Gb, -2.0)
                mxb = Gb.max()
                cntb = np.sum(Gb > mxb - 1e-7, axis=1)
                ratt = np.where(cntb == 0)[0]
                for idx in ratt:
                    if rng.random() < 0.5:
                        # place far from existing: try few candidates pick min max-dot
                        cand = rng.standard_normal((8, d))
                        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                        dots = np.abs(cand @ child.T).max(axis=1)
                        child[idx] = cand[int(np.argmin(dots))]
                child /= np.linalg.norm(child, axis=1, keepdims=True)
        except Exception:
            pass
        # smooth child (short, to enter new basin)
        try:
            if mode == 1:
                steps, b0, b1 = 1100, 25.0, 300.0
            elif mode == 2:
                steps, b0, b1 = 1300, 15.0, 280.0
            else:
                steps, b0, b1 = 1000, 20.0, 300.0
            child = _smooth(child, steps, b0, b1, 0.020, 0.002)
            v = _max_cos(child)
        except Exception:
            continue
        # maintain diversity archive
        try:
            # insert if better than worst and not duplicate
            vals = [p[0] for p in pop]
            worst = max(vals) if vals else 1e9
            if v < worst - 1e-12 and len(pop) < 6:
                pop.append((v, child.copy()))
                pop = sorted(pop, key=lambda t: t[0])[:6]
            elif v < worst - 1e-12:
                # replace worst
                wi = int(np.argmax(vals))
                pop[wi] = (v, child.copy())
                pop = sorted(pop, key=lambda t: t[0])[:6]
            elif v < worst + 0.005 and len(pop) < 6 and (rng.random() < 0.2):
                pop.append((v, child.copy()))
        except Exception:
            pass
        # LP snap promising children
        try:
            if v < best_val + 0.015 and remain > 5.0:
                if v < best_val + 0.001:
                    budget = min(6.0, remain - 2.0)
                else:
                    budget = min(2.5, remain - 2.0)
                if budget >= 1.0:
                    Xp = _lp_polish(child, budget)
                    vp = _max_cos(Xp)
                    if vp < v - 1e-13:
                        v = vp
                        child = Xp
                    # update archive with polished
                    try:
                        vals2 = [p[0] for p in pop]
                        worst2 = max(vals2) if vals2 else 1e9
                        if v < worst2 - 1e-12:
                            if len(pop) >= 6:
                                wi = int(np.argmax(vals2))
                                pop[wi] = (v, child.copy())
                            else:
                                pop.append((v, child.copy()))
                            pop = sorted(pop, key=lambda t: t[0])[:6]
                    except Exception:
                        pass
            if v < best_val - 1e-12:
                best_val = v
                best = child.copy()
        except Exception:
            if v < best_val - 1e-12:
                best_val = v
                try:
                    best = child.copy()
                except Exception:
                    pass
        # occasional pure-mutation fallback to keep exploring if stuck
        if trial % 10 == 9 and time.time() - t0 < 120.0:
            try:
                Xm = best + 0.03 * rng.standard_normal((n, d))
                Xm /= np.linalg.norm(Xm, axis=1, keepdims=True)
                Xm = _smooth(Xm, 700, 30.0, 320.0, 0.020, 0.002)
                vm = _max_cos(Xm)
                if vm < best_val - 1e-12:
                    best_val = vm
                    best = Xm.copy()
            except Exception:
                pass
    # final polish of best with remaining time
    try:
        remain = HARD - (time.time() - t0) - 1.0
        if remain > 2.0:
            best = _lp_polish(best, min(remain, 26.0))
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
