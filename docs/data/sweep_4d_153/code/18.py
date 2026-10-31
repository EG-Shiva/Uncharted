import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_descent(x, steps=550, T0=0.025, T1=0.003, s0=0.015, s1=0.0015):
    x = x.copy()
    n = x.shape[0]
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        s = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        norms = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        step = Gt / norms
        x = x - s * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            # keep tightest 2200
            idx = np.argpartition(-c[sel], 2200)[:2200]
            I, J = I[idx], J[idx]
            p = 2200
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[I, J], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def hole_for(others, rng, M=6000, refine=25, step=0.05):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    b = cand[bi].copy()
    bw = float(worst[bi])
    for _ in range(refine):
        prop = b + rng.normal(size=4) * step
        prop /= np.linalg.norm(prop)
        w = float((prop @ others.T).max())
        if w < bw:
            b, bw = prop, w
            step *= 1.05
        else:
            step *= 0.92
    return b

def worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 145.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None:
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append((max_cos(a), a))
    if len(cands) == 0:
        x0 = rng.normal(size=(153, 4))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda t: t[0])
        x0 = cands[0][1].copy()
    n = 153
    best = x0.copy()
    best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    Tb = 0.0018
    it = 0
    # precompute for speed
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 28:
        it += 1
        u = rng.random()
        y = None
        steps, T0, T1 = 500, 0.02, 0.0025
        if u < 0.22:
            sig = 0.006 + 0.009 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 450
            T0, T1 = 0.018, 0.0022
        elif u < 0.34:
            k = int(rng.integers(1, 6))
            idx = rng.choice(n, size=k, replace=False)
            y = cur.copy()
            y[idx] += rng.normal(size=(k, 4)) * 0.28
            y += rng.normal(size=y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 600
            T0, T1 = 0.025, 0.003
        elif u < 0.54:
            # overload-to-hole: pick bottleneck points
            y = cur.copy()
            worst, G = worst_per_point(y)
            order = np.argsort(-worst)
            nr = 1 if rng.random() < 0.65 else 2
            # pick among top 12 with randomness
            pool = order[:12]
            ridx = pool[rng.choice(len(pool), size=nr, replace=False)]
            # add small jitter to others to diversify
            y += rng.normal(size=y.shape) * 0.0015
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            for r in ridx:
                others = np.delete(y, r, axis=0)
                y[r] = hole_for(others, rng, M=5000, refine=20, step=0.06)
            y += rng.normal(size=y.shape) * 0.0015
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 650
            T0, T1 = 0.022, 0.0025
        elif u < 0.64:
            # active-edge break
            y = cur.copy()
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            iu0 = np.where(G > m - 2e-3)
            # keep i<j
            mask = iu0[0] < iu0[1]
            Ii, Ji = iu0[0][mask], iu0[1][mask]
            if len(Ii) > 0:
                e = int(rng.integers(len(Ii)))
                i, j = int(Ii[e]), int(Ji[e])
                # push i away from j tangentially, j away from i
                dij = y[j] - y[i] * float(y[i] @ y[j])
                dji = y[i] - y[j] * float(y[i] @ y[j])
                ni = np.linalg.norm(dij) + 1e-12
                nj = np.linalg.norm(dji) + 1e-12
                y[i] = y[i] - (dij / ni) * (0.05 + 0.05 * rng.random())
                y[j] = y[j] - (dji / nj) * (0.05 + 0.05 * rng.random())
                y += rng.normal(size=y.shape) * 0.002
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            else:
                y = y + rng.normal(size=y.shape) * 0.01
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 600
            T0, T1 = 0.022, 0.0025
        elif u < 0.79:
            # cluster shake around random center
            y = cur.copy()
            c = int(rng.integers(n))
            dots = y @ y[c]
            k = int(rng.integers(5, 18))
            idx = np.argsort(-dots)[:k]
            y[idx] += rng.normal(size=(k, 4)) * 0.12
            y += rng.normal(size=y.shape) * 0.0025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 600
            T0, T1 = 0.025, 0.003
        elif u < 0.92:
            sig = 0.035 + 0.03 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 550
            T0, T1 = 0.03, 0.004
        else:
            # random restart quench
            if rng.random() < 0.4 and time.time() < EXPLORE_END - 30:
                y = rng.normal(size=(153, 4))
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                steps = 700
                T0, T1 = 0.03, 0.003
            else:
                sig = 0.06
                y = cur + rng.normal(size=cur.shape) * sig
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                steps = 550
                T0, T1 = 0.03, 0.004
        s0 = 0.015 if rng.random() < 0.5 else 0.012
        ys = smooth_descent(y, steps=steps, T0=T0, T1=T1, s0=s0, s1=0.0012)
        v = max_cos(ys)
        accept = False
        if v < cur_val:
            accept = True
        else:
            if rng.random() < np.exp((cur_val - v) / Tb):
                accept = True
        if accept:
            cur, cur_val = ys, v
        # selective LP refinement
        if accept and v < best_val + 0.002 and time.time() < EXPLORE_END - 2:
            # only polish fairly often but with short budget
            # tighten condition as time goes
            do_polish = (v < best_val - 1e-12) or (v < best_val + 0.0008) or (rng.random() < 0.12)
            if do_polish:
                z = lp_polish(ys, budget=0.7)
                vz = max_cos(z)
                if vz < best_val - 1e-12:
                    best, best_val = z, vz
                    cur, cur_val = z.copy(), vz
                elif vz < cur_val:
                    cur, cur_val = z, vz
            else:
                if v < best_val - 1e-12:
                    best, best_val = ys, v
        else:
            if v < best_val - 1e-12:
                best, best_val = ys, v
                if accept:
                    cur, cur_val = ys.copy(), v
        if cur_val > best_val + 0.012:
            cur = best.copy() + rng.normal(size=best.shape) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = max_cos(cur)
            Tb = max(0.0010, Tb * 0.98)
        # slow anneal
        Tb = Tb * 0.9997 + 0.0009 * 0.0003
        if Tb < 0.0009:
            Tb = 0.0009
    # final intensive polish
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 4:
        # try a few perturbed variants of best
        tries = []
        tries.append(best.copy())
        # small jitters + quench
        for k in range(3):
            if DEADLINE - time.time() < 8:
                break
            sig = 0.003 + 0.002 * k
            yp = best + rng.normal(size=best.shape) * sig
            yp /= np.linalg.norm(yp, axis=1, keepdims=True)
            yq = smooth_descent(yp, steps=350, T0=0.012, T1=0.0018, s0=0.010, s1=0.001)
            tries.append(yq)
        # allocate LP budget
        per = min(8.0, (remaining - 1.0) / max(1, len(tries)))
        for t in tries:
            if time.time() > DEADLINE - 3:
                break
            z = lp_polish(t, budget=per)
            vz = max_cos(z)
            if vz < best_val:
                best, best_val = z, vz
        # final polish of best
        if time.time() < DEADLINE - 3:
            best = lp_polish(best, budget=min(10.0, DEADLINE - time.time() - 2.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
