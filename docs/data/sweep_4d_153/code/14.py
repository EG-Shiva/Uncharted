import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def smooth_descent(x, steps, T0, T1, s0, s1, rng, noise=0.0, mu=0.0):
    x = x.copy()
    n = x.shape[0]
    vel = np.zeros_like(x)
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
        direc = Gt / norms
        if mu > 0:
            vel = mu * vel + (1.0 - mu) * direc
            step = vel
        else:
            step = direc
        x = x - s * step
        if noise > 0:
            x = x + rng.normal(size=x.shape) * noise
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-30
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
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def hole_candidate(others, rng, M=4000):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)].copy()

def greedy_push(x, iters=3000, eps0=0.002):
    x = x.copy()
    best = max_cos(x)
    best_x = x.copy()
    n = x.shape[0]
    for it in range(iters):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G.max(axis=1)))
        # find partner
        row = G[k]
        j = int(np.argmax(row))
        eps = eps0 * (1.0 - it / iters) + 1e-5
        dik = float(x[k] @ x[j])
        # push apart symmetrically along tangent
        gk = x[j] - dik * x[k]
        gj = x[k] - dik * x[j]
        x[k] = x[k] - eps * gk
        x[j] = x[j] - eps * gj
        x[k] /= np.linalg.norm(x[k]) + 1e-30
        x[j] /= np.linalg.norm(x[j]) + 1e-30
        if (it & 63) == 0:
            v = max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
            elif v > best + 0.0005:
                x = best_x.copy()
    v = max_cos(best_x)
    return best_x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 143.0
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
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
                if np.all(np.isfinite(a)):
                    cands.append((max_cos(a), a.copy()))
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
    pool = [(best_val, best.copy())]
    Tb = 0.0022
    it = 0
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 28:
        it += 1
        u = rng.random()
        y = cur.copy()
        if u < 0.22:
            sig = 0.006 + 0.010 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, s0, s1, mu, nz = 420, 0.018, 0.0022, 0.014, 0.0012, 0.4, 0.0
        elif u < 0.40:
            # worst-point hole reinsertion (targeted)
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            worst = Gc.max(axis=1)
            k = 1 if rng.random() < 0.6 else 2
            idx = np.argsort(worst)[-k:]
            # sometimes random instead
            if rng.random() < 0.25:
                idx = rng.choice(n, size=k, replace=False)
            for r in idx:
                others = np.delete(y, r, axis=0)
                y[r] = hole_candidate(others, rng, M=5000)
            y += rng.normal(size=y.shape) * 0.0025
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, s0, s1, mu, nz = 650, 0.022, 0.0022, 0.015, 0.0011, 0.3, 0.0
        elif u < 0.60:
            # cluster shake around overloaded node
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            cm = Gc.max()
            thr = cm - 0.004
            deg = (Gc > thr).sum(axis=1)
            # weighted pick among high degree
            order = np.argsort(deg)
            seed = order[int(rng.integers(max(0, n - 40), n))]
            nbrs = np.where(Gc[seed] > thr)[0].tolist()
            nbrs.append(seed)
            # expand to second neighbors to reach size
            want = int(rng.integers(6, 16))
            if len(nbrs) < want:
                # add closest points geometrically
                d2 = Gc[seed].copy()
                extra = np.argsort(-d2)
                for e in extra:
                    if e not in nbrs:
                        nbrs.append(int(e))
                    if len(nbrs) >= want:
                        break
            nbrs = np.array(nbrs[:want], dtype=int)
            sigc = 0.08 + 0.08 * rng.random()
            y[nbrs] += rng.normal(size=(len(nbrs), 4)) * sigc
            # small global jitter
            y += rng.normal(size=y.shape) * 0.0015
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, s0, s1, mu, nz = 650, 0.025, 0.0028, 0.015, 0.0012, 0.2, 0.0
        elif u < 0.76:
            # cap shift: coherent move of one hemisphere/cap
            au = rng.normal(size=4); au /= np.linalg.norm(au)
            proj = cur @ au
            th = float(rng.uniform(-0.1, 0.6))
            mask = proj > th
            nm = int(mask.sum())
            if nm < 8 or nm > 130:
                # fallback global
                y = cur + rng.normal(size=cur.shape) * 0.03
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                steps, T0, T1, s0, s1, mu, nz = 550, 0.028, 0.0035, 0.015, 0.0012, 0.0, 0.0
            else:
                sh = rng.normal(size=4) * (0.06 + 0.08 * rng.random())
                # make roughly tangent to cap center
                y[mask] += sh
                y += rng.normal(size=y.shape) * 0.0015
                y /= np.linalg.norm(y, axis=1, keepdims=True)
                steps, T0, T1, s0, s1, mu, nz = 650, 0.024, 0.0025, 0.015, 0.0011, 0.3, 0.0
        else:
            sig = 0.030 + 0.035 * rng.random()
            y = cur + rng.normal(size=cur.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps, T0, T1, s0, s1, mu, nz = 550, 0.030, 0.0035, 0.016, 0.0012, 0.0, 0.0004
            # add slight Langevin noise during descent
            nz = 0.0004
        # randomize T slightly
        T0 *= 0.85 + 0.3 * rng.random()
        T1 *= 0.8 + 0.4 * rng.random()
        ys = smooth_descent(y, steps=steps, T0=T0, T1=T1, s0=s0, s1=s1, rng=rng, noise=nz, mu=mu)
        v = max_cos(ys)
        accept = False
        if v < cur_val:
            accept = True
        else:
            if rng.random() < np.exp((cur_val - v) / max(1e-6, Tb)):
                accept = True
        if accept:
            cur, cur_val = ys, v
            if len(pool) < 6:
                pool.append((v, ys.copy()))
            else:
                # replace worst in pool if better
                pool.sort(key=lambda t: t[0])
                if v < pool[-1][0]:
                    pool[-1] = (v, ys.copy())
        if v < best_val - 1e-12:
            best, best_val = ys.copy(), v
            cur, cur_val = ys.copy(), v
            pool.sort(key=lambda t: t[0])
        elif v < best_val + 0.002 and accept and time.time() < EXPLORE_END - 2:
            # quick LP refine promising basins
            if (it % 2) == 0 or v < best_val + 0.0005:
                z = lp_polish(ys, budget=0.9)
                vz = max_cos(z)
                if vz < best_val - 1e-12:
                    best, best_val = z, vz
                    cur, cur_val = z.copy(), vz
                elif vz < cur_val:
                    cur, cur_val = z, vz
        if cur_val > best_val + 0.012:
            cur = best.copy() + rng.normal(size=best.shape) * 0.008
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = max_cos(cur)
            Tb = max(0.0011, Tb * 0.985)
    # final selection polish
    pool.append((best_val, best.copy()))
    pool.sort(key=lambda t: t[0])
    # keep top 3 unique
    tops = []
    for vv, xx in pool:
        if len(tops) >= 3:
            break
        tops.append(xx.copy())
    remaining = DEADLINE - time.time() - 2.0
    if remaining > 4:
        per = min(7.0, remaining / max(1, len(tops) + 1))
        refined = []
        for xx in tops:
            if time.time() > DEADLINE - 9:
                break
            z = lp_polish(xx, budget=per)
            # extra fine gradient squeeze
            z2 = smooth_descent(z, steps=300, T0=0.003, T1=0.0006, s0=0.002, s1=0.0004, rng=rng, noise=0.0, mu=0.5)
            z3 = lp_polish(z2, budget=per * 0.6)
            refined.append((max_cos(z3), z3))
        # also greedy push on current best
        if time.time() < DEADLINE - 6 and refined:
            refined.sort(key=lambda t: t[0])
            gb = greedy_push(refined[0][1], iters=2000)
            refined.append((max_cos(gb), gb))
            # final LP after greedy
            if time.time() < DEADLINE - 5:
                zf = lp_polish(gb, budget=DEADLINE - time.time() - 3.0)
                refined.append((max_cos(zf), zf))
        if refined:
            refined.sort(key=lambda t: t[0])
            if refined[0][0] < best_val:
                best, best_val = refined[0][1], refined[0][0]
        else:
            best = lp_polish(best, budget=max(1.0, remaining - 1.0))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
