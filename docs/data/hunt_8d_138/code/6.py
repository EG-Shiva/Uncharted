import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def anneal(x, steps, beta0, beta1, lr, rng, noise_amp=0.0, mu=0.8):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        beta = beta0 + (beta1 - beta0) * frac
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr_k * vel
        if noise_amp > 0:
            nz = rng.standard_normal(x.shape)
            nz -= np.sum(nz * x, axis=1, keepdims=True) * x
            x = x + noise_amp * (1.0 - frac) * nz
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(15.0 * radius, 5e-10)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            order = np.argsort(c[sel])[-2500:]
            I, J = I[order], J[order]
            cc = c[sel][order]
            p = 2500
        else:
            cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def contact_counts(x, bv, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > bv - tol, axis=1)


def splice(a, b, rng, noise=0.02):
    n, d = a.shape
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u) + 1e-12
    pa = a @ u
    pb = b @ u
    # take complementary caps; randomly flip side
    if rng.random() < 0.5:
        Aset = a[pa > 0]
        Bset = b[pb <= 0]
    else:
        Aset = a[pa <= 0]
        Bset = b[pb > 0]
    # random rotation of one cap to break symmetry / align differently
    if rng.random() < 0.5 and len(Bset) > 0:
        # random orthogonal-ish jitter via QR
        M = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(M)
        if rng.random() < 0.5:
            Bset = Bset @ Q.T
        else:
            Aset = Aset @ Q.T
    child_list = []
    if len(Aset) > 0:
        child_list.append(Aset)
    if len(Bset) > 0:
        child_list.append(Bset)
    if child_list:
        child = np.concatenate(child_list, axis=0)
    else:
        child = a.copy()
    # fix size to n
    if len(child) > n:
        idx = rng.choice(len(child), size=n, replace=False)
        child = child[idx]
    elif len(child) < n:
        need = n - len(child)
        extra = rng.standard_normal((need, d))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        child = np.concatenate([child, extra], axis=0)
    child = child + noise * rng.standard_normal(child.shape)
    child /= np.linalg.norm(child, axis=1, keepdims=True)
    return child


def greedy_single_repair(x, rng, ntry=2500):
    # reposition the single worst point by sampling
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    w = int(np.argmax(rowmax))
    cand = rng.standard_normal((ntry, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T  # (ntry, n)
    dots[:, w] = -2.0
    score = dots.max(axis=1)
    b = int(np.argmin(score))
    if score[b] < rowmax[w]:
        x = x.copy()
        x[w] = cand[b]
    return x


def solve():
    t0 = time.time()
    T_LIMIT = 170.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (138, 8):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((138, 8))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best = None
    best_val = np.inf
    for c in cands:
        v = max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()

    pool = [best.copy()]
    pool_val = [best_val]

    def add_to_pool(x, v, maxsize=6):
        pool.append(x.copy())
        pool_val.append(v)
        # keep best maxsize distinct-ish (by value gap)
        order = np.argsort(np.array(pool_val))
        kept_x, kept_v = [], []
        for o in order:
            vv = pool_val[o]
            ok = True
            for kv in kept_v:
                if abs(kv - vv) < 1e-12 and len(kept_v) >= 2:
                    pass
            kept_x.append(pool[o])
            kept_v.append(vv)
            if len(kept_x) >= maxsize:
                break
        # actually just keep top maxsize by value
        pool.clear()
        pool_val.clear()
        for xx, vv in zip(kept_x[:maxsize], kept_v[:maxsize]):
            pool.append(xx)
            pool_val.append(vv)

    # Phase 1: build diverse pool with big kicks (fast, short snaps)
    for rep in range(5):
        if time.time() - t0 > 40:
            break
        sigma = 0.45 + 0.15 * rng.random()
        x = best + sigma * rng.standard_normal(best.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        try:
            x = anneal(x, 900, 15.0, 120.0, 0.03 + 0.02 * rng.random(), rng,
                       noise_amp=0.01, mu=0.8)
            v = max_cos(x)
            if v < best_val + 0.03:
                tl = T_LIMIT - (time.time() - t0)
                if tl < 20:
                    break
                x = lp_polish(x, 0.7)
                v = max_cos(x)
            add_to_pool(x, v)
            if v < best_val - 1e-12:
                best, best_val = x.copy(), v
        except Exception:
            continue

    trial = 0
    while time.time() - t0 < 122.0:
        trial += 1
        tl = T_LIMIT - (time.time() - t0)
        if tl < 14:
            break
        mode = trial % 4
        try:
            if mode in (0, 1) and len(pool) >= 2:
                # crossover between two pool members
                order = np.argsort(np.array(pool_val))
                # bias pick: best + random
                if rng.random() < 0.7:
                    i1 = int(order[0])
                else:
                    i1 = int(rng.integers(0, len(pool)))
                i2 = int(rng.integers(0, len(pool)))
                if i2 == i1:
                    i2 = (i2 + 1) % len(pool)
                A, B = pool[i1], pool[i2]
                noise = 0.015 + 0.03 * rng.random()
                x_try = splice(A, B, rng, noise=noise)
                # occasional cluster dispersal: wipe k nearest of random center
                if rng.random() < 0.4:
                    c0 = int(rng.integers(0, 138))
                    d2 = np.sum((x_try - x_try[c0]) ** 2 if False else None
                    G = x_try @ x_try[c0]
                    k = int(12 + rng.integers(0, 20))
                    idx = np.argsort(G)[-k:]
                    x_try[idx] = rng.standard_normal((k, 8))
                    x_try[idx] /= np.linalg.norm(x_try[idx], axis=1, keepdims=True)
                # greedy repair of worst 1-2 points
                if rng.random() < 0.5:
                    x_try = greedy_single_repair(x_try, rng, ntry=3000)
            elif mode == 2:
                # subset kick biased to jammed points
                try:
                    cc = contact_counts(best, best_val)
                except Exception:
                    cc = np.ones(138)
                sigma = 0.15 + 0.35 * rng.random()
                k = int(8 + rng.integers(0, 22))
                w = (cc.astype(float) + 1.0)
                w /= w.sum()
                idx = rng.choice(138, size=k, replace=False, p=w)
                x_try = best + 0.012 * rng.standard_normal(best.shape)
                x_try[idx] += (sigma * 2.0) * rng.standard_normal((k, 8))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            else:
                # full large kick for fresh basin
                sigma = 0.25 + 0.35 * rng.random()
                x_try = best + sigma * rng.standard_normal(best.shape)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

            beta0 = 12.0 + rng.random() * 25.0
            beta1 = 90.0 + rng.random() * 140.0
            lr = 0.022 + rng.random() * 0.025
            steps = 1600
            if time.time() - t0 > 95:
                steps = 1100
            noise_amp = 0.004 * rng.random()
            x_try = anneal(x_try, steps, beta0, beta1, lr, rng,
                           noise_amp=noise_amp, mu=0.8)
            v_try = max_cos(x_try)
            # extra single-point repair before snap sometimes helps rattlers
            if v_try < best_val + 0.015 and rng.random() < 0.3:
                x_try = greedy_single_repair(x_try, rng, ntry=2000)
                x_try = anneal(x_try, 250, 60.0, 150.0, 0.015, rng, mu=0.85)
                v_try = max_cos(x_try)
            if v_try < best_val + 0.012:
                ps = 1.4
                if v_try < best_val:
                    ps = 2.2
                tl2 = T_LIMIT - (time.time() - t0)
                ps = min(ps, max(0.3, tl2 - 10.0))
                if ps < 0.25:
                    break
                x_pol = lp_polish(x_try, ps)
                v_pol = max_cos(x_pol)
                # keep pool diverse: add if decent
                if v_pol < (max(pool_val) if pool_val else best_val) or len(pool) < 6:
                    add_to_pool(x_pol, v_pol)
                if v_pol < best_val - 1e-12:
                    best, best_val = x_pol.copy(), v_pol
            else:
                # add mediocre but diverse to pool rarely to keep gene pool fresh
                if rng.random() < 0.05:
                    add_to_pool(x_try, v_try)
        except Exception:
            continue

    tl = T_LIMIT - (time.time() - t0)
    if tl > 2.5:
        try:
            best = lp_polish(best, max(1.0, min(30.0, tl - 1.5)))
            best_val = max_cos(best)
        except Exception:
            pass
    while time.time() - t0 < T_LIMIT - 1.0:
        try:
            x_try = best + 0.004 * rng.standard_normal(best.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_pol = lp_polish(x_try, 0.7)
            v_pol = max_cos(x_pol)
            if v_pol < best_val:
                best, best_val = x_pol, v_pol
            else:
                break
        except Exception:
            break
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
