import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def anneal_lse(x, steps, beta0, beta1, lr0, lr1, rng=None):
    n, d = x.shape
    x = x.copy()
    for s in range(steps):
        t = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** t
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = float(np.sqrt(np.mean(Tg * Tg))) + 1e-12
        x = x - (lr / rms) * Tg
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def fingerprint(x):
    G = x @ x.T
    ev = np.linalg.eigvalsh(G)
    ev = np.sort(ev)[::-1]
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    top = np.sort(c)[::-1][:200]
    return ev, top

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': maxiter * 3})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def lp_polish(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.4:
            break
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                yy = x + res.x[:-1].reshape(n, d) * shrink
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv2 = max_cos(yy)
                if vv2 < best:
                    x, best = yy, vv2
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return x

def gaussian_start(rng, n=68, d=12):
    x = rng.standard_normal((n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def sobol_start(seed, n=68, d=12):
    try:
        from scipy.stats import qmc, norm
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        # power of two for balance
        m = 128
        u = sampler.random_base2(m=int(np.log2(m)))
        u = np.clip(u, 1e-6, 1 - 1e-6)[:n]
        z = norm.ppf(u)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    except Exception:
        rng = np.random.default_rng(seed)
        return gaussian_start(rng, n, d)

def d12_pool(rng, poolsize, d=12):
    pool = []
    tries = 0
    while len(pool) < poolsize and tries < poolsize * 50:
        tries += 1
        v = rng.integers(-2, 3, size=d)
        if np.all(v == 0):
            continue
        if int(np.sum(v)) % 2 != 0:
            continue
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            continue
        pool.append(v / nrm)
    if len(pool) == 0:
        z = rng.standard_normal((poolsize, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    return np.array(pool)

def d12_start(rng, n=68, d=12):
    pool = d12_pool(rng, 4000, d)
    idx = rng.choice(len(pool), size=n, replace=False)
    x = pool[idx].astype(float).copy()
    # break exact symmetries with tiny jitter
    x += 1e-3 * rng.standard_normal(x.shape)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def greedy_maximin(pool, rng, n=68):
    P = len(pool)
    sel = []
    first = int(rng.integers(0, P))
    sel.append(first)
    best = np.full(P, 1e9)
    # max dot to selected set (smaller is better hole)
    v0 = pool[first]
    dots = pool @ v0
    best = np.minimum(best, dots)
    best[first] = 1e9
    for _ in range(1, n):
        # pick pool point with minimal best (most isolated), excluding selected
        nxt = int(np.argmin(best))
        sel.append(nxt)
        best[nxt] = 1e9
        dots = pool @ pool[nxt]
        best = np.minimum(best, dots)
        # keep selected excluded
        for s in sel:
            best[s] = 1e9
    x = pool[np.array(sel)].copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    DEADLINE = t0 + 165.0
    n, d = 68, 12
    master = np.random.default_rng(20261031)
    seeds = [int(master.integers(0, 10**9)) for _ in range(100)]

    starts = []
    # 10 Gaussian
    for i in range(10):
        rng = np.random.default_rng(seeds[i])
        starts.append(('gauss%d' % i, gaussian_start(rng, n, d)))
    # 6 Sobol
    for i in range(6):
        starts.append(('sobol%d' % i, sobol_start(seeds[10 + i], n, d)))
    # 6 D12
    for i in range(6):
        rng = np.random.default_rng(seeds[20 + i])
        starts.append(('d12_%d' % i, d12_start(rng, n, d)))
    # 6 greedy maximin: alternate random pool / D12 pool
    for i in range(6):
        rng = np.random.default_rng(seeds[30 + i])
        if i % 2 == 0:
            pool = rng.standard_normal((4000, d))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        else:
            pool = d12_pool(rng, 4000, d)
            # pad with random to reach 4000 if short
            if len(pool) < 4000:
                extra = rng.standard_normal((4000 - len(pool), d))
                extra /= np.linalg.norm(extra, axis=1, keepdims=True)
                pool = np.vstack([pool, extra])
        starts.append(('greedy%d' % i, greedy_maximin(pool, rng, n)))

    reps = []  # list of dicts: x, val, ev, top
    EIG_TOL = 0.15
    TOP_TOL = 0.006
    for si, (name, x0) in enumerate(starts):
        if time.time() > t0 + 45.0:
            break
        steps = 200 + (si * 53 + 17) % 401  # 200..600 deterministic spread
        beta0 = 30.0
        beta1 = 300.0
        # slight diversity in lr
        lr0 = 0.04 + 0.02 * ((si * 7) % 5) / 4.0
        lr1 = 0.004 + 0.003 * ((si * 11) % 4) / 3.0
        b0 = beta0 * (1.0 + 0.2 * ((si % 3) - 1))
        b1 = beta1 * (1.0 + 0.2 * (((si + 1) % 3) - 1))
        x = anneal_lse(x0, steps, b0, b1, lr0, lr1)
        v = max_cos(x)
        ev, top = fingerprint(x)
        # cluster check: keep at most one per cluster (best wins)
        found = -1
        for ri, r in enumerate(reps):
            de = float(np.linalg.norm(ev - r['ev']))
            dt = float(np.linalg.norm(top - r['top']) / np.sqrt(200.0))
            if de < EIG_TOL and dt < TOP_TOL:
                found = ri
                break
        if found >= 0:
            if v < reps[found]['val']:
                reps[found] = {'x': x.copy(), 'val': v, 'ev': ev, 'top': top, 'name': name}
        else:
            reps.append({'x': x.copy(), 'val': v, 'ev': ev, 'top': top, 'name': name})

    if len(reps) == 0:
        rng = np.random.default_rng(0)
        x = gaussian_start(rng, n, d)
        reps.append({'x': x, 'val': max_cos(x), 'ev': fingerprint(x)[0], 'top': fingerprint(x)[1], 'name': 'fallback'})

    # sort by cheap score, greedily pick dissimilar finalists
    reps_sorted = sorted(reps, key=lambda r: r['val'])
    finalists = []
    for r in reps_sorted:
        ok = True
        for f in finalists:
            de = float(np.linalg.norm(r['ev'] - f['ev']))
            dt = float(np.linalg.norm(r['top'] - f['top']) / np.sqrt(200.0))
            if de < EIG_TOL and dt < TOP_TOL:
                ok = False
                break
        if ok:
            finalists.append(r)
        if len(finalists) >= 5:
            break
    # if too few due to clustering, just take top-5 by score
    if len(finalists) < min(5, len(reps_sorted)):
        for r in reps_sorted:
            if all(r['name'] != f['name'] for f in finalists):
                finalists.append(r)
            if len(finalists) >= min(5, len(reps_sorted)):
                break

    # Polish dissimilar finalists with escalating L-BFGS + LP, sharing time
    polished = []
    for fi, f in enumerate(finalists):
        if time.time() > DEADLINE - 8.0:
            # no time: keep unpolished
            polished.append({'x': f['x'], 'val': f['val']})
            continue
        x = f['x'].copy()
        v = f['val']
        # sharp L-BFGS rungs (cheap local convergence)
        for beta, it in [(600.0, 60), (2500.0, 60)]:
            if time.time() > DEADLINE - 8.0:
                break
            y = run_lbfgs(x, beta, it)
            vy = max_cos(y)
            if vy < v:
                x, v = y, vy
        # LP budget: share remaining evenly among remaining finalists
        remain = DEADLINE - time.time() - 4.0
        nrem = len(finalists) - fi
        budget = max(1.0, remain / max(nrem, 1) * 0.85)
        # first finalist gets slightly less? keep equal for diversity
        budget = min(budget, 25.0)
        if remain > 2.0:
            x2 = lp_polish(x, budget, r0=8e-4)
            v2 = max_cos(x2)
            if v2 < v:
                x, v = x2, v2
            else:
                # try one sharp L-BFGS after LP
                if time.time() < DEADLINE - 6.0:
                    y = run_lbfgs(x, 6000.0, 40)
                    vy = max_cos(y)
                    if vy < v:
                        x, v = y, vy
        polished.append({'x': x, 'val': v})

    # overall winner among dissimilar polished
    polished_sorted = sorted(polished, key=lambda r: r['val'])
    x_best = polished_sorted[0]['x']
    best_val = polished_sorted[0]['val']
    # extra squeeze on winner if time remains
    if time.time() < DEADLINE - 6.0:
        extra = DEADLINE - time.time() - 3.0
        if extra > 2.0:
            x2 = lp_polish(x_best, min(extra, 18.0), r0=3e-4)
            v2 = max_cos(x2)
            if v2 < best_val:
                x_best, best_val = x2, v2

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    # FINAL comparison: now (and only now) load best.npy / record.npy
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        vb = max_cos(xb)
        if vb < best_val:
            x_best, best_val = xb, vb
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        vr = max_cos(xr)
        if vr < best_val:
            x_best, best_val = xr, vr
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
