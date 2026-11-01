import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

_T0 = None
_IU = None

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _make_fg(n, d, IU, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    fg = _make_fg(n, d, IU, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-6, 'maxfun': maxiter * 3})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0 / np.linalg.norm(x0, axis=1, keepdims=True)

def _anneal_short(x0, schedule):
    x = x0.copy()
    for beta, it in schedule:
        x = _run_lbfgs(x, beta, it)
    return x

def _lp_polish(x, deadline, r0=8e-4, cap_p=1600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.25 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        win = max(20.0 * radius, 2e-7)
        sel = c > m - win
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        I = iu[0][sel]; J = iu[1][sel]
        if len(I) > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-12:
                break
    return x

def _sobol_sphere_block(n, d, seed):
    try:
        from scipy.stats import qmc, norm
        samp = None
        # power of 2 for Sobol balance
        m = 1
        while m < n:
            m *= 2
        gen = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = gen.random_base2(m=int(np.log2(m)))
        u = np.clip(u[:n], 1e-6, 1 - 1e-6)
        z = norm.ppf(u)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    except Exception:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

def _d12_pool(rng, m_target=4000):
    d = 12
    pool = []
    seen = set()
    tries = 0
    while len(pool) < m_target and tries < m_target * 30:
        tries += 1
        v = rng.integers(-2, 3, size=d)
        if np.all(v == 0):
            continue
        if int(v.sum()) % 2 != 0:
            continue
        key = tuple(v.tolist())
        # identify antipode as same direction family but keep one
        key2 = tuple((-v).tolist())
        if key in seen or key2 in seen:
            continue
        seen.add(key)
        f = v.astype(float)
        f /= np.linalg.norm(f)
        pool.append(f)
    if len(pool) == 0:
        z = rng.standard_normal((m_target, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    return np.array(pool)

def _greedy_maximin(pool, n_pick, rng):
    m = pool.shape[0]
    if m <= n_pick:
        # pad with random
        extra = rng.standard_normal((n_pick - m, pool.shape[1]))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        return np.vstack([pool, extra])
    start = int(rng.integers(0, m))
    sel = [start]
    selmask = np.zeros(m, dtype=bool)
    selmask[start] = True
    # current max dot with selected set
    curmax = pool @ pool[start]
    curmax[start] = np.inf
    for _ in range(1, n_pick):
        # pick minimizer of curmax among unselected
        idx = int(np.argmin(np.where(selmask, np.inf, curmax)))
        sel.append(idx)
        selmask[idx] = True
        d = pool @ pool[idx]
        curmax = np.minimum(curmax, d)
        curmax[selmask] = np.inf
    return pool[np.array(sel)]

def _fingerprint(x):
    G = x @ x.T
    np.fill_diagonal(G, 1.0)
    ev = np.linalg.eigvalsh(G)
    ev = np.sort(ev)  # ascending
    iu = np.triu_indices(x.shape[0], 1)
    c = (x @ x.T)[iu]
    top = np.sort(c)[-200:][::-1]  # descending, top200
    return ev, top

def _fp_dist(ev1, top1, ev2, top2):
    de = float(np.linalg.norm(ev1 - ev2) / np.sqrt(len(ev1)))
    dt = float(np.linalg.norm(top1 - top2) / np.sqrt(len(top1)))
    return de + 5.0 * dt

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline_final = t0 + TLIM
    n, d = 68, 12
    rng = np.random.default_rng(20261031)

    census_end = t0 + 95.0

    # ---- build pools for greedy selections (fixed, deterministic) ----
    try:
        rng_pool = np.random.default_rng(999)
        gauss_pool = rng_pool.standard_normal((4000, d))
        gauss_pool /= np.linalg.norm(gauss_pool, axis=1, keepdims=True)
        d12pool = _d12_pool(np.random.default_rng(777), m_target=3000)
        mix_pool = np.vstack([gauss_pool, d12pool])
    except Exception:
        mix_pool = None
        d12pool = None

    starts = []
    # type counts: aim ~44 starts
    N_A = 12  # gaussian
    N_B = 10  # sobol
    N_C = 10  # D12 greedy / raw
    N_D = 12  # greedy maximin from mix
    for i in range(N_A):
        r = np.random.default_rng(1000 + i)
        z = r.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        starts.append(('g%d' % i, z))
    for i in range(N_B):
        z = _sobol_sphere_block(n, d, seed=2000 + i)
        # random orthogonal scramble to diversify blocks
        r = np.random.default_rng(2000 + i)
        Q, _ = np.linalg.qr(r.standard_normal((d, d)))
        z = z @ Q.T
        # random sign flips
        s = r.choice([-1.0, 1.0], size=d)
        z = z * s
        starts.append(('s%d' % i, z))
    for i in range(N_C):
        r = np.random.default_rng(3000 + i)
        # sample 68 distinct D12 directions; if not enough, pad gaussian
        if d12pool is not None and len(d12pool) >= 68:
            idx = r.choice(len(d12pool), size=68, replace=False)
            z = d12pool[idx].copy()
            z += 1e-3 * r.standard_normal(z.shape)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
        else:
            z = r.standard_normal((n, d))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
        starts.append(('d%d' % i, z))
    for i in range(N_D):
        r = np.random.default_rng(4000 + i)
        if mix_pool is not None:
            # subsample pool slice for diversity then greedy
            perm = r.permutation(len(mix_pool))[:2500]
            sub = mix_pool[perm]
            z = _greedy_maximin(sub, n, r)
        else:
            z = r.standard_normal((n, d))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
        # tiny jitter so LBFGS not stuck at lattice symmetry
        z += 2e-4 * r.standard_normal(z.shape)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        starts.append(('m%d' % i, z))

    # short anneal schedules varying per start (200-500 steps)
    schedules = [
        [(30.0, 70), (80.0, 70), (180.0, 70), (300.0, 70)],
        [(40.0, 60), (100.0, 60), (300.0, 90)],
        [(30.0, 50), (60.0, 50), (150.0, 60), (300.0, 60)],
        [(50.0, 80), (200.0, 80), (350.0, 60)],
    ]

    survivors = []
    for si, (tag, x0) in enumerate(starts):
        if time.time() > census_end:
            break
        sched = schedules[si % len(schedules)]
        try:
            x = _anneal_short(x0, sched)
        except Exception:
            continue
        v = _max_cos(x)
        try:
            ev, top = _fingerprint(x)
        except Exception:
            continue
        survivors.append((v, x, ev, top, tag))
        # also early stop if very good? continue census for diversity

    if len(survivors) == 0:
        rng0 = np.random.default_rng(0)
        x = rng0.standard_normal((n, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        x = _anneal_short(x, schedules[0])
        survivors.append((_max_cos(x), x, *_fingerprint(x)[:2], 'fallback'))

    # ---- fingerprint clustering: keep at most one per cluster ----
    survivors.sort(key=lambda t: t[0])
    kept = []
    THRESH = 0.004  # fingerprint distance threshold
    for v, x, ev, top, tag in survivors:
        dup = False
        for _, _, evk, topk, _ in kept:
            if _fp_dist(ev, top, evk, topk) < THRESH:
                dup = True
                break
        if not dup:
            kept.append((v, x, ev, top, tag))
    # if too few kept (all similar), relax: keep top spread anyway
    # choose finalists: best dissimilar, up to 5
    K = min(5, len(kept))
    finalists = kept[:K]
    # if kept < 5 but survivors has more distinct-ish, add next survivors far from finalists
    if len(finalists) < 5:
        for v, x, ev, top, tag in survivors:
            if len(finalists) >= 5:
                break
            skip = False
            for vv, xx, evk, topk, _ in finalists:
                if _fp_dist(ev, top, evk, topk) < 0.0015:
                    skip = True
                    break
            if not skip:
                # check not already in finalists by tag
                tags = [t for _, _, _, _, t in finalists]
                if tag not in tags:
                    finalists.append((v, x, ev, top, tag))

    # ---- spend remaining time LP-polishing best dissimilar finalists ----
    # time slice per finalist
    remain = deadline_final - time.time()
    # reserve 6s for final load+compare
    budget = max(remain - 6.0, 5.0)
    per = budget / max(len(finalists), 1)
    polished = []
    for fi, (v, x, ev, top, tag) in enumerate(finalists):
        fdead = time.time() + per
        if fi == len(finalists) - 1:
            fdead = deadline_final - 6.0
        try:
            # medium-high beta LBFGS before LP
            xw = _run_lbfgs(x, 500.0, 90)
            if time.time() > fdead - 1.0:
                polished.append((_max_cos(xw), xw))
                continue
            xw = _run_lbfgs(xw, 900.0, 70)
        except Exception:
            xw = x
        try:
            xw = _lp_polish(xw, fdead, r0=8e-4, cap_p=1600)
        except Exception:
            pass
        polished.append((_max_cos(xw), xw))
        if time.time() > deadline_final - 7.0:
            break

    # also include unpolished survivors best in case polish hurt
    for v, x, _, _, _ in kept[:3]:
        polished.append((v, x))

    polished.sort(key=lambda t: t[0])
    x_best = polished[0][1].copy()
    best_val = polished[0][0]

    # final comparison with best.npy (allowed only now)
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if xb.shape == (68, 12):
            vb = _max_cos(xb)
            if vb < best_val:
                x_best = xb.copy()
                best_val = vb
    except Exception:
        pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
