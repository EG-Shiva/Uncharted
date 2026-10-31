import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
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
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def micro_hill(x, seconds, rng):
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).copy()
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best = float(G.max())
    t_end = time.time() + seconds
    it = 0
    sigmas = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    si = 0
    sigma = sigmas[0]
    cnt_rej = 0
    while time.time() < t_end:
        it += 1
        if it % 800 == 0:
            si = (si + 1) % len(sigmas)
            sigma = sigmas[si]
        i = int(rng.integers(0, n))
        r = rng.normal(size=d)
        dot = float(r @ x[i])
        r -= dot * x[i]
        rn = float(np.linalg.norm(r))
        if rn < 1e-15:
            continue
        r /= rn
        y = x[i] + sigma * r
        y /= np.linalg.norm(y)
        # max with others
        dvec = y @ x.T
        dvec[i] = -2.0
        dmax = float(dvec.max())
        # max among pairs not involving i
        rowmax = float(G[i].max())
        if rowmax < best - 1e-12:
            m_other = best
        else:
            # recompute excluding i
            # fast: mask
            tmp_max = -2.0
            # use numpy max on masked copy (n=116 cheap)
            # avoid full copy: compute max of G with row/col i ignored
            # G is symmetric, do:
            m_other = float(np.max(np.delete(np.delete(G, i, axis=0), i, axis=1)))
        new_best = dmax if dmax > m_other else m_other
        if new_best < best - 1e-15:
            x[i] = y
            nr = x[i] @ x.T
            nr[i] = -2.0
            G[i, :] = nr
            G[:, i] = nr
            # G[i,i] already -2
            best = new_best
            cnt_rej = 0
        else:
            cnt_rej += 1
    return x

def perturb_to_new_basin(x, rng, trial):
    y = x.copy()
    n, d = y.shape
    G = y @ y.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    contacts = (G > m - 1e-7).sum(axis=1)
    s = trial % 5
    if s == 0:
        y = y + rng.normal(0, 0.03, size=y.shape)
    elif s == 1:
        y = y + rng.normal(0, 0.015, size=y.shape)
        ratt = np.where(contacts == 0)[0]
        if len(ratt) > 0:
            y[ratt] = y[ratt] + rng.normal(0, 0.45, size=(len(ratt), d))
        else:
            # kick lightest 10%
            light = np.argsort(contacts)[:12]
            y[light] = y[light] + rng.normal(0, 0.3, size=(len(light), d))
    elif s == 2:
        k = 3 if (trial // 5) % 2 == 0 else 5
        worst = np.argsort(contacts)[-k:]
        worst_set = set(worst.tolist())
        for idx in worst:
            cand = rng.normal(size=(1500, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            Gm = cand @ y.T
            # ignore columns of other worst (being moved) to spread
            for w in worst_set:
                Gm[:, w] = -2.0
            # restore column idx? keep -2, we want far from kept points only
            # actually we already set all worst to -2, good
            bi = int(np.argmin(Gm.max(axis=1)))
            y[idx] = cand[bi]
        y = y + rng.normal(0, 0.008, size=y.shape)
    elif s == 3:
        k = int(rng.integers(5, 16))
        idx = rng.choice(n, size=k, replace=False)
        y[idx] = y[idx] + rng.normal(0, 0.22, size=(k, d))
        y = y + rng.normal(0, 0.005, size=y.shape)
    else:
        idx = int(rng.integers(0, n))
        cand = rng.normal(size=(2000, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        Gc = cand @ y.T
        Gc[:, idx] = -2.0
        bi = int(np.argmin(Gc.max(axis=1)))
        y[idx] = cand[bi]
        y = y + rng.normal(0, 0.01, size=y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def solve():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    BUDGET = 172.0
    # load starts
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape == (116, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.normal(size=(116, 6))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_x = min(cands, key=max_cos)
    best_v = max_cos(best_x)
    # baseline tighten
    if time.time() - t0 < BUDGET - 5:
        best_x = lp_polish(best_x, min(4.0, BUDGET - (time.time() - t0) - 2))
        best_v = max_cos(best_x)
    if time.time() - t0 < BUDGET - 5:
        best_x = micro_hill(best_x, 3.0, rng)
        best_v = max_cos(best_x)
    trial = 0
    # basin hopping phase until ~120s elapsed
    while time.time() - t0 < 120.0 and time.time() - t0 < BUDGET - 30:
        trial += 1
        y0 = perturb_to_new_basin(best_x, rng, trial)
        # short LP reconverge
        remain = BUDGET - (time.time() - t0) - 25
        if remain < 0.5:
            break
        secs = min(1.4, remain)
        y1 = lp_polish(y0, secs)
        v1 = max_cos(y1)
        # quick micro squeeze if promising (within 0.002 of best)
        if v1 < best_v + 0.002 and time.time() - t0 < BUDGET - 25:
            y1 = micro_hill(y1, 0.35, rng)
            v1 = max_cos(y1)
        if v1 < best_v - 5e-13:
            best_x, best_v = y1, v1
    # final intensive polish on best
    while time.time() - t0 < BUDGET - 2:
        remain = BUDGET - (time.time() - t0) - 1.0
        if remain < 0.5:
            break
        s1 = min(6.0, remain * 0.55)
        best_x = lp_polish(best_x, s1)
        best_v = max_cos(best_x)
        remain2 = BUDGET - (time.time() - t0) - 0.5
        if remain2 < 0.5:
            break
        s2 = min(6.0, remain2)
        best_x = micro_hill(best_x, s2, rng)
        best_v = max_cos(best_x)
        # if very tight on time, break
        if BUDGET - (time.time() - t0) < 2.0:
            break
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
