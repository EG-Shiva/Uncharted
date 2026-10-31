import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = (g >= m - tol).sum(axis=1)
    pw = g.max(axis=1)
    return m, pw, cnt, g

def _adam_moving(x, mov, steps, tau0, tau1, lr0, lr1):
    mov = np.asarray(mov, dtype=int)
    k = len(mov)
    if k == 0:
        return x
    m = np.zeros((k, 4))
    v = np.zeros((k, 4))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau0 != tau1 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        g = gt[mov]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x[mov] = x[mov] - lr * step
        x[mov] = x[mov] / np.linalg.norm(x[mov], axis=1, keepdims=True)
    return x

def _subset_lp(x, mov, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    mov = np.asarray(mov, dtype=int)
    msz = len(mov)
    if msz == 0:
        return x
    pos = np.full(n, -1, dtype=int)
    pos[mov] = np.arange(msz)
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        cfull = cmat[iu0]
        mcur = float(cfull.max())
        sel = cfull > mcur - max(20.0 * radius, 1e-9)
        I0 = iu0[0][sel]; J0 = iu0[1][sel]
        keep = np.logical_or(np.isin(I0, mov), np.isin(J0, mov))
        I = I0[keep]; J = J0[keep]
        c = cfull[sel][keep]
        p = len(I)
        if p == 0:
            break
        nv = msz * d + 1
        rows_list = []
        cols_list = []
        vals_list = []
        for r in range(p):
            i = int(I[r]); j = int(J[r])
            pi = int(pos[i]); pj = int(pos[j])
            if pi >= 0 and pj >= 0:
                for a in range(d):
                    rows_list.append(r); cols_list.append(pi * d + a); vals_list.append(float(x[j, a]))
                    rows_list.append(r); cols_list.append(pj * d + a); vals_list.append(float(x[i, a]))
                rows_list.append(r); cols_list.append(msz * d); vals_list.append(-1.0)
            elif pi >= 0:
                for a in range(d):
                    rows_list.append(r); cols_list.append(pi * d + a); vals_list.append(float(x[j, a]))
                rows_list.append(r); cols_list.append(msz * d); vals_list.append(-1.0)
            else:
                for a in range(d):
                    rows_list.append(r); cols_list.append(pj * d + a); vals_list.append(float(x[i, a]))
                rows_list.append(r); cols_list.append(msz * d); vals_list.append(-1.0)
        A_ub = coo_matrix((np.array(vals_list), (np.array(rows_list), np.array(cols_list))), shape=(p, nv)).tocsr()
        # equality: <x_i, dx_i>=0
        er = np.repeat(np.arange(msz), d)
        ec = np.arange(msz * d)
        ev = x[mov].ravel()
        A_eq = coo_matrix((ev, (er, ec)), shape=(msz, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(msz),
                          bounds=[(-radius, radius)] * (msz * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = res.x[:-1].reshape(msz, d)
        y = x.copy()
        y[mov] = y[mov] + dx
        y[mov] = y[mov] / np.linalg.norm(y[mov], axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _full_lp(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_moved(x, thr_neigh=0.75):
    m, pw, cnt, g = _contacts(x, tol=1e-9)
    rattlers = np.where(cnt == 0)[0]
    if len(rattlers) == 0:
        rattlers = np.argsort(pw)[:3]
    low = np.where(cnt <= 4)[0]
    base = set(rattlers.tolist()) | set(low.tolist())
    # neighbours with cosine above thr
    mov = set(base)
    for i in list(base):
        nbrs = np.where(g[i] > thr_neigh)[0]
        for j in nbrs:
            mov.add(int(j))
    mov = np.array(sorted(mov), dtype=int)
    return mov, rattlers, cnt, g, m

def _rattler_pool(x, rattler, fixed_idx, rng, n_rand=25000, n_best=12):
    # dense random voids vs all except rattlers
    cand = rng.standard_normal((n_rand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # exclude rattler rows: use mask
    others = x[fixed_idx]
    sims = cand @ others.T
    mx = sims.max(axis=1)
    order = np.argsort(mx)
    pool = [cand[order[k]] for k in range(min(n_best, len(order)))]
    # equidistant from 4 nearby frozen points
    # find 12 nearest fixed points to rattler position
    rpos = x[rattler]
    s = others @ rpos
    ni = np.argsort(-s)[:12]
    near = others[ni]
    # random 4-tuples
    for _ in range(60):
        idx = rng.choice(len(near), size=4, replace=False)
        A = near[idx]
        try:
            y = np.linalg.solve(A, np.ones(4))
        except Exception:
            continue
        nrm = np.linalg.norm(y)
        if not np.finite(nrm) or nrm < 1e-8:
            continue
        for sgn in (1.0, -1.0):
            v = sgn * y / nrm
            v = v / np.linalg.norm(v)
            # check feasibility roughly: max to others
            mm = float((v @ others.T).max())
            if mm < 0.905:
                pool.append(v)
    if len(pool) == 0:
        pool.append(rpos.copy())
    return np.array(pool)

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    x_best = x0.copy()
    best_val = _max_cos(x_best)
    record = 0.898160315150
    targets = [0.8975, 0.8970, 0.8965]

    # initial moved set
    thr = 0.75
    mov, rattlers, cnt, G, mcur = _build_moved(x_best, thr_neigh=thr)
    # adjust size to 25-45
    for _ in range(4):
        if len(mov) < 25 and thr > 0.60:
            thr -= 0.05
            mov, rattlers, cnt, G, mcur = _build_moved(x_best, thr_neigh=thr)
        elif len(mov) > 48 and thr < 0.82:
            thr += 0.03
            mov, rattlers, cnt, G, mcur = _build_moved(x_best, thr_neigh=thr)
        else:
            break
    mov_set = set(mov.tolist())
    # pools per rattler
    fixed_for_pool = np.array([i for i in range(272) if i not in set(rattlers.tolist())], dtype=int)
    pools = {}
    for r in rattlers:
        try:
            pools[int(r)] = _rattler_pool(x_best, int(r), fixed_for_pool, rng)
        except Exception:
            pools[int(r)] = x_best[[int(r)]].copy()

    sigmas = [0.03, 0.015, 0.05, 0.008, 0.04, 0.02]
    taus = [(0.012, 0.0012), (0.02, 0.002), (0.007, 0.0007), (0.03, 0.003)]
    lrs = [(0.008, 0.0008), (0.012, 0.001), (0.005, 0.0005)]

    mov_other = np.array([i for i in mov if i not in set(rattlers.tolist())], dtype=int)
    trial = 0
    # surgery loop
    while time.time() < deadline - 40.0:
        trial += 1
        # periodically rebuild mov/pools if stuck or improved
        if trial % 12 == 1 and trial > 1:
            # expand neighbourhood progressively if no improvement recently
            # rebuild from current best
            thr_try = thr
            # cycle thresholds to unfreeze larger neighbourhoods
            if trial % 24 == 1:
                thr_try = max(0.65, thr - 0.05)
            mov2, ratt2, cnt2, G2, m2 = _build_moved(x_best, thr_neigh=thr_try)
            if 15 <= len(mov2) <= 60:
                mov, rattlers = mov2, ratt2
                cnt, G = cnt2, G2
                mov_other = np.array([i for i in mov if i not in set(rattlers.tolist())], dtype=int)
                fixed_for_pool = np.array([i for i in range(272) if i not in set(rattlers.tolist())], dtype=int)
                pools = {}
                for r in rattlers:
                    try:
                        pools[int(r)] = _rattler_pool(x_best, int(r), fixed_for_pool, rng)
                    except Exception:
                        pools[int(r)] = x_best[[int(r)]].copy()
        sigma = sigmas[trial % len(sigmas)]
        tau0, tau1 = taus[trial % len(taus)]
        lr0, lr1 = lrs[trial % len(lrs)]
        x_try = x_best.copy()
        # relocate each rattler to a candidate
        for r in rattlers:
            r = int(r)
            pool = pools.get(r, None)
            if pool is None or len(pool) == 0:
                continue
            # bias to best voids early, random later
            if trial % 3 == 0:
                # best of pool by max-to-fixed
                sims = pool @ x_best[fixed_for_pool].T
                mx = sims.max(axis=1)
                choice = int(np.argmin(mx))
                # sometimes pick 2nd/3rd best
                if rng.random() < 0.5 and len(pool) > 3:
                    order = np.argsort(mx)
                    choice = int(order[rng.integers(0, min(4, len(pool)))])
            else:
                choice = int(rng.integers(0, len(pool)))
            v = pool[choice] + 0.005 * rng.standard_normal(4)
            v = v / np.linalg.norm(v)
            x_try[r] = v
        # perturb other moved points
        if len(mov_other) > 0:
            x_try[mov_other] = x_try[mov_other] + sigma * rng.standard_normal((len(mov_other), 4))
            x_try[mov_other] = x_try[mov_other] / np.linalg.norm(x_try[mov_other], axis=1, keepdims=True)
        steps = 550 if (deadline - time.time()) > 80 else 350
        try:
            x_try = _adam_moving(x_try, mov, steps, tau0, tau1, lr0, lr1)
        except Exception:
            continue
        # small LP feasibility / polish on moved set every other trial
        if trial % 2 == 0:
            try:
                x_try = _subset_lp(x_try, mov, seconds=1.2, deadline=deadline - 38.0)
            except Exception:
                pass
        try:
            v = _max_cos(x_try)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        # check rattler near-contacts for each target logic
        # compute per-rattler contacts near new max
        accept = False
        if v < best_val - 1e-12 and v < record + 1e-12:
            # require rattler gains >=3 near-contacts (within 5e-3 of new max) unless big drop
            try:
                gtry = x_try @ x_try.T
                np.fill_diagonal(gtry, -2.0)
                ok = True
                for r in rattlers:
                    r = int(r)
                    row = gtry[r]
                    # near-contacts: above v-0.005 and above current target-0.001
                    nc = int((row >= v - 0.005).sum())
                    if nc < 3 and (best_val - v) < 5e-5:
                        # also check against targets: need 3 above 0.8965-ish
                        nc2 = int((row >= targets[-1] - 0.001).sum())
                        if nc2 < 3:
                            ok = False
                            break
                if ok or (best_val - v) > 1e-4:
                    accept = True
                else:
                    # still accept if strictly below record and better, but count as weak
                    # accept weak improvement to allow topology change to propagate
                    if v < best_val - 2e-7:
                        accept = True
            except Exception:
                accept = True
        if accept:
            x_best = _norm(x_try)
            best_val = v
            # rebuild moved set around new optimum
            mov, rattlers, cnt, G, mcur = _build_moved(x_best, thr_neigh=thr)
            if len(mov) < 15 or len(mov) > 60:
                thr = 0.75
                mov, rattlers, cnt, G, mcur = _build_moved(x_best, thr_neigh=thr)
            mov_other = np.array([i for i in mov if i not in set(rattlers.tolist())], dtype=int)
            fixed_for_pool = np.array([i for i in range(272) if i not in set(rattlers.tolist())], dtype=int)
            pools = {}
            for r in rattlers:
                try:
                    pools[int(r)] = _rattler_pool(x_best, int(r), fixed_for_pool, rng)
                except Exception:
                    pools[int(r)] = x_best[[int(r)]].copy()
        # early exit if we beat lowest target
        if best_val < 0.8965:
            break
    # final full LP polish
    remain = deadline - time.time() - 2.0
    if remain > 5:
        try:
            x_pol = _full_lp(x_best, seconds=min(remain, 35.0), deadline=deadline - 1.0)
            vpol = _max_cos(x_pol)
            if vpol < best_val and np.all(np.isfinite(x_pol)):
                x_best = x_pol
                best_val = vpol
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
