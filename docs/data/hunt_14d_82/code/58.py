import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_pool(rng, target=18000):
    base = []
    # weight-2 all: 91*4=364
    for combo in itertools.combinations(range(14), 2):
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v = np.zeros(14)
                v[combo[0]] = s1
                v[combo[1]] = s2
                v /= np.linalg.norm(v)
                base.append(v)
    base = np.array(base)  # 364
    w3 = []
    for combo in itertools.combinations(range(14), 3):
        for mask in range(8):
            v = np.zeros(14)
            for k, idx in enumerate(combo):
                v[idx] = 1.0 if ((mask >> k) & 1) else -1.0
            v /= np.sqrt(3.0)
            w3.append(v)
    w3 = np.array(w3)  # 2912
    n_w4 = 5000
    w4 = np.zeros((n_w4, 14))
    for i in range(n_w4):
        idx = rng.choice(14, size=4, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=4)
        w4[i, idx] = signs
    w4 = w4 / np.linalg.norm(w4, axis=1, keepdims=True)
    n_pm = 3000
    pm = rng.choice(np.array([-1.0, 1.0]), size=(n_pm, 14)).astype(float)
    pm = pm / np.sqrt(14.0)
    n_g = 6000
    g = rng.standard_normal(size=(n_g, 14))
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    pool = np.vstack([base, w3, w4, pm, g])
    # trim / pad to target
    if pool.shape[0] > target:
        sel = rng.choice(pool.shape[0], size=target, replace=False)
        # keep all sparse base+w3 always: ensure first 3276 kept
        # simple: keep first 3276 + random rest
        keep_head = 3276
        rest = pool[keep_head:]
        need = target - keep_head
        idx = rng.choice(rest.shape[0], size=need, replace=False)
        pool = np.vstack([pool[:keep_head], rest[idx]])
    return pool

def _greedy_select(pool, start_idx, n_select=82):
    P = pool.shape[0]
    cur = pool @ pool[start_idx]
    cur = cur.copy()
    cur[start_idx] = np.inf
    selected = [int(start_idx)]
    for _ in range(1, n_select):
        nxt = int(np.argmin(cur))
        selected.append(nxt)
        d = pool @ pool[nxt]
        # max
        np.maximum(cur, d, out=cur)
        cur[nxt] = np.inf
        # keep previous selected inf (already inf, max keeps inf)
        for s in selected:
            cur[s] = np.inf
    return pool[np.array(selected, dtype=int)].copy()

def _adam_full(X0, steps, tau0, tau1, lr, deadline=None):
    X = _norm(X0.copy())
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = X.copy()
    bestv = _max_cos(X)
    for s in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau0 > 0 and tau1 > 0 else tau0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp((G - cm) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.finite(S) or S <= 0:
            S = 1.0
        W = E / S
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (s + 1))
        vhat = v / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip global step
        X = X - step
        X = _norm(X)
        # track occasionally
        if (s & 31) == 31 or s == steps - 1:
            vv = _max_cos(X)
            if vv < bestv:
                bestv = vv
                best = X.copy()
    return best, bestv

def _expand_orbit(seeds):
    # seeds (ns,d), ns=6 -> 82 rows: 5*14+12
    ns, d = seeds.shape
    nrows = 82
    X = np.empty((nrows, d))
    mp = []
    r = 0
    for i in range(5):
        nrm = float(np.linalg.norm(seeds[i])) + 1e-30
        for k in range(14):
            X[r] = np.roll(seeds[i], k) / nrm
            mp.append((i, k))
            r += 1
    i = 5
    nrm = float(np.linalg.norm(seeds[i])) + 1e-30
    for k in range(12):
        X[r] = np.roll(seeds[i], k) / nrm
        mp.append((i, k))
        r += 1
    return X, mp

def _orbit_adam(seeds0, mp_static, steps, tau0, tau1, lr, deadline=None):
    seeds = seeds0.copy()
    ns, d = seeds.shape
    m = np.zeros_like(seeds)
    v = np.zeros_like(seeds)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # build map list
    # mp_static computed for shape, but recompute each time? fixed
    # precompute row->(i,k)
    rows = []
    r = 0
    for i in range(5):
        for k in range(14):
            rows.append((i, k))
            r += 1
    for k in range(12):
        rows.append((5, k))
    best_seeds = seeds.copy()
    Xb, _ = _expand_orbit(seeds)
    bestv = _max_cos(Xb)
    for s in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac
        X, _ = _expand_orbit(seeds)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp((G - cm) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.finite(S) or S <= 0:
            S = 1.0
        W = E / S
        Gr = W @ X  # (82,d) ambient grad
        # backprop to seeds
        Gs = np.zeros_like(seeds)
        for ridx, (i, k) in enumerate(rows):
            x = X[ridx]
            gg = Gr[ridx]
            c = float(gg @ x)
            tang = gg - c * x
            # inverse roll
            contrib = np.roll(tang, -k)
            nrm = float(np.linalg.norm(seeds[i])) + 1e-30
            Gs[i] += contrib / nrm
        # adam on seeds
        m = b1 * m + (1 - b1) * Gs
        v = b2 * v + (1 - b2) * (Gs * Gs)
        mhat = m / (1 - b1 ** (s + 1))
        vhat = v / (1 - b2 ** (s + 1))
        seeds = seeds - lr * mhat / (np.sqrt(vhat) + eps)
        # avoid collapse
        nr = np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-30
        # keep scale ~1 to avoid drift; renormalize seeds to unit norm each step
        seeds = seeds / nr
        if (s & 63) == 63 or s == steps - 1:
            Xt, _ = _expand_orbit(seeds)
            vv = _max_cos(Xt)
            if vv < bestv:
                bestv = vv
                best_seeds = seeds.copy()
    return best_seeds, bestv

def _lp_polish(X, time_budget, deadline, radius_init=3e-3, cap=1200):
    X = _norm(X.copy())
    best = _max_cos(X)
    radius = float(radius_init)
    t_start = time.time()
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    while time.time() - t_start < time_budget:
        if time.time() > deadline:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(np.max(c_all))
        thr = m - max(20 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            tl = max(0.5, min(3.0, deadline - time.time() - 0.1))
            if tl <= 0.15:
                break
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": float(tl)})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    BUDGET = 168.0
    deadline = t0 + BUDGET
    hard_ret = t0 + 174.0
    rng = np.random.default_rng(12345)
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = _norm(rng.standard_normal((n, d)))
        cands.append(X0)
    cands.sort(key=lambda c: _max_cos(c))
    global_best = cands[0].copy()
    best_val = _max_cos(global_best)
    # ---- structured pool greedy (cheap, capped ~20k) ----
    try:
        pool = _build_pool(rng, target=18000)
        # few greedy starts
        starts = [int(rng.integers(0, pool.shape[0])) for _ in range(4)]
        # include a sparse and dense start deterministically
        starts[0] = 0
        starts[1] = pool.shape[0] // 2
        greedy_best = None
        greedy_val = 1e9
        for si in starts:
            if time.time() > deadline - 100:
                break
            Xg = _greedy_select(pool, si, n_select=82)
            vg = _max_cos(Xg)
            if vg < greedy_val:
                greedy_val = vg
                greedy_best = Xg.copy()
        if greedy_best is not None:
            # short smooth-max Adam then LP polish
            Gb, vb = _adam_full(greedy_best, steps=220, tau0=0.02, tau1=0.003, lr=0.02, deadline=deadline - 90)
            # quick LP (small budget)
            if time.time() < deadline - 90:
                Gb2, vb2 = _lp_polish(Gb, time_budget=1.2, deadline=deadline - 88, radius_init=2e-3, cap=1000)
                Gb, vb = (Gb2, vb2) if vb2 < vb else (Gb, vb)
            if vb < best_val:
                global_best = _norm(Gb.copy())
                best_val = vb
            # keep structured best for later release even if worse
            struct_base = Gb.copy()
            struct_val = vb
        else:
            struct_base = None
    except Exception:
        struct_base = None
    # ---- group-orbit seed optimization in low-dim space ----
    try:
        if time.time() < deadline - 85:
            ns = 6
            seeds0 = rng.standard_normal((ns, d))
            seeds0 = seeds0 / np.linalg.norm(seeds0, axis=1, keepdims=True)
            # if structured greedy decent, seed from it? use random to stay different basin
            seeds_opt, _ = _orbit_adam(seeds0, None, steps=700, tau0=0.015, tau1=0.002, lr=0.03, deadline=deadline - 70)
            Xo, _ = _expand_orbit(seeds_opt)
            vo = _max_cos(Xo)
            # release all coordinates: short Adam
            Xo2, vo2 = _adam_full(Xo, steps=250, tau0=0.01, tau1=0.002, lr=0.015, deadline=deadline - 60)
            if vo2 < vo:
                Xo, vo = Xo2, vo2
            # short LP
            if time.time() < deadline - 58:
                Xo3, vo3 = _lp_polish(Xo, time_budget=1.2, deadline=deadline - 56, radius_init=2e-3, cap=1000)
                if vo3 < vo:
                    Xo, vo = Xo3, vo3
            if vo < best_val:
                global_best = _norm(Xo.copy())
                best_val = vo
    except Exception:
        pass
    # ---- remaining time: polish incumbent with cheap shakes + Adam + LP ----
    # use leftover budget to try to improve global_best slightly
    sigmas = [0.015, 0.03, 0.06]
    it = 0
    cur = global_best.copy()
    cur_val = best_val
    while time.time() < deadline - 4.0:
        if time.time() > hard_ret - 8:
            break
        it += 1
        remain = deadline - 4.0 - time.time()
        if remain < 1.0:
            break
        # small perturbation of global best (different basin kick occasionally larger)
        sigma = sigmas[it % len(sigmas)]
        if it % 12 == 0:
            sigma = 0.12
        trial = global_best + sigma * rng.standard_normal(size=(n, d))
        trial = _norm(trial)
        # fast Adam quench (few steps)
        steps = 120 if trial is not None else 120
        # adapt steps to remain
        if remain < 6:
            steps = 60
        trial_b, trial_v0 = _adam_full(trial, steps=steps, tau0=0.008, tau1=0.002, lr=0.012, deadline=deadline - 3.0)
        # filter: only LP if promising
        if trial_v0 > best_val + 0.015 and (it % 5 != 0):
            if trial_v0 < cur_val:
                cur = trial_b.copy(); cur_val = trial_v0
            continue
        blp = 0.8
        if trial_v0 < best_val + 0.002:
            blp = 1.2
        blp = min(blp, deadline - 3.0 - time.time())
        if blp < 0.3:
            # skip LP, just check
            if trial_v0 < best_val - 1e-13:
                global_best = trial_b.copy(); best_val = trial_v0; cur = trial_b.copy(); cur_val = trial_v0
            continue
        trial_p, trial_v = _lp_polish(trial_b, time_budget=blp, deadline=deadline - 2.5, radius_init=2e-3, cap=1100)
        if trial_v < best_val - 1e-13:
            global_best = trial_p.copy()
            best_val = trial_v
            cur = trial_p.copy(); cur_val = trial_v
        elif trial_v < cur_val:
            cur = trial_p.copy(); cur_val = trial_v
        # periodic reset to best
        if it % 15 == 0:
            cur = global_best.copy(); cur_val = best_val
        # quick exit if time low
        if time.time() > deadline - 4.0:
            break
    global_best = _norm(global_best)
    # final safety: ensure finite, no zero rows
    assert global_best.shape == (82, 14)
    assert np.all(np.isfinite(global_best))
    nr = np.linalg.norm(global_best, axis=1)
    assert np.all(nr > 1e-8)
    return global_best
