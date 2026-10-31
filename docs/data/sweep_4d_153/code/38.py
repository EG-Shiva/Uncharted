import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_counts(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - tol
    I = iu[0][sel]; J = iu[1][sel]
    cnt = np.zeros(x.shape[0], dtype=int)
    np.add.at(cnt, I, 1)
    np.add.at(cnt, J, 1)
    return cnt, float(m)

def _smooth(x, steps=300, T0=0.02, T1=0.003, s0=0.015, s1=0.0012):
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
        x = x - s * (Gt / norms)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
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
        # cap constraints for speed
        if p > 2500:
            # keep tightest 2500
            idx = np.argpartition(-c[sel], 2500)[:2500]
            I, J = I[idx], J[idx]
            ss = np.zeros(int(sel.sum()), dtype=bool); ss[idx] = True
            csel = c[sel][idx]
            p = 2500
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _best_hole(others, rng, M=20000):
    cand = rng.normal(size=(M, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    return cand[np.argmin(worst)]

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 142.0  # reserve ~30s final polish
    # load records
    loaded = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    loaded.append(a)
        except Exception:
            pass
    if len(loaded) == 0:
        rng0 = np.random.default_rng(0)
        a = rng0.normal(size=(153, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        loaded.append(a)
    vals = [(_max_cos(a), a) for a in loaded]
    vals.sort(key=lambda t: t[0])
    gbest_val, gbest = vals[0][0], vals[0][1].copy()
    fallback = gbest.copy()

    def time_left():
        return DEADLINE - time.time()

    # branch schedule: (kind, target, seed)
    # kind: 'down' greedy, 'downr' random-high, 'up' overshoot
    schedule = [
        ('down', 148, 101),
        ('up', 154, 201),
        ('downr', 149, 102),
        ('up', 155, 202),
        ('down', 150, 103),
        ('downr', 148, 104),
        ('up', 156, 203),
        ('down', 152, 105),
        ('downr', 150, 106),
        ('downr', 151, 107),
        ('down', 149, 108),
        ('up', 154, 204),
    ]

    si = 0
    try:
        while time.time() < EXPLORE_END and time_left() > 32 and si < len(schedule):
            kind, target, seed = schedule[si]
            si += 1
            rng = np.random.default_rng(seed)
            try:
                x = gbest.copy()
                if kind in ('down', 'downr'):
                    # remove down to target
                    while x.shape[0] > target:
                        if time.time() > EXPLORE_END or time_left() < 32:
                            break
                        cnt, _ = _contact_counts(x, tol=1e-9)
                        if kind == 'down':
                            # highest-contact point (tie: smallest index for determinism)
                            mx = cnt.max()
                            cands_idx = np.where(cnt == mx)[0]
                            rm = int(cands_idx[0])
                        else:
                            # random among top-5 loaded points
                            order = np.argsort(-cnt, kind='stable')
                            K = min(5, len(order))
                            # weighted toward top but random
                            pick = int(rng.integers(0, K))
                            rm = int(order[pick])
                        x = np.delete(x, rm, axis=0)
                        # brief smooth + LP
                        x = _smooth(x, steps=150, T0=0.02, T1=0.004, s0=0.012, s1=0.0012)
                        x = _lp_polish(x, budget=0.7)
                    if x.shape[0] != target:
                        continue
                    # rebuild to 153
                    while x.shape[0] < 153:
                        if time.time() > EXPLORE_END or time_left() < 32:
                            break
                        hole = _best_hole(x, rng, M=20000)
                        x = np.vstack([x, hole.reshape(1, 4)])
                        x = _smooth(x, steps=280, T0=0.02, T1=0.0028, s0=0.014, s1=0.0012)
                        x = _lp_polish(x, budget=0.7)
                        if x.shape[0] == 153:
                            v = _max_cos(x)
                            if v < gbest_val - 1e-12:
                                gbest_val, gbest = v, x.copy()
                else:  # overshoot then prune
                    Nup = int(target)
                    # grow
                    while x.shape[0] < Nup:
                        if time.time() > EXPLORE_END or time_left() < 32:
                            break
                        hole = _best_hole(x, rng, M=20000)
                        x = np.vstack([x, hole.reshape(1, 4)])
                        x = _smooth(x, steps=250, T0=0.02, T1=0.003, s0=0.014, s1=0.0012)
                        x = _lp_polish(x, budget=0.7)
                    # prune one-by-one to 153
                    while x.shape[0] > 153:
                        if time.time() > EXPLORE_END or time_left() < 32:
                            break
                        ncur = x.shape[0]
                        G = x @ x.T
                        np.fill_diagonal(G, -2.0)
                        # quick score for removing i = max of G without row/col i
                        # brute force per i (n<=156, cheap)
                        quick = np.empty(ncur)
                        for i in range(ncur):
                            m = G[np.arange(ncur) != i][:, np.arange(ncur) != i].max() if ncur > 2 else -2.0
                            quick[i] = m
                        order = np.argsort(quick, kind='stable')
                        K = min(6, ncur)
                        shortlist = order[:K]
                        best_c = None
                        best_cv = 1e9
                        for rm in shortlist:
                            if time.time() > EXPLORE_END:
                                break
                            y = np.delete(x, int(rm), axis=0)
                            y = _lp_polish(y, budget=0.45)
                            vy = _max_cos(y)
                            if vy < best_cv:
                                best_cv = vy
                                best_c = y.copy()
                        if best_c is None:
                            # fallback: remove worst quick (largest quick)
                            rm = int(order[-1])
                            x = np.delete(x, rm, axis=0)
                            x = _lp_polish(x, budget=0.5)
                        else:
                            x = best_c
                        if x.shape[0] == 153:
                            v = _max_cos(x)
                            if v < gbest_val - 1e-12:
                                gbest_val, gbest = v, x.copy()
            except Exception:
                continue
    except Exception:
        pass

    # retain best 153 result; final LP polish with remaining time (25-30s)
    try:
        gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
        gbest_val = _max_cos(gbest)
        rem = DEADLINE - time.time() - 2.0
        if rem > 3:
            budget = min(rem, 28.0)
            gbest = _lp_polish(gbest, budget=budget)
            gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
            v = _max_cos(gbest)
            if v < gbest_val and np.all(np.isfinite(gbest)):
                gbest_val = v
            else:
                # keep better of polished vs pre-polish
                pass
    except Exception:
        gbest = fallback.copy()
    # strict fallback to loaded record if failed
    try:
        gbest = np.asarray(gbest, dtype=float)
        if gbest.shape != (153, 4) or not np.all(np.isfinite(gbest)):
            raise ValueError("bad shape")
        nrm = np.linalg.norm(gbest, axis=1, keepdims=True)
        if np.any(nrm < 1e-12) or not np.all(np.isfinite(nrm)):
            raise ValueError("zero row")
        gbest = gbest / nrm
        fv = _max_cos(fallback / np.linalg.norm(fallback, axis=1, keepdims=True))
        gv = _max_cos(gbest)
        if gv > fv + 5e-4 and False:
            pass
        # always ensure valid; if our result is NaN, use fallback
    except Exception:
        gbest = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
    gbest = gbest / np.linalg.norm(gbest, axis=1, keepdims=True)
    return gbest
