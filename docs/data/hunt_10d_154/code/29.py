import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax(x, steps, lr, T0, T1):
    n, d = x.shape
    m_ad = np.zeros_like(x)
    v_ad = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum() * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step / np.maximum(sn / 1.0, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            # keep most tight
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            p = 2500
            thr = float(c[sel][order[-1]])
            bvals = -c[sel][order]
        else:
            bvals = -c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=bvals, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_templates():
    d = 10
    Pa, Pb, Pc, Pd = [], [], [], []
    s2 = np.sqrt(2.0)
    for comb in itertools.combinations(range(d), 2):
        for s in itertools.product((-1.0, 1.0), repeat=2):
            v = np.zeros(d)
            v[comb[0]] = s[0] / s2
            v[comb[1]] = s[1] / s2
            Pa.append(v)
    Pa = np.array(Pa)
    s3 = np.sqrt(3.0)
    for comb in itertools.combinations(range(d), 3):
        for s in itertools.product((-1.0, 1.0), repeat=3):
            v = np.zeros(d)
            for k, c in enumerate(comb):
                v[c] = s[k] / s3
            Pb.append(v)
    Pb = np.array(Pb)
    s6 = np.sqrt(6.0)
    for i2 in range(d):
        rest = [c for c in range(d) if c != i2]
        for comb in itertools.combinations(rest, 2):
            for s in itertools.product((-1.0, 1.0), repeat=3):
                v = np.zeros(d)
                v[i2] = 2.0 * s[0] / s6
                v[comb[0]] = s[1] / s6
                v[comb[1]] = s[2] / s6
                Pc.append(v)
    Pc = np.array(Pc)
    for comb in itertools.combinations(range(d), 4):
        for s in itertools.product((-1.0, 1.0), repeat=4):
            v = np.zeros(d)
            for k, c in enumerate(comb):
                v[c] = s[k] * 0.5
            Pd.append(v)
    Pd = np.array(Pd)
    return Pa, Pb, Pc, Pd

def _random_sparse_pool(rng, m, d=10):
    if m <= 0:
        return None
    out = np.zeros((m, d))
    for i in range(m):
        k = int(rng.integers(2, 5))  # 2..4
        supp = rng.choice(d, size=k, replace=False)
        vals = rng.standard_normal(k)
        n = np.linalg.norm(vals)
        if n == 0:
            vals[0] = 1.0
            n = 1.0
        vals = vals / n
        out[i, supp] = vals
    return _norm(out)

def _greedy_select(P, rng, tau, Ktop):
    Pp = P.shape[0]
    # random start
    idx0 = int(rng.integers(Pp))
    sel = [idx0]
    selmask = np.zeros(Pp, dtype=bool)
    selmask[idx0] = True
    p0 = P[idx0]
    best_scores = P @ p0  # max dot with selected
    best_scores[idx0] = np.inf
    # sum exp tie breaker
    # avoid overflow: exp((v-1)/tau)*exp(1/tau) -> use exp(v/tau) clipped
    # tau>=0.02 so exp(1/0.02)=e50 ~ 5e21 fits in double
    sum_exp = np.exp(np.clip(best_scores / tau, -50, 50))
    sum_exp[idx0] = np.inf
    n_target = 154
    for _ in range(1, n_target):
        # min max
        # best_scores has inf for selected
        m_min = float(np.min(best_scores))
        # ties
        tie = np.where(best_scores <= m_min + 1e-12)[0]
        # tie already excludes selected (inf)
        if len(tie) > Ktop:
            se = sum_exp[tie]
            part = np.argpartition(se, Ktop - 1)[:Ktop]
            cand = tie[part]
            r = rng.random()
            if r < 0.55:
                # best
                b = int(np.argmin(sum_exp[cand]))
                if rng.random() < 0.25:
                    pick = int(cand[int(rng.integers(Ktop))])
                else:
                    pick = int(cand[b])
            else:
                pick = int(cand[int(rng.integers(Ktop))])
        elif len(tie) == 1:
            pick = int(tie[0])
        else:
            if rng.random() < 0.6:
                pick = int(tie[int(np.argmin(sum_exp[tie]))])
            else:
                pick = int(tie[int(rng.integers(len(tie)))])
        sel.append(pick)
        selmask[pick] = True
        nd = P @ P[pick]
        # update
        # best_scores = max
        np.maximum(best_scores, nd, out=best_scores)
        best_scores[pick] = np.inf
        sum_exp += np.exp(np.clip(nd / tau, -50, 50))
        sum_exp[pick] = np.inf
    X = P[np.array(sel)]
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    Pa, Pb, Pc, Pd = _build_templates()
    EXPL_END = t0 + 138.0
    passes = 0
    # quick estimate of greedy cost; adapt
    while time.time() < EXPL_END:
        passes += 1
        # random mixture
        pools = []
        # include probs
        def maybe_include(Pbase, prob=0.85):
            if rng.random() < prob:
                frac = 0.5 + 0.5 * rng.random()
                m = Pbase.shape[0]
                k = int(m * frac)
                if k >= m:
                    return Pbase
                idx = rng.choice(m, size=k, replace=False)
                return Pbase[idx]
            return None
        a = maybe_include(Pa, 0.90)
        if a is not None:
            pools.append(a)
        b = maybe_include(Pb, 0.85)
        if b is not None:
            pools.append(b)
        c = maybe_include(Pc, 0.85)
        if c is not None:
            pools.append(c)
        e = maybe_include(Pd, 0.85)
        if e is not None:
            pools.append(e)
        nrand = int(rng.integers(0, 1200))
        if nrand > 0:
            pools.append(_random_sparse_pool(rng, nrand, d))
        if len(pools) == 0:
            pools.append(Pa)
        Ppool = np.concatenate(pools, axis=0)
        # dedup not needed
        tau = float(0.02 + rng.random() * 0.06)  # 0.02..0.08
        Ktop = int(rng.integers(3, 13))  # 3..12
        try:
            X0 = _greedy_select(Ppool, rng, tau, Ktop)
        except Exception:
            continue
        pre = _max_cos(X0)
        # break symmetry with noise
        sigma = float(0.004 + rng.random() * 0.022)
        X = _norm(X0 + rng.standard_normal((n, d)) * sigma)
        # cheap relax: two-stage params randomised
        if passes % 3 == 0:
            X = _relax(X, steps=380, lr=0.016, T0=0.014, T1=0.0025)
            X = _relax(X, steps=280, lr=0.010, T0=0.004, T1=0.0008)
        elif passes % 3 == 1:
            X = _relax(X, steps=500, lr=0.014, T0=0.010, T1=0.0010)
        else:
            X = _relax(X, steps=450, lr=0.012, T0=0.012, T1=0.0012)
        val = _max_cos(X)
        # polish only if promising (threshold ~0.43, allow slightly above pre)
        if val > 0.435:
            # occasional second try with different noise? skip to save time
            if time.time() > EXPL_END:
                break
            continue
        # extra cold snap for very promising
        if val < best_val + 0.005:
            # short LP
            lp_budget = 1.4
            if val < best_val - 0.0005:
                lp_budget = 2.2
            # ensure time
            if time.time() + lp_budget + 1.0 > EXPL_END + 8:
                lp_budget = max(0.6, EXPL_END + 8 - time.time() - 1.0)
            try:
                Xp = _lp_polish(X, lp_budget, deadline=DEAD - 12.0)
                valp = _max_cos(Xp)
                if valp < val:
                    X, val = Xp, valp
            except Exception:
                pass
        if val < best_val:
            best = X.copy()
            best_val = val
            # save? not needed
        if time.time() > EXPL_END:
            break
    # final polish remaining time
    remain = DEAD - time.time() - 1.0
    if remain > 3 and best is not None:
        lp_time = min(max(5.0, remain - 1.0), 30.0)
        try:
            best = _lp_polish(best, lp_time, deadline=DEAD)
        except Exception:
            pass
    best = _norm(best)
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.standard_normal((int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
