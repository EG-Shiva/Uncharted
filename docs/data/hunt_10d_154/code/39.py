import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
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
        if T < 1e-12:
            T = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        mv = G.max()
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
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
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
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _build_struct_pool():
    d = 10
    vecs = []
    # (1,1,0^8)/sqrt2
    for combo in itertools.combinations(range(d), 2):
        for s in itertools.product((-1.0, 1.0), repeat=2):
            v = np.zeros(d)
            v[combo[0]] = s[0]
            v[combo[1]] = s[1]
            v /= np.sqrt(2.0)
            vecs.append(v)
    # (1,1,1,0^7)/sqrt3
    for combo in itertools.combinations(range(d), 3):
        for s in itertools.product((-1.0, 1.0), repeat=3):
            v = np.zeros(d)
            for k, si in zip(combo, s):
                v[k] = si
            v /= np.sqrt(3.0)
            vecs.append(v)
    # (2,1,1,0^7)/sqrt6
    for i in range(d):
        rem = [j for j in range(d) if j != i]
        for combo in itertools.combinations(rem, 2):
            for s0 in (-1.0, 1.0):
                for s1 in itertools.product((-1.0, 1.0), repeat=2):
                    v = np.zeros(d)
                    v[i] = 2.0 * s0
                    v[combo[0]] = s1[0]
                    v[combo[1]] = s1[1]
                    v /= np.sqrt(6.0)
                    vecs.append(v)
    # (1,1,1,1,0^6)/2
    for combo in itertools.combinations(range(d), 4):
        for s in itertools.product((-1.0, 1.0), repeat=4):
            v = np.zeros(d)
            for k, si in zip(combo, s):
                v[k] = si
            v /= 2.0
            vecs.append(v)
    P = np.stack(vecs, axis=0)
    return _norm(P)

def _random_pool(rng, m_total, d=10):
    m_dense = m_total // 2
    m_sparse = m_total - m_dense
    Pd = rng.normal(size=(m_dense, d))
    Pd = _norm(Pd)
    Ps = np.zeros((m_sparse, d))
    for i in range(m_sparse):
        k = int(rng.integers(2, 6))  # 2..5
        supp = rng.choice(d, size=k, replace=False)
        vals = rng.normal(size=k)
        Ps[i, supp] = vals
    Ps = _norm(Ps)
    return np.concatenate([Pd, Ps], axis=0)

def _greedy_select(P, rng, eps, T2, jit):
    M, d = P.shape
    N = 154
    sel = np.empty(N, dtype=np.int64)
    used = np.zeros(M, dtype=bool)
    first = int(rng.integers(M))
    sel[0] = first
    used[first] = True
    cmax = P @ P[first]
    cmax[first] = np.inf
    for k in range(1, N):
        # current minimum
        m = np.min(np.where(used, np.inf, cmax))
        cand = np.where((~used) & (cmax <= m + eps))[0]
        if cand.size == 0:
            # fallback: take global minimizer
            cand = np.array([int(np.argmin(np.where(used, np.inf, cmax)))])
        if cand.size == 1:
            best_idx = int(cand[0])
        else:
            Lc = cand.size
            if Lc > 600:
                cand = rng.choice(cand, size=600, replace=False)
                Lc = 600
            S = P[sel[:k]]
            Pc = P[cand]
            D = Pc @ S.T
            # sum of exponentials tie-breaker
            # clip for stability
            S2 = np.exp(D / T2).sum(axis=1)
            jr = rng.random(size=S2.shape)
            scores = np.log(S2 + 1e-300) + jit * jr
            best_idx = int(cand[int(np.argmin(scores))])
        sel[k] = best_idx
        used[best_idx] = True
        dnew = P @ P[best_idx]
        # max update
        # use out-of-place maximum
        cmax = np.maximum(cmax, dnew)
        cmax[best_idx] = np.inf
    return P[sel]

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = float(v)
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    try:
        Pstruct = _build_struct_pool()
    except Exception:
        Pstruct = _norm(rng.normal(size=(7000, d)))
    Ms = Pstruct.shape[0]
    HOP_END = DEAD - 34.0
    polished = 0
    passes = 0
    # track backlog of candidates if none below threshold
    backlog = []  # list of (pre, X0)
    while time.time() < HOP_END:
        passes += 1
        # random mixture params
        keep_p = 0.75 + 0.25 * rng.random()
        mask = rng.random(Ms) < keep_p
        if mask.sum() < 2000:
            mask = np.ones(Ms, dtype=bool)
        Nr = int(rng.integers(4000, 8000))
        try:
            Pr = _random_pool(rng, Nr, d)
        except Exception:
            Pr = _norm(rng.normal(size=(Nr, d)))
        Pact = np.concatenate([Pstruct[mask], Pr], axis=0)
        eps = 0.015 + rng.random() * 0.035
        T2 = 0.03 + rng.random() * 0.05
        jit = 0.15 + rng.random() * 0.35
        try:
            X0 = _greedy_select(Pact, rng, eps, T2, jit)
        except Exception:
            continue
        try:
            pre = _mc(X0)
        except Exception:
            continue
        # keep backlog (small)
        if len(backlog) < 4:
            backlog.append((pre, X0.copy()))
            backlog.sort(key=lambda z: z[0])
        else:
            if pre < backlog[-1][0]:
                backlog[-1] = (pre, X0.copy())
                backlog.sort(key=lambda z: z[0])
        do_polish = False
        if pre < 0.43:
            do_polish = True
        elif polished < 2 and pre < 0.50 and passes > 5:
            # ensure at least some polishes to use time
            do_polish = True
        if not do_polish:
            # limit passes count to avoid infinite cheap loop without polish
            if passes > 400:
                break
            # if we have many passes and time is going, force polish best backlog occasionally
            if passes % 60 == 0 and time.time() < HOP_END - 10:
                do_polish = True
                # polish best backlog entry
                X0 = backlog[0][1].copy()
                pre = backlog[0][0]
            else:
                continue
        if time.time() > HOP_END - 4:
            break
        # break symmetry with tiny noise
        sigma = 0.002 + rng.random() * 0.008
        X = X0 + rng.normal(scale=sigma, size=X0.shape)
        X = _norm(X)
        # annealed relax
        try:
            steps = int(350 + rng.integers(250))
            lr = 0.012 + rng.random() * 0.008
            T0a = 0.009 + rng.random() * 0.005
            T1a = 0.0008 + rng.random() * 0.001
            X = _relax(X, steps=steps, lr=lr, T0=T0a, T1=T1a)
        except Exception:
            pass
        # light LP polish (bounded)
        try:
            X = _lp_polish(X, seconds=2.2, deadline=HOP_END)
        except Exception:
            pass
        try:
            v = _mc(X)
        except Exception:
            continue
        polished += 1
        if v < best_val:
            best_val = float(v)
            best = X.copy()
            if best_val < 0.4012522 - 1e-6:
                # keep searching but note improvement
                pass
        # early exit if time low
        if time.time() > HOP_END:
            break
    # final polish of best
    remain = DEAD - time.time() - 1.0
    if remain < 5:
        remain = 5
    if remain > 32:
        remain = 32
    try:
        best = _lp_polish(best, seconds=remain, deadline=DEAD - 0.5)
    except Exception:
        pass
    best = _norm(best)
    return best
