import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _random_sphere(rng, n, d):
    Y = rng.normal(size=(n, d))
    return _norm(Y)


def _two_sparse_pool():
    d = 13
    v = 1.0 / np.sqrt(2.0)
    rows = []
    for a in range(d):
        for b in range(a + 1, d):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    x = np.zeros(d)
                    x[a] = sa * v
                    x[b] = sb * v
                    rows.append(x)
    return np.array(rows)  # 312 x 13


def _three_sparse_pool(rng, m=2288):
    # subsample of (±ea±eb±ec)/sqrt3 for speed/diversity
    d = 13
    v = 1.0 / np.sqrt(3.0)
    rows = []
    # enumerate all would be 2288; take all if m>=2288
    import itertools
    all_idx = list(itertools.combinations(range(d), 3))
    rng.shuffle(all_idx)
    cnt = 0
    for (a, b, c) in all_idx:
        if cnt >= m:
            break
        # 8 signs: take random 4 to keep pool bounded if needed
        for sa, sb, sc in [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                           (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]:
            if cnt >= m:
                break
            # subsample signs randomly to hit m
            x = np.zeros(d)
            x[a] = sa * v
            x[b] = sb * v
            x[c] = sc * v
            rows.append(x)
            cnt += 1
    return np.array(rows)


def _random_orth(rng, d):
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    return Q


def _greedy_from_pool(pool, rng, k=73):
    m = pool.shape[0]
    start = int(rng.integers(0, m))
    sel = [start]
    selset = {start}
    # iterative greedy maximin (minimize max dot)
    chosen = pool[start:start + 1, :]
    # keep best-max array
    # compute max dot of every pool member to chosen set
    maxd = (pool @ chosen.T)[:, 0]
    maxd[start] = 1e9
    for _ in range(1, k):
        # tie-break with tiny noise for diversity
        noise = rng.random(m) * 1e-12
        cand = np.argmin(maxd + noise * 0 + noise * 0)  # placeholder
        # pick argmin excluding selected (selected have 1e9)
        # add small random jitter to allow diversity among ties
        jit = rng.random(m) * 1e-9
        # don't jitter selected
        score = maxd + jit
        # force selected to stay large
        for s in sel:
            score[s] = 1e9
        cand = int(np.argmin(score))
        sel.append(cand)
        selset.add(cand)
        dvec = pool[cand:cand + 1, :]
        dots = (pool @ dvec.T)[:, 0]
        maxd = np.maximum(maxd, dots)
        for s in sel:
            maxd[s] = 1e9
    return pool[np.array(sel)]


def _cyclic_seed(rng):
    d = 13
    base = rng.normal(size=(5, d))
    base = _norm(base)
    pts = []
    for v in base:
        for s in range(d):
            pts.append(np.roll(v, s))
    X65 = np.array(pts)  # 65 x 13
    X65 = _norm(X65)
    # 8 greedily chosen holes from random candidates
    ncand = 3000
    C = _random_sphere(rng, ncand, d)
    # sequential greedy
    cur = X65.copy()
    for _ in range(8):
        dots = C @ cur.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        cur = np.vstack([cur, C[bi:bi + 1, :]])
        # remove chosen to avoid reuse (optional)
        # keep C as is
    return _norm(cur)


def _cheap_adam(X0, steps=350, s0=25.0, s1=170.0, lr0=0.02, lr1=0.003, t_end=1e30):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    n = X.shape[0]
    for k in range(steps):
        if k % 50 == 0 and time.time() > t_end:
            break
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = float(np.exp(ls0 + (ls1 - ls0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        E = np.exp(s * (G - cmax))
        np.fill_diagonal(E, 0.0)
        sumE = float(E.sum() * 0.5)
        if not np.isfinite(sumE) or sumE == 0:
            continue
        W = E / sumE
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1
        m1 += (1.0 - b1) * gt
        v1 *= b2
        v1 += (1.0 - b2) * (gt * gt)
        mhat = m1 / (1.0 - b1p)
        vhat = v1 / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X -= lr * step
        X = _norm(X)
    return X


def _lbfgs_refine(X0, alpha=150.0, maxiter=120, t_end=1e30):
    n, d = X0.shape
    iu0 = np.triu_indices(n, 1)
    Y = X0.ravel().copy().astype(float)

    def fun(Yflat):
        Ym = Yflat.reshape(n, d)
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Ym / rn
        G = X @ X.T
        dots = G[iu0]
        M = float(dots.max())
        e = np.exp(alpha * (dots - M))
        S = float(e.sum())
        loss = M + np.log(S) / alpha
        p = e / S
        W = np.zeros((n, n))
        W[iu0] = p
        W = W + W.T
        grad_X = W @ X
        dg = (grad_X * X).sum(axis=1, keepdims=True)
        grad_Xt = grad_X - dg * X
        grad_Y = (grad_Xt / rn).ravel()
        return float(loss), grad_Y.astype(float)

    try:
        res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                       options={"maxiter": int(maxiter), "ftol": 1e-13, "gtol": 1e-7, "maxcor": 10})
        Y = res.x
    except Exception:
        pass
    Ym = Y.reshape(n, d)
    return _norm(Ym)


def _lp_polish(x, seconds, t_end=1e30):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    n, d = 73, 13
    rng = np.random.default_rng(12345)

    # incumbent fallback so we never regress
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = _norm(xb)
    best_val = _max_cos(xb)
    X_best = xb.copy()

    base_pool = _two_sparse_pool()  # 312 x 13

    seeds = []  # (X0, family)

    # (a) 30 purely random
    for _ in range(30):
        seeds.append((_random_sphere(rng, n, d), "rand"))

    # (b) greedy algebraic: 10 seeds
    for i in range(10):
        try:
            if i % 2 == 0:
                Q = _random_orth(rng, d)
                pool = base_pool @ Q.T
                pool = _norm(pool)
            else:
                pool = _three_sparse_pool(rng, m=1200)
                pool = _norm(pool)
                Q = _random_orth(rng, d)
                pool = pool @ Q.T
                pool = _norm(pool)
            g = _greedy_from_pool(pool, rng, k=n)
            g = _norm(g)
            # small jitter to break exact symmetries
            g = _norm(g + rng.normal(size=g.shape) * 1e-3)
            seeds.append((g, "alg"))
        except Exception:
            seeds.append((_random_sphere(rng, n, d), "alg"))

    # (c) cyclic-orbit: 5 seeds
    for _ in range(5):
        try:
            seeds.append((_cyclic_seed(rng), "cyc"))
        except Exception:
            seeds.append((_random_sphere(rng, n, d), "cyc"))

    # cheap relax each seed (<= ~2s each, but we use ~350 Adam steps ~0.1-0.3s)
    # reserve time for finalists: need ~ 3*(lbfgs+lp) ~ 60s
    CHEAP_END = min(T_END - 65.0, t0 + 75.0)
    cands = []  # (val, X, family)
    # also consider incumbent as candidate? no, keep separate
    for (X0, fam) in seeds:
        if time.time() > CHEAP_END:
            break
        if time.time() > T_END - 70.0:
            break
        try:
            Xr = _cheap_adam(X0, steps=350, s0=25.0, s1=170.0,
                             lr0=0.02, lr1=0.003, t_end=T_END - 65.0)
            v = _max_cos(Xr)
            cands.append((v, Xr, fam))
            if v < best_val:
                best_val = v
                X_best = Xr.copy()
        except Exception:
            continue

    if len(cands) == 0:
        return _norm(X_best)

    # rank + diversity: keep best of each family
    fams = ["rand", "alg", "cyc"]
    finalists = []
    for f in fams:
        flt = [(v, X) for (v, X, ff) in cands if ff == f]
        if len(flt) > 0:
            flt.sort(key=lambda t: t[0])
            finalists.append((flt[0][0], flt[0][1], f))
    # sort finalists by value (best first)
    finalists.sort(key=lambda t: t[0])
    # keep at most 3 (one per family already <=3)
    finalists = finalists[:3]

    # L-BFGS + LP for finalists only, with global clock guard
    for (v0, X0, fam) in finalists:
        # need enough time to do something useful; else stop
        remain = T_END - time.time()
        if remain < 6.0:
            break
        # L-BFGS refinement (only if enough time)
        Xc = X0
        if remain > 10.0:
            try:
                # shorter maxiter if little time
                mi = 120 if remain > 40 else 60
                Xc = _lbfgs_refine(X0, alpha=150.0, maxiter=mi, t_end=T_END)
                vc = _max_cos(Xc)
                if vc < best_val:
                    best_val = vc
                    X_best = Xc.copy()
            except Exception:
                Xc = X0
        # LP polish at most 15s each, only if enough time remains
        remain = T_END - time.time()
        if remain < 5.0:
            # keep L-BFGS result already tracked
            continue
        lp_sec = min(15.0, remain - 2.0)
        if lp_sec < 2.5:
            continue
        try:
            Xp = _lp_polish(Xc, seconds=float(lp_sec), t_end=T_END - 1.0)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp
                X_best = Xp.copy()
        except Exception:
            continue
        if time.time() > T_END - 2.0:
            break

    X_best = _norm(np.asarray(X_best, dtype=float))
    # strict feasibility: no NaN/zero rows
    assert X_best.shape == (73, 13)
    assert np.all(np.isfinite(X_best))
    nr = np.linalg.norm(X_best, axis=1)
    assert np.all(nr > 1e-8)
    return X_best
