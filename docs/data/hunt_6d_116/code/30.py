import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, iters, a0, a1, lr0, lr1):
    X = _norm(X.copy())
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, iters + 1):
        f = it / float(max(iters, 1))
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        W = np.exp(alpha * (G - cmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        Gr = W @ X
        proj = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * Gt * Gt
        mh = m / (1.0 - b1p); vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _deep_hole(others, rng, ntry=2000, refine=25):
    d = others.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-15)
    dots = C @ others.T
    w = dots.max(axis=1)
    y = C[int(np.argmin(w))].copy()
    step = 0.05
    for _ in range(refine):
        dd = others @ y
        j = int(np.argmax(dd))
        g = others[j] - float(others[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        y /= (np.linalg.norm(y) + 1e-15)
        step *= 0.97
    return y / (np.linalg.norm(y) + 1e-15)

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _build_pool():
    d = 6
    P = []
    # D6 roots (±1,±1,0,0,0,0)/sqrt2
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si; v[j] = sj
                    v /= np.sqrt(2.0)
                    P.append(v)
    # axis (±1,0,0,0,0,0)
    for i in range(d):
        for s in (1.0, -1.0):
            v = np.zeros(d); v[i] = s
            P.append(v)
    # (±1^6)/sqrt6 all 64
    for mask in range(64):
        v = np.ones(d)
        for i in range(d):
            if (mask >> i) & 1:
                v[i] = -1.0
        v /= np.sqrt(6.0)
        P.append(v)
    # triples (±1,±1,±1,0,0,0)/sqrt3 : subsample to keep pool moderate
    # all C(6,3)=20 position sets, all 8 signs =160
    for combo in itertools.combinations(range(d), 3):
        for mask in range(8):
            v = np.zeros(d)
            for k, pos in enumerate(combo):
                v[pos] = -1.0 if ((mask >> k) & 1) else 1.0
            v /= np.sqrt(3.0)
            P.append(v)
    # quadruples (±1,±1,±1,±1,0,0)/2 : subsample half to limit size
    cnt = 0
    for combo in itertools.combinations(range(d), 4):
        for mask in range(16):
            cnt += 1
            if (cnt % 2) == 0:
                continue
            v = np.zeros(d)
            for k, pos in enumerate(combo):
                v[pos] = -1.0 if ((mask >> k) & 1) else 1.0
            v /= 2.0
            P.append(v)
    P = np.array(P, dtype=float)
    # deduplicate roughly
    # round and unique
    Pr = np.round(P, decimals=8)
    # use view unique
    _, idx = np.unique(Pr, axis=0, return_index=True)
    P = P[np.sort(idx)]
    return P

def _greedy_from_pool(P, k, rng):
    M = P.shape[0]
    # seed with most opposite pair
    # sample subset for speed if M large
    G = P @ P.T
    # find min entry (most opposite)
    np.fill_diagonal(G, 2.0)
    ij = int(np.argmin(G))
    i0 = ij // M; j0 = ij % M
    sel = [int(i0), int(j0)]
    selset = set(sel)
    # iterative greedy
    S = P[sel]
    for _ in range(k - 2):
        dots = P @ S.T
        mx = dots.max(axis=1)
        # exclude selected
        for s in sel:
            mx[s] = np.inf
        bi = int(np.argmin(mx))
        sel.append(bi)
        selset.add(bi)
        S = P[sel]
        if len(sel) >= k:
            break
    X = P[np.array(sel[:k])].copy()
    # tiny symmetry break
    X = X + 1e-3 * rng.standard_normal(X.shape)
    return _norm(X)

def _grow_to_116(X0, rng, quench_short=160, quench_mid=350):
    X = _norm(X0.copy())
    # light initial polish
    X = _quench(X, iters=180, a0=20, a1=180, lr0=0.03, lr1=0.004)
    while X.shape[0] < 116:
        y = _deep_hole(X, rng, ntry=1800, refine=25)
        X = np.vstack([X, y.reshape(1, -1)])
        # short re-opt; vary alpha slightly by size
        X = _quench(X, iters=quench_short, a0=20, a1=220, lr0=0.03, lr1=0.004)
    X = _quench(X, iters=quench_mid, a0=25, a1=280, lr0=0.025, lr1=0.003)
    return X

def _shrink_to_116(X0, rng, quench_mid=350):
    X = _norm(X0.copy())
    X = _quench(X, iters=180, a0=20, a1=180, lr0=0.03, lr1=0.004)
    while X.shape[0] > 116:
        pm = _per_point_max(X)
        # delete worst 1 (highest per-point max)
        k = int(np.argmax(pm))
        # occasionally delete random among top-3 to diversify
        if rng.random() < 0.25:
            top = np.argsort(-pm)[:3]
            k = int(top[int(rng.integers(0, len(top)))])
        X = np.delete(X, k, axis=0)
        if (X.shape[0] % 3) == 0 and X.shape[0] > 116:
            X = _quench(X, iters=90, a0=20, a1=150, lr0=0.02, lr1=0.004)
    X = _quench(X, iters=quench_mid, a0=25, a1=280, lr0=0.025, lr1=0.003)
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1200:
            o = np.argsort(c[idx])[-1200:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    EXPL = t0 + 132.0
    rng_master = np.random.default_rng(916 healthy if False else 91611)
    # fallback best
    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        n = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(n < 1e-12):
            return None
        return a / n
    best_x = None; best_v = 2.0
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                v = _maxcos(c)
                if v < best_v:
                    best_v = v; best_x = c.copy()
        except Exception:
            pass
    if best_x is None:
        r = np.random.default_rng(0)
        best_x = _norm(r.standard_normal((116, 6)))
        best_v = _maxcos(best_x)
    pool = _build_pool()
    chains = []  # (v, X)
    # include fallback so top-3 includes record basin
    chains.append((best_v, best_x.copy()))
    chain_id = 0
    # time-sliced many chains
    while time.time() < EXPL:
        if time.time() > HARD - 30.0:
            break
        typ = chain_id % 4
        rng = np.random.default_rng(100000 + chain_id * 7919)
        try:
            if typ == 0:
                # grow random 90-100 -> 116
                N0 = int(rng.integers(90, 101))
                X0 = rng.standard_normal((N0, 6))
                X0 = _norm(X0)
                X = _grow_to_116(X0, rng, quench_short=150, quench_mid=320)
            elif typ == 1:
                # grow from lattice subset 90-100 -> 116
                N0 = int(rng.integers(90, 101))
                perm = rng.permutation(pool.shape[0])[:N0]
                X0 = pool[perm].copy() + 0.01 * rng.standard_normal((N0, 6))
                X0 = _norm(X0)
                X = _grow_to_116(X0, rng, quench_short=150, quench_mid=320)
            elif typ == 2:
                # shrink random 122-128 -> 116
                N1 = int(rng.integers(122, 129))
                # mix random + pool for diversity on alternate chains
                if (chain_id // 4) % 2 == 0:
                    X0 = _norm(rng.standard_normal((N1, 6)))
                else:
                    # random 116 + extra random to reach N1 via holes
                    base = _norm(rng.standard_normal((100, 6)))
                    base = _quench(base, iters=120, a0=15, a1=120, lr0=0.03, lr1=0.005)
                    while base.shape[0] < N1:
                        y = _deep_hole(base, rng, ntry=1200, refine=15)
                        base = np.vstack([base, y.reshape(1, -1)])
                    X0 = base
                X = _shrink_to_116(X0, rng, quench_mid=320)
            else:
                # direct lattice greedy 116
                X0 = _greedy_from_pool(pool, 116, rng)
                X = _quench(X0, iters=380, a0=25, a1=300, lr0=0.028, lr1=0.003)
            v = _maxcos(X)
            chains.append((v, X.copy()))
            if v < best_v - 1e-12:
                best_v = v; best_x = X.copy()
        except Exception:
            pass
        chain_id += 1
        # strict per-chain slice already enforced by short quenches; check time
        if chain_id > 60:
            break
    # rank chains, LP-polish top three (excluding already polished fallback if top)
    chains_sorted = sorted(chains, key=lambda t: t[0])[:3]
    # total 25s for LP
    lp_budget = min(25.0, max(4.0, HARD - 4.0 - time.time()))
    per = lp_budget / max(len(chains_sorted), 1)
    for (v, Xc) in chains_sorted:
        if time.time() > HARD - 3.0:
            break
        dl = time.time() + per
        # quick strong quench before LP for fresh chains
        try:
            if time.time() < dl - 1.0:
                Xc = _quench(Xc, iters=250, a0=40, a1=400, lr0=0.02, lr1=0.002)
            bx, bv = _lp_polish(Xc, dl)
            if bv < best_v:
                best_v = bv; best_x = bx.copy()
        except Exception:
            continue
    best_x = _norm(best_x)
    return best_x
