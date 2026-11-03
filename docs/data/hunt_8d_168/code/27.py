import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _quench_free(X, steps, beta0, beta1, lr0, lr1):
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    m = np.zeros_like(X); v = np.zeros_like(X)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        mhat = m / bc1; vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _cyclic_group(m, mults):
    Rs = np.zeros((m, 8, 8))
    for k in range(m):
        R = np.eye(8)
        for p, mu in enumerate(mults):
            ang = 2.0 * np.pi * mu * k / m
            c = np.cos(ang); s = np.sin(ang)
            R[2*p, 2*p] = c; R[2*p, 2*p+1] = -s
            R[2*p+1, 2*p] = s; R[2*p+1, 2*p+1] = c
        Rs[k] = R
    return Rs

def _expand(S, Rs):
    # X[s*m+k] = R_k @ S_s
    Xs = np.einsum('kij,sj->ski', Rs, S)
    return Xs.reshape(-1, 8)

def _quench_sym(S, Rs, steps, beta0, beta1, lr0, lr1):
    nseed = S.shape[0]; gm = Rs.shape[0]
    RsT = Rs.transpose(0, 2, 1)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    mad = np.zeros_like(S); vad = np.zeros_like(S)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        X = _expand(S, Rs)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = W.sum()
        if Z == 0:
            continue
        Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gtr = Gt.reshape(nseed, gm, 8)
        GS = np.einsum('kij,skj->si', RsT, Gtr)
        # project to seed tangent
        d = (GS * S).sum(axis=1, keepdims=True)
        GtS = GS - d * S
        mad *= b1; mad += (1.0 - b1) * GtS
        vad *= b2; vad += (1.0 - b2) * (GtS * GtS)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        mh = mad / bc1; vh = vad / bc2
        step = mh / (np.sqrt(vh) + eps)
        S = S - lr * step
        nrm = np.sqrt((S * S).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        S /= nrm
    return S

def _e8_roots():
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8); v[i] = si; v[j] = sj
                    pts.append(v)
    for mask in range(256):
        signs = np.ones(8); nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -1.0; nm += 1
        if nm % 2 == 0:
            pts.append(0.5 * signs)
    a = np.array(pts, dtype=float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a

def _e8_greedy(e8, n_keep, rng):
    N = e8.shape[0]
    G = e8 @ e8.T
    alive = np.ones(N, dtype=bool)
    # count pairs at ~1/2
    # iteratively remove worst
    cur_n = N
    # precompute mask of close pairs
    close = (G > 0.495)
    while cur_n > n_keep:
        idx = np.where(alive)[0]
        # count close neighbours among alive
        sub = close[np.ix_(idx, idx)].sum(axis=1)
        # tie-break by sum of G
        worst_pos = int(np.argmax(sub))
        # if all zero, remove one with largest sum
        if sub[worst_pos] == 0:
            ssum = G[np.ix_(idx, idx)].sum(axis=1)
            worst_pos = int(np.argmax(ssum + rng.random(len(idx)) * 1e-9))
        alive[idx[worst_pos]] = False
        cur_n -= 1
    return e8[alive].copy()

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if int(sel.sum()) > 4000:
            idx = np.argsort(-c)
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[:4000]] = True
            sel = sel & keep
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    n, d = 168, 8
    rng = np.random.default_rng(0)
    best_x = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)

    # --- symmetry-constrained search from scratch ---
    sym_best = None; sym_val = 1e9
    configs = [
        (7, 24, [1, 2, 3], 3),
        (14, 12, [1, 3, 5], 2),
        (21, 8, [1, 2, 4], 2),
        (28, 6, [1, 3, 5], 2),
    ]
    for mm, ns, mults, trials in configs:
        if time.time() - t0 > 75:
            break
        Rs = _cyclic_group(mm, mults)
        for tr in range(trials):
            if time.time() - t0 > 75:
                break
            S0 = rng.normal(size=(ns, d))
            S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
            # vary schedule deterministically
            b0 = 12.0 + 8.0 * tr
            b1 = 260.0 + 60.0 * tr
            l0 = 0.006 if tr % 2 == 0 else 0.009
            S = _quench_sym(S0, Rs, 1500, b0, b1, l0, l0 * 0.08)
            X = _expand(S, Rs)
            # X already unit (rotations preserve norm)
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            v = _max_cos(X)
            if v < sym_val:
                sym_val = v; sym_best = X.copy()
            # quick free micro-quench to judge released potential (short)
            Xq = _quench_free(X.copy(), 400, 60.0, 400.0, 0.003, 0.0004)
            vq = _max_cos(Xq)
            if vq < sym_val:
                sym_val = vq; sym_best = Xq.copy()

    # --- E8-subset construction: greedy delete 72 of 240 ---
    try:
        if time.time() - t0 < 95:
            e8 = _e8_roots()
            sub = _e8_greedy(e8, n, rng)
            sub = sub + rng.normal(size=(n, d)) * 0.015
            sub /= np.linalg.norm(sub, axis=1, keepdims=True)
            subq = _quench_free(sub, 1400, 20.0, 350.0, 0.005, 0.0005)
            vsub = _max_cos(subq)
            if vsub < sym_val:
                sym_val = vsub; sym_best = subq.copy()
    except Exception:
        pass

    # --- staged release of symmetry ---
    cur = sym_best if sym_best is not None else best_x.copy()
    cur_val = sym_val if sym_best is not None else best_val
    stages = [(1e-3, 1000), (1e-2, 1300), (5e-2, 1500)]
    for amp, steps in stages:
        if time.time() - t0 > BUDGET - 55:
            break
        for rep in range(2):
            if time.time() - t0 > BUDGET - 55:
                break
            # orbit-cluster style: common + individual parts
            X = cur + rng.normal(size=(n, d)) * amp
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            b0 = float(rng.uniform(20, 50)); b1 = float(rng.uniform(250, 600))
            l0 = float(rng.uniform(0.003, 0.007))
            Xq = _quench_free(X, steps, b0, b1, l0, l0 * 0.12)
            vq = _max_cos(Xq)
            if vq < cur_val:
                cur_val = vq; cur = Xq.copy()
    # merge with record basin
    glob = best_x.copy(); gval = best_val
    if cur_val < gval:
        glob = cur.copy(); gval = cur_val
    else:
        # also try releasing record-free? keep cur as alternate basin
        pass

    # --- free basin-hopping from global best (topology changes) ---
    it = 0
    while time.time() - t0 < BUDGET - 34:
        it += 1
        X = glob.copy()
        r = rng.random()
        if r < 0.5:
            sigma = float(10.0 ** rng.uniform(-2.4, -1.0))
            X = X + rng.normal(size=(n, d)) * sigma
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif r < 0.8:
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            ind = G.max(axis=1)
            k = int(rng.integers(3, 10))
            worst = np.argsort(-ind)[:k]
            X[worst] = rng.normal(size=(k, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # shake overloaded
            G = X @ X.T; np.fill_diagonal(G, -2.0)
            m_ = G.max()
            cnt = (G > m_ - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            pool = order[:30]
            k = int(rng.integers(4, 10))
            sel = rng.choice(pool, size=min(k, len(pool)), replace=False)
            X[sel] += rng.normal(size=(len(sel), d)) * float(rng.uniform(0.1, 0.4))
            X += rng.normal(size=(n, d)) * 0.005
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        beta0 = float(rng.uniform(20, 60)); beta1 = float(rng.uniform(200, 700))
        lr0 = float(rng.uniform(0.0025, 0.008))
        steps = int(rng.integers(900, 1800))
        Xq = _quench_free(X, steps, beta0, beta1, lr0, lr0 * 0.12)
        v = _max_cos(Xq)
        if v < gval - 1e-12:
            gval = v; glob = Xq.copy()
        # occasionally try alternate basin cur
        if it % 7 == 0 and cur_val < 0.6:
            X2 = cur + rng.normal(size=(n, d)) * 0.02
            X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
            X2q = _quench_free(X2, 800, 30.0, 400.0, 0.004, 0.0005)
            v2 = _max_cos(X2q)
            if v2 < cur_val:
                cur_val = v2; cur = X2q.copy()
            if v2 < gval:
                gval = v2; glob = X2q.copy()

    # final choice
    final = glob if gval <= cur_val else cur
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 4:
        final = _lp_polish(final, min(remain, 32.0))
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    return final
