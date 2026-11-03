import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def anneal_refine(v, steps, lr0, lr1, a0, a1, beta=0.9, track=True):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac) if a0 > 0 else a1
        lr = lr0 + (lr1 - lr0) * frac
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        cur = cur - lr * (mom / bc)
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 250 == 249) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def greedy_reinsert(base, idx, rng, n_cand=4000):
    x0 = base.copy()
    n, d = base.shape
    idx = list(idx)
    # temporarily remove: for scoring ignore points in idx not yet placed
    # sequential farthest insertion
    placed = np.ones(n, dtype=bool)
    placed[idx] = False
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # dots vs fixed points
        fixed = x0[placed]
        if fixed.shape[0] > 0:
            dots_fixed = cand @ fixed.T
            mf = dots_fixed.max(axis=1)
        else:
            mf = np.full(n_cand, -2.0)
        dots_new = cand @ x0[k].reshape(d, 1) if False else None
        # also avoid other idx points already placed? placed tracks fixed; k itself excluded
        md = mf
        # penalize closeness among candidates is handled sequentially; add small repulsion among new set via already reinserted
        best_c = np.argmin(md)
        x0[k] = cand[best_c]
        placed[k] = True
    return x0

def cap_rotation(base, rng, strength=0.3):
    n, d = base.shape
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    proj = base @ u
    # choose threshold to get cap of size 5..20
    k_target = int(rng.integers(5, 21))
    thr = np.sort(proj)[n - k_target]
    idx = np.where(proj >= thr)[0]
    # random small rotation: Q = expm of skew; approximate via QR of perturbed identity
    A = rng.standard_normal((d, d)) * strength
    A = A - A.T
    # first-order Cayley: Q = (I+A/2)(I-A/2)^{-1} approx; use QR for exact orthogonal
    M = np.eye(d) + A
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    x0 = base.copy()
    x0[idx] = x0[idx] @ Q.T
    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    return x0, idx

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x

def solve():
    t0 = time.time()
    TIME_BUDGET = 172.0
    EXPLORE_UNTIL = 132.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    # quick screening of from-scratch quenches to find different basin
    # a few fast random starts
    pop = [(best_val, best_x.copy())]
    # include record as diversity if different
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (99, 13):
            vr = max_cos(xr)
            if vr < best_val + 1e-6:
                pop.append((vr, xr.copy()))
    except Exception:
        pass

    it = 0
    stall = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 500:
        it += 1
        remain = EXPLORE_UNTIL - (time.time() - t0)
        if remain < 5:
            break
        # base selection: mostly best, sometimes diverse
        r = rng.random()
        if r < 0.65 or len(pop) == 1:
            base = best_x
            bv = best_val
        else:
            k = rng.integers(0, len(pop))
            bv, base = pop[k][0], pop[k][1]
        # adapt strength with stall
        extra = min(stall * 0.02, 0.15)
        u = rng.random()
        if u < 0.18:
            # from-scratch fast quench seed (topology reset)
            x0 = rng.standard_normal(size=(n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(900, 1600))
            lr0 = float(rng.uniform(0.03, 0.07)); lr1 = lr0 * 0.25
            a0 = float(rng.uniform(12, 25)); a1 = float(rng.uniform(80, 160))
            x1, s1 = anneal_refine(x0, steps, lr0, lr1, a0, a1)
        elif u < 0.48:
            # contact-neighborhood surgery (main topology change)
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # contacts count for hub
            thr_c = m - 1e-7
            cnt = (G > thr_c).sum(axis=1)
            # hub biased to overloaded
            if rng.random() < 0.8:
                hub = int(np.argmax(cnt + rng.random(n) * 2.0))
            else:
                rowmax = G.max(axis=1)
                hub = int(np.argmax(rowmax + rng.random(n) * 0.01))
            K = int(rng.integers(6, 15))
            order = np.argsort(-G[hub])
            idx = order[:K]
            # include hub
            if hub not in idx:
                idx = np.append(idx[:-1], hub)
            x0 = greedy_reinsert(base, idx, rng, n_cand=4000)
            # small shake of reinserted to break symmetry
            sig = 0.02 + extra
            x0[idx] += rng.standard_normal(size=(len(idx), d)) * sig
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1000, 2200))
            lr0 = float(rng.uniform(0.02, 0.055)); lr1 = lr0 * 0.3
            a0 = float(rng.uniform(15, 30)); a1 = float(rng.uniform(90, 200))
            x1, s1 = anneal_refine(x0, steps, lr0, lr1, a0, a1)
        elif u < 0.63:
            # cap rigid rotation
            st = float(rng.uniform(0.15, 0.5 + extra))
            x0, idx = cap_rotation(base, rng, strength=st)
            # light shake on boundary? add noise to all
            sig = 0.01 + extra * 0.5
            x0 += rng.standard_normal(size=(n, d)) * sig * 0.3
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(1000, 2000))
            lr0 = float(rng.uniform(0.02, 0.05)); lr1 = lr0 * 0.3
            a0 = 20.0; a1 = float(rng.uniform(100, 180))
            x1, s1 = anneal_refine(x0, steps, lr0, lr1, a0, a1)
        elif u < 0.83:
            # worst-k kick + noise
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            kk = int(rng.integers(2, 9))
            idx = np.argsort(rowmax)[-kk:]
            x0 = base.copy()
            sigma = float(rng.uniform(0.05, 0.25)) + extra
            x0[idx] += rng.standard_normal(size=(kk, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(900, 2000))
            lr0 = float(rng.uniform(0.015, 0.05)); lr1 = lr0 * 0.3
            a0 = float(rng.uniform(15, 30)); a1 = float(rng.uniform(80, 180))
            x1, s1 = anneal_refine(x0, steps, lr0, lr1, a0, a1)
        else:
            # global shake
            sigma = float(rng.uniform(0.02, 0.20)) + extra * 0.5
            x0 = base + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            steps = int(rng.integers(900, 2000))
            lr0 = float(rng.uniform(0.02, 0.055)); lr1 = lr0 * 0.3
            a0 = float(rng.uniform(15, 30)); a1 = float(rng.uniform(90, 190))
            x1, s1 = anneal_refine(x0, steps, lr0, lr1, a0, a1)
        s1 = max_cos(x1)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
            stall = 0
        else:
            stall += 1
            if len(pop) < 5 and s1 < pop[-1][0] + 0.005:
                # keep near-best diverse if far in config space? simple keep
                pop.append((s1, x1.copy()))
                pop = sorted(pop, key=lambda z: z[0])[:5]
            elif rng.random() < 0.03 and len(pop) < 5:
                pop.append((s1, x1.copy()))
                pop = sorted(pop, key=lambda z: z[0])[:5]
    remain = TIME_BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        best_x = lp_polish(best_x, remain * 0.9)
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
