import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, deadline, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    iuI = iu0[0]; iuJ = iu0[1]
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc1 = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vv, (rows, cc1)), shape=(p, nv)).tocsr()
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
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def contact_info(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cm = G > m - tol
    cnt = cm.sum(axis=1)
    # nearest max (excluding self)
    G2 = x @ x.T
    np.fill_diagonal(G2, -2.0)
    nmax = G2.max(axis=1)
    return cnt, m, nmax, G2

def anneal_movable(x, mov, iters, k0, k1, lr0, lr1, mu, noise, rng):
    d = x.shape[1]
    m = len(mov)
    if m == 0:
        return x
    v = np.zeros((m, d))
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 and k1 > 0 else k0
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = float(G.max())
        # exp in stable way
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        Wm = W[mov]
        rs = Wm.sum(axis=1, keepdims=True) + 1e-12
        grad = (Wm @ x) / rs
        Xm = x[mov]
        dot = np.sum(grad * Xm, axis=1, keepdims=True)
        gt = grad - dot * Xm
        v = mu * v + gt
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        Xn = Xm - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(Xn.shape)
            pd = np.sum(pert * Xn, axis=1, keepdims=True)
            pert -= pd * Xn
            Xn = Xn + amp * pert
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        x[mov] = Xn
    return x

def largest_void(fixed, n_samp, rng):
    # random samples on S^3, pick minimizing max dot to fixed
    R = rng.standard_normal((n_samp, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    D = R @ fixed.T  # (samp, nfixed)
    md = D.max(axis=1)
    i = int(np.argmin(md))
    return R[i]

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 145.0
    rng = np.random.default_rng(1)
    n, d = 153, 4
    cands = []
    try:
        cands.append(np.load("best.npy"))
    except Exception:
        pass
    try:
        cands.append(np.load("record.npy"))
    except Exception:
        pass
    best_val = 1e9
    base = None
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (n, d):
            continue
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best_val:
            best_val = v
            base = c.copy()
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        best_val = max_cos(base)
    incumbent = base.copy()
    inc_val = best_val
    # quick initial polish to lock exact local optimum
    incumbent = lp_polish(incumbent, min(t0 + 12.0, HARD_END - 5))
    inc_val = max_cos(incumbent)

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        x = incumbent.copy()
        cnt, m0, nmax, Gfull = contact_info(x)
        order = np.argsort(cnt, kind="stable")
        # choose movable set size
        cs = int(rng.choice([6, 10, 14, 20]))
        # always include rattlers / lowest
        mov = list(order[:cs])
        # add a few random points to diversify
        if rng.random() < 0.5:
            extra = rng.integers(0, n, size=int(rng.integers(2, 6)))
            for e in extra:
                if int(e) not in mov:
                    mov.append(int(e))
        mov = np.array(sorted(set(int(v) for v in mov)), dtype=int)
        fixed_mask = np.ones(n, dtype=bool)
        fixed_mask[mov] = False
        fixed = x[fixed_mask]
        # rattler void reinsertion: for the lowest-contact point, jump to void
        if rng.random() < 0.7:
            # pick the very lowest (rattler) to reinsert
            r0 = int(order[0])
            # void w.r.t. all other points (excluding r0 itself)
            others = np.delete(x, r0, axis=0)
            nv = 12000 if rng.random() < 0.5 else 30000
            vv = largest_void(others, nv, rng)
            # blend: mostly void, sometimes jittered
            if rng.random() < 0.2:
                vv = vv + 0.05 * rng.standard_normal(4)
                vv /= np.linalg.norm(vv)
            x[r0] = vv
            # also shake neighbours of r0 (closest points) to make room
            dists = others @ vv  # max dot = closest
            nn = np.argsort(-dists)[:max(2, cs // 3)]
            # map back to original indices
            # others excludes r0, fix index shift
            for k in nn:
                idx = int(k) if int(k) < r0 else int(k) + 1
                if idx not in mov:
                    mov = np.append(mov, idx)
            mov = np.array(sorted(set(int(v) for v in mov)), dtype=int)
            fixed_mask = np.ones(n, dtype=bool); fixed_mask[mov] = False
        # tangent Gaussian shake on movable
        sigma = float(rng.choice([0.015, 0.03, 0.06, 0.10, 0.16]))
        pert = rng.standard_normal((len(mov), d))
        pd = np.sum(pert * x[mov], axis=1, keepdims=True)
        pert -= pd * x[mov]
        x[mov] = x[mov] + sigma * pert
        x[mov] /= np.linalg.norm(x[mov], axis=1, keepdims=True)
        # anneal movable with randomized schedule
        k0 = float(rng.choice([40.0, 80.0, 150.0]))
        k1 = float(rng.choice([400.0, 800.0, 1500.0]))
        if k1 < k0:
            k0, k1 = k1, k0
        lr0 = float(rng.choice([0.005, 0.01, 0.02]))
        lr1 = float(rng.choice([0.0005, 0.001, 0.002]))
        mu = float(rng.choice([0.0, 0.5, 0.8, 0.9]))
        noise = float(rng.choice([0.0, 0.002, 0.005, 0.01]))
        iters = int(rng.choice([400, 700, 1100]))
        # time guard: anneal is cheap, but cap iters if late
        if HARD_END - time.time() < 30:
            iters = min(iters, 400)
        x = anneal_movable(x, mov, iters, k0, k1, lr0, lr1, mu, noise, rng)
        # short LP unfreezing
        remain = HARD_END - time.time()
        if remain < 5:
            break
        lp_budget = min(2.2, max(0.8, remain - 22.0)) if trial < 40 else min(1.5, max(0.7, remain - 21.0))
        # occasionally give longer polish
        if rng.random() < 0.12:
            lp_budget = min(4.0, max(1.0, remain - 20.0))
        x = lp_polish(x, time.time() + lp_budget, radius_init=float(rng.choice([5e-4, 1e-3, 2e-3])))
        v = max_cos(x)
        if v < inc_val - 1e-12:
            incumbent = x.copy()
            inc_val = v
        # else: also accept tiny sideways moves occasionally to change basin? keep strict
        # early exit if big win
        if time.time() >= SEARCH_END:
            break
    # final long polish of incumbent
    remain = HARD_END - time.time()
    if remain > 3:
        incumbent = lp_polish(incumbent, HARD_END - 1.0, radius_init=1e-3)
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
