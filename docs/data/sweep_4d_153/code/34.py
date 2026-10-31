import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def sym_generate_X(R, c1, s1, c2, s2):
    n_orb = R.shape[0]
    m = c1.shape[0]
    R0 = R[:, 0:1]
    R1 = R[:, 1:2]
    R2 = R[:, 2:3]
    R3 = R[:, 3:4]
    C1 = c1[None, :]
    S1 = s1[None, :]
    C2 = c2[None, :]
    S2 = s2[None, :]
    B0 = R0 * C1 - R1 * S1
    B1 = R0 * S1 + R1 * C1
    B2 = R2 * C2 - R3 * S2
    B3 = R2 * S2 + R3 * C2
    N = n_orb * m
    X = np.empty((N, 4), dtype=float)
    X[:, 0] = B0.reshape(-1)
    X[:, 1] = B1.reshape(-1)
    X[:, 2] = B2.reshape(-1)
    X[:, 3] = B3.reshape(-1)
    # numerical renorm (should already be unit)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    X /= nrm
    return X

def sym_relax(Rinit, m, p1, p2, iters, k0, k1, lr0, lr1, mu, rng, noise=0.0):
    R = Rinit.copy().astype(float)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
    n_orb = R.shape[0]
    t = np.arange(m)
    c1 = np.cos(2 * np.pi * p1 * t / m)
    s1 = np.sin(2 * np.pi * p1 * t / m)
    c2 = np.cos(2 * np.pi * p2 * t / m)
    s2 = np.sin(2 * np.pi * p2 * t / m)
    V = np.zeros_like(R)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 and k1 > 0 else k0
        lr = lr0 * (lr1 / lr0) ** frac if iters > 1 else lr0
        X = sym_generate_X(R, c1, s1, c2, s2)
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mv = float(G.max())
        # weights
        A = k * (G - mv)
        # clip for exp overflow (A<=0)
        A = np.clip(A, -80.0, 0.0)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        GX = W @ X  # (N,4)
        S = float(W.sum()) + 1e-12
        scale = S / max(1, n_orb)
        GXr = GX.reshape(n_orb, m, 4)
        g0 = (c1[None, :] * GXr[:, :, 0] + s1[None, :] * GXr[:, :, 1]).sum(axis=1)
        g1 = (-s1[None, :] * GXr[:, :, 0] + c1[None, :] * GXr[:, :, 1]).sum(axis=1)
        g2 = (c2[None, :] * GXr[:, :, 2] + s2[None, :] * GXr[:, :, 3]).sum(axis=1)
        g3 = (-s2[None, :] * GXr[:, :, 2] + c2[None, :] * GXr[:, :, 3]).sum(axis=1)
        grad = np.stack([g0, g1, g2, g3], axis=1) / scale
        dot = np.sum(grad * R, axis=1, keepdims=True)
        gt = grad - dot * R
        # clip extreme grads
        # momentum
        if mu > 0:
            V = mu * V + gt
            step = V
        else:
            step = gt
        R = R - lr * step
        if noise > 0 and frac < 0.4:
            amp = noise * (1.0 - frac / 0.4)
            P = rng.standard_normal(R.shape)
            P -= np.sum(P * R, axis=1, keepdims=True) * R
            R = R + amp * P
        nrm = np.linalg.norm(R, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        R /= nrm
    X = sym_generate_X(R, c1, s1, c2, s2)
    return R, X

def exp_relax(x, iters, k0, k1, lr0, lr1, mu, noise_amp, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = np.clip(k * (G - m), -80.0, 0.0)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        x = x - lr * v
        if noise_amp > 0 and frac < 0.5:
            amp = noise_amp * (1.0 - frac / 0.5)
            pert = rng.standard_normal(x.shape)
            pert -= np.sum(pert * x, axis=1, keepdims=True) * x
            x = x + amp * pert
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(C.max())
        sel = C > m - max(25.0 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argsort(-C[sel])[:3000]
            I, J = I[idx], J[idx]
            p = len(I)
            bvals = -(x @ x.T)[I, J]
        else:
            bvals = -C[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=bvals,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def random_valid_pp(m, rng):
    for _ in range(50):
        p1 = int(rng.integers(0, m))
        p2 = int(rng.integers(0, m))
        if p1 == 0 and p2 == 0:
            continue
        if math.gcd(math.gcd(m, p1), p2) != 1:
            continue
        # avoid both zero-mod trivial collapse to tiny orbits for generic pts
        # need lcm(m/gcd(m,p1) or 1 if p1==0 ...) == m for generic; just accept if gcd cond holds
        # reject p1==p2==0 mod m already done; reject degenerate where both multiples
        return p1, p2
    return 1, 2 % m

def solve():
    t0 = time.time()
    rng = np.random.default_rng(17291753)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        best = rng.standard_normal((n, d))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    try:
        best_val = max_cos(best)
    except Exception:
        best_val = 1.0
    incumbent = best.copy()
    inc_val = float(best_val)

    families = [(17, 9), (51, 3), (9, 17), (3, 51)]
    fam_w = [0.42, 0.28, 0.18, 0.12]

    sym_cands = []  # (val, X, (m,p1,p2,R))
    SCREEN_END = t0 + 58.0
    trial = 0
    # quick screening of symmetric basins
    while time.time() < SCREEN_END and trial < 220:
        trial += 1
        r = rng.random()
        cum = 0.0
        fi = 0
        for i, w in enumerate(fam_w):
            cum += w
            if r < cum:
                fi = i
                break
        m, n_orb = families[fi]
        p1, p2 = random_valid_pp(m, rng)
        Rinit = rng.standard_normal((n_orb, d))
        Rinit /= np.linalg.norm(Rinit, axis=1, keepdims=True) + 1e-16
        # random schedule cheap
        k0 = float(rng.choice(np.array([4.0, 8.0, 15.0])))
        k1 = float(rng.choice(np.array([120.0, 250.0, 500.0])))
        lr0 = float(rng.choice(np.array([0.08, 0.15, 0.25])))
        lr1 = float(rng.choice(np.array([0.004, 0.010, 0.02])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        iters = int(rng.choice(np.array([350, 500, 700])))
        try:
            Ropt, Xopt = sym_relax(Rinit, m, p1, p2, iters, k0, k1, lr0, lr1, mu, rng, noise=0.0)
        except Exception:
            continue
        try:
            v = max_cos(Xopt)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        sym_cands.append((v, Xopt.copy(), (m, p1, p2, Ropt.copy())))
        sym_cands = sorted(sym_cands, key=lambda z: z[0])[:8]
        # early keep incumbent if symmetric already good (unlikely)
        if v < inc_val:
            inc_val = v
            incumbent = Xopt.copy()

    sym_cands = sorted(sym_cands, key=lambda z: z[0])[:8]
    # full relaxation of best few symmetric candidates
    FULL_END = t0 + 82.0
    relaxed = []
    for i, (v, X0, info) in enumerate(sym_cands[:4]):
        if time.time() > FULL_END:
            break
        m, p1, p2, R0 = info
        try:
            Ropt, Xopt = sym_relax(R0, m, p1, p2, 1600, 10.0, 1200.0, 0.12, 0.003, 0.85, rng, noise=0.0)
            v2 = max_cos(Xopt)
            relaxed.append((v2, Xopt, (m, p1, p2, Ropt)))
        except Exception:
            relaxed.append((v, X0, info))
    if len(relaxed) == 0 and len(sym_cands) > 0:
        relaxed = [(sym_cands[0][0], sym_cands[0][1], sym_cands[0][2])]
    relaxed = sorted(relaxed, key=lambda z: z[0])

    # symmetry breaking: gradually increase perturbation 0 -> 0.05-0.15
    BREAK_END = t0 + 128.0
    broken_cands = []
    # include relaxed symmetric as seeds
    seeds = [r[1] for r in relaxed[:3]]
    if len(seeds) == 0:
        seeds = [incumbent]
    amps = [0.015, 0.04, 0.07, 0.11, 0.15]
    bi = 0
    while time.time() < BREAK_END and bi < 28:
        bi += 1
        base = seeds[int(rng.integers(len(seeds)))] if len(seeds) > 0 else incumbent
        amp = amps[int(rng.integers(len(amps)))] if rng.random() < 0.85 else float(rng.uniform(0.02, 0.15))
        x = base.copy()
        pert = rng.standard_normal(x.shape)
        # tangential part only to keep near sphere before renorm
        pert -= np.sum(pert * x, axis=1, keepdims=True) * x
        x = x + amp * pert
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        iters = int(rng.choice(np.array([700, 1000, 1400])))
        k0 = float(rng.choice(np.array([8.0, 15.0, 25.0])))
        k1 = float(rng.choice(np.array([300.0, 700.0, 1500.0])))
        lr0 = float(rng.choice(np.array([0.03, 0.06])))
        lr1 = float(rng.choice(np.array([0.0015, 0.004])))
        mu = float(rng.choice(np.array([0.0, 0.85, 0.9])))
        try:
            xo = exp_relax(x, iters, k0, k1, lr0, lr1, mu, 0.0, rng)
        except Exception:
            continue
        try:
            v = max_cos(xo)
        except Exception:
            continue
        broken_cands.append((v, xo))
        broken_cands = sorted(broken_cands, key=lambda z: z[0])[:6]
        if v < inc_val:
            inc_val = v
            incumbent = xo.copy()
            seeds.append(xo.copy())
            seeds = seeds[-5:]

    # final pool: best broken + incumbent + best symmetric-broken + original best
    pool = []
    for v, xo in broken_cands[:3]:
        pool.append((v, xo))
    pool.append((float(max_cos(incumbent)), incumbent.copy()))
    pool.append((float(best_val), best.copy()))
    # dedup sort
    pool = sorted(pool, key=lambda z: z[0])[:4]

    time_left = 172.0 - (time.time() - t0)
    if time_left < 10:
        time_left = 10
    per = time_left / max(1, min(3, len(pool)))
    polished_best = None
    polished_val = 1e9
    for i, (vv, xc) in enumerate(pool[:3]):
        if time.time() - t0 > 171:
            break
        budget = per * (1.5 if i == 0 else 1.0)
        budget = min(budget, 30.0)
        try:
            xp = lp_polish(xc, budget)
        except Exception:
            xp = xc
        try:
            vp = max_cos(xp)
        except Exception:
            continue
        if vp < polished_val:
            polished_val = vp
            polished_best = xp.copy()
    if polished_best is None:
        polished_best = incumbent
        polished_val = float(max_cos(polished_best))
    # return best among polished and original record
    if float(best_val) < polished_val:
        rb = best / np.linalg.norm(best, axis=1, keepdims=True)
        return rb
    polished_best /= np.linalg.norm(polished_best, axis=1, keepdims=True)
    return polished_best
