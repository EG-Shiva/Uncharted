import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _anneal(x, steps, tau0, tau1, lr0, lr1, seed_off=0):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _void_against(fixed, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ fixed.T
    best = s.max(axis=1)
    bi = int(best.argmin())
    b = cand[bi].copy()
    bs = float(best[bi])
    step = 0.06
    for _ in range(120):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ fixed.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.06
        else:
            step *= 0.95
        if step < 1e-5:
            break
    return b

def _greedy_refill(x_fixed_list, k, rng):
    # x_fixed_list: (nf,4) array, returns (nf+k,4)
    cur = x_fixed_list.copy()
    for _ in range(k):
        b = _void_against(cur, rng, n_cand=6000)
        cur = np.vstack([cur, b.reshape(1, 4)])
    return cur

def _personal(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _contact_count(x, tol=2e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    return (G >= m - tol).sum(axis=1), m

def _fiber_init(rng, nfib, per_fib):
    # Hopf fibers: base points on S2 via Fibonacci, fiber circles
    n = nfib * per_fib
    # Fibonacci sphere for base
    ga = np.pi * (3.0 - np.sqrt(5.0))
    th = rng.random() * 2 * np.pi
    base = []
    for i in range(nfib):
        z = 1 - 2 * (i + 0.5) / nfib
        r = np.sqrt(max(0.0, 1 - z * z))
        ph = th + ga * i
        # Hopf base (theta, phi1-phi2): map S2 to theta/eta
        # use standard Hopf: theta in [0,pi/2], eta in [0,2pi), xi in [0,2pi)
        # sample base uniformly: theta = acos-ish, eta uniform
        # derive from Fibonacci point: use z->theta, ph->eta
        theta = np.arccos(np.clip(z, -1, 1)) / 2.0  # 0..pi/2
        eta = ph
        base.append((theta, eta))
    pts = []
    off = rng.random() * 2 * np.pi
    for (theta, eta) in base:
        c, s = np.cos(theta), np.sin(theta)
        for j in range(per_fib):
            xi = off + 2 * np.pi * j / per_fib + 0.1 * rng.standard_normal()
            # S3 coords: (c*cos((eta+xi)/2)?) use standard param
            # x1+i x2 = c*e^{i(xi+eta)/2?} simplify: use two angles
            a1 = (eta + xi) * 0.5
            a2 = (xi - eta) * 0.5
            pts.append([c * np.cos(a1), c * np.sin(a1), s * np.cos(a2), s * np.sin(a2)])
    P = np.array(pts, dtype=float)
    if P.shape[0] != n:
        # fallback pad
        extra = rng.standard_normal((n - P.shape[0], 4))
        P = np.vstack([P, extra]) if P.shape[0] < n else P[:n]
    # random SO(4) rotation
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    P = P @ Q.T
    # small jitter to break exact symmetry
    P = P + 0.01 * rng.standard_normal(P.shape)
    return _norm(P)

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        vv = _mc(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = _norm(np.asarray(x0, dtype=float))
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        x0 = flat[:need].reshape(272, 4)
        x0 = _norm(x0)
    x_best = x0.copy()
    best_val = _mc(x_best)
    cur = x_best.copy()

    fiber_cfgs = [(34, 8), (68, 4), (17, 16), (136, 2), (43, 4), (22, 8)]
    # pad configs to 272 where needed
    trial = 0
    # hopping budget ~ until deadline-35
    while time.time() < t0 + 128 and time.time() < deadline - 40:
        trial += 1
        mode = trial % 8
        y = None
        tau0, tau1, lr0, lr1, steps = 0.02, 0.002, 0.01, 0.001, 500
        if mode == 0:
            # worst-block 24 refill
            pw, _ = _personal(cur)
            order = np.argsort(-pw)
            kill = order[:24]
            mask = np.ones(272, dtype=bool); mask[kill] = False
            fixed = cur[mask]
            ref = _greedy_refill(fixed, 24, rng)
            y = np.empty_like(cur); y[mask] = fixed; y[~mask] = ref[-24:]
            # shuffle to avoid ordering bias
            tau0, tau1, lr0, lr1, steps = 0.018, 0.0012, 0.010, 0.0008, 550
        elif mode == 1:
            # random cap 32 refill (coordinated region move)
            u = rng.standard_normal((4,)); u /= np.linalg.norm(u)
            proj = cur @ u
            kill = np.argsort(-proj)[:32]
            mask = np.ones(272, dtype=bool); mask[kill] = False
            fixed = cur[mask]
            ref = _greedy_refill(fixed, 32, rng)
            y = np.empty_like(cur); y[mask] = fixed; y[~mask] = ref[-32:]
            tau0, tau1, lr0, lr1, steps = 0.022, 0.0018, 0.011, 0.0009, 550
        elif mode == 2:
            cfg = fiber_cfgs[(trial // 8) % len(fiber_cfgs)]
            nf, pf = cfg
            tot = nf * pf
            if tot == 272:
                y = _fiber_init(rng, nf, pf)
            elif tot < 272:
                part = _fiber_init(rng, nf, pf)
                need = 272 - tot
                extra = rng.standard_normal((need, 4)); extra = _norm(extra)
                # greedy add
                cur2 = part
                for _ in range(need):
                    b = _void_against(cur2, rng, n_cand=4000)
                    cur2 = np.vstack([cur2, b.reshape(1, 4)])
                y = cur2
            else:
                y = _fiber_init(rng, nf, pf)[:272]
                y = _norm(y)
            tau0, tau1, lr0, lr1, steps = 0.03, 0.002, 0.012, 0.001, 650
        elif mode == 3:
            # rattlers + worst + random (eliminate wasted points)
            cc, mcur = _contact_count(cur, tol=5e-7)
            ratt = np.where(cc == 0)[0]
            pw, _ = _personal(cur)
            order = np.argsort(-pw)
            killset = set(ratt.tolist())
            for o in order:
                if len(killset) >= 20:
                    break
                killset.add(int(o))
            while len(killset) < 20:
                killset.add(int(rng.integers(0, 272)))
            kill = np.array(sorted(killset), dtype=int)
            mask = np.ones(272, dtype=bool); mask[kill] = False
            fixed = cur[mask]
            ref = _greedy_refill(fixed, len(kill), rng)
            y = np.empty_like(cur); y[mask] = fixed; y[~mask] = ref[-len(kill):]
            tau0, tau1, lr0, lr1, steps = 0.014, 0.001, 0.009, 0.0007, 550
        elif mode == 4:
            # global large shake (basin jump) no kill
            y = cur + 0.22 * rng.standard_normal((272, 4))
            y = _norm(y)
            tau0, tau1, lr0, lr1, steps = 0.025, 0.0015, 0.011, 0.0008, 500
        elif mode == 5:
            # large 48 kill
            pw, _ = _personal(cur)
            order = np.argsort(-pw)
            k = 48
            # mix worst 32 + random 16
            kill = set(order[:32].tolist())
            while len(kill) < k:
                kill.add(int(rng.integers(0, 272)))
            kill = np.array(sorted(kill), dtype=int)
            mask = np.ones(272, dtype=bool); mask[kill] = False
            fixed = cur[mask]
            ref = _greedy_refill(fixed, len(kill), rng)
            y = np.empty_like(cur); y[mask] = fixed; y[~mask] = ref[-len(kill):]
            tau0, tau1, lr0, lr1, steps = 0.028, 0.0022, 0.012, 0.001, 600
        elif mode == 6:
            cfg = fiber_cfgs[(trial // 8 + 3) % len(fiber_cfgs)]
            nf, pf = cfg
            if nf * pf == 272:
                y = _fiber_init(rng, nf, pf)
            else:
                y = rng.standard_normal((272, 4)); y = _norm(y)
            tau0, tau1, lr0, lr1, steps = 0.035, 0.003, 0.013, 0.0012, 600
        else:
            # small cap reshuffle + low-temp focus
            u = rng.standard_normal((4,)); u /= np.linalg.norm(u)
            proj = cur @ u
            kill = np.argsort(-proj)[:14]
            mask = np.ones(272, dtype=bool); mask[kill] = False
            fixed = cur[mask]
            ref = _greedy_refill(fixed, len(kill), rng)
            y = np.empty_like(cur); y[mask] = fixed; y[~mask] = ref[-len(kill):]
            # add small noise to all
            y = y + 0.015 * rng.standard_normal(y.shape)
            y = _norm(y)
            tau0, tau1, lr0, lr1, steps = 0.008, 0.0006, 0.005, 0.0004, 450
        # anneal
        y = _anneal(y, steps, tau0, tau1, lr0, lr1)
        val = _mc(y)
        if val < best_val:
            best_val = val
            x_best = y.copy()
            cur = y.copy()
        else:
            # move cur sometimes to diversify (accept worse if not too bad)
            curval = _mc(cur)
            if val < curval + 0.004 and rng.random() < 0.25:
                cur = y.copy()
            elif rng.random() < 0.05:
                cur = x_best.copy() + 0.05 * rng.standard_normal((272, 4))
                cur = _norm(cur)
        # time guard inside refill loops already; check deadline
        if time.time() > deadline - 40:
            break
    # short refinement of best at low temp
    for _ in range(2):
        if time.time() > deadline - 38:
            break
        y = _anneal(x_best.copy(), 350, 0.006, 0.0006, 0.004, 0.0004)
        vv = _mc(y)
        if vv < best_val:
            best_val = vv
            x_best = y.copy()
    remain = deadline - time.time() - 1.5
    if remain > 6:
        xp = _lp_polish(x_best, min(remain, 34.0), deadline - 1.0)
        vp = _mc(xp)
        if vp <= best_val:
            x_best = xp
            best_val = vp
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best = _norm(x_best)
    return x_best
