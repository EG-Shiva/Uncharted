import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def quench_lse(x0, steps, tau0, tau1, lr0, lr1, beta, rng, noise0=0.0):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gmax = float(G.max())
        if Gmax < bv:
            bv = Gmax
            bx = x.copy()
        E = np.exp((G - Gmax) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-30)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        # optional decaying Langevin noise
        if noise0 > 0:
            ns = noise0 * (1.0 - frac)
            if ns > 1e-12:
                nz = rng.standard_normal((n, d))
                nz -= np.sum(nz * x, axis=1, keepdims=True) * x
                tang = tang + ns * nz
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def quench_pow(x0, steps, p0, p1, lr0, lr1, beta, rng):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    lp = np.log(p1 / p0) if p0 != p1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        p = p0 * np.exp(lp * frac)
        lr = lr0 * np.exp(llr * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gmax = float(G.max())
        if Gmax < bv:
            bv = Gmax
            bx = x.copy()
        # weights ~ ((1+G)/2)^p
        B = np.clip((1.0 + G) * 0.5, 0.0, 1.0)
        # avoid underflow: scale by max
        E = np.power(B, p)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-30)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        np.fill_diagonal(Gfull, -2.0)
        m = float(Gfull.max())
        c_full = (x @ x.T)[iu]
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, best_x = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x


def hole_centers(x, k, moseeds, rng):
    # find k directions minimizing max dot to x (largest holes)
    n, d = x.shape
    cand = rng.standard_normal((moseeds, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T  # Moseeds x n
    md = dots.max(axis=1)
    idx = np.argpartition(md, min(k, moseeds - 1))[:k]
    # sort by quality
    idx = idx[np.argsort(md[idx])]
    return cand[idx]


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    PHASE1_END = t0 + 122.0
    rng = np.random.default_rng(7)

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (153, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if len(cand) == 0:
        a = rng.standard_normal((153, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    vals = [max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best_x = cand[bi].copy()
    best_val = float(vals[bi])
    n, d = 153, 4

    trial = 0
    # precompute contact info of best periodically
    while time.time() < PHASE1_END and time.time() < HARD_END - 18:
        trial += 1
        r = rng.random()
        # --- mutation to get x_try ---
        if r < 0.02:
            # far random restart
            x_try = rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.38:
            # global Gaussian kick, log-uniform sigma
            ls = rng.uniform(np.log(0.004), np.log(0.22))
            sigma = float(np.exp(ls))
            x_try = best_x + sigma * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.58:
            # subset heavy kick
            k = int(rng.integers(3, 24))
            idx = rng.choice(n, size=k, replace=False)
            sigma_big = float(np.exp(rng.uniform(np.log(0.08), np.log(0.5))))
            sigma_small = float(np.exp(rng.uniform(np.log(0.001), np.log(0.01))))
            noise = sigma_small * rng.standard_normal((n, d))
            noise[idx] = sigma_big * rng.standard_normal((k, d))
            x_try = best_x + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.75:
            # contact-targeted shake: perturb points involved in near-max pairs
            ppm = per_point_max(best_x)
            # threshold for involved
            thr = best_val - float(rng.uniform(0.002, 0.02))
            involved = np.where(ppm > thr)[0]
            if len(involved) < 2:
                involved = np.argsort(-ppm)[:10]
            sigma_in = float(np.exp(rng.uniform(np.log(0.03), np.log(0.3))))
            sigma_out = 0.002
            noise = sigma_out * rng.standard_normal((n, d))
            noise[involved] += sigma_in * rng.standard_normal((len(involved), d))
            x_try = best_x + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.88:
            # rattler-to-hole relocation
            ppm = per_point_max(best_x)
            # rattlers: points well below max
            gap = float(rng.uniform(0.005, 0.03))
            ratt = np.where(ppm < best_val - gap)[0]
            x_try = best_x + 0.005 * rng.standard_normal((n, d))
            if len(ratt) > 0:
                nh = int(rng.integers(1, min(4, len(ratt) + 1)))
                pick = rng.choice(ratt, size=nh, replace=False)
                hc = hole_centers(x_try, nh, 3000, rng)
                # add small jitter to hole centers
                hc = hc + 0.01 * rng.standard_normal(hc.shape)
                hc /= np.linalg.norm(hc, axis=1, keepdims=True)
                x_try[pick] = hc
            else:
                # no clear rattlers: relocate worst-hole single point (min per-point max)
                j = int(np.argmin(ppm))
                hc = hole_centers(x_try, 1, 3000, rng)[0]
                x_try[j] = hc
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # moderate global + subset mix
            sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(0.08))))
            x_try = best_x + sigma * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        # --- choose quench hyperparams ---
        steps = int(rng.choice([500, 650, 800, 1000, 1200]))
        # shrink steps if late
        if time.time() > PHASE1_END - 25 and steps > 700:
            steps = 700
        beta = float(rng.uniform(0.80, 0.96))
        use_lse = (rng.random() < 0.65)
        if use_lse:
            tau0 = float(np.exp(rng.uniform(np.log(0.008), np.log(0.05))))
            tau1 = float(np.exp(rng.uniform(np.log(0.0007), np.log(0.004))))
            if tau1 >= tau0:
                tau1 = tau0 * 0.1
            lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.12))))
            lr1 = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.012))))
            nz = 0.0
            if rng.random() < 0.25:
                nz = float(np.exp(rng.uniform(np.log(0.0005), np.log(0.005))))
            bx, bv = quench_lse(x_try, steps, tau0, tau1, lr0, lr1, beta, rng, noise0=nz)
        else:
            p0 = float(rng.uniform(12, 40))
            p1 = float(rng.uniform(80, 300))
            lr0 = float(np.exp(rng.uniform(np.log(0.02), np.log(0.10))))
            lr1 = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.01))))
            bx, bv = quench_pow(x_try, steps, p0, p1, lr0, lr1, beta, rng)

        if bv < best_val - 1e-12:
            remain1 = PHASE1_END - time.time()
            if remain1 > 6:
                bx_p = lp_polish(bx, min(2.0, remain1 - 1.0))
                bv_p = max_cos(bx_p)
                if bv_p < bv:
                    bx, bv = bx_p, bv_p
            if bv < best_val:
                best_x, best_val = bx, bv

    remain = HARD_END - time.time() - 2.0
    if remain > 3:
        use = min(remain, 45.0)
        best_x = lp_polish(best_x, use)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
