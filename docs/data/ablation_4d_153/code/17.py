import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_quench(x0, steps, tau0, tau1, lr0, lr1, beta=0.9, noise0=0.0, noise1=0.0, rng=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 and tau0 > 0 and tau1 > 0 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    lno = 0.0
    if noise0 > 0 and noise1 > 0 and noise0 != noise1:
        lno = np.log(noise1 / noise0)
    elif noise0 == 0 or noise1 == 0:
        lno = 0.0
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        if noise0 == 0.0:
            namp = 0.0
        elif noise1 == 0.0:
            namp = noise0 * (1.0 - frac)
        else:
            namp = noise0 * np.exp(lno * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        Gmax = float(G.max())
        if Gmax < bv:
            bv = Gmax
            bx = x.copy()
        tau = max(tau, 1e-12)
        E = np.exp((G - Gmax) / tau)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-30)
        avg = (E @ x) / rs
        dot = np.sum(avg * x, axis=1, keepdims=True)
        tang = avg - dot * x
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        if namp > 0 and rng is not None:
            nz = rng.standard_normal((n, d))
            # project to tangent
            nd = np.sum(nz * x, axis=1, keepdims=True)
            nz = nz - nd * x
            x = x + namp * nz
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x /= nrm
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


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    PHASE1_END = t0 + 122.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4

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
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    vals = [max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best_x = cand[bi].copy()
    best_val = float(vals[bi])
    # pool of diverse good solutions (keep up to 3)
    pool = [(best_val, best_x.copy())]
    # also add second candidate if distinct
    for a, v in zip(cand, vals):
        if v > best_val + 1e-12 and len(pool) < 3:
            pool.append((v, a.copy()))
    pool.sort(key=lambda t: t[0])

    sigmas = [0.008, 0.015, 0.03, 0.055, 0.09, 0.15]
    trial = 0
    while time.time() < PHASE1_END and time.time() < HARD_END - 20:
        trial += 1
        # choose base
        if trial % 13 == 0:
            base = None  # fresh random
        elif len(pool) > 1 and trial % 6 == 0:
            # second best for diversity
            base = pool[1][1]
        else:
            base = pool[0][1]
            # keep pool[0] synced with best
            if best_val < pool[0][0] - 1e-15:
                pass

        if base is None:
            x_try = rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            r = rng.random()
            if r < 0.32:
                sigma = sigmas[rng.integers(len(sigmas))]
                # occasionally mix two sigmas (bimodal: rattler-like)
                x_try = base + sigma * rng.standard_normal((n, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif r < 0.54:
                # rattler-targeted kick: large move for low-contact points
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                deg = (G > m - 0.002).sum(axis=1)
                sig_small = float(rng.uniform(0.005, 0.015))
                sig_big = float(rng.uniform(0.15, 0.4))
                sig = np.where(deg <= 1, sig_big, sig_small)[:, None]
                x_try = base + sig * rng.standard_normal((n, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            elif r < 0.76:
                # void teleport: move one endpoint of a max pair to largest void
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                ii = flat // n
                jj = flat % n
                victim = ii if rng.random() < 0.5 else jj
                # sample voids
                S = rng.standard_normal((3000, d))
                S /= np.linalg.norm(S, axis=1, keepdims=True)
                mask = np.ones(n, dtype=bool)
                mask[victim] = False
                D = S @ base[mask].T
                worst = D.max(axis=1)
                bidx = int(np.argmin(worst))
                void_pt = S[bidx]
                x_try = base + 0.008 * rng.standard_normal((n, d))
                x_try[victim] = void_pt + 0.02 * rng.standard_normal(d)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            else:
                # cap reshuffle
                u = rng.standard_normal(d)
                u /= np.linalg.norm(u)
                dots = base @ u
                k = int(rng.integers(4, 13))
                idx = np.argsort(dots)[-k:]
                x_try = base + 0.008 * rng.standard_normal((n, d))
                for vv in idx:
                    x_try[vv] = u + 0.35 * rng.standard_normal(d)
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        STEPS = int(rng.integers(650, 1050))
        # shrink steps near end of phase to fit more trials
        if time.time() > PHASE1_END - 20 and STEPS > 700:
            STEPS = 700
        TAU0 = float(rng.uniform(0.015, 0.035))
        TAU1 = float(rng.uniform(0.0008, 0.003))
        LR0 = float(rng.uniform(0.04, 0.09))
        LR1 = float(rng.uniform(0.002, 0.007))
        beta = float(rng.uniform(0.85, 0.94))
        if rng.random() < 0.45:
            nz0 = 0.0
            nz1 = 0.0
        else:
            nz0 = float(rng.uniform(0.0005, 0.0025))
            nz1 = nz0 * float(rng.uniform(0.0, 0.2))

        bx, bv = smooth_quench(x_try, STEPS, TAU0, TAU1, LR0, LR1, beta=beta,
                               noise0=nz0, noise1=nz1, rng=rng)

        if bv < best_val - 1e-12:
            remain1 = PHASE1_END - time.time()
            # quick polish to get true basin value
            if remain1 > 4 and bv < best_val + 0.002:
                use = min(1.2, max(0.4, remain1 - 1.0))
                # only polish if promising enough
                bx_p = lp_polish(bx, use, t_deadline=min(HARD_END - 15, PHASE1_END + 10))
                bv_p = max_cos(bx_p)
                if bv_p < bv:
                    bx, bv = bx_p, bv_p
            if bv < best_val:
                best_x = bx.copy()
                best_val = float(bv)
                pool.append((best_val, best_x.copy()))
                pool.sort(key=lambda t: t[0])
                pool = pool[:3]
        elif bv < pool[-1][0] - 5e-6 and len(pool) < 3:
            pool.append((bv, bx.copy()))
            pool.sort(key=lambda t: t[0])
            pool = pool[:3]
        elif bv < best_val + 0.001 and rng.random() < 0.06:
            # occasional diversity insert
            # check distinctness
            dup = False
            for _, px in pool:
                if float(np.abs(px * bx).sum() / n) > 0.9999:
                    pass
            if len(pool) < 3 and not dup:
                pool.append((bv, bx.copy()))
                pool.sort(key=lambda t: t[0])

    # final polish with remaining time
    remain = HARD_END - time.time() - 1.5
    if remain > 2:
        use = min(remain, 45.0)
        best_x = lp_polish(best_x, use, t_deadline=HARD_END - 1.0)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
