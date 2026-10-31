import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_quench(x0, steps, tau0, tau1, lr0, lr1, beta=0.9):
    """Annealed smoothed-max descent on S^3. Returns best encountered."""
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    # initial value
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
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
        vel = beta * vel + (1.0 - beta) * tang
        x = x - lr * vel
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x /= nrm
    # check final
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    gf = float(Gf.max())
    if gf < bv:
        bv = gf
        bx = x.copy()
    return bx, bv


def lp_polish(x, seconds, t_deadline=None):
    """Trust-region LP polish. Minimise max cosine."""
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
        # current max (off diag)
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
    PHASE1_END = t0 + 125.0

    rng = np.random.default_rng(0)

    # load starting points
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

    # also consider random init quenched quickly as alternative start
    n, d = 153, 4
    sigmas = [0.008, 0.015, 0.03, 0.055, 0.09, 0.15]
    trial = 0
    # parameters for quench
    STEPS = 1000
    TAU0, TAU1 = 0.025, 0.0015
    LR0, LR1 = 0.06, 0.004

    while time.time() < PHASE1_END and time.time() < HARD_END - 15:
        if trial % 12 == 11:
            # far random restart
            x_try = rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            sigma = sigmas[trial % len(sigmas)]
            noise = rng.standard_normal((n, d))
            x_try = best_x + sigma * noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        bx, bv = smooth_quench(x_try, STEPS, TAU0, TAU1, LR0, LR1, beta=0.9)

        if bv < best_val - 1e-12:
            # polish promising new basin briefly to get true local value
            # give 1.5s for intermediate polish if time allows
            remain1 = PHASE1_END - time.time()
            if remain1 > 5:
                bx_p = lp_polish(bx, min(1.5, remain1 - 1.0))
                bv_p = max_cos(bx_p)
                if bv_p < bv:
                    bx, bv = bx_p, bv_p
            if bv < best_val:
                best_x, best_val = bx, bv
        elif bv < best_val + 0.0008 and (trial % 5 == 0):
            # occasionally polish near-best to see if it would cross over
            pass

        trial += 1
        # adapt steps slightly if running out of time: reduce steps
        if time.time() > PHASE1_END - 20 and STEPS > 600:
            STEPS = 600

    # final polish with remaining time
    remain = HARD_END - time.time() - 2.0
    if remain > 3:
        # cap to ~40s to stay safe, but use available
        use = min(remain, 40.0)
        best_x = lp_polish(best_x, use)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
