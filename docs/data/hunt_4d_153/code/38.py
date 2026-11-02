import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _expand(rho, theta, phi, marr):
    No = len(rho)
    M = 17
    n = No * M
    X = np.empty((n, 4))
    idx = 0
    twopi = 2 * np.pi
    for j in range(No):
        r = np.cos(rho[j])
        s = np.sin(rho[j])
        # clamp
        if r < 1e-6: r = 1e-6
        if s < 1e-6: s = 1e-6
        # renorm to unit (r,s) (in case)
        nr = np.sqrt(r*r + s*s)
        r /= nr; s /= nr
        m = int(marr[j])
        for k in range(M):
            a = theta[j] + twopi * k / M
            b = phi[j] + twopi * m * k / M
            X[idx, 0] = r * np.cos(a)
            X[idx, 1] = r * np.sin(a)
            X[idx, 2] = s * np.cos(b)
            X[idx, 3] = s * np.sin(b)
            idx += 1
    # normalize rows
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, budget, t0=None, hard=172.0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        if t0 is not None and time.time() - t0 > hard - 2:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 2e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        csel = c_all[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2500:
            idx = np.argpartition(-csel, 2500)[:2500]
            I = I[idx]; J = J[idx]; csel = csel[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                              J[:, None] * d + np.arange(d),
                              np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth_free(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu0, Ju0 = iu
    lb0 = np.log(lr0); lb1 = np.log(lr1)
    gb0 = np.log(beta0); gb1 = np.log(beta1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(gb0 + (gb1 - gb0) * f)
        lr = np.exp(lb0 + (lb1 - lb0) * f)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gm = g.max()
        e = np.exp(np.clip(beta * (g - gm), -30, 0.0))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _sym_opt(marr, rho0, th0, ph0, steps, beta0, beta1, lr0, lr1):
    No = len(marr)
    M = 17
    n = No * M
    orbit_id = np.repeat(np.arange(No), M)
    kvals = np.tile(np.arange(M), No)
    m_per_pt = marr[orbit_id].astype(float)
    rho = rho0.copy(); theta = th0.copy(); phi = ph0.copy()
    mr = np.zeros(No); vr = np.zeros(No)
    mt = np.zeros(No); vt = np.zeros(No)
    mp = np.zeros(No); vp = np.zeros(No)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    gb0 = np.log(beta0); gb1 = np.log(beta1)
    lb0 = np.log(lr0); lb1 = np.log(lr1)
    twopi = 2 * np.pi
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = np.exp(gb0 + (gb1 - gb0) * f)
        lr = np.exp(lb0 + (lb1 - lb0) * f)
        r_orb = np.cos(rho); s_orb = np.sin(rho)
        # keep away from poles
        r_orb = np.clip(r_orb, 1e-4, 1 - 1e-4)
        s_orb = np.clip(s_orb, 1e-4, 1 - 1e-4)
        # renorm (r,s) per orbit
        nn = np.sqrt(r_orb**2 + s_orb**2)
        r_orb /= nn; s_orb /= nn
        rp = r_orb[orbit_id]; sp = s_orb[orbit_id]
        thp = theta[orbit_id] + twopi * kvals / M
        php = phi[orbit_id] + twopi * m_per_pt * kvals / M
        Cth = np.cos(thp[:, None] - thp[None, :])
        Sth = np.sin(thp[:, None] - thp[None, :])
        Cph = np.cos(php[:, None] - php[None, :])
        Sph = np.sin(php[:, None] - php[None, :])
        RR = rp[:, None] * rp[None, :]
        SS = sp[:, None] * sp[None, :]
        G = RR * Cth + SS * Cph
        g = G[Iu0, Ju0]
        gm = g.max()
        e = np.exp(np.clip(beta * (g - gm), -30, 0.0))
        Z = e.sum()
        w = e / Z
        Wsym = np.zeros((n, n))
        Wsym[Iu0, Ju0] = w
        Wsym = Wsym + Wsym.T
        # grads
        A = Wsym * RR * Sth
        rsA = A.sum(axis=1)
        B = Wsym * SS * Sph
        rsB = B.sum(axis=1)
        E = -(sp[:, None] * rp[None, :]) * Cth + (rp[:, None] * sp[None, :]) * Cph
        WE = Wsym * E
        rsE = WE.sum(axis=1)
        gr = np.zeros(No); gt = np.zeros(No); gp = np.zeros(No)
        for j in range(No):
            mask = (orbit_id == j)
            gt[j] = -rsA[mask].sum()
            gp[j] = -rsB[mask].sum()
            gr[j] = rsE[mask].sum()
        # Adam updates
        mr = b1 * mr + (1 - b1) * gr
        vr = b2 * vr + (1 - b2) * (gr * gr)
        mt = b1 * mt + (1 - b1) * gt
        vt = b2 * vt + (1 - b2) * (gt * gt)
        mp = b1 * mp + (1 - b1) * gp
        vp = b2 * vp + (1 - b2) * (gp * gp)
        c1 = 1 - b1 ** (s + 1); c2 = 1 - b2 ** (s + 1)
        sr = (mr / c1) / (np.sqrt(vr / c2) + eps)
        st = (mt / c1) / (np.sqrt(vt / c2) + eps)
        spp = (mp / c1) / (np.sqrt(vp / c2) + eps)
        rho = rho - lr * sr
        theta = theta - lr * st
        phi = phi - lr * spp
        # keep rho in bounds, wrap angles
        rho = np.clip(rho, 0.08, 1.4908)
        # wrap to [-pi,pi] to avoid drift
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    # final expand
    X = _expand(rho, theta, phi, marr)
    return X, _max_cos(X), (rho, theta, phi)

def solve():
    t0 = time.time()
    HARD = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=float).reshape(153, 4)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n = 153
    iu = np.triu_indices(n, 1)

    # ---- Phase 1: symmetric torus search ----
    sym_best_val = 1e9
    sym_best_X = None
    # time budget for sym phase
    SYM_BUDGET = 62.0
    n_restart = 0
    # candidate twist sets: avoid +-1 (1,16) and 0
    allowed = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    while time.time() - t0 < SYM_BUDGET and n_restart < 12:
        n_restart += 1
        # pattern: random allowed, ensure not all equal
        marr = rng.choice(allowed, size=9, replace=True)
        # occasionally force diverse set
        if n_restart == 1:
            marr = np.array([2,3,4,5,6,7,8,9,10])
        elif n_restart == 2:
            marr = np.array([2,4,6,8,10,12,14,3,5])
        elif n_restart == 3:
            marr = np.array([5,5,5,5,5,5,5,5,5], dtype=float)
            # 5 inverse is 7, pairing good
        # init params
        r2 = rng.uniform(0.12, 0.88, size=9)
        rho0 = np.arccos(np.sqrt(r2))
        th0 = rng.uniform(-np.pi, np.pi, size=9)
        ph0 = rng.uniform(-np.pi, np.pi, size=9)
        remain_sym = SYM_BUDGET - (time.time() - t0)
        if remain_sym < 4:
            break
        # steps adapt
        if remain_sym > 40:
            steps = 1100
        elif remain_sym > 20:
            steps = 800
        else:
            steps = 500
        # beta schedule: start low, end high
        Xc, vc, _ = _sym_opt(marr.astype(float), rho0, th0, ph0, steps, 25.0, 1600.0, 0.05, 0.004)
        if vc < sym_best_val:
            sym_best_val = vc
            sym_best_X = Xc.copy()
        # early exit if already excellent
        if sym_best_val < 0.846 and (time.time() - t0) > 25:
            break

    # ---- Phase 2: release symmetry + polish ----
    cands = []
    if sym_best_X is not None:
        cands.append(sym_best_X)
        # a couple of perturbed copies will be made on the fly
    cands.append(x_best.copy())

    # evaluate cands with quick smooth+polish, keep best
    # allocate remaining time
    for ci, xc in enumerate(list(cands)):
        if time.time() - t0 > HARD - 15:
            break
        # short free smooth
        remain = HARD - 12 - (time.time() - t0)
        if remain < 4:
            break
        steps = 700 if remain > 40 else 400
        y = _smooth_free(xc, steps, 150.0, 4000.0, 0.012, 0.0008, iu)
        v = _max_cos(y)
        # polish briefly
        bud = 2.0 if remain > 35 else 1.0
        if time.time() - t0 + bud > HARD - 10:
            bud = max(0.4, HARD - 10 - (time.time() - t0))
        yp, vp = _lp_polish(y, bud, t0, HARD)
        y, v = yp, vp
        if v < best_val - 1e-12:
            best_val = v
            x_best = y.copy()

    # extra symmetry-breaking kicks from sym best
    if sym_best_X is not None:
        nkick = 0
        while time.time() - t0 < HARD - 12 and nkick < 6:
            nkick += 1
            remain = HARD - 12 - (time.time() - t0)
            if remain < 5:
                break
            y = sym_best_X + rng.normal(0, 0.02 + 0.01 * nkick, size=sym_best_X.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 600 if remain > 30 else 350
            y = _smooth_free(y, steps, 120.0, 3500.0, 0.012, 0.0008, iu)
            v = _max_cos(y)
            if v < best_val + 0.004:
                bud = 1.5 if remain > 25 else 0.8
                y, v = _lp_polish(y, bud, t0, HARD)
            if v < best_val - 1e-12:
                best_val = v
                x_best = y.copy()
                # if improved a lot, keep kicking from new best
                if best_val < sym_best_val - 0.002:
                    pass

    # final polish on incumbent with leftover
    left = HARD - (time.time() - t0)
    if left > 1.5:
        x_best, best_val = _lp_polish(x_best, min(left - 0.5, 7.0), t0, HARD)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
