import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget, t0=None, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= budget:
            break
        if deadline is not None and time.time() >= deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
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
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _full_adam(x0, steps, beta0, beta1, lr0, lr1, iu, deadline=None):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu, Ju = iu
    if steps <= 1:
        lb = np.log(beta1 / beta0); lr_b = np.log(lr1 / lr0)
    else:
        lb = np.log(beta1 / beta0) / (steps - 1)
        lr_b = np.log(lr1 / lr0) / (steps - 1)
    for s in range(steps):
        if deadline is not None and (s % 50 == 0) and time.time() >= deadline:
            break
        beta = beta0 * np.exp(lb * s)
        lr = lr0 * np.exp(lr_b * s)
        G = x @ x.T
        g = G[Iu, Ju]
        gm = g.max()
        e = np.exp(beta * (g - gm))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu, Ju] = w
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

def _build_cyclic(u, th, ph, m_vec, N):
    n_orb = len(m_vec)
    n = n_orb * N
    X = np.empty((n, 4))
    s = 1.0 / (1.0 + np.exp(-np.clip(u, -10, 10)))
    s = np.clip(s, 1e-6, 1 - 1e-6)
    r = np.sqrt(s); q = np.sqrt(1 - s)
    alpha = 2 * np.pi * np.arange(N) / N
    for j in range(n_orb):
        a = th[j] + alpha
        b = ph[j] + m_vec[j] * alpha
        ca = np.cos(a); sa = np.sin(a)
        cb = np.cos(b); sb = np.sin(b)
        sl = slice(j * N, (j + 1) * N)
        X[sl, 0] = r[j] * ca
        X[sl, 1] = r[j] * sa
        X[sl, 2] = q[j] * cb
        X[sl, 3] = q[j] * sb
    return X

def _sym_run(N, m_vec, rng, steps, beta0, beta1, lr):
    n_orb = len(m_vec)
    u = rng.normal(0, 0.5, size=n_orb)
    th = rng.uniform(0, 2 * np.pi, size=n_orb)
    ph = rng.uniform(0, 2 * np.pi, size=n_orb)
    alpha = 2 * np.pi * np.arange(N) / N
    P = 3 * n_orb
    m1 = np.zeros(P); v1 = np.zeros(P)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb = np.log(beta1 / beta0) / max(steps - 1, 1)
    best_val = 1e9
    best_tup = (u.copy(), th.copy(), ph.copy())
    n = n_orb * N
    iu = np.triu_indices(n, 1)
    Iu, Ju = iu
    for s_ in range(steps):
        beta = beta0 * np.exp(lb * s_)
        sv = 1.0 / (1.0 + np.exp(-np.clip(u, -10, 10)))
        sv = np.clip(sv, 1e-6, 1 - 1e-6)
        r = np.sqrt(sv); q = np.sqrt(1 - sv)
        ds = sv * (1 - sv)
        dr = ds / (2 * r + 1e-12)
        dq = -ds / (2 * q + 1e-12)
        # build X
        X = np.empty((n, 4))
        ca_list = []; sa_list = []; cb_list = []; sb_list = []
        for j in range(n_orb):
            a = th[j] + alpha
            b = ph[j] + m_vec[j] * alpha
            ca = np.cos(a); sa = np.sin(a)
            cb = np.cos(b); sb = np.sin(b)
            ca_list.append(ca); sa_list.append(sa); cb_list.append(cb); sb_list.append(sb)
            sl = slice(j * N, (j + 1) * N)
            X[sl, 0] = r[j] * ca
            X[sl, 1] = r[j] * sa
            X[sl, 2] = q[j] * cb
            X[sl, 3] = q[j] * sb
        G = X @ X.T
        g = G[Iu, Ju]
        gm = g.max()
        if gm < best_val:
            best_val = float(gm)
            best_tup = (u.copy(), th.copy(), ph.copy())
        e = np.exp(beta * (g - gm))
        w = e / e.sum()
        W = np.zeros((n, n))
        W[Iu, Ju] = w
        W = W + W.T
        gradX = W @ X
        gu = np.empty(n_orb); gt = np.empty(n_orb); gp = np.empty(n_orb)
        for j in range(n_orb):
            sl = slice(j * N, (j + 1) * N)
            Gj = gradX[sl]
            ca = ca_list[j]; sa = sa_list[j]; cb = cb_list[j]; sb = sb_list[j]
            g1 = Gj[:, 0]; g2 = Gj[:, 1]; g3 = Gj[:, 2]; g4 = Gj[:, 3]
            gu[j] = np.sum((g1 * ca + g2 * sa) * dr[j] + (g3 * cb + g4 * sb) * dq[j])
            gt[j] = r[j] * np.sum(-g1 * sa + g2 * ca)
            gp[j] = q[j] * np.sum(-g3 * sb + g4 * cb)
        gpar = np.concatenate([gu, gt, gp])
        m1 = b1 * m1 + (1 - b1) * gpar
        v1 = b2 * v1 + (1 - b2) * (gpar * gpar)
        c1 = 1 - b1 ** (s_ + 1); c2 = 1 - b2 ** (s_ + 1)
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        upd = lr * step
        u -= upd[:n_orb]
        th -= upd[n_orb:2 * n_orb]
        ph -= upd[2 * n_orb:]
        th = np.mod(th, 2 * np.pi)
        ph = np.mod(ph, 2 * np.pi)
        u = np.clip(u, -6, 6)
    ub, thb, phb = best_tup
    Xb = _build_cyclic(ub, thb, phb, m_vec, N)
    return Xb, _max_cos(Xb)

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(153, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (153, 4):
        xb = xb.reshape(153, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = np.triu_indices(n, 1)

    # Phase A: symmetric torus search (prioritise N=9 feasible, then N=3, then N=17)
    sym_cands = []
    tA_end = t0 + 32.0
    # patterns: list of (N, n_orb, coprime set)
    configs = []
    # N=9, 17 orbits
    for _ in range(8):
        mv = rng.choice(np.array([1, 2, 4, 5, 7, 8]), size=17)
        configs.append((9, mv))
    for _ in range(3):
        mv = rng.choice(np.array([1, 2]), size=51)
        configs.append((3, mv))
    for _ in range(3):
        mv = rng.choice(np.array([1, 2, 3, 4, 5, 6, 7, 8, 11, 13]), size=9)
        configs.append((17, mv))
    for (N, mv) in configs:
        if time.time() >= min(tA_end, DEAD - 120):
            break
        try:
            Xs, vs = _sym_run(N, np.asarray(mv), rng, steps=700, beta0=40.0, beta1=500.0, lr=0.03)
        except Exception:
            continue
        sym_cands.append((vs, Xs))
        # keep global best if symmetric already beats (unlikely for N=17)
        if vs < best_val:
            best_val = vs
            x_best = Xs.copy()
    sym_cands.sort(key=lambda t: t[0])
    # Phase B: release symmetry from top symmetric seeds
    tB_end = t0 + 75.0
    for idx in range(min(3, len(sym_cands))):
        if time.time() >= min(tB_end, DEAD - 90):
            break
        vs, Xs = sym_cands[idx]
        # skip extremely bad seeds to save time (intra 0.93)
        if vs > 0.92 and idx >= 2:
            continue
        y = _full_adam(Xs, steps=900, beta0=80.0, beta1=1500.0, lr0=0.008, lr1=0.001, iu=iu, deadline=min(DEAD - 85, tB_end + 10))
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            x_best = y.copy()
        # light LP snap on this seed
        if time.time() < min(tB_end, DEAD - 90):
            try:
                yl, vl = _lp_polish(y, budget=3.0, deadline=min(DEAD - 85, tB_end + 10))
            except Exception:
                continue
            if vl < best_val:
                best_val = vl
                x_best = yl.copy()

    # Phase C: basin hopping around incumbent (balanced kicks + rattler reinsert)
    it = 0
    while time.time() < DEAD - 28.0:
        it += 1
        # contact analysis for targeted kicks
        try:
            G = x_best @ x_best.T
            gall = G[iu]
            mx = gall.max()
            tight = gall > mx - 1e-7
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, iu[0][tight], 1)
            np.add.at(cnt, iu[1][tight], 1)
        except Exception:
            cnt = np.full(n, 5)
        ratt = np.where(cnt <= 1)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 5
        if mode == 0:
            sigma = 0.012; steps = 450; beta1 = 1200.0
        elif mode == 1:
            sigma = 0.03; steps = 450; beta1 = 1500.0
        elif mode == 2:
            sigma = 0.06; steps = 350; beta1 = 1000.0
        elif mode == 3:
            sigma = 0.02; steps = 550; beta1 = 2000.0
        else:
            sigma = 0.10; steps = 300; beta1 = 800.0
        y = x_best + rng.normal(0, sigma, size=(n, d))
        # reinsert rattlers / perturb hubs
        if len(ratt) > 0 and rng.random() < 0.7:
            for j in ratt:
                if rng.random() < 0.5:
                    y[j] = rng.normal(size=4)
                else:
                    y[j] = x_best[j] + rng.normal(0, 0.3, size=4)
        if len(hubs) > 0:
            y[hubs] += rng.normal(0, sigma * 0.7, size=(len(hubs), d))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _full_adam(y, steps=steps, beta0=60.0, beta1=beta1, lr0=0.008, lr1=0.001, iu=iu, deadline=DEAD - 26.0)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best_val:
            # quick LP to confirm basin
            try:
                yl, vl = _lp_polish(y, budget=2.5, deadline=DEAD - 24.0)
            except Exception:
                yl, vl = y, v
            if vl < best_val:
                best_val = vl
                x_best = yl.copy()
            elif v < best_val:
                best_val = v
                x_best = y.copy()
        if time.time() >= DEAD - 28.0:
            break

    # Final polish
    try:
        remain = DEAD - time.time() - 1.0
        if remain > 2.0:
            x_best, best_val = _lp_polish(x_best, budget=min(remain, 22.0))
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: return incumbent if something went wrong
    if x_best.shape != (153, 4) or not np.all(np.isfinite(x_best)):
        return xb
    return x_best
