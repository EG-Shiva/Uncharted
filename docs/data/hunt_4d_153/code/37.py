import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
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
        if time.time() - start >= budget:
            break
    return x, best

def _smooth_opt(x0, steps, beta0, beta1, lr0, lr1, iu):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    Iu0, Ju0 = iu
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
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
        mhat = m1 / c1
        vhat = v1 / c2
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(nr, 1e-12, out=nr)
        x /= nr
    return x

def _expand_torus(a, th, ph, mvec, L):
    norb = len(a)
    n = norb * L
    # sigmoid for r
    r = 1.0 / (1.0 + np.exp(-a))
    r = np.clip(r, 1e-3, 1 - 1e-3)
    s = np.sqrt(np.clip(1 - r * r, 1e-12, 1))
    X = np.empty((n, 4))
    tw = 2 * np.pi / L
    for j in range(norb):
        ks = np.arange(L)
        c1 = np.cos(th[j] + tw * ks)
        s1 = np.sin(th[j] + tw * ks)
        c2 = np.cos(ph[j] + tw * mvec[j] * ks)
        s2 = np.sin(ph[j] + tw * mvec[j] * ks)
        X[j * L:(j + 1) * L, 0] = r[j] * c1
        X[j * L:(j + 1) * L, 1] = r[j] * s1
        X[j * L:(j + 1) * L, 2] = s[j] * c2
        X[j * L:(j + 1) * L, 3] = s[j] * s2
    return X, r, s

def _sym_run(mvec, L, steps, beta0, beta1, lr, seed):
    rng = np.random.default_rng(seed)
    norb = len(mvec)
    # init params
    r0 = rng.uniform(0.25, 0.92, size=norb)
    a = np.log(r0 / (1 - r0))
    th = rng.uniform(0, 2 * np.pi, size=norb)
    ph = rng.uniform(0, 2 * np.pi, size=norb)
    m1 = np.zeros(3 * norb); v1 = np.zeros(3 * norb)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    tw = 2 * np.pi / L
    ks_all = np.arange(L)
    n = norb * L
    iu = np.triu_indices(n, 1)
    Iu0, Ju0 = iu
    best_X = None; best_v = 1e9
    for s in range(steps):
        beta = beta0 * np.exp(betb * s)
        X, r, ss = _expand_torus(a, th, ph, mvec, L)
        G = X @ X.T
        g = G[Iu0, Ju0]
        gmax = g.max()
        e = np.exp(beta * (g - gmax))
        Z = e.sum()
        w = e / Z
        W = np.zeros((n, n))
        W[Iu0, Ju0] = w
        W = W + W.T
        gradX = W @ X  # (n,4)
        # chain to params
        ga = np.zeros(norb); gth = np.zeros(norb); gph = np.zeros(norb)
        for j in range(norb):
            ks = ks_all
            a1 = th[j] + tw * ks
            a2 = ph[j] + tw * mvec[j] * ks
            c1 = np.cos(a1); s1n = np.sin(a1)
            c2 = np.cos(a2); s2n = np.sin(a2)
            gx = gradX[j * L:(j + 1) * L]
            # dth
            dth_x0 = -r[j] * s1n
            dth_x1 = r[j] * c1
            gth[j] = np.sum(gx[:, 0] * dth_x0 + gx[:, 1] * dth_x1)
            # dph
            dph_x2 = -ss[j] * s2n
            dph_x3 = ss[j] * c2
            gph[j] = np.sum(gx[:, 2] * dph_x2 + gx[:, 3] * dph_x3)
            # da via dr
            drda = r[j] * (1 - r[j])
            dsdr = -r[j] / max(ss[j], 1e-9)
            dsda = dsdr * drda
            dx0dr = c1; dx1dr = s1n
            dx2ds = c2; dx3ds = s2n
            # dX/da = dX/dr*drda for first two, dX/ds*dsda for last two
            ga[j] = np.sum(gx[:, 0] * dx0dr * drda + gx[:, 1] * dx1dr * drda +
                           gx[:, 2] * dx2ds * dsda + gx[:, 3] * dx3ds * dsda)
        gvec = np.concatenate([ga, gth, gph])
        # Adam
        m1 = b1 * m1 + (1 - b1) * gvec
        v1 = b2 * v1 + (1 - b2) * (gvec * gvec)
        c1_ = 1 - b1 ** (s + 1); c2_ = 1 - b2 ** (s + 1)
        mh = m1 / c1_; vh = v1 / c2_
        upd = mh / (np.sqrt(vh) + eps)
        # scale angles vs a same lr
        pvec = np.concatenate([a, th, ph]) - lr * upd
        a = pvec[0:norb]; th = pvec[norb:2*norb]; ph = pvec[2*norb:3*norb]
        # wrap angles
        # track best by true max
        if (s % 50 == 0) or (s == steps - 1):
            Xc, _, _ = _expand_torus(a, th, ph, mvec, L)
            v = _max_cos(Xc)
            if v < best_v:
                best_v = v; best_X = Xc.copy()
    return best_X, best_v

def solve():
    t0 = time.time()
    HARD = 172.0
    n, d = 153, 4
    iu = np.triu_indices(n, 1)
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        xb = xb.reshape(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    # ---- symmetric torus search (time-boxed ~30s) ----
    sym_best_X = None; sym_best_v = 1e9
    L1 = 17; norb1 = 9
    L2 = 9; norb2 = 17
    patterns = []
    # L=17 patterns
    patterns.append((L1, np.ones(norb1, dtype=int)))
    patterns.append((L1, np.full(norb1, 2, dtype=int)))
    patterns.append((L1, np.full(norb1, 3, dtype=int)))
    patterns.append((L1, np.array([1,2,3,4,5,6,7,8,9], dtype=int)))
    patterns.append((L1, np.array([1,3,5,7,9,11,13,15,2], dtype=int)))
    prng = np.random.default_rng(123)
    for _ in range(2):
        patterns.append((L1, prng.integers(1, 17, size=norb1)))
    patterns.append((L2, np.ones(norb2, dtype=int)))
    patterns.append((L2, np.full(norb2, 2, dtype=int)))
    patterns.append((L2, prng.integers(1, 9, size=norb2)))
    # filter m coprime for L2
    fpatterns = []
    for L, mv in patterns:
        mv = np.asarray(mv, dtype=int)
        if L == 9:
            mv = np.where(np.gcd(mv, 9) == 1, mv, 1)
        fpatterns.append((L, mv))
    sid = 0
    sym_start = time.time()
    for (L, mv) in fpatterns:
        if time.time() - t0 > 38.0:
            break
        if time.time() - sym_start > 28.0:
            break
        for rep in range(2):
            if time.time() - t0 > 38.0:
                break
            sid += 1
            try:
                Xc, vc = _sym_run(mv, L, steps=900, beta0=30.0, beta1=320.0, lr=0.02, seed=1000+sid)
            except Exception:
                continue
            if vc < sym_best_v:
                sym_best_v = vc; sym_best_X = Xc.copy()
    # release symmetry: unrestricted descent + polish on sym best
    sym_released = None; sym_released_v = 1e9
    if sym_best_X is not None:
        try:
            y = _smooth_opt(sym_best_X, 900, 40.0, 300.0, 0.006, 0.0004, iu)
            v = _max_cos(y)
            # light polish
            if v < best_val + 0.01 and (HARD - (time.time()-t0)) > 30:
                y2, v2 = _lp_polish(y, min(4.0, max(1.0, (HARD-(time.time()-t0))/10)))
            else:
                y2, v2 = y, v
            sym_released, sym_released_v = y2, v2
            if v2 < best_val - 1e-12:
                x_best, best_val = y2, v2
        except Exception:
            pass
        # second perturbed release
        try:
            if (HARD - (time.time()-t0)) > 40:
                yp = sym_best_X + rng.normal(0, 0.01, size=sym_best_X.shape)
                yp /= np.linalg.norm(yp, axis=1, keepdims=True)
                y = _smooth_opt(yp, 800, 30.0, 300.0, 0.006, 0.0004, iu)
                v = _max_cos(y)
                if v < best_val + 0.005:
                    y2, v2 = _lp_polish(y, 2.0)
                    if v2 < best_val - 1e-12:
                        x_best, best_val = y2, v2
                    if v2 < sym_released_v:
                        sym_released, sym_released_v = y2, v2
        except Exception:
            pass
    # if sym failed, use incumbent as sym_released proxy
    if sym_released is None:
        sym_released = x_best.copy(); sym_released_v = best_val
    # ---- basin hopping with rattler-targeted kicks, two basins ----
    it = 0
    while time.time() - t0 < 140.0:
        it += 1
        if time.time() - t0 > HARD - 30:
            break
        # choose seed basin: mostly incumbent, sometimes symmetric basin
        use_sym = (it % 4 == 3) and (sym_released is not None)
        seed = sym_released if use_sym else x_best
        G = seed @ seed.T
        gall = G[iu]
        mx = gall.max()
        tight = gall > mx - 1e-7
        cnt = np.zeros(n, dtype=int)
        np.add.at(cnt, iu[0][tight], 1)
        np.add.at(cnt, iu[1][tight], 1)
        ratt = np.where(cnt <= 2)[0]
        hubs = np.where(cnt >= 7)[0]
        mode = it % 6
        if mode == 0:
            sigma, nrep, extra = 0.015, 0, 0.0
        elif mode == 1:
            sigma, nrep, extra = 0.03, 1, 0.25
        elif mode == 2:
            sigma, nrep, extra = 0.07, 2, 0.25
        elif mode == 3:
            sigma, nrep, extra = 0.02, 1, 0.4
        elif mode == 4:
            sigma, nrep, extra = 0.12, 2, 0.0
        else:
            sigma, nrep, extra = 0.05, 3, 0.15
        y = seed.copy()
        y += rng.normal(0, sigma, size=y.shape)
        if len(ratt) > 0 and extra > 0:
            for idx in ratt:
                if rng.random() < 0.6:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, extra, size=4)
        if len(hubs) > 0:
            y[hubs] += rng.normal(0, sigma * 0.8, size=(len(hubs), d))
        if nrep > 0:
            rep = rng.choice(n, size=nrep, replace=False)
            y[rep] = rng.normal(size=(nrep, 4))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        steps = 1000
        if mode in (2, 4):
            beta0, beta1 = 25.0, 350.0
        else:
            beta0, beta1 = 40.0, 280.0
        y2 = _smooth_opt(y, steps, beta0, beta1, 0.006, 0.0004, iu)
        v2 = _max_cos(y2)
        thresh = best_val + (0.004 if mode in (2, 4, 5) else 0.0015)
        # also allow sym basin updates
        if use_sym and v2 < sym_released_v - 1e-9:
            # polish lightly to keep basin alive
            remain = HARD - (time.time() - t0) - 22.0
            if remain > 2.0:
                yp, vp = _lp_polish(y2, min(2.0, remain / 8.0))
                if vp < sym_released_v:
                    sym_released, sym_released_v = yp, vp
                if vp < best_val - 1e-12:
                    x_best, best_val = yp, vp
            else:
                if v2 < sym_released_v:
                    sym_released, sym_released_v = y2, v2
        if v2 < thresh:
            remain = HARD - (time.time() - t0) - 22.0
            if remain < 1.0:
                break
            b = min(3.0, max(1.2, remain / 8.0))
            yp, vp = _lp_polish(y2, b)
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
            elif use_sym and vp < sym_released_v:
                sym_released = yp; sym_released_v = vp
        if time.time() - t0 > HARD - 25:
            break
    remain = HARD - (time.time() - t0) - 2.0
    if remain > 2.0:
        remain = min(remain, 25.0)
        try:
            x_best, best_val = _lp_polish(x_best, remain)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
