import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_TRIU = np.triu_indices(153, 1)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-7):
    n = x.shape[0]
    G = x @ x.T
    gall = G[_TRIU]
    mx = float(gall.max())
    tight = gall > mx - thr
    cnt = np.zeros(n, dtype=int)
    np.add.at(cnt, _TRIU[0][tight], 1)
    np.add.at(cnt, _TRIU[1][tight], 1)
    return mx, cnt, G

def _adam_descend(x0, steps, beta0, beta1, lr0, lr1, mode='soft', s0=2.0, s1=8.0, switch_frac=0.5):
    n, d = x0.shape
    x = x0.copy()
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    Iu0, Ju0 = _TRIU
    lrb = np.log(lr1 / lr0) / max(steps - 1, 1)
    betb = np.log(beta1 / beta0) / max(steps - 1, 1)
    sb = (s1 - s0) / max(steps * switch_frac - 1, 1) if steps * switch_frac > 1 else 0.0
    W = np.zeros((n, n))
    for s in range(steps):
        lr = lr0 * np.exp(lrb * s)
        G = x @ x.T
        g = G[Iu0, Ju0]
        use_riesz = (mode == 'riesz') or (mode == 'mix' and s < steps * switch_frac)
        if use_riesz:
            cur_s = s0 + sb * s if mode == 'mix' else (s0 + (s1 - s0) * s / max(steps - 1, 1))
            p = cur_s * 0.5 + 1.0
            r2 = np.maximum(2.0 - 2.0 * g, 1e-12)
            e = r2 ** (-p)
            # normalize to keep step scale stable
            e = e / (e.sum() + 1e-30) * len(e)
            w = e
        else:
            if mode == 'mix':
                frac = (s - steps * switch_frac) / max(steps * (1 - switch_frac) - 1, 1)
                beta = beta0 * np.exp(betb * s) if beta0 > 0 else 40.0
                # ramp beta a bit with frac already via schedule
            else:
                beta = beta0 * np.exp(betb * s)
            gmax = g.max()
            # subtract max for stability
            e = np.exp(beta * (g - gmax))
            Z = e.sum()
            w = e / (Z + 1e-30)
        W.fill(0.0)
        W[Iu0, Ju0] = w
        W[Ju0, Iu0] = w
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

def _lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = _TRIU
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

def _rand_orth(rng):
    A = rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-30)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _splice(a, b, rng, jitter=0.008):
    n = a.shape[0]
    R = _rand_orth(rng)
    bR = b @ R.T
    u = rng.normal(size=4); u /= np.linalg.norm(u)
    pa = a @ u
    pb = bR @ u
    k = n // 2 + int(rng.integers(0, 2))
    ia = np.argpartition(-pa, k - 1)[:k]
    kb = n - k
    ib = np.argpartition(pb, kb - 1)[:kb] if kb > 0 else np.zeros(0, dtype=int)
    # take top-k from a, bottom-kb from bR => clean cut
    # for b we want smallest pb
    c = np.concatenate([a[ia], bR[ib]], axis=0)
    c = c + rng.normal(0, jitter, size=c.shape)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    # shuffle order to avoid bias
    perm = rng.permutation(n)
    return c[perm]

def _cap_resample(x, rng, m):
    y = x.copy()
    cen = rng.normal(size=4); cen /= np.linalg.norm(cen)
    # sometimes center on existing point to hit dense region
    if rng.random() < 0.5:
        cen = x[rng.integers(0, x.shape[0])] + rng.normal(0, 0.2, size=4)
        cen /= np.linalg.norm(cen)
    dots = y @ cen
    idx = np.argpartition(-dots, m - 1)[:m]
    y[idx] = rng.normal(size=(m, 4))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _sym_seed(m, twist, rng):
    n = 153
    th1 = 2 * np.pi / m
    th2 = 2 * np.pi * twist / m
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    R = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, c2, -s2], [0, 0, s2, c2]])
    n_orb = n // m
    rem = n - n_orb * m
    bases = rng.normal(size=(n_orb, 4))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    # random global rotation
    Q = _rand_orth(rng)
    bases = bases @ Q.T
    pts = []
    Rm = np.eye(4)
    for t in range(m):
        for b in bases:
            pts.append(Rm @ b)
        Rm = R @ Rm
    if rem > 0:
        extra = rng.normal(size=(rem, 4))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        for e in extra:
            pts.append(e)
    X = np.array(pts[:n])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    HARD = 171.0
    rng = np.random.default_rng(7)
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
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float).reshape(153, 4)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best_val = _max_cos(xb)
    x_best = xb.copy()
    n, d = 153, 4
    iu = _TRIU

    # build small diverse pool quickly
    pool = [x_best.copy()]
    pool_vals = [best_val]
    # symmetric seeds + quick descend for diversity
    sym_params = [(9, 2), (17, 3), (3, 1)]
    for (mm, tw) in sym_params:
        if time.time() - t0 > 30:
            break
        try:
            s0 = _sym_seed(mm, tw, rng)
            s0d = _adam_descend(s0, 500, 8.0, 200.0, 0.008, 0.0006, mode='mix', s0=1.5, s1=6.0, switch_frac=0.5)
            v = _max_cos(s0d)
            pool.append(s0d); pool_vals.append(v)
            if v < best_val - 1e-12:
                x_best, best_val = s0d.copy(), v
        except Exception:
            pass
    # a couple of Coulomb-uniformized random starts
    for _ in range(2):
        if time.time() - t0 > 40:
            break
        r0 = rng.normal(size=(n, d))
        r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
        r0d = _adam_descend(r0, 500, 10.0, 220.0, 0.01, 0.0006, mode='mix', s0=1.0, s1=6.0, switch_frac=0.6)
        v = _max_cos(r0d)
        pool.append(r0d); pool_vals.append(v)
        if v < best_val - 1e-12:
            x_best, best_val = r0d.copy(), v
    # keep pool size bounded, keep best + most diverse good ones
    def _trim_pool():
        global pool, pool_vals
        idx = np.argsort(np.array(pool_vals))
        # always keep best overall first
        keep = idx[:8].tolist()
        pool = [pool[i] for i in keep]
        pool_vals = [pool_vals[i] for i in keep]

    _trim_pool()
    # ensure x_best is pool[0] if it is global best
    # main evolutionary loop
    it = 0
    while time.time() - t0 < HARD - 32:
        it += 1
        remain_total = HARD - (time.time() - t0) - 24.0
        if remain_total < 2.0:
            break
        mode = it % 7
        y = None
        desc_mode = 'soft'
        beta0, beta1 = 40.0, 280.0
        lr0, lr1 = 0.006, 0.0004
        steps = 800
        s0p, s1p, sw = 1.5, 7.0, 0.5
        # pick parents
        order = np.argsort(np.array(pool_vals))
        elite = pool[order[0]]
        second = pool[order[1]] if len(pool) > 1 else elite
        if mode == 0:
            # small gaussian exploit around best
            y = x_best + rng.normal(0, 0.018, size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            beta0, beta1 = 45.0, 320.0
            steps = 700
        elif mode == 1:
            # medium kick + rattler replacement
            G = x_best @ x_best.T
            gall = G[iu]
            mx = gall.max()
            tight = gall > mx - 1e-7
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, iu[0][tight], 1)
            np.add.at(cnt, iu[1][tight], 1)
            y = x_best + rng.normal(0, 0.035, size=x_best.shape)
            ratt = np.where(cnt <= 2)[0]
            hubs = np.where(cnt >= 7)[0]
            for idx in ratt:
                if rng.random() < 0.7:
                    y[idx] = rng.normal(size=4)
                else:
                    y[idx] += rng.normal(0, 0.3, size=4)
            if len(hubs) > 0:
                y[hubs] += rng.normal(0, 0.03, size=(len(hubs), d))
            nrep = 1
            rep = rng.choice(n, size=nrep, replace=False)
            y[rep] = rng.normal(size=(nrep, 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            beta0, beta1 = 35.0, 300.0
            steps = 850
        elif mode == 2:
            # hemisphere splice: big topology change
            # choose two distinct pool members randomly
            a_idx, b_idx = rng.choice(len(pool), size=2, replace=len(pool) < 2)
            y = _splice(pool[a_idx], pool[b_idx], rng, jitter=0.01)
            desc_mode = 'mix'; beta0, beta1 = 20.0, 320.0
            lr0, lr1 = 0.008, 0.0004
            steps = 950
            s0p, s1p, sw = 1.0, 6.0, 0.45
        elif mode == 3:
            # cap deletion / reinsertion
            m = int(rng.integers(8, 22))
            y = _cap_resample(x_best, rng, m)
            y += rng.normal(0, 0.01, size=y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            desc_mode = 'mix'; beta0, beta1 = 15.0, 300.0
            lr0, lr1 = 0.008, 0.0004
            steps = 950
            s0p, s1p, sw = 1.0, 7.0, 0.5
        elif mode == 4:
            # large kick + pure Riesz pre-spread then softmax
            y = x_best + rng.normal(0, 0.12, size=x_best.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            desc_mode = 'mix'; beta0, beta1 = 25.0, 350.0
            lr0, lr1 = 0.01, 0.0005
            steps = 1000
            s0p, s1p, sw = 2.0, 8.0, 0.55
        elif mode == 5:
            # crossover with fresh random (inject novelty)
            r0 = rng.normal(size=(n, d))
            r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
            # align not needed; splice elite with random
            y = _splice(x_best, r0, rng, jitter=0.005)
            desc_mode = 'mix'; beta0, beta1 = 12.0, 280.0
            lr0, lr1 = 0.009, 0.0004
            steps = 900
            s0p, s1p, sw = 1.0, 6.0, 0.5
        else:
            # hub-focused shake: move overloaded cluster together
            G = x_best @ x_best.T
            gall = G[iu]
            mx = gall.max()
            tight = gall > mx - 1e-7
            cnt = np.zeros(n, dtype=int)
            np.add.at(cnt, iu[0][tight], 1)
            np.add.at(cnt, iu[1][tight], 1)
            hub = int(np.argmax(cnt))
            dots = x_best @ x_best[hub]
            k = int(rng.integers(6, 14))
            clust = np.argpartition(-dots, k - 1)[:k]
            y = x_best.copy()
            y += rng.normal(0, 0.015, size=y.shape)
            y[clust] += rng.normal(0, 0.09, size=(k, d))
            # also relocate rattler if exists
            ratt = np.where(cnt <= 1)[0]
            if len(ratt) > 0:
                y[ratt] = rng.normal(size=(len(ratt), 4))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            beta0, beta1 = 40.0, 300.0
            steps = 800
        # descend
        try:
            y2 = _adam_descend(y, steps, beta0, beta1, lr0, lr1, mode=desc_mode, s0=s0p, s1=s1p, switch_frac=sw)
        except Exception:
            continue
        try:
            v2 = _max_cos(y2)
        except Exception:
            continue
        # decide LP polish
        thresh = best_val + (0.004 if mode in (2, 3, 4, 5) else 0.0015)
        if v2 < thresh:
            rem = HARD - (time.time() - t0) - 23.0
            if rem < 1.0:
                break
            b = min(2.5, max(1.0, rem / 10.0))
            try:
                yp, vp = _lp_polish(y2, b)
            except Exception:
                continue
            if vp < best_val - 1e-12:
                x_best, best_val = yp, vp
                pool.append(yp); pool_vals.append(vp)
                _trim_pool()
            elif vp < best_val + 0.002:
                # keep diverse near-best to enable further splices
                pool.append(yp); pool_vals.append(vp)
                _trim_pool()
            elif v2 < best_val + 0.002 and len(pool) < 8:
                pool.append(y2); pool_vals.append(v2)
        else:
            # still keep occasionally for diversity if quite good
            if v2 < best_val + 0.001 and len(pool) < 8:
                pool.append(y2); pool_vals.append(v2)
        if time.time() - t0 > HARD - 26:
            break
    remain = HARD - (time.time() - t0) - 1.5
    if remain > 2.0:
        remain = min(remain, 26.0)
        try:
            x_best, best_val = _lp_polish(x_best, remain)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
