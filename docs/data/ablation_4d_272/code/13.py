import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def lp_polish(x, seconds, cap=5000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
            best = v
            x = y
            bx = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return bx

def smooth_gd(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        if s0 != s1:
            smax = s0 * (s1 / s0) ** frac
        else:
            smax = s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = np.linalg.norm(vel, axis=1).max()
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def refine_single(Y, c0, steps=200, s=0.03, mu=0.5):
    c = c0 / np.linalg.norm(c0)
    vel = np.zeros(4)
    for _ in range(steps):
        dots = Y @ c
        m = dots.max()
        sel = dots > m - 2e-3
        if np.count_nonzero(sel) == 0:
            g = Y[np.argmax(dots)]
        else:
            g = Y[sel].sum(axis=0)
        gt = g - np.dot(g, c) * c
        n = np.linalg.norm(gt)
        if n < 1e-14:
            break
        gt = gt / n
        vel = mu * vel + gt
        vn = np.linalg.norm(vel)
        if vn > 1e-14:
            c = c - (s / vn) * vel
            c /= np.linalg.norm(c)
            s *= 0.996
    return c

def hole_reinsert(x, rng, n_samples=15000):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    k = int(np.argmax(G))
    i, j = divmod(k, n)
    Gi = G[i].copy(); Gj = G[j].copy()
    Gi[j] = -2; Gj[i] = -2
    move = i if Gi.max() > Gj.max() else j
    if rng.random() < 0.25:
        move = i if rng.random() < 0.5 else j
    Y = np.delete(x, move, axis=0)
    C = rng.standard_normal((n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_c = None; best_s = 2.0
    bs = 3000
    for a in range(0, n_samples, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sm = S.max(axis=1)
        kk = int(np.argmin(sm))
        if sm[kk] < best_s:
            best_s = float(sm[kk])
            best_c = Cb[kk].copy()
    best_c = refine_single(Y, best_c, steps=250, s=0.03)
    y = x.copy()
    y[move] = best_c / np.linalg.norm(best_c)
    return y

def subset_twist(x, rng, frac=0.4, angle=0.15):
    n, d = x.shape
    # random plane
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    u = Q[:, 0]; v = Q[:, 1]
    # random subset: hemisphere or random mask
    if rng.random() < 0.5:
        c = rng.standard_normal(d); c /= np.linalg.norm(c)
        dots = x @ c
        k = max(10, int(n * frac))
        idx = np.argpartition(-dots, k - 1)[:k]
    else:
        k = max(10, int(n * frac))
        idx = rng.choice(n, size=k, replace=False)
    ca = np.cos(angle); sa = np.sin(angle)
    y = x.copy()
    for ii in idx:
        xi = x[ii]
        a = float(xi @ u); b = float(xi @ v)
        perp = xi - a * u - b * v
        a2 = ca * a + sa * b
        b2 = -sa * a + ca * b
        yi = perp + a2 * u + b2 * v
        yi /= np.linalg.norm(yi)
        y[ii] = yi
    return y

def hopf_init(rng, n=272):
    nl = int(rng.integers(9, 17))
    u = rng.random(nl)
    theta = np.arcsin(np.sqrt(np.clip(u, 1e-6, 1 - 1e-6)))
    theta = np.sort(theta)
    theta += rng.standard_normal(nl) * 0.02
    theta = np.clip(theta, 0.05, np.pi/2 - 0.05)
    w = np.sin(theta) * np.cos(theta) + 0.15
    w = w / w.sum()
    counts = np.maximum(3, np.round(n * w).astype(int))
    # adjust to sum n
    while counts.sum() > n:
        k = int(np.argmax(counts))
        counts[k] -= 1
    while counts.sum() < n:
        k = int(np.argmax(w / np.maximum(counts, 1)))
        counts[k] += 1
    X = []
    for th, ck in zip(theta, counts):
        o1 = rng.random() * 2 * np.pi
        o2 = rng.random() * 2 * np.pi
        p1 = rng.random(ck) * 2 * np.pi + o1
        p2 = rng.random(ck) * 2 * np.pi + o2
        ct = np.cos(th); st = np.sin(th)
        for a, b in zip(p1, p2):
            X.append([ct*np.cos(a), ct*np.sin(a), st*np.cos(b), st*np.sin(b)])
    X = np.array(X[:n], dtype=np.float64)
    # shuffle
    perm = rng.permutation(n)
    return X[perm]

def solve():
    t0 = time.time()
    BUDGET = 170.0
    rng = np.random.default_rng(0)
    n, d = 272, 4
    try:
        bx = np.load("best.npy")
    except Exception:
        bx = None
    try:
        rx = np.load("record.npy")
    except Exception:
        rx = None
    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a
    best_x = None; best_v = 2.0
    for cand in (bx, rx):
        if cand is not None:
            try:
                c = normed(cand)
                if c.shape == (272, 4):
                    v = max_cos(c)
                    if v < best_v:
                        best_v = v; best_x = c.copy()
            except Exception:
                pass
    if best_x is None:
        best_x = normed(rng.standard_normal((n, d)))
        best_v = max_cos(best_x)
    base_best = best_x.copy()

    # Phase A: diversification with Hopf + random starts, fast GD only
    phaseA_end = t0 + 32.0
    starts = []
    for _ in range(6):
        starts.append(hopf_init(rng))
    for _ in range(3):
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        starts.append(a)
    # also twisted record
    for _ in range(3):
        starts.append(subset_twist(base_best, rng, frac=0.3+rng.random()*0.3, angle=0.2+rng.random()*0.4))
    for s in starts:
        if time.time() > phaseA_end:
            break
        a0 = float(rng.choice([20.0, 40.0, 60.0]))
        a1 = float(rng.choice([300.0, 500.0, 800.0]))
        y = smooth_gd(s.copy(), steps=450, a0=a0, a1=a1, s0=0.02, s1=0.004, mu=0.85)
        v = max_cos(y)
        if v < best_v + 0.002:
            # short lock
            yl = lp_polish(y, seconds=1.2, cap=4000)
            vl = max_cos(yl)
            if vl < best_v:
                best_v = vl; best_x = yl.copy()
        elif v < best_v:
            best_v = v; best_x = y.copy()
        if time.time() > phaseA_end:
            break

    # Phase B: basin hopping
    it = 0
    sigmas = [0.004, 0.010, 0.022, 0.05, 0.10, 0.03, 0.008]
    while time.time() - t0 < BUDGET - 13.0:
        it += 1
        time_left = BUDGET - 13.0 - (time.time() - t0)
        if time_left <= 0:
            break
        op = it % 6
        y = None
        if op in (0, 1, 2):
            sigma = sigmas[(it - 1) % len(sigmas)] * (0.7 + 0.6 * rng.random())
            rowm, mcur = row_max(best_x)
            is_rat = rowm < (mcur - 5e-4)
            noise = rng.standard_normal(best_x.shape)
            # project to tangent for stability
            noise = noise - np.sum(noise * best_x, axis=1, keepdims=True) * best_x
            y = best_x.copy()
            if op == 1 and np.count_nonzero(is_rat) > 0:
                # rattler-targeted: big kick to rattlers, tiny to others
                y[is_rat] += noise[is_rat] * (sigma * 3.0)
                y[~is_rat] += noise[~is_rat] * (sigma * 0.25)
            elif op == 2:
                # cluster kick around worst pair
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G))
                wi, wj = divmod(k, n)
                # neighbors of wi,wj
                neigh = set([wi, wj])
                for qq in (wi, wj):
                    top = np.argpartition(-G[qq], 8)[:8]
                    for t in top:
                        neigh.add(int(t))
                neigh = np.array(list(neigh), dtype=int)
                y = best_x.copy()
                y[neigh] += noise[neigh] * (sigma * 2.0)
                mask = np.ones(n, dtype=bool); mask[neigh] = False
                y[mask] += noise[mask] * (sigma * 0.2)
            else:
                y = best_x + noise * sigma
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif op == 3:
            y = hole_reinsert(best_x, rng, n_samples=15000)
        elif op == 4:
            # double hole: two successive reinsertions
            y = hole_reinsert(best_x, rng, n_samples=10000)
            y = hole_reinsert(y, rng, n_samples=10000)
        else:
            frac = 0.2 + rng.random() * 0.4
            ang = 0.05 + rng.random() * 0.35
            y = subset_twist(best_x, rng, frac=frac, angle=ang)
            # small jitter after twist
            y += rng.standard_normal(y.shape) * 0.002
            y /= np.linalg.norm(y, axis=1, keepdims=True)

        # annealed smoothing with randomized schedule
        a0 = float(rng.choice([25.0, 45.0, 80.0]))
        a1 = float(rng.choice([350.0, 600.0, 1000.0]))
        steps = int(rng.choice([280, 400, 550]))
        s0 = 0.015 + rng.random() * 0.015
        s1 = 0.003 + rng.random() * 0.005
        mu = 0.8 + rng.random() * 0.12
        y = smooth_gd(y, steps=steps, a0=a0, a1=a1, s0=s0, s1=s1, mu=mu)
        v = max_cos(y)
        # only LP-polish if promising to save time
        thresh_lp = best_v + 0.004
        if v < thresh_lp:
            # adaptive LP time: longer if very close
            lp_t = 1.6 if v > best_v else 2.5
            # don't exceed remaining
            rem = BUDGET - 13.0 - (time.time() - t0)
            lp_t = min(lp_t, max(0.6, rem - 0.5))
            if lp_t > 0.5:
                y = lp_polish(y, seconds=lp_t, cap=4500)
                v = max_cos(y)
        if v < best_v:
            best_v = v
            best_x = y.copy()

    # final long polish
    rem = BUDGET - (time.time() - t0)
    ftime = max(2.0, min(12.0, rem - 1.0))
    if ftime > 0.5:
        try:
            best_x = lp_polish(best_x, seconds=ftime, cap=6000)
        except Exception:
            pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
