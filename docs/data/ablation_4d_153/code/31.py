import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), float(np.max(G))

def _random_sphere(n, d, rng):
    x = rng.standard_normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _torus_init(n, rng, nt, jitter=0.02):
    # Clifford torus coords: [cos t cos a, cos t sin a, sin t cos b, sin t sin b]
    # nt tori, per = n // nt points each with golden-slope line + shifts
    per = n // nt
    rem = n - per * nt
    xs = []
    phi = 0.6180339887498949
    psi = 0.7548776662466927
    for k in range(nt):
        # equal-area tori
        t = np.arcsin(np.sqrt((k + 0.5) / nt))
        ct, st = np.cos(t), np.sin(t)
        m = per + (1 if k < rem else 0)
        s1 = rng.random()
        s2 = rng.random()
        for i in range(m):
            a = 2.0 * np.pi * ((i * phi + s1) % 1.0)
            b = 2.0 * np.pi * ((i * psi + s2) % 1.0)
            xs.append([ct * np.cos(a), ct * np.sin(a), st * np.cos(b), st * np.sin(b)])
    x = np.array(xs, dtype=float)
    if jitter > 0:
        x = x + rng.standard_normal(size=x.shape) * jitter
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _smooth_opt(x, steps, tau0, tau1, lr0, lr1, mom=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        np.maximum(row, 1e-12, out=row)
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        # clip
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        np.maximum(gn, 1e-12, out=gn)
        scale = np.minimum(1.0, 2.0 / gn)
        grad = grad * scale
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _hole_center(x, rng, nsamp=12000, refine=60):
    n, d = x.shape
    cand = rng.standard_normal(size=(nsamp, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ x.T
    best_dot = np.max(dots, axis=1)
    bi = int(np.argmin(best_dot))
    h = cand[bi].copy()
    # hill climb: move away from neighbours
    for _ in range(refine):
        dts = x @ h  # (n,)
        # softmax weight to nearest
        tau = 0.01
        m = float(np.max(dts))
        w = np.exp((dts - m) / tau)
        w = w / (np.sum(w) + 1e-12)
        g = w @ x  # direction of neighbours
        # move opposite, tangent
        g = g - np.dot(g, h) * h
        gn = float(np.linalg.norm(g))
        if gn < 1e-12:
            break
        g = g / gn
        step = 0.02
        hn = h - step * g
        hn = hn / np.linalg.norm(hn)
        # accept if better
        nd = float(np.max(hn @ x.T))
        od = float(np.max(h @ x.T))
        if nd < od:
            h = hn
        else:
            break
    return h

def _lp_polish(x, deadline):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            break
        if idx.size > 2800:
            order = np.argpartition(c[idx], -2800)[-2800:]
            idx = idx[order]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = t0 + 118.0
    rng = np.random.default_rng(12345)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _random_sphere(n, d, rng)
        best_v = _max_cos(best_x)
    base_record = best_x.copy()
    # pool of smoothed candidates (keep top few for LP)
    pool = []  # (val, x)
    def consider(x, v=None):
        nonlocal best_v, best_x
        if v is None:
            v = _max_cos(x)
        if len(pool) < 6:
            pool.append((v, x.copy()))
            pool.sort(key=lambda t: t[0])
        elif v < pool[-1][0]:
            pool[-1] = (v, x.copy())
            pool.sort(key=lambda t: t[0])
        return v

    # quick diverse torus/random smoothed seeds first
    init_jobs = []
    # structured torus inits
    for nt in (9, 17, 3, 6):
        try:
            init_jobs.append(_torus_init(n, rng, nt, jitter=0.05))
        except Exception:
            pass
    for _ in range(2):
        init_jobs.append(_random_sphere(n, d, rng))
    # large-kicked record inits
    for sig in (0.35, 0.55):
        y = base_record + rng.standard_normal(size=(n, d)) * sig
        # tangent-ish then normalize
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        init_jobs.append(y)

    # smooth each init job briefly, keep
    for y in init_jobs:
        if time.time() > EXPLORE_END - 20.0:
            break
        z = _smooth_opt(y.copy(), steps=900, tau0=0.06, tau1=0.004, lr0=0.25, lr1=0.02, mom=0.85)
        v = _max_cos(z)
        consider(z, v)
        # also a longer anneal variant for the first two
        if time.time() > EXPLORE_END:
            break

    # main surgery + kick loop
    it = 0
    sigmas = [0.02, 0.06, 0.15, 0.30, 0.50]
    while time.time() < EXPLORE_END:
        it += 1
        mode = it % 6
        # choose base: mostly best, sometimes pool member or record
        if len(pool) >= 2 and (it % 4 == 0):
            base = pool[rng.integers(len(pool))][1]
        elif it % 3 == 0:
            base = base_record
        else:
            # current best known (global)
            base = best_x if best_x is not None else base_record
            if len(pool) > 0 and rng.random() < 0.4:
                base = pool[0][1]
        y = base.copy()
        if mode == 0:
            # global kick
            sig = sigmas[(it // 6) % len(sigmas)]
            noise = rng.standard_normal(size=(n, d)) * sig
            dot = np.sum(noise * y, axis=1, keepdims=True)
            noise = noise - dot * y
            y = y + noise
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        elif mode == 1:
            # subset re-randomize
            k = int(rng.integers(10, 36))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] = _random_sphere(k, d, rng)
        elif mode == 2:
            # worst-to-hole surgery: move worst points to holes
            rm, _ = _row_max(y)
            order = np.argsort(-rm)
            nh = 1 + (it % 3)  # 1-3 holes
            for h_i in range(nh):
                h = _hole_center(y, rng, nsamp=6000, refine=30)
                # small jitter around hole so multiple holes differ
                h = h + rng.standard_normal(size=(d,)) * 0.01
                h /= np.linalg.norm(h)
                victim = int(order[h_i % len(order)])
                y[victim] = h
        elif mode == 3:
            # rattler shake + neighbour shake: find low-contact points
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            thr = best_v - 1e-4 if best_v < 1 else 0.84
            deg = np.sum(G > thr, axis=1)
            low = np.where(deg <= 1)[0]
            if low.size == 0:
                low = np.argsort(deg)[:6]
            k = min(int(low.size), 4 + (it % 5))
            sel = rng.choice(low, size=k, replace=False) if low.size >= k else low
            noise = rng.standard_normal(size=(sel.size, d)) * 0.25
            y[sel] = y[sel] + noise
            y[sel] /= np.linalg.norm(y[sel], axis=1, keepdims=True)
            # also kick worst cluster
            rm, _ = _row_max(y)
            w = int(np.argmax(rm))
            y[w] = y[w] + rng.standard_normal(size=(d,)) * 0.1
            y[w] /= np.linalg.norm(y[w])
        elif mode == 4:
            # pair-swap / twist: random rotation of a subset
            k = int(rng.integers(15, 45))
            idx = rng.choice(n, size=k, replace=False)
            # random small rotation in 4D
            A = rng.standard_normal(size=(d, d)) * 0.35
            A = A - A.T
            # exp approx: Q = I + A (orthonormalized)
            Q, _ = np.linalg.qr(np.eye(d) + A)
            if float(np.linalg.det(Q)) < 0:
                Q[:, 0] = -Q[:, 0]
            y[idx] = (y[idx] @ Q.T)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # fresh torus jitter restart
            nt = [9, 17, 3, 6][it % 4]
            y = _torus_init(n, rng, nt, jitter=0.10 + 0.05 * ((it // 4) % 3))

        # anneal smooth: depth depends on mode
        if mode == 2 or mode == 5:
            steps, tau0, tau1, lr0, lr1 = 1200, 0.05, 0.003, 0.22, 0.015
        else:
            steps, tau0, tau1, lr0, lr1 = 800, 0.05, 0.004, 0.20, 0.02
        z = _smooth_opt(y, steps=steps, tau0=tau0, tau1=tau1, lr0=lr0, lr1=lr1)
        v = _max_cos(z)
        consider(z, v)
        if v < best_v - 1e-12:
            # immediate adopt as global best (pre-LP)
            best_v = v; best_x = z.copy()
        # time guard
        if time.time() > EXPLORE_END:
            break

    # LP polish phase: polish pool members + global best, best first
    cands = []
    if best_x is not None:
        cands.append(best_x.copy())
    for _, px in pool:
        cands.append(px.copy())
    # dedup not needed; sort by value
    cands_vals = [( _max_cos(c), c) for c in cands]
    cands_vals.sort(key=lambda t: t[0])
    # keep top 4 unique-ish
    top = []
    for v, c in cands_vals:
        if len(top) >= 4:
            break
        top.append(c)
    # allocate remaining time
    for i, c in enumerate(top):
        remain = HARD_END - time.time()
        if remain < 4.0:
            break
        # give more time to the best
        budget = remain / (len(top) - i)
        budget = min(budget, 30.0)
        deadline = time.time() + budget - 0.3
        pc, pv = _lp_polish(c, deadline)
        if pv < best_v:
            best_v = pv; best_x = pc.copy()
        if time.time() > HARD_END - 3.0:
            break
    # final safety polish of global best if time left
    if time.time() < HARD_END - 4.0 and best_x is not None:
        pc, pv = _lp_polish(best_x.copy(), HARD_END - 1.0)
        if pv < best_v:
            best_v = pv; best_x = pc.copy()
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
