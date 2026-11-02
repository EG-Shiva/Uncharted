import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    n = x.shape[0]
    G = x @ x.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    m = float(c.max())
    sel = c > m - thr
    I = iu0[sel]; J = iu1[sel]
    deg = np.zeros(n, dtype=int)
    for a, b in zip(I, J):
        deg[a] += 1; deg[b] += 1
    return m, I, J, deg, G

def _rand_rot(rng):
    A = rng.randn(4, 4)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _anneal(x0, steps, T0, T1, lr, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    na = int(act.sum())
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lrk = lr * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        if na != n:
            Grad[~act] = 0.0
        vel = mu * vel + (1.0 - mu) * Grad
        if na != n:
            vel[~act] = 0.0
        x[act] = x[act] - lrk * vel[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0 and na > 0:
                nz = rng.randn(na, d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                xa = xa + nz
                xa /= np.maximum(np.linalg.norm(xa, axis=1, keepdims=True), 1e-12)
                x[act] = xa
        xn = np.linalg.norm(x[act], axis=1, keepdims=True)
        x[act] /= np.maximum(xn, 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.003, 0.0009, 0.00035), iters=(70, 70, 90)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _lp_polish(x, deadline, r0=0.002, cap=1200):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _greedy_hole_fill(R, k, rng, pool_size=16000):
    # R: (nr,4) remaining points, return k new points far from R and each other
    nr = R.shape[0]
    P = rng.randn(pool_size, 4)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    # max dot to R
    D = P @ R.T
    cur = D.max(axis=1)
    chosen = []
    selmask = np.zeros(pool_size, dtype=bool)
    for _ in range(k):
        tmp = cur.copy()
        tmp[selmask] = 1e9
        j = int(np.argmin(tmp))
        chosen.append(j)
        selmask[j] = True
        d = P @ P[j]
        cur = np.maximum(cur, d)
        cur[selmask] = 1e9
    return P[np.array(chosen)]

def _torus_init(rng):
    n = 163
    nt = int(rng.randint(5, 10))
    thetas = rng.uniform(0.35, 1.22, size=nt)
    area = np.sin(2 * thetas) + 0.15
    area /= area.sum()
    alloc = np.floor(area * n).astype(int)
    while alloc.sum() < n:
        alloc[int(rng.randint(nt))] += 1
    while alloc.sum() > n:
        i = int(rng.randint(nt))
        if alloc[i] > 1:
            alloc[i] -= 1
    pts = []
    for nk, th in zip(alloc, thetas):
        nk = int(nk)
        if nk <= 0:
            continue
        ct, st = float(np.cos(th)), float(np.sin(th))
        ct = max(ct, 0.08); st = max(st, 0.08)
        cot = ct / st
        nx = int(round(np.sqrt(max(nk * cot, 1.0))))
        nx = max(1, min(nk, nx))
        ny = int((nk + nx - 1) // nx)
        ox, oy = float(rng.rand()), float(rng.rand())
        tw = float(rng.uniform(-0.35, 0.35))
        pat = rng.rand() < 0.5
        uu = []; vv = []
        for ix in range(nx):
            for iy in range(ny):
                u = (ix + ox) / nx % 1.0
                v = (iy + oy) / ny % 1.0
                if pat:
                    u = (u + 0.5 * (iy % 2) / nx) % 1.0
                uu.append(u); vv.append(v)
        uu = np.array(uu); vv = np.array(vv)
        if len(uu) > nk:
            idx = rng.choice(len(uu), nk, replace=False)
            uu = uu[idx]; vv = vv[idx]
        vv = (vv + tw * uu) % 1.0
        uu = (uu + rng.randn(len(uu)) * 0.0025) % 1.0
        vv = (vv + rng.randn(len(vv)) * 0.0025) % 1.0
        a = 2 * np.pi * uu; b = 2 * np.pi * vv
        X = np.stack([ct * np.cos(a), ct * np.sin(a), st * np.cos(b), st * np.sin(b)], axis=1)
        pts.append(X)
    X = np.vstack(pts)
    if X.shape[0] > n:
        idx = rng.choice(X.shape[0], n, replace=False)
        X = X[idx]
    elif X.shape[0] < n:
        extra = rng.randn(n - X.shape[0], 4)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        X = np.vstack([X, extra])
    X = X @ _rand_rot(rng).T
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _cap_rotate(X, rng):
    n = X.shape[0]
    u = rng.randn(4); u /= np.linalg.norm(u)
    dots = X @ u
    # random cap size: threshold quantile
    q = float(rng.uniform(0.80, 0.96))
    thr = float(np.quantile(dots, q))
    idx = np.where(dots >= thr)[0]
    if len(idx) < 3 or len(idx) > 60:
        # fallback to random subset of size 8-20
        k = int(rng.randint(8, 21))
        idx = rng.choice(n, k, replace=False)
    # random small rotation in 4D
    A = rng.randn(4, 4) * 1.0
    A = A - A.T
    ang = float(rng.uniform(0.05, 0.35))
    # exp approx via QR of I+ang*A
    M = np.eye(4) + ang * A
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    Y = X.copy()
    Y[idx] = (Y[idx] @ Q.T)
    # add jitter to cap
    amp = float(rng.uniform(0.005, 0.03))
    nz = rng.randn(len(idx), 4) * amp
    nz -= (nz * Y[idx]).sum(axis=1, keepdims=True) * Y[idx]
    Y[idx] = Y[idx] + nz
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _hub_avalanche(X, rng):
    n = X.shape[0]
    try:
        m, I, J, deg, G = _contacts(X, thr=2e-7)
    except Exception:
        deg = np.zeros(n, dtype=int)
    order = np.argsort(-deg)
    k = int(rng.randint(7, 15))
    # take top hub + its neighbors to break the jammed cluster
    hubs = list(order[:max(2, k // 3)])
    doomed = set(hubs)
    # add neighbors of hubs
    # build adjacency
    adj = [set() for _ in range(n)]
    for a, b in zip(I, J):
        adj[a].add(b); adj[b].add(a)
    for h in list(hubs):
        for nb in adj[h]:
            if len(doomed) >= k and rng.rand() < 0.7:
                continue
            doomed.add(int(nb))
            if len(doomed) >= k:
                break
        if len(doomed) >= k:
            break
    while len(doomed) < k:
        doomed.add(int(rng.randint(n)))
    # occasionally add a random far point to create rattler
    if rng.rand() < 0.5:
        doomed.add(int(rng.randint(n)))
    doomed = sorted(doomed)[:16]
    k2 = len(doomed)
    mask = np.ones(n, dtype=bool); mask[doomed] = False
    R = X[mask]
    try:
        fill = _greedy_hole_fill(R, k2, rng, pool_size=16000)
    except Exception:
        fill = rng.randn(k2, 4)
        fill /= np.linalg.norm(fill, axis=1, keepdims=True)
    Y = X.copy()
    Y[np.array(doomed)] = fill
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    act = np.zeros(n, dtype=bool); act[doomed] = True
    return Y, act

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SCREEN_END = t0 + 122.0
    POLISH_END = DEADLINE - 2.0
    rng = np.random.RandomState(1630403)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    # quick diverse elite pool
    elites = [(best_val, best_x.copy())]
    # seed pool with a few torus inits quickly annealed
    it = 0
    # main basin-hopping loop
    while time.time() < SCREEN_END:
        it += 1
        # pick base: usually best, sometimes random elite
        if len(elites) > 1 and rng.rand() < 0.25:
            bi = int(rng.randint(len(elites)))
            base = elites[bi][1]
        else:
            base = best_x
        r = rng.rand()
        X0 = None
        act0 = None
        act_two_stage = False
        if r < 0.38:
            # hub avalanche + hole fill (primary topology jump)
            try:
                Y, act = _hub_avalanche(base, rng)
            except Exception:
                Y = base.copy(); act = None
            X0 = Y; act0 = act; act_two_stage = True
        elif r < 0.58:
            # rigid cap rotation
            try:
                X0 = _cap_rotate(base, rng)
            except Exception:
                X0 = base.copy()
        elif r < 0.72:
            # large tangential shake (break contacts)
            amp = float(rng.uniform(0.02, 0.12))
            Y = base.copy()
            nz = rng.randn(n, 4) * amp
            nz -= (nz * Y).sum(axis=1, keepdims=True) * Y
            Y = Y + nz
            Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X0 = Y
        elif r < 0.86:
            # fresh torus collective mode
            try:
                X0 = _torus_init(rng)
                # mix a little toward base to keep quality? no, keep fresh
                if rng.rand() < 0.3:
                    # crossover: replace random half with base points far from torus points
                    k = int(rng.randint(20, 60))
                    idx = rng.choice(n, k, replace=False)
                    X0[idx] = base[idx]
            except Exception:
                X0 = rng.randn(n, 4)
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        else:
            # uniform fresh + greedy? just random
            X0 = rng.randn(n, 4)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        # anneal params randomized (dual temperature)
        T0 = float(rng.uniform(0.006, 0.016))
        T1 = float(rng.uniform(0.00025, 0.0007))
        lr = float(rng.uniform(0.30, 0.60))
        mu = float(rng.uniform(0.86, 0.93))
        noise0 = float(rng.uniform(0.0, 0.006))
        steps = int(rng.choice([900, 1200, 1600]))
        dl = min(SCREEN_END, POLISH_END)
        try:
            if act_two_stage and act0 is not None and act0.sum() > 0:
                # stage 1: move only avalanche set (creates rattler/floppy), high T
                Xa, va = _anneal(X0, steps // 3, T0, max(T0 * 0.25, 0.0015), lr, mu, rng, active=act0, noise0=noise0 + 0.004, deadline=dl)
                # stage 2: full anneal low T
                Xb, vb = _anneal(Xa, steps, max(T0 * 0.3, 0.002), T1, lr, mu, rng, active=None, noise0=noise0, deadline=dl)
            else:
                Xb, vb = _anneal(X0, steps, T0, T1, lr, mu, rng, active=None, noise0=noise0, deadline=dl)
        except Exception:
            continue
        if not np.all(np.isfinite(Xb)):
            continue
        try:
            v = _max_cos(Xb)
        except Exception:
            continue
        # keep elite diversity
        if v < best_val + 0.003:
            elites.append((v, Xb.copy()))
            elites = sorted(elites, key=lambda e: e[0])[:8]
        if v + 1e-12 < best_val:
            # refine promising with lbfgs + occasional LP
            Xr = Xb
            try:
                if time.time() < SCREEN_END - 6.0:
                    Xr = _lbfgs(Xb, min(time.time() + 4.0, SCREEN_END))
                    vr = _max_cos(Xr)
                    if vr < v:
                        Xb, v = Xr, vr
                    else:
                        Xr = Xb
            except Exception:
                pass
            # LP polish only if very promising to save time
            if v < best_val - 2e-7 and time.time() < SCREEN_END - 4.0:
                try:
                    Xp, vp = _lp_polish(Xb, min(time.time() + 5.0, SCREEN_END + 4.0), r0=0.002, cap=1100)
                    if vp < v:
                        Xb, v = Xp, vp
                except Exception:
                    pass
            if v + 1e-12 < best_val:
                best_val = float(v)
                best_x = Xb.copy()
                elites[0] = (best_val, best_x.copy())
                elites = sorted(elites, key=lambda e: e[0])[:8]
        if time.time() > SCREEN_END:
            break
    # final intensive polish of best
    try:
        if time.time() < POLISH_END:
            Xf = _lbfgs(best_x, POLISH_END - 8.0, taus=(0.002, 0.0007, 0.0003), iters=(90, 90, 120))
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best_x = vf, Xf
    except Exception:
        pass
    # large-radius topology-jump LP attempt then small LP
    try:
        if time.time() < POLISH_END - 6.0:
            Xp, vp = _lp_polish(best_x, POLISH_END - 4.0, r0=0.008, cap=1600)
            if vp < best_val:
                best_val, best_x = vp, Xp
    except Exception:
        pass
    try:
        if time.time() < POLISH_END:
            Xp2, vp2 = _lp_polish(best_x, POLISH_END, r0=0.0015, cap=1200)
            if vp2 < best_val:
                best_val, best_x = vp2, Xp2
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # strict feasibility: tiny shrink not needed but ensure no nan
    best_x[~np.isfinite(best_x)] = 0.25
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
