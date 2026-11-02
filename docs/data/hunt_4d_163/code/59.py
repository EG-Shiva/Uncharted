import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, csr_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

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
    lratio = float(np.log(T1 / T0))
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

def _lbfgs(x_init, deadline, taus=(0.003, 0.0009, 0.00035), iters=(60, 60, 70)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.5:
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
            L = m + _tau * float(np.log(s / Np))
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

def _lp_polish(x, deadline, r0=0.002, cap=1100):
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
                s2 = np.zeros_like(sel, dtype=bool)
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
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 4.0})
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _refine_hole(h0, Xf, steps=200, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(float(np.linalg.norm(h0)), 1e-12)
    lratio = float(np.log(T1 / T0))
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = max(T0 * np.exp(lratio * frac), 1e-12)
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = float(np.linalg.norm(h))
        if nh < 1e-12:
            break
        h /= nh
    return h

def _self_stress_lamb(x, thr=1e-7):
    n = x.shape[0]
    iu0, iu1 = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iu0, iu1]
    m = float(c.max())
    sel = c > m - thr
    if int(sel.sum()) < 10:
        sel = np.zeros_like(sel, dtype=bool)
        idx = np.argsort(-c)[:max(100, 3 * n)]
        sel[idx] = True
    I = iu0[sel]; J = iu1[sel]
    p = len(I)
    # tangent bases
    T = np.zeros((n, 4, 3))
    for i in range(n):
        xi = x[i]
        am = int(np.argmax(np.abs(xi)))
        cols = []
        for k in range(4):
            if k == am:
                continue
            e = np.zeros(4); e[k] = 1.0
            v = e - float(e @ xi) * xi
            cols.append(v)
        M = np.stack(cols, axis=1)
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    AT = np.zeros((3 * n, p))
    for t in range(p):
        i = int(I[t]); j = int(J[t])
        AT[i * 3:(i + 1) * 3, t] = T[i].T @ x[j]
        AT[j * 3:(j + 1) * 3, t] = T[j].T @ x[i]
    nd = 3 * n
    nv = p + nd
    try:
        # LP: min sum u s.t. AT lam - u <=0, -AT lam - u <=0, sum lam=1
        cobj = np.zeros(nv); cobj[p:] = 1.0
        Aub = np.zeros((2 * nd, nv))
        Aub[:nd, :p] = AT.T if False else 0  # placeholder
        # Build properly: rows 0..nd-1: AT^T? No, constraint is AT*lam - u <=0 per dim row
        # AT is (nd x p), so AT*lam is (nd)
        Aub[:nd, :p] = AT.T.T if False else AT.T  # this is wrong shape; fix below
    except Exception:
        pass
    # build correctly with explicit arrays
    try:
        Aub = np.zeros((2 * nd, nv))
        # top block: [AT^T? ] we need (nd x nv): first p cols = AT, last nd = -I
        Aub[0:nd, 0:p] = AT
        Aub[0:nd, p:] = -np.eye(nd)
        Aub[nd:2 * nd, 0:p] = -AT
        Aub[nd:2 * nd, p:] = -np.eye(nd)
        bub = np.zeros(2 * nd)
        Aeq = np.zeros((1, nv)); Aeq[0, :p] = 1.0
        beq = np.ones(1)
        bounds = [(0, None)] * p + [(0, None)] * nd
        res = linprog(cobj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs",
                      options={"time_limit": 8.0})
        if res.status == 0:
            lam = np.maximum(res.x[:p], 0)
            s = lam.sum()
            if s > 0:
                lam /= s
            return lam, I, J
    except Exception:
        pass
    lam = np.ones(p) / max(p, 1)
    return lam, I, J

def _bfs_patch_maxstress(i0, j0, adj, target):
    # adj: list of lists of (nbr, lam)
    visited = set([int(i0), int(j0)])
    if target <= 2:
        return list(visited)[:target]
    # frontier: dict nbr -> best connecting lam to visited
    import heapq
    # use negative lam for max heap via nlargest each step (small sizes, simple loop)
    while len(visited) < target:
        best_v = -1
        best_w = -1e100
        for u in list(visited):
            for (v, w) in adj[u]:
                if v in visited:
                    continue
                if w > best_w:
                    best_w = w
                    best_v = v
        if best_v < 0:
            # disconnected: add lowest-degree unvisited to keep contiguous? just add any
            for u in range(len(adj)):
                if u not in visited:
                    best_v = u
                    break
            if best_v < 0:
                break
        visited.add(int(best_v))
    return list(visited)

def _greedy_reinsert(Xf, removed_pts, rng, n_pool=20000, per_jit=100):
    nf = Xf.shape[0]
    k = removed_pts.shape[0]
    Yrand = rng.randn(n_pool, 4)
    Yrand /= np.maximum(np.linalg.norm(Yrand, axis=1, keepdims=True), 1e-12)
    if k > 0 and per_jit > 0:
        rep = np.repeat(removed_pts, per_jit, axis=0)
        nz = rng.randn(rep.shape[0], 4) * 0.15
        nz -= (nz * rep).sum(axis=1, keepdims=True) * rep
        rep = rep + nz
        rep /= np.maximum(np.linalg.norm(rep, axis=1, keepdims=True), 1e-12)
        Ypool = np.vstack([Yrand, rep])
    else:
        Ypool = Yrand
    Df = Ypool @ Xf.T
    vals_f = Df.max(axis=1)
    cur_max = vals_f.copy()
    used = np.zeros(Ypool.shape[0], dtype=bool)
    placed = []
    Xall = Xf.copy()
    for s in range(k):
        # pick best unused
        tmp = cur_max.copy()
        tmp[used] = 1e9
        idx = int(np.argmin(tmp))
        used[idx] = True
        h0 = Ypool[idx]
        h = _refine_hole(h0, Xall, steps=220, T0=0.02, T1=0.002, lr0=0.08)
        placed.append(h)
        Xall = np.vstack([Xall, h.reshape(1, 4)])
        # update cur_max with new point
        dnew = Ypool @ h
        cur_max = np.maximum(cur_max, dnew)
    if len(placed) == 0:
        return np.zeros((0, 4))
    return np.stack(placed, axis=0)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 126.0
    rng = np.random.RandomState(713163)
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
    # compute self-stress on current best
    try:
        lam, EI, EJ = _self_stress_lamb(best_x, thr=1e-7)
    except Exception:
        lam = None
        EI = EJ = None
    if lam is None or len(lam) == 0:
        # fallback to degree hubs
        G = best_x @ best_x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu0, iu1 = np.triu_indices(n, 1)
        c = G[iu0, iu1]
        sel = c > m - 1e-7
        EI = iu0[sel]; EJ = iu1[sel]
        lam = np.ones(len(EI)) / max(len(EI), 1)
    order = np.argsort(-lam)
    topK = min(4, len(order))
    top_edges = order[:topK]
    # adjacency with lambda weights
    adj = [[] for _ in range(n)]
    emap = {}
    for t in range(len(EI)):
        i = int(EI[t]); j = int(EJ[t]); w = float(lam[t])
        adj[i].append((j, w))
        adj[j].append((i, w))
        emap[(i, j)] = w
        emap[(j, i)] = w
    radii = [12, 18, 24]
    trials = []
    for ei in top_edges:
        for R in radii:
            trials.append((int(ei), int(R)))
    # run trials
    for (ei, R) in trials:
        if time.time() > SEARCH_END:
            break
        if ei >= len(EI):
            continue
        i0 = int(EI[ei]); j0 = int(EJ[ei])
        try:
            patch = _bfs_patch_maxstress(i0, j0, adj, R)
        except Exception:
            continue
        patch = sorted(set(int(v) for v in patch))
        k = len(patch)
        if k < 8 or k > 30:
            continue
        frozen_mask = np.ones(n, dtype=bool)
        frozen_mask[patch] = False
        Xf = best_x[frozen_mask]
        rem = best_x[np.array(patch)]
        try:
            placed = _greedy_reinsert(Xf, rem, rng, n_pool=20000, per_jit=80)
        except Exception:
            continue
        if placed.shape[0] != k:
            continue
        Xt = best_x.copy()
        Xt[np.array(patch)] = placed
        # stage 1: patch-only softmax anneal T 0.004 -> 0.0004, 1500 steps
        active = np.zeros(n, dtype=bool)
        active[np.array(patch)] = True
        try:
            Xp, vp = _anneal(Xt, steps=1500, T0=0.004, T1=0.0004, lr=0.09, mu=0.85,
                             rng=rng, active=active, noise0=0.0,
                             deadline=SEARCH_END)
        except Exception:
            continue
        # stage 2: all-point anneal short
        try:
            Xa, va = _anneal(Xp, steps=700, T0=0.0015, T1=0.00025, lr=0.05, mu=0.85,
                             rng=rng, active=None, noise0=0.0,
                             deadline=SEARCH_END)
        except Exception:
            Xa, va = Xp, vp
        # lbfgs quick if promising
        if va < best_val + 0.002 and time.time() < SEARCH_END - 3.0:
            try:
                Xl = _lbfgs(Xa, deadline=SEARCH_END, taus=(0.002, 0.0007), iters=(50, 60))
                vl = _max_cos(Xl)
                if vl < va:
                    Xa, va = Xl, vl
            except Exception:
                pass
        # LP polish if promising
        if va < best_val + 0.001 and time.time() < SEARCH_END - 2.0:
            try:
                Xp2, vp2 = _lp_polish(Xa, deadline=min(SEARCH_END, time.time() + 6.0), r0=0.002, cap=1100)
                if vp2 < va:
                    Xa, va = Xp2, vp2
            except Exception:
                pass
        if va < best_val:
            best_val = float(va)
            best_x = Xa.copy() / np.maximum(np.linalg.norm(Xa, axis=1, keepdims=True), 1e-12)
            # refresh adjacency? keep original stress topology for remaining trials (deliberate)
            # but update best value for acceptance
        if time.time() > SEARCH_END - 1.0:
            break
    # final polish of global best
    try:
        if time.time() < DEADLINE - 8.0:
            Xl = _lbfgs(best_x, deadline=DEADLINE - 6.0, taus=(0.002, 0.0007, 0.0003), iters=(50, 60, 70))
            vl = _max_cos(Xl)
            if vl < best_val:
                best_val = vl
                best_x = Xl
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 5.0:
            Xp, vp = _lp_polish(best_x, deadline=DEADLINE - 0.5, r0=0.002, cap=1300)
            if vp < best_val:
                best_val = vp
                best_x = Xp
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
