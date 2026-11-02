import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _degs(x, m=None, tol=0.002):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(G.max())
    deg = (G > m - tol).sum(axis=1)
    rowm = G.max(axis=1)
    return G, m, deg, rowm

def _rand_rot_small(rng, eps):
    M = np.eye(4) + rng.randn(4, 4) * eps
    Q, R = np.linalg.qr(M)
    d = np.sign(np.diag(R) + 1e-12)
    Q = Q * d
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _refine_hole(h0, Xf, steps=200, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(np.linalg.norm(h0), 1e-12)
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

def _find_holes(Xf, rng, n_samples=16000, n_holes=4):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:8]
    outs = []
    for ii in idx:
        h = _refine_hole(Y[ii], Xf)
        v = float((Xf @ h).max())
        outs.append((v, h))
    outs.sort(key=lambda t: t[0])
    return [h for _, h in outs[:n_holes]]

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

def _lbfgs(x_init, deadline, taus=(0.003, 0.0009, 0.00035), iters=(60, 60, 80)):
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

def _torus_init(rng, n=163, jitter=0.05):
    # Clifford torus grid
    a = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(a):
        for j in range(a):
            if len(pts) >= n:
                break
            th = 2 * np.pi * i / a + 0.1 * (j % 2)
            ph = 2 * np.pi * j / a
            p = np.array([np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)]) / np.sqrt(2.0)
            pts.append(p)
    X = np.array(pts[:n])
    X = X + rng.randn(n, 4) * jitter
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 132.0
    FINAL_END = t0 + BUDGET - 1.5
    rng = np.random.RandomState(20261102)
    n = 163
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.asarray(np.load(fn), dtype=float)
            if v.shape == (163, 4):
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                cands.append(v)
        except Exception:
            pass
    if not cands:
        c = rng.randn(n, 4)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = best_x.copy()
    cur_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    hop_T = 0.00035
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 34:
            break
        trial += 1
        mode = trial % 7
        # periodic torus restart (symmetry basin)
        if trial % 23 == 0:
            cur = _torus_init(rng, n=n, jitter=0.08 + 0.04 * rng.rand())
            dl = min(time.time() + 4.5, SEARCH_END)
            cur_b, v = _anneal(cur, steps=3800, T0=0.02, T1=0.0007, lr=0.05, mu=0.88,
                               rng=rng, noise0=0.0015, deadline=dl)
            cur_b, v = _anneal(cur_b, steps=2200, T0=0.004, T1=0.00025, lr=0.035, mu=0.90,
                               rng=rng, noise0=0.0, deadline=dl)
            if v < cur_val or rng.rand() < np.exp(-max(0.0, v - cur_val) / hop_T):
                cur_x, cur_val = cur_b, v
                if v < best_val:
                    f = _lbfgs(cur_b, deadline=time.time() + 2.0)
                    vf = _max_cos(f)
                    if vf < best_val:
                        best_x, best_val = f, vf
                        cur_x, cur_val = f, vf
                    elif v < best_val:
                        best_x, best_val = cur_b, v
            continue
        base = cur_x.copy()
        G, m0, deg, rowm = _degs(base)
        if mode == 0:
            sig = 0.015 + 0.03 * rng.rand()
            cur = base + rng.randn(n, 4) * sig
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            # hub-avalanche: remove top hubs to deep holes
            score = rowm + 0.0006 * deg + rng.randn(n) * 2e-6
            k = 3 + (trial % 3)  # 3-5
            worst = np.argsort(score)[-k:]
            keep = np.ones(n, dtype=bool); keep[worst] = False
            Xf = base[keep].copy()
            cur = base.copy()
            holes = _find_holes(Xf, rng, n_samples=14000, n_holes=k + 1)
            for t, w in enumerate(worst):
                h = holes[t % len(holes)] + rng.randn(4) * 0.006
                h /= max(np.linalg.norm(h), 1e-12)
                cur[w] = h
                Xf = np.vstack([Xf, h[None, :]])
            cur += rng.randn(n, 4) * 0.004
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            # rigid-cap collective rotation (symmetry-enforced mode)
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = 0.35 + 0.4 * rng.rand()
            dots = base @ u
            mask = dots > thr
            cnt = int(mask.sum())
            if cnt < 4 or cnt > 60:
                idx = rng.choice(n, 14, replace=False)
                cur = base.copy()
                cur[idx] += rng.randn(14, 4) * 0.06
            else:
                Q = _rand_rot_small(rng, eps=0.10 + 0.15 * rng.rand())
                cur = base.copy()
                cur[mask] = cur[mask] @ Q.T
                cur[mask] += rng.randn(cnt, 4) * 0.008
                cur += rng.randn(n, 4) * 0.003
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 3:
            # floppy subcluster: BFS from worst hub, frozen-outside anneal later
            start = int(np.argmax(rowm + 0.0005 * deg))
            adj = G > m0 - 0.002
            seen = np.zeros(n, dtype=bool); seen[start] = True
            frontier = [start]
            want = 7 + int(rng.randint(0, 5))
            while frontier and int(seen.sum()) < want:
                u0 = frontier.pop(0)
                nbrs = np.where(adj[u0])[0]
                rng.shuffle(nbrs)
                for nb in nbrs:
                    if not seen[nb]:
                        seen[nb] = True
                        frontier.append(int(nb))
                        if int(seen.sum()) >= want:
                            break
            if int(seen.sum()) < 5:
                seen[np.argsort(rowm)[-8:]] = True
            cur = base.copy()
            cur[seen] += rng.randn(int(seen.sum()), 4) * 0.09
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
            # two-stage: first only subcluster moves (creates rattler slack)
            dl = min(time.time() + 4.0, SEARCH_END)
            sub_b, sub_v = _anneal(cur, steps=1500, T0=0.012, T1=0.001, lr=0.05, mu=0.85,
                                   rng=rng, active=seen, noise0=0.002, deadline=dl)
            cur = sub_b
        elif mode == 4:
            # pair-breaking push apart top contacts
            c = G[iu0, iu1]
            top = np.argsort(c)[-10:]
            cur = base.copy()
            for tt in top[:6]:
                i = int(iu0[tt]); j = int(iu1[tt])
                d = cur[i] - cur[j]
                nd = float(np.linalg.norm(d))
                if nd < 1e-9:
                    d = rng.randn(4); nd = float(np.linalg.norm(d))
                d /= nd
                step = 0.02 + 0.03 * rng.rand()
                cur[i] += d * step * 0.5 + rng.randn(4) * 0.005
                cur[j] -= d * step * 0.5 + rng.randn(4) * 0.005
            cur += rng.randn(n, 4) * 0.004
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 5:
            # cap shake with common tangent
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = 0.45 + 0.3 * rng.rand()
            mask = (base @ u) > thr
            cnt = int(mask.sum())
            cur = base.copy()
            if cnt == 0 or cnt > n // 2:
                idx = rng.choice(n, 18, replace=False)
                cur[idx] += rng.randn(18, 4) * 0.05
            else:
                vd = rng.randn(4); vd -= float(vd @ u) * u
                vd /= max(np.linalg.norm(vd), 1e-12)
                cur[mask] += vd[None, :] * (0.05 + 0.05 * rng.rand()) + rng.randn(cnt, 4) * 0.012
                cur += rng.randn(n, 4) * 0.003
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        else:
            # strong global shake to different basin
            cur = base + rng.randn(n, 4) * 0.075
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        dl = min(time.time() + 4.2, SEARCH_END)
        # dual anneal: hot then cold
        mid, vm = _anneal(cur, steps=2600, T0=0.016, T1=0.0008, lr=0.048, mu=0.87,
                          rng=rng, noise0=0.0012 if mode in (0, 5, 6) else 0.0004, deadline=dl)
        fin, vf = _anneal(mid, steps=1800, T0=0.003, T1=0.00022, lr=0.032, mu=0.90,
                          rng=rng, noise0=0.0, deadline=dl)
        new_x, new_v = fin, vf
        # accept to current via Metropolis (allows topology hopping)
        if new_v < cur_val or rng.rand() < float(np.exp(-max(0.0, new_v - cur_val) / hop_T)):
            cur_x, cur_val = new_x, new_v
        if new_v < best_val - 1e-12:
            # quick smooth before committing
            sm = _lbfgs(new_x, deadline=time.time() + 1.6)
            vs = _max_cos(sm)
            if vs < new_v:
                new_x, new_v = sm, vs
                if new_v < cur_val:
                    cur_x, cur_val = new_x, new_v
            best_x, best_val = new_x, new_v
        elif new_v < best_val + 0.0004 and (trial % 5 == 0):
            sm = _lbfgs(new_x, deadline=time.time() + 1.0)
            vs = _max_cos(sm)
            if vs < best_val:
                best_x, best_val = sm, vs
        # slow cool hopping temp
        hop_T *= 0.9995
        hop_T = max(hop_T, 8e-05)
    # final polish best
    try:
        best_x = _lbfgs(best_x, deadline=FINAL_END - 6.0)
        best_val = _max_cos(best_x)
        best_x, best_val = _lp_polish(best_x, deadline=FINAL_END, r0=0.002)
    except Exception:
        pass
    best_x /= np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
