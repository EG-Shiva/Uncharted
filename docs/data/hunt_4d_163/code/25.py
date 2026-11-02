import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=1e-3):
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
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
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

def _lp_jump(x, radius=0.18, topk=1400):
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    try:
        G = x @ x.T
        c = G[iu0, iu1]
        idx = np.argsort(c)[-topk:]
        I = iu0[idx]; J = iu1[idx]
        csel = c[idx]
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
        A_ub = coo_matrix((cc.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        if res.status != 0:
            return None
        y = x + res.x[:-1].reshape(n, d)
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        return y
    except Exception:
        return None

def _gd_soft(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _gd_power(x0, steps, p0, p1, lr0, mu, rng, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        p = p0 + (p1 - p0) * frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        D = 1.0 - G
        D[D < 1e-6] = 1e-6
        # weights, clipped for stability
        W = np.power(D, -p - 1.0)
        np.fill_diagonal(W, 0.0)
        # normalize by mean to keep step scale stable
        mn = float(W.mean()) + 1e-300
        W = W / mn
        grad = (W @ x)
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        # scale down
        gn = float(np.linalg.norm(grad) / n) + 1e-300
        grad = grad / gn * 0.2
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(90, 90, 130)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
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
            Yt = y.reshape(n, d)
            Yt /= np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
            y = Yt.ravel().copy()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf, _max_cos(Xf)

def _refine_hole(h0, Xf, steps=200, T0=0.02, T1=0.002, lr0=0.08):
    h = h0 / max(np.linalg.norm(h0), 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xf @ h
        m = float(dots.max())
        e = np.exp((dots - m) / max(T, 1e-12))
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        grad = w @ Xf
        grad = grad - float(grad @ h) * h
        h = h - lr * grad
        nh = np.linalg.norm(h)
        if nh < 1e-12:
            break
        h /= nh
    return h

def _find_hole(Xf, rng, n_samples=16000):
    Y = rng.randn(n_samples, 4)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = np.argsort(vals)[:6]
    best_h = Y[idx[0]]
    best_v = float(vals[idx[0]])
    for ii in idx:
        h = _refine_hole(Y[ii], Xf)
        v = float((Xf @ h).max())
        if v < best_v:
            best_v = v
            best_h = h
    return best_h, best_v

def _rand_rot4(rng, ang):
    A = rng.randn(4, 4)
    A = A - A.T
    # first-order exp approx via QR of I+ang*A
    M = np.eye(4) + ang * A
    Q, R = np.linalg.qr(M)
    d = np.diag(R)
    d[d == 0] = 1.0
    Q = Q * np.sign(d)[None, :]
    return Q

def solve():
    t0 = time.time()
    BUDGET = 172.0
    SEARCH_END = t0 + 130.0
    FINAL_END = t0 + BUDGET - 1.0
    rng = np.random.RandomState(12345)
    n = 163
    cands = []
    try:
        bx = np.load("best.npy")
        bx = np.asarray(bx, dtype=float)
        if bx.shape == (163, 4):
            cands.append(bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12))
    except Exception:
        pass
    try:
        rec = np.load("record.npy")
        rec = np.asarray(rec, dtype=float)
        if rec.shape == (163, 4):
            cands.append(rec / np.maximum(np.linalg.norm(rec, axis=1, keepdims=True), 1e-12))
    except Exception:
        pass
    if len(cands) == 0:
        c = rng.randn(n, 4)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        cands.append(c)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    # elite archive
    elites = [(best_val, best_x.copy())]
    # quick baseline polish
    if time.time() < SEARCH_END:
        px, pv = _lp_polish(best_x, deadline=min(time.time() + 4.0, SEARCH_END))
        if pv < best_val:
            best_x, best_val = px, pv
            elites[0] = (best_val, best_x.copy())

    trial = 0
    sigmas = [0.02, 0.06, 0.13, 0.008, 0.05, 0.10]
    while time.time() < SEARCH_END:
        if time.time() > t0 + BUDGET - 34:
            break
        # pick base: mostly best, sometimes other elite / random elite mix
        if len(elites) > 1 and (trial % 7 == 6):
            elites_sorted = sorted(elites, key=lambda e: e[0])
            base = elites_sorted[(trial // 7) % len(elites_sorted)][1].copy()
        else:
            base = best_x.copy()
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 8
        cur = base.copy()
        if mode == 0:
            cur = cur + rng.randn(*cur.shape) * sigma
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 1:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            rowm = G.max(axis=1)
            order = np.argsort(rowm + 0.0005 * deg)
            worst = order[-12:]
            noise = rng.randn(*cur.shape) * (sigma * 0.25)
            noise[worst] += rng.randn(len(worst), 4) * (sigma * 2.4)
            cur = cur + noise
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 2:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rowm = G.max(axis=1)
            m = float(G.max())
            deg = (G > m - 0.002).sum(axis=1)
            score = rowm + 0.0004 * deg + rng.randn(n) * 1e-6
            k = 2 + (trial % 3)
            worst = np.argsort(score)[-k:]
            keep = np.ones(n, dtype=bool)
            keep[worst] = False
            Xf = cur[keep].copy()
            for w in worst:
                h, _ = _find_hole(Xf, rng, n_samples=14000)
                h = h + rng.randn(4) * 0.008
                h /= max(np.linalg.norm(h), 1e-12)
                cur[w] = h
                Xf = np.vstack([Xf, h[None, :]])
            cur = cur + rng.randn(*cur.shape) * (sigma * 0.2)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 3:
            # rigid cap rotation
            u = rng.randn(4); u /= np.linalg.norm(u)
            thr = 0.40 + 0.40 * rng.rand()
            dots = cur @ u
            mask = dots > thr
            cnt = int(mask.sum())
            if cnt < 4 or cnt > n // 2:
                idx = rng.choice(n, 16, replace=False)
                cur[idx] += rng.randn(16, 4) * sigma
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
            else:
                ang = sigma * (2.0 + 3.0 * rng.rand())
                R = _rand_rot4(rng, ang)
                cur[mask] = (R @ cur[mask].T).T
                cur += rng.randn(*cur.shape) * (sigma * 0.15)
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 4:
            # avalanche around worst point via contact graph
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            rowm = G.max(axis=1)
            seed = int(np.argmax(rowm + rng.randn(n) * 1e-6))
            adj = (G > m - 0.0015)
            # BFS depth 2
            seen = set([seed])
            frontier = [seed]
            for _ in range(2):
                nxt = []
                for u_ in frontier:
                    nb = np.where(adj[u_])[0]
                    for w_ in nb:
                        if w_ not in seen:
                            seen.add(int(w_))
                            nxt.append(int(w_))
                frontier = nxt
                if not frontier:
                    break
            seen = list(seen)
            cur[seen] += rng.randn(len(seen), 4) * (sigma * 1.8)
            # also shake neighbors lightly
            cur += rng.randn(*cur.shape) * (sigma * 0.12)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 5:
            # large-radius LP jump then noise
            jr = 0.12 + 0.10 * rng.rand()
            y = _lp_jump(cur, radius=jr, topk=1400)
            if y is not None:
                cur = y + rng.randn(*cur.shape) * (0.01 + sigma * 0.2)
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
            else:
                cur = cur + rng.randn(*cur.shape) * sigma
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
        elif mode == 6:
            # cold / structured restart mixed with best
            if trial % 16 == 6:
                # Hopf-like toroidal grid + noise
                a = rng.randn(4); a /= np.linalg.norm(a)
                # random fibonacci on S3
                Y = rng.randn(n, 4)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                # mix half best half random to keep diversity
                perm = rng.permutation(n)
                take = n // 3
                cur[perm[:take]] = Y[perm[:take]]
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            else:
                cur = rng.randn(n, 4)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        else:
            # pair-break push
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            iu0, iu1 = np.triu_indices(n, 1)
            c = G[iu0, iu1]
            top = np.argsort(c)[-10:]
            for tt in top[:6]:
                i = int(iu0[tt]); j = int(iu1[tt])
                if rng.rand() < 0.7:
                    d = cur[i] - cur[j]
                    dn = np.linalg.norm(d)
                    if dn > 1e-9:
                        d /= dn
                        cur[i] += d * sigma * 0.9
                        cur[j] -= d * sigma * 0.9
            cur += rng.randn(*cur.shape) * (sigma * 0.2)
            cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)

        # choose optimizer
        dl = min(time.time() + 4.5, SEARCH_END, t0 + BUDGET - 33)
        use_power = (trial % 5 == 4)
        try:
            if use_power:
                p0 = 12.0 + (trial % 3) * 8.0
                p1 = 60.0 + (trial % 2) * 30.0
                cur_b, v = _gd_power(cur, steps=3500, p0=p0, p1=p1, lr0=0.05, mu=0.85, rng=rng, deadline=dl)
                # follow with short softmax
                cur_b, v2 = _gd_soft(cur_b, steps=2500, T0=0.006, T1=0.0006, lr0=0.035, mu=0.88, rng=rng, noise0=0.0, deadline=dl)
                v = min(v, v2) if v2 < 1 else v
                # need best of the two: recompute
                if v2 <= v:
                    pass
                else:
                    # _gd_soft returns its own best
                    v = v2
                    # cur_b already from second call
            else:
                T0 = 0.02 + 0.01 * ((trial % 3) == 1)
                T1 = 0.0007 + 0.0004 * ((trial % 2))
                cur_b, v = _gd_soft(cur, steps=6000, T0=T0, T1=T1, lr0=0.040, mu=0.88, rng=rng, noise0=0.0015 if mode in (0, 3) else 0.0005, deadline=dl)
        except Exception:
            trial += 1
            continue
        # opportunistic smoothing + polish only if promising
        if v < best_val + 0.004 and time.time() < SEARCH_END - 1.0:
            try:
                cur_s, vs = _lbfgs_smooth(cur_b, deadline=time.time() + 1.8)
                if vs < v:
                    cur_b, v = cur_s, vs
            except Exception:
                pass
        if v < best_val:
            # polish immediately
            try:
                pol, vp = _lp_polish(cur_b, deadline=min(time.time() + 2.5, FINAL_END))
                if vp < best_val:
                    best_x, best_val = pol, vp
                    elites.append((best_val, best_x.copy()))
                    if len(elites) > 6:
                        elites = sorted(elites, key=lambda e: e[0])[:6]
                elif v < best_val:
                    best_x, best_val = cur_b, v
                    elites.append((best_val, best_x.copy()))
                    if len(elites) > 6:
                        elites = sorted(elites, key=lambda e: e[0])[:6]
            except Exception:
                best_x, best_val = cur_b, v
        elif v < best_val + 0.0015:
            # keep diverse elite if far
            try:
                dmin = min(float(np.linalg.norm(cur_b - e[1]).mean()) for e in elites)
            except Exception:
                dmin = 1.0
            if dmin > 0.02 and len(elites) < 6:
                elites.append((v, cur_b.copy()))
        trial += 1
    # final intensive polish
    try:
        best_x, best_val = _lp_polish(best_x, deadline=FINAL_END, r0=1e-3)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
